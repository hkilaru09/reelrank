"""Two-tower retrieval model.

Item tower: movie id + genres + release-year bucket + tags -> MLP -> unit vector.
User tower: user id + mean-pooled embeddings of the last `max_hist` liked
movies -> MLP -> unit vector. Score = cosine / temperature.

Training predicts each liked movie from the movies liked *before* it, with a
softmax over the other positives in the batch (in-batch negatives) plus
uniformly sampled extra negatives. In-batch negatives are drawn in
proportion to popularity, so without a correction the model learns to push
popular movies down; the logQ correction (Yi et al., 2019) subtracts the log
of each negative's expected count before the softmax. The user id is
randomly dropped to an "unknown user" slot during training so the same
tower serves brand-new users from their history alone.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Recommender
from .split import EvalSet, Fold, ItemFeatures


@dataclass
class TwoTowerConfig:
    dim: int = 64
    hidden: int = 256
    max_hist: int = 50
    temperature: float = 0.05
    logq: bool = True
    n_uniform_neg: int = 1024
    user_id_dropout: float = 0.25
    batch_size: int = 1024
    lr: float = 2e-3
    weight_decay: float = 0.0
    epochs: int = 20
    patience: int = 3
    year_bucket: int = 5
    seed: int = 0


def pad_lists(lists: list[np.ndarray], width: int | None = None) -> np.ndarray:
    """Right-pad index lists with -1 into a 2-D array."""
    width = width or max(1, max((len(x) for x in lists), default=1))
    out = np.full((len(lists), width), -1, dtype=np.int64)
    for i, x in enumerate(lists):
        out[i, : min(len(x), width)] = x[:width]
    return out


def last_n(seqs: list[np.ndarray], n: int) -> np.ndarray:
    """Most recent n items of each sequence, right-aligned is not needed for mean pooling."""
    return pad_lists([s[-n:] for s in seqs], n)


class MeanBag(nn.Module):
    """Mean of embeddings over -1-padded index lists (zero vector for an empty list)."""

    def __init__(self, n: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n + 1, dim, padding_idx=0)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        e = self.emb(idx + 1)                                 # -1 -> padding row 0
        cnt = (idx >= 0).sum(1, keepdim=True).clamp(min=1)
        return e.sum(1) / cnt


def mlp(d_in: int, hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, hidden), nn.ReLU(), nn.Linear(hidden, d_out))


class TwoTowerNet(nn.Module):
    def __init__(self, n_users: int, n_items: int, n_genres: int, n_tags: int, n_years: int,
                 cfg: TwoTowerConfig):
        super().__init__()
        d = cfg.dim
        self.user_id = nn.Embedding(n_users + 1, d)           # row 0 = unknown user
        self.hist = MeanBag(n_items, d)
        self.user_mlp = mlp(2 * d, cfg.hidden, d)
        self.item_id = nn.Embedding(n_items, d)
        self.genre = MeanBag(max(n_genres, 1), d)
        self.tag = MeanBag(max(n_tags, 1), d)
        self.year = nn.Embedding(n_years, d)
        self.item_mlp = mlp(4 * d, cfg.hidden, d)
        for e in (self.user_id, self.item_id, self.year):
            nn.init.normal_(e.weight, std=0.05)
        for bag in (self.hist, self.genre, self.tag):
            nn.init.normal_(bag.emb.weight, std=0.05)
            with torch.no_grad():
                bag.emb.weight[0].zero_()

    def user(self, uid: torch.Tensor, hist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([self.user_id(uid), self.hist(hist)], dim=1)
        return F.normalize(self.user_mlp(x), dim=1)

    def item(self, iid, genres, tags, year) -> torch.Tensor:
        x = torch.cat([self.item_id(iid), self.genre(genres), self.tag(tags), self.year(year)], dim=1)
        return F.normalize(self.item_mlp(x), dim=1)


def default_device() -> torch.device:
    # On an M2, MPS was 1.4-2.8x *slower* per step than CPU for this model at
    # every size tried (embedding-heavy backward passes, small matmuls), so
    # CPU is the default; pass device="mps" or "cuda" to override.
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoTower(Recommender):
    name = "two_tower"

    def __init__(self, cfg: TwoTowerConfig | None = None, device: str | None = None,
                 verbose: bool = True, **overrides):
        self.cfg = TwoTowerConfig(**{**asdict(cfg or TwoTowerConfig()), **overrides})
        self.device = torch.device(device) if device else default_device()
        self.verbose = verbose

    # ----- feature tensors -------------------------------------------------
    def _item_inputs(self, items: ItemFeatures):
        year = items.year
        lo = np.nanmin(year) if np.isfinite(year).any() else 0
        bucket = np.where(np.isfinite(year), (np.nan_to_num(year, nan=lo) - lo) // self.cfg.year_bucket + 1, 0)
        self.year_lo_ = float(lo)
        n_years = int(bucket.max()) + 1
        t = lambda a: torch.as_tensor(a, device=self.device)
        return (t(np.arange(len(year))), t(pad_lists(items.genres)), t(pad_lists(items.tags)),
                t(bucket.astype(np.int64))), n_years

    def _item_embeddings(self) -> torch.Tensor:
        return self.net.item(*self.item_inputs_)

    # ----- training ----------------------------------------------------------
    def fit(self, fold: Fold, items: ItemFeatures, eval_fn=None) -> "TwoTower":
        """Train on fold.seqs. If eval_fn(model) -> float is given, early-stop on it."""
        super().fit(fold, items)
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        rng = np.random.default_rng(cfg.seed)
        self.n_items_ = fold.n_items
        self.item_inputs_, n_years = self._item_inputs(items)
        self.net = TwoTowerNet(fold.n_users, fold.n_items, items.n_genres, items.n_tags,
                               n_years, cfg).to(self.device)

        # Every liked movie is one example; its history is what the user liked before it.
        lengths = np.array([len(s) for s in fold.seqs])
        flat = np.concatenate([s for s in fold.seqs if len(s)]) if lengths.sum() else np.zeros(0, int)
        offsets = np.r_[0, np.cumsum(lengths)[:-1]]
        ex_user = np.repeat(np.arange(fold.n_users), lengths)
        ex_pos = np.arange(len(flat))
        freq = np.bincount(flat, minlength=fold.n_items).astype(np.float64)
        p_item = torch.as_tensor(freq / freq.sum(), dtype=torch.float32, device=self.device)

        opt = torch.optim.AdamW(self.net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        best, best_state, bad, self.history_ = -np.inf, None, 0, []
        arange_l = np.arange(-cfg.max_hist, 0)
        for epoch in range(cfg.epochs):
            self.net.train()
            t0, total, n_batches = time.time(), 0.0, 0
            perm = rng.permutation(len(ex_pos))
            for s in range(0, len(perm), cfg.batch_size):
                b = perm[s : s + cfg.batch_size]
                users, pos = ex_user[b], ex_pos[b]
                hidx = pos[:, None] + arange_l
                hist = np.where(hidx >= offsets[users][:, None], flat[np.maximum(hidx, 0)], -1)
                uid = users + 1
                uid[rng.random(len(uid)) < cfg.user_id_dropout] = 0
                loss = self._loss(torch.as_tensor(uid, device=self.device),
                                  torch.as_tensor(hist, device=self.device),
                                  torch.as_tensor(flat[pos], device=self.device), p_item, rng)
                opt.zero_grad()
                loss.backward()
                opt.step()
                total += loss.item()
                n_batches += 1
            rec = {"epoch": epoch + 1, "loss": total / max(n_batches, 1), "secs": time.time() - t0}
            if eval_fn is not None:
                self._cache_items()
                rec["val"] = eval_fn(self)
            self.history_.append(rec)
            if self.verbose:
                print(f"  epoch {rec['epoch']:2d} loss {rec['loss']:.4f}"
                      + (f" val {rec['val']:.4f}" if "val" in rec else "") + f" ({rec['secs']:.0f}s)",
                      flush=True)
            if eval_fn is not None:
                if rec["val"] > best:
                    best, bad = rec["val"], 0
                    best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        break
        if best_state is not None:
            self.net.load_state_dict(best_state)
            self.best_epoch_ = int(np.argmax([h["val"] for h in self.history_]) + 1)
        else:
            self.best_epoch_ = len(self.history_)
        self._cache_items()
        return self

    def _loss(self, uid, hist, target, p_item, rng) -> torch.Tensor:
        cfg = self.cfg
        u = self.net.user(uid, hist)                                     # (B, d)
        cand = target
        if cfg.n_uniform_neg:
            extra = torch.as_tensor(rng.integers(0, self.n_items_, cfg.n_uniform_neg), device=self.device)
            cand = torch.cat([target, extra])
        iid, genres, tags, year = self.item_inputs_
        v = self.net.item(iid[cand], genres[cand], tags[cand], year[cand])  # (B + M, d)
        logits = u @ v.T / cfg.temperature
        if cfg.logq:
            B, M = len(target), cfg.n_uniform_neg
            # Expected number of times each candidate appears among the negatives.
            expected = B * p_item[cand]
            if M:
                expected = expected + M / self.n_items_
            logits = logits - torch.log(expected)[None, :]
        # A batch can hold the same movie twice; never treat it as its own negative.
        same = target[:, None] == cand[None, :]
        same[torch.arange(len(target)), torch.arange(len(target))] = False
        logits = logits.masked_fill(same, float("-inf"))
        labels = torch.arange(len(target), device=self.device)
        return F.cross_entropy(logits, labels)

    # ----- inference ---------------------------------------------------------
    @torch.no_grad()
    def _cache_items(self):
        self.net.eval()
        self.item_emb_ = self._item_embeddings().float().cpu().numpy()

    @torch.no_grad()
    def user_embeddings(self, ev: EvalSet) -> np.ndarray:
        self.net.eval()
        uid = torch.as_tensor(np.where(ev.users >= 0, ev.users + 1, 0), device=self.device)
        hist = torch.as_tensor(last_n(ev.hist_seqs, self.cfg.max_hist), device=self.device)
        return self.net.user(uid, hist).float().cpu().numpy()

    def score(self, ev: EvalSet) -> np.ndarray:
        return self.user_embeddings(ev) @ self.item_emb_.T

    def params(self):
        return asdict(self.cfg)

    # ----- export for torch-free serving ---------------------------------------
    def export_user_tower(self) -> dict[str, np.ndarray]:
        sd = {k: v.detach().float().cpu().numpy() for k, v in self.net.state_dict().items()}
        return {
            "user_id": sd["user_id.weight"], "hist": sd["hist.emb.weight"],
            "w1": sd["user_mlp.0.weight"], "b1": sd["user_mlp.0.bias"],
            "w2": sd["user_mlp.2.weight"], "b2": sd["user_mlp.2.bias"],
            "max_hist": np.array(self.cfg.max_hist),
        }
