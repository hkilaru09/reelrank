"""Load exported artifacts and answer recommendation requests."""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp

from ..metrics import mask_seen, topk
from ..ranker import build_features
from ..retrieval import ItemIndex
from ..split import EvalSet


class NumpyUserTower:
    """Same forward pass as TwoTowerNet.user, in NumPy."""

    def __init__(self, weights: dict[str, np.ndarray]):
        self.w = weights
        self.max_hist = int(weights["max_hist"])

    def __call__(self, uid: np.ndarray, hist: np.ndarray) -> np.ndarray:
        """uid: dense user index + 1 (0 = unknown). hist: (n, L) item ids, -1 padded."""
        w = self.w
        e = w["hist"][hist + 1]
        cnt = np.maximum((hist >= 0).sum(1, keepdims=True), 1)
        x = np.concatenate([w["user_id"][uid], e.sum(1) / cnt], axis=1)
        h = np.maximum(x @ w["w1"].T + w["b1"], 0) @ w["w2"].T + w["b2"]
        return (h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)).astype(np.float32)


class Engine:
    def __init__(self, artifacts: str | Path):
        d = Path(artifacts)
        self.meta = json.loads((d / "meta.json").read_text())
        self.tower = NumpyUserTower(dict(np.load(d / "user_tower.npz")))
        self.index = ItemIndex(np.load(d / "item_emb.npy"))
        self.ranker = lgb.Booster(model_file=str(d / "ranker.txt"))
        st = np.load(d / "item_stats.npz")
        self.stats = SimpleNamespace(**{k: st[k] for k in st.files})
        self.stats.end_ts = float(self.stats.end_ts)
        u = np.load(d / "users.npz")
        self.users = {k: u[k] for k in u.files}
        self.user_index = {int(r): i for i, r in enumerate(self.users["raw_ids"]) if self.users["in_train"][i]}
        items = json.loads((d / "items.json").read_text())
        self.item_raw = np.asarray(items["raw_ids"])
        self.item_index = {int(r): i for i, r in enumerate(self.item_raw)}
        self.titles, self.genres = items["titles"], items["genres"]
        self.n_items = len(self.item_raw)
        self.n_candidates = int(self.meta["n_candidates"])

    # ------------------------------------------------------------------ requests
    def known_user(self, raw_id: int) -> bool:
        return raw_id in self.user_index

    def for_user(self, raw_id: int, k: int = 10) -> dict:
        u = self.user_index[raw_id]
        U = self.users
        seq = U["seq_flat"][U["seq_ptr"][u]: U["seq_ptr"][u + 1]]
        seen = U["seen_idx"][U["seen_ptr"][u]: U["seen_ptr"][u + 1]]
        return self._recommend(u, seq, seen, int(U["last_ts"][u]), k)

    def for_history(self, liked_raw: list[int], k: int = 10, now_ts: int | None = None) -> dict:
        """A user the model has never seen, described only by movies they liked."""
        seq = np.array([self.item_index[m] for m in liked_raw if m in self.item_index], dtype=np.int64)
        return self._recommend(-1, seq, seq, int(now_ts or self.stats.end_ts), k)

    def _recommend(self, u: int, seq: np.ndarray, seen: np.ndarray, last_ts: int, k: int) -> dict:
        t0 = time.perf_counter()
        ev = self._eval_set(u, seq, seen, last_ts)
        if len(seq) == 0:
            # Nothing liked yet: same fallback the offline evaluation uses.
            recs = topk(mask_seen(self.stats.pop_scores[None, :].copy(), ev.hist_seen), k)[0]
            return self._response(recs, np.full(len(recs), np.nan), "popular_fallback", t0)
        emb = self.tower(np.array([u + 1]), self._last_n(seq))
        cs, ci = self.index.search(emb, self.n_candidates, ev.hist_seen)
        t1 = time.perf_counter()
        # One thread: 200 rows are too few to split, and per-request OpenMP thread
        # pools fighting across workers were the source of the p99 tail under load.
        pred = self.ranker.predict(build_features(ev, ci, cs, self.stats), num_threads=1).reshape(ci.shape)
        pred[ci < 0] = -np.inf
        order = np.argsort(-pred[0], kind="stable")[:k]
        out = self._response(ci[0][order], pred[0][order], "two_tower+ranker", t0)
        out["timing_ms"]["retrieve"] = round((t1 - t0) * 1000, 3)
        return out

    def _last_n(self, seq: np.ndarray) -> np.ndarray:
        h = np.full((1, self.tower.max_hist), -1, dtype=np.int64)
        tail = seq[-self.tower.max_hist:]
        h[0, : len(tail)] = tail
        return h

    def _eval_set(self, u: int, seq: np.ndarray, seen: np.ndarray, last_ts: int) -> EvalSet:
        row = lambda idx: sp.csr_matrix((np.ones(len(idx), np.float32), (np.zeros(len(idx), int), idx)),
                                        shape=(1, self.n_items))
        return EvalSet(np.array([u]), row(seq), row(seen), [seq], np.array([last_ts]), [np.zeros(0, int)])

    def _response(self, ids: np.ndarray, scores: np.ndarray, source: str, t0: float) -> dict:
        ids = ids[ids >= 0]
        return {
            "source": source,
            "items": [{"movie_id": int(self.item_raw[i]), "title": self.titles[i], "genres": self.genres[i],
                       "score": None if not np.isfinite(s) else round(float(s), 5)}
                      for i, s in zip(ids, scores)],
            "timing_ms": {"total": round((time.perf_counter() - t0) * 1000, 3)},
        }
