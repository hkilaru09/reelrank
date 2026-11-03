"""Most-popular, item-kNN and ALS baselines."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from .base import Recommender
from .split import EvalSet, Fold, ItemFeatures

DAY = 86400


class MostPopular(Recommender):
    """Positive counts, optionally decayed by age so recent hits count more."""
    name = "popular"

    def __init__(self, half_life_days: float | None = None):
        self.half_life_days = half_life_days

    def fit(self, fold, items):
        super().fit(fold, items)
        pos = fold.train[fold.train["rating"] >= 4]
        w = np.ones(len(pos))
        if self.half_life_days:
            age = (pos["ts"].max() - pos["ts"].to_numpy()) / DAY
            w = 0.5 ** (age / self.half_life_days)
        self.scores_ = np.bincount(pos["item"], weights=w, minlength=fold.n_items).astype(np.float32)
        return self

    def score(self, ev):
        return np.broadcast_to(self.scores_, (len(ev), len(self.scores_)))

    def params(self):
        return {"half_life_days": self.half_life_days}


class ItemKNN(Recommender):
    """Item-item cosine with shrinkage, keeping the k nearest neighbours of each item.

    score(u, j) = sum over liked items i of sim(i, j). Built in column blocks so
    it scales to catalogs where a dense item x item matrix would not fit.
    """
    name = "item_knn"

    def __init__(self, k: int = 100, shrink: float = 10.0, block: int | None = None):
        self.k, self.shrink, self.block = k, shrink, block

    def fit(self, fold, items):
        super().fit(fold, items)
        X = fold.train_pos.tocsc().astype(np.float32)
        norms = np.sqrt(np.asarray(X.multiply(X).sum(0)).ravel())
        XT = X.T.tocsr()
        n = X.shape[1]
        block = self.block or int(np.clip(2e7 // n, 64, 2048))   # ~80 MB dense blocks
        rows, cols, vals = [], [], []
        for s in range(0, n, block):
            e = min(s + block, n)
            co = (XT @ X[:, s:e]).toarray()                  # (n, e-s) co-occurrence
            sim = co / (np.outer(norms, norms[s:e]) + self.shrink + 1e-9)
            sim[np.arange(s, e), np.arange(e - s)] = 0.0      # no self-similarity
            k = min(self.k, n - 1)
            top = np.argpartition(-sim, k - 1, axis=0)[:k]     # (k, e-s)
            v = np.take_along_axis(sim, top, axis=0)
            keep = v > 0
            rows.append(top[keep])
            cols.append(np.broadcast_to(np.arange(s, e), top.shape)[keep])
            vals.append(v[keep])
        self.sim_ = sp.csr_matrix((np.concatenate(vals).astype(np.float32),
                                   (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        return self

    def score(self, ev):
        return (ev.hist_pos @ self.sim_).toarray()

    def params(self):
        return {"k": self.k, "shrink": self.shrink}


class ALS(Recommender):
    """Implicit-feedback matrix factorisation (Hu, Koren & Volinsky 2008) via `implicit`.

    Users the model never saw are folded in with one exact least-squares step
    against the fixed item factors, which is what ALS would do for them anyway.
    """
    name = "als"

    def __init__(self, factors: int = 128, regularization: float = 0.1, alpha: float = 10.0,
                 iterations: int = 20, seed: int = 0):
        self.factors, self.regularization, self.alpha = factors, regularization, alpha
        self.iterations, self.seed = iterations, seed

    def fit(self, fold, items):
        from implicit.cpu.als import AlternatingLeastSquares

        super().fit(fold, items)
        with threadpool_limits(1, "blas"):
            model = AlternatingLeastSquares(
                factors=self.factors, regularization=self.regularization, alpha=self.alpha,
                iterations=self.iterations, random_state=self.seed, calculate_training_loss=False)
            model.fit(fold.train_pos, show_progress=False)
        self.user_factors_ = np.asarray(model.user_factors)
        self.item_factors_ = np.asarray(model.item_factors)
        return self

    def _fold_in(self, hist: sp.csr_matrix) -> np.ndarray:
        V = self.item_factors_
        gram = V.T @ V + self.regularization * np.eye(V.shape[1], dtype=V.dtype)
        out = np.zeros((hist.shape[0], V.shape[1]), dtype=V.dtype)
        for r in range(hist.shape[0]):
            idx = hist[r].indices
            if len(idx):
                Vi = V[idx]
                out[r] = np.linalg.solve(gram + self.alpha * Vi.T @ Vi, (1 + self.alpha) * Vi.sum(0))
        return out

    def user_vectors(self, ev: EvalSet) -> np.ndarray:
        U = np.zeros((len(ev), self.item_factors_.shape[1]), dtype=np.float32)
        known = ev.users >= 0
        U[known] = self.user_factors_[ev.users[known]]
        if (~known).any():
            U[~known] = self._fold_in(ev.hist_pos[np.flatnonzero(~known)])
        return U

    def score(self, ev):
        return self.user_vectors(ev) @ self.item_factors_.T

    def params(self):
        return {"factors": self.factors, "regularization": self.regularization,
                "alpha": self.alpha, "iterations": self.iterations}


def recent_history(ev: EvalSet, k: int) -> EvalSet:
    """The same users described only by their k most recent likes, as unknown users.

    Seen items stay masked in full; only what the model is told changes.
    """
    seqs = [s[-k:] for s in ev.hist_seqs]
    rows = np.repeat(np.arange(len(seqs)), [len(s) for s in seqs])
    cols = np.concatenate(seqs) if len(rows) else np.zeros(0, dtype=np.int64)
    pos = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=ev.hist_pos.shape)
    return EvalSet(np.full(len(ev), -1), pos, ev.hist_seen, seqs, ev.hist_last_ts, ev.targets)


class RecentHistory(Recommender):
    """Score a fitted history-based model (item-kNN, ALS fold-in) from the last k likes only.

    A control for the two-tower, which pools only a user's latest likes: does
    recency alone explain its advantage on a temporal split?
    """

    def __init__(self, model: Recommender, k: int):
        self.model, self.k = model, k
        self.name = f"{model.name}_last{k}"

    def fit(self, fold, items):
        super().fit(fold, items)
        self.model.fit(fold, items)
        return self

    def score(self, ev):
        return self.model.score(recent_history(ev, self.k))

    def params(self):
        return {**self.model.params(), "recent_k": self.k}
