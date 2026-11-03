"""Second stage: LightGBM LambdaRank over the two-tower's top candidates.

Features for each (user, candidate) pair:

* retrieval: two-tower score, its rank in the candidate list, gap to the top score
* popularity: log count of positives, log count in the last 90 days of training, percentile
* genre match: cosine between the user's liked-genre profile and the movie's genres
* recency: movie age when the user last rated, gap between the movie's year and
  the mean year of the user's liked movies, days since the user's last rating ended
  (relative to the end of training)
* user context: number of liked movies in the history

An optional variant (`extra=`) also feeds the scores of other fitted models
(ALS, item-kNN) for each candidate, to separate "LambdaRank cannot help" from
"these features cannot help".

The ranker is trained on the validation window (candidates from a two-tower
fit on train_inner) and evaluated on the test window (candidates from a
two-tower refit on the full training set), so it never sees test labels.
"""

from __future__ import annotations

import numpy as np
import lightgbm as lgb

from .base import Recommender
from .metrics import auto_batch, mask_seen, topk
from .retrieval import ItemIndex
from .split import EvalSet, Fold, ItemFeatures

DAY = 86400.0
YEAR = 365.25 * DAY
FEATURES = ["tt_score", "tt_rank", "tt_gap", "pop_log", "pop_recent_log", "pop_pct",
            "genre_match", "item_age", "year_gap", "user_idle_days", "user_n_hist"]


class ItemStats:
    """Item-level statistics computed from one training fold."""

    def __init__(self, fold: Fold, items: ItemFeatures, recent_days: float = 90):
        pos = fold.train[fold.train["rating"] >= 4]
        self.end_ts = float(fold.train["ts"].max())
        pop = np.bincount(pos["item"], minlength=fold.n_items).astype(float)
        recent = pos[pos["ts"] >= self.end_ts - recent_days * DAY]
        self.pop_log = np.log1p(pop)
        self.pop_recent_log = np.log1p(np.bincount(recent["item"], minlength=fold.n_items))
        pct = np.empty(fold.n_items)
        pct[np.argsort(pop, kind="stable")] = np.linspace(0, 1, fold.n_items)
        self.pop_pct = pct
        g = items.genre_matrix().toarray()
        self.genres = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-9)
        self.genre_raw = g
        self.year = items.year


def build_features(ev: EvalSet, cand_ids: np.ndarray, cand_scores: np.ndarray,
                   stats: ItemStats) -> np.ndarray:
    """(n_users * n_cand, n_features) matrix, rows grouped by user in candidate order."""
    n, c = cand_ids.shape
    ids = np.maximum(cand_ids, 0)
    valid = cand_ids >= 0
    s = np.where(valid, cand_scores, np.nan)

    hist_genre = ev.hist_pos @ stats.genre_raw                         # (n, n_genres)
    hist_genre = hist_genre / np.maximum(np.linalg.norm(hist_genre, axis=1, keepdims=True), 1e-9)
    genre_match = np.einsum("ng,ncg->nc", hist_genre, stats.genres[ids])

    n_hist = np.diff(ev.hist_pos.indptr).astype(float)
    year_sum = ev.hist_pos @ np.nan_to_num(stats.year, nan=0.0)
    year_cnt = ev.hist_pos @ np.isfinite(stats.year).astype(float)
    user_mean_year = np.where(year_cnt > 0, year_sum / np.maximum(year_cnt, 1), np.nan)
    user_year = 1970 + ev.hist_last_ts / YEAR

    item_year = stats.year[ids]
    feats = np.stack([
        s,
        np.broadcast_to(np.arange(c), (n, c)).astype(float),
        np.nanmax(s, axis=1, keepdims=True) - s,
        stats.pop_log[ids],
        stats.pop_recent_log[ids],
        stats.pop_pct[ids],
        genre_match,
        user_year[:, None] - item_year,
        item_year - user_mean_year[:, None],
        np.broadcast_to(((stats.end_ts - ev.hist_last_ts) / DAY)[:, None], (n, c)),
        np.broadcast_to(np.log1p(n_hist)[:, None], (n, c)),
    ], axis=-1)
    return feats.reshape(n * c, len(FEATURES)).astype(np.float32)


def labels_for(cand_ids: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    return np.stack([np.isin(cand_ids[i], t) for i, t in enumerate(targets)]).astype(np.int32)


class TwoStage(Recommender):
    """Two-tower + FAISS candidates, re-ranked by LambdaRank."""
    name = "two_tower+ranker"

    def __init__(self, retriever, n_candidates: int = 200, lgb_params: dict | None = None,
                 index_kind: str = "flat", seed: int = 0, extra: dict[str, Recommender] | None = None):
        self.retriever = retriever
        self.extra = extra or {}
        if self.extra:
            self.name = "two_tower+ranker+cf"
        self.feature_names_ = FEATURES + [f"{n}_score" for n in self.extra]
        self.n_candidates = n_candidates
        self.index_kind = index_kind
        self.seed = seed
        self.lgb_params = {"objective": "lambdarank", "metric": "ndcg", "eval_at": [10, 20],
                           "learning_rate": 0.05, "num_leaves": 31, "min_child_samples": 100,
                           "feature_fraction": 0.9, "lambdarank_truncation_level": 30,
                           "n_estimators": 1000, "verbose": -1, **(lgb_params or {})}

    def candidates(self, ev: EvalSet) -> tuple[np.ndarray, np.ndarray]:
        return self.index_.search(self.retriever.user_embeddings(ev), self.n_candidates, ev.hist_seen)

    def attach(self, retriever, fold: Fold, items: ItemFeatures,
               extra: dict[str, Recommender] | None = None) -> "TwoStage":
        """Point the stage-1 index (and any extra scorers) at models refitted on `fold`."""
        self.retriever = retriever
        if extra is not None:
            assert list(extra) == list(self.extra), "extra scorers must keep their order"
            self.extra = extra
        self.index_ = ItemIndex(retriever.item_emb_, self.index_kind)
        self.stats_ = ItemStats(fold, items)
        self.pop_scores_ = fold.item_pop().astype(np.float32)
        return self

    def fit_ranker(self, ev: EvalSet, fold: Fold, items: ItemFeatures, valid_frac: float = 0.2,
                   max_users: int = 20_000):
        """Learn the re-ranker on `ev` (a window the retriever was not trained on)."""
        self.attach(self.retriever, fold, items)
        if len(ev) > max_users:  # bounds memory: features are users x candidates x features
            ev = ev.subset(np.sort(np.random.default_rng(self.seed).choice(len(ev), max_users, replace=False)))
        cs, ci = self.candidates(ev)
        X = self._features(ev, ci, cs).reshape(len(ev), self.n_candidates, -1)
        y = labels_for(ci, ev.targets)
        self.train_candidate_recall_ = float(np.mean(
            [y[i].sum() / len(t) for i, t in enumerate(ev.targets)]))
        has_pos = y.sum(1) > 0                       # users with no hit carry no ranking signal
        users = np.flatnonzero(has_pos)
        rng = np.random.default_rng(self.seed)
        rng.shuffle(users)
        n_val = int(len(users) * valid_frac)
        va, tr = users[:n_val], users[n_val:]
        flat = lambda idx: (X[idx].reshape(-1, X.shape[-1]), y[idx].ravel(),
                            np.full(len(idx), self.n_candidates))
        Xtr, ytr, gtr = flat(tr)
        Xva, yva, gva = flat(va)
        self.model_ = lgb.LGBMRanker(**{k: v for k, v in self.lgb_params.items() if k != "eval_at"},
                                     random_state=self.seed)
        self.model_.fit(Xtr, ytr, group=gtr, eval_set=[(Xva, yva)], eval_group=[gva],
                        eval_at=self.lgb_params["eval_at"], feature_name=self.feature_names_,
                        callbacks=[lgb.early_stopping(50, verbose=False)])
        return self

    def recommend(self, ev: EvalSet, k: int) -> np.ndarray:
        cs, ci = self.candidates(ev)
        X = self._features(ev, ci, cs)
        pred = self.model_.predict(X, num_iteration=self.model_.best_iteration_)
        pred = pred.reshape(ci.shape)
        pred[ci < 0] = -np.inf
        order = np.argsort(-pred, axis=1, kind="stable")[:, :k]
        recs = np.take_along_axis(ci, order, axis=1)
        # Same rule as every other model: no liked movies at all -> most popular.
        empty = np.flatnonzero((ev.users < 0) & (np.diff(ev.hist_pos.indptr) == 0))
        if len(empty):
            pop = np.tile(self.pop_scores_, (len(empty), 1))
            recs[empty] = topk(mask_seen(pop, ev.hist_seen[empty]), recs.shape[1])
        if recs.shape[1] < k:  # asked for more than the candidate list holds
            recs = np.hstack([recs, np.full((len(recs), k - recs.shape[1]), -1)])
        return recs

    def _features(self, ev: EvalSet, ci: np.ndarray, cs: np.ndarray) -> np.ndarray:
        X = build_features(ev, ci, cs, self.stats_)
        if not self.extra:
            return X
        cols = np.full((len(ev), ci.shape[1], len(self.extra)), np.nan, dtype=np.float32)
        b = auto_batch(len(self.pop_scores_))
        for s in range(0, len(ev), b):   # full score rows are big: gather a batch at a time
            rows = np.arange(s, min(s + b, len(ev)))
            sub = ev.subset(rows)
            for j, model in enumerate(self.extra.values()):
                cols[rows, :, j] = np.take_along_axis(np.asarray(model.score(sub)), np.maximum(ci[rows], 0), 1)
        cols[ci < 0] = np.nan
        return np.hstack([X, cols.reshape(-1, len(self.extra))])

    def score(self, ev):
        raise NotImplementedError("the ranker only scores its candidates; see recommend()")

    def feature_importance(self) -> dict[str, float]:
        gain = self.model_.booster_.feature_importance("gain")
        return dict(zip(self.feature_names_, (gain / gain.sum()).round(4).tolist()))

    def params(self):
        return {"n_candidates": self.n_candidates, "index": self.index_kind, "extra": list(self.extra),
                **{f"lgb_{k}": v for k, v in self.lgb_params.items() if k != "eval_at"}}
