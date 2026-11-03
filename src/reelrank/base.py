"""The interface every model implements, so one harness evaluates them all."""

from __future__ import annotations

import numpy as np

from .metrics import mask_seen, topk
from .split import EvalSet, Fold, ItemFeatures


class Recommender:
    name = "base"

    def fit(self, fold: Fold, items: ItemFeatures) -> "Recommender":
        # Fallback for a brand-new user who has not liked anything yet.
        self.pop_scores_ = fold.item_pop().astype(np.float32)
        return self

    def score(self, ev: EvalSet) -> np.ndarray:
        """(len(ev), n_items) relevance scores, higher is better; seen items not masked."""
        raise NotImplementedError

    def recommend(self, ev: EvalSet, k: int) -> np.ndarray:
        scores = np.array(self.score(ev), dtype=np.float32, copy=True)
        empty = (ev.users < 0) & (np.diff(ev.hist_pos.indptr) == 0)
        scores[empty] = self.pop_scores_
        return topk(mask_seen(scores, ev.hist_seen), k)

    def params(self) -> dict:
        return {}
