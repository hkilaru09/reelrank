"""Full-catalog evaluation, plus the sampled-negative protocol for comparison.

Full catalog: every item the user has not rated in their history is a
candidate, so Recall@K means "of the movies this user went on to like, how
many made the top K out of all ~N items".

Sampled (Krichene & Rendle, 2020): each held-out positive is ranked against
only 100 random unseen items. It is cheap and popular, and it can disagree
with the full-catalog ranking of models; `sampled_eval` exists to show
whether that happens here.
"""

from __future__ import annotations

import numpy as np

from .split import EvalSet


def mask_seen(scores: np.ndarray, seen) -> np.ndarray:
    rows, cols = seen.nonzero()
    scores[rows, cols] = -np.inf
    return scores


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def hit_matrix(recs: np.ndarray, targets: list[np.ndarray]) -> np.ndarray:
    hits = np.zeros(recs.shape, dtype=bool)
    for i, t in enumerate(targets):
        hits[i] = np.isin(recs[i], t)
    return hits


def ranking_metrics(recs: np.ndarray, targets: list[np.ndarray], ks=(10, 20, 50),
                    map_k: int = 100) -> dict[str, np.ndarray]:
    """Per-user Recall@K, NDCG@K and AP@map_k for (n_users, >=max K) recommendation lists."""
    hits = hit_matrix(recs, targets)
    n_rel = np.array([len(t) for t in targets], dtype=float)
    discount = 1.0 / np.log2(np.arange(recs.shape[1]) + 2)
    out = {}
    for k in ks:
        h = hits[:, :k]
        out[f"recall@{k}"] = h.sum(1) / n_rel
        idcg = np.array([discount[: int(min(k, n))].sum() for n in n_rel])
        out[f"ndcg@{k}"] = (h * discount[:k]).sum(1) / idcg
    h = hits[:, :map_k]
    prec_at = np.cumsum(h, axis=1) / np.arange(1, h.shape[1] + 1)
    out[f"map@{map_k}"] = (prec_at * h).sum(1) / np.minimum(n_rel, map_k)
    return out


def beyond_accuracy(recs: np.ndarray, item_pop: np.ndarray, k: int = 20) -> dict[str, float]:
    """Catalog coverage and popularity bias of the top-k lists.

    pop_pct: mean popularity percentile of recommended items (100 = the most
    popular movie in training). tail_share: share of recommendations outside
    the 20% most popular items.
    """
    r = recs[:, :k].ravel()
    r = r[r >= 0]
    n_items = len(item_pop)
    pct = np.empty(n_items)
    pct[np.argsort(item_pop, kind="stable")] = np.arange(n_items) / (n_items - 1) * 100
    head = pct >= 80
    return {
        f"coverage@{k}": len(np.unique(r)) / n_items,
        f"pop_pct@{k}": float(pct[r].mean()),
        f"tail_share@{k}": float((~head[r]).mean()),
    }


def auto_batch(n_items: int, budget: float = 2.5e7) -> int:
    """Users per scoring batch so a dense (users x items) block stays ~100 MB."""
    return int(np.clip(budget // max(n_items, 1), 32, 1024))


def evaluate(model, ev: EvalSet, item_pop: np.ndarray, ks=(10, 20, 50), map_k: int = 100,
             batch: int | None = None) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    """Recommend for every user in `ev` and score against the held-out targets."""
    batch = batch or auto_batch(len(item_pop))
    kmax = max(max(ks), map_k)
    recs = np.vstack([model.recommend(ev.subset(np.arange(s, min(s + batch, len(ev)))), kmax)
                      for s in range(0, len(ev), batch)])
    per_user = ranking_metrics(recs, ev.targets, ks, map_k)
    summary = {k: float(v.mean()) for k, v in per_user.items()}
    summary.update(beyond_accuracy(recs, item_pop, k=20))
    summary["n_users"] = len(ev)
    return summary, per_user


def sampled_eval(model, ev: EvalSet, n_neg: int = 100, k: int = 10, seed: int = 0,
                 batch: int | None = None) -> dict[str, float]:
    """HR@k and NDCG@k with each held-out positive ranked against n_neg random unseen items."""
    rng = np.random.default_rng(seed)
    n_items = ev.hist_seen.shape[1]
    batch = batch or auto_batch(n_items)
    hr, ndcg = [], []
    for s in range(0, len(ev), batch):
        sub = ev.subset(np.arange(s, min(s + batch, len(ev))))
        scores = model.score(sub)
        for i, targets in enumerate(sub.targets):
            excluded = np.zeros(n_items, dtype=bool)
            excluded[sub.hist_seen[i].indices] = True
            excluded[targets] = True
            pool = np.flatnonzero(~excluded)
            for t in targets:
                negs = rng.choice(pool, size=min(n_neg, len(pool)), replace=False)
                # Ties count against the positive: a constant scorer gets no credit.
                rank = int((scores[i, negs] >= scores[i, t]).sum())
                hr.append(rank < k)
                ndcg.append(1.0 / np.log2(rank + 2) if rank < k else 0.0)
    return {f"sampled_hr@{k}": float(np.mean(hr)), f"sampled_ndcg@{k}": float(np.mean(ndcg))}


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0) -> dict:
    """Mean of a - b over users with a 95% bootstrap interval (a, b are per-user metrics)."""
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    boots = d[idx].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi)}
