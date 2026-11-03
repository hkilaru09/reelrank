"""Candidate retrieval: a FAISS inner-product index over item-tower embeddings."""

from __future__ import annotations

import faiss

faiss.omp_set_num_threads(1)
import numpy as np


class ItemIndex:
    """Exact (flat) or approximate (HNSW) maximum-inner-product search.

    Seen items are filtered after the search, so it over-fetches by the
    largest history in the batch to still return `k` fresh candidates.
    """

    def __init__(self, item_emb: np.ndarray, kind: str = "flat", hnsw_m: int = 32,
                 ef_search: int = 256):
        self.item_emb = np.ascontiguousarray(item_emb, dtype=np.float32)
        d = self.item_emb.shape[1]
        if kind == "flat":
            self.index = faiss.IndexFlatIP(d)
        elif kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(d, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efSearch = ef_search
        else:
            raise ValueError(kind)
        self.index.add(self.item_emb)
        self.kind = kind

    @property
    def n_items(self) -> int:
        return self.item_emb.shape[0]

    def search(self, user_emb: np.ndarray, k: int, seen=None, max_fetch: int = 4096) -> tuple[np.ndarray, np.ndarray]:
        """Top-k unseen items per user: (scores, ids), ids padded with -1 if too few."""
        user_emb = np.ascontiguousarray(user_emb, dtype=np.float32)
        extra = int(np.diff(seen.indptr).max()) if seen is not None and seen.nnz else 0
        fetch = min(self.n_items, k + extra, max(k, max_fetch))
        scores, ids = self.index.search(user_emb, fetch)
        out_s = np.full((len(ids), k), -np.inf, dtype=np.float32)
        out_i = np.full((len(ids), k), -1, dtype=np.int64)
        for r in range(len(ids)):
            keep = ids[r] >= 0
            if seen is not None:
                keep &= ~np.isin(ids[r], seen[r].indices)
            sel_i, sel_s = ids[r][keep][:k], scores[r][keep][:k]
            out_i[r, : len(sel_i)] = sel_i
            out_s[r, : len(sel_s)] = sel_s
        return out_s, out_i
