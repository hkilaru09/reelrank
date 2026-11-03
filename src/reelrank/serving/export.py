"""Write everything the API needs into one directory (no torch required to load it)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _ragged(lists) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([len(x) for x in lists], dtype=np.int64)
    flat = np.concatenate([np.asarray(x, dtype=np.int64) for x in lists]) if lengths.sum() else np.zeros(0, np.int64)
    return np.r_[0, np.cumsum(lengths)], flat


def export_artifacts(split, two_tower, ranker, out: str | Path) -> Path:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    fold, items, stats = split.test, split.items, ranker.stats_

    np.savez(out / "user_tower.npz", **two_tower.export_user_tower())
    np.save(out / "item_emb.npy", two_tower.item_emb_.astype(np.float32))
    ranker.model_.booster_.save_model(str(out / "ranker.txt"), num_iteration=ranker.model_.best_iteration_)
    np.savez(out / "item_stats.npz", pop_log=stats.pop_log, pop_recent_log=stats.pop_recent_log,
             pop_pct=stats.pop_pct, genres=stats.genres, genre_raw=stats.genre_raw, year=stats.year,
             end_ts=np.array(stats.end_ts), pop_scores=fold.item_pop().astype(np.float32))

    seq_ptr, seq_flat = _ragged(fold.seqs)
    seen = fold.train_seen.tocsr()
    last_ts = fold.train.groupby("user")["ts"].max().reindex(range(fold.n_users)).fillna(0).to_numpy()
    in_train = np.diff(seen.indptr) > 0      # cold-start users were held out of training
    np.savez(out / "users.npz", raw_ids=split.user_raw_ids, in_train=in_train,
             seq_ptr=seq_ptr, seq_flat=seq_flat, seen_ptr=seen.indptr, seen_idx=seen.indices,
             last_ts=last_ts.astype(np.int64))

    (out / "items.json").write_text(json.dumps({
        "raw_ids": items.raw_ids.tolist(), "titles": items.titles.tolist(),
        "genres": [[items.genre_names[g] for g in gs] for gs in items.genres]}))
    (out / "meta.json").write_text(json.dumps({
        "n_candidates": ranker.n_candidates, "two_tower": two_tower.params(),
        "ranker_features": ranker.model_.booster_.feature_name()}, indent=2))
    return out
