"""Temporal per-user split, implicit-feedback labels and the cold-start slice.

For every warm user, ratings are sorted by time and the last `test_frac` of
them (at least one) are held out as the test window. The same rule applied
again to what is left gives the validation window used for tuning. Only
ratings >= `pos_threshold` count as positives: they are the training signal
and the targets. Everything a user rated in their history, positive or not,
is masked from their recommendations.

A random `cold_frac` of users never appear in training. At evaluation they
reveal only their first `cold_k` ratings (fewer than 5) and are scored on the
same last-window targets a warm user would be, so the cold-start slice
answers: what can each model do with almost no history?
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import scipy.sparse as sp


@dataclass
class ItemFeatures:
    titles: np.ndarray
    raw_ids: np.ndarray
    year: np.ndarray              # float, NaN when unknown
    genres: list[np.ndarray]      # genre indices per item
    tags: list[np.ndarray]        # tag indices per item
    genre_names: list[str]
    tag_names: list[str]

    @property
    def n_genres(self) -> int:
        return len(self.genre_names)

    @property
    def n_tags(self) -> int:
        return len(self.tag_names)

    def genre_matrix(self) -> sp.csr_matrix:
        rows = np.repeat(np.arange(len(self.genres)), [len(g) for g in self.genres])
        cols = np.concatenate(self.genres) if rows.size else np.array([], dtype=int)
        return sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                             shape=(len(self.genres), max(self.n_genres, 1)))


@dataclass
class EvalSet:
    """Users to score, what each model may see about them, and what they did next."""
    users: np.ndarray             # dense user index, -1 if the model never saw the user
    hist_pos: sp.csr_matrix       # (n_eval, n_items) history positives
    hist_seen: sp.csr_matrix      # (n_eval, n_items) everything rated in history
    hist_seqs: list[np.ndarray]   # chronological history positives
    hist_last_ts: np.ndarray      # time of the last history rating
    targets: list[np.ndarray]     # positives in the held-out window

    def __len__(self) -> int:
        return len(self.users)

    def subset(self, idx: np.ndarray) -> "EvalSet":
        return EvalSet(self.users[idx], self.hist_pos[idx], self.hist_seen[idx],
                       [self.hist_seqs[i] for i in idx], self.hist_last_ts[idx],
                       [self.targets[i] for i in idx])


@dataclass
class Fold:
    """A training set and the warm users' evaluation window that follows it."""
    n_users: int
    n_items: int
    train: pd.DataFrame           # user, item, rating, ts (dense indices)
    train_pos: sp.csr_matrix      # (n_users, n_items) positives
    train_seen: sp.csr_matrix     # (n_users, n_items) all ratings
    seqs: list[np.ndarray]        # chronological positives per user
    seq_ts: list[np.ndarray]
    eval: EvalSet
    extra: dict = field(default_factory=dict)

    def item_pop(self) -> np.ndarray:
        return np.asarray(self.train_pos.sum(axis=0)).ravel().astype(np.float64)


@dataclass
class Split:
    items: ItemFeatures
    valid: Fold                   # train_inner -> validation window (tuning, ranker training)
    test: Fold                    # train -> test window (reported numbers)
    cold: EvalSet                 # cold-start users, scored by models fit on test.train
    user_raw_ids: np.ndarray
    cold_users: np.ndarray        # dense ids of the cold-start users


def _csr(users, items, n_users, n_items) -> sp.csr_matrix:
    m = sp.csr_matrix((np.ones(len(users), dtype=np.float32), (users, items)),
                      shape=(n_users, n_items))
    m.sum_duplicates()
    m.data[:] = 1.0
    return m


def _sequences(df: pd.DataFrame, n_users: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Chronological item lists per user (df must already be time-sorted per user)."""
    seqs = [np.zeros(0, dtype=np.int64) for _ in range(n_users)]
    ts = [np.zeros(0, dtype=np.int64) for _ in range(n_users)]
    if len(df):
        users = df["user"].to_numpy()
        bounds = np.flatnonzero(np.diff(users)) + 1
        starts = np.r_[0, bounds]
        items_all, ts_all = df["item"].to_numpy(), df["ts"].to_numpy()
        for s, e in zip(starts, np.r_[bounds, len(df)]):
            seqs[users[s]] = items_all[s:e].astype(np.int64)
            ts[users[s]] = ts_all[s:e].astype(np.int64)
    return seqs, ts


def _tail_mask(df: pd.DataFrame, frac: float) -> np.ndarray:
    """True for each user's last `frac` of rows (at least one). df sorted by user, time."""
    n = df.groupby("user", sort=False)["user"].transform("size").to_numpy()
    rank = df.groupby("user", sort=False).cumcount().to_numpy()
    n_tail = np.maximum(1, np.round(frac * n)).astype(int)
    return rank >= n - n_tail


def _eval_set(hist: pd.DataFrame, future: pd.DataFrame, users: np.ndarray, user_index: np.ndarray,
              n_items: int, pos_threshold: float) -> EvalSet:
    """users: dense ids of the people to evaluate; user_index: id given to the model (-1 = unknown)."""
    pos_future = future[future["rating"] >= pos_threshold]
    targets_by_user = pos_future.groupby("user")["item"].apply(lambda s: s.to_numpy())
    keep = np.array([u in targets_by_user.index for u in users], dtype=bool)
    users, user_index = users[keep], user_index[keep]
    row_of = pd.Series(np.arange(len(users)), index=users)

    h = hist[hist["user"].isin(users)]
    rows = row_of.loc[h["user"]].to_numpy()
    seen = _csr(rows, h["item"].to_numpy(), len(users), n_items)
    hp = h[h["rating"] >= pos_threshold]
    pos = _csr(row_of.loc[hp["user"]].to_numpy(), hp["item"].to_numpy(), len(users), n_items)
    hp_rows = hp.assign(user=row_of.loc[hp["user"]].to_numpy()).sort_values(["user", "ts"], kind="stable")
    seqs, _ = _sequences(hp_rows, len(users))
    last_ts = h.groupby("user")["ts"].max().reindex(users).fillna(0).to_numpy().astype(np.int64)
    targets = [targets_by_user[u] for u in users]
    return EvalSet(user_index.astype(np.int64), pos, seen, seqs, last_ts, targets)


def _fold(train: pd.DataFrame, future: pd.DataFrame, n_users: int, n_items: int,
          warm_users: np.ndarray, pos_threshold: float) -> Fold:
    pos = train[train["rating"] >= pos_threshold]
    seqs, seq_ts = _sequences(pos, n_users)
    ev = _eval_set(train, future, warm_users, warm_users, n_items, pos_threshold)
    return Fold(n_users, n_items, train.reset_index(drop=True),
                _csr(pos["user"].to_numpy(), pos["item"].to_numpy(), n_users, n_items),
                _csr(train["user"].to_numpy(), train["item"].to_numpy(), n_users, n_items),
                seqs, seq_ts, ev)


def build_items(items: pd.DataFrame, item_ids: np.ndarray) -> ItemFeatures:
    items = items.set_index("item").reindex(item_ids)
    genres = sorted({g for gs in items["genres"].dropna() for g in gs})
    tags = sorted({t for ts in items["tags"].dropna() for t in ts})
    g_idx = {g: i for i, g in enumerate(genres)}
    t_idx = {t: i for i, t in enumerate(tags)}
    as_list = lambda v: v if isinstance(v, list) else []
    return ItemFeatures(
        titles=items["title"].fillna("").to_numpy(),
        raw_ids=np.asarray(item_ids),
        year=items["year"].to_numpy(dtype=float),
        genres=[np.array([g_idx[g] for g in as_list(gs)], dtype=np.int64) for gs in items["genres"]],
        tags=[np.array([t_idx[t] for t in as_list(ts)], dtype=np.int64) for ts in items["tags"]],
        genre_names=genres, tag_names=tags,
    )


def make_split(ratings: pd.DataFrame, items: pd.DataFrame, test_frac: float = 0.1,
               pos_threshold: float = 4.0, cold_frac: float = 0.1, cold_k: int = 4,
               seed: int = 0) -> Split:
    rng = np.random.default_rng(seed)
    df = ratings[["user", "item", "rating", "ts"]].copy()
    item_ids = np.sort(df["item"].unique())
    user_ids = np.sort(df["user"].unique())
    df["item"] = np.searchsorted(item_ids, df["item"].to_numpy())
    df["user"] = np.searchsorted(user_ids, df["user"].to_numpy())
    n_users, n_items = len(user_ids), len(item_ids)

    # Many ratings share a timestamp; break ties randomly rather than by item id.
    df["tie"] = rng.random(len(df))
    df = df.sort_values(["user", "ts", "tie"], kind="stable").drop(columns="tie").reset_index(drop=True)

    cold_users = np.sort(rng.choice(n_users, size=int(round(cold_frac * n_users)), replace=False))
    is_cold = np.zeros(n_users, dtype=bool)
    is_cold[cold_users] = True
    warm_users = np.flatnonzero(~is_cold)

    test_mask = _tail_mask(df, test_frac)
    warm_rows = ~is_cold[df["user"].to_numpy()]
    train = df[~test_mask & warm_rows]
    test = df[test_mask & warm_rows]
    val_mask = _tail_mask(train, test_frac)
    train_inner, val = train[~val_mask], train[val_mask]

    valid_fold = _fold(train_inner, val, n_users, n_items, warm_users, pos_threshold)
    test_fold = _fold(train, test, n_users, n_items, warm_users, pos_threshold)

    cold_df = df[is_cold[df["user"].to_numpy()]]
    cold_hist = cold_df[cold_df.groupby("user").cumcount().to_numpy() < cold_k]
    cold_future = cold_df[test_mask[is_cold[df["user"].to_numpy()]]]
    cold = _eval_set(cold_hist, cold_future, cold_users, np.full(len(cold_users), -1),
                     n_items, pos_threshold)

    return Split(build_items(items, item_ids), valid_fold, test_fold, cold, user_ids, cold_users)
