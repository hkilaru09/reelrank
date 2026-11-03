import numpy as np

from reelrank.split import make_split


def test_train_strictly_precedes_test_per_user(split):
    tr = split.test.train
    last_train = tr.groupby("user")["ts"].max()
    for row, u in enumerate(split.test.eval.users):
        # every target was rated no earlier than anything in that user's training history
        assert split.test.eval.hist_last_ts[row] == last_train[u]
    assert len(split.test.eval) > 0


def test_targets_are_unseen_positives(split):
    ev = split.test.eval
    for i in range(len(ev)):
        seen = set(ev.hist_seen[i].indices)
        assert len(ev.targets[i]) > 0
        assert not seen & set(ev.targets[i])


def test_validation_train_is_prefix_of_test_train(split):
    inner = set(map(tuple, split.valid.train[["user", "item"]].to_numpy()))
    full = set(map(tuple, split.test.train[["user", "item"]].to_numpy()))
    assert inner < full


def test_cold_users_never_trained_on_and_reveal_under_five(split):
    cold_rows = split.cold.users
    assert (cold_rows == -1).all()
    assert np.diff(split.cold.hist_seen.indptr).max() <= 4
    assert len(split.cold_users) > 0
    for fold in (split.valid, split.test):
        assert not set(fold.train["user"].unique()) & set(split.cold_users)


def test_positive_threshold(frames):
    ratings, movies = frames
    s = make_split(ratings, movies, pos_threshold=5.0)
    pos = s.test.train[s.test.train["rating"] >= 5]
    assert s.test.train_pos.nnz == len(pos.drop_duplicates(["user", "item"]))


def test_split_is_deterministic(frames):
    a, b = make_split(*frames, seed=3), make_split(*frames, seed=3)
    assert a.test.train.equals(b.test.train)
    assert all(np.array_equal(x, y) for x, y in zip(a.test.eval.targets, b.test.eval.targets))
