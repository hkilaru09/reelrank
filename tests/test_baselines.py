import numpy as np
import pytest

from reelrank import metrics
from reelrank.baselines import ALS, ItemKNN, MostPopular


def _recall(model, split, ev=None):
    ev = ev or split.test.eval
    return metrics.evaluate(model, ev, split.test.item_pop())[0]["recall@20"]


def test_personalised_models_beat_popularity(split):
    pop = _recall(MostPopular().fit(split.test, split.items), split)
    for model in (ItemKNN(k=50, shrink=10), ALS(factors=16, iterations=10)):
        assert _recall(model.fit(split.test, split.items), split) > pop


def test_itemknn_matches_dense_cosine(split):
    model = ItemKNN(k=10_000, shrink=0, block=7).fit(split.test, split.items)
    X = split.test.train_pos.toarray()
    co = X.T @ X
    n = np.sqrt(np.diag(co))
    dense = co / (np.outer(n, n) + 1e-9)
    np.fill_diagonal(dense, 0)
    np.testing.assert_allclose(model.sim_.toarray(), dense, rtol=1e-4, atol=1e-6)


def test_als_fold_in_recovers_trained_user(split):
    """Folding a trained user's own history back in should land near their learned vector."""
    model = ALS(factors=8, regularization=0.1, alpha=10, iterations=30).fit(split.test, split.items)
    u = int(np.argmax(np.diff(split.test.train_pos.indptr)))
    folded = model._fold_in(split.test.train_pos[[u]])[0]
    learned = model.user_factors_[u]
    cos = folded @ learned / np.linalg.norm(folded) / np.linalg.norm(learned)
    assert cos > 0.95


def test_cold_users_get_history_based_recs(split):
    model = ALS(factors=16, iterations=10).fit(split.test, split.items)
    recs = model.recommend(split.cold, 10)
    assert recs.shape == (len(split.cold), 10)
    for i in range(len(split.cold)):
        assert not set(recs[i]) & set(split.cold.hist_seen[i].indices)


@pytest.mark.parametrize("half_life", [None, 30])
def test_popular_is_user_independent(split, half_life):
    model = MostPopular(half_life).fit(split.test, split.items)
    s = model.score(split.test.eval)
    assert (s == s[0]).all()


def test_recent_history_keeps_mask_and_trims_history(split):
    from reelrank.baselines import recent_history

    ev = split.test.eval
    r = recent_history(ev, 3)
    assert (r.users == -1).all()
    assert np.diff(r.hist_pos.indptr).max() <= 3
    assert (r.hist_seen != ev.hist_seen).nnz == 0
    for i in range(5):
        assert set(r.hist_pos[i].indices) == set(ev.hist_seqs[i][-3:])
