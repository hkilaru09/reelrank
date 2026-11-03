import numpy as np

from reelrank import metrics
from reelrank.ranker import FEATURES, ItemStats, build_features, labels_for


def test_feature_matrix_shape_and_values(pipeline, split):
    _, ranker = pipeline
    ev = split.test.eval.subset(np.arange(10))
    cs, ci = ranker.candidates(ev)
    X = build_features(ev, ci, cs, ItemStats(split.test, split.items))
    assert X.shape == (10 * ranker.n_candidates, len(FEATURES))
    tt_rank = X[:, FEATURES.index("tt_rank")].reshape(10, -1)
    assert (tt_rank[:, 0] == 0).all()
    gm = X[:, FEATURES.index("genre_match")]
    assert (gm >= -1e-6).all() and (gm <= 1 + 1e-6).all()


def test_labels():
    y = labels_for(np.array([[3, 1, 2], [4, 5, 6]]), [np.array([2]), np.array([9])])
    assert y.tolist() == [[0, 0, 1], [0, 0, 0]]


def test_two_stage_returns_unseen_candidates(pipeline, split):
    _, ranker = pipeline
    ev = split.test.eval
    recs = ranker.recommend(ev, 20)
    cs, ci = ranker.candidates(ev)
    for i in range(len(ev)):
        assert set(recs[i]) <= set(ci[i])
        assert not set(recs[i]) & set(ev.hist_seen[i].indices)
    summary, _ = metrics.evaluate(ranker, ev, split.test.item_pop())
    assert 0 < summary["recall@20"] <= 1


def test_ranker_with_extra_scorers(split, pipeline):
    from reelrank.baselines import ALS
    from reelrank.ranker import TwoStage

    tt, _ = pipeline
    als_v = ALS(factors=8, iterations=5).fit(split.valid, split.items)
    als_t = ALS(factors=8, iterations=5).fit(split.test, split.items)
    r = TwoStage(tt, n_candidates=30, extra={"als": als_v},
                 lgb_params={"n_estimators": 20, "min_child_samples": 5})
    r.fit_ranker(split.valid.eval, split.valid, split.items)
    r.attach(tt, split.test, split.items, extra={"als": als_t})
    assert r.feature_names_[-1] == "als_score"
    ev = split.test.eval.subset(np.arange(8))
    cs, ci = r.candidates(ev)
    X = r._features(ev, ci, cs).reshape(8, 30, -1)
    expected = np.take_along_axis(als_t.score(ev), ci, 1)
    np.testing.assert_allclose(X[:, :, -1], expected, rtol=1e-5)
    assert r.recommend(ev, 10).shape == (8, 10)
