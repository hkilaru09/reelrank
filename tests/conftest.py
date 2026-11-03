import numpy as np
import pytest

from reelrank.split import make_split
from reelrank.synthetic import make_movielens_like

TINY_TT = dict(dim=16, hidden=32, n_uniform_neg=64, batch_size=256, epochs=3, verbose=False, device="cpu")


@pytest.fixture(scope="session")
def frames():
    return make_movielens_like(n_users=200, n_items=120, seed=1)


@pytest.fixture(scope="session")
def split(frames):
    return make_split(*frames, seed=0)


@pytest.fixture(scope="session")
def pipeline(split):
    """A tiny trained two-tower + ranker, built exactly as the experiment builds it."""
    from reelrank.ranker import TwoStage
    from reelrank.twotower import TwoTower

    tt_valid = TwoTower(**TINY_TT).fit(split.valid, split.items)
    tt_test = TwoTower(**TINY_TT).fit(split.test, split.items)
    ranker = TwoStage(tt_valid, n_candidates=50, lgb_params={"n_estimators": 30, "min_child_samples": 5})
    ranker.fit_ranker(split.valid.eval, split.valid, split.items)
    ranker.attach(tt_test, split.test, split.items)
    return tt_test, ranker


@pytest.fixture(scope="session")
def artifacts(split, pipeline, tmp_path_factory):
    from reelrank.serving.export import export_artifacts

    tt, ranker = pipeline
    return export_artifacts(split, tt, ranker, tmp_path_factory.mktemp("artifacts"))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
