import numpy as np
import torch

from reelrank.serving.engine import NumpyUserTower
from reelrank.twotower import TwoTower, last_n
from conftest import TINY_TT


def test_training_loss_decreases(split):
    model = TwoTower(**{**TINY_TT, "epochs": 4}).fit(split.test, split.items)
    losses = [h["loss"] for h in model.history_]
    assert losses[-1] < losses[0]


def test_numpy_user_tower_matches_torch(pipeline, split):
    tt, _ = pipeline
    ev = split.test.eval.subset(np.arange(20))
    ref = tt.user_embeddings(ev)
    tower = NumpyUserTower(tt.export_user_tower())
    out = tower(np.where(ev.users >= 0, ev.users + 1, 0), last_n(ev.hist_seqs, tt.cfg.max_hist))
    np.testing.assert_allclose(out, ref, atol=1e-5)


def test_cold_user_uses_unknown_slot_and_history(pipeline, split):
    tt, _ = pipeline
    emb = tt.user_embeddings(split.cold)
    assert np.isfinite(emb).all()
    np.testing.assert_allclose(np.linalg.norm(emb, axis=1), 1, atol=1e-5)
    # different histories -> different users
    assert np.unique(emb.round(5), axis=0).shape[0] > 1


def test_logq_correction_and_duplicate_masking(split):
    model = TwoTower(**{**TINY_TT, "epochs": 0, "n_uniform_neg": 0}).fit(split.test, split.items)
    rng = np.random.default_rng(0)
    uid = torch.tensor([1, 2, 3])
    hist = torch.full((3, model.cfg.max_hist), -1)
    target = torch.tensor([5, 5, 7])          # rows 0 and 1 share a positive
    p_item = torch.full((split.test.n_items,), 1.0 / split.test.n_items)
    # with a uniform item distribution logQ shifts every logit equally -> same loss
    with torch.no_grad():
        model.cfg.logq = True
        a = model._loss(uid, hist, target, p_item, rng).item()
        model.cfg.logq = False
        b = model._loss(uid, hist, target, p_item, rng).item()
    assert np.isclose(a, b, atol=1e-5)
    # the loss stays finite even though a duplicate positive is masked to -inf
    assert np.isfinite(a)
