import numpy as np
import pytest
import scipy.sparse as sp

from reelrank import metrics
from reelrank.split import EvalSet


def test_recall_ndcg_map_by_hand():
    recs = np.array([[5, 1, 7, 2]])
    targets = [np.array([1, 2, 9])]
    m = metrics.ranking_metrics(recs, targets, ks=(2, 4), map_k=4)
    assert m["recall@2"][0] == pytest.approx(1 / 3)
    assert m["recall@4"][0] == pytest.approx(2 / 3)
    dcg = 1 / np.log2(3) + 1 / np.log2(5)
    idcg = 1 + 1 / np.log2(3) + 1 / np.log2(4)
    assert m["ndcg@4"][0] == pytest.approx(dcg / idcg)
    # hits at ranks 2 and 4: precision 1/2 and 2/4, divided by min(3, 4)
    assert m["map@4"][0] == pytest.approx((0.5 + 0.5) / 3)


def test_topk_and_mask_seen():
    scores = np.array([[0.1, 0.9, 0.5, 0.7]], dtype=np.float32)
    seen = sp.csr_matrix(np.array([[0, 1, 0, 0]]))
    assert metrics.topk(metrics.mask_seen(scores.copy(), seen), 2).tolist() == [[3, 2]]


def test_beyond_accuracy():
    pop = np.array([10.0, 1.0, 5.0, 0.0, 3.0])
    out = metrics.beyond_accuracy(np.array([[0, 2], [0, 4]]), pop, k=2)
    assert out["coverage@2"] == pytest.approx(3 / 5)
    assert out["pop_pct@2"] == pytest.approx((100 + 75 + 100 + 50) / 4)


class _Fixed:
    def __init__(self, scores):
        self.scores = scores

    def score(self, ev):
        return np.tile(self.scores, (len(ev), 1))


def _ev(n_items=50):
    seen = sp.csr_matrix((1, n_items))
    return EvalSet(np.array([0]), seen, seen, [np.zeros(0, int)], np.array([0]), [np.array([3])])


def test_sampled_eval_oracle_and_constant():
    oracle = np.zeros(50)
    oracle[3] = 1
    assert metrics.sampled_eval(_Fixed(oracle), _ev(), n_neg=20)["sampled_hr@10"] == 1.0
    # ties count against the positive, so a constant scorer earns nothing
    assert metrics.sampled_eval(_Fixed(np.zeros(50)), _ev(), n_neg=20)["sampled_hr@10"] == 0.0


def test_paired_bootstrap_contains_true_difference(rng):
    a = rng.normal(1.0, 1, 2000)
    b = a - 0.2 + rng.normal(0, 0.1, 2000)
    out = metrics.paired_bootstrap(a, b)
    assert out["lo"] < 0.2 < out["hi"]
