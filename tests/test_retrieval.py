import numpy as np
import scipy.sparse as sp

from reelrank.metrics import mask_seen, topk
from reelrank.retrieval import ItemIndex


def test_flat_index_matches_exact_numpy(rng):
    items = rng.normal(size=(500, 16)).astype(np.float32)
    users = rng.normal(size=(30, 16)).astype(np.float32)
    seen = sp.random(30, 500, density=0.05, format="csr", random_state=0)
    seen.data[:] = 1
    _, ids = ItemIndex(items, "flat").search(users, 50, seen)
    exact = topk(mask_seen(users @ items.T, seen), 50)
    assert np.array_equal(ids, exact)


def test_hnsw_recovers_most_of_exact(rng):
    items = rng.normal(size=(2000, 16)).astype(np.float32)
    users = rng.normal(size=(20, 16)).astype(np.float32)
    _, a = ItemIndex(items, "flat").search(users, 100)
    _, b = ItemIndex(items, "hnsw").search(users, 100)
    overlap = np.mean([len(np.intersect1d(x, y)) / 100 for x, y in zip(a, b)])
    assert overlap > 0.9


def test_pads_when_catalog_smaller_than_k(rng):
    items = rng.normal(size=(5, 4)).astype(np.float32)
    seen = sp.csr_matrix(np.array([[1, 0, 0, 0, 0]]))
    s, ids = ItemIndex(items).search(rng.normal(size=(1, 4)).astype(np.float32), 10, seen)
    assert (ids[0, :4] >= 0).all() and (ids[0, 4:] == -1).all()
    assert 0 not in ids[0]
