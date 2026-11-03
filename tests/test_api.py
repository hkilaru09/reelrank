import numpy as np
import pytest
from fastapi.testclient import TestClient

from reelrank.serving.engine import Engine


@pytest.fixture(scope="module")
def client(artifacts, monkeypatch_module):
    from reelrank.serving import app as app_mod

    monkeypatch_module.setenv("REELRANK_ARTIFACTS", str(artifacts))
    app_mod.engine.cache_clear()
    with TestClient(app_mod.app) as c:
        yield c
    app_mod.engine.cache_clear()


@pytest.fixture(scope="module")
def monkeypatch_module():
    mp = pytest.MonkeyPatch()
    yield mp
    mp.undo()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["n_items"] > 0


def test_recommend_known_user(client, split):
    raw = int(split.user_raw_ids[split.test.eval.users[0]])
    body = client.get(f"/recommend/{raw}?k=5").json()
    assert body["source"] == "two_tower+ranker"
    assert len(body["items"]) == 5
    assert len({i["movie_id"] for i in body["items"]}) == 5


def test_unknown_user_404(client):
    assert client.get("/recommend/987654321").status_code == 404


def test_new_user_by_history(client, split):
    liked = [int(x) for x in split.items.raw_ids[:3]]
    body = client.post("/recommend", json={"liked": liked, "k": 5}).json()
    assert body["source"] == "two_tower+ranker"
    assert not {i["movie_id"] for i in body["items"]} & set(liked)
    assert client.post("/recommend", json={"liked": [], "k": 5}).json()["source"] == "popular_fallback"


def test_k_validated(client, split):
    raw = int(split.user_raw_ids[split.test.eval.users[0]])
    assert client.get(f"/recommend/{raw}?k=0").status_code == 422


def test_served_recs_match_offline_pipeline(artifacts, pipeline, split):
    """The torch-free server must return exactly what the evaluated pipeline returns."""
    _, ranker = pipeline
    engine = Engine(artifacts)
    ev = split.test.eval
    offline = ranker.recommend(ev.subset(np.arange(15)), 10)
    for row in range(15):
        raw = int(split.user_raw_ids[ev.users[row]])
        served = [engine.item_index[i["movie_id"]] for i in engine.for_user(raw, 10)["items"]]
        assert served == offline[row].tolist()
