"""FastAPI service.

    GET  /recommend/{user_id}?k=10   known user: two-tower -> FAISS top 200 -> LambdaRank
    POST /recommend                  new user: {"liked": [movie ids], "k": 10}
    GET  /health
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from .engine import Engine

app = FastAPI(title="ReelRank", version="0.1.0")


@lru_cache(maxsize=1)
def engine() -> Engine:
    return Engine(os.environ.get("REELRANK_ARTIFACTS", "artifacts/ml-1m"))


class History(BaseModel):
    liked: list[int] = Field(default_factory=list, description="MovieLens movie ids the user liked")
    k: int = Field(10, ge=1, le=100)


@app.on_event("startup")
def _warm() -> None:
    engine()


@app.get("/health")
def health() -> dict:
    e = engine()
    return {"status": "ok", "n_items": e.n_items, "n_users": len(e.user_index)}


@app.get("/recommend/{user_id}")
def recommend(user_id: int, k: int = Query(10, ge=1, le=100)) -> dict:
    e = engine()
    if not e.known_user(user_id):
        raise HTTPException(404, f"user {user_id} not in training data; POST /recommend with their liked movies")
    return {"user_id": user_id, **e.for_user(user_id, k)}


@app.post("/recommend")
def recommend_new_user(body: History) -> dict:
    return {"user_id": None, **engine().for_history(body.liked, body.k)}
