"""Small synthetic data in MovieLens shape, for tests and CI (no real data needed).

Users and movies get latent taste vectors; each user rates movies chosen in
proportion to taste + popularity, over time, with ratings driven by affinity,
so every model has real structure to find.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GENRES = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi"]
TAGS = ["classic", "twist", "funny", "dark", "slow", "visual"]


def make_movielens_like(n_users: int = 300, n_items: int = 200, dim: int = 6,
                        min_ratings: int = 20, max_ratings: int = 60,
                        seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n_users, dim))
    V = rng.normal(size=(n_items, dim))
    pop = rng.normal(scale=1.0, size=n_items)

    rows = []
    t0 = 946684800  # 2000-01-01
    for u in range(n_users):
        n = rng.integers(min_ratings, max_ratings + 1)
        logits = U[u] @ V.T / np.sqrt(dim) + pop
        p = np.exp(logits - logits.max())
        p /= p.sum()
        chosen = rng.choice(n_items, size=n, replace=False, p=p)
        aff = U[u] @ V[chosen].T / np.sqrt(dim)
        rating = np.clip(np.round(3.2 + 1.2 * aff + rng.normal(scale=0.7, size=n)), 1, 5)
        ts = t0 + np.sort(rng.integers(0, 3 * 365 * 86400, size=n))
        rows.append(pd.DataFrame({"user": u + 1, "item": chosen + 1, "rating": rating, "ts": ts}))
    ratings = pd.concat(rows, ignore_index=True)

    top_dim = np.argmax(V[:, : len(GENRES)], axis=1)
    movies = pd.DataFrame({
        "item": np.arange(1, n_items + 1),
        "title": [f"Movie {i} ({1950 + i % 50})" for i in range(1, n_items + 1)],
        "year": [float(1950 + i % 50) for i in range(1, n_items + 1)],
        "genres": [[GENRES[g]] + ([GENRES[(g + 1) % len(GENRES)]] if i % 3 == 0 else [])
                   for i, g in enumerate(top_dim)],
        "tags": [[TAGS[g]] if i % 2 else [] for i, g in enumerate(top_dim)],
    })
    return ratings, movies
