"""Load MovieLens into one tidy format.

Every dataset becomes two frames:

* interactions: user, item, rating, ts   (raw MovieLens ids, ts in seconds)
* items:        item, title, year, genres (list[str]), tags (list[str])

Downstream code never sees the raw file layouts.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

YEAR_RE = re.compile(r"\((\d{4})\)\s*$")


def _parse_year(title: str) -> float:
    m = YEAR_RE.search(title.strip())
    return float(m.group(1)) if m else np.nan


def load_ml1m(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    ratings = pd.read_csv(
        root / "ratings.dat", sep="::", engine="python", encoding="latin-1",
        names=["user", "item", "rating", "ts"],
    )
    movies = pd.read_csv(
        root / "movies.dat", sep="::", engine="python", encoding="latin-1",
        names=["item", "title", "genres"],
    )
    movies["year"] = movies["title"].map(_parse_year)
    movies["genres"] = movies["genres"].str.split("|")
    movies["tags"] = [[] for _ in range(len(movies))]  # ML-1M ships no tags
    return ratings, movies[["item", "title", "year", "genres", "tags"]]


def load_ml32m(root: str | Path, top_tags: int = 1000, min_tag_count: int = 3,
               max_tags_per_item: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    root = Path(root)
    ratings = pd.read_csv(
        root / "ratings.csv", engine="pyarrow",
        dtype={"userId": "int32", "movieId": "int32", "rating": "float32", "timestamp": "int64"},
    ).rename(columns={"userId": "user", "movieId": "item", "timestamp": "ts"})
    movies = pd.read_csv(root / "movies.csv").rename(columns={"movieId": "item"})
    movies["year"] = movies["title"].map(_parse_year)
    movies["genres"] = movies["genres"].str.split("|").map(
        lambda g: [] if g == ["(no genres listed)"] else g)

    tags = pd.read_csv(root / "tags.csv", usecols=["movieId", "tag"]).dropna()
    tags["tag"] = tags["tag"].astype(str).str.strip().str.lower()
    vocab = tags["tag"].value_counts().head(top_tags).index
    tags = tags[tags["tag"].isin(vocab)]
    counts = tags.groupby(["movieId", "tag"]).size().rename("n").reset_index()
    counts = counts[counts["n"] >= min_tag_count].sort_values(["movieId", "n"], ascending=[True, False])
    per_item = counts.groupby("movieId")["tag"].apply(lambda s: list(s[:max_tags_per_item]))
    movies["tags"] = movies["item"].map(per_item).map(lambda t: t if isinstance(t, list) else [])
    return ratings, movies[["item", "title", "year", "genres", "tags"]]


LOADERS = {"ml-1m": load_ml1m, "ml-32m": load_ml32m}


def load(name: str, raw_dir: str | Path = "data/raw") -> tuple[pd.DataFrame, pd.DataFrame]:
    return LOADERS[name](Path(raw_dir) / name)
