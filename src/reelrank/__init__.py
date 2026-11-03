"""ReelRank: a two-stage (retrieve, then rank) movie recommender."""

import os

# The macOS wheels of torch and faiss each bundle their own libomp, and loading
# both aborts the process. Allowing the duplicate is the standard workaround;
# tests/test_retrieval.py checks FAISS against an exact NumPy top-k so a wrong
# result would not go unnoticed. The serving image ships no torch at all.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
