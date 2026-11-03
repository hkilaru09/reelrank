# ReelRank: a two-stage movie recommender, measured honestly

ReelRank retrieves 200 candidate movies with a **two-tower neural model** and a **FAISS** index, then re-ranks them
with **LightGBM LambdaRank**. It is served by a **FastAPI** endpoint in **Docker**. The evaluation harness ranks the
**full catalog** (not 100 sampled negatives) on a **temporal** split. The project asks two things: does the
two-stage system beat a properly tuned ALS, and which parts of it actually earn their keep?

## TL;DR (MovieLens-1M, 5,195 test users, all 3,706 movies ranked)

- **The two-tower beats tuned ALS by +0.061 Recall@20** (0.238 vs 0.178; 95% bootstrap CI +0.053 to +0.069). **Almost half of
  that is recency, not deep learning.** ALS that is told only each user's last 5 likes closes 0.027 of the 0.061 gap. The
  two-tower pools only a user's 20 most recent likes, and a temporal split rewards exactly that.
- **The ranker, with the features in the brief, barely helps.** Its features are the two-tower score, popularity, genre match
  and recency. It adds **+0.005 Recall@20** (CI +0.001 to +0.008) and makes recommendations *more* popular (popularity
  percentile 86.9 → 90.8). Give the same LambdaRank ranker ALS and item-kNN scores as features and it adds **+0.026**
  (CI +0.022 to +0.031), reaching **Recall@20 0.264**, the best model here. The ranker was limited by its features, not by the method.
- **For brand-new users (4 ratings), most-popular beats every personalized model**, the two-tower included (Recall@20 0.054 vs 0.045).
- **Sampled metrics would have picked the wrong model.** Rank each held-out movie against 100 random negatives and the
  two-tower comes **3rd of 4** (HR@10 0.630 vs ALS 0.682). Rank the full catalog and it is **1st, 41% ahead of ALS** (Recall@10 0.154 vs
  0.109). This is the effect Krichene & Rendle (2020) warn about.
- **logQ correction is the most important training choice.** Without it, validation Recall@20 drops 0.271 → 0.234.
- **At scale the picture changes, and the README says how** (ML-32M, 20% user sample, 50,034 movies). The two-tower's lead over ALS
  shrinks to +0.015 Recall@20 and the recency explanation stops working. Meanwhile the ranker gains more (+0.008 with the brief's
  features, +0.032 with ALS/kNN scores), and most-popular still wins cold start.
- **Serving**: in Docker on an M2 laptop, **p99 = 8.6 ms at 100 req/s and 11.7 ms at 600 req/s** (k6, zero errors). A load
  test found a tail-latency bug: LightGBM spun up an OpenMP thread pool per request. Fixing it cut p99 at 100 req/s from 27.4 ms to 8.6 ms.

## Results: MovieLens-1M

Each user's last 10% of ratings (by time) is held out; a rating of 4 or more is a positive. Every metric ranks all 3,706 movies, minus the ones
the user already rated. Hyperparameters were chosen on a validation window carved the same way out of the training data;
the test window was used once. *pop pct* is the average popularity percentile of the recommended movies (100 = the most popular
movie). *coverage* is the share of the catalog that appears in anyone's top 20.

| Model | Recall@10 | Recall@20 | Recall@50 | NDCG@10 | NDCG@20 | MAP@100 | coverage@20 | pop pct@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Most popular | 0.0528 | 0.0900 | 0.1680 | 0.0599 | 0.0694 | 0.0355 | 0.059 | 99.4 |
| Item-kNN | 0.0894 | 0.1518 | 0.2761 | 0.0871 | 0.1061 | 0.0584 | 0.376 | 95.7 |
| ALS (`implicit`) | 0.1092 | 0.1775 | 0.3140 | 0.1010 | 0.1217 | 0.0690 | 0.285 | 95.1 |
| Item-kNN, last 10 likes only | 0.1083 | 0.1742 | 0.2950 | 0.1012 | 0.1218 | 0.0684 | 0.694 | 87.9 |
| ALS fold-in, last 5 likes only | 0.1331 | 0.2045 | 0.3306 | 0.1249 | 0.1468 | 0.0839 | 0.373 | 91.7 |
| Two-tower (retrieval only) | 0.1540 | 0.2380 | 0.3884 | 0.1419 | 0.1677 | 0.1022 | 0.643 | 86.9 |
| Two-tower + LambdaRank | 0.1539 | 0.2426 | 0.3913 | 0.1440 | 0.1709 | 0.1026 | 0.600 | 90.8 |
| Two-tower + LambdaRank + ALS/kNN scores | **0.1729** | **0.2641** | **0.4172** | **0.1631** | **0.1902** | **0.1154** | 0.556 | 91.0 |

Paired bootstrap over users (2,000 resamples), Recall@20 difference with 95% interval:

| Comparison | Recall@20 diff | NDCG@20 diff |
|---|---:|---:|
| Two-tower − ALS | +0.0605 [+0.0525, +0.0686] | +0.0460 [+0.0403, +0.0521] |
| ALS on last 5 likes − ALS | +0.0271 [+0.0200, +0.0338] | +0.0250 [+0.0199, +0.0301] |
| Two-tower + ranker − two-tower | +0.0046 [+0.0010, +0.0083] | +0.0032 [+0.0008, +0.0057] |
| Two-tower + ranker + ALS/kNN scores − two-tower | +0.0261 [+0.0218, +0.0305] | +0.0225 [+0.0196, +0.0253] |

### Cold start: 575 users the models never saw, 4 ratings each

These users were removed from training entirely. At test time each one reveals only their first 4 ratings (2.2 positives on
average) and is scored on the same last-10% window as a warm user. ALS folds them in with one least-squares step. The
two-tower uses its "unknown user" slot plus the pooled history; it was trained with the user ID randomly dropped for exactly
this case. Users with no positives at all get the popular list from every model.

| Model | Recall@10 | Recall@20 | Recall@50 | NDCG@20 | coverage@20 | pop pct@20 |
|---|---:|---:|---:|---:|---:|---:|
| Most popular | **0.0286** | **0.0539** | **0.1228** | **0.0343** | 0.006 | 99.7 |
| Item-kNN | 0.0171 | 0.0374 | 0.0896 | 0.0258 | 0.352 | 93.7 |
| ALS (fold-in) | 0.0178 | 0.0392 | 0.0903 | 0.0259 | 0.228 | 95.9 |
| Two-tower | 0.0229 | 0.0436 | 0.0946 | 0.0286 | 0.179 | 97.2 |
| Two-tower + LambdaRank | 0.0248 | 0.0449 | 0.0974 | 0.0296 | 0.163 | 97.5 |
| Two-tower + LambdaRank + ALS/kNN scores | 0.0260 | 0.0479 | 0.1029 | 0.0328 | 0.166 | 97.8 |

The two-tower does beat ALS here, but the difference is not significant (+0.004 Recall@20, CI −0.006 to +0.015). Most-popular beats ALS
significantly (+0.015, CI +0.003 to +0.027). With two liked movies to go on, a personal guess is worse than the crowd's.
The two-tower's recommendations for these users are already 97th-percentile popular. It has partly learned that.

### Sampled vs full-catalog evaluation

| Model | Sampled HR@10 (1 vs 100 random) | Sampled NDCG@10 | Full-catalog Recall@10 | Full-catalog NDCG@10 |
|---|---:|---:|---:|---:|
| Most popular | 0.5224 | 0.2938 | 0.0528 | 0.0599 |
| Item-kNN | 0.6517 | 0.3807 | 0.0894 | 0.0871 |
| ALS | **0.6819** | **0.4036** | 0.1092 | 0.1010 |
| Two-tower | 0.6300 | 0.3970 | **0.1540** | **0.1419** |

Sampled HR@10 orders the models ALS > item-kNN > two-tower > popular. The full catalog orders them two-tower > ALS > item-kNN > popular.
The likely mechanism: random negatives are mostly obscure movies. The two-tower is trained with logQ correction and has
genre/year features, so it gives obscure movies more credit than ALS does. That makes it "lose" to easy negatives that a real
ranking never has to beat. (Sampled numbers use 2,000 test users; ties count against the positive.)

### What mattered in the two-tower (validation window)

Each line changes one setting from the base model (dim 64, temperature 0.05, last 50 likes). The model is early-stopped on
validation Recall@20.

| Variant | Recall@20 | NDCG@20 | coverage@20 | pop pct@20 |
|---|---:|---:|---:|---:|
| base (3 seeds) | 0.2528 / 0.2569 / 0.2598 | 0.1682 / 0.1720 / 0.1767 | 0.71–0.73 | 86.3–86.9 |
| temperature 0.1 | 0.2617 | 0.1765 | 0.627 | 87.2 |
| temperature 0.02 | 0.2586 | 0.1702 | 0.743 | 86.0 |
| dim 128 | 0.2625 | 0.1761 | 0.758 | 85.1 |
| last 20 likes | 0.2657 | 0.1769 | 0.724 | 86.4 |
| last 100 likes | 0.2573 | 0.1706 | 0.715 | 86.2 |
| learning rate 5e-4 | 0.2469 | 0.1618 | 0.715 | 86.1 |
| no logQ correction | 0.2188 | 0.1410 | 0.793 | 79.2 |
| no extra uniform negatives | 0.2567 | 0.1725 | 0.764 | 84.6 |
| no user-ID dropout | 0.2564 | 0.1715 | 0.719 | 86.2 |
| user ID always dropped (history only) | 0.2180 | 0.1441 | 0.608 | 89.5 |
| **combined: temperature 0.1, dim 128, last 20 (3 seeds)** | **0.2714 / 0.2727 / 0.2699** | 0.1854 / 0.1844 / 0.1841 | 0.63–0.64 | 86.8–87.3 |
| combined, no logQ correction | 0.2340 | 0.1517 | 0.730 | 80.3 |

Seed-to-seed spread is about 0.007 for the base model and 0.003 for the combined one. Single-setting changes of 0.005–0.013 are
therefore at most about twice the noise. The combined model's lead over base holds across all three seeds, and so do the two large
effects: logQ correction (−0.037) and the user-ID embedding (−0.035 when dropped). Removing logQ correction makes the model
*less* popularity-biased (pop pct 80) but less accurate. In-batch negatives are drawn in proportion to popularity, so without the
correction the model learns to push popular movies down too hard.

### Tuning the baselines

- **Most-popular**: positive counts with exponential time decay; 7 half-lives tried (best: 365 days, essentially no decay).
- **Item-kNN**: cosine with shrinkage, top-k neighbours; 20 configs (best: k = 50, shrink = 0).
- **ALS**: 129 configs. The first 60-config grid put the best setting on its edge (64 factors, regularization 10), so the grid was
  widened to 125 configs and then extended past the new edge. The final choice is 512 factors, regularization 30, alpha 2;
  going from 256 to 512 factors added only 0.0009, so ALS has plateaued.
- **Recency controls**: the tuned item-kNN and ALS scored from only a user's last *k* likes, with *k* in {5, 10, 20, 50}
  chosen on validation (item-kNN: 10, ALS: 5).

## Results: MovieLens-32M (20% user sample)

To check that the ML-1M story holds at scale, the run was repeated on a seeded 20% sample of ML-32M users: 40,189 users,
5.1M training ratings, and **50,034 movies ranked** (13.5× the ML-1M catalog). Movie tags are used here. The baseline grid was
smaller (ALS: 27 configs, again extended until 256 → 512 factors added under 0.001). Only the winning two-tower configuration and
its no-logQ ablation were trained. Test: 33,988 users, 268,856 held-out positives.

| Model | Recall@10 | Recall@20 | Recall@50 | NDCG@20 | MAP@100 | coverage@20 | pop pct@20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Most popular | 0.0516 | 0.0869 | 0.1555 | 0.0599 | 0.0317 | 0.006 | 100.0 |
| Item-kNN | 0.1017 | 0.1632 | 0.2861 | 0.1105 | 0.0633 | 0.097 | 99.5 |
| ALS (`implicit`) | 0.1260 | 0.2005 | 0.3430 | 0.1359 | 0.0789 | 0.044 | 99.5 |
| ALS fold-in, last 10 likes only | 0.1274 | 0.1985 | 0.3282 | 0.1343 | 0.0774 | 0.054 | 99.2 |
| Two-tower (retrieval only) | 0.1380 | 0.2150 | 0.3569 | 0.1441 | 0.0852 | 0.143 | 98.6 |
| Two-tower + LambdaRank | 0.1433 | 0.2234 | 0.3684 | 0.1515 | 0.0891 | 0.134 | 98.9 |
| Two-tower + LambdaRank + ALS/kNN scores | **0.1627** | **0.2469** | **0.3962** | **0.1736** | **0.1033** | 0.122 | 98.9 |

What changed with scale:

- **The two-tower's lead over ALS shrinks to +0.0145 Recall@20** (CI +0.012 to +0.017), from +0.061 on ML-1M. The recency
  control no longer helps ALS (−0.0020, CI −0.0041 to +0.0001). The ML-1M "it's mostly recency" explanation does not carry over. I have not
  isolated why. The sample has *fewer* positives per movie than ML-1M (52 vs 127), so it is not simply more data per item.
- **The ranker earns more.** The brief's features add +0.0084 (CI +0.0070 to +0.0097); adding ALS/kNN scores adds +0.0320
  (CI +0.0300 to +0.0338). The two-stage system with CF features beats ALS by +0.0465.
- **logQ correction matters even more.** Validation Recall@20 drops from 0.2389 to 0.1724 without it.
- **Cold start (3,783 users, 4 ratings each): most-popular still wins** (Recall@20 0.0661). Here the two-tower alone is
  significantly *worse* than ALS fold-in (−0.0055, CI −0.0102 to −0.0009). Only the CF-feature ranker gets close to popularity (0.0653).
- **Sampled metrics flip a different pair.** Sampled HR@10 ranks most-popular (0.899) above item-kNN (0.859) and ALS (0.947) just
  above the two-tower (0.946). On the full catalog, item-kNN has about twice popular's Recall@10, and the two-tower leads ALS.
- **Everything is popular.** Personalized recommendations sit at the 98.6–99.5th popularity percentile, and the best model shows just 12% of
  the catalog to anyone. With 50k movies, a long-tail fix (or diversity objective) would be the next project, not a tuning knob.
- Serving cost grows modestly. Offline retrieval plus ranking takes 1.35 ms per user (p99 2.3 ms), versus 2.3 ms (p99 5.8 ms) to
  score all 50k movies with ALS. HNSW returns 99.8% of the exact top 200, and the 200 candidates hold 62% of test positives.

## How it works

```
               ┌──────────── offline (src/reelrank) ─────────────┐
 ratings ──► temporal split ──► baselines (pop, item-kNN, ALS)  │
               │              └► two-tower (PyTorch) ──► item embeddings ──► FAISS index
               │                         │                                   │
               └► validation window ──► LambdaRank on 200 candidates ◄────────┘
                                          │
               ┌──────────── online (Docker, no torch) ───────────┐
 GET /recommend/{user} ─► NumPy user tower ─► FAISS top 200 (seen removed) ─► LightGBM ─► top k
```

**Two-tower** (`src/reelrank/twotower.py`)
- *Item tower*: movie-ID embedding, mean of genre embeddings, release-year bucket and mean of tag embeddings (ML-32M only;
  ML-1M has no tags), fed through an MLP and L2-normalized.
- *User tower*: user-ID embedding plus the mean embedding of the user's last *n* liked movies, fed through an MLP.
- *Training*: each liked movie is predicted from the movies liked before it. The softmax runs over the batch's other positives
  plus 1,024 uniformly sampled movies. Each negative's logit is reduced by the log of its expected count among the negatives
  (logQ correction, Yi et al. 2019). Duplicate positives in a batch are masked.
- *Cold start*: the user ID is replaced by an "unknown" slot 25% of the time, so the same tower serves new users from history alone.

**Ranker** (`src/reelrank/ranker.py`)
- LambdaRank is trained on the *validation* window, with candidates from a two-tower that never saw that window. It is then
  attached to a two-tower refit on the full training set and scored once on the test window.
- *Features*: two-tower score, rank and gap to the top score; all-time and recent popularity; cosine between the user's genre
  profile and the movie's genres; movie age when the user last rated; gap from the user's usual release year; user activity.
- Gain importance is 51% gap to the top score and 19% candidate rank, versus about 1% for genre match. Most of what it learned
  is "trust the retriever".
- The 200 candidates contain 64% of the test positives, which caps any re-ranker.

**Serving** (`src/reelrank/serving/`)
- The API image has no torch: the user tower is about 10 lines of NumPy, exported from the trained weights.
- A test checks that the served top 10 is identical to the evaluated pipeline's for the same users.
- FAISS flat search is exact here. HNSW returns 99.9% of the exact top 200 and is there for larger catalogs.

## Engineering

| | |
|---|---|
| Tests | 37 pytest tests on synthetic MovieLens-shaped data: metric values checked by hand, split leakage checks, item-kNN vs a dense cosine, ALS fold-in, NumPy tower = torch tower, FAISS = exact top-k, API endpoints, served output = offline output |
| CI | GitHub Actions: tests and an end-to-end synthetic run of every stage, plus a Docker build. No MovieLens data is ever used in CI |
| Tracking | Every tuning and final run is logged to MLflow (`sqlite:///mlflow.db`; `mlflow ui --backend-store-uri sqlite:///mlflow.db`) |
| Reproducibility | Seeded throughout; rerunning the final stage reproduced every test metric to 4 decimals |
| Load test | `loadtest/recommend.js` (k6, constant arrival rate). Results below, Docker Desktop on an 8 GB M2, 2 uvicorn workers |

| Load | p50 | p95 | p99 | errors |
|---|---:|---:|---:|---:|
| 100 req/s, before fix | 3.9 ms | 14.6 ms | 27.4 ms | 0 |
| 100 req/s | 3.5 ms | 4.9 ms | **8.6 ms** | 0 |
| 300 req/s | 2.1 ms | 5.1 ms | 13.2 ms | 0 |
| 600 req/s | 1.3 ms | 2.9 ms | 11.7 ms | 0 |

Inside the service, retrieval plus ranking takes 0.64 ms per user at the median (p99 1.6 ms). The rest of the request time is
HTTP, JSON and the container. Training the two-tower on Apple-silicon MPS was 1.4–2.8× *slower* than the CPU at this model
size (embedding-heavy backward passes), so training runs on the CPU.

## Reproduce

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # or your platform's wheel
pip install -r requirements.txt
```

```bash
python scripts/download_data.py ml-1m
```

```bash
python scripts/run_experiments.py --dataset ml-1m        # about 1.5 hours on an M2 laptop; writes results/ml-1m/
```

```bash
python scripts/run_experiments.py --dataset ml-32m --user-frac 0.2 --grid medium --tt-variants "combined: temperature=0.1, dim=128, max_hist=20" "combined, no logQ correction"
```

```bash
pytest -q
```

Serve and load-test:

```bash
docker build -t reelrank .
```

```bash
docker run -p 8000:8000 -v "$PWD/artifacts/ml-1m:/artifacts:ro" reelrank
```

```bash
curl "localhost:8000/recommend/1?k=10"
```

```bash
curl -X POST localhost:8000/recommend -H 'content-type: application/json' -d '{"liked": [1, 2355, 3114], "k": 10}'
```

```bash
python scripts/k6_users.py > loadtest/users.json && k6 run loadtest/recommend.js
```

## Caveats

- **A per-user temporal split leaks global time.** User A's test window can predate user B's training ratings, so popularity
  and item embeddings can carry a little of A's future. A single global cutoff would be stricter. The per-user split was the
  brief, and every model gets the same leak.
- **One seed for the final comparison.** The validation repeats put two-tower seed noise at about 0.003–0.007 Recall@20. That is
  small next to the gaps reported here, but the test numbers come from one training run per model.
- **Cold start is simulated.** ML-1M and ML-32M only include users with at least 20 ratings, so "new users" are real users
  truncated to their first 4 ratings.
- **ML-32M is a 20% user sample**, run with a smaller tuning grid. That is what fits on an 8 GB laptop alongside everything else.
- **Data licence**: MovieLens may not be redistributed, so no data, model weights or derived artifacts are in this repository.
  `scripts/download_data.py` fetches the data from GroupLens. Code is MIT-licensed.

## Layout

```
src/reelrank/
  data.py          MovieLens loaders            split.py       temporal split, cold-start slice
  metrics.py       full-catalog + sampled eval  baselines.py   popular, item-kNN, ALS, recency controls
  twotower.py      two-tower model              retrieval.py   FAISS index
  ranker.py        LambdaRank stage             experiment.py  tuning, final evaluation, report
  serving/         export, NumPy engine, FastAPI app
scripts/           download, run experiments, k6 user list
results/           JSON + markdown from the runs reported above
tests/  loadtest/  .github/workflows/
```
