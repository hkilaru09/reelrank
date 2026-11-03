Test: 33988 warm users, 268856 held-out positives, 50034 movies ranked (full catalog).

| Model | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 | map@100 | coverage@20 | pop_pct@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Most popular | 0.0516 | 0.0869 | 0.1555 | 0.0492 | 0.0599 | 0.0317 | 0.0063 | 100.0 |
| Item-kNN | 0.1017 | 0.1632 | 0.2861 | 0.0916 | 0.1105 | 0.0633 | 0.0966 | 99.5 |
| ALS (implicit) | 0.1260 | 0.2005 | 0.3430 | 0.1132 | 0.1359 | 0.0789 | 0.0443 | 99.5 |
| Item-kNN, last-k likes only | 0.1052 | 0.1653 | 0.2782 | 0.0937 | 0.1121 | 0.0636 | 0.2626 | 97.7 |
| ALS fold-in, last-k likes only | 0.1274 | 0.1985 | 0.3282 | 0.1124 | 0.1343 | 0.0774 | 0.0541 | 99.2 |
| Two-tower (retrieval only) | 0.1380 | 0.2150 | 0.3569 | 0.1198 | 0.1441 | 0.0852 | 0.1427 | 98.6 |
| Two-tower + LambdaRank | 0.1433 | 0.2234 | 0.3684 | 0.1272 | 0.1515 | 0.0891 | 0.1335 | 98.9 |
| Two-tower + LambdaRank, + ALS/kNN score features | **0.1627** | **0.2469** | **0.3962** | **0.1489** | **0.1736** | **0.1033** | 0.1218 | 98.9 |

Cold start: 3783 users never seen in training, each revealing only their first 4 ratings (2.3 positives on average).

| Model | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 | map@100 | coverage@20 | pop_pct@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Most popular | **0.0370** | **0.0661** | 0.1265 | **0.0276** | **0.0376** | **0.0203** | 0.0005 | 100.0 |
| Item-kNN | 0.0312 | 0.0559 | 0.1191 | 0.0243 | 0.0323 | 0.0181 | 0.0824 | 98.5 |
| ALS (implicit) | 0.0340 | 0.0574 | 0.1228 | 0.0255 | 0.0331 | 0.0184 | 0.0339 | 99.5 |
| Two-tower (retrieval only) | 0.0277 | 0.0519 | 0.1136 | 0.0204 | 0.0284 | 0.0153 | 0.0389 | 99.5 |
| Two-tower + LambdaRank | 0.0320 | 0.0605 | 0.1294 | 0.0239 | 0.0333 | 0.0179 | 0.0326 | 99.7 |
| Two-tower + LambdaRank, + ALS/kNN score features | 0.0340 | 0.0653 | **0.1335** | 0.0262 | 0.0365 | 0.0198 | 0.0372 | 99.6 |

Paired bootstrap, difference vs ALS (mean over users, 95% interval):

| Model | slice | Recall@20 diff | NDCG@20 diff |
|---|---|---:|---:|
| Most popular | test | -0.1136 [-0.1166, -0.1107] | -0.0760 [-0.0780, -0.0741] |
| Item-kNN | test | -0.0373 [-0.0395, -0.0351] | -0.0254 [-0.0267, -0.0240] |
| Item-kNN, last-k likes only | test | -0.0351 [-0.0377, -0.0325] | -0.0237 [-0.0255, -0.0219] |
| ALS fold-in, last-k likes only | test | -0.0020 [-0.0041, +0.0001] | -0.0016 [-0.0031, -0.0000] |
| Two-tower (retrieval only) | test | +0.0145 [+0.0117, +0.0173] | +0.0082 [+0.0062, +0.0102] |
| Two-tower + LambdaRank | test | +0.0229 [+0.0201, +0.0256] | +0.0156 [+0.0138, +0.0175] |
| Two-tower + LambdaRank, + ALS/kNN score features | test | +0.0465 [+0.0441, +0.0489] | +0.0377 [+0.0362, +0.0394] |
| Most popular | cold | +0.0088 [+0.0028, +0.0149] | +0.0045 [+0.0014, +0.0079] |
| Item-kNN | cold | -0.0015 [-0.0050, +0.0023] | -0.0008 [-0.0027, +0.0012] |
| Two-tower (retrieval only) | cold | -0.0055 [-0.0102, -0.0009] | -0.0047 [-0.0071, -0.0022] |
| Two-tower + LambdaRank | cold | +0.0031 [-0.0020, +0.0082] | +0.0002 [-0.0024, +0.0027] |
| Two-tower + LambdaRank, + ALS/kNN score features | cold | +0.0079 [+0.0032, +0.0129] | +0.0034 [+0.0008, +0.0060] |

Re-ranking vs retrieval alone (paired bootstrap, difference vs two-tower):

| Model | slice | Recall@20 diff | NDCG@20 diff |
|---|---|---:|---:|
| Two-tower + LambdaRank | test | +0.0084 [+0.0070, +0.0097] | +0.0074 [+0.0065, +0.0084] |
| Two-tower + LambdaRank, + ALS/kNN score features | test | +0.0320 [+0.0300, +0.0338] | +0.0295 [+0.0282, +0.0308] |
| Two-tower + LambdaRank | cold | +0.0086 [+0.0059, +0.0112] | +0.0049 [+0.0035, +0.0061] |
| Two-tower + LambdaRank, + ALS/kNN score features | cold | +0.0134 [+0.0100, +0.0170] | +0.0080 [+0.0062, +0.0101] |

Sampled (1 positive vs 100 random negatives) vs full-catalog ranking of models:

| Model | sampled HR@10 | sampled NDCG@10 | full Recall@10 | full NDCG@10 |
|---|---:|---:|---:|---:|
| Most popular | 0.8985 | 0.6347 | 0.0516 | 0.0492 |
| Item-kNN | 0.8593 | 0.7126 | 0.1017 | 0.0916 |
| ALS (implicit) | 0.9470 | 0.7759 | 0.1260 | 0.1132 |
| Two-tower (retrieval only) | 0.9463 | 0.7704 | 0.1380 | 0.1198 |

Two-tower variants (validation window, early-stopped on Recall@20):

| Variant | Recall@20 | NDCG@20 | coverage@20 | pop_pct@20 | best epoch |
|---|---:|---:|---:|---:|---:|
| combined: temperature=0.1, dim=128, max_hist=20 | 0.2389 | 0.1551 | 0.1048 | 98.7 | 8 |
| combined, no logQ correction | 0.1724 | 0.1079 | 0.1929 | 95.9 | 11 |

Tuned baselines (best on validation Recall@20):

- Most popular: `{'half_life_days': None}` (validation Recall@20 0.0901, 3 configs tried)
- Item-kNN: `{'k': 50, 'shrink': 0}` (validation Recall@20 0.1734, 4 configs tried)
- ALS (implicit): `{'factors': 512, 'regularization': 100.0, 'alpha': 5.0, 'iterations': 15}` (validation Recall@20 0.2063, 27 configs tried)
