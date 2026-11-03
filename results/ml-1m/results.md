Test: 5195 warm users, 46145 held-out positives, 3706 movies ranked (full catalog).

| Model | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 | map@100 | coverage@20 | pop_pct@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Most popular | 0.0528 | 0.0900 | 0.1680 | 0.0599 | 0.0694 | 0.0355 | 0.0586 | 99.4 |
| Item-kNN | 0.0894 | 0.1518 | 0.2761 | 0.0871 | 0.1061 | 0.0584 | 0.3764 | 95.7 |
| ALS (implicit) | 0.1092 | 0.1775 | 0.3140 | 0.1010 | 0.1217 | 0.0690 | 0.2847 | 95.1 |
| Item-kNN, last-k likes only | 0.1083 | 0.1742 | 0.2950 | 0.1012 | 0.1218 | 0.0684 | 0.6943 | 87.9 |
| ALS fold-in, last-k likes only | 0.1331 | 0.2045 | 0.3306 | 0.1249 | 0.1468 | 0.0839 | 0.3726 | 91.7 |
| Two-tower (retrieval only) | 0.1540 | 0.2380 | 0.3884 | 0.1419 | 0.1677 | 0.1022 | 0.6433 | 86.9 |
| Two-tower + LambdaRank | 0.1539 | 0.2426 | 0.3913 | 0.1440 | 0.1709 | 0.1026 | 0.6001 | 90.8 |
| Two-tower + LambdaRank, + ALS/kNN score features | **0.1729** | **0.2641** | **0.4172** | **0.1631** | **0.1902** | **0.1154** | 0.5559 | 91.0 |

Cold start: 575 users never seen in training, each revealing only their first 4 ratings (2.2 positives on average).

| Model | recall@10 | recall@20 | recall@50 | ndcg@10 | ndcg@20 | map@100 | coverage@20 | pop_pct@20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Most popular | **0.0286** | **0.0539** | **0.1228** | **0.0254** | **0.0343** | **0.0181** | 0.0062 | 99.7 |
| Item-kNN | 0.0171 | 0.0374 | 0.0896 | 0.0191 | 0.0258 | 0.0129 | 0.3524 | 93.7 |
| ALS (implicit) | 0.0178 | 0.0392 | 0.0903 | 0.0181 | 0.0259 | 0.0118 | 0.2275 | 95.9 |
| Two-tower (retrieval only) | 0.0229 | 0.0436 | 0.0946 | 0.0209 | 0.0286 | 0.0140 | 0.1786 | 97.2 |
| Two-tower + LambdaRank | 0.0248 | 0.0449 | 0.0974 | 0.0222 | 0.0296 | 0.0151 | 0.1630 | 97.5 |
| Two-tower + LambdaRank, + ALS/kNN score features | 0.0260 | 0.0479 | 0.1029 | 0.0249 | 0.0328 | 0.0160 | 0.1659 | 97.8 |

Paired bootstrap, difference vs ALS (mean over users, 95% interval):

| Model | slice | Recall@20 diff | NDCG@20 diff |
|---|---|---:|---:|
| Most popular | test | -0.0875 [-0.0947, -0.0806] | -0.0524 [-0.0572, -0.0478] |
| Item-kNN | test | -0.0257 [-0.0310, -0.0205] | -0.0156 [-0.0186, -0.0125] |
| Item-kNN, last-k likes only | test | -0.0033 [-0.0101, +0.0038] | +0.0000 [-0.0050, +0.0051] |
| ALS fold-in, last-k likes only | test | +0.0271 [+0.0200, +0.0338] | +0.0250 [+0.0199, +0.0301] |
| Two-tower (retrieval only) | test | +0.0605 [+0.0525, +0.0686] | +0.0460 [+0.0403, +0.0521] |
| Two-tower + LambdaRank | test | +0.0651 [+0.0574, +0.0728] | +0.0492 [+0.0438, +0.0549] |
| Two-tower + LambdaRank, + ALS/kNN score features | test | +0.0867 [+0.0793, +0.0941] | +0.0685 [+0.0631, +0.0739] |
| Most popular | cold | +0.0147 [+0.0029, +0.0267] | +0.0084 [+0.0014, +0.0152] |
| Item-kNN | cold | -0.0018 [-0.0102, +0.0070] | -0.0001 [-0.0044, +0.0044] |
| Two-tower (retrieval only) | cold | +0.0044 [-0.0060, +0.0154] | +0.0027 [-0.0039, +0.0093] |
| Two-tower + LambdaRank | cold | +0.0057 [-0.0045, +0.0162] | +0.0037 [-0.0031, +0.0105] |
| Two-tower + LambdaRank, + ALS/kNN score features | cold | +0.0087 [-0.0002, +0.0170] | +0.0070 [+0.0011, +0.0127] |

Re-ranking vs retrieval alone (paired bootstrap, difference vs two-tower):

| Model | slice | Recall@20 diff | NDCG@20 diff |
|---|---|---:|---:|
| Two-tower + LambdaRank | test | +0.0046 [+0.0010, +0.0083] | +0.0032 [+0.0008, +0.0057] |
| Two-tower + LambdaRank, + ALS/kNN score features | test | +0.0261 [+0.0218, +0.0305] | +0.0225 [+0.0196, +0.0253] |
| Two-tower + LambdaRank | cold | +0.0012 [-0.0038, +0.0061] | +0.0010 [-0.0023, +0.0041] |
| Two-tower + LambdaRank, + ALS/kNN score features | cold | +0.0043 [-0.0029, +0.0115] | +0.0043 [+0.0001, +0.0083] |

Sampled (1 positive vs 100 random negatives) vs full-catalog ranking of models:

| Model | sampled HR@10 | sampled NDCG@10 | full Recall@10 | full NDCG@10 |
|---|---:|---:|---:|---:|
| Most popular | 0.5224 | 0.2938 | 0.0528 | 0.0599 |
| Item-kNN | 0.6517 | 0.3807 | 0.0894 | 0.0871 |
| ALS (implicit) | 0.6819 | 0.4036 | 0.1092 | 0.1010 |
| Two-tower (retrieval only) | 0.6300 | 0.3970 | 0.1540 | 0.1419 |

Two-tower variants (validation window, early-stopped on Recall@20):

| Variant | Recall@20 | NDCG@20 | coverage@20 | pop_pct@20 | best epoch |
|---|---:|---:|---:|---:|---:|
| base | 0.2528 | 0.1682 | 0.7072 | 86.5 | 7 |
| temperature=0.1 | 0.2617 | 0.1765 | 0.6266 | 87.2 | 16 |
| temperature=0.02 | 0.2586 | 0.1702 | 0.7426 | 86.0 | 11 |
| dim=128 | 0.2625 | 0.1761 | 0.7577 | 85.1 | 13 |
| max_hist=20 | 0.2657 | 0.1769 | 0.7242 | 86.4 | 10 |
| max_hist=100 | 0.2573 | 0.1706 | 0.7151 | 86.2 | 11 |
| lr=5e-4 | 0.2469 | 0.1618 | 0.7153 | 86.1 | 13 |
| no logQ correction | 0.2188 | 0.1410 | 0.7933 | 79.2 | 11 |
| no extra uniform negatives | 0.2567 | 0.1725 | 0.7642 | 84.6 | 13 |
| no user-id dropout | 0.2564 | 0.1715 | 0.7194 | 86.2 | 9 |
| user-id always dropped (history only) | 0.2180 | 0.1441 | 0.6082 | 89.5 | 6 |
| combined: temperature=0.1, dim=128, max_hist=20 | 0.2714 | 0.1854 | 0.6409 | 86.8 | 13 |
| base [seed 1] | 0.2569 | 0.1720 | 0.7345 | 86.3 | 11 |
| base [seed 2] | 0.2598 | 0.1767 | 0.7156 | 86.9 | 9 |
| combined [seed 1] | 0.2727 | 0.1844 | 0.6422 | 87.3 | 11 |
| combined [seed 2] | 0.2699 | 0.1841 | 0.6282 | 87.3 | 9 |
| combined, no logQ correction | 0.2340 | 0.1517 | 0.7299 | 80.3 | 11 |

Tuned baselines (best on validation Recall@20):

- Most popular: `{'half_life_days': 365}` (validation Recall@20 0.0939, 7 configs tried)
- Item-kNN: `{'k': 50, 'shrink': 0}` (validation Recall@20 0.1592, 20 configs tried)
- ALS (implicit): `{'factors': 512, 'regularization': 30.0, 'alpha': 2.0, 'iterations': 15}` (validation Recall@20 0.1774, 129 configs tried)
