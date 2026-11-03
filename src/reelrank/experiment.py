"""End-to-end experiment: tune on the validation window, report on the test window.

Stages (each writes JSON to results/<dataset>/ and can be rerun alone):

1. baselines  - grid-search most-popular, item-kNN and ALS on validation Recall@20
2. two_tower  - one-at-a-time variations around a base config, early-stopped on
                validation Recall@20 (doubles as the ablation table)
3. final      - refit every winner on the full training set, train the ranker on
                the validation window, then score everything on the test window
                and the cold-start slice; sampled-vs-full comparison; bootstrap
                intervals; latency; export serving artifacts
"""

from __future__ import annotations

import itertools
import json
import os
import pickle
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import data as data_mod
from . import metrics
from .baselines import ALS, ItemKNN, MostPopular, RecentHistory
from .split import Split, make_split

PRIMARY = "recall@20"


# ----------------------------------------------------------------------------- tracking
class Tracker:
    """Thin MLflow wrapper; a no-op when MLflow is not installed or disabled."""

    def __init__(self, experiment: str, uri: str | None = "sqlite:///mlflow.db"):
        self.mlflow = None
        if uri is None:
            return
        try:
            os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
            import mlflow
        except ImportError:
            return
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment)
        self.mlflow = mlflow

    def log(self, run_name: str, params: dict, metrics_: dict, tags: dict | None = None):
        if self.mlflow is None:
            return
        with self.mlflow.start_run(run_name=run_name):
            self.mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
            self.mlflow.log_metrics({k.replace("@", "_at_"): float(v) for k, v in metrics_.items()
                                     if isinstance(v, (int, float)) and np.isfinite(v)})
            if tags:
                self.mlflow.set_tags(tags)


@contextmanager
def timer(label: str):
    t = time.time()
    print(f"[{label}] ...", flush=True)
    yield
    print(f"[{label}] done in {time.time() - t:.0f}s", flush=True)


def _json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def _round(d: dict, n: int = 4) -> dict:
    return {k: (round(v, n) if isinstance(v, float) else v) for k, v in d.items()}


# ----------------------------------------------------------------------------- data
def load_split(dataset: str, cache_dir: str | Path = "data/processed", user_frac: float = 1.0,
               **split_kw) -> Split:
    """Build (or load the cached) split. 'synthetic' needs no download.

    user_frac < 1 keeps a seeded random sample of users (all their ratings),
    which is how ML-32M fits on an 8 GB laptop.
    """
    if dataset == "synthetic":
        from .synthetic import make_movielens_like
        return make_split(*make_movielens_like(), **split_kw)
    cache = Path(cache_dir) / f"{dataset}-split.pkl"
    use_cache = not split_kw and user_frac == 1.0
    if use_cache and cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    with timer(f"split {dataset}"):
        ratings, movies = data_mod.load(dataset)
        if user_frac < 1.0:
            users = ratings["user"].unique()
            keep = np.random.default_rng(0).choice(users, int(len(users) * user_frac), replace=False)
            ratings = ratings[ratings["user"].isin(keep)].reset_index(drop=True)
        s = make_split(ratings, movies, **split_kw)
        del ratings
    if use_cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        with open(cache, "wb") as f:
            pickle.dump(s, f, protocol=pickle.HIGHEST_PROTOCOL)
    return s


def tuning_users(ev, cap: int = 10_000, seed: int = 0):
    """Validation users used for model selection: all of them, or a seeded sample of `cap`.

    Selection only; every reported test number uses every test user.
    """
    if len(ev) <= cap:
        return ev
    return ev.subset(np.sort(np.random.default_rng(seed).choice(len(ev), cap, replace=False)))


# ----------------------------------------------------------------------------- stage 1
BASELINE_GRIDS = {
    "popular": (MostPopular, {"half_life_days": [None, 30, 90, 180, 365, 730, 1460]}),
    "item_knn": (ItemKNN, {"k": [20, 50, 100, 200, 400], "shrink": [0, 10, 50, 200]}),
    # A first 3x4x5 grid put the optimum on its edge (64 factors, reg 10), so it was widened;
    # the second grid checks past the new edge (256 factors) around the best region.
    "als": (ALS, [{"factors": [16, 32, 64, 128, 256], "regularization": [0.1, 1.0, 10.0, 30.0, 100.0],
                   "alpha": [1.0, 2.0, 5.0, 10.0, 40.0], "iterations": [15]},
                  {"factors": [512], "regularization": [30.0, 100.0], "alpha": [1.0, 2.0], "iterations": [15]}]),
}
MEDIUM_GRIDS = {  # ML-32M sample: centred on the ML-1M optimum, sized for a laptop
    "popular": (MostPopular, {"half_life_days": [None, 90, 365]}),
    "item_knn": (ItemKNN, {"k": [50, 200], "shrink": [0, 50]}),
    "als": (ALS, {"factors": [64, 128], "regularization": [1.0, 10.0, 100.0],
                  "alpha": [2.0, 5.0, 10.0], "iterations": [15]}),
}
SMALL_GRIDS = {  # synthetic / CI
    "popular": (MostPopular, {"half_life_days": [None, 365]}),
    "item_knn": (ItemKNN, {"k": [20, 50], "shrink": [0, 10]}),
    "als": (ALS, {"factors": [16], "regularization": [0.1], "alpha": [10.0], "iterations": [10]}),
}


def _configs(grid: dict | list[dict]):
    for g in grid if isinstance(grid, list) else [grid]:
        for values in itertools.product(*g.values()):
            yield dict(zip(g.keys(), values))


def tune_baselines(split: Split, out: Path, tracker: Tracker, grids=BASELINE_GRIDS,
                   models: list[str] | None = None) -> dict:
    """Grid-search each baseline; with `models`, retune only those and keep the rest."""
    fold, pop = split.valid, split.valid.item_pop()
    val_ev = tuning_users(fold.eval)
    path = out / "baselines_tuning.json"
    results = json.loads(path.read_text()) if models and path.exists() else {}
    for name, (cls, grid) in grids.items():
        if models and name not in models:
            continue
        trials, configs = [], list(_configs(grid))
        with timer(f"tune {name} ({len(configs)} configs)"):
            for params in configs:
                t = time.time()
                model = cls(**params).fit(fold, split.items)
                summary, _ = metrics.evaluate(model, val_ev, pop, ks=(10, 20, 50))
                summary["fit_eval_secs"] = time.time() - t
                trials.append({"params": params, "valid": _round(summary)})
                tracker.log(f"tune/{name}", params, summary, {"stage": "tune", "model": name})
        best = max(trials, key=lambda r: r["valid"][PRIMARY])
        print(f"  best {name}: {best['params']} {PRIMARY}={best['valid'][PRIMARY]}")
        results[name] = {"best": best, "trials": trials}
    _json(results, path)
    return results


# ----------------------------------------------------------------------------- stage 2
TT_VARIANTS = {
    "base": {},
    "temperature=0.1": {"temperature": 0.1},
    "temperature=0.02": {"temperature": 0.02},
    "dim=128": {"dim": 128},
    "max_hist=20": {"max_hist": 20},
    "max_hist=100": {"max_hist": 100},
    "lr=5e-4": {"lr": 5e-4},
    "no logQ correction": {"logq": False},
    "no extra uniform negatives": {"n_uniform_neg": 0},
    "no user-id dropout": {"user_id_dropout": 0.0},
    "user-id always dropped (history only)": {"user_id_dropout": 1.0},
    "combined: temperature=0.1, dim=128, max_hist=20": {"temperature": 0.1, "dim": 128, "max_hist": 20},
    "combined, no logQ correction": {"temperature": 0.1, "dim": 128, "max_hist": 20, "logq": False},
    # Reseeded repeats measure run-to-run noise; they are never selected as the final model.
    "base [seed 1]": {"seed": 1},
    "base [seed 2]": {"seed": 2},
    "combined [seed 1]": {"temperature": 0.1, "dim": 128, "max_hist": 20, "seed": 1},
    "combined [seed 2]": {"temperature": 0.1, "dim": 128, "max_hist": 20, "seed": 2},
}


def selectable(tt_res: dict) -> dict:
    return {k: v for k, v in tt_res.items() if "[seed" not in k}


def tune_two_tower(split: Split, out: Path, tracker: Tracker, variants=TT_VARIANTS,
                   base: dict | None = None, epochs: int = 30) -> dict:
    from .twotower import TwoTower

    fold, pop = split.valid, split.valid.item_pop()
    path = out / "two_tower_tuning.json"
    done = json.loads(path.read_text()) if path.exists() else {}
    val_ev = tuning_users(fold.eval)
    eval_fn = lambda m: metrics.evaluate(m, val_ev, pop, ks=(20,), map_k=20)[0][PRIMARY]
    for name, over in variants.items():
        if name in done:
            continue
        with timer(f"two-tower variant: {name}"):
            model = TwoTower(epochs=epochs, **{**(base or {}), **over})
            model.fit(fold, split.items, eval_fn=eval_fn)
            summary, _ = metrics.evaluate(model, val_ev, pop)
        done[name] = {"overrides": over, "config": asdict(model.cfg), "best_epoch": model.best_epoch_,
                      "valid": _round(summary), "history": model.history_}
        tracker.log(f"tune/two_tower/{name}", asdict(model.cfg), {**summary, "best_epoch": model.best_epoch_},
                    {"stage": "tune", "model": "two_tower"})
        _json(done, path)  # checkpoint after every variant
    best = max(selectable(done).values(), key=lambda r: r["valid"][PRIMARY])
    print(f"  best two-tower: {best['overrides']} {PRIMARY}={best['valid'][PRIMARY]} epoch {best['best_epoch']}")
    return done


# ----------------------------------------------------------------------------- stage 3
def _per_user_latency(fn, ev, n: int = 300, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ev), size=min(n, len(ev)), replace=False)
    times = []
    for i in idx:
        sub = ev.subset(np.array([i]))
        t = time.perf_counter()
        fn(sub)
        times.append((time.perf_counter() - t) * 1000)
    return {"p50_ms": float(np.percentile(times, 50)), "p99_ms": float(np.percentile(times, 99))}


def final(split: Split, out: Path, tracker: Tracker, artifacts_dir: Path | None = None,
          n_candidates: int = 200, sampled_users: int | None = 2000, tt_override: dict | None = None) -> dict:
    from .ranker import TwoStage
    from .retrieval import ItemIndex
    from .twotower import TwoTower, TwoTowerConfig

    base_res = json.loads((out / "baselines_tuning.json").read_text())
    tt_res = json.loads((out / "two_tower_tuning.json").read_text())
    best_tt = max(selectable(tt_res).values(), key=lambda r: r["valid"][PRIMARY])
    tt_cfg = {**best_tt["config"], **(tt_override or {})}

    vf, tf = split.valid, split.test
    pop_test = tf.item_pop()
    models = {}
    for name, cls in [("popular", MostPopular), ("item_knn", ItemKNN), ("als", ALS)]:
        with timer(f"refit {name}"):
            models[name] = cls(**base_res[name]["best"]["params"]).fit(tf, split.items)

    # Recency controls: the tuned kNN / ALS told only about each user's last k likes,
    # with k picked on the validation window.
    recency_tuning, valid_models = {}, {}
    for name, cls in [("item_knn", ItemKNN), ("als", ALS)]:
        fitted = cls(**base_res[name]["best"]["params"]).fit(vf, split.items)
        valid_models[name] = fitted
        scores = {}
        for k in (5, 10, 20, 50):
            wrapped = RecentHistory(fitted, k)
            wrapped.pop_scores_ = fitted.pop_scores_
            scores[k] = metrics.evaluate(wrapped, tuning_users(vf.eval), vf.item_pop(), ks=(20,), map_k=20)[0][PRIMARY]
        best_k = max(scores, key=scores.get)
        recency_tuning[name] = {"valid_recall@20_by_k": scores, "best_k": best_k}
        print(f"  recency control {name}: {scores} -> k={best_k}")
        wrapped = RecentHistory(models[name], best_k)
        wrapped.pop_scores_ = models[name].pop_scores_
        models[f"{name}_recent"] = wrapped
        valid_models[f"{name}_recent"] = RecentHistory(fitted, best_k)

    # Ranker training data: candidates from a two-tower that has not seen the validation window.
    val_ev = tuning_users(vf.eval)
    eval_fn = lambda m: metrics.evaluate(m, val_ev, vf.item_pop(), ks=(20,), map_k=20)[0][PRIMARY]
    with timer("two-tower on train_inner (for ranker labels)"):
        tt_valid = TwoTower(TwoTowerConfig(**tt_cfg)).fit(vf, split.items, eval_fn=eval_fn)
    with timer("two-tower refit on full train"):
        tt_test = TwoTower(TwoTowerConfig(**{**tt_cfg, "epochs": tt_valid.best_epoch_})).fit(tf, split.items)
    models["two_tower"] = tt_test
    with timer("ranker"):
        ranker = TwoStage(tt_valid, n_candidates=n_candidates).fit_ranker(vf.eval, vf, split.items)
        ranker_valid_candidate_recall = ranker.train_candidate_recall_
        ranker.attach(tt_test, tf, split.items)
    models["two_tower+ranker"] = ranker
    cf_names = ["als", "als_recent", "item_knn"]
    with timer("ranker + collaborative-filtering scores as features"):
        ranker_cf = TwoStage(tt_valid, n_candidates=n_candidates, extra={n: valid_models[n] for n in cf_names})
        ranker_cf.fit_ranker(vf.eval, vf, split.items)
        ranker_cf.attach(tt_test, tf, split.items, extra={n: models[n] for n in cf_names})
    models["two_tower+ranker+cf"] = ranker_cf

    results = {"dataset_stats": dataset_stats(split), "configs": {}, "test": {}, "cold": {},
               "bootstrap_vs_als": {"test": {}, "cold": {}}, "sampled": {}}
    per_user = {"test": {}, "cold": {}}
    for name, model in models.items():
        results["configs"][name] = model.params()
        for slice_name, ev in [("test", tf.eval), ("cold", split.cold)]:
            if slice_name == "cold" and name.endswith("_recent"):
                continue  # cold users have <= 4 ratings, so "last k" is their whole history
            with timer(f"evaluate {name} on {slice_name}"):
                summary, pu = metrics.evaluate(model, ev, pop_test)
            results[slice_name][name] = _round(summary)
            per_user[slice_name][name] = pu
            tracker.log(f"final/{name}/{slice_name}", model.params(), summary,
                        {"stage": "final", "model": name, "slice": slice_name})
    for slice_name in ("test", "cold"):
        for name in per_user[slice_name]:
            if name == "als":
                continue
            results["bootstrap_vs_als"][slice_name][name] = {
                m: _round(metrics.paired_bootstrap(per_user[slice_name][name][m], per_user[slice_name]["als"][m]))
                for m in ("recall@20", "ndcg@20")}
    # Does re-ranking beat retrieval alone?
    results["bootstrap_vs_two_tower"] = {
        sl: {name: {m: _round(metrics.paired_bootstrap(per_user[sl][name][m], per_user[sl]["two_tower"][m]))
                    for m in ("recall@20", "ndcg@20")}
             for name in ("two_tower+ranker", "two_tower+ranker+cf")}
        for sl in ("test", "cold")}

    # How many of the test positives the 200 candidates contain at all (the ranker's ceiling).
    cs, ci = ranker.candidates(tf.eval)
    from .ranker import labels_for
    hits = labels_for(ci, tf.eval.targets)
    results["candidate_recall@%d" % n_candidates] = {
        "test": float(np.mean([hits[i].sum() / len(t) for i, t in enumerate(tf.eval.targets)])),
        "valid_ranker_training": ranker_valid_candidate_recall}
    results["ranker_feature_importance"] = ranker.feature_importance()
    results["ranker_best_iteration"] = int(ranker.model_.best_iteration_)
    results["ranker_cf_feature_importance"] = ranker_cf.feature_importance()
    results["ranker_cf_best_iteration"] = int(ranker_cf.model_.best_iteration_)
    results["two_tower_best_epoch"] = int(tt_valid.best_epoch_)
    results["recency_tuning"] = recency_tuning

    # Sampled metrics vs full-catalog: does the ordering of models change?
    ev_s = tf.eval
    if sampled_users and len(ev_s) > sampled_users:
        ev_s = ev_s.subset(np.sort(np.random.default_rng(0).choice(len(ev_s), sampled_users, replace=False)))
    for name in ("popular", "item_knn", "als", "two_tower"):
        with timer(f"sampled eval {name}"):
            results["sampled"][name] = _round(metrics.sampled_eval(models[name], ev_s, n_neg=100, k=10))
    results["sampled"]["n_users"] = len(ev_s)

    # Approximate search: how much of the exact top-200 does HNSW return?
    U = tt_test.user_embeddings(tf.eval)
    flat = ItemIndex(tt_test.item_emb_, "flat")
    hnsw = ItemIndex(tt_test.item_emb_, "hnsw")
    _, a = flat.search(U, n_candidates, tf.eval.hist_seen)
    _, b = hnsw.search(U, n_candidates, tf.eval.hist_seen)
    results["hnsw_overlap@%d" % n_candidates] = float(np.mean([len(np.intersect1d(x, y)) / n_candidates
                                                              for x, y in zip(a, b)]))

    # Offline single-request latency of each stage (no HTTP).
    results["offline_latency_ms"] = {
        "user_tower": _per_user_latency(tt_test.user_embeddings, tf.eval),
        "retrieve_200": _per_user_latency(ranker.candidates, tf.eval),
        "retrieve+rank": _per_user_latency(lambda e: ranker.recommend(e, 20), tf.eval),
        "als_full_catalog": _per_user_latency(lambda e: models["als"].recommend(e, 20), tf.eval),
    }

    _json(results, out / "final.json")
    (out / "results.md").write_text(results_markdown(results, base_res, tt_res))
    if artifacts_dir is not None:
        from .serving.export import export_artifacts
        export_artifacts(split, tt_test, ranker, artifacts_dir)
    return results


def dataset_stats(split: Split) -> dict:
    tf = split.test
    return {"n_users": int(split.user_raw_ids.size), "n_items": int(tf.n_items),
            "train_ratings": int(len(tf.train)), "train_positives": int(tf.train_pos.nnz),
            "test_users": len(tf.eval), "test_positives": int(sum(len(t) for t in tf.eval.targets)),
            "cold_users": len(split.cold),
            "cold_hist_positives_mean": float(np.diff(split.cold.hist_pos.indptr).mean()) if len(split.cold) else 0.0}


# ----------------------------------------------------------------------------- report
LABELS = {"popular": "Most popular", "item_knn": "Item-kNN", "als": "ALS (implicit)",
          "item_knn_recent": "Item-kNN, last-k likes only", "als_recent": "ALS fold-in, last-k likes only",
          "two_tower": "Two-tower (retrieval only)", "two_tower+ranker": "Two-tower + LambdaRank",
          "two_tower+ranker+cf": "Two-tower + LambdaRank, + ALS/kNN score features"}


def results_markdown(r: dict, base_res: dict, tt_res: dict) -> str:
    cols = ["recall@10", "recall@20", "recall@50", "ndcg@10", "ndcg@20", "map@100",
            "coverage@20", "pop_pct@20"]
    lines = []

    def table(slice_name):
        lines.append("| Model | " + " | ".join(cols) + " |")
        lines.append("|---|" + "---:|" * len(cols))
        best = {c: max(v[c] for v in r[slice_name].values()) for c in cols[:6]}
        for name, v in r[slice_name].items():
            cells = []
            for c in cols:
                s = f"{v[c]:.1f}" if c == "pop_pct@20" else f"{v[c]:.4f}"
                cells.append(f"**{s}**" if c in best and v[c] == best[c] else s)
            lines.append(f"| {LABELS.get(name, name)} | " + " | ".join(cells) + " |")

    s = r["dataset_stats"]
    lines.append(f"Test: {s['test_users']} warm users, {s['test_positives']} held-out positives, "
                 f"{s['n_items']} movies ranked (full catalog).\n")
    table("test")
    lines.append(f"\nCold start: {s['cold_users']} users never seen in training, each revealing only "
                 f"their first 4 ratings ({s['cold_hist_positives_mean']:.1f} positives on average).\n")
    table("cold")

    lines.append("\nPaired bootstrap, difference vs ALS (mean over users, 95% interval):\n")
    lines.append("| Model | slice | Recall@20 diff | NDCG@20 diff |")
    lines.append("|---|---|---:|---:|")
    for sl in ("test", "cold"):
        for name, d in r["bootstrap_vs_als"][sl].items():
            f = lambda x: f"{x['diff']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
            lines.append(f"| {LABELS.get(name, name)} | {sl} | {f(d['recall@20'])} | {f(d['ndcg@20'])} |")

    lines.append("\nRe-ranking vs retrieval alone (paired bootstrap, difference vs two-tower):\n")
    lines.append("| Model | slice | Recall@20 diff | NDCG@20 diff |")
    lines.append("|---|---|---:|---:|")
    for sl in ("test", "cold"):
        for name, d in r["bootstrap_vs_two_tower"][sl].items():
            f = lambda x: f"{x['diff']:+.4f} [{x['lo']:+.4f}, {x['hi']:+.4f}]"
            lines.append(f"| {LABELS.get(name, name)} | {sl} | {f(d['recall@20'])} | {f(d['ndcg@20'])} |")

    lines.append("\nSampled (1 positive vs 100 random negatives) vs full-catalog ranking of models:\n")
    lines.append("| Model | sampled HR@10 | sampled NDCG@10 | full Recall@10 | full NDCG@10 |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in ("popular", "item_knn", "als", "two_tower"):
        sm, fm = r["sampled"][name], r["test"][name]
        lines.append(f"| {LABELS[name]} | {sm['sampled_hr@10']:.4f} | {sm['sampled_ndcg@10']:.4f} | "
                     f"{fm['recall@10']:.4f} | {fm['ndcg@10']:.4f} |")

    lines.append("\nTwo-tower variants (validation window, early-stopped on Recall@20):\n")
    lines.append("| Variant | Recall@20 | NDCG@20 | coverage@20 | pop_pct@20 | best epoch |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for name, v in tt_res.items():
        m = v["valid"]
        lines.append(f"| {name} | {m['recall@20']:.4f} | {m['ndcg@20']:.4f} | {m['coverage@20']:.4f} | "
                     f"{m['pop_pct@20']:.1f} | {v['best_epoch']} |")

    lines.append("\nTuned baselines (best on validation Recall@20):\n")
    for name in ("popular", "item_knn", "als"):
        b = base_res[name]["best"]
        lines.append(f"- {LABELS[name]}: `{b['params']}` (validation Recall@20 {b['valid']['recall@20']:.4f}, "
                     f"{len(base_res[name]['trials'])} configs tried)")
    return "\n".join(lines) + "\n"
