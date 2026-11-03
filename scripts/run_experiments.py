"""Run the ReelRank study.

    python scripts/run_experiments.py --dataset ml-1m                   # all stages
    python scripts/run_experiments.py --dataset ml-1m --stages final    # one stage
    python scripts/run_experiments.py --dataset synthetic --quick       # CI-sized smoke run
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reelrank import experiment as ex  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ml-1m", choices=["ml-1m", "ml-32m", "synthetic"])
    p.add_argument("--stages", nargs="+", default=["baselines", "two_tower", "final"],
                   choices=["baselines", "two_tower", "final", "report"])
    p.add_argument("--quick", action="store_true", help="tiny grids and few epochs (smoke test)")
    p.add_argument("--out", default="results")
    p.add_argument("--artifacts", default="artifacts")
    p.add_argument("--no-mlflow", action="store_true")
    p.add_argument("--grid", default="full", choices=["full", "medium"], help="baseline grid size")
    p.add_argument("--baselines", nargs="+", help="retune only these baselines, keep the others")
    p.add_argument("--tt-variants", nargs="+", help="two-tower variants to run (default: all)")
    p.add_argument("--user-frac", type=float, default=1.0, help="keep this share of users (seeded)")
    args = p.parse_args()

    name = args.dataset if args.user_frac == 1.0 else f"{args.dataset}-u{round(args.user_frac * 100)}"
    out = Path(args.out) / name
    tracker = ex.Tracker(f"reelrank-{name}", uri=None if args.no_mlflow else "sqlite:///mlflow.db")
    if args.stages == ["report"]:  # rebuild results.md from the saved JSON, no training
        ex.write_report(out)
        return
    split = ex.load_split(args.dataset, user_frac=args.user_frac)

    if "baselines" in args.stages:
        grids = ex.SMALL_GRIDS if args.quick else {"full": ex.BASELINE_GRIDS, "medium": ex.MEDIUM_GRIDS}[args.grid]
        ex.tune_baselines(split, out, tracker, grids=grids, models=args.baselines)
    if "two_tower" in args.stages:
        if args.quick:
            ex.tune_two_tower(split, out, tracker, variants={"base": {}, "no logQ correction": {"logq": False}},
                              base={"dim": 16, "hidden": 32, "n_uniform_neg": 64, "batch_size": 256}, epochs=3)
        else:
            variants = {k: v for k, v in ex.TT_VARIANTS.items() if not args.tt_variants or k in args.tt_variants}
            ex.tune_two_tower(split, out, tracker, variants=variants)
    if "final" in args.stages:
        ex.final(split, out, tracker, artifacts_dir=Path(args.artifacts) / name,
                 sampled_users=200 if args.quick else 2000)
    if "report" in args.stages:
        ex.write_report(out)


if __name__ == "__main__":
    main()
