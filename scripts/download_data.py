"""Download MovieLens datasets into data/raw/.

MovieLens data may not be redistributed (see the GroupLens license), so it is
never committed: data/ is git-ignored and this script fetches it on demand.

    python scripts/download_data.py ml-1m
    python scripts/download_data.py ml-32m   # ~240 MB zip, ~900 MB extracted
"""

import argparse
import hashlib
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-32m": "https://files.grouplens.org/datasets/movielens/ml-32m.zip",
}
# Rough space needed (zip + extracted + processed parquet), in GB.
SPACE_NEEDED_GB = {"ml-1m": 0.1, "ml-32m": 2.0}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(URLS))
    parser.add_argument("--out", default="data/raw")
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / args.dataset
    if target.exists():
        print(f"{target} already exists, skipping")
        return

    free_gb = shutil.disk_usage(out).free / 1e9
    if free_gb < SPACE_NEEDED_GB[args.dataset] * 1.5:
        sys.exit(f"Only {free_gb:.1f} GB free; {args.dataset} needs about "
                 f"{SPACE_NEEDED_GB[args.dataset]} GB. Free some space first.")

    zip_path = out / f"{args.dataset}.zip"
    print(f"Downloading {URLS[args.dataset]}")
    urllib.request.urlretrieve(URLS[args.dataset], zip_path)
    md5 = hashlib.md5(zip_path.read_bytes()).hexdigest()
    print(f"md5 {md5}")
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if name.startswith("/") or ".." in Path(name).parts:
                sys.exit(f"Refusing suspicious path in archive: {name}")
        zf.extractall(out)
    zip_path.unlink()  # keep the footprint small
    print(f"Extracted to {target}")


if __name__ == "__main__":
    main()
