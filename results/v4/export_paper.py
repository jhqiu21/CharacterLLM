"""
Copy the report's tables (.tex) and figures (.pdf) into the LaTeX source of the report.

    python results/v4/export_paper.py --paper /path/to/paper     # -> <paper>/tables/v4/*.tex, <paper>/image/v4/*.pdf

Run results/v4/aggregate_v4.py first. Files in the target folders are replaced; nothing else in the paper is touched.
"""

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    args = ap.parse_args()
    paper = Path(args.paper)
    for src, dst, pattern in [(HERE / "tables", paper / "tables" / "v4", "*.tex"), (HERE / "figures", paper / "image" / "v4", "*.pdf")]:
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob(pattern))
        for f in files:
            shutil.copy2(f, dst / f.name)
        print(f"{len(files)} files -> {dst}")


if __name__ == "__main__":
    main()
