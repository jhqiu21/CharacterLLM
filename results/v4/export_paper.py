"""
Copy the report's tables (.tex) and figures (.pdf) into the LaTeX source of the report.

    python results/v4/export_paper.py --paper /path/to/paper     # -> <paper>/tables/vN/*.tex, <paper>/image/vN/*.pdf

For every results folder of VERSIONS (v4: main study; v5: repairs of RoPE beyond the training length; v6: capacity and
accuracy). Run results/vN/aggregate_vN.py first. Files in the target folders are replaced; nothing else in the paper is
touched.
"""

import argparse
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
VERSIONS = ("v4", "v5", "v6")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paper", required=True)
    args = ap.parse_args()
    paper = Path(args.paper)
    pairs = [(HERE.parent / v / "tables", paper / "tables" / v, "*.tex") for v in VERSIONS]
    pairs += [(HERE.parent / v / "figures", paper / "image" / v, "*.pdf") for v in VERSIONS]
    for src, dst, pattern in pairs:
        dst.mkdir(parents=True, exist_ok=True)
        files = sorted(src.glob(pattern))
        for f in files:
            shutil.copy2(f, dst / f.name)
        print(f"{len(files)} files -> {dst}")


if __name__ == "__main__":
    main()
