"""
Create data/text8_train.txt and data/text8_test.txt from text8 (Mahoney, http://mattmahoney.net/dc/textdata.html).

    python scripts/prepare_data.py                    # download text8.zip (31 MB) into data/ and split it
    python scripts/prepare_data.py --zip path/to/text8.zip

text8 has 100,000,000 characters (a-z and space). text8_train.txt is the first 90,000,000; text8_test.txt is the last
5,000,000 (characters 95,000,000-99,999,999); the 5,000,000 in between are not used. train.py splits text8_train.txt
into training (first 95%) and validation (last 5%). The md5 sums of both files are checked against the files every
result in results/ was produced with (results/v3/env.txt).
"""

import argparse
import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://mattmahoney.net/dc/text8.zip"
N_TOTAL, N_TRAIN, N_TEST = 100_000_000, 90_000_000, 5_000_000
MD5 = {"text8_train.txt": "c11c16d3c5dea4e747fdd2d4e14ae571", "text8_test.txt": "4aa15c339c5d1d7a85ecf52989ce1113"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=None, help="existing text8.zip (default: download to data/text8.zip)")
    ap.add_argument("--out-dir", default=str(ROOT / "data"))
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zip_path = Path(args.zip) if args.zip else out / "text8.zip"
    if not zip_path.exists():
        print(f"downloading {URL} -> {zip_path}", flush=True)
        urllib.request.urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as z:
        text = z.read("text8").decode("ascii")
    if len(text) != N_TOTAL:
        sys.exit(f"text8 has {len(text):,} characters, expected {N_TOTAL:,}")
    parts = {"text8_train.txt": text[:N_TRAIN], "text8_test.txt": text[N_TOTAL - N_TEST:]}
    ok = True
    for name, part in parts.items():
        (out / name).write_text(part)
        md5 = hashlib.md5(part.encode("ascii")).hexdigest()
        ok &= md5 == MD5[name]
        print(f"{out / name}: {len(part):,} characters, md5 {md5} ({'ok' if md5 == MD5[name] else 'MISMATCH, expected ' + MD5[name]})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
