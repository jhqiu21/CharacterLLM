"""
Check that every number listed in results/v4/claims.md appears in its source table.

    python results/v4/check_claims.py

claims.md rows: | id | value | file | row | column |
- file: a Markdown table under results/v4/tables/ (e.g. pe_comparison.md), or a path relative to results/ for the other
  rounds (e.g. v6/tables/paper_errors.md);
- row: the text of the table's first cell, or 'notes' for the text below the table; a row whose first cell repeats is
  given as 'cell1 @ cell2 [@ cell3 ...]' (the leading cells of that row);
- column: the header text, or '-' for notes. The cell (or the notes) must contain `value` exactly as written, or, for a
  value rounded for the text, every number of the value must equal the corresponding number of the cell rounded half-up to
  the same number of decimals (sign ignored when the value has none, e.g. "0.062 ± 0.004" for "-0.0620 ± 0.0041").
Rows whose file is 'derived', 'external' or 'pilot' give their source in the row column and are not checked here.
"""

import re
import sys
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent


def parse_md_table(text):
    lines = [ln for ln in text.splitlines() if ln.startswith("|")]
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for ln in lines[2:]:
        rows.append([c.strip() for c in ln.strip("|").split("|")])
    notes = "\n".join(ln for ln in text.splitlines()[1:] if ln and not ln.startswith("|"))
    return header, rows, notes


NUM = re.compile(r"[-+−]?\d[\d,]*(?:\.\d+)?")


def numbers(text):
    return [t.replace(",", "").replace("−", "-") for t in NUM.findall(text)]


def rounded_match(value, text):
    """Every number of `value` equals a consecutive run of numbers in `text`, rounded half-up to the value's decimals."""
    v = numbers(value)
    t = numbers(text)
    if not v:
        return False
    for start in range(len(t) - len(v) + 1):
        ok = True
        for a, b in zip(v, t[start:start + len(v)]):
            dec = len(a.split(".")[1]) if "." in a else 0
            q = Decimal(1).scaleb(-dec)
            bb = Decimal(b).quantize(q, rounding=ROUND_HALF_UP)
            aa = Decimal(a)
            if not (aa == bb or (a[0] not in "+-" and aa == abs(bb))):
                ok = False
                break
        if ok:
            return True
    return False


def main():
    claims = [ln for ln in (HERE / "claims.md").read_text().splitlines() if re.match(r"^\| *[A-Z]+\d+", ln)]
    bad, n = [], 0
    for ln in claims:
        cid, value, fname, row, col = [c.strip().strip("`") for c in ln.strip("|").split("|")[:5]]
        if fname in ("derived", "external", "pilot"):
            continue
        n += 1
        path = HERE.parent / fname if "/" in fname else HERE / "tables" / fname
        header, rows, notes = parse_md_table(path.read_text())
        if row == "notes":
            ok = value in notes or rounded_match(value, notes)
        else:
            key = [k.strip() for k in row.split(" @ ")]
            match = [r for r in rows if r[:len(key)] == key]
            cell = match[0][header.index(col)] if len(match) == 1 and col in header else None
            ok = cell is not None and (value in cell or rounded_match(value, cell))
        if not ok:
            bad.append(f"{cid}: {value!r} not found in {fname} [{row}][{col}]")
    print(f"{n} claims checked, {len(bad)} failed")
    for b in bad:
        print("  " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
