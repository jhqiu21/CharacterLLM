"""
Check that every number listed in results/v4/claims.md appears in its source table.

    python results/v4/check_claims.py

claims.md rows: | id | value | file | row | column |
- file: a Markdown table under results/v4/tables/ (e.g. pe_comparison.md);
- row: the text of the table's first cell, or 'notes' for the text below the table; a row whose first cell repeats is
  given as 'cell1 @ cell2 [@ cell3 ...]' (the leading cells of that row);
- column: the header text, or '-' for notes. The cell (or the notes) must contain `value` exactly as written.
Rows whose file is 'derived' give a formula in the row column and are not checked here.
"""

import re
import sys
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


def main():
    claims = [ln for ln in (HERE / "claims.md").read_text().splitlines() if re.match(r"^\| *[A-Z]+\d+", ln)]
    bad, n = [], 0
    for ln in claims:
        cid, value, fname, row, col = [c.strip().strip("`") for c in ln.strip("|").split("|")[:5]]
        if fname == "derived":
            continue
        n += 1
        header, rows, notes = parse_md_table((HERE / "tables" / fname).read_text())
        if row == "notes":
            ok = value in notes
        else:
            key = [k.strip() for k in row.split(" @ ")]
            match = [r for r in rows if r[:len(key)] == key]
            ok = len(match) == 1 and col in header and value in match[0][header.index(col)]
        if not ok:
            bad.append(f"{cid}: {value!r} not found in {fname} [{row}][{col}]")
    print(f"{n} claims checked, {len(bad)} failed")
    for b in bad:
        print("  " + b)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
