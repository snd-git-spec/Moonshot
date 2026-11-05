#!/usr/bin/env python3
"""CLI: bucket paper trades and write runs/paper_expectancy_<ts>.md."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moonshot.paper_expectancy import locked_criteria_text, write_expectancy_report


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", default=str(ROOT / "moonshot.db"))
    p.add_argument("--out", default=str(ROOT / "runs"))
    args = p.parse_args()
    print(locked_criteria_text())
    path = write_expectancy_report(args.db, args.out)
    print(f"wrote {path}")
    print(path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
