#!/usr/bin/env python3
"""Audit moonshot.db trades → docs/paper_audit_<ts>.md (work unit 0.1a)."""

from __future__ import annotations

import argparse
import math
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moonshot.paths import DB_PATH, ROOT as MS_ROOT  # noqa: E402


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    k = (len(ys) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return ys[int(k)]
    return ys[f] * (c - k) + ys[c] * (k - f)


def audit(db_path: Path, out_dir: Path) -> Path:
    if not db_path.exists():
        raise FileNotFoundError(f"DB not found: {db_path}")
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(trades)")}
        required = {"ts", "symbol", "qty", "r_multiple", "fees", "funding"}
        missing = required - cols
        if missing:
            raise RuntimeError(f"trades missing columns: {sorted(missing)}")

        rows = list(con.execute("SELECT * FROM trades"))
        n = len(rows)
        symbols: dict[str, int] = {}
        null_fees = null_funding = null_r = zero_fee = 0
        rs: list[float] = []
        anomalies: list[str] = []
        ts_vals: list[str] = []

        for r in rows:
            sym = str(r["symbol"] or "?")
            symbols[sym] = symbols.get(sym, 0) + 1
            ts = r["ts"]
            if ts:
                ts_vals.append(str(ts))
            fees = r["fees"]
            funding = r["funding"]
            rm = r["r_multiple"]
            qty = r["qty"]
            if fees is None:
                null_fees += 1
            elif float(fees) == 0.0:
                zero_fee += 1
            if funding is None:
                null_funding += 1
            if rm is None:
                null_r += 1
            else:
                rv = float(rm)
                rs.append(rv)
                if rv < -5.0 or rv > 50.0:
                    anomalies.append(f"r_multiple={rv} symbol={sym} ts={ts}")
            if qty is not None and float(qty) < 0:
                anomalies.append(f"negative qty={qty} symbol={sym} ts={ts}")

        ts_sorted = sorted(ts_vals)
        date_range = (
            f"{ts_sorted[0]} → {ts_sorted[-1]}" if ts_sorted else "(no timestamps)"
        )

        dist = {
            "min": min(rs) if rs else float("nan"),
            "p5": _pct(rs, 5),
            "p25": _pct(rs, 25),
            "median": _pct(rs, 50),
            "p75": _pct(rs, 75),
            "p95": _pct(rs, 95),
            "max": max(rs) if rs else float("nan"),
        }

        ts_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"paper_audit_{ts_stamp}.md"

        lines = [
            f"# Paper audit {ts_stamp}",
            "",
            f"- DB: `{db_path}`",
            f"- Total trades: **{n}**",
            f"- Date range: {date_range}",
            "",
            "## Per-symbol counts",
            "",
        ]
        for sym, c in sorted(symbols.items(), key=lambda x: (-x[1], x[0])):
            lines.append(f"- {sym}: {c}")
        if not symbols:
            lines.append("- (none)")

        lines += [
            "",
            "## NULL / zero counts",
            "",
            f"- fees NULL: {null_fees}",
            f"- funding NULL: {null_funding}",
            f"- r_multiple NULL: {null_r}",
            f"- fees == 0: {zero_fee}",
            "",
            "## r_multiple distribution",
            "",
        ]
        for k, v in dist.items():
            lines.append(f"- {k}: {v if v == v else 'n/a'}")

        lines += ["", "## Anomalies", ""]
        if anomalies:
            for a in anomalies[:200]:
                lines.append(f"- {a}")
            if len(anomalies) > 200:
                lines.append(f"- … {len(anomalies) - 200} more")
        else:
            lines.append("- (none)")

        lines += [
            "",
            "## Notes",
            "",
            "- Legacy trades (pre-config_hash) will be bucketed separately after 0.2.",
            "- r_multiple is already net of fees/funding at close; do not rebuild gross R.",
            "",
        ]
        out.write_text("\n".join(lines), encoding="utf-8")
        return out
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--out-dir", type=Path, default=MS_ROOT / "docs")
    args = p.parse_args()
    path = audit(args.db, args.out_dir)
    print(path)


if __name__ == "__main__":
    main()
