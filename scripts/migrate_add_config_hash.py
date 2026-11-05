#!/usr/bin/env python3
"""Idempotent trades migration: config_hash, config_json, initial_risk_dollars (0.2)."""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moonshot.paths import DB_PATH  # noqa: E402


def _cols(con: sqlite3.Connection, table: str) -> set[str]:
    return {r[1] for r in con.execute(f"PRAGMA table_info({table})")}


def migrate(db_path: Path, *, backup: bool = True) -> dict:
    if not db_path.exists():
        raise FileNotFoundError(db_path)
    backup_path = None
    if backup:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        backup_path = db_path.with_suffix(db_path.suffix + f".bak_{ts}")
        shutil.copy2(db_path, backup_path)

    con = sqlite3.connect(str(db_path))
    try:
        before = _cols(con, "trades")
        added = []
        for col, decl in (
            ("config_hash", "TEXT"),
            ("config_json", "TEXT"),
            ("initial_risk_dollars", "REAL"),
        ):
            if col not in before:
                con.execute(f"ALTER TABLE trades ADD COLUMN {col} {decl}")
                added.append(col)
        con.execute("CREATE INDEX IF NOT EXISTS idx_trades_config_hash ON trades(config_hash)")
        con.commit()
        after = _cols(con, "trades")
        # Legacy rows keep NULL config_hash — no recoverable backfill without snapshots
        null_hash = con.execute(
            "SELECT COUNT(*) FROM trades WHERE config_hash IS NULL"
        ).fetchone()[0]
        return {
            "backup": str(backup_path) if backup_path else None,
            "added": added,
            "cols": sorted(after),
            "legacy_null_hash": int(null_hash),
        }
    finally:
        con.close()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DB_PATH)
    p.add_argument("--no-backup", action="store_true")
    args = p.parse_args()
    info = migrate(args.db, backup=not args.no_backup)
    print(info)


if __name__ == "__main__":
    main()
