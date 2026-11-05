"""SQLite experiment log for hunt/backtest runs (A.6)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from moonshot.paths import ROOT

DEFAULT_DB = ROOT / "runs" / "experiments.db"


def _connect(db_path: Path | str = DEFAULT_DB) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.row_factory = sqlite3.Row
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            started_ts TEXT NOT NULL,
            config_hash TEXT,
            data_hash TEXT,
            metrics_json TEXT,
            meta_json TEXT
        );
        CREATE TABLE IF NOT EXISTS trial_counter (
            config_hash TEXT PRIMARY KEY,
            first_seen_ts TEXT NOT NULL,
            run_ids_json TEXT NOT NULL
        );
        """
    )
    con.commit()
    return con


def log_run(
    *,
    run_id: str,
    kind: str,
    config_hash: str | None = None,
    data_hash: str | None = None,
    metrics: Mapping[str, Any] | None = None,
    meta: Mapping[str, Any] | None = None,
    started_ts: str | None = None,
    db_path: Path | str = DEFAULT_DB,
) -> None:
    ts = started_ts or datetime.now(timezone.utc).isoformat()
    con = _connect(db_path)
    try:
        con.execute(
            """INSERT OR REPLACE INTO runs
               (run_id, kind, started_ts, config_hash, data_hash, metrics_json, meta_json)
               VALUES (?,?,?,?,?,?,?)""",
            (
                run_id,
                kind,
                ts,
                config_hash,
                data_hash,
                json.dumps(dict(metrics or {}), default=str),
                json.dumps(dict(meta or {}), default=str),
            ),
        )
        if config_hash:
            row = con.execute(
                "SELECT run_ids_json FROM trial_counter WHERE config_hash = ?",
                (config_hash,),
            ).fetchone()
            if row is None:
                con.execute(
                    "INSERT INTO trial_counter (config_hash, first_seen_ts, run_ids_json) VALUES (?,?,?)",
                    (config_hash, ts, json.dumps([run_id])),
                )
            else:
                ids = json.loads(row["run_ids_json"] or "[]")
                if run_id not in ids:
                    ids.append(run_id)
                con.execute(
                    "UPDATE trial_counter SET run_ids_json = ? WHERE config_hash = ?",
                    (json.dumps(ids), config_hash),
                )
        con.commit()
    finally:
        con.close()


def get_runs(
    filters: Mapping[str, Any] | None = None,
    *,
    db_path: Path | str = DEFAULT_DB,
) -> list[dict[str, Any]]:
    con = _connect(db_path)
    try:
        sql = "SELECT * FROM runs WHERE 1=1"
        args: list[Any] = []
        f = dict(filters or {})
        if "kind" in f:
            sql += " AND kind = ?"
            args.append(f["kind"])
        if "config_hash" in f:
            sql += " AND config_hash = ?"
            args.append(f["config_hash"])
        if "data_hash" in f:
            sql += " AND data_hash = ?"
            args.append(f["data_hash"])
        sql += " ORDER BY started_ts"
        return [dict(r) for r in con.execute(sql, args).fetchall()]
    finally:
        con.close()


def trial_counter_count(*, db_path: Path | str = DEFAULT_DB) -> int:
    con = _connect(db_path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM trial_counter").fetchone()[0])
    finally:
        con.close()
