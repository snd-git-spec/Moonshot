from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot._pull import pull
import moonshot.types as _moonshot_types
pull(_moonshot_types, globals())
import moonshot.config as _moonshot_config
pull(_moonshot_config, globals())
from moonshot.paths import ROOT, DB_PATH

class Persistence:
    """SQLite log for paper/live. WAL + named columns so the dashboard can read safely."""

    def __init__(self, path: str, mode: str = "paper"):
        self.path = path
        self.mode = mode if mode in ("paper", "live") else "paper"
        self.conn = sqlite3.connect(path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                ts TEXT, symbol TEXT, direction TEXT, entry REAL, exit REAL, qty REAL,
                pnl REAL, r_multiple REAL, liquidated INTEGER, exit_reason TEXT, adds INTEGER,
                fees REAL, funding REAL, entry_time TEXT, partial_taken INTEGER, mode TEXT);
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT, equity REAL, peak REAL, drawdown REAL, mode TEXT);
            CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, state TEXT);
            CREATE TABLE IF NOT EXISTS risk_snapshot (
                ts TEXT, mode TEXT, equity REAL, peak REAL, drawdown REAL,
                halt_reason TEXT, kill INTEGER, streak INTEGER, daily_halt INTEGER,
                pause_until TEXT);
            CREATE TABLE IF NOT EXISTS events (
                ts TEXT, mode TEXT, symbol TEXT, kind TEXT, payload TEXT);
            """
        )
        self._migrate()
        self.conn.commit()

    def _cols(self, table: str) -> set:
        return {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _ensure_col(self, table: str, col: str, decl: str) -> None:
        if col not in self._cols(table):
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def _migrate(self) -> None:
        for col, decl in (("entry_time", "TEXT"), ("partial_taken", "INTEGER"), ("mode", "TEXT")):
            self._ensure_col("trades", col, decl)
        self._ensure_col("equity", "mode", "TEXT")

    def log_trade(self, t: TradeResult) -> None:
        self.conn.execute(
            """INSERT INTO trades
               (ts, symbol, direction, entry, exit, qty, pnl, r_multiple, liquidated,
                exit_reason, adds, fees, funding, entry_time, partial_taken, mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pd.Timestamp(t.exit_time).isoformat(), t.symbol, t.direction.value, t.entry_price,
             t.exit_price, t.initial_qty, t.pnl, t.r_multiple, int(t.liquidated),
             t.exit_reason.value, t.max_adds, t.fees, t.funding,
             pd.Timestamp(t.entry_time).isoformat(), int(t.partial_taken), self.mode))
        self.conn.commit()

    def log_equity(self, ts: datetime, equity: float, peak: float) -> None:
        dd = (peak - equity) / peak if peak > 0 else 0.0
        self.conn.execute(
            "INSERT INTO equity (ts, equity, peak, drawdown, mode) VALUES (?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), equity, peak, dd, self.mode))
        self.conn.commit()

    def log_risk(self, ts: datetime, risk: "RiskEngine") -> None:
        peak = risk.peak_equity
        eq = risk.equity
        dd = (peak - eq) / peak if peak > 0 else 0.0
        pause = risk.pause_until.isoformat() if risk.pause_until is not None else None
        self.conn.execute(
            """INSERT INTO risk_snapshot
               (ts, mode, equity, peak, drawdown, halt_reason, kill, streak, daily_halt, pause_until)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (pd.Timestamp(ts).isoformat(), self.mode, eq, peak, dd, risk.halt_reason or "",
             int(risk.kill_switch_active), int(risk.consecutive_losses), int(risk.daily_halt), pause))
        self.conn.commit()

    def log_event(self, ts: datetime, symbol: str, kind: str, payload: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, mode, symbol, kind, payload) VALUES (?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), self.mode, symbol, kind,
             json.dumps(payload or {}, default=str)))
        self.conn.commit()

    def save_position(self, symbol: str, pos: Optional[Position]) -> None:
        if pos is None:
            self.conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        else:
            self.conn.execute("INSERT OR REPLACE INTO positions VALUES (?,?)",
                              (symbol, json.dumps(pos.to_dict())))
        self.conn.commit()

    def load_positions(self) -> dict:
        rows = self.conn.execute("SELECT symbol, state FROM positions").fetchall()
        return {s: Position.from_dict(json.loads(j)) for s, j in rows}

    def closed_pnl_sum(self, mode: Optional[str] = None) -> float:
        """Sum of closed-trade PnL for cash reconstruction after a paper restart."""
        q = "SELECT COALESCE(SUM(pnl), 0) FROM trades"
        args: tuple = ()
        if mode:
            q += " WHERE mode=?"
            args = (mode,)
        row = self.conn.execute(q, args).fetchone()
        return float(row[0] if row and row[0] is not None else 0.0)

    def load_equity(self, mode: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT ts, equity, peak, drawdown, mode FROM equity"
        args: tuple = ()
        if mode:
            q += " WHERE mode=? OR mode IS NULL"
            args = (mode,)
        q += " ORDER BY ts"
        df = pd.read_sql_query(q, self.conn, params=args)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def load_trades(self, mode: Optional[str] = None) -> pd.DataFrame:
        q = """SELECT ts, symbol, direction, entry, exit, qty, pnl, r_multiple, liquidated,
                      exit_reason, adds, fees, funding, entry_time, partial_taken, mode
               FROM trades"""
        args: tuple = ()
        if mode:
            q += " WHERE mode=? OR mode IS NULL"
            args = (mode,)
        q += " ORDER BY ts"
        df = pd.read_sql_query(q, self.conn, params=args)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            if "entry_time" in df.columns:
                df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        return df

    def load_events(self, mode: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        q = "SELECT ts, mode, symbol, kind, payload FROM events"
        args: list = []
        if mode:
            q += " WHERE mode=?"
            args.append(mode)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        df = pd.read_sql_query(q, self.conn, params=args)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def load_risk(self, mode: Optional[str] = None) -> Optional[dict]:
        q = """SELECT ts, mode, equity, peak, drawdown, halt_reason, kill, streak,
                      daily_halt, pause_until FROM risk_snapshot"""
        args: tuple = ()
        if mode:
            q += " WHERE mode=?"
            args = (mode,)
        q += " ORDER BY ts DESC LIMIT 1"
        row = self.conn.execute(q, args).fetchone()
        if not row:
            return None
        cols = ["ts", "mode", "equity", "peak", "drawdown", "halt_reason", "kill",
                "streak", "daily_halt", "pause_until"]
        return dict(zip(cols, row))

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def open_readonly(path: str) -> sqlite3.Connection:
        """Dashboard-safe reader (WAL-friendly). Raises if the file is missing."""
        uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn


class Notifier:
    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

    def send(self, msg: str) -> None:
        log.info("NOTIFY: %s", msg)
        if not self.enabled:
            return
        try:
            import urllib.request
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data=json.dumps({"chat_id": self.chat_id, "text": msg}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:  # noqa: BLE001
            log.warning("Notifier failed: %s", e)


