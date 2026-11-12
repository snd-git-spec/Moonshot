from __future__ import annotations

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    TradeResult,
)
from moonshot.position import (
    Position,
)

if TYPE_CHECKING:
    from moonshot.risk import RiskEngine

# Extra trade columns for observability spine (additive migrate).
_TRADE_OBS_COLS: tuple[tuple[str, str], ...] = (
    ("intended_qty", "REAL"),
    ("filled_qty", "REAL"),
    ("slippage_bps", "REAL"),
    ("assumed_slippage_bps", "REAL"),
    ("spread_bps", "REAL"),
    ("funding_rate_entry", "REAL"),
    ("mae_r", "REAL"),
    ("mfe_r", "REAL"),
    ("bars_held", "INTEGER"),
    ("open_config_hash", "TEXT"),
)

_RISK_OBS_COLS: tuple[tuple[str, str], ...] = (
    ("effective_lev", "REAL"),
    ("net_lev", "REAL"),
    ("margin_used", "REAL"),
    ("free_margin", "REAL"),
    ("min_liq_distance_pct", "REAL"),
    ("config_hash", "TEXT"),
    ("feed_age_s", "REAL"),
)


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
            CREATE TABLE IF NOT EXISTS rejected_entries (
                ts TEXT, mode TEXT, symbol TEXT, direction TEXT,
                entry REAL, stop REAL, reason TEXT);
            CREATE TABLE IF NOT EXISTS cycle_snapshot (
                ts TEXT, mode TEXT, equity REAL, peak REAL, drawdown REAL,
                effective_lev REAL, net_lev REAL, open_symbols TEXT,
                pause_until TEXT, halt_reason TEXT, kill INTEGER,
                config_hash TEXT, feed_age_s REAL, cycle_id TEXT);
            CREATE TABLE IF NOT EXISTS expectancy_rollup (
                ts TEXT, mode TEXT, config_hash TEXT, n INTEGER,
                mean_r REAL, ci_low REAL, ci_high REAL, cost_share REAL,
                max_dd REAL, liquidations INTEGER, label TEXT);
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
        for col, decl in (
            ("config_hash", "TEXT"),
            ("config_json", "TEXT"),
            ("initial_risk_dollars", "REAL"),
        ):
            self._ensure_col("trades", col, decl)
        for col, decl in _TRADE_OBS_COLS:
            self._ensure_col("trades", col, decl)
        for col, decl in _RISK_OBS_COLS:
            self._ensure_col("risk_snapshot", col, decl)
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_config_hash ON trades(config_hash)"
        )
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_trades_open_config_hash ON trades(open_config_hash)"
        )
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_events_kind_ts ON events(kind, ts)")
        self.conn.execute("CREATE INDEX IF NOT EXISTS idx_cycle_snapshot_ts ON cycle_snapshot(ts)")
        self.conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_expectancy_rollup_ts ON expectancy_rollup(ts)"
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS rejected_entries (
                ts TEXT, mode TEXT, symbol TEXT, direction TEXT,
                entry REAL, stop REAL, reason TEXT)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cycle_snapshot (
                ts TEXT, mode TEXT, equity REAL, peak REAL, drawdown REAL,
                effective_lev REAL, net_lev REAL, open_symbols TEXT,
                pause_until TEXT, halt_reason TEXT, kill INTEGER,
                config_hash TEXT, feed_age_s REAL, cycle_id TEXT)
            """
        )
        self.conn.execute(
            """
            CREATE TABLE IF NOT EXISTS expectancy_rollup (
                ts TEXT, mode TEXT, config_hash TEXT, n INTEGER,
                mean_r REAL, ci_low REAL, ci_high REAL, cost_share REAL,
                max_dd REAL, liquidations INTEGER, label TEXT)
            """
        )

    def _soft(self, label: str, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 — observability must never abort trading
            log.warning("persist %s failed: %s", label, e)

    def log_rejected_entry(
        self,
        ts: datetime,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        reason: str,
    ) -> None:
        def _do() -> None:
            self.conn.execute(
                """INSERT INTO rejected_entries
                   (ts, mode, symbol, direction, entry, stop, reason)
                   VALUES (?,?,?,?,?,?,?)""",
                (
                    pd.Timestamp(ts).isoformat(),
                    self.mode,
                    symbol,
                    direction,
                    float(entry),
                    float(stop),
                    reason,
                ),
            )
            self.conn.commit()

        self._soft("rejected_entry", _do)

    def log_trade(self, t: TradeResult) -> None:
        def _do() -> None:
            self.conn.execute(
                """INSERT INTO trades
                   (ts, symbol, direction, entry, exit, qty, pnl, r_multiple, liquidated,
                    exit_reason, adds, fees, funding, entry_time, partial_taken, mode,
                    config_hash, config_json, initial_risk_dollars,
                    intended_qty, filled_qty, slippage_bps, assumed_slippage_bps, spread_bps,
                    funding_rate_entry, mae_r, mfe_r, bars_held, open_config_hash)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pd.Timestamp(t.exit_time).isoformat(),
                    t.symbol,
                    t.direction.value,
                    t.entry_price,
                    t.exit_price,
                    t.initial_qty,
                    t.pnl,
                    t.r_multiple,
                    int(t.liquidated),
                    t.exit_reason.value,
                    t.max_adds,
                    t.fees,
                    t.funding,
                    pd.Timestamp(t.entry_time).isoformat(),
                    int(t.partial_taken),
                    self.mode,
                    getattr(t, "config_hash", None),
                    getattr(t, "config_json", None),
                    float(getattr(t, "initial_risk_dollars", 0.0) or 0.0),
                    getattr(t, "intended_qty", None),
                    getattr(t, "filled_qty", None),
                    getattr(t, "slippage_bps", None),
                    getattr(t, "assumed_slippage_bps", None),
                    getattr(t, "spread_bps", None),
                    getattr(t, "funding_rate_entry", None),
                    getattr(t, "mae_r", None),
                    getattr(t, "mfe_r", None),
                    getattr(t, "bars_held", None),
                    getattr(t, "open_config_hash", None),
                ),
            )
            self.conn.commit()

        self._soft("trade", _do)

    def log_equity(self, ts: datetime, equity: float, peak: float) -> None:
        def _do() -> None:
            dd = (peak - equity) / peak if peak > 0 else 0.0
            self.conn.execute(
                "INSERT INTO equity (ts, equity, peak, drawdown, mode) VALUES (?,?,?,?,?)",
                (pd.Timestamp(ts).isoformat(), equity, peak, dd, self.mode),
            )
            self.conn.commit()

        self._soft("equity", _do)

    def log_risk(
        self,
        ts: datetime,
        risk: "RiskEngine",
        *,
        effective_lev: float | None = None,
        net_lev: float | None = None,
        margin_used: float | None = None,
        free_margin: float | None = None,
        min_liq_distance_pct: float | None = None,
        config_hash: str | None = None,
        feed_age_s: float | None = None,
    ) -> None:
        def _do() -> None:
            peak = risk.peak_equity
            eq = risk.equity
            dd = (peak - eq) / peak if peak > 0 else 0.0
            pause = risk.pause_until.isoformat() if risk.pause_until is not None else None
            self.conn.execute(
                """INSERT INTO risk_snapshot
                   (ts, mode, equity, peak, drawdown, halt_reason, kill, streak, daily_halt,
                    pause_until, effective_lev, net_lev, margin_used, free_margin,
                    min_liq_distance_pct, config_hash, feed_age_s)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pd.Timestamp(ts).isoformat(),
                    self.mode,
                    eq,
                    peak,
                    dd,
                    risk.halt_reason or "",
                    int(risk.kill_switch_active),
                    int(risk.consecutive_losses),
                    int(risk.daily_halt),
                    pause,
                    effective_lev,
                    net_lev,
                    margin_used,
                    free_margin,
                    min_liq_distance_pct,
                    config_hash,
                    feed_age_s,
                ),
            )
            self.conn.commit()

        self._soft("risk", _do)

    def log_cycle_snapshot(
        self,
        ts: datetime,
        *,
        equity: float,
        peak: float,
        effective_lev: float,
        net_lev: float,
        open_symbols: list[str] | str,
        pause_until: str | None,
        halt_reason: str | None,
        kill: bool,
        config_hash: str | None,
        feed_age_s: float | None,
        cycle_id: str | None,
    ) -> None:
        def _do() -> None:
            dd = (peak - equity) / peak if peak > 0 else 0.0
            syms = (
                open_symbols
                if isinstance(open_symbols, str)
                else ",".join(str(s) for s in open_symbols)
            )
            self.conn.execute(
                """INSERT INTO cycle_snapshot
                   (ts, mode, equity, peak, drawdown, effective_lev, net_lev, open_symbols,
                    pause_until, halt_reason, kill, config_hash, feed_age_s, cycle_id)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pd.Timestamp(ts).isoformat(),
                    self.mode,
                    float(equity),
                    float(peak),
                    float(dd),
                    float(effective_lev),
                    float(net_lev),
                    syms,
                    pause_until,
                    halt_reason or "",
                    int(bool(kill)),
                    config_hash,
                    feed_age_s,
                    cycle_id or "",
                ),
            )
            self.conn.commit()

        self._soft("cycle_snapshot", _do)

    def log_fill(
        self,
        ts: datetime,
        symbol: str,
        *,
        side: str,
        reason: str,
        intended_qty: float,
        filled_qty: float,
        ref_price: float | None,
        fill_price: float,
        slippage_bps: float | None,
        assumed_slippage_bps: float | None,
        spread_bps: float | None = None,
    ) -> None:
        payload = {
            "intended_qty": float(intended_qty),
            "filled_qty": float(filled_qty),
            "ref_price": None if ref_price is None else float(ref_price),
            "fill_price": float(fill_price),
            "slippage_bps": None if slippage_bps is None else float(slippage_bps),
            "assumed_slippage_bps": (
                None if assumed_slippage_bps is None else float(assumed_slippage_bps)
            ),
            "spread_bps": None if spread_bps is None else float(spread_bps),
            "side": side,
            "reason": reason,
        }
        self.log_event(ts, symbol, "fill", payload)

    def log_expectancy_rollup(
        self,
        ts: datetime,
        *,
        config_hash: str | None,
        n: int,
        mean_r: float,
        ci_low: float,
        ci_high: float,
        cost_share: float,
        max_dd: float,
        liquidations: int,
        label: str,
    ) -> None:
        def _do() -> None:
            self.conn.execute(
                """INSERT INTO expectancy_rollup
                   (ts, mode, config_hash, n, mean_r, ci_low, ci_high, cost_share,
                    max_dd, liquidations, label)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    pd.Timestamp(ts).isoformat(),
                    self.mode,
                    config_hash,
                    int(n),
                    float(mean_r) if mean_r == mean_r else None,
                    float(ci_low) if ci_low == ci_low else None,
                    float(ci_high) if ci_high == ci_high else None,
                    float(cost_share) if cost_share == cost_share else None,
                    float(max_dd),
                    int(liquidations),
                    label,
                ),
            )
            self.conn.commit()

        self._soft("expectancy_rollup", _do)

    def last_expectancy_rollup_ts(self) -> Optional[pd.Timestamp]:
        try:
            row = self.conn.execute(
                "SELECT ts FROM expectancy_rollup WHERE mode=? ORDER BY ts DESC LIMIT 1",
                (self.mode,),
            ).fetchone()
            if not row:
                return None
            return pd.Timestamp(row[0])
        except Exception:
            return None

    def log_event(
        self, ts: datetime, symbol: str, kind: str, payload: Optional[dict] = None
    ) -> None:
        def _do() -> None:
            self.conn.execute(
                "INSERT INTO events (ts, mode, symbol, kind, payload) VALUES (?,?,?,?,?)",
                (
                    pd.Timestamp(ts).isoformat(),
                    self.mode,
                    symbol,
                    kind,
                    json.dumps(payload or {}, default=str),
                ),
            )
            self.conn.commit()

        self._soft(f"event:{kind}", _do)

    def save_position(self, symbol: str, pos: Optional[Position]) -> None:
        def _do() -> None:
            if pos is None:
                self.conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
            else:
                self.conn.execute(
                    "INSERT OR REPLACE INTO positions VALUES (?,?)",
                    (symbol, json.dumps(pos.to_dict())),
                )
            self.conn.commit()

        self._soft("position", _do)

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
        cols = self._cols("trades")
        base = [
            "ts",
            "symbol",
            "direction",
            "entry",
            "exit",
            "qty",
            "pnl",
            "r_multiple",
            "liquidated",
            "exit_reason",
            "adds",
            "fees",
            "funding",
            "entry_time",
            "partial_taken",
            "mode",
            "config_hash",
            "initial_risk_dollars",
        ]
        extra = [c for c, _ in _TRADE_OBS_COLS]
        select = [c for c in base + extra if c in cols]
        q = f"SELECT {', '.join(select)} FROM trades"
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

    def load_cycle_snapshots(self, mode: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        q = """SELECT ts, mode, equity, peak, drawdown, effective_lev, net_lev,
                      open_symbols, pause_until, halt_reason, kill, config_hash,
                      feed_age_s, cycle_id FROM cycle_snapshot"""
        args: list[Any] = []
        if mode:
            q += " WHERE mode=?"
            args.append(mode)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        try:
            df = pd.read_sql_query(q, self.conn, params=args)
        except Exception:
            return pd.DataFrame()
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def load_expectancy_rollups(self, mode: Optional[str] = None, limit: int = 50) -> pd.DataFrame:
        q = """SELECT ts, mode, config_hash, n, mean_r, ci_low, ci_high, cost_share,
                      max_dd, liquidations, label FROM expectancy_rollup"""
        args: list[Any] = []
        if mode:
            q += " WHERE mode=?"
            args.append(mode)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        try:
            df = pd.read_sql_query(q, self.conn, params=args)
        except Exception:
            return pd.DataFrame()
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def load_risk(self, mode: Optional[str] = None) -> Optional[dict]:
        cols = self._cols("risk_snapshot")
        wanted = [
            "ts",
            "mode",
            "equity",
            "peak",
            "drawdown",
            "halt_reason",
            "kill",
            "streak",
            "daily_halt",
            "pause_until",
            "effective_lev",
            "net_lev",
            "margin_used",
            "free_margin",
            "min_liq_distance_pct",
            "config_hash",
            "feed_age_s",
        ]
        select = [c for c in wanted if c in cols]
        q = f"SELECT {', '.join(select)} FROM risk_snapshot"
        args: tuple = ()
        if mode:
            q += " WHERE mode=?"
            args = (mode,)
        q += " ORDER BY ts DESC LIMIT 1"
        row = self.conn.execute(q, args).fetchone()
        if not row:
            return None
        return dict(zip(select, row, strict=False))

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
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:  # noqa: BLE001
            log.warning("Notifier failed: %s", e)
