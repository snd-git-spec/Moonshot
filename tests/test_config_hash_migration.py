"""0.2 — config_hash helpers + migration + persist fields."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from moonshot.config import Config
from moonshot.config_hash import canonical_json, config_hash
from moonshot.persist import Persistence
from moonshot.types import Direction, ExitReason, TradeResult
from scripts.migrate_add_config_hash import migrate


def test_config_hash_shared_for_identical_configs():
    a = Config(leverage=10, timeframe="5m", strategy="momentum_15")
    b = Config(leverage=10, timeframe="5m", strategy="momentum_15")
    assert config_hash(a) == config_hash(b)
    assert canonical_json(a) == canonical_json(b)


def test_config_hash_differs_for_different_configs():
    a = Config(leverage=10, timeframe="5m")
    b = Config(leverage=5, timeframe="5m")
    assert config_hash(a) != config_hash(b)


def test_migration_idempotent(tmp_path: Path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(str(db))
    con.execute(
        """CREATE TABLE trades (
            ts TEXT, symbol TEXT, direction TEXT, entry REAL, exit REAL, qty REAL,
            pnl REAL, r_multiple REAL, liquidated INTEGER, exit_reason TEXT, adds INTEGER,
            fees REAL, funding REAL, entry_time TEXT, partial_taken INTEGER, mode TEXT)"""
    )
    con.commit()
    con.close()
    r1 = migrate(db, backup=False)
    r2 = migrate(db, backup=False)
    assert "config_hash" in r1["cols"]
    assert r2["added"] == []
    con = sqlite3.connect(str(db))
    # SQLite may name the index; ensure ours exists via sql
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='index'")]
    assert "idx_trades_config_hash" in names
    con.close()


def test_initial_risk_dollars_populated(tmp_path: Path):
    db = tmp_path / "p.db"
    p = Persistence(str(db), mode="paper")
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tr = TradeResult(
        symbol="BTC/USDT:USDT",
        direction=Direction.LONG,
        entry_time=ts,
        exit_time=ts,
        entry_price=100.0,
        exit_price=110.0,
        initial_qty=1.0,
        pnl=10.0,
        r_multiple=2.0,
        liquidated=False,
        exit_reason=ExitReason.STOP,
        max_adds=0,
        partial_taken=False,
        fees=0.1,
        funding=0.0,
        initial_risk_dollars=5.0,
        config_hash="abc",
        config_json="{}",
    )
    p.log_trade(tr)
    row = p.conn.execute("SELECT initial_risk_dollars, config_hash FROM trades").fetchone()
    assert float(row[0]) == 5.0
    assert row[1] == "abc"
