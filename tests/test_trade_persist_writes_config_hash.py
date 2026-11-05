"""A.5-write — closed trades persist config_hash + initial_risk_dollars."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from moonshot.config import Config
from moonshot.config_hash import canonical_json, config_hash
from moonshot.persist import Persistence
from moonshot.types import Direction, ExitReason, TradeResult


def test_trade_persist_writes_config_hash(tmp_path: Path):
    cfg = Config(leverage=10, timeframe="5m", strategy="momentum_15")
    db = Persistence(str(tmp_path / "t.db"), mode="paper")
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tr = TradeResult(
        symbol="BTC/USDT:USDT",
        direction=Direction.LONG,
        entry_time=ts,
        exit_time=ts,
        entry_price=100.0,
        exit_price=105.0,
        initial_qty=1.0,
        pnl=4.5,
        r_multiple=1.5,
        liquidated=False,
        exit_reason=ExitReason.STOP,
        max_adds=0,
        partial_taken=False,
        fees=0.1,
        funding=0.0,
        initial_risk_dollars=3.0,
        config_hash=config_hash(cfg),
        config_json=canonical_json(cfg),
    )
    db.log_trade(tr)
    row = db.conn.execute(
        "SELECT config_hash, config_json, initial_risk_dollars FROM trades"
    ).fetchone()
    assert row[0] == config_hash(cfg)
    assert row[1] == canonical_json(cfg)
    assert float(row[2]) == 3.0
