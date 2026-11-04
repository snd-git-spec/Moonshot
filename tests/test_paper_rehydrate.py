"""PaperBroker last_mark + equity_stale across restart."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from moonshot.brokers import PaperBroker
from moonshot.config import Config
from moonshot.position import Position
from moonshot.types import Direction


def test_rehydrate_restores_last_mark_and_stale_until_tick(cfg: Config, tmp_path: Path) -> None:
    cfg = Config(starting_equity=50.0, db_path=str(tmp_path / "t.db"))
    pb = PaperBroker(cfg)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    pos = Position.open("BTC/USDT:USDT", Direction.LONG, 5, 100.0, 1.0, 98.0, t0, 0.02)
    # Snapshot after mark moved away from entry
    pb.rehydrate({"BTC/USDT:USDT": pos}, closed_pnl=0.0, last_marks={"BTC/USDT:USDT": 110.0})
    assert pb.equity_stale is True
    assert pb.last["BTC/USDT:USDT"] == 110.0
    eq_snap = pb.get_equity()
    # Unrealized uses restored mark, not avg_entry
    assert abs(eq_snap - (50.0 - 0.02 + (110.0 - 100.0) * 1.0)) < 1e-9

    # Persist marks as LiveRunner would
    runtime = {
        "last_mark": pb.mark_snapshot(),
        "equity_stale": True,
        "pause_until": None,
    }
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime))

    # Restart broker from same snapshot marks
    pb2 = PaperBroker(cfg)
    marks = json.loads(path.read_text())["last_mark"]
    pb2.rehydrate({"BTC/USDT:USDT": pos}, closed_pnl=0.0, last_marks=marks)
    assert abs(pb2.get_equity() - eq_snap) < 1e-9
    assert pb2.equity_stale is True

    # First tick clears stale and updates equity
    pb2.on_price("BTC/USDT:USDT", 120.0)
    assert pb2.equity_stale is False
    assert abs(pb2.get_equity() - (50.0 - 0.02 + 20.0)) < 1e-9


def test_rehydrate_without_marks_falls_back_to_avg_and_stale(cfg: Config) -> None:
    pb = PaperBroker(cfg)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    pos = Position.open("ETH/USDT:USDT", Direction.SHORT, 5, 50.0, 2.0, 52.0, t0, 0.01)
    pb.rehydrate({"ETH/USDT:USDT": pos}, closed_pnl=-1.0, last_marks=None)
    assert pb.last["ETH/USDT:USDT"] == 50.0
    assert pb.equity_stale is True
