"""PaperBroker fills, slippage, PnL, reduce-only."""

from __future__ import annotations

import pandas as pd
import pytest

from moonshot.brokers import PaperBroker
from moonshot.config import Config
from moonshot.types import Direction, ExecutionError


def test_open_and_equity_tracks_mark(cfg: Config) -> None:
    pb = PaperBroker(cfg)
    fill = pb.market_order("BTC/USDT:USDT", "buy", 0.01, 100.0)
    assert fill.side == "buy"
    assert fill.qty == pytest.approx(0.01)
    # Slippage bumps buy price up
    assert fill.price > 100.0
    pb.on_price("BTC/USDT:USDT", 110.0)
    eq = pb.get_equity()
    assert eq > cfg.starting_equity - 1.0  # fees small; mark profit


def test_reduce_only_rejects_same_direction(cfg: Config) -> None:
    pb = PaperBroker(cfg)
    pb.market_order("ETH/USDT:USDT", "buy", 0.1, 50.0)
    with pytest.raises(ExecutionError, match="reduce-only"):
        pb.market_order("ETH/USDT:USDT", "buy", 0.05, 51.0, reduce_only=True)


def test_reduce_only_closes_and_realizes_pnl(cfg: Config) -> None:
    pb = PaperBroker(cfg)
    pb.market_order("ETH/USDT:USDT", "buy", 1.0, 100.0)
    cash_after_open = pb.cash
    fill = pb.market_order("ETH/USDT:USDT", "sell", 1.0, 110.0, reduce_only=True)
    assert fill.side == "sell"
    assert pb.signed.get("ETH/USDT:USDT", 0.0) == 0.0
    # Realized gain credited (minus fees)
    assert pb.cash > cash_after_open


def test_exchange_stop_fires_on_price(cfg: Config) -> None:
    pb = PaperBroker(cfg)
    pb.market_order("BTC/USDT:USDT", "buy", 0.5, 100.0)
    pb.set_stop("BTC/USDT:USDT", Direction.LONG, 0.5, 95.0)
    pb.on_price("BTC/USDT:USDT", 96.0)
    assert pb.pop_stop_fill("BTC/USDT:USDT") is None
    pb.on_price("BTC/USDT:USDT", 94.0)
    fill = pb.pop_stop_fill("BTC/USDT:USDT")
    assert fill is not None
    assert fill.side == "sell"


def test_rehydrate_restores_position(cfg: Config) -> None:
    from moonshot.position import Position

    pb = PaperBroker(cfg)
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    restored = {
        "BTC/USDT:USDT": Position.open(
            "BTC/USDT:USDT",
            Direction.LONG,
            5,
            100.0,
            1.0,
            98.0,
            t0,
            0.02,
        ),
    }
    pb.rehydrate(restored, closed_pnl=-1.5)
    assert abs(pb.signed["BTC/USDT:USDT"] - 1.0) < 1e-12
    assert abs(pb.stops["BTC/USDT:USDT"] - 98.0) < 1e-12
