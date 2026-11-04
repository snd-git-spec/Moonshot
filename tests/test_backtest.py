"""BacktestEngine determinism on synthetic OHLCV."""

from __future__ import annotations

import math

from moonshot.backtest import BacktestEngine
from moonshot.config import Config
from moonshot.data import synthetic_ohlcv
from moonshot.features import prepare_data


def _run_twice(cfg: Config):
    raw = {
        s: synthetic_ohlcv(s, 1200, tf=cfg.timeframe) for s in ("BTC/USDT:USDT", "ETH/USDT:USDT")
    }
    data = prepare_data(raw, cfg)
    a = BacktestEngine(data, cfg).run()
    b = BacktestEngine(data, cfg).run()
    return a, b


def test_backtest_deterministic(cfg: Config) -> None:
    a, b = _run_twice(cfg)
    assert a.trades == b.trades
    assert a.final_equity == b.final_equity
    assert a.max_drawdown == b.max_drawdown
    assert a.total_return == b.total_return
    assert list(a.equity_curve.values) == list(b.equity_curve.values)


def test_backtest_finite_equity(cfg: Config) -> None:
    a, _ = _run_twice(cfg)
    assert math.isfinite(a.final_equity)
    assert a.final_equity >= 0
    assert a.equity_curve.index.is_monotonic_increasing
