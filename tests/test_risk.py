"""RiskEngine daily halt, streak pause, DD kill, paper drift."""
from __future__ import annotations

import pandas as pd

from moonshot.config import Config
from moonshot.risk import RiskEngine


def test_daily_halt_resets_next_day() -> None:
    r = RiskEngine(Config(max_daily_loss_pct=0.05), silent=True)
    r.update_equity(50.0, pd.Timestamp("2025-01-01 04:00", tz="UTC"))
    r.update_equity(46.0, pd.Timestamp("2025-01-01 08:00", tz="UTC"))
    assert not r.trading_allowed(pd.Timestamp("2025-01-01 09:00", tz="UTC"))
    r.update_equity(46.0, pd.Timestamp("2025-01-02 00:00", tz="UTC"))
    assert r.trading_allowed(pd.Timestamp("2025-01-02 01:00", tz="UTC"))


def test_peak_drawdown_kills() -> None:
    r = RiskEngine(Config(max_peak_drawdown_pct=0.20), silent=True)
    r.update_equity(100.0, pd.Timestamp("2025-01-01", tz="UTC"))
    r.update_equity(79.0, pd.Timestamp("2025-01-02", tz="UTC"))
    assert r.kill_switch_active
    assert not r.trading_allowed(pd.Timestamp("2025-01-02 01:00", tz="UTC"))


def test_streak_pause() -> None:
    cfg = Config(streak_losses=3, streak_pause_hours=2)
    r = RiskEngine(cfg, silent=True)
    t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    for i in range(3):
        r.register_trade_result(-1.0, t0 + pd.Timedelta(hours=i))
    assert r.pause_until is not None
    assert not r.trading_allowed(t0 + pd.Timedelta(hours=1))
    assert r.trading_allowed(r.pause_until + pd.Timedelta(minutes=1))


def test_paper_drift_halts_on_bad_exp() -> None:
    r = RiskEngine(Config(), silent=True)
    r.set_backtest_band(soft_ret=0.10, week_ret=0.20, min_trades=5, exp_floor=-0.20)
    assert not r.check_paper_drift(n_trades=3, paper_exp=-0.5, paper_ret=-0.1)
    assert r.check_paper_drift(n_trades=10, paper_exp=-0.5, paper_ret=-0.1)
    assert r.kill_switch_active
    assert "paper drift" in r.halt_reason


def test_paper_drift_halts_on_soft_gap() -> None:
    r = RiskEngine(Config(), silent=True)
    r.set_backtest_band(soft_ret=0.40, min_trades=5, soft_slack=0.30, exp_floor=-0.99)
    # paper ret 0% is 40pp under soft 40% with 30% slack → halt
    assert r.check_paper_drift(n_trades=10, paper_exp=0.0, paper_ret=0.0)
    assert r.drift_halt
