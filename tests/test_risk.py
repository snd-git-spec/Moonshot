"""RiskEngine daily halt, streak pause, DD circuit, paper drift."""

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


def test_peak_drawdown_pauses_then_resumes_in_backtest() -> None:
    r = RiskEngine(
        Config(
            max_peak_drawdown_pct=0.20,
            kill_resume_hours=24,
            max_daily_loss_pct=0.99,  # isolate DD trip from daily halt
        ),
        silent=True,
    )
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    r.update_equity(100.0, t0)
    r.update_equity(79.0, t0 + pd.Timedelta(days=1))
    # Backtest: circuit pause, not permanent kill
    assert not r.kill_switch_active
    assert r.pause_until is not None
    assert not r.trading_allowed(t0 + pd.Timedelta(days=1, hours=1))
    # After pause window, trading resumes with re-anchored peak
    later = r.pause_until + pd.Timedelta(minutes=1)
    assert r.trading_allowed(later)
    assert r.pause_until is None
    assert abs(r.peak_equity - r.equity) < 1e-9


def test_peak_drawdown_permanent_when_live() -> None:
    r = RiskEngine(Config(max_peak_drawdown_pct=0.20), silent=False)
    r.update_equity(100.0, pd.Timestamp("2025-01-01", tz="UTC"))
    r.update_equity(79.0, pd.Timestamp("2025-01-02", tz="UTC"))
    assert r.kill_switch_active
    assert not r.trading_allowed(pd.Timestamp("2025-01-10", tz="UTC"))


def test_streak_pause() -> None:
    cfg = Config(streak_losses=3, streak_pause_hours=2)
    r = RiskEngine(cfg, silent=True)
    t0 = pd.Timestamp("2025-01-01 00:00", tz="UTC")
    for i in range(3):
        r.register_trade_result(-1.0, t0 + pd.Timedelta(hours=i))
    assert r.pause_until is not None
    assert not r.trading_allowed(t0 + pd.Timedelta(hours=1))
    assert r.trading_allowed(r.pause_until + pd.Timedelta(minutes=1))


def test_expectancy_trip_resumes_in_backtest() -> None:
    r = RiskEngine(
        Config(expectancy_window=5, expectancy_floor=-0.10, kill_resume_hours=48),
        silent=True,
    )
    t0 = pd.Timestamp("2025-06-01", tz="UTC")
    for i in range(5):
        r.register_trade_result(-1.0, t0 + pd.Timedelta(hours=i))
    assert not r.kill_switch_active
    assert r.pause_until is not None
    assert not r.trading_allowed(t0 + pd.Timedelta(hours=6))
    assert r.trading_allowed(r.pause_until + pd.Timedelta(minutes=1))


def test_paper_drift_halts_on_bad_exp() -> None:
    r = RiskEngine(Config(), silent=False)
    r.set_backtest_band(soft_ret=0.10, week_ret=0.20, min_trades=5, exp_floor=-0.20)
    assert not r.check_paper_drift(n_trades=3, paper_exp=-0.5, paper_ret=-0.1)
    assert r.check_paper_drift(n_trades=10, paper_exp=-0.5, paper_ret=-0.1)
    assert r.kill_switch_active
    assert "paper drift" in r.halt_reason


def test_paper_drift_halts_on_soft_gap() -> None:
    r = RiskEngine(Config(), silent=False)
    r.set_backtest_band(soft_ret=0.40, min_trades=5, soft_slack=0.30, exp_floor=-0.99)
    assert r.check_paper_drift(n_trades=10, paper_exp=0.0, paper_ret=0.0)
    assert r.drift_halt
