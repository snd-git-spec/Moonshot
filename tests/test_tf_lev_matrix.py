"""TF×lev matrix helpers — cells, freeze_matrix, knob scaling, rank, memory."""

from __future__ import annotations

from dataclasses import replace

import pytest

from moonshot.config import Config
from moonshot.hunt.constants import HUNT_TF_ALLOW, HUNT_TARGETS
from moonshot.hunt.scoring import (
    WEEK_DAYS,
    _best_by_tf_raise,
    _best_by_tf_seed_ok,
    _freeze_risk,
    _freeze_risk_matrix,
    _lev_tf_allowed,
    _period_robust_ok,
    _rank_matrix_rows,
    _scale_bar_knobs_for_tf,
    _tf_lev_matrix_cfgs,
    _tf_minutes,
)


def _base(**kw) -> Config:
    defaults = dict(
        strategy="momentum_15",
        timeframe="5m",
        leverage=10,
        aggressive_mode=True,
        risk_pct=0.03,
        max_daily_loss_pct=0.32,
        max_peak_drawdown_pct=0.55,
        mom_lookback=12,
        cooldown_bars=3,
        breakout_lookback=28,
        time_stop_bars=32,
        squeeze_lookback=8,
        pullback_lookback=3,
    )
    defaults.update(kw)
    return Config(**defaults)


def test_matrix_cells_exact():
    base = _base()
    cfgs = _tf_lev_matrix_cfgs(base)
    pairs = {(str(c.timeframe), int(c.leverage)) for c in cfgs}
    expected = set()
    for tf in HUNT_TF_ALLOW:
        for lev in (5, 7, 10, 15, 25):
            if _lev_tf_allowed(lev, tf):
                expected.add((tf, lev))
    assert pairs == expected
    assert ("15m", 25) not in pairs
    assert ("5m", 25) in pairs
    assert len(pairs) == 15  # 5+4+3+3


def test_freeze_risk_matrix_allows_lev_tf():
    base = _base(leverage=10, timeframe="5m", risk_pct=0.03)
    cand = replace(base, leverage=15, timeframe="5m", risk_pct=0.05)
    out = _freeze_risk_matrix(base, cand)
    assert int(out.leverage) == 15
    assert str(out.timeframe) == "5m"
    assert float(out.max_account_leverage) == 15.0
    assert float(out.max_net_leverage) == 15.0
    assert float(out.risk_pct) == pytest.approx(0.03)
    assert float(out.max_daily_loss_pct) == float(base.max_daily_loss_pct)
    assert float(out.max_peak_drawdown_pct) == float(base.max_peak_drawdown_pct)
    # 1h cell with allowed lev
    out_1h = _freeze_risk_matrix(base, replace(base, leverage=7, timeframe="1h"))
    assert int(out_1h.leverage) == 7
    assert str(out_1h.timeframe) == "1h"


def test_freeze_risk_a1_still_pins_lev():
    base = _base(leverage=10, timeframe="15m", risk_pct=0.03)
    cand = replace(base, leverage=25, timeframe="15m", risk_pct=0.05)
    out = _freeze_risk(base, cand)
    assert int(out.leverage) == 10
    assert float(out.risk_pct) == pytest.approx(0.03)


def test_scale_bar_knobs_5m_to_1h():
    base = _base(timeframe="5m", mom_lookback=12, cooldown_bars=3)
    scaled = _scale_bar_knobs_for_tf(base, "1h")
    # 1h / 5m = 12× coarser → divide bar counts (same wall-clock)
    assert _tf_minutes("1h") / _tf_minutes("5m") == pytest.approx(12.0)
    assert int(scaled.mom_lookback) == 3  # max(3, round(12/12))
    assert int(scaled.cooldown_bars) == 0  # max(0, round(3/12))
    assert str(scaled.timeframe) == "1h"


def test_rank_prefers_higher_week_among_robust():
    rows = [
        {"tf": "5m", "leverage": 10, "week_ret": 0.20, "robust_ok": True, "skip": False},
        {"tf": "1h", "leverage": 10, "week_ret": 0.10, "robust_ok": True, "skip": False},
        {"tf": "5m", "leverage": 25, "week_ret": 0.50, "robust_ok": False, "skip": False},
    ]
    winner = _rank_matrix_rows(rows)
    assert winner is not None
    assert winner["tf"] == "5m"
    assert winner["week_ret"] == 0.20


def test_rank_accepts_zero_week_ret():
    """0.0 must not be treated as missing (falsy `or` bug)."""
    rows = [
        {"tf": "1h", "leverage": 5, "week_ret": 0.0, "robust_ok": True, "skip": False},
    ]
    winner = _rank_matrix_rows(rows)
    assert winner is not None
    assert winner["week_ret"] == 0.0


def test_best_by_tf_raise_only_per_tf():
    state: dict = {"best_by_tf": {}}
    _best_by_tf_raise(
        state,
        tf="5m",
        knobs={"timeframe": "5m", "leverage": 10},
        week_ret=0.20,
        oos_ret=0.05,
        generation=1,
        ts="t1",
    )
    _best_by_tf_raise(
        state,
        tf="1h",
        knobs={"timeframe": "1h", "leverage": 10},
        week_ret=0.15,
        oos_ret=0.02,
        generation=2,
        ts="t2",
    )
    # weaker 5m no-op
    _best_by_tf_raise(
        state,
        tf="5m",
        knobs={"timeframe": "5m", "leverage": 7},
        week_ret=0.10,
        oos_ret=0.01,
        generation=3,
        ts="t3",
    )
    assert state["best_by_tf"]["5m"]["week_ret"] == 0.20
    assert state["best_by_tf"]["5m"]["knobs"]["leverage"] == 10
    assert state["best_by_tf"]["1h"]["week_ret"] == 0.15


def test_best_by_tf_stale_rejected(monkeypatch):
    monkeypatch.setitem(HUNT_TARGETS, "require_period_robust", True)
    monkeypatch.setitem(HUNT_TARGETS, "prior_week_floor", 0.0)
    monkeypatch.setitem(HUNT_TARGETS, "oos_floor", 0.0)
    blob = {
        "knobs": {"timeframe": "5m"},
        "week_ret": 0.4,
        "oos_ret": 0.1,
        "prior_ret": -0.2,
    }
    assert _best_by_tf_seed_ok(blob, prior_ret=-0.2, oos_ret=0.1) is False
    assert _best_by_tf_seed_ok(
        {**blob, "prior_ret": 0.05}, prior_ret=0.05, oos_ret=0.1
    )


def test_empty_robust_set_keeps_baseline():
    assert _rank_matrix_rows([]) is None
    assert _rank_matrix_rows(
        [{"tf": "5m", "week_ret": 0.9, "robust_ok": False, "skip": False}]
    ) is None


def test_truncated_holdout_skipped():
    from moonshot.hunt.scoring import _holdout_span_ok

    assert _holdout_span_ok(days=float(WEEK_DAYS)) is True
    assert _holdout_span_ok(days=float(WEEK_DAYS) - 1.0) is True
    assert _holdout_span_ok(days=float(WEEK_DAYS) - 1.5) is False
