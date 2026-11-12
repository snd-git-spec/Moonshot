"""Fold eval wrapper — p25 Sortino primary for hunt ranking."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from moonshot.hunt.fold_eval import eval_cfg_folds
from moonshot.hunt.fold_runner import lower_quantile_metric, sortino_from_daily_returns


def test_lower_quantile_matches_p25_helper():
    vals = [0.1, 0.5, 1.0, 2.0]
    assert abs(lower_quantile_metric(vals) - float(np.quantile(vals, 0.25))) < 1e-12


def test_eval_cfg_folds_empty_data():
    from moonshot.config import Config

    cfg = Config(strategy="momentum_15", timeframe="5m", leverage=10)
    out = eval_cfg_folds({}, cfg, end=pd.Timestamp("2026-11-01", tz="UTC"), smoke=True)
    assert out["ok"] is False
    assert math.isnan(out["p25_sortino"]) or out["why"]


def test_sortino_from_synthetic_returns():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.001, 0.01, size=100)
    s = sortino_from_daily_returns(rets)
    assert math.isfinite(s)


def test_rank_key_orders_fold_before_week():
    from moonshot.backtest import BacktestResult
    from moonshot.hunt.scoring import _rank_key

    def _br(ret=0.0):
        return BacktestResult(
            pyramiding=False,
            partial=False,
            leverage=10,
            final_equity=100.0 * (1.0 + ret),
            total_return=ret,
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            expectancy_r=0.0,
            exp_ci=(0.0, 0.0),
            profit_factor=1.0,
            max_drawdown=0.1,
            liquidations=0,
            trades=10,
            win_rate=0.5,
            top5_pct=0.0,
            avg_adds=0.0,
            halted="",
            equity_curve=pd.Series([100.0]),
            trade_list=[],
        )

    # High week but low fold loses to low week high fold
    low_fold = _rank_key(
        _br(0.5),
        _br(0.5),
        _br(0.5),
        week={"ret": 0.9},
        fold={"p25_sortino": -1.0},
        cost_ok=True,
        roi=0.5,
    )
    high_fold = _rank_key(
        _br(0.1),
        _br(0.1),
        _br(0.1),
        week={"ret": 0.0},
        fold={"p25_sortino": 1.5},
        cost_ok=True,
        roi=0.1,
    )
    assert high_fold > low_fold


def test_cost_ok_sorts_above_cost_fail():
    from moonshot.backtest import BacktestResult
    from moonshot.hunt.scoring import _rank_key

    def _br(ret=0.0):
        return BacktestResult(
            pyramiding=False,
            partial=False,
            leverage=10,
            final_equity=100.0 * (1.0 + ret),
            total_return=ret,
            cagr=0.0,
            sharpe=0.0,
            sortino=0.0,
            expectancy_r=0.0,
            exp_ci=(0.0, 0.0),
            profit_factor=1.0,
            max_drawdown=0.1,
            liquidations=0,
            trades=10,
            win_rate=0.5,
            top5_pct=0.0,
            avg_adds=0.0,
            halted="",
            equity_curve=pd.Series([100.0]),
            trade_list=[],
        )

    pass_cost = _rank_key(
        _br(0.1),
        _br(0.1),
        fold={"p25_sortino": 0.5},
        cost_ok=True,
        roi=0.1,
    )
    fail_cost = _rank_key(
        _br(0.9),
        _br(0.9),
        fold={"p25_sortino": 0.5},
        cost_ok=False,
        roi=0.9,
    )
    assert pass_cost > fail_cost


def test_should_ban_candidate_gate_fails():
    from moonshot.hunt.scoring import _should_ban_candidate

    ok, why = _should_ban_candidate(cost_ok=False)
    assert ok and "cost" in why
    ok, why = _should_ban_candidate(dsr_fail=True)
    assert ok and "dsr" in why
    ok, why = _should_ban_candidate(sealed_fail=True)
    assert ok and "sealed" in why
    ok, why = _should_ban_candidate(
        fold={"p25_sortino": -1.0, "ok": False, "why": "p25 -1 < floor"}
    )
    assert ok and "fold" in why


def test_champion_raise_fold_primary_not_week():
    from moonshot.hunt.scoring import _raise_champion_peak, _resolve_champion_peak

    state: dict = {}
    _raise_champion_peak(
        state,
        knobs={"strategy": "momentum_15", "timeframe": "5m", "leverage": 5},
        week_ret=0.9,
        score=50,
        week={"ret": 0.9},
        ts="2026-01-01T00:00:00Z",
        generation=1,
        p25_sortino=0.2,
        roi=0.1,
        cost_ok=True,
    )
    # Higher week but lower p25 must not replace
    out = _raise_champion_peak(
        state,
        knobs={"strategy": "momentum_15", "timeframe": "5m", "leverage": 10},
        week_ret=1.5,
        score=90,
        week={"ret": 1.5},
        ts="2026-01-02T00:00:00Z",
        generation=2,
        p25_sortino=0.1,
        roi=0.5,
        cost_ok=True,
    )
    assert float(out["p25_sortino"]) == 0.2
    assert int(out["knobs"]["leverage"]) == 5
    # Higher p25 raises
    out2 = _raise_champion_peak(
        state,
        knobs={"strategy": "momentum_15", "timeframe": "5m", "leverage": 8},
        week_ret=0.05,
        score=40,
        week={"ret": 0.05},
        ts="2026-01-03T00:00:00Z",
        generation=3,
        p25_sortino=1.0,
        roi=0.2,
        cost_ok=True,
    )
    assert float(out2["p25_sortino"]) == 1.0
    assert int(_resolve_champion_peak(state)["knobs"]["leverage"]) == 8
    assert out2.get("primary_metric") == "p25_fold_sortino"


def test_stage_a_sort_prefers_p25_not_week():
    """Stage A row shape: (p25, trades, roi, ...) — high week alone loses."""
    # Mimic improve Stage A sort key
    rows = [
        (-0.5, 10, 0.8, "week_winner"),  # great week ROI diagnostic, bad fold
        (1.2, 8, 0.05, "fold_winner"),
    ]
    rows.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    assert rows[0][3] == "fold_winner"
