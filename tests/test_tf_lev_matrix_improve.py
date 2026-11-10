"""Integration tests for TF×lev matrix selection (no network)."""

from __future__ import annotations

from dataclasses import replace

from moonshot.config import Config
from moonshot.hunt.scoring import (
    _dedupe_cfgs_matrix,
    _freeze_risk_matrix,
    _period_robust_ok,
    _rank_matrix_rows,
    _run_tf_lev_matrix_from_rows,
    _select_polish_base,
    _tf_lev_matrix_cfgs,
)


def _base(**kw) -> Config:
    d = dict(
        strategy="momentum_15",
        timeframe="5m",
        leverage=10,
        aggressive_mode=True,
        risk_pct=0.03,
        max_daily_loss_pct=0.32,
        max_peak_drawdown_pct=0.55,
    )
    d.update(kw)
    return Config(**d)


def test_matrix_winner_1h_beats_5m():
    live = _base()
    cells = {
        ("5m", 10): replace(live, timeframe="5m", leverage=10),
        ("1h", 10): replace(live, timeframe="1h", leverage=10),
    }
    rows = [
        {
            "tf": "5m",
            "leverage": 10,
            "week_ret": 0.10,
            "robust_ok": True,
            "skip": False,
            "test_ret": 0.05,
            "prior_ret": 0.02,
            "soft_ret": 0.04,
        },
        {
            "tf": "1h",
            "leverage": 10,
            "week_ret": 0.25,
            "robust_ok": True,
            "skip": False,
            "test_ret": 0.08,
            "prior_ret": 0.03,
            "soft_ret": 0.06,
        },
    ]
    cfg, winner = _run_tf_lev_matrix_from_rows(rows, live_base=live, cell_cfgs=cells)
    assert winner["tf"] == "1h"
    assert cfg is not None
    assert str(cfg.timeframe) == "1h"
    polish = _select_polish_base(
        live_base=live,
        matrix_winner_cfg=cfg,
        best_by_tf_cfg=None,
        champion_cfg=replace(live, timeframe="5m"),
    )
    assert str(polish.timeframe) == "1h"


def test_red_oos_excluded_from_matrix_winner():
    rows = [
        {
            "tf": "5m",
            "leverage": 10,
            "week_ret": 0.50,
            "robust_ok": False,
            "skip": False,
            "test_ret": -0.45,
        },
        {
            "tf": "15m",
            "leverage": 7,
            "week_ret": 0.12,
            "robust_ok": True,
            "skip": False,
            "test_ret": 0.05,
        },
    ]
    w = _rank_matrix_rows(rows)
    assert w["tf"] == "15m"
    assert w["week_ret"] == 0.12


def test_freeze_matrix_evaluates_multiple_levers():
    base = _base(leverage=10)
    seen = set()
    for c in _tf_lev_matrix_cfgs(base):
        out = _freeze_risk_matrix(base, c)
        seen.add(int(out.leverage))
    assert 5 in seen and 15 in seen and 10 in seen
    assert 25 in seen  # 5m cell


def test_denser_seeds_inherit_winner_tf():
    polish = _base(timeframe="1h", leverage=7)
    denser = [
        replace(
            polish,
            strategy="momentum_15",
            leverage=5,
            timeframe=str(polish.timeframe),
        ),
        replace(
            polish,
            strategy="trend_mtf",
            leverage=10,
            timeframe=str(polish.timeframe),
        ),
    ]
    out = _dedupe_cfgs_matrix(polish, denser)
    assert out
    assert all(str(c.timeframe) == "1h" for c in out)


def test_period_robust_blocks_hot_red_week():
    ok, why = _period_robust_ok(
        {"prior_ret": -0.26, "ret": 0.40},
        -0.45,
        -0.49,
    )
    assert ok is False
    assert "oos" in why or "prior" in why
