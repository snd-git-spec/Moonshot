"""Phase 6.4 lite — unique strat/tf/lev coverage counts."""

from __future__ import annotations

from moonshot.config import Config
from moonshot.obs import coverage_unique_counts, hunt_funnel_metrics


def test_coverage_unique_counts():
    cfgs = [
        Config(strategy="momentum_15", timeframe="5m", leverage=10),
        Config(strategy="momentum_15", timeframe="1h", leverage=5),
        Config(strategy="trend_mtf", timeframe="1h", leverage=5),
    ]
    rows = [
        {"strategy": "combo_or", "tf": "15m", "leverage": 7},
        {"skip": True, "strategy": "ignored", "tf": "4h", "leverage": 25},
    ]
    out = coverage_unique_counts(cfgs, matrix_rows=rows)
    assert out["n_unique_strat"] == 3  # mom, trend, combo
    assert out["n_unique_tf"] == 3  # 5m, 1h, 15m
    assert out["n_unique_lev"] == 3  # 10, 5, 7
    assert out["n_cand"] >= 3


def test_funnel_includes_coverage_fields():
    f = hunt_funnel_metrics(
        scout_in=10,
        scout_keep=5,
        confirm_b=3,
        confirm_hits=1,
        confirm_rejects={},
        matrix_rows=[],
        finalist_ok=False,
        gates_pass=False,
        promote=False,
        n_unique_strat=2,
        n_unique_tf=3,
        n_unique_lev=4,
        n_cand=40,
    )
    assert f["n_unique_strat"] == 2
    assert f["n_unique_tf"] == 3
    assert f["primary_metric"] == "p25_fold_sortino"
