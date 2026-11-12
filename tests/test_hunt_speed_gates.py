"""Gates for hunt-speed package: hard overlap, soft diversity, stuck funnel."""

from __future__ import annotations

from moonshot.hunt.constants import (
    HUNT_HALVING_A_KEEP,
    HUNT_MATRIX_SOFT_CAP,
    HUNT_STUCK_SOFT_BUDGET,
)
from moonshot.hunt.scoring import (
    _matrix_hard_top_keys,
    _matrix_hard_top_overlap,
    _matrix_soft_pick,
)


def _row(tf: str, lev: int, week: float, prior: float, *, dense: bool = True) -> dict:
    return {
        "tf": tf,
        "leverage": lev,
        "strategy": "momentum_15",
        "week_ret": week,
        "prior_ret": prior,
        "dense_ok": dense,
        "skip": False,
    }


def test_soft_cap_and_stuck_constants():
    assert int(HUNT_MATRIX_SOFT_CAP) == 5
    assert int(HUNT_HALVING_A_KEEP) == 16
    assert int(HUNT_STUCK_SOFT_BUDGET) == 14


def test_hard_top_overlap_gate_passes_on_live_like_matrix():
    # Mirrors cloud matrix snapshot used for Phase B gate (overlap ≥ 4/5).
    rows = [
        _row("5m", 10, 0.337, -0.129),
        _row("5m", 7, 0.316, -0.090),
        _row("5m", 5, 0.243, -0.064),
        _row("5m", 15, 0.161, -0.175),
        _row("5m", 25, -0.017, -0.328),
        _row("4h", 5, -0.071, -0.052),
        _row("4h", 7, -0.071, -0.052),
        _row("4h", 10, -0.079, -0.052),
        _row("1h", 5, -0.105, 0.018),
        _row("1h", 7, -0.111, 0.017),
        _row("1h", 10, -0.111, 0.017),
        _row("15m", 5, -0.165, -0.147),
        _row("15m", 7, -0.195, -0.169),
        _row("15m", 10, -0.214, -0.181),
    ]
    overlap = _matrix_hard_top_overlap(rows, n=5)
    assert overlap >= 4
    week_keys = _matrix_hard_top_keys(rows, n=5, dual_composite=False)
    dual_keys = _matrix_hard_top_keys(rows, n=5, dual_composite=True)
    assert ("5m", 25) in week_keys  # weak prior still in week top-5
    assert ("1h", 5) in dual_keys  # prior helps 1h into dual top-5


def test_soft_pick_prefers_tf_diversity():
    rows = [
        _row("5m", 10, 0.40, 0.0),
        _row("5m", 7, 0.35, 0.0),
        _row("5m", 5, 0.30, 0.0),
        _row("5m", 15, 0.25, 0.0),
        _row("5m", 25, 0.20, 0.0),
        _row("4h", 5, 0.10, 0.0),
        _row("1h", 5, 0.05, 0.0),
        _row("15m", 5, 0.01, 0.0),
    ]
    picked = _matrix_soft_pick(rows, cap=5)
    tfs = {r["tf"] for r in picked}
    assert len(picked) == 5
    # Best-per-TF first → all four TFs present, then fill with next 5m.
    assert tfs == {"5m", "4h", "1h", "15m"}
    assert sum(1 for r in picked if r["tf"] == "5m") == 2


def test_stuck_fat_funnel_is_churn_proxy():
    """Cloud history proxy: soft_runs≥18 escape rate lower than soft≤14 → thin.

    Fat escape rate must not be >1.5× thin; otherwise do not thin.
    """
    # Counts from cloud history (explore/breakout) at ship time.
    fat_n, fat_esc = 171, 3
    thin_n, thin_esc = 40, 3
    fat_rate = fat_esc / fat_n
    thin_rate = thin_esc / thin_n
    assert fat_rate <= thin_rate * 1.5


def test_book_leverage_bakeoff_locks_strategy_and_tf():
    from moonshot.hunt.scoring import _book_leverage_bakeoff

    rows = [
        _row("5m", 10, 0.40, 0.0),
        _row("5m", 15, 0.44, 0.0),
        _row("1h", 10, 0.90, 0.0),  # different TF — must not win the 10× slot
        {
            **_row("5m", 5, 0.20, 0.0),
            "strategy": "trend_mtf",  # different strategy — excluded
        },
    ]
    out = _book_leverage_bakeoff(rows, strategy="momentum_15", timeframe="5m")
    # trend_mtf@5m/5× excluded; 1h@10× must not steal the 10× slot
    assert [r["leverage"] for r in out] == [10, 15]
    assert all(r["strategy"] == "momentum_15" and r["timeframe"] == "5m" for r in out)
    assert float(out[0]["week_ret"]) == 0.40
    assert float(out[1]["week_ret"]) == 0.44
