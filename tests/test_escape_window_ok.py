"""Survive-first escape windows: no week-led OOS-red champions."""

from __future__ import annotations

from moonshot.hunt.scoring import _escape_window_ok, _period_robust_ok


def test_week_led_style_oos_red_blocked():
    """Holdout +2pp with deep OOS/soft red must fail escape floors."""
    ok, why = _escape_window_ok(
        {"ret": 0.32},
        -0.40,
        -0.50,
        {"ret": 0.30},
        -0.41,
        -0.52,
    )
    assert ok is False
    assert "escape_oos" in why or "escape_soft" in why


def test_relative_escape_with_floors_ok():
    # Absolute floors are escape_oos/soft_floor=-0.10; soft must clear that
    # while still beating Live by escape_rel_*_pp.
    ok, why = _escape_window_ok(
        {"ret": 0.20},
        -0.08,
        -0.08,
        {"ret": 0.18},
        -0.20,
        -0.22,
    )
    assert ok is True, why
    assert why == "escape_windows_ok"


def test_relative_escape_below_soft_floor_blocked():
    ok, why = _escape_window_ok(
        {"ret": 0.40},
        -0.10,
        -0.60,
        {"ret": 0.30},
        -0.20,
        -0.70,
    )
    assert ok is False
    assert "escape_soft" in why


def test_relative_must_beat_live_oos():
    ok, why = _escape_window_ok(
        {"ret": 0.40},
        -0.19,
        -0.10,
        {"ret": 0.30},
        -0.20,
        -0.20,
    )
    assert ok is False
    assert "escape_oos" in why


def test_period_robust_still_rejects_oos_red():
    ok, why = _period_robust_ok({"ret": 0.40, "prior_ret": 0.05}, -0.36, -0.57)
    assert ok is False
    assert "oos" in why
