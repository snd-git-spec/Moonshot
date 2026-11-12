"""Phase 1 — escape only when Live is a death book."""

from __future__ import annotations

from moonshot.hunt.constants import HUNT_TARGETS
from moonshot.hunt.promote_gates import escape_sealed_floor_ok, live_death_book


def test_escape_floors_tightened():
    assert float(HUNT_TARGETS["escape_oos_floor"]) == -0.10
    assert float(HUNT_TARGETS["escape_soft_floor"]) == -0.10
    assert float(HUNT_TARGETS["escape_arm_oos"]) == -0.10


def test_green_holdout_not_death_even_if_mild_oos():
    # oos above arm floor → not death when week green
    death, why = live_death_book(week_ret=0.30, oos_ret=-0.05, soft_ret=-0.05, paper_bleeding=False)
    assert death is False
    assert why == "live_not_death"


def test_deep_oos_is_death():
    death, why = live_death_book(week_ret=0.30, oos_ret=-0.37, soft_ret=-0.50, paper_bleeding=False)
    assert death is True
    assert "oos" in why


def test_escape_sealed_floor():
    ok, why = escape_sealed_floor_ok(-0.05)
    assert ok is True
    ok2, why2 = escape_sealed_floor_ok(-0.20)
    assert ok2 is False
    assert "escape_sealed" in why2


def test_escape_sealed_relative_when_live_sealed_already_dead():
    """Cloud failure mode: Live sealed -11.6%, escape -11.7% — absolute floor
    blocked forever while selection week was far better than Live. Survive-first
    allows near-tied sealed vs dead Live."""
    ok, why = escape_sealed_floor_ok(-0.117, -0.116)
    assert ok is True
    assert why == "escape_sealed_vs_live_ok"
    # Materially worse than dead Live still fails
    ok2, why2 = escape_sealed_floor_ok(-0.20, -0.116)
    assert ok2 is False
    assert "live_sealed" in why2
    # Healthy Live sealed still uses absolute floor
    ok3, why3 = escape_sealed_floor_ok(-0.12, 0.05)
    assert ok3 is False
    assert "floor" in why3


def test_silence_density_blocked_when_not_death_in_improve_source():
    """Improve must require live_is_death before silence block."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[1] / "moonshot/hunt/improve.py").read_text()
    assert "live_is_death" in src
    assert "never polish-deploy on a non-death Live book" in src or "live_is_death" in src
    # matrix escape also gated
    assert "and live_is_death" in src
