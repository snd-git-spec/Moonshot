"""Period-robust promote floors — no red prior / OOS crowning."""

from moonshot.hunt.constants import HUNT_TARGETS
from moonshot.hunt.scoring import _period_robust_ok


def test_period_robust_rejects_red_oos(monkeypatch):
    monkeypatch.setitem(HUNT_TARGETS, "require_period_robust", True)
    monkeypatch.setitem(HUNT_TARGETS, "oos_floor", 0.0)
    monkeypatch.setitem(HUNT_TARGETS, "prior_week_floor", 0.0)
    ok, why = _period_robust_ok({"prior_ret": 0.1, "ret": 0.2}, -0.45, 0.05)
    assert ok is False
    assert "oos" in why


def test_period_robust_rejects_red_prior(monkeypatch):
    monkeypatch.setitem(HUNT_TARGETS, "require_period_robust", True)
    ok, why = _period_robust_ok({"prior_ret": -0.26, "ret": 0.20}, 0.05, 0.02)
    assert ok is False
    assert "prior" in why


def test_period_robust_accepts_green_windows(monkeypatch):
    monkeypatch.setitem(HUNT_TARGETS, "require_period_robust", True)
    monkeypatch.setitem(HUNT_TARGETS, "oos_floor", 0.0)
    monkeypatch.setitem(HUNT_TARGETS, "prior_week_floor", 0.0)
    monkeypatch.setitem(HUNT_TARGETS, "soft_promote_floor", 0.0)
    ok, why = _period_robust_ok({"prior_ret": 0.05, "ret": 0.30}, 0.12, 0.08)
    assert ok is True
    assert why == "period_robust"


def test_period_robust_can_disable(monkeypatch):
    monkeypatch.setitem(HUNT_TARGETS, "require_period_robust", False)
    ok, why = _period_robust_ok({"prior_ret": -0.5}, -0.9, -0.9)
    assert ok is True
    assert why == "robust_off"
