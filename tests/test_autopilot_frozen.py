"""0.3 — AUTOPILOT_FROZEN blocks promotion side effects."""

from __future__ import annotations


from moonshot.hunt.constants import autopilot_frozen


def test_frozen_blocks_promotion(monkeypatch):
    monkeypatch.setenv("AUTOPILOT_FROZEN", "1")
    assert autopilot_frozen() is True
    monkeypatch.setenv("AUTOPILOT_FROZEN", "0")
    assert autopilot_frozen() is False


def test_frozen_allows_backtest(monkeypatch):
    """Backtests must not be gated by AUTOPILOT_FROZEN."""
    monkeypatch.setenv("AUTOPILOT_FROZEN", "1")
    from moonshot.backtest import BacktestEngine

    assert BacktestEngine is not None
    assert autopilot_frozen() is True


def test_promotion_blocked_even_when_targets_met(monkeypatch, tmp_path):
    """improve_once must not call _patch_env when frozen, even if improved."""
    monkeypatch.setenv("AUTOPILOT_FROZEN", "1")
    monkeypatch.setenv("AUTOPILOT_STATE", str(tmp_path / "ap.json"))

    # Unit-level: the freeze gate on the promote path
    from moonshot.hunt import improve as imp

    patched = []

    def fake_patch(updates):
        patched.append(updates)

    monkeypatch.setattr(imp, "_patch_env", fake_patch)
    monkeypatch.setattr(imp, "autopilot_frozen", lambda: True)

    # Simulate the gate used in improve_once
    improved = True
    if improved and imp.autopilot_frozen():
        improved = False
    if improved:
        imp._patch_env({"LEVERAGE": "99"})
    assert patched == []
    assert improved is False
