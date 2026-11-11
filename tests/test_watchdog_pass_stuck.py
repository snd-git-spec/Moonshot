"""Watchdog must CRITICAL + force_matrix_slim on stuck / no-finish passes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from moonshot.hunt.constants import HUNT_STALE_FINISH_HOURS


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_watchdog_sets_force_slim_on_no_finish(tmp_path, monkeypatch):
    from moonshot.hunt import autopilot as ap

    state = {
        "mode": "hunt",
        "busy": True,
        "started_at": _iso(0.1),
        "last_improve_at": _iso(float(HUNT_STALE_FINISH_HOURS) + 1.5),
        "pass_started_at": _iso(0.2),
        "hunt_event": {"kind": "improve_start", "ts": _iso(0.05)},
        "champion": {
            "knobs": {"strategy": "momentum_15", "timeframe": "5m", "leverage": 10},
            "week_ret_live": 0.20,
        },
        "live_deployed": {"week_ret": 0.20},
        "history": [],
        "pending_champion_rescue": False,
    }
    saved = {}

    monkeypatch.setattr(ap, "_load_autopilot_state", lambda: state)
    monkeypatch.setattr(ap, "_env_book_fingerprint", lambda: ("momentum_15", "5m", "10"))
    monkeypatch.setattr(ap, "_save_autopilot_state", lambda st: saved.update(st))
    monkeypatch.setattr(ap, "_live_feed", lambda *a, **k: None)
    monkeypatch.setattr(ap, "recent_oom_hint", lambda: False)

    out = ap._run_scoreboard_watchdog()
    assert out["critical"] is True
    assert state["force_matrix_slim"] is True
    assert state["hunt_recover_reason"] in ("no_finish", "pass_stuck", "oom")
    assert any("HUNT_NO_FINISH" in a for a in out["alerts"])


def test_watchdog_oom_hint_sets_reason_oom(monkeypatch):
    from moonshot.hunt import autopilot as ap

    state = {
        "mode": "hunt",
        "busy": True,
        "last_improve_at": _iso(float(HUNT_STALE_FINISH_HOURS) + 2.0),
        "pass_started_at": _iso(0.1),
        "hunt_event": {"kind": "improve_start", "ts": _iso(0.05)},
        "champion": {},
        "live_deployed": {},
        "history": [],
    }
    monkeypatch.setattr(ap, "_load_autopilot_state", lambda: state)
    monkeypatch.setattr(ap, "_env_book_fingerprint", lambda: ("", "", ""))
    monkeypatch.setattr(ap, "_save_autopilot_state", lambda st: None)
    monkeypatch.setattr(ap, "_live_feed", lambda *a, **k: None)
    monkeypatch.setattr(ap, "recent_oom_hint", lambda: True)

    out = ap._run_scoreboard_watchdog()
    assert out["critical"] is True
    assert state["hunt_recover_reason"] == "oom"
    assert state["force_matrix_slim"] is True
