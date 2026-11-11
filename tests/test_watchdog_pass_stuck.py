"""Watchdog must CRITICAL + force_matrix_slim on stuck / no-finish passes."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from moonshot.hunt.constants import HUNT_STALE_FINISH_HOURS


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _patch_state_io(monkeypatch, disk: dict):
    """Patch constants load/save used by flock _update_autopilot_state."""
    import moonshot.hunt.constants as const
    from moonshot.hunt import autopilot as ap

    def _load():
        return dict(disk)

    def _save(st):
        disk.clear()
        disk.update(st)

    def _update(mutator):
        st = _load()
        mutator(st)
        _save(st)
        return st

    monkeypatch.setattr(const, "_load_autopilot_state", _load)
    monkeypatch.setattr(const, "_save_autopilot_state", _save)
    monkeypatch.setattr(const, "_update_autopilot_state", _update)
    monkeypatch.setattr(ap, "_load_autopilot_state", _load)
    monkeypatch.setattr(ap, "_save_autopilot_state", _save)
    monkeypatch.setattr(ap, "_update_autopilot_state", _update)
    return ap


def test_watchdog_sets_force_slim_on_no_finish(tmp_path, monkeypatch):
    disk = {
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
    ap = _patch_state_io(monkeypatch, disk)
    monkeypatch.setattr(ap, "_env_book_fingerprint", lambda: ("momentum_15", "5m", "10"))
    monkeypatch.setattr(ap, "_live_feed", lambda *a, **k: None)
    monkeypatch.setattr(ap, "recent_oom_hint", lambda: False)

    out = ap._run_scoreboard_watchdog()
    assert out["critical"] is True
    assert disk["force_matrix_slim"] is True
    assert disk["hunt_recover_reason"] in ("no_finish", "pass_stuck", "oom")
    assert any("HUNT_NO_FINISH" in a for a in out["alerts"])


def test_watchdog_oom_hint_sets_reason_oom(monkeypatch):
    disk = {
        "mode": "hunt",
        "busy": True,
        "last_improve_at": _iso(float(HUNT_STALE_FINISH_HOURS) + 2.0),
        "pass_started_at": _iso(0.1),
        "hunt_event": {"kind": "improve_start", "ts": _iso(0.05)},
        "champion": {},
        "live_deployed": {},
        "history": [],
    }
    ap = _patch_state_io(monkeypatch, disk)
    monkeypatch.setattr(ap, "_env_book_fingerprint", lambda: ("", "", ""))
    monkeypatch.setattr(ap, "_live_feed", lambda *a, **k: None)
    monkeypatch.setattr(ap, "recent_oom_hint", lambda: True)

    out = ap._run_scoreboard_watchdog()
    assert out["critical"] is True
    assert disk["hunt_recover_reason"] == "oom"
    assert disk["force_matrix_slim"] is True


def test_watchdog_save_preserves_pass_started_at(monkeypatch):
    """Regression: watchdog must not clobber improve_once pass lifecycle fields."""
    disk = {
        "mode": "hunt",
        "busy": True,
        "last_improve_at": _iso(float(HUNT_STALE_FINISH_HOURS) + 2.0),
        "pass_started_at": "2026-11-11T11:32:11+00:00",
        "pass_heartbeat_at": "2026-11-11T11:32:40+00:00",
        "hunt_event": {"kind": "improve_start", "ts": _iso(0.05)},
        "champion": {},
        "live_deployed": {},
        "history": [],
        "generation": 1904,
    }
    ap = _patch_state_io(monkeypatch, disk)
    monkeypatch.setattr(ap, "_env_book_fingerprint", lambda: ("", "", ""))
    monkeypatch.setattr(ap, "_live_feed", lambda *a, **k: None)
    monkeypatch.setattr(ap, "recent_oom_hint", lambda: False)

    ap._run_scoreboard_watchdog()
    assert disk["pass_started_at"] == "2026-11-11T11:32:11+00:00"
    assert disk["pass_heartbeat_at"] == "2026-11-11T11:32:40+00:00"
    assert disk["force_matrix_slim"] is True
