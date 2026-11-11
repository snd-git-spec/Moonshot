"""OOM-restart loops must FAIL scoreboard — not carve out on fresh improve_start."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from moonshot.hunt.constants import HUNT_PASS_MAX_HOURS, HUNT_STALE_FINISH_HOURS
from moonshot.hunt.health import hunt_pass_health


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def test_oom_loop_fresh_start_old_improve_fails_no_finish():
    """Fresh started_at + improve_start must NOT hide a frozen last_improve_at."""
    st = {
        "mode": "hunt",
        "busy": True,
        "started_at": _iso(0.1),
        "last_improve_at": _iso(float(HUNT_STALE_FINISH_HOURS) + 1.0),
        "pass_started_at": _iso(0.1),
        "hunt_event": {
            "kind": "improve_start",
            "ts": _iso(0.05),
            "message": "HUNT pass #1904",
        },
        "improve_passes": 1904,
    }
    out = hunt_pass_health(st, process_alive=True, mode="hunt")
    assert out["ok"] is False
    assert any("HUNT_NO_FINISH" in f for f in out["failures"])


def test_busy_stale_heartbeat_fails_pass_stuck():
    """PASS_STUCK = stale pass_heartbeat only (not wall-clock pass age alone)."""
    from moonshot.hunt.constants import HUNT_HEARTBEAT_STALE_MIN

    st = {
        "mode": "hunt",
        "busy": True,
        "last_improve_at": _iso(1.0),  # recent enough for NO_FINISH
        "pass_started_at": _iso(3.0),  # old wall-clock OK if heartbeat fresh
        "pass_heartbeat_at": _iso((float(HUNT_HEARTBEAT_STALE_MIN) + 5) / 60.0),
        "hunt_event": {"kind": "improve_start", "ts": _iso(0.1)},
    }
    out = hunt_pass_health(st, process_alive=True, mode="hunt")
    assert out["ok"] is False
    assert any("HUNT_PASS_STUCK" in f for f in out["failures"])


def test_heartbeat_fresh_no_pass_stuck_even_if_pass_old():
    st = {
        "mode": "hunt",
        "busy": True,
        "last_improve_at": _iso(0.5),
        "pass_started_at": _iso(float(HUNT_PASS_MAX_HOURS) + 2.0),
        "pass_heartbeat_at": _iso(0.05),  # fresh heartbeat
        "hunt_event": {"kind": "improve_start", "ts": _iso(0.05)},
    }
    out = hunt_pass_health(st, process_alive=True, mode="hunt")
    assert not any("HUNT_PASS_STUCK" in f for f in out["failures"])


def test_terminal_hold_grace_skips_stale():
    st = {
        "mode": "hunt",
        "busy": False,
        "last_improve_at": _iso(float(HUNT_STALE_FINISH_HOURS) + 2.0),
        "hunt_event": {"kind": "hold", "ts": _iso(0.05)},
    }
    out = hunt_pass_health(st, process_alive=True, mode="hunt")
    assert out["ok"] is True
    assert out["failures"] == []


def test_scoreboard_check_wires_no_finish(monkeypatch):
    import scripts.scoreboard_check as sc

    st = {
        "mode": "hunt",
        "busy": True,
        "started_at": _iso(0.1),
        "last_improve_at": _iso(float(HUNT_STALE_FINISH_HOURS) + 2.0),
        "pass_started_at": _iso(0.1),
        "hunt_event": {"kind": "improve_start", "ts": _iso(0.05)},
        "champion": {},
        "live": {},
        "history": [],
        "improve_passes": 10,
        "generation": 10,
    }
    monkeypatch.setattr(sc, "_load_autopilot_state", lambda: st)
    monkeypatch.setattr(sc, "_env_book_fingerprint", lambda: ("momentum_15", "5m", "10"))
    monkeypatch.setattr(
        sc,
        "_autopilot_process",
        lambda: {"pid": 1, "alive": True, "pid_path": "", "source": "test"},
    )
    monkeypatch.setattr(sc, "_paper_closes", lambda n=10: [])
    monkeypatch.setattr(sc, "_paper_bleed", lambda: {"bleeding": False})
    monkeypatch.setattr(sc, "_risk_halt", lambda: None)
    monkeypatch.setattr(sc, "ENV_PATH", type("P", (), {"exists": lambda self: False})())

    report = sc.check()
    assert report["ok"] is False
    assert any("HUNT_NO_FINISH" in f for f in report["failures"])
