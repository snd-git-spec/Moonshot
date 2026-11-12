"""Stuck pending_champion_rescue must clear when already on champion book."""

from __future__ import annotations

from moonshot.config import Config
from moonshot.hunt.scoring import _cfg_fingerprint


def test_same_fingerprint_means_already_on_champ():
    a = Config(strategy="momentum_15", timeframe="5m", leverage=10, adx_enter=22.0)
    b = Config(strategy="momentum_15", timeframe="5m", leverage=10, adx_enter=22.0)
    assert _cfg_fingerprint(a) == _cfg_fingerprint(b)


def test_different_adx_is_not_same_book():
    a = Config(strategy="momentum_15", timeframe="5m", leverage=10, adx_enter=22.0)
    b = Config(strategy="momentum_15", timeframe="5m", leverage=10, adx_enter=20.0)
    assert _cfg_fingerprint(a) != _cfg_fingerprint(b)


def test_watchdog_clears_stuck_pending_when_env_matches_champ(tmp_path, monkeypatch):
    import moonshot.hunt.constants as const
    from moonshot.hunt import autopilot as ap

    from moonshot.hunt.constants import _utc_now

    # Fresh finish — avoid HUNT_NO_FINISH critical from stale last_improve_at.
    now = _utc_now().isoformat()
    disk = {
        "mode": "hunt",
        "busy": False,
        "last_improve_at": now,
        "pending_champion_rescue": True,
        "champ_trail_streak": 3,
        "champion": {
            "knobs": {"strategy": "momentum_15", "timeframe": "5m", "leverage": 10},
            "week_ret_live": 0.25,
        },
        "live_deployed": {"week_ret": 0.25},
        "history": [],
        "hunt_event": {"kind": "hold", "ts": now},
    }

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
    monkeypatch.setattr(ap, "_env_book_fingerprint", lambda: ("momentum_15", "5m", "10"))
    monkeypatch.setattr(ap, "_live_feed", lambda *a, **k: None)
    monkeypatch.setattr(ap, "recent_oom_hint", lambda: False)

    out = ap._run_scoreboard_watchdog()
    assert disk["pending_champion_rescue"] is False
    assert disk["champ_trail_streak"] == 0
    assert out["critical"] is False
