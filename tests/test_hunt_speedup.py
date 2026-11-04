"""Faster-hunt helpers: search mix, banlist, dual-week, research equity, rescue streak."""

from __future__ import annotations

from dataclasses import replace

from moonshot.config import Config
from moonshot.hunt.constants import (
    HUNT_CHAMP_TRAIL_PASSES,
    HUNT_CHAMP_TRAIL_PP,
    _hunt_soft_workers,
)
from moonshot.hunt.scoring import (
    _ban_key,
    _banlist_add,
    _banlist_set,
    _dual_week_ok,
    _filter_banned,
    _hunt_search_mode,
    _should_ban_week,
)


def test_search_mode_mix_with_champion() -> None:
    modes = [_hunt_search_mode(i, 0, True, has_champion=True) for i in range(20)]
    n = len(modes)
    assert modes.count("breakout") / n == 0.1  # every 10th
    assert 0.15 <= modes.count("explore") / n <= 0.25
    assert modes.count("polish") / n >= 0.65


def test_search_mode_stuck_prefers_breakout_or_explore() -> None:
    assert _hunt_search_mode(0, 10, True, has_champion=True) == "breakout"
    assert _hunt_search_mode(1, 10, True, has_champion=False) in ("explore", "breakout")


def test_banlist_add_filter_and_cap() -> None:
    cfg = Config(adx_enter=18.0, leverage=5)
    other = replace(cfg, adx_enter=22.0)
    state: dict = {}
    _banlist_add(state, cfg, week_ret=-0.5, reason="week<-20%")
    banned = _banlist_set(state)
    assert _ban_key(cfg) in banned
    kept, n = _filter_banned([cfg, other], banned)
    assert n == 1
    assert kept == [other]


def test_should_ban_week() -> None:
    ok, _ = _should_ban_week({"ret": -0.05})
    assert ok is False
    bad, why = _should_ban_week({"ret": -0.25})
    assert bad is True
    assert "week" in why
    kill, _ = _should_ban_week({"ret": 0.1}, kill_illusion=True)
    assert kill is True


def test_dual_week_gate() -> None:
    assert _dual_week_ok({"ret": 1.0, "prior_ret": 0.0}) is True
    assert _dual_week_ok({"ret": 0.30, "prior_ret": 0.30}) is True
    assert _dual_week_ok({"ret": 0.30, "prior_ret": -0.05}) is False
    assert _dual_week_ok({"ret": 0.10, "prior_ret": 0.10}) is False
    assert _dual_week_ok(None) is False


def test_research_starting_equity_separate_from_paper(monkeypatch) -> None:
    monkeypatch.setenv("STARTING_EQUITY", "1000.0")
    monkeypatch.setenv("RESEARCH_STARTING_EQUITY", "50.0")
    c = Config.from_env()
    assert abs(c.starting_equity - 1000.0) < 1e-9
    assert abs(c.research_starting_equity - 50.0) < 1e-9


def test_soft_workers_clamped(monkeypatch) -> None:
    monkeypatch.setenv("HUNT_SOFT_WORKERS", "99")
    assert _hunt_soft_workers() == 4
    monkeypatch.setenv("HUNT_SOFT_WORKERS", "0")
    assert _hunt_soft_workers() == 1


def test_champ_trail_streak_logic() -> None:
    """Rescue eligibility mirrors improve_once policy-3 thresholds."""
    champ, live = 0.52, -0.47
    trail = champ - live
    assert trail >= HUNT_CHAMP_TRAIL_PP
    streak = 0
    for _ in range(HUNT_CHAMP_TRAIL_PASSES):
        if trail >= HUNT_CHAMP_TRAIL_PP:
            streak += 1
        else:
            streak = 0
    assert streak >= HUNT_CHAMP_TRAIL_PASSES
