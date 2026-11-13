"""Stage C dual filter may beat Live into _consider without +25% absolute."""

from __future__ import annotations

from moonshot.hunt.scoring import RESEARCH_MIN_WEEK_EDGE, _stage_c_dual_allows


def test_stage_c_allows_beat_live_without_hard_floor():
    live = -0.017
    # Just above Live + edge, prior non-negative — must not dual-reject.
    week = {"ret": live + RESEARCH_MIN_WEEK_EDGE + 0.001, "prior_ret": 0.01}
    assert _stage_c_dual_allows(week, live_week_ret=live) is True


def test_stage_c_rejects_below_live():
    live = -0.017
    week = {"ret": live - 0.01, "prior_ret": 0.0}
    assert _stage_c_dual_allows(week, live_week_ret=live) is False


def test_stage_c_rejects_beat_live_with_red_prior():
    live = -0.017
    week = {"ret": live + 0.05, "prior_ret": -0.1}
    assert _stage_c_dual_allows(week, live_week_ret=live) is False


def test_stage_c_allows_hard_floor_unchanged():
    week = {"ret": 0.30, "prior_ret": 0.30}
    assert _stage_c_dual_allows(week, live_week_ret=-0.02) is True


def test_stage_c_allows_dual_ok_hundred():
    week = {"ret": 1.0, "prior_ret": 0.0}
    assert _stage_c_dual_allows(week, live_week_ret=-0.02) is True
