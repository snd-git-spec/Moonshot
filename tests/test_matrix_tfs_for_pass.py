"""TF×lev matrix must not slim to Live TF while hunt is stuck."""

from __future__ import annotations

from moonshot.hunt.constants import HUNT_MATRIX_TFS, HUNT_STUCK_HELD
from moonshot.hunt.scoring import _matrix_tfs_for_pass


def test_stuck_gets_full_matrix_even_when_live_week_strong():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=int(HUNT_STUCK_HELD),
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.30,
    )
    assert tfs == tuple(HUNT_MATRIX_TFS)
    assert "15m" in tfs and "1h" in tfs and "4h" in tfs


def test_explore_gets_full_matrix_when_live_strong():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="explore",
        skip_heavy_search=False,
        live_week_ret=0.25,
    )
    assert tfs == tuple(HUNT_MATRIX_TFS)


def test_breakout_gets_full_matrix_when_live_strong():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=1,
        search_mode="breakout",
        skip_heavy_search=False,
        live_week_ret=0.40,
    )
    assert tfs == tuple(HUNT_MATRIX_TFS)


def test_polish_strong_live_may_slim():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.20,
    )
    assert tfs == ("5m",)


def test_polish_weak_live_full_matrix():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.05,
    )
    assert tfs == tuple(HUNT_MATRIX_TFS)


def test_exhausted_forces_full_matrix():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.50,
        hunt_exhausted=True,
    )
    assert tfs == tuple(HUNT_MATRIX_TFS)


def test_skip_heavy_slims_only_when_not_stuck():
    slim = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=True,
        live_week_ret=0.10,
    )
    assert slim == ("5m",)
    full = _matrix_tfs_for_pass(
        base_tf="5m",
        held=int(HUNT_STUCK_HELD) + 2,
        search_mode="polish",
        skip_heavy_search=True,
        live_week_ret=0.10,
    )
    assert full == tuple(HUNT_MATRIX_TFS)


def test_force_slim_overrides_stuck_full_matrix():
    """OOM / pass-stuck recovery: Live TF only — stuck must not force FULL TFs."""
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=int(HUNT_STUCK_HELD) + 5,
        search_mode="breakout",
        skip_heavy_search=False,
        live_week_ret=0.01,
        hunt_exhausted=True,
        force_slim=True,
    )
    assert tfs == ("5m",)
