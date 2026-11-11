"""TF×lev matrix is always the full allow-list — no TF slim path."""

from __future__ import annotations

from moonshot.hunt.constants import HUNT_MATRIX_TFS, HUNT_STUCK_HELD
from moonshot.hunt.scoring import _matrix_tfs_for_pass


def _assert_full(tfs: tuple[str, ...]) -> None:
    assert tfs == tuple(HUNT_MATRIX_TFS)
    assert tfs == ("5m", "15m", "1h", "4h")


def test_stuck_gets_full_matrix_even_when_live_week_strong():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=int(HUNT_STUCK_HELD),
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.30,
    )
    _assert_full(tfs)


def test_explore_gets_full_matrix_when_live_strong():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="explore",
        skip_heavy_search=False,
        live_week_ret=0.25,
    )
    _assert_full(tfs)


def test_breakout_gets_full_matrix_when_live_strong():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=1,
        search_mode="breakout",
        skip_heavy_search=False,
        live_week_ret=0.40,
    )
    _assert_full(tfs)


def test_polish_strong_live_still_full():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.20,
    )
    _assert_full(tfs)


def test_polish_weak_live_full_matrix():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.05,
    )
    _assert_full(tfs)


def test_exhausted_forces_full_matrix():
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=False,
        live_week_ret=0.50,
        hunt_exhausted=True,
    )
    _assert_full(tfs)


def test_skip_heavy_never_slims_matrix():
    slim = _matrix_tfs_for_pass(
        base_tf="5m",
        held=0,
        search_mode="polish",
        skip_heavy_search=True,
        live_week_ret=0.10,
    )
    _assert_full(slim)
    full = _matrix_tfs_for_pass(
        base_tf="5m",
        held=int(HUNT_STUCK_HELD) + 2,
        search_mode="polish",
        skip_heavy_search=True,
        live_week_ret=0.10,
    )
    _assert_full(full)


def test_force_slim_kwarg_ignored_always_full():
    """Legacy force_slim must not cut TFs — denser skip is separate."""
    tfs = _matrix_tfs_for_pass(
        base_tf="5m",
        held=int(HUNT_STUCK_HELD) + 5,
        search_mode="breakout",
        skip_heavy_search=False,
        live_week_ret=0.01,
        hunt_exhausted=True,
        force_slim=True,
    )
    _assert_full(tfs)
