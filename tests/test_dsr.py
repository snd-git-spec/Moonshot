"""A.3b — DSR monotone in noise; cumulative trial_counter."""

from __future__ import annotations

from pathlib import Path

from moonshot.hunt.dsr import (
    deflated_sharpe_ratio,
    dsr_passes,
    record_trial,
    trial_count,
)


def test_dsr_monotone_in_noise():
    # Holding T and trials fixed, higher observed SR → higher DSR
    low = deflated_sharpe_ratio(0.5, n_obs=10, n_trials=2)
    mid = deflated_sharpe_ratio(1.0, n_obs=10, n_trials=2)
    high = deflated_sharpe_ratio(1.5, n_obs=10, n_trials=2)
    assert 0.0 < low < mid < high < 1.0
    # More trials (selection noise / multiple-testing) deflates DSR for same SR
    few = deflated_sharpe_ratio(1.5, n_obs=40, n_trials=2)
    many = deflated_sharpe_ratio(1.5, n_obs=40, n_trials=50)
    assert few > many
    assert dsr_passes(0.6)
    assert not dsr_passes(0.0)


def test_trial_count_cumulative(tmp_path: Path):
    db = tmp_path / "trials.db"
    assert trial_count(db_path=db) == 0
    n1 = record_trial("abc", "run1", first_seen_ts="2026-01-01T00:00:00Z", db_path=db)
    assert n1 == 1
    n2 = record_trial("abc", "run2", first_seen_ts="2026-01-02T00:00:00Z", db_path=db)
    assert n2 == 1  # same hash
    n3 = record_trial("def", "run3", first_seen_ts="2026-01-03T00:00:00Z", db_path=db)
    assert n3 == 2
    assert trial_count(db_path=db) == 2
