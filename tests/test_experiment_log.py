"""A.6 — experiment_log writes + cumulative trial_counter."""

from __future__ import annotations

from pathlib import Path

from moonshot.experiment_log import get_runs, log_run, trial_counter_count


def test_experiment_log_writes(tmp_path: Path):
    db = tmp_path / "experiments.db"
    log_run(
        run_id="r1",
        kind="hunt",
        config_hash="abc",
        data_hash="d1",
        metrics={"sharpe": 1.2},
        db_path=db,
    )
    rows = get_runs({"kind": "hunt"}, db_path=db)
    assert len(rows) == 1
    assert rows[0]["config_hash"] == "abc"
    assert rows[0]["data_hash"] == "d1"


def test_trial_counter_cumulative(tmp_path: Path):
    db = tmp_path / "experiments.db"
    log_run(run_id="a", kind="backtest", config_hash="h1", db_path=db)
    log_run(run_id="b", kind="backtest", config_hash="h1", db_path=db)
    log_run(run_id="c", kind="hunt", config_hash="h2", db_path=db)
    assert trial_counter_count(db_path=db) == 2
    assert len(get_runs({"config_hash": "h1"}, db_path=db)) == 2
