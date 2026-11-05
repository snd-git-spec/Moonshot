"""A.3d — artifact layout, two run dirs, reproducibility."""

from __future__ import annotations

from pathlib import Path

from moonshot.hunt.artifacts import (
    REQUIRED_ARTIFACTS,
    artifact_layout_complete,
    metrics_equal,
    new_hunt_run_dir,
    write_artifact_layout,
)


def test_artifact_layout_complete(tmp_path: Path):
    run = write_artifact_layout(
        tmp_path / "hunt_one",
        config={"x": 1},
        metrics={"p25_fold_sortino": 1.23},
        equity_values=[100.0, 101.0],
    )
    assert artifact_layout_complete(run)
    for name in REQUIRED_ARTIFACTS:
        assert (run / name).exists()


def test_two_runs_produce_two_dirs(tmp_path: Path):
    a = new_hunt_run_dir(tmp_path, ts="20260101_000000_000001")
    b = new_hunt_run_dir(tmp_path, ts="20260101_000000_000002")
    assert a != b
    assert a.exists() and b.exists()
    write_artifact_layout(a, metrics={"v": 1.0})
    write_artifact_layout(b, metrics={"v": 2.0})
    assert artifact_layout_complete(a)
    assert artifact_layout_complete(b)


def test_reproducibility(tmp_path: Path):
    metrics = {"p25_fold_sortino": 1.234567890123, "n_folds": 4}
    a = write_artifact_layout(tmp_path / "r1", metrics=metrics)
    b = write_artifact_layout(tmp_path / "r2", metrics=dict(metrics))
    import json

    ma = json.loads((a / "metrics.json").read_text())
    mb = json.loads((b / "metrics.json").read_text())
    assert metrics_equal(ma, mb, tol=1e-9)
