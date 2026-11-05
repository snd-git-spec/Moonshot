"""A.3a — fold boundaries, Sortino lock, lower-quantile primary metric."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from moonshot.hunt.fold_runner import (
    ANNUALIZATION,
    ANNUALIZATION_FACTOR,
    N_FOLDS,
    SORTINO_TARGET,
    fold_boundaries,
    lower_quantile_metric,
    purge_embargo_bars,
    sortino_definition_meta,
    sortino_from_daily_returns,
    write_fold_artifacts,
)


def test_fold_boundaries():
    purge, embargo = purge_embargo_bars(20, 10)
    assert purge == 30
    assert embargo == 10
    folds = fold_boundaries(1000, N_FOLDS, purge_bars=purge, embargo_bars=embargo)
    assert len(folds) == N_FOLDS
    # Non-overlapping eval spans
    for i in range(len(folds) - 1):
        assert folds[i]["end"] <= folds[i + 1]["start"]
        gap = folds[i + 1]["start"] - folds[i]["end"]
        assert gap == purge + embargo
    assert folds[0]["start"] == 0
    assert folds[-1]["end"] == 1000
    widths = [f["end"] - f["start"] for f in folds]
    assert min(widths) >= 1
    # Non-last folds share equal length
    assert len(set(widths[:-1])) == 1


def test_lower_quantile_metric():
    vals = [1.0, 2.0, 3.0, 4.0]
    q = lower_quantile_metric(vals, 0.25)
    assert abs(q - float(np.quantile(vals, 0.25))) < 1e-12
    assert math.isnan(lower_quantile_metric([]))
    assert math.isnan(lower_quantile_metric([float("nan")]))


def test_sortino_definition_reproducible(tmp_path: Path):
    meta = sortino_definition_meta()
    assert meta["target"] == SORTINO_TARGET == 0.0
    assert meta["annualization"] == ANNUALIZATION
    assert abs(meta["annualization_factor"] - ANNUALIZATION_FACTOR) < 1e-12

    rng = np.random.default_rng(42)
    rets = rng.normal(0.001, 0.01, size=200)
    a = sortino_from_daily_returns(rets)
    b = sortino_from_daily_returns(rets)
    assert math.isfinite(a) and a == b

    # Hand-check formula
    excess = rets - 0.0
    dd = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    expected = float(np.mean(excess) / dd * math.sqrt(365))
    assert abs(a - expected) < 1e-12

    folds = fold_boundaries(400, 4, purge_bars=5, embargo_bars=5)
    paths = write_fold_artifacts(
        tmp_path / "hunt_test",
        folds=folds,
        fold_sortinos=[a, a * 0.9, a * 1.1, a * 0.8],
        fold_sharpes=[1.0, 0.5, 1.2, 0.8],
    )
    metrics = json.loads(paths["metrics"].read_text())
    assert metrics["annualization"] == ANNUALIZATION
    assert metrics["target"] == 0.0
    assert "p25_fold_sortino" in metrics
    assert paths["folds"].exists()
