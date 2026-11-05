"""Fold-aware research backtest helpers (A.3a).

Primary objective: 25th-percentile per-fold Sortino (target return = 0).
Annualization: daily returns × sqrt(365) — locked here and in metrics.json.

Diagnostic: 25th-percentile per-fold Sharpe (same annualization).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

N_FOLDS = 4
# Default purge/embargo when lookbacks unknown (bars on the evaluation TF).
DEFAULT_FEATURE_LOOKBACK = 50
DEFAULT_MAX_HOLDING_BARS = 48
SORTINO_TARGET = 0.0
# Locked annualization for daily returns (matches moonshot.backtest).
ANNUALIZATION = "daily_returns * sqrt(365)"
ANNUALIZATION_FACTOR = math.sqrt(365.0)
PRIMARY_QUANTILE = 0.25


def purge_embargo_bars(
    feature_lookback: int = DEFAULT_FEATURE_LOOKBACK,
    max_holding_bars: int = DEFAULT_MAX_HOLDING_BARS,
) -> tuple[int, int]:
    """PURGE_BARS / EMBARGO_BARS from feature lookback + max holding."""
    purge = max(0, int(feature_lookback) + int(max_holding_bars))
    embargo = max(0, int(max_holding_bars))
    return purge, embargo


def fold_boundaries(
    n_bars: int,
    n_folds: int = N_FOLDS,
    *,
    purge_bars: int = 0,
    embargo_bars: int = 0,
) -> list[dict[str, int]]:
    """
    Sequential non-overlapping evaluation folds over [0, n_bars).

    Between fold i eval end and fold i+1 eval start we skip purge+embargo bars
    so label leakage across fold edges is reduced. Fold widths are equal on the
    remaining evaluable span after reserving (n_folds-1)*(purge+embargo) gaps.
    """
    if n_bars < 1:
        raise ValueError("n_bars must be >= 1")
    if n_folds < 1:
        raise ValueError("n_folds must be >= 1")
    gap = max(0, int(purge_bars)) + max(0, int(embargo_bars))
    reserved = gap * (n_folds - 1)
    if n_bars <= reserved + n_folds:
        # Degenerate: split evenly with zero gaps so tests/smoke still run.
        edges = [int(round(i * n_bars / n_folds)) for i in range(n_folds + 1)]
        return [
            {
                "fold": k,
                "start": edges[k],
                "end": edges[k + 1],
                "purge_bars": int(purge_bars),
                "embargo_bars": int(embargo_bars),
            }
            for k in range(n_folds)
            if edges[k + 1] > edges[k]
        ]

    usable = n_bars - reserved
    fold_len = usable // n_folds
    folds: list[dict[str, int]] = []
    cursor = 0
    for k in range(n_folds):
        start = cursor
        # Last fold absorbs remainder up to n_bars (no trailing gap).
        end = n_bars if k == n_folds - 1 else start + fold_len
        folds.append(
            {
                "fold": k,
                "start": int(start),
                "end": int(end),
                "purge_bars": int(purge_bars),
                "embargo_bars": int(embargo_bars),
            }
        )
        cursor = end + gap
    return folds


def sortino_from_daily_returns(
    daily_returns: Sequence[float] | np.ndarray,
    *,
    target: float = SORTINO_TARGET,
) -> float:
    """
    Sortino ratio on daily returns.

    downside deviation = sqrt(mean(min(r - target, 0)^2))
    Sortino = mean(r - target) / dd * sqrt(365)
    """
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    excess = r - float(target)
    downside = np.minimum(excess, 0.0)
    dd = float(np.sqrt(np.mean(downside**2)))
    if dd <= 0.0:
        return float("nan")
    return float(np.mean(excess) / dd * ANNUALIZATION_FACTOR)


def sharpe_from_daily_returns(daily_returns: Sequence[float] | np.ndarray) -> float:
    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return float("nan")
    std = float(r.std(ddof=0))
    if std <= 0.0:
        return float("nan")
    return float(r.mean() / std * ANNUALIZATION_FACTOR)


def lower_quantile_metric(
    fold_metrics: Iterable[float],
    q: float = PRIMARY_QUANTILE,
) -> float:
    """Primary research metric: lower quantile across folds (default p25)."""
    vals = [float(x) for x in fold_metrics if x is not None and math.isfinite(float(x))]
    if not vals:
        return float("nan")
    return float(np.quantile(np.asarray(vals, dtype=float), q))


def sortino_definition_meta() -> dict[str, Any]:
    return {
        "target": SORTINO_TARGET,
        "annualization": ANNUALIZATION,
        "annualization_factor": ANNUALIZATION_FACTOR,
        "primary_metric": "p25_fold_sortino",
        "diagnostic_metric": "p25_fold_sharpe",
        "n_folds": N_FOLDS,
    }


def write_fold_artifacts(
    run_dir: Path | str,
    *,
    folds: list[dict[str, Any]],
    fold_sortinos: Sequence[float],
    fold_sharpes: Sequence[float] | None = None,
    extra_metrics: dict[str, Any] | None = None,
) -> dict[str, Path]:
    """Write folds.json + metrics.json under runs/hunt_<ts>/ (no root overwrite)."""
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    folds_path = root / "folds.json"
    metrics_path = root / "metrics.json"
    folds_path.write_text(json.dumps({"folds": folds}, indent=2, default=str) + "\n")
    metrics: dict[str, Any] = {
        **sortino_definition_meta(),
        "fold_sortino": [None if (x is None or not math.isfinite(float(x))) else float(x) for x in fold_sortinos],
        "p25_fold_sortino": lower_quantile_metric(fold_sortinos),
    }
    if fold_sharpes is not None:
        metrics["fold_sharpe"] = [
            None if (x is None or not math.isfinite(float(x))) else float(x) for x in fold_sharpes
        ]
        metrics["p25_fold_sharpe"] = lower_quantile_metric(fold_sharpes)
    if extra_metrics:
        metrics.update(extra_metrics)
    metrics_path.write_text(json.dumps(metrics, indent=2, default=str) + "\n")
    return {"folds": folds_path, "metrics": metrics_path}


def daily_returns_from_equity(equity: pd.Series) -> np.ndarray:
    if equity is None or len(equity) < 2:
        return np.asarray([], dtype=float)
    dr = equity.resample("1D").last().dropna().pct_change().dropna()
    return dr.to_numpy(dtype=float)
