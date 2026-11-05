"""2× cost stress + sealed holdout (A.3c)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

HOLDOUT_FRAC = 0.22  # final ~20–25% by time


@dataclass(frozen=True)
class CostStressResult:
    pass_1x: bool
    pass_2x: bool
    metric_1x: float
    metric_2x: float

    @property
    def eligible(self) -> bool:
        return bool(self.pass_1x and self.pass_2x)


def cost_stress_gate(
    metric_1x: float,
    metric_2x: float,
    *,
    min_metric: float = 0.0,
) -> CostStressResult:
    """Eligible only if gates pass at 1× and 2× fee+slippage."""
    p1 = metric_1x > min_metric
    p2 = metric_2x > min_metric
    return CostStressResult(pass_1x=p1, pass_2x=p2, metric_1x=metric_1x, metric_2x=metric_2x)


def sealed_holdout_slice(
    n_bars: int,
    *,
    holdout_frac: float = HOLDOUT_FRAC,
) -> tuple[slice, slice]:
    """Return (selection_slice, sealed_holdout_slice). Holdout is final frac by time."""
    if n_bars < 2:
        raise ValueError("n_bars must be >= 2")
    cut = max(1, int(n_bars * (1.0 - holdout_frac)))
    cut = min(cut, n_bars - 1)
    return slice(0, cut), slice(cut, n_bars)


class SealedHoldoutGuard:
    """Ensures sealed holdout indices are never read during selection."""

    def __init__(self, holdout: slice):
        self.holdout = holdout
        self._read_attempted = False

    def assert_not_read(self, index: int) -> None:
        start = self.holdout.start or 0
        stop = self.holdout.stop if self.holdout.stop is not None else 10**18
        if start <= index < stop:
            self._read_attempted = True
            raise RuntimeError(f"sealed holdout bar {index} must not be read during selection")

    def assert_range_not_read(self, start: int, end: int) -> None:
        for i in range(start, end):
            self.assert_not_read(i)

    @property
    def was_read(self) -> bool:
        return self._read_attempted


def sealed_holdout_audit_fields(
    *,
    git_sha: str,
    data_slice_hash: str,
    holdout_start_idx: int,
    holdout_end_idx: int,
    metric: float | None,
    passed: bool | None,
) -> dict[str, Any]:
    return {
        "git_sha": git_sha,
        "data_slice_hash": data_slice_hash,
        "holdout_start_idx": holdout_start_idx,
        "holdout_end_idx": holdout_end_idx,
        "metric": metric,
        "passed": passed,
        "retired_on_fail": bool(passed is False),
    }


def hash_data_slice(payload: Sequence[Any] | Mapping[str, Any] | bytes | str) -> str:
    if isinstance(payload, bytes):
        raw = payload
    elif isinstance(payload, str):
        raw = payload.encode()
    else:
        raw = json.dumps(payload, sort_keys=True, default=str).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def write_cost_holdout_artifacts(
    run_dir: Path | str,
    *,
    stress: CostStressResult,
    holdout_audit: dict[str, Any],
) -> dict[str, Path]:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    cost_path = root / "cost_stress.json"
    hold_path = root / "sealed_holdout.json"
    cost_path.write_text(
        json.dumps(
            {
                "pass_1x": stress.pass_1x,
                "pass_2x": stress.pass_2x,
                "metric_1x": stress.metric_1x,
                "metric_2x": stress.metric_2x,
                "eligible": stress.eligible,
            },
            indent=2,
        )
        + "\n"
    )
    hold_path.write_text(json.dumps(holdout_audit, indent=2) + "\n")
    return {"cost_stress": cost_path, "sealed_holdout": hold_path}
