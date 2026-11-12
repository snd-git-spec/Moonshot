"""Promote-path gates — single import surface so wiring cannot silently regress.

Research promote order (after candidate selection):
  period-robust → sealed holdout → DSR → material/WF/finalist/transfer/2×/hysteresis/dwell → apply

Escape only when ``live_death_book``; still needs escape windows + sealed floor.
DSR is v1 multiple-testing (PBO/CSCV still TODO in dsr.py).
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import pandas as pd

from moonshot.hunt.constants import HUNT_TARGETS
from moonshot.hunt.cost_holdout import sealed_holdout_audit_fields
from moonshot.hunt.dsr import (
    DSR_THRESHOLD,
    deflated_sharpe_ratio,
    dsr_passes,
    record_trial,
    trial_count,
)

# Names referenced from improve.py — test_promote_gate_wiring asserts these strings.
PROMOTE_GATE_SYMBOLS = (
    "live_death_book",
    "selection_end_for_sealed",
    "assert_selection_before_sealed",
    "sealed_holdout_gate",
    "research_dsr_gate",
    "escape_sealed_floor_ok",
    "record_pass_trials",
)


def live_death_book(
    *,
    week_ret: float | None,
    oos_ret: float | None,
    soft_ret: float | None,
    paper_bleeding: bool = False,
) -> tuple[bool, str]:
    """True when Live is a death book — only then may escape arm."""
    if paper_bleeding:
        return True, "paper_bleeding"
    arm_oos = float(HUNT_TARGETS.get("escape_arm_oos", -0.10))
    arm_soft = float(HUNT_TARGETS.get("escape_arm_soft", -0.10))
    try:
        wr = float(week_ret) if week_ret is not None else None
    except (TypeError, ValueError):
        wr = None
    try:
        oos = float(oos_ret) if oos_ret is not None else None
    except (TypeError, ValueError):
        oos = None
    try:
        soft = float(soft_ret) if soft_ret is not None else None
    except (TypeError, ValueError):
        soft = None
    if wr is not None and wr < 0.0:
        return True, f"week {wr:+.0%} < 0"
    if oos is not None and oos < arm_oos:
        return True, f"oos {oos:+.0%} < arm {arm_oos:+.0%}"
    if soft is not None and soft < arm_soft:
        return True, f"soft {soft:+.0%} < arm {arm_soft:+.0%}"
    return False, "live_not_death"


def sealed_holdout_days() -> int:
    return int(HUNT_TARGETS.get("sealed_holdout_days", 14))


def selection_end_for_sealed(data_end: pd.Timestamp) -> pd.Timestamp:
    """Last timestamp search may use — sealed bars start after this."""
    end = pd.Timestamp(data_end)
    end = end.tz_localize("UTC") if end.tzinfo is None else end.tz_convert("UTC")
    return end - pd.Timedelta(days=int(sealed_holdout_days()))


def assert_selection_before_sealed(
    selection_end: pd.Timestamp,
    data_end: pd.Timestamp,
) -> None:
    """Fail closed if selection window eats into sealed bars."""
    sealed_start = selection_end_for_sealed(data_end)
    sel = pd.Timestamp(selection_end)
    sel = sel.tz_localize("UTC") if sel.tzinfo is None else sel.tz_convert("UTC")
    # Allow equality: selection ends at sealed_start (sealed is (sealed_start, data_end]).
    if sel > sealed_start + pd.Timedelta(seconds=1):
        raise RuntimeError(
            f"selection end {sel.isoformat()} past sealed start {sealed_start.isoformat()}"
        )


def sealed_holdout_gate(
    *,
    sealed_week_ret: float | None,
    live_sealed_week_ret: float | None,
    kill_illusion: bool = False,
    research_min_week_edge: float | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """Research sealed exam: non-negative sealed week and beat Live sealed by edge."""
    floor = float(HUNT_TARGETS.get("sealed_week_floor", 0.0))
    edge = (
        float(research_min_week_edge)
        if research_min_week_edge is not None
        else float(HUNT_TARGETS.get("sealed_vs_live_edge", 0.008))
    )
    meta: dict[str, Any] = {
        "evaluated": True,
        "sealed_week_ret": sealed_week_ret,
        "live_sealed_week_ret": live_sealed_week_ret,
        "floor": floor,
        "edge": edge,
    }
    if kill_illusion:
        meta["passed"] = False
        meta["why"] = "sealed_kill_illusion"
        return False, "sealed_kill_illusion", meta
    try:
        sw = float(sealed_week_ret) if sealed_week_ret is not None else None
    except (TypeError, ValueError):
        sw = None
    if sw is None:
        meta["passed"] = False
        meta["why"] = "sealed_missing"
        return False, "sealed_missing", meta
    if sw < floor:
        meta["passed"] = False
        meta["why"] = f"sealed_week {sw:+.0%} < floor {floor:+.0%}"
        return False, str(meta["why"]), meta
    try:
        live_s = float(live_sealed_week_ret) if live_sealed_week_ret is not None else None
    except (TypeError, ValueError):
        live_s = None
    if live_s is not None and sw + 1e-12 < live_s + edge:
        why = f"sealed_week {sw:+.0%} < live_sealed {live_s:+.0%}+{edge:.0%}"
        meta["passed"] = False
        meta["why"] = why
        return False, why, meta
    meta["passed"] = True
    meta["why"] = "sealed_ok"
    return True, "sealed_ok", meta


def escape_sealed_floor_ok(
    sealed_week_ret: float | None,
    live_sealed_week_ret: float | None = None,
) -> tuple[bool, str]:
    """Escape sealed gate — absolute floor, or relative when Live sealed is already dead.

    Absolute ``escape_sealed_floor`` still blocks trash when Live sealed is healthy.
    When Live sealed is already below that floor (death book), survive-first:
    allow the escape if its sealed week is not materially worse than Live sealed
    (``escape_sealed_vs_live_tol_pp``). Otherwise every less-bad TF/lev escape
    dies forever while capital stays on the worse Live book.
    """
    floor = float(HUNT_TARGETS.get("escape_sealed_floor", -0.10))
    rel_tol = float(HUNT_TARGETS.get("escape_sealed_vs_live_tol_pp", 0.02))
    try:
        sw = float(sealed_week_ret) if sealed_week_ret is not None else None
    except (TypeError, ValueError):
        sw = None
    try:
        live_s = float(live_sealed_week_ret) if live_sealed_week_ret is not None else None
    except (TypeError, ValueError):
        live_s = None
    if sw is None:
        return False, "escape_sealed_missing"
    # Live sealed already under absolute floor → relative survive-first.
    if live_s is not None and live_s < floor:
        if sw + 1e-12 >= live_s - rel_tol:
            return True, "escape_sealed_vs_live_ok"
        return False, (
            f"escape_sealed {sw:+.0%} < live_sealed {live_s:+.0%}"
            f"-{rel_tol:.0%} (live sealed already < floor {floor:+.0%})"
        )
    if sw < floor:
        return False, f"escape_sealed {sw:+.0%} < floor {floor:+.0%}"
    return True, "escape_sealed_ok"


def sharpe_from_returns(returns: list[float] | tuple[float, ...]) -> tuple[float, int]:
    """Simple Sharpe from per-trade or per-bar returns; n_obs = len."""
    xs = [float(x) for x in returns if x is not None and math.isfinite(float(x))]
    n = len(xs)
    if n < 2:
        return 0.0, n
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / max(n - 1, 1)
    if var <= 0:
        return (0.0 if mean == 0 else (1e6 if mean > 0 else -1e6)), n
    return mean / math.sqrt(var), n


def research_dsr_gate(
    *,
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    threshold: float | None = None,
) -> tuple[bool, str, dict[str, Any]]:
    """v1 multiple-testing gate — PBO/CSCV still TODO in dsr.py."""
    thr = float(DSR_THRESHOLD if threshold is None else threshold)
    dsr = deflated_sharpe_ratio(float(observed_sr), int(n_obs), int(n_trials))
    ok = dsr_passes(dsr, thr)
    meta = {
        "evaluated": True,
        "observed_sr": float(observed_sr),
        "n_obs": int(n_obs),
        "n_trials": int(n_trials),
        "dsr": float(dsr),
        "threshold": thr,
        "passed": bool(ok),
        "why": "dsr_ok" if ok else "dsr_fail",
    }
    return bool(ok), str(meta["why"]), meta


def record_pass_trials(
    config_hashes: list[str],
    *,
    run_id: str,
    first_seen_ts: str,
    db_path: Any = None,
) -> tuple[int, int]:
    """Record distinct configs scored this pass; return (n_pass, n_cum)."""
    seen: set[str] = set()
    n_pass = 0
    n_cum = 0
    kwargs = {}
    if db_path is not None:
        kwargs["db_path"] = db_path
    for h in config_hashes:
        key = str(h or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        n_pass += 1
        n_cum = record_trial(key, run_id, first_seen_ts=first_seen_ts, **kwargs)
    if n_pass == 0:
        n_cum = trial_count(**kwargs) if kwargs else trial_count()
    return n_pass, int(n_cum)


def sealed_audit_payload(
    *,
    git_sha: str = "",
    data_slice_hash: str = "",
    holdout_start_idx: int = 0,
    holdout_end_idx: int = 0,
    metric: float | None = None,
    passed: bool | None = None,
) -> dict[str, Any]:
    return sealed_holdout_audit_fields(
        git_sha=git_sha,
        data_slice_hash=data_slice_hash,
        holdout_start_idx=holdout_start_idx,
        holdout_end_idx=holdout_end_idx,
        metric=metric,
        passed=passed,
    )


def nearest_miss_payload(
    *,
    week_ret: float | None = None,
    prior_ret: float | None = None,
    oos_ret: float | None = None,
    soft_ret: float | None = None,
    why: str = "",
    search_mode: str = "",
    book: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "week_ret": week_ret,
        "prior_ret": prior_ret,
        "oos_ret": oos_ret,
        "soft_ret": soft_ret,
        "why": why,
        "search_mode": search_mode,
        "book": dict(book or {}),
    }
