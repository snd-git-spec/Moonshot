"""Observability helpers — portfolio path, expectancy rollup, hunt funnel packing.

Fail-soft: callers should wrap with contextlib.suppress / try-except.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import pandas as pd

log = logging.getLogger("moonshot")

EXPECTANCY_ROLLUP_MIN_SEC = 15 * 60  # 15 minutes


def portfolio_lev_metrics(
    positions: Mapping[str, Any],
    last_price: Mapping[str, float],
    equity: float,
    *,
    mmr: float = 0.004,
) -> dict[str, float | None]:
    """Effective / net leverage, margin, and min distance-to-liq across open book."""
    if equity <= 0:
        return {
            "effective_lev": 0.0,
            "net_lev": 0.0,
            "margin_used": 0.0,
            "free_margin": float(equity),
            "min_liq_distance_pct": None,
        }
    margin_used = 0.0
    open_notional = 0.0
    net_notional = 0.0
    min_liq: float | None = None
    for sym, pos in positions.items():
        px = float(last_price.get(sym, getattr(pos, "avg_entry", 0.0) or 0.0))
        qty = float(getattr(pos, "current_qty", 0.0) or 0.0)
        if qty <= 0:
            continue
        sign = int(getattr(pos, "sign", 1) or 1)
        notional = qty * px
        open_notional += notional
        net_notional += sign * notional
        margin_used += float(pos.margin()) if hasattr(pos, "margin") else notional / max(
            1, int(getattr(pos, "leverage", 1) or 1)
        )
        try:
            liq = float(pos.liq_price(mmr))
            if px > 0 and liq == liq:
                dist = abs(px - liq) / px
                min_liq = dist if min_liq is None else min(min_liq, dist)
        except Exception:
            pass
    return {
        "effective_lev": open_notional / equity,
        "net_lev": net_notional / equity,
        "margin_used": margin_used,
        "free_margin": max(0.0, float(equity) - margin_used),
        "min_liq_distance_pct": None if min_liq is None else float(min_liq),
    }


def write_expectancy_rollups(db, *, force: bool = False) -> int:
    """Bucket paper trades by config_hash and persist expectancy_rollup rows.

    Returns number of buckets written. Throttled unless force=True.
    """
    from moonshot.paper_expectancy import bucket_trades, load_trade_rows

    last = db.last_expectancy_rollup_ts()
    now = pd.Timestamp.now(tz="UTC")
    if not force and last is not None:
        try:
            age = (now - pd.Timestamp(last)).total_seconds()
            if age < EXPECTANCY_ROLLUP_MIN_SEC:
                return 0
        except Exception:
            pass
    rows = load_trade_rows(db.path)
    buckets = bucket_trades(rows)
    if not buckets:
        # Still stamp an empty rollup so the schedule advances.
        db.log_expectancy_rollup(
            now.to_pydatetime(),
            config_hash=None,
            n=0,
            mean_r=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            cost_share=float("nan"),
            max_dd=0.0,
            liquidations=0,
            label="EMPTY",
        )
        return 0
    n_written = 0
    for b in buckets:
        db.log_expectancy_rollup(
            now.to_pydatetime(),
            config_hash=b.config_hash,
            n=b.n,
            mean_r=b.mean_r,
            ci_low=b.ci_low,
            ci_high=b.ci_high,
            cost_share=b.cost_share,
            max_dd=b.max_dd,
            liquidations=b.liquidations,
            label=b.decision_label(),
        )
        n_written += 1
    return n_written


def kill_histogram(
    matrix_rows: Sequence[Mapping[str, Any]] | None,
    confirm_rejects: Mapping[str, Any] | None = None,
    material_why: str | None = None,
) -> dict[str, int]:
    hist: dict[str, int] = {}
    for r in matrix_rows or []:
        if r.get("skip"):
            w = str(r.get("why") or "skip")
        elif r.get("why"):
            w = str(r.get("why"))
        elif not r.get("robust_ok", True):
            w = "period_robust"
        else:
            w = "ok"
        hist[w] = hist.get(w, 0) + 1
    for k, v in (confirm_rejects or {}).items():
        try:
            n = int(v or 0)
        except (TypeError, ValueError):
            n = 0
        if n:
            hist[f"confirm_{k}"] = n
    if material_why:
        key = f"material:{str(material_why)[:96]}"
        hist[key] = hist.get(key, 0) + 1
    return hist


def coverage_unique_counts(
    cfgs: Sequence[Any] | None = None,
    *,
    matrix_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, int]:
    """Count unique strategy / timeframe / leverage seen this pass."""
    strats: set[str] = set()
    tfs: set[str] = set()
    levs: set[int] = set()
    n = 0
    for c in cfgs or []:
        try:
            strats.add(str(getattr(c, "strategy", "") or "").strip().lower())
            tfs.add(str(getattr(c, "timeframe", "") or "").strip().lower())
            levs.add(int(getattr(c, "leverage", 0) or 0))
            n += 1
        except Exception:
            continue
    for r in matrix_rows or []:
        if r.get("skip"):
            continue
        try:
            if r.get("strategy"):
                strats.add(str(r.get("strategy") or "").strip().lower())
            if r.get("tf") or r.get("timeframe"):
                tfs.add(str(r.get("tf") or r.get("timeframe") or "").strip().lower())
            if r.get("leverage") is not None:
                levs.add(int(r.get("leverage") or 0))
            n += 1
        except Exception:
            continue
    strats.discard("")
    tfs.discard("")
    levs.discard(0)
    return {
        "n_cand": int(n),
        "n_unique_strat": len(strats),
        "n_unique_tf": len(tfs),
        "n_unique_lev": len(levs),
    }


def hunt_funnel_metrics(
    *,
    scout_in: int,
    scout_keep: int,
    confirm_b: int,
    confirm_hits: int,
    confirm_rejects: Mapping[str, Any] | None,
    matrix_rows: Sequence[Mapping[str, Any]] | None,
    finalist_ok: bool,
    gates_pass: bool,
    promote: bool,
    cost_pass: int | None = None,
    fold_pass: int | None = None,
    promote_attempt: int | None = None,
    primary_metric: str = "p25_fold_sortino",
    n_unique_strat: int | None = None,
    n_unique_tf: int | None = None,
    n_unique_lev: int | None = None,
    n_cand: int | None = None,
) -> dict[str, Any]:
    scored = 0
    for r in matrix_rows or []:
        if not r.get("skip"):
            scored += 1
    dual_fail = int((confirm_rejects or {}).get("dual") or 0)
    out: dict[str, Any] = {
        "scout_in": int(scout_in),
        "scout_keep": int(scout_keep),
        "confirm_soft": int(confirm_hits),
        "confirm_b": int(confirm_b),
        "confirm_dual_fail": dual_fail,
        "matrix_scored": scored,
        "finalist": int(bool(finalist_ok)),
        "gates_pass": int(bool(gates_pass)),
        "promote": int(bool(promote)),
        "primary_metric": str(primary_metric),
    }
    if cost_pass is not None:
        out["cost_pass"] = int(cost_pass)
    if fold_pass is not None:
        out["fold_pass"] = int(fold_pass)
    if promote_attempt is not None:
        out["promote_attempt"] = int(promote_attempt)
    if n_unique_strat is not None:
        out["n_unique_strat"] = int(n_unique_strat)
    if n_unique_tf is not None:
        out["n_unique_tf"] = int(n_unique_tf)
    if n_unique_lev is not None:
        out["n_unique_lev"] = int(n_unique_lev)
    if n_cand is not None:
        out["n_cand"] = int(n_cand)
    return out


def enrich_ladder_rows(
    ladder: Sequence[Mapping[str, Any]] | None,
    *,
    book_exp: float | None = None,
    book_dd: float | None = None,
    cost_2x_ok: bool | None = None,
) -> list[dict[str, Any]]:
    """Normalize ladder rows; never invent Sortino/liqs — null when unknown."""
    out: list[dict[str, Any]] = []
    for r in ladder or []:
        row = dict(r)
        row.setdefault("sortino_p25", row.get("sortino_p25"))
        row.setdefault("exp", row.get("exp", book_exp))
        row.setdefault("dd", row.get("dd", book_dd))
        row.setdefault("liqs", row.get("liqs"))
        if "cost_2x_ok" not in row:
            row["cost_2x_ok"] = cost_2x_ok
        row.setdefault("week_ret", row.get("week_ret"))
        out.append(row)
    return out


def log_hunt_pass_experiment(
    *,
    generation: int,
    config_hash: str | None,
    data_hash: str | None,
    funnel: Mapping[str, Any],
    kill_hist: Mapping[str, Any],
    ladder: Sequence[Mapping[str, Any]] | None,
    timing: Mapping[str, Any] | None,
    cache: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
    sealed_holdout: Mapping[str, Any] | None = None,
    dsr: Mapping[str, Any] | None = None,
    db_path: Path | str | None = None,
) -> None:
    from moonshot.experiment_log import log_run
    from moonshot.paths import ROOT

    path = Path(db_path) if db_path else (ROOT / "runs" / "experiments.db")
    run_id = f"hunt_pass_{generation}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"
    sealed = dict(sealed_holdout) if sealed_holdout is not None else {
        "evaluated": False,
        "passed": None,
        "why": "not_provided",
    }
    metrics = {
        "funnel": dict(funnel),
        "kill_histogram": dict(kill_hist),
        "leverage_ladder": list(ladder or []),
        "timing": dict(timing or {}),
        "cache": dict(cache or {}),
        "sealed_holdout": sealed,
        "dsr": dict(dsr) if dsr is not None else {"evaluated": False, "passed": None},
    }
    log_run(
        run_id=run_id,
        kind="hunt_pass",
        config_hash=config_hash,
        data_hash=data_hash,
        metrics=metrics,
        meta=dict(meta or {}),
        db_path=path,
    )


def finite_or_none(x: Any) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v
