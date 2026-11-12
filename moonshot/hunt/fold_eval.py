"""Fold-based cfg evaluation for hunt ranking (p25 Sortino primary).

Uses fold_runner boundaries + BacktestEngine slices. Smoke = 2 folds on a
short calendar window; full = 4 folds up to selection ``end``.
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd

from moonshot.backtest import BacktestEngine
from moonshot.config import Config
from moonshot.hunt.constants import HUNT_HALVING_A_DAYS, HUNT_TARGETS
from moonshot.hunt.fold_runner import (
    N_FOLDS,
    daily_returns_from_equity,
    fold_boundaries,
    lower_quantile_metric,
    purge_embargo_bars,
    sortino_from_daily_returns,
)


def _bar_index(data: dict, end: pd.Timestamp) -> pd.DatetimeIndex:
    """Sorted unique bar timestamps across symbols, clipped to ``end``."""
    end = pd.Timestamp(end)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    stamps: list[pd.Timestamp] = []
    for sd in data.values():
        bars = getattr(sd, "bars", None)
        if bars is None or len(bars) == 0:
            continue
        for b in bars:
            ts = pd.Timestamp(b.Index)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            if ts <= end:
                stamps.append(ts)
    if not stamps:
        return pd.DatetimeIndex([], tz="UTC")
    return pd.DatetimeIndex(sorted(set(stamps)))


def eval_cfg_folds(
    data: dict,
    cfg: Config,
    *,
    end: Optional[pd.Timestamp] = None,
    smoke: bool = False,
    n_folds: int | None = None,
    calendar_days: int | None = None,
) -> dict[str, Any]:
    """Run fold backtests; return p25 Sortino + ROI summary.

    smoke=True → 2 folds on ~HUNT_HALVING_A_DAYS (cheap Stage A).
    smoke=False → 4 folds on data ending at ``end`` (selection window).
    """
    if end is None:
        try:
            end = max(sd.bars[-1].Index for sd in data.values())
        except Exception:
            return {
                "ok": False,
                "why": "no_bars",
                "p25_sortino": float("nan"),
                "fold_sortinos": [],
                "roi": 0.0,
                "final_equity": float(getattr(cfg, "starting_equity", 0) or 0),
                "trades": 0,
            }
    end = pd.Timestamp(end)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")

    folds_n = int(n_folds) if n_folds is not None else (2 if smoke else int(N_FOLDS))
    days = int(
        calendar_days
        if calendar_days is not None
        else (int(HUNT_HALVING_A_DAYS) if smoke else 0)
    )
    idx = _bar_index(data, end)
    if days > 0 and len(idx) > 0:
        start_cut = end - pd.Timedelta(days=days)
        idx = idx[idx >= start_cut]
    n_bars = int(len(idx))
    min_folds = 1 if smoke else 2
    if n_bars < max(folds_n * 3, 10):
        return {
            "ok": False,
            "why": f"thin_bars n={n_bars}",
            "p25_sortino": float("nan"),
            "fold_sortinos": [],
            "roi": 0.0,
            "final_equity": float(cfg.starting_equity),
            "trades": 0,
            "n_bars": n_bars,
            "smoke": bool(smoke),
        }

    lookback = int(getattr(cfg, "ema_entry_period", 50) or 50)
    hold = int(getattr(cfg, "time_stop_bars", 48) or 48)
    purge, embargo = purge_embargo_bars(lookback, hold)
    # Smoke: shrink purge so short windows still get 2 folds.
    if smoke:
        purge = min(purge, max(5, n_bars // 20))
        embargo = min(embargo, max(2, n_bars // 40))

    bounds = fold_boundaries(n_bars, folds_n, purge_bars=purge, embargo_bars=embargo)
    lab = (
        bool(cfg.enable_pyramiding),
        bool(cfg.enable_partial),
        int(cfg.leverage),
    )
    fold_sortinos: list[float] = []
    total_trades = 0
    last_final = float(cfg.starting_equity)
    last_ret = 0.0
    finite = 0

    for fb in bounds:
        i0, i1 = int(fb["start"]), int(fb["end"])
        if i1 <= i0 or i1 > n_bars:
            fold_sortinos.append(float("nan"))
            continue
        f_start = idx[i0]
        f_end = idx[min(i1 - 1, n_bars - 1)]
        try:
            res = BacktestEngine(data, cfg, start=f_start, end=f_end).run(lab)
        except Exception:
            fold_sortinos.append(float("nan"))
            continue
        total_trades += int(res.trades or 0)
        last_final = float(res.final_equity)
        last_ret = float(res.total_return)
        try:
            dr = daily_returns_from_equity(res.equity_curve)
            s = sortino_from_daily_returns(dr)
        except Exception:
            s = float("nan")
        if s is not None and math.isfinite(float(s)):
            finite += 1
            fold_sortinos.append(float(s))
        else:
            # Fall back to engine sortino if daily series too short.
            eng_s = float(res.sortino) if res.sortino == res.sortino else float("nan")
            if math.isfinite(eng_s):
                finite += 1
                fold_sortinos.append(eng_s)
            else:
                fold_sortinos.append(float("nan"))

    p25 = lower_quantile_metric(fold_sortinos)
    floor = float(HUNT_TARGETS.get("fold_sortino_floor", 0.0))
    ok = finite >= min_folds and math.isfinite(p25) and p25 >= floor
    why = (
        "passed"
        if ok
        else (
            f"finite_folds={finite}<{min_folds}"
            if finite < min_folds
            else (
                f"p25_sortino={p25}"
                if not math.isfinite(p25)
                else f"p25 {p25:+.3f} < floor {floor:+.3f}"
            )
        )
    )
    return {
        "ok": bool(ok),
        "why": why,
        "p25_sortino": float(p25) if math.isfinite(p25) else float("nan"),
        "fold_sortinos": fold_sortinos,
        "roi": float(last_ret),
        "final_equity": float(last_final),
        "trades": int(total_trades),
        "n_bars": n_bars,
        "n_folds": folds_n,
        "finite_folds": finite,
        "smoke": bool(smoke),
        "span": f"{idx[0]:%Y-%m-%d} → {idx[-1]:%Y-%m-%d}" if n_bars else "",
    }


def cost_ok_for_cfg(
    data: dict,
    cfg: Config,
    *,
    end: Optional[pd.Timestamp] = None,
    days: int = 14,
) -> tuple[bool, float, float]:
    """1× vs 2× fee+slip window ROI; True if both > 0 (cost_stress_gate spirit)."""
    from dataclasses import replace

    from moonshot.hunt.cost_holdout import cost_stress_gate

    if end is None:
        end = max(sd.bars[-1].Index for sd in data.values())
    end = pd.Timestamp(end)
    start = end - pd.Timedelta(days=int(days))
    lab = (bool(cfg.enable_pyramiding), bool(cfg.enable_partial), int(cfg.leverage))
    fee = float(getattr(cfg, "fee_taker", 0.0008) or 0.0008)
    slip_bps = float(getattr(cfg, "slippage_bps", 3.0) or 3.0)
    maker = float(getattr(cfg, "fee_maker", fee) or fee)

    def _ret(mult: float) -> float:
        c2 = replace(
            cfg,
            fee_taker=fee * mult,
            fee_maker=maker * mult,
            slippage_bps=slip_bps * mult,
        )
        try:
            r = BacktestEngine(data, c2, start=start, end=end).run(lab)
            return float(r.total_return)
        except Exception:
            return float("nan")

    m1 = _ret(1.0)
    m2 = _ret(2.0)
    gate = cost_stress_gate(
        m1 if math.isfinite(m1) else -1.0, m2 if math.isfinite(m2) else -1.0
    )
    return (
        bool(gate.eligible),
        float(m1) if math.isfinite(m1) else -1.0,
        float(m2) if math.isfinite(m2) else -1.0,
    )