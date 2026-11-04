from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.config import (
    Config,
)
from moonshot.features import (
    adx, prepare_data,
)
from moonshot.backtest import (
    BacktestEngine,
    BacktestResult,
)
from moonshot.hunt.constants import (
    HUNT_BREAKOUT_N,
    HUNT_EXPLORE_N,
    HUNT_LEV_BREAKOUT,
    HUNT_LEV_FORCE,
    HUNT_LEV_POLISH,
    HUNT_STRATEGIES,
    HUNT_TARGETS,
    HUNT_TF_ALLOW,
    _lev_slip_stress,
)

from moonshot.paths import (
    ROOT, ENV_PATH, DB_PATH, AUTOPILOT_STATE, LIVE_FEED,
    PAPER_RUNTIME_PATH, RULES_PATH,
)

def _soft_kill_cfg(cfg: Config) -> Config:
    """Kills effectively off — used to detect kill-switch-only 'wins'."""
    return replace(
        cfg,
        max_peak_drawdown_pct=0.99,
        expectancy_floor=-99.0,
        streak_losses=999,
        max_daily_loss_pct=0.99,
    )


def _freeze_risk(base: Config, c: Config) -> Config:
    """Risk/margin/filters/exits/strategy/TF hunt-able. Aggressive: leverage up to 25x."""
    aggressive = bool(getattr(base, "aggressive_mode", False))
    # Strategy + TF may change in explore/breakout (Layer 4 / 4c)
    strat = str(getattr(c, "strategy", base.strategy) or base.strategy).strip().lower()
    if strat not in HUNT_STRATEGIES and strat not in ("momentum_rule", "combo_and", "regime_breakout"):
        strat = (base.strategy or "trend_mtf").strip().lower()
    tf = str(getattr(c, "timeframe", base.timeframe) or base.timeframe).strip().lower()
    if tf not in HUNT_TF_ALLOW:
        tf = str(base.timeframe) if str(base.timeframe) in HUNT_TF_ALLOW else "15m"

    if aggressive:
        lev = int(max(5, min(25, int(getattr(c, "leverage", base.leverage) or base.leverage))))
        # 25x only on short clocks
        if lev >= 25 and tf == "15m":
            tf = "5m"
            lev = 25
        if lev >= 15 and tf == "15m":
            # allow 15x@15m but prefer not forcing; keep as sampled
            pass
        # Risk / book size scale down with leverage
        # Keep enough concurrent slots — density dies if open_hi is 2–4 on a 10-coin book
        if lev >= 25:
            risk_hi, open_hi, stop_lo = 0.02, 4, 2.5
        elif lev >= 15:
            risk_hi, open_hi, stop_lo = 0.025, 6, 2.25
        elif lev >= 10:
            risk_hi, open_hi, stop_lo = 0.03, 8, 2.0
        else:
            risk_hi, open_hi, stop_lo = 0.05, 8, 1.5
        risk = float(max(0.015, min(risk_hi, c.risk_pct)))
        max_open = int(max(1, min(open_hi, int(
            getattr(c, "max_open_positions", getattr(base, "max_open_positions", 4)) or 4
        ))))
        max_acct = float(max(lev, min(25.0, float(
            getattr(c, "max_account_leverage", float(lev)) or lev
        ))))
        max_net = float(max(lev, min(25.0, float(
            getattr(c, "max_net_leverage", float(lev)) or lev
        ))))
        stop_atr = float(max(stop_lo, min(4.0, c.initial_stop_atr)))
    else:
        lev = int(base.leverage)
        risk = float(max(0.02, min(0.05, c.risk_pct)))
        max_acct = float(base.max_account_leverage)
        max_net = float(getattr(base, "max_net_leverage", 3.0))
        max_open = int(getattr(base, "max_open_positions", 3))
        stop_atr = float(max(1.5, min(4.0, c.initial_stop_atr)))
        strat = (base.strategy or "trend_mtf").strip().lower()
        tf = str(base.timeframe)

    margin = float(max(0.16, min(0.80, c.initial_margin_pct)))
    margin = max(margin, min(0.80, risk * 8.0))
    # Kill rails for leveraged minute books
    if aggressive and (tf.endswith("m") or int(lev) >= 5):
        daily_lo, daily_hi = 0.28, 0.40
        dd_lo, dd_hi = 0.50, 0.65
        exp_floor = float(min(-0.10, max(-0.35, getattr(c, "expectancy_floor", -0.15))))
        streak_n = int(max(6, min(12, getattr(c, "streak_losses", 7) or 7)))
        streak_hrs = float(max(6.0, min(24.0, getattr(c, "streak_pause_hours", 12.0) or 12.0)))
    else:
        daily_lo, daily_hi = 0.10, 0.35
        dd_lo, dd_hi = 0.35, 0.60
        exp_floor = float(getattr(c, "expectancy_floor", base.expectancy_floor))
        streak_n = int(getattr(c, "streak_losses", base.streak_losses))
        streak_hrs = float(getattr(c, "streak_pause_hours", base.streak_pause_hours))
    daily = float(max(daily_lo, min(daily_hi, max(c.max_daily_loss_pct, risk * 3))))
    peak_dd = float(max(dd_lo, min(dd_hi, c.max_peak_drawdown_pct)))
    h4_fast = int(max(8, min(50, int(getattr(c, "h4_ema_fast", base.h4_ema_fast) or base.h4_ema_fast))))
    h4_slow = int(max(h4_fast + 5, min(100, int(getattr(c, "h4_ema_slow", base.h4_ema_slow) or base.h4_ema_slow))))
    ema_entry = int(max(8, min(50, int(getattr(c, "ema_entry_period", base.ema_entry_period) or base.ema_entry_period))))
    pullback_lb = int(max(2, min(12, int(getattr(c, "pullback_lookback", base.pullback_lookback) or base.pullback_lookback))))
    btc_adx = float(max(0.0, min(30.0, float(getattr(c, "btc_adx_min", 0.0) or 0.0))))
    breadth = float(max(0.0, min(0.8, float(getattr(c, "breadth_min", 0.0) or 0.0))))
    # Cap ADX/vol so hunt cannot "improve" into a silent zero-setup book.
    # Keep ADX modest on minute TFs — 28+ starves entries (~3/day).
    adx_hi = 22.0 if str(tf).endswith("m") else 40.0
    adx_enter = float(max(12.0, min(adx_hi, c.adx_enter)))
    adx_exit = float(max(10.0, min(adx_enter - 4.0, c.adx_exit)))
    vol_mult = float(max(1.05, min(1.6 if str(tf).endswith("m") else 3.0, c.volume_multiplier)))
    cool_hi = 4 if str(tf).endswith("m") else 12
    cool_bars = int(max(0, min(cool_hi, int(c.cooldown_bars))))
    return replace(
        c,
        strategy=strat,
        initial_stop_atr=stop_atr,
        leverage=lev,
        max_account_leverage=max_acct,
        max_net_leverage=max_net,
        max_open_positions=max_open,
        enable_pyramiding=bool(base.enable_pyramiding),
        timeframe=tf,
        require_btc_regime=bool(c.require_btc_regime),
        require_rising_adx=bool(c.require_rising_adx),
        require_squeeze=bool(c.require_squeeze),
        require_rsi_veto=bool(c.require_rsi_veto),
        use_asymmetric_trail=bool(c.use_asymmetric_trail),
        trail_atr_loser=float(max(1.0, min(3.5, c.trail_atr_loser))),
        breakout_lookback=int(max(8, min(48, c.breakout_lookback))),
        breakout_buffer_atr=float(max(0.0, min(1.0, c.breakout_buffer_atr))),
        h4_ema_fast=h4_fast,
        h4_ema_slow=h4_slow,
        h4_adx_min_short=float(max(10.0, min(28.0, c.h4_adx_min_short))),
        ema_entry_period=ema_entry,
        pullback_lookback=pullback_lb,
        btc_adx_min=btc_adx,
        breadth_min=breadth,
        risk_pct=risk,
        initial_margin_pct=margin,
        aggressive_margin_pct=float(max(margin, min(0.80, c.aggressive_margin_pct))),
        max_daily_loss_pct=daily,
        max_peak_drawdown_pct=peak_dd,
        expectancy_floor=exp_floor,
        streak_losses=streak_n,
        streak_pause_hours=streak_hrs,
        adx_enter=adx_enter,
        adx_exit=adx_exit,
        volume_multiplier=vol_mult,
        cooldown_bars=cool_bars,
        partial_pct=float(max(0.25, min(0.75, c.partial_pct))),
        partial_at_r=float(max(0.5, min(3.0, c.partial_at_r))),
        time_stop_bars=int(max(4, min(48, c.time_stop_bars))),
        mom_lookback=int(max(3, min(24, c.mom_lookback))),
        mom_min_atr=float(max(0.25, min(2.0, c.mom_min_atr))),
    )


def _held_streak(state: dict) -> int:
    """Count consecutive non-improving history rows from the newest back."""
    hist = list(state.get("history") or [])
    streak = 0
    for h in reversed(hist):
        if h.get("improved"):
            break
        streak += 1
    return streak


# No settle lock — a better week book writes immediately (user wants climb, not wait).
RESEARCH_PROMOTE_MIN_SEC = 0
# Tiny material floors — any clear edge promotes (rampant improvement).
RESEARCH_MIN_WEEK_EDGE = 0.005
RESEARCH_MIN_SCORE_EDGE = 1
RESEARCH_MIN_OOS_RET_EDGE = 0.008
POST_PROMOTE_HUNT_PAUSE_SEC = 5
# Pending paper entry older than this many bars is dropped (no late fantasy fills).
PEND_ENTRY_MAX_BARS = 2
# If paper has not closed a trade for this long, hunt may rotate to a different
# tradeable book even when it does not beat the champion holdout week. Otherwise
# a ghost +40% week champion can freeze Live on zero setups indefinitely.
PAPER_SILENCE_ROTATE_HOURS = 6.0


def _paper_idle_hours() -> float:
    """Hours since last closed paper trade (DB). Large number if none."""
    try:
        con = sqlite3.connect(str(DB_PATH))
        try:
            row = con.execute(
                "SELECT ts FROM trades WHERE mode = 'paper' ORDER BY ts DESC LIMIT 1"
            ).fetchone()
        finally:
            con.close()
        if not row or not row[0]:
            return 999.0
        last = pd.Timestamp(row[0])
        if last.tzinfo is None:
            last = last.tz_localize("UTC")
        else:
            last = last.tz_convert("UTC")
        return float((pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600.0)
    except Exception:
        return 0.0


def _cfg_fingerprint(cfg: Config) -> tuple:
    """Stable identity of tradable knobs — used to skip no-op promotes."""
    return (
        str(cfg.strategy), str(cfg.timeframe), int(cfg.leverage),
        round(float(cfg.adx_enter), 3), round(float(cfg.adx_exit), 3),
        round(float(cfg.volume_multiplier), 3),
        round(float(cfg.breakout_buffer_atr), 3), int(cfg.breakout_lookback),
        round(float(cfg.trail_atr_multiple), 3), round(float(cfg.trail_atr_loser), 3),
        round(float(cfg.initial_stop_atr), 3), int(cfg.cooldown_bars),
        bool(cfg.require_rising_adx), bool(cfg.require_squeeze),
        bool(cfg.require_rsi_veto), bool(cfg.require_btc_regime),
        bool(cfg.use_asymmetric_trail), bool(cfg.enable_partial),
        bool(cfg.enable_time_stop), int(cfg.time_stop_bars),
        int(cfg.mom_lookback), round(float(cfg.mom_min_atr), 3),
        int(cfg.h4_ema_fast), int(cfg.h4_ema_slow),
        round(float(cfg.btc_adx_min), 3), round(float(cfg.breadth_min), 3),
    )


def _champion_from_history(hist_rows: list, current: dict | None) -> dict | None:
    """Keep the highest holdout-week knobs ever seen — never let a wipe lower the floor."""
    best = dict(current) if isinstance(current, dict) and current.get("knobs") else None
    best_w = float((best or {}).get("week_ret") or -999)
    for row in hist_rows or []:
        wr = row.get("week_return")
        kn = row.get("knobs") if isinstance(row.get("knobs"), dict) else None
        if wr is None or not kn:
            continue
        try:
            wr_f = float(wr)
        except (TypeError, ValueError):
            continue
        if wr_f > best_w + 0.0005:
            best_w = wr_f
            best = {
                "ts": row.get("ts"),
                "generation": row.get("generation"),
                "week_ret": wr_f,
                "score": row.get("score"),
                "knobs": kn,
                "week": row.get("week") if isinstance(row.get("week"), dict) else None,
            }
    return best


def _material_research_edge(
    *,
    best_cfg: Config, base_cfg: Config,
    best_grade: dict, base_grade: dict,
    best_week: dict, base_week: dict,
    best_o: BacktestResult, base_o: BacktestResult,
    best_soft: Optional[BacktestResult], base_soft: Optional[BacktestResult],
) -> tuple[bool, str]:
    """True only when the candidate is a real step up — not ±1% thrash.

    Hard rule: never research-promote a *worse* holdout week just because
    score/OOS look prettier. That is how Live showed “Upgraded” into −26%
    after a density reseed while a +2% candidate sat on a HOLD row.
    """
    if _cfg_fingerprint(best_cfg) == _cfg_fingerprint(base_cfg):
        return False, "identical knobs"
    best_w = float(best_week.get("ret") or 0)
    base_w = float(base_week.get("ret") or 0)
    if best_w < base_w - 0.001:
        return False, f"worse_week {base_w:+.0%}→{best_w:+.0%}"
    week_edge = best_w >= base_w + RESEARCH_MIN_WEEK_EDGE
    score_edge = (
        int(best_grade.get("score_raw", best_grade.get("score", 0)))
        >= int(base_grade.get("score_raw", base_grade.get("score", 0)))
        + RESEARCH_MIN_SCORE_EDGE
    )
    oos_edge = float(best_o.total_return) >= float(base_o.total_return) + RESEARCH_MIN_OOS_RET_EDGE
    soft_edge = False
    if best_soft is not None and base_soft is not None:
        soft_edge = (
            float(best_soft.total_return)
            >= float(base_soft.total_return) + RESEARCH_MIN_OOS_RET_EDGE
        )
    # Score/OOS can confirm a promote only when week did not go backwards
    # (already enforced above) AND week at least ties within noise.
    if week_edge:
        return True, "week_edge"
    if score_edge and (oos_edge or soft_edge) and best_w >= base_w - 0.001:
        return True, "score_edge"
    if oos_edge and soft_edge and best_w >= base_w - 0.001:
        return True, "oos_soft_edge"
    return False, "noise"


def _hunt_search_mode(generation: int, held_streak: int, hunt: bool) -> str:
    """
    polish = micro-nudges around current best.
    explore = wider random (default when stuck a bit).
    breakout = large jumps across filters/exits when held forever.
    Aggressive: spend most passes exploring / breaking out of local peaks.
    """
    if not hunt:
        return "explore"
    # Aggressive: breakout often, explore otherwise — almost never stay in polish
    if held_streak >= 2 or (generation % 3 == 0):
        return "breakout"
    return "explore"


def _dedupe_cfgs(base: Config, seeds: list) -> list:
    seen = set()
    uniq = []
    for c in seeds:
        c = _freeze_risk(base, c)
        key = (
            c.strategy, c.timeframe, c.adx_enter, c.adx_exit, c.volume_multiplier,
            c.breakout_buffer_atr, c.breakout_lookback, c.trail_atr_multiple,
            c.trail_atr_loser, round(c.initial_stop_atr, 2), c.cooldown_bars,
            c.require_rising_adx, c.require_squeeze, c.require_rsi_veto, c.require_btc_regime,
            c.use_asymmetric_trail, round(c.h4_adx_min_short, 1),
            c.rsi_long_max, c.rsi_short_min, c.squeeze_lookback,
            c.enable_partial, c.partial_at_r, c.partial_pct,
            c.enable_time_stop, c.time_stop_bars,
            round(c.risk_pct, 3), round(c.initial_margin_pct, 3),
            int(c.mom_lookback), round(c.mom_min_atr, 3),
            int(c.leverage), int(c.max_open_positions), round(float(c.max_net_leverage), 2),
            int(c.h4_ema_fast), int(c.h4_ema_slow), int(c.ema_entry_period),
            int(c.pullback_lookback), round(float(c.btc_adx_min), 1),
            round(float(c.breadth_min), 2),
        )
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _polish_cfgs(base: Config) -> list:
    """~24–32 nudges around the current best — find small upgrades fast."""
    seeds = [base]
    for d in (-3, -2, -1, 1, 2, 3):
        adx = max(14.0, min(40.0, base.adx_enter + d))
        seeds.append(replace(base, adx_enter=adx, adx_exit=max(10.0, adx - 6.0)))
    for d in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3):
        seeds.append(replace(
            base, volume_multiplier=max(1.1, min(3.0, round(base.volume_multiplier + d, 1))),
        ))
    for d in (-3, -2, -1, 1, 2, 3):
        seeds.append(replace(base, mom_lookback=max(3, min(24, int(base.mom_lookback + d)))))
    for d in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3):
        seeds.append(replace(
            base, mom_min_atr=max(0.25, min(2.0, round(base.mom_min_atr + d, 2))),
        ))
    for d in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
        seeds.append(replace(
            base, trail_atr_multiple=max(1.25, min(4.0, round(base.trail_atr_multiple + d, 2))),
        ))
    for d in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
        seeds.append(replace(
            base, initial_stop_atr=max(1.5, min(4.0, round(base.initial_stop_atr + d, 2))),
        ))
    for d in (-6, -4, -2, 2, 4, 6):
        seeds.append(replace(base, time_stop_bars=max(4, min(48, int(base.time_stop_bars + d)))))
    for d in (-2, -1, 1, 2, 3):
        seeds.append(replace(base, cooldown_bars=max(0, min(12, int(base.cooldown_bars + d)))))
    for d in (-0.02, -0.01, 0.01, 0.02, 0.03):
        rp = max(0.02, min(0.05, round(base.risk_pct + d, 3)))
        seeds.append(replace(base, risk_pct=rp, initial_margin_pct=max(0.16, min(0.80, rp * 8))))
    for d in (-0.5, -0.25, 0.25, 0.5):
        seeds.append(replace(base, partial_at_r=max(0.5, min(3.0, round(base.partial_at_r + d, 2)))))
    # Filter flips — often the only way out of a stuck peak
    seeds.append(replace(base, require_btc_regime=not bool(base.require_btc_regime)))
    seeds.append(replace(base, require_rising_adx=not bool(base.require_rising_adx)))
    seeds.append(replace(base, use_asymmetric_trail=not bool(base.use_asymmetric_trail)))
    # Leverage dial (aggressive only) — 5/7/10 in polish; 15/25 in breakout
    if bool(getattr(base, "aggressive_mode", False)):
        for lev in HUNT_LEV_POLISH:
            if lev == int(base.leverage):
                continue
            tf = base.timeframe
            if lev >= 25:
                tf = "5m" if str(base.timeframe) == "15m" else base.timeframe
            elif lev >= 10 and str(base.timeframe) == "15m":
                tf = base.timeframe  # 10x may stay 15m; breakout samples 5m
            seeds.append(replace(
                base, leverage=lev, timeframe=str(tf),
                max_account_leverage=float(max(lev, float(base.max_account_leverage))),
                max_net_leverage=float(max(lev, float(base.max_net_leverage))),
            ))
        for n_open in (2, 3, 4, 5, 6):
            if n_open != int(base.max_open_positions):
                seeds.append(replace(base, max_open_positions=n_open))
        for h4s in (12.0, 15.0, 18.0, 20.0, 22.0):
            seeds.append(replace(base, h4_adx_min_short=h4s))
        for ema_e in (12, 20, 34):
            seeds.append(replace(base, ema_entry_period=ema_e))
        for pb in (3, 4, 6):
            seeds.append(replace(base, pullback_lookback=pb))
        for badx in (0.0, 14.0, 18.0):
            seeds.append(replace(base, btc_adx_min=badx))
        for br in (0.0, 0.3, 0.5):
            seeds.append(replace(base, breadth_min=br))
    for lb in (-2, -1, 1, 2):
        for thr in (-0.15, -0.1, 0.1, 0.15):
            seeds.append(replace(
                base,
                mom_lookback=max(3, min(24, int(base.mom_lookback + lb))),
                mom_min_atr=max(0.25, min(2.0, round(base.mom_min_atr + thr, 2))),
            ))
    return seeds


def _sample_cfg(base: Config, rng, *, wide: bool = False) -> Config:
    """One random candidate; wide=True stretches farther for breakout mode."""
    aggressive = bool(getattr(base, "aggressive_mode", False))
    # Density-first sampling — avoid ADX28 / squeeze / long cooldown starvation
    adx_opts = [12, 14, 16, 18, 20, 22] if wide else [12, 14, 16, 18, 20]
    vol_opts = [1.05, 1.1, 1.15, 1.2, 1.3, 1.4] if wide else [1.05, 1.1, 1.15, 1.2, 1.3]
    trail_opts = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0] if wide else [2.0, 2.5, 3.0, 3.5, 4.0]
    stop_opts = [2.0, 2.5, 3.0, 3.5, 4.0] if wide else [1.5, 2.0, 2.5, 3.0, 3.5]
    cool_opts = [0, 0, 1, 1, 2, 2, 3] if wide else [0, 1, 1, 2, 2, 3]
    partial_opts = [True, True, True, False]
    partial_r_opts = [0.5, 0.75, 1.0, 1.5, 2.0] if wide else [0.75, 1.0, 1.5]
    tstop_bars_opts = [4, 6, 8, 12, 16, 24, 32] if wide else [6, 8, 12, 16]
    risk_opts = [0.015, 0.02, 0.025, 0.03] if wide else [0.02, 0.025, 0.03, 0.04]
    mom_lb_opts = [3, 4, 6, 8, 12, 16, 20] if wide else [4, 6, 8, 12, 16]
    mom_thr_opts = [0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5] if wide else [0.35, 0.5, 0.75, 1.0, 1.25]
    bo_lb_opts = [10, 14, 20, 28, 36] if wide else [14, 20, 28]
    bo_buf_opts = [0.0, 0.1, 0.2, 0.35, 0.5] if wide else [0.0, 0.15, 0.3]
    loser_opts = [1.25, 1.5, 2.0, 2.5, 3.0]
    h4_adx_opts = [10.0, 12.0, 15.0, 18.0, 20.0, 22.0, 25.0]
    lev_opts = list(HUNT_LEV_BREAKOUT if wide else HUNT_LEV_POLISH) if aggressive else [int(base.leverage)]
    open_opts = [4, 5, 6, 7, 8] if aggressive else [int(base.max_open_positions)]
    strat_opts = list(HUNT_STRATEGIES) if (aggressive and wide) else (
        list(HUNT_STRATEGIES) if (aggressive and rng.random() < 0.45) else
        [(base.strategy or "trend_mtf").strip().lower()]
    )
    adx = float(rng.choice(adx_opts))
    risk = float(rng.choice(risk_opts))
    lev = int(rng.choice(lev_opts))
    n_open = int(rng.choice(open_opts))
    strat = str(rng.choice(strat_opts))
    # Prefer faster clocks so more setups exist per day — never 3m (1m resample).
    if lev >= 15:
        tf = "5m"
    elif lev >= 10:
        tf = str(rng.choice(["5m", "5m", "15m"]))
    else:
        tf = str(rng.choice(["5m", "15m", "15m"])) if wide else (
            str(base.timeframe) if str(base.timeframe) in HUNT_TF_ALLOW else "15m"
        )
        if rng.random() < 0.35:
            tf = "5m"
    return replace(
        base,
        strategy=strat,
        timeframe=tf,
        adx_enter=adx,
        adx_exit=max(10.0, adx - float(rng.choice([4, 5, 6, 8, 10]))),
        volume_multiplier=float(rng.choice(vol_opts)),
        trail_atr_multiple=float(rng.choice(trail_opts)),
        trail_atr_loser=float(rng.choice(loser_opts)),
        use_asymmetric_trail=bool(rng.choice([True, True, False])),
        initial_stop_atr=float(rng.choice(stop_opts)),
        cooldown_bars=int(rng.choice(cool_opts)),
        require_rising_adx=False if rng.random() < 0.85 else True,
        require_squeeze=False if rng.random() < 0.9 else True,
        require_rsi_veto=False if rng.random() < 0.85 else True,
        require_btc_regime=bool(rng.choice([True, False, False])),
        breakout_lookback=int(rng.choice(bo_lb_opts)),
        breakout_buffer_atr=float(rng.choice(bo_buf_opts)),
        h4_adx_min_short=float(rng.choice(h4_adx_opts)),
        h4_ema_fast=int(rng.choice([12, 20, 34])),
        h4_ema_slow=int(rng.choice([40, 50, 80])),
        ema_entry_period=int(rng.choice([12, 20, 34])),
        pullback_lookback=int(rng.choice([3, 4, 6, 8])),
        btc_adx_min=float(rng.choice([0.0, 0.0, 0.0, 14.0])),
        breadth_min=float(rng.choice([0.0, 0.0, 0.0, 0.3])),
        enable_partial=bool(rng.choice(partial_opts)),
        partial_at_r=float(rng.choice(partial_r_opts)),
        partial_pct=0.50,
        enable_time_stop=True,
        time_stop_bars=int(rng.choice(tstop_bars_opts)),
        time_stop_min_r=1.0,
        risk_pct=risk,
        initial_margin_pct=max(0.16, min(0.80, risk * 8)),
        mom_lookback=int(rng.choice(mom_lb_opts)),
        mom_min_atr=float(rng.choice(mom_thr_opts)),
        leverage=lev,
        max_account_leverage=float(max(lev, float(base.max_account_leverage))),
        max_net_leverage=float(max(lev, float(base.max_net_leverage))),
        max_open_positions=n_open,
    )


def _explore_cfgs(base: Config, generation: int) -> list:
    """Wider random sample — escape local peaks (fast batches)."""
    rng = np.random.default_rng(generation * 9973 + 17)
    seeds = [base]
    for _ in range(int(HUNT_EXPLORE_N)):
        seeds.append(_sample_cfg(base, rng, wide=False))
    seeds.extend(_polish_cfgs(base)[:10])
    return seeds


def _breakout_cfgs(base: Config, generation: int) -> list:
    """Large jumps when polish/explore stall — fast batches."""
    rng = np.random.default_rng(generation * 7919 + 101)
    seeds = [base]
    for _ in range(int(HUNT_BREAKOUT_N)):
        seeds.append(_sample_cfg(base, rng, wide=True))
    # Explicit filter matrix around current best (bias rising-ADX OFF for 15m)
    for btc in (True, False):
        for rise in (False, True):
            for asym in (True, False):
                seeds.append(replace(
                    base,
                    require_btc_regime=btc,
                    require_rising_adx=rise,
                    use_asymmetric_trail=asym,
                ))
    seeds.extend(_polish_cfgs(base)[:8])
    return seeds


def _tf_for_leverage(lev: int, base_tf: str) -> str:
    """Pair high leverage with faster clocks so 25x is actually testable.

    Hunt never uses 3m (1m-resample cost). 15/25× → 5m; ≤10× keep base TF.
    """
    lev = int(lev)
    bt = str(base_tf or "15m").strip().lower()
    if bt not in HUNT_TF_ALLOW:
        bt = "15m"
    if lev >= 15:
        return "5m"
    return bt


def _leverage_ladder_cfgs(base: Config) -> list:
    """Explicit 5x/10x/15x/25x clones of the *current* book — one cfg per lev."""
    seeds = []
    for lev in HUNT_LEV_FORCE:
        tf = _tf_for_leverage(int(lev), str(base.timeframe))
        seeds.append(replace(
            base,
            leverage=int(lev),
            timeframe=tf,
            max_account_leverage=float(max(float(lev), float(base.max_account_leverage))),
            max_net_leverage=float(max(float(lev), float(getattr(base, "max_net_leverage", lev)))),
        ))
    return seeds


def _candidate_cfgs(base: Config, generation: int = 0, hunt: bool = True,
                    search_mode: str = "polish") -> list:
    """
    polish: nudges around current best.
    explore: wider random + local ring.
    breakout: large jumps across filters/exits when held forever.

    Leverage bake-off runs once after soft-check (not duplicated into this list).
    """
    mode = (search_mode or "polish").strip().lower()
    if not hunt:
        mode = "polish"
    if mode == "breakout":
        seeds = _breakout_cfgs(base, generation)
    elif mode == "explore":
        seeds = _explore_cfgs(base, generation)
    else:
        seeds = _polish_cfgs(base)
    return _dedupe_cfgs(base, seeds)


def _feature_cache_key(cfg: Config) -> tuple:
    """Knobs that change build_features / prepare_data — exit-only knobs excluded."""
    return (
        str(cfg.timeframe),
        int(cfg.mom_lookback),
        int(cfg.adx_period),
        float(cfg.adx_enter),
        float(cfg.adx_exit),
        int(cfg.ema_htf_period),
        int(cfg.ema_daily_period),
        int(cfg.h4_ema_fast),
        int(cfg.h4_ema_slow),
        int(cfg.ema_entry_period),
        int(cfg.pullback_lookback),
        int(cfg.breakout_lookback),
        int(cfg.volume_lookback),
        int(cfg.atr_period),
        int(cfg.rsi_period),
        int(cfg.bb_period),
        float(cfg.bb_std),
        int(cfg.kc_period),
        float(cfg.kc_atr_mult),
        int(cfg.squeeze_lookback),
        bool(cfg.require_btc_regime),
    )


def _prepare_cached(raw: dict, cfg: Config, cache: dict) -> dict:
    key = _feature_cache_key(cfg)
    hit = cache.get(key)
    if hit is not None:
        return hit
    data = prepare_data(raw, cfg)
    cache[key] = data
    return data


def _hard_promising(ci: BacktestResult, co: BacktestResult,
                    best_i: BacktestResult, best_o: BacktestResult,
                    best_score: int, *, slack: int = 2) -> bool:
    """True if hard book is worth the soft-run cost."""
    if not _is_screenable(ci, co):
        return False
    g = _grade(ci, co, None)
    cand_score = int(g.get("score_raw", g["score"]))
    if cand_score >= int(best_score) - int(slack):
        return True
    # Relative: better (or less bad) OOS than incumbent — critical on losing regimes
    if co.total_return >= best_o.total_return - 0.02:
        return True
    if co.expectancy_r >= best_o.expectancy_r - 0.05:
        return True
    if co.win_rate >= best_o.win_rate - 0.05:
        return True
    if _rank_key(ci, co, None) > _rank_key(best_i, best_o, None):
        return True
    return False


def _eval_cfg(data: dict, cfg: Config, split, *, end=None) -> tuple:
    """IS/OOS on the fit window only (end defaults to full data; pass holdout_start)."""
    run_cfg = _lev_slip_stress(cfg)
    label = (run_cfg.enable_pyramiding, run_cfg.enable_partial, run_cfg.leverage)
    fit_end = end
    ins_end = split if fit_end is None else min(pd.Timestamp(split), pd.Timestamp(fit_end))
    ins = BacktestEngine(data, run_cfg, end=ins_end).run(label)
    oos = BacktestEngine(data, run_cfg, start=split, end=fit_end).run(label)
    return ins, oos


def _eval_soft_oos(data: dict, cfg: Config, split, *, end=None) -> BacktestResult:
    """OOS with kills off — exposes configs that only 'win' by shutting down early."""
    soft = _soft_kill_cfg(_lev_slip_stress(cfg))
    label = (soft.enable_pyramiding, soft.enable_partial, soft.leverage)
    return BacktestEngine(data, soft, start=split, end=end).run(label)


# ------------------------------------------------------------
# Walk-forward: judge a config on history the hunt NEVER saw.
# Depth comes from HUNT_WF_DAYS (calendar) via _hunt_fetch_wf_raw.
# ------------------------------------------------------------
WF_FOLDS = 3
WF_WARMUP_DAYS = 60              # skip indicator warm-up at the very start
WF_MIN_FOLD_TRADES = 8
WF_MIN_FOLDS_USED = 2
WF_WORST_FOLD_FLOOR = -0.15      # no unseen fold may lose more than 15%
# Legacy bar map (unused for fetch; kept so old notebooks don't NameError).
WF_BARS = {"4h": 16000, "15m": 20000, "5m": 36000, "3m": 45000}


def _wf_eval(wf_data: dict, cfg: Config, hunt_start, cache_note: str = "") -> dict:
    """
    Run the config (kills OFF = honest) over consecutive folds that end where the
    hunt window begins. Pass = positive trade-weighted expectancy, majority of
    folds green, worst fold not a disaster.
    """
    soft = _soft_kill_cfg(_lev_slip_stress(cfg))
    label = (soft.enable_pyramiding, soft.enable_partial, soft.leverage)
    t_first = min(sd.bars[0].Index for sd in wf_data.values())
    a0 = t_first + pd.Timedelta(days=WF_WARMUP_DAYS)
    edges = pd.date_range(a0, hunt_start, periods=WF_FOLDS + 1)
    folds = []
    for k in range(WF_FOLDS):
        a, b = edges[k], edges[k + 1]
        res = BacktestEngine(wf_data, soft, start=a, end=b).run(label)
        folds.append({
            "from": f"{a:%Y-%m-%d}", "to": f"{b:%Y-%m-%d}",
            "trades": int(res.trades),
            "exp": round(float(res.expectancy_r), 3),
            "ret": round(float(res.total_return), 3),
            "win": round(float(res.win_rate), 3),
        })
    used = [f for f in folds if f["trades"] >= WF_MIN_FOLD_TRADES]
    n_tot = sum(f["trades"] for f in used)
    w_exp = (sum(f["exp"] * f["trades"] for f in used) / n_tot) if n_tot else 0.0
    green = sum(1 for f in used if f["ret"] > 0)
    worst = min((f["ret"] for f in used), default=0.0)
    ok = bool(
        len(used) >= WF_MIN_FOLDS_USED
        and w_exp > 0
        and green * 3 >= len(used) * 2          # ≥ 2/3 of used folds green
        and worst >= WF_WORST_FOLD_FLOOR
    )
    if len(used) < WF_MIN_FOLDS_USED:
        why = f"only {len(used)} fold(s) had ≥{WF_MIN_FOLD_TRADES} trades"
    elif w_exp <= 0:
        why = f"unseen avg profit {w_exp:+.2f}R ≤ 0"
    elif green * 3 < len(used) * 2:
        why = f"only {green}/{len(used)} unseen periods profitable"
    elif worst < WF_WORST_FOLD_FLOOR:
        why = f"worst unseen period {worst:+.0%}"
    else:
        why = "passed"
    return {
        "ok": ok, "why": why, "folds": folds, "trades": int(n_tot),
        "exp": round(float(w_exp), 3), "green": f"{green}/{len(used)}",
        "worst": round(float(worst), 3),
        "span": f"{a0:%Y-%m-%d} → {pd.Timestamp(hunt_start):%Y-%m-%d}",
    }


FWD_MIN_TRADES = 20      # closed forward paper trades required before the score can exceed 85
FWD_SCORE_CAP = 85

# Recent-window signal (not a hard promote veto while the book is climbing)
RECENT_DAYS = 90
RECENT_MIN_TRADES = 12
RECENT_EXP_FLOOR = -0.25   # climb-friendly; deep bleed still flagged, mild drag OK

# Primary money holdout (calendar days frozen out of fit / scored for promote).
# 14d > old 7d — more trades / power before research promote.
WEEK_DAYS = 14


def _window_eval(data: dict, cfg: Config, days: int,
                 end: Optional[pd.Timestamp] = None, *, soft: bool = True,
                 min_trades: int = 0, exp_floor: float = -99.0) -> dict:
    """Hard or soft backtest over the last `days` ending at `end`."""
    if end is None:
        end = max(sd.bars[-1].Index for sd in data.values())
    start = pd.Timestamp(end) - pd.Timedelta(days=int(days))
    run_cfg = _soft_kill_cfg(cfg) if soft else cfg
    try:
        lab = (
            bool(run_cfg.enable_pyramiding),
            bool(run_cfg.enable_partial),
            int(run_cfg.leverage),
        )
        res = BacktestEngine(data, run_cfg, start=start, end=end).run(lab)
    except Exception as e:
        return {
            "ok": False, "why": f"{days}d eval failed: {e}",
            "trades": 0, "exp": 0.0, "ret": 0.0, "final": 0.0,
            "span": f"{start:%Y-%m-%d} → {pd.Timestamp(end):%Y-%m-%d}",
            "days": int(days), "soft": bool(soft),
        }
    thin = int(min_trades) > 0 and res.trades < int(min_trades)
    ok = (not thin) and float(res.expectancy_r) >= float(exp_floor)
    why = (
        "passed" if ok else
        (f"thin {days}d sample n={res.trades}" if thin
         else f"{days}d exp {res.expectancy_r:+.2f}R < {exp_floor}")
    )
    return {
        "ok": ok, "why": why,
        "trades": int(res.trades),
        "exp": round(float(res.expectancy_r), 3),
        "ret": round(float(res.total_return), 3),
        "final": round(float(res.final_equity), 2),
        "win": round(float(res.win_rate), 3),
        "span": f"{start:%Y-%m-%d} → {pd.Timestamp(end):%Y-%m-%d}",
        "days": int(days), "soft": bool(soft),
    }


def _recent_eval(data: dict, cfg: Config, end: Optional[pd.Timestamp] = None) -> dict:
    """Soft (kills off) expectancy over the last RECENT_DAYS ending at `end`."""
    return _window_eval(
        data, cfg, RECENT_DAYS, end=end, soft=True,
        min_trades=RECENT_MIN_TRADES, exp_floor=RECENT_EXP_FLOOR,
    )


def _week_eval(data: dict, cfg: Config, end: Optional[pd.Timestamp] = None) -> dict:
    """PRIMARY: hard + soft P&L over the last WEEK_DAYS calendar days."""
    week_min = int(HUNT_TARGETS.get("week_min_trades", 2))
    hard = _window_eval(
        data, cfg, WEEK_DAYS, end=end, soft=False,
        min_trades=week_min, exp_floor=-0.50,
    )
    soft = _window_eval(
        data, cfg, WEEK_DAYS, end=end, soft=True,
        min_trades=0, exp_floor=-99.0,
    )
    week_tgt = float(HUNT_TARGETS.get("week_return", 1.0))
    met = bool(hard.get("ok") and float(hard.get("ret") or 0) >= week_tgt)
    return {
        "ok": bool(hard.get("ok")),
        "met": met,
        "why": hard.get("why"),
        "ret": hard.get("ret"),
        "soft_ret": soft.get("ret"),
        "final": hard.get("final"),
        "soft_final": soft.get("final"),
        "trades": hard.get("trades"),
        "soft_trades": soft.get("trades"),
        "exp": hard.get("exp"),
        "win": hard.get("win"),
        "span": hard.get("span"),
        "goal": week_tgt,
        "days": WEEK_DAYS,
    }


def _forward_record(symbols: list, since_iso: str) -> dict:
    """Closed PAPER trades since the strategy went live in paper — the only proof that isn't a backtest."""
    n, pnl, wins = 0, 0.0, 0
    try:
        conn = sqlite3.connect(str(DB_PATH))
        try:
            q = "SELECT pnl FROM trades WHERE mode='paper' AND ts >= ?"
            args: list = [since_iso]
            if symbols:
                q += " AND symbol IN (%s)" % ",".join("?" * len(symbols))
                args += list(symbols)
            for (p,) in conn.execute(q, args):
                n += 1
                pnl += float(p)
                wins += 1 if float(p) > 0 else 0
        finally:
            conn.close()
    except Exception as e:
        log.warning("forward record unavailable: %s", e)
    return {
        "n": n, "pnl": round(pnl, 2), "win": round(wins / n, 3) if n else 0.0,
        "need": FWD_MIN_TRADES, "since": since_iso,
        "ok": bool(n >= FWD_MIN_TRADES and pnl > 0),
    }


def _wf_msg(wf: Optional[dict]) -> str:
    if not wf:
        return "unseen-history check not run"
    verdict = "PASS" if wf.get("ok") else "FAIL"
    return (
        f"unseen {wf.get('span','')} {verdict} ({wf.get('green','?')} periods green, "
        f"{wf.get('exp', 0):+.2f}R avg, {wf.get('trades', 0)} trades"
        + ("" if wf.get("ok") else f" — {wf.get('why','')}") + ")"
    )


def _score(ins: BacktestResult, oos: BacktestResult,
           soft_oos: Optional[BacktestResult] = None) -> float:
    """Tie-break scalar only — display / upgrades use _grade()['score']."""
    min_test = int(HUNT_TARGETS.get("test_n", 40))
    min_train = int(HUNT_TARGETS.get("train_n", 40))
    if oos.trades < min_test or ins.trades < min_train:
        return -1e9
    if ins.expectancy_r <= 0 or oos.expectancy_r <= 0:
        return -1e9
    if oos.total_return <= 0 or ins.total_return <= 0:
        return -1e9
    soft_floor = float(HUNT_TARGETS.get("soft_return_floor", -0.10))
    soft_ret = float(soft_oos.total_return) if soft_oos is not None else float(oos.total_return)
    if soft_ret < soft_floor:
        return -1e9
    hard_ret = float(oos.total_return)
    hard_halted = bool((oos.halted or "").strip())
    # Kill illusion: hard looks rich only because it quit early while soft bleeds.
    illusion_pen = 0.0
    if hard_halted and soft_ret < hard_ret - 0.05:
        illusion_pen = min(8.0, (hard_ret - soft_ret) * 12.0)
    # Over-killed: soft edge is real but kills crushed hard — mild penalty so hunt
    # prefers books that still print with rails on (without rewarding illusion).
    overkill_pen = 0.0
    if hard_halted and soft_ret > hard_ret + 0.10:
        overkill_pen = min(3.0, (soft_ret - hard_ret) * 4.0)
    n_bonus = min(oos.trades, 160) / 160.0
    win_tgt = float(HUNT_TARGETS.get("win_rate", 0.50))
    win_bonus = float(oos.win_rate) / win_tgt if win_tgt else float(oos.win_rate)
    ret_tgt = float(HUNT_TARGETS.get("test_return", 0.25))
    ret_bonus = hard_ret / ret_tgt if ret_tgt else hard_ret
    # Dollar growth dominates — ~100%/week path is return + soft edge, not tidy win-rate.
    return (
        float(oos.expectancy_r) * 1.2
        + float(ins.expectancy_r) * 0.4
        + hard_ret * 16.0
        + float(ins.total_return) * 2.5
        + float(soft_ret) * 5.0
        + min(hard_ret, soft_ret) * 4.0  # reward equity that survives both clocks
        - float(oos.max_drawdown) * 0.8
        + n_bonus * 0.12
        + win_bonus * 0.7
        + ret_bonus * 6.0
        + float(oos.win_rate) * 0.15
        - illusion_pen
        - overkill_pen
    )


def _rank_key(ins: BacktestResult, oos: BacktestResult,
              soft_oos: Optional[BacktestResult] = None,
              week: Optional[dict] = None) -> tuple:
    """
    Primary sort = quality /100 the user sees (now includes $ return + soft-run).
    Holdout week return is part of the key when supplied (F1).
    """
    g = _grade(ins, oos, soft_oos, week=week)
    soft_ret = float(soft_oos.total_return) if soft_oos is not None else float(oos.total_return)
    week_ret = float((week or {}).get("ret") or -999.0)
    return (
        int(g.get("score_raw", g["score"])),  # hunt on uncapped research quality
        week_ret,
        float(oos.total_return),
        soft_ret,
        float(oos.win_rate),
        float(oos.expectancy_r),
        int(oos.trades),
        -float(oos.max_drawdown),
        _score(ins, oos, soft_oos),
    )


def _beats(a_i: BacktestResult, a_o: BacktestResult,
           b_i: BacktestResult, b_o: BacktestResult,
           a_soft: Optional[BacktestResult] = None,
           b_soft: Optional[BacktestResult] = None,
           a_week: Optional[dict] = None,
           b_week: Optional[dict] = None) -> bool:
    """True if A is strictly better than B for hunting."""
    return _rank_key(a_i, a_o, a_soft, a_week) > _rank_key(b_i, b_o, b_soft, b_week)


def _grade(ins: BacktestResult, oos: BacktestResult,
           soft_oos: Optional[BacktestResult] = None,
           wf: Optional[dict] = None, wf_required: bool = False,
           fwd: Optional[dict] = None, fwd_required: bool = False,
           week: Optional[dict] = None) -> dict:
    """0–100 score vs HUNT_TARGETS + checklist for the Live page.

    Primary money bar is 7-day return when `week` is supplied.
    wf: walk-forward (unseen history) result. When required and missing/failed the
    targets cannot be met and the displayed score is capped at 60.
    """
    t = HUNT_TARGETS
    win_tgt = float(t.get("win_rate", 0.50))
    ret_tgt = float(t.get("test_return", 0.25))
    week_tgt = float(t.get("week_return", 1.0))
    soft_floor = float(t.get("soft_return_floor", -0.10))
    soft_ret = float(soft_oos.total_return) if soft_oos is not None else float(oos.total_return)
    soft_final = float(soft_oos.final_equity) if soft_oos is not None else float(oos.final_equity)
    hard_halted = bool((oos.halted or "").strip())
    hard_ret = float(oos.total_return)
    week_ret = float((week or {}).get("ret") or 0.0)
    week_soft = float((week or {}).get("soft_ret") or 0.0) if week is not None else 0.0
    week_hard_floor = float(t.get("week_hard_floor", 0.25))
    kill_illusion = (
        hard_halted and soft_oos is not None and soft_ret < hard_ret - 0.05
    )
    over_killed = (
        hard_halted and soft_oos is not None and soft_ret > hard_ret + 0.10
    )
    checks = {
        "train_exp": ins.expectancy_r >= t["train_exp"],
        "test_exp": oos.expectancy_r >= t["test_exp"],
        "train_n": ins.trades >= t["train_n"],
        "test_n": oos.trades >= t["test_n"],
        "max_dd": oos.max_drawdown <= t["max_dd"],
        "win_rate": oos.win_rate >= win_tgt,
        "test_return": hard_ret >= ret_tgt,
        "soft_ok": soft_ret >= soft_floor,
        "both_positive": (
            ins.expectancy_r > 0 and oos.expectancy_r > 0
            and ins.total_return > 0 and hard_ret > 0
        ),
    }
    if week is not None:
        checks["week_return"] = bool(week.get("ok")) and week_ret >= week_tgt
        checks["week_hard_ok"] = bool(week.get("ok")) and week_ret >= week_hard_floor
    if t.get("block_kill_illusion", True):
        checks["no_kill_illusion"] = not kill_illusion

    def _pct(val, tgt, higher_better=True):
        if higher_better:
            return float(max(0.0, min(1.0, val / tgt if tgt else 0.0)))
        return float(max(0.0, min(1.0, 1.0 - (val / tgt) if tgt else 0.0)))

    soft_span = max(1e-6, ret_tgt - soft_floor)
    soft_pct = float(max(0.0, min(1.0, (soft_ret - soft_floor) / soft_span)))
    week_pct = _pct(week_ret, week_tgt) if week is not None else 0.0
    honest_pct = _pct(min(hard_ret, soft_ret), ret_tgt) if ret_tgt else 0.0
    parts = [
        _pct(ins.expectancy_r, t["train_exp"]),
        _pct(oos.expectancy_r, t["test_exp"]),
        _pct(ins.trades, t["train_n"]),
        _pct(oos.trades, t["test_n"]),
        _pct(oos.max_drawdown, t["max_dd"], higher_better=False),
        _pct(oos.win_rate, win_tgt),
        _pct(hard_ret, ret_tgt) * 0.5,
        soft_pct * 0.5,
        honest_pct * 0.5,
        1.0 if checks["both_positive"] else 0.0,
        0.0 if kill_illusion else 1.0,
    ]
    if week is not None:
        # Week is PRIMARY for score and for targets_met (+100%/7d with leverage).
        # Density bonus — sparse ~20-trade weeks must not outrank active books.
        week_n = float(week.get("trades") or week.get("soft_trades") or 0)
        week_n_tgt = float(t.get("week_target_trades", 70))
        dens_pct = float(max(0.0, min(1.0, week_n / week_n_tgt if week_n_tgt else 0.0)))
        parts.extend([week_pct, week_pct, dens_pct, dens_pct])
    score_raw = int(round(100.0 * sum(parts) / len(parts)))
    score = score_raw
    if kill_illusion:
        score = min(score, 55)
        score_raw = min(score_raw, 70)
    wf_ok = bool(wf and wf.get("ok"))
    if wf_required:
        checks["walk_forward"] = wf_ok
        if not wf_ok:
            score = min(score, 60)
    if fwd_required:
        fwd_ok = bool(fwd and fwd.get("ok"))
        checks["forward_proof"] = fwd_ok
        if not fwd_ok:
            score = min(score, FWD_SCORE_CAP)
    met = all(checks.values())
    return {
        "score": score,
        "score_raw": score_raw,
        "targets_met": met,
        "walk_forward": wf,
        "forward": fwd,
        "checks": checks,
        "targets": t,
        "train_exp": round(ins.expectancy_r, 3),
        "test_exp": round(oos.expectancy_r, 3),
        "train_n": ins.trades,
        "test_n": oos.trades,
        "max_dd": round(oos.max_drawdown, 3),
        "win": round(oos.win_rate, 3),
        "test_return": round(hard_ret, 3),
        "train_return": round(ins.total_return, 3),
        "oos_final": round(oos.final_equity, 2),
        "train_final": round(ins.final_equity, 2),
        "soft_return": round(soft_ret, 3),
        "soft_final": round(soft_final, 2),
        "hard_halted": hard_halted,
        "halt_reason": (oos.halted or "")[:120],
        "kill_illusion": kill_illusion,
        "over_killed": over_killed,
        "soft_hard_gap": round(hard_ret - soft_ret, 3),
        "week": week,
        "week_return": round(week_ret, 3) if week is not None else None,
        "week_soft_return": round(week_soft, 3) if week is not None else None,
        "week_final": (week or {}).get("final"),
        "week_span": (week or {}).get("span"),
    }


def _is_screenable(cand_i: BacktestResult, cand_o: BacktestResult) -> bool:
    """Loose gate for hunt pool entry — enough trades to rank, not 'must already be profitable'.

    Requiring positive expectancy here emptied hard_pool on losing regimes and
    blocked soft-check / week-ladder entirely.
    """
    min_n = max(6, min(12, int(HUNT_TARGETS.get("test_n", 20)) // 2))
    if cand_o.trades < min_n or cand_i.trades < min_n:
        return False
    if cand_o.max_drawdown > 0.85:
        return False
    return True


def _is_valid_candidate(cand_i: BacktestResult, cand_o: BacktestResult,
                        soft_oos: Optional[BacktestResult] = None) -> bool:
    """Strict gate for 'healthy book' claims (elite / targets). Not used to enter soft pool."""
    min_test = int(HUNT_TARGETS.get("test_n", 40))
    min_train = int(HUNT_TARGETS.get("train_n", 40))
    if cand_o.trades < min_test or cand_i.trades < min_train:
        return False
    if cand_i.expectancy_r <= 0 or cand_o.expectancy_r <= 0:
        return False
    if cand_o.max_drawdown > 0.55:
        return False
    soft_floor = float(HUNT_TARGETS.get("soft_return_floor", -0.10))
    if soft_oos is not None and soft_oos.total_return < soft_floor:
        return False
    if (
        HUNT_TARGETS.get("block_kill_illusion", True)
        and soft_oos is not None
        and (cand_o.halted or "").strip()
        and soft_oos.total_return < cand_o.total_return - 0.05
    ):
        return False
    return True


def _is_climb_ok(cand_i: BacktestResult, cand_o: BacktestResult,
                 soft_oos: Optional[BacktestResult],
                 base_i: Optional[BacktestResult] = None,
                 base_o: Optional[BacktestResult] = None) -> bool:
    """Research-promote gate: allow relative climb from a losing baseline.

    Soft floor loosens while baseline is underwater so soft-check can feed
    _consider. Kill-illusion still blocked. Does NOT demand exp>0 while ≤0.
    """
    if not _is_screenable(cand_i, cand_o):
        return False
    baseline_losing = (
        base_i is not None and base_o is not None
        and (base_i.expectancy_r <= 0 or base_o.expectancy_r <= 0
             or base_o.total_return <= 0)
    )
    soft_floor = float(HUNT_TARGETS.get("soft_return_floor", -0.30))
    if baseline_losing:
        soft_floor = min(soft_floor, -0.60)
    if soft_oos is not None and soft_oos.total_return < soft_floor:
        return False
    if (
        HUNT_TARGETS.get("block_kill_illusion", True)
        and soft_oos is not None
        and (cand_o.halted or "").strip()
        and soft_oos.total_return < cand_o.total_return - 0.05
    ):
        return False
    if baseline_losing:
        return True
    # When baseline is healthy, still prefer non-negative OOS expectancy
    return cand_o.expectancy_r > -0.05


def _discover_seed_cfgs(base: Config, generation: int, held: int) -> list:
    """Inject momentum_rule / combo_or seeds from .moonshot_rules.json when stuck.

    Discover uses different barrier exits — we only take the strategy family tip;
    BacktestEngine re-validation in the hunt loop is the real gate.
    """
    if held < 8 and (generation % 11 != 0):
        return []
    path = RULES_PATH
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text())
    except Exception as e:
        log.warning("discover rules load failed: %s", e)
        return []
    top = list(payload.get("top") or [])[:3]
    if not top:
        return []
    seeds = []
    for i, _rule in enumerate(top):
        seeds.append(replace(
            base,
            strategy="momentum_rule" if i % 2 == 0 else "combo_or",
            timeframe=str(base.timeframe) if str(base.timeframe) in HUNT_TF_ALLOW else "15m",
        ))
    log.info("discover inject: %d seed(s) from %s", len(seeds), path.name)
    return seeds


