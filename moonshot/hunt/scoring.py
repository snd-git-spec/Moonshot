from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import replace
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.config import (
    Config,
)
from moonshot.features import (
    prepare_data,
)
from moonshot.backtest import (
    BacktestEngine,
    BacktestResult,
)
from moonshot.hunt.constants import (
    HUNT_BANLIST_CAP,
    HUNT_BAN_WEEK_FLOOR,
    HUNT_BREAKOUT_N,
    HUNT_EXPLORE_N,
    HUNT_LEV15_MIN_SOFT_TRADES,
    HUNT_LEV15_MIN_WEEK_TRADES,
    HUNT_LEV25_MIN_SOFT_TRADES,
    HUNT_LEV25_MIN_WEEK_TRADES,
    HUNT_LEV_BREAKOUT,
    HUNT_LEV_POLISH,
    HUNT_MATRIX_SOFT_CAP,
    HUNT_MATRIX_TFS,
    HUNT_STRATEGIES,
    HUNT_STUCK_HELD,
    HUNT_TARGETS,
    HUNT_TF_ALLOW,
    _lev_slip_stress,
    _utc_now,
)

from moonshot.paths import (
    DB_PATH,
    RULES_PATH,
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
    """Pin risk rails to base (A.1); filters/exits/HTF remain hunt-able.

    Leverage, risk_pct, daily-loss halt, and peak-DD kill always match ``base``.
    Lev×TF matrix helpers remain in-tree for post-KEEP research but are not
    applied here.
    """
    aggressive = bool(getattr(base, "aggressive_mode", False))
    # Strategy + TF may change in explore/breakout (Layer 4 / 4c)
    strat = str(getattr(c, "strategy", base.strategy) or base.strategy).strip().lower()
    if strat not in HUNT_STRATEGIES and strat not in (
        "momentum_rule",
        "combo_and",
        "regime_breakout",
    ):
        strat = (base.strategy or "trend_mtf").strip().lower()
    tf = str(getattr(c, "timeframe", base.timeframe) or base.timeframe).strip().lower()
    if tf not in HUNT_TF_ALLOW:
        tf = str(base.timeframe) if str(base.timeframe) in HUNT_TF_ALLOW else "15m"

    # A.1 — risk rails pinned to deployed/base book
    lev = int(base.leverage)
    risk = float(base.risk_pct)
    max_acct = float(base.max_account_leverage)
    max_net = float(getattr(base, "max_net_leverage", lev))
    max_open = int(getattr(c, "max_open_positions", getattr(base, "max_open_positions", 4)) or 4)
    if aggressive:
        max_open = int(max(1, min(8, max_open)))
        stop_atr = float(max(1.5, min(4.0, c.initial_stop_atr)))
    else:
        max_open = int(getattr(base, "max_open_positions", 3))
        stop_atr = float(max(1.5, min(4.0, c.initial_stop_atr)))
        strat = (base.strategy or "trend_mtf").strip().lower()
        tf = str(base.timeframe)

    margin = float(max(0.16, min(0.80, c.initial_margin_pct)))
    margin = max(margin, min(0.80, risk * 8.0))
    # Kill rails pinned to base (A.1)
    daily = float(base.max_daily_loss_pct)
    peak_dd = float(base.max_peak_drawdown_pct)
    exp_floor = float(getattr(base, "expectancy_floor", -0.15))
    streak_n = int(getattr(base, "streak_losses", 5))
    streak_hrs = float(getattr(base, "streak_pause_hours", 48.0))
    h4_fast = int(
        max(8, min(50, int(getattr(c, "h4_ema_fast", base.h4_ema_fast) or base.h4_ema_fast)))
    )
    h4_slow = int(
        max(
            h4_fast + 5,
            min(100, int(getattr(c, "h4_ema_slow", base.h4_ema_slow) or base.h4_ema_slow)),
        )
    )
    ema_entry = int(
        max(
            8,
            min(
                50,
                int(getattr(c, "ema_entry_period", base.ema_entry_period) or base.ema_entry_period),
            ),
        )
    )
    pullback_lb = int(
        max(
            2,
            min(
                12,
                int(
                    getattr(c, "pullback_lookback", base.pullback_lookback)
                    or base.pullback_lookback
                ),
            ),
        )
    )
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
        require_h4_bias=bool(getattr(c, "require_h4_bias", False)),
        require_daily_align=bool(getattr(c, "require_daily_align", False)),
        require_weekly_bias=bool(getattr(c, "require_weekly_bias", False)),
        ema_weekly_period=int(
            max(2, min(30, int(getattr(c, "ema_weekly_period", 10) or 10)))
        ),
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


def knobs_to_cfg(kn: dict, base: Optional[Config] = None) -> Config:
    """Rebuild a Config from hunt/champion knobs (shared by hunt + Live UI).

    Uses matrix freeze so strategy/timeframe/leverage from ``kn`` survive —
    plain ``_freeze_risk`` pinned lev (and non-aggressive strat/TF) to base,
    which silently turned elite 7×@15m deploys into 10×@5m.
    """
    base = base or Config.from_env()
    kn = kn or {}
    return _freeze_risk_matrix(
        base,
        replace(
            base,
            strategy=str(kn.get("strategy", base.strategy)),
            timeframe=str(kn.get("timeframe", base.timeframe)),
            adx_enter=float(kn["adx_enter"]),
            adx_exit=float(kn.get("adx_exit", float(kn["adx_enter"]) - 6)),
            volume_multiplier=float(kn["volume_multiplier"]),
            breakout_buffer_atr=float(kn.get("breakout_buffer_atr", base.breakout_buffer_atr)),
            breakout_lookback=int(kn.get("breakout_lookback", base.breakout_lookback)),
            trail_atr_multiple=float(kn.get("trail_atr", base.trail_atr_multiple)),
            initial_stop_atr=float(kn.get("initial_stop_atr", base.initial_stop_atr)),
            cooldown_bars=int(kn.get("cooldown_bars", base.cooldown_bars)),
            require_rising_adx=bool(kn.get("require_rising_adx", base.require_rising_adx)),
            require_squeeze=bool(kn.get("require_squeeze", base.require_squeeze)),
            require_rsi_veto=bool(kn.get("require_rsi_veto", base.require_rsi_veto)),
            require_btc_regime=bool(kn.get("require_btc_regime", base.require_btc_regime)),
            require_h4_bias=bool(kn.get("require_h4_bias", base.require_h4_bias)),
            require_daily_align=bool(kn.get("require_daily_align", base.require_daily_align)),
            require_weekly_bias=bool(kn.get("require_weekly_bias", base.require_weekly_bias)),
            ema_weekly_period=int(kn.get("ema_weekly_period", base.ema_weekly_period)),
            rsi_long_max=float(kn.get("rsi_long_max", base.rsi_long_max)),
            rsi_short_min=float(kn.get("rsi_short_min", base.rsi_short_min)),
            squeeze_lookback=int(kn.get("squeeze_lookback", base.squeeze_lookback)),
            enable_partial=bool(kn.get("enable_partial", base.enable_partial)),
            partial_at_r=float(kn.get("partial_at_r", base.partial_at_r)),
            partial_pct=float(kn.get("partial_pct", base.partial_pct)),
            enable_time_stop=bool(kn.get("enable_time_stop", base.enable_time_stop)),
            time_stop_bars=int(kn.get("time_stop_bars", base.time_stop_bars)),
            time_stop_min_r=float(kn.get("time_stop_min_r", base.time_stop_min_r)),
            trail_atr_loser=float(kn.get("trail_atr_loser", base.trail_atr_loser)),
            use_asymmetric_trail=bool(kn.get("use_asymmetric_trail", base.use_asymmetric_trail)),
            risk_pct=float(kn.get("risk_pct", base.risk_pct)),
            initial_margin_pct=float(kn.get("initial_margin_pct", base.initial_margin_pct)),
            aggressive_margin_pct=float(
                kn.get("aggressive_margin_pct", base.aggressive_margin_pct)
            ),
            max_daily_loss_pct=float(kn.get("max_daily_loss_pct", base.max_daily_loss_pct)),
            max_peak_drawdown_pct=float(
                kn.get("max_peak_drawdown_pct", base.max_peak_drawdown_pct)
            ),
            mom_lookback=int(kn.get("mom_lookback", base.mom_lookback)),
            mom_min_atr=float(kn.get("mom_min_atr", base.mom_min_atr)),
            leverage=int(kn.get("leverage", base.leverage)),
            max_account_leverage=float(
                kn.get("max_account_leverage", kn.get("leverage", base.max_account_leverage))
            ),
            max_net_leverage=float(
                kn.get("max_net_leverage", kn.get("leverage", base.max_net_leverage))
            ),
            max_open_positions=int(kn.get("max_open_positions", base.max_open_positions)),
            h4_ema_fast=int(kn.get("h4_ema_fast", base.h4_ema_fast)),
            h4_ema_slow=int(kn.get("h4_ema_slow", base.h4_ema_slow)),
            ema_entry_period=int(kn.get("ema_entry_period", base.ema_entry_period)),
            pullback_lookback=int(kn.get("pullback_lookback", base.pullback_lookback)),
            btc_adx_min=float(kn.get("btc_adx_min", base.btc_adx_min)),
            breadth_min=float(kn.get("breadth_min", base.breadth_min)),
            starting_equity=float(
                getattr(base, "research_starting_equity", base.starting_equity)
                or base.starting_equity
            ),
        ),
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


# Legacy settle lock (0 = off). Non-escape dwell uses RESEARCH_PROMOTE_DWELL_SEC.
RESEARCH_PROMOTE_MIN_SEC = 0
# Tiny material floors for in-memory climb; promote hysteresis uses ≥2pp below.
RESEARCH_MIN_WEEK_EDGE = 0.005
RESEARCH_MIN_SCORE_EDGE = 1
RESEARCH_MIN_OOS_RET_EDGE = 0.008
POST_PROMOTE_HUNT_PAUSE_SEC = 5
# Product A anti-flap (non-escape research only; escape/rescue bypass).
RESEARCH_PROMOTE_DWELL_SEC = 24 * 3600
RESEARCH_HYSTERESIS_WEEK_PP = 0.02
RESEARCH_HYSTERESIS_PASSES = 2
RESEARCH_LEV_UP_WEEK_PP = 0.05
HUNT_EXHAUSTED_GENS = 50
PAPER_RETIRE_MIN_N = 40
PAPER_RETIRE_REARM_SEC = 24 * 3600
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
        last = last.tz_localize("UTC") if last.tzinfo is None else last.tz_convert("UTC")
        return float((pd.Timestamp.now(tz="UTC") - last).total_seconds() / 3600.0)
    except Exception:
        return 0.0


def _paper_bleed(n: int | None = None) -> dict:
    """True when the last N closed paper trades are all losers (live book is dying).

    After a hunt promote clears the circuit, only closes *after* that deploy count
    — otherwise the old losing streak would re-arm the halt immediately.
    """
    from moonshot.hunt.constants import PAPER_BLEED_N, _load_autopilot_state

    need = int(n if n is not None else PAPER_BLEED_N)
    out = {"bleed": False, "n": 0, "losses": 0, "pnl": 0.0, "need": need}
    since = None
    try:
        st = _load_autopilot_state()
        since = st.get("paper_bleed_window_start") or None
    except Exception:
        since = None
    try:
        con = sqlite3.connect(str(DB_PATH))
        try:
            if since:
                rows = con.execute(
                    "SELECT pnl FROM trades WHERE mode = 'paper' AND ts > ? "
                    "ORDER BY ts DESC LIMIT ?",
                    (str(since), need),
                ).fetchall()
            else:
                rows = con.execute(
                    "SELECT pnl FROM trades WHERE mode = 'paper' ORDER BY ts DESC LIMIT ?",
                    (need,),
                ).fetchall()
        finally:
            con.close()
        if not rows:
            return out
        pnls = [float(r[0] or 0.0) for r in rows]
        out["n"] = len(pnls)
        out["losses"] = sum(1 for p in pnls if p <= 0)
        out["pnl"] = round(sum(pnls), 2)
        out["bleed"] = bool(len(pnls) >= need and all(p <= 0 for p in pnls))
    except Exception:
        return out
    return out


def _cfg_fingerprint(cfg: Config) -> tuple:
    """Stable identity of tradable knobs — used to skip no-op promotes."""
    return (
        str(cfg.strategy),
        str(cfg.timeframe),
        int(cfg.leverage),
        round(float(cfg.adx_enter), 3),
        round(float(cfg.adx_exit), 3),
        round(float(cfg.volume_multiplier), 3),
        round(float(cfg.breakout_buffer_atr), 3),
        int(cfg.breakout_lookback),
        round(float(cfg.trail_atr_multiple), 3),
        round(float(cfg.trail_atr_loser), 3),
        round(float(cfg.initial_stop_atr), 3),
        int(cfg.cooldown_bars),
        bool(cfg.require_rising_adx),
        bool(cfg.require_squeeze),
        bool(cfg.require_rsi_veto),
        bool(cfg.require_btc_regime),
        bool(getattr(cfg, "require_h4_bias", False)),
        bool(getattr(cfg, "require_daily_align", False)),
        bool(getattr(cfg, "require_weekly_bias", False)),
        bool(cfg.use_asymmetric_trail),
        bool(cfg.enable_partial),
        bool(cfg.enable_time_stop),
        int(cfg.time_stop_bars),
        int(cfg.mom_lookback),
        round(float(cfg.mom_min_atr), 3),
        int(cfg.h4_ema_fast),
        int(cfg.h4_ema_slow),
        round(float(cfg.btc_adx_min), 3),
        round(float(cfg.breadth_min), 3),
    )


def _ban_key(cfg: Config) -> str:
    """Stable string key for loser banlist (same identity as fingerprint)."""
    return repr(_cfg_fingerprint(cfg))


def _banlist_set(state: dict) -> set[str]:
    rows = state.get("banlist") or []
    out: set[str] = set()
    if not isinstance(rows, list):
        return out
    for row in rows:
        if isinstance(row, dict) and row.get("hash"):
            out.add(str(row["hash"]))
        elif isinstance(row, str):
            out.add(row)
    return out


def _banlist_add(state: dict, cfg: Config, *, week_ret: float, reason: str) -> None:
    """Append a loser; FIFO cap HUNT_BANLIST_CAP."""
    key = _ban_key(cfg)
    rows = list(state.get("banlist") or [])
    if not isinstance(rows, list):
        rows = []
    # de-dupe: refresh existing
    rows = [r for r in rows if not (isinstance(r, dict) and str(r.get("hash")) == key)]
    rows.append(
        {
            "hash": key,
            "week_ret": round(float(week_ret), 4),
            "reason": str(reason)[:120],
            "ts": _utc_now().isoformat(),
        }
    )
    if len(rows) > int(HUNT_BANLIST_CAP):
        rows = rows[-int(HUNT_BANLIST_CAP) :]
    state["banlist"] = rows


def _should_ban_week(week: dict | None, *, kill_illusion: bool = False) -> tuple[bool, str]:
    if kill_illusion:
        return True, "kill_illusion"
    if not week:
        return False, ""
    try:
        wr = float(week.get("ret") or 0)
    except (TypeError, ValueError):
        return False, ""
    if wr < float(HUNT_BAN_WEEK_FLOOR):
        return True, f"week<{HUNT_BAN_WEEK_FLOOR:.0%}"
    return False, ""


def _filter_banned(cfgs: list, banned: set[str]) -> tuple[list, int]:
    if not banned:
        return list(cfgs), 0
    kept, n = [], 0
    for c in cfgs:
        if _ban_key(c) in banned:
            n += 1
            continue
        kept.append(c)
    return kept, n


def _champion_from_history(hist_rows: list, current: dict | None) -> dict | None:
    """Keep the highest holdout-week knobs ever seen — never let a wipe lower the floor."""
    best = dict(current) if isinstance(current, dict) and current.get("knobs") else None
    best_w = float((best or {}).get("week_ret_peak") or (best or {}).get("week_ret") or -999)
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
                "week_ret_peak": wr_f,
                "score": row.get("score"),
                "knobs": kn,
                "week": row.get("week") if isinstance(row.get("week"), dict) else None,
            }
    return best


def _resolve_champion_peak(state: dict) -> dict | None:
    """Peak holdout champion — prefer champion_peak, else champion (+ history raise)."""
    peak = state.get("champion_peak") if isinstance(state.get("champion_peak"), dict) else None
    champ = state.get("champion") if isinstance(state.get("champion"), dict) else None
    hist = list(state.get("history") or [])
    merged = _champion_from_history(hist, peak or champ)
    if not merged:
        return None
    # Preserve highest peak ever recorded on the blob
    peak_w = float(merged.get("week_ret_peak") or merged.get("week_ret") or -999)
    for src in (peak, champ):
        if not isinstance(src, dict):
            continue
        try:
            peak_w = max(peak_w, float(src.get("week_ret_peak") or src.get("week_ret") or -999))
        except (TypeError, ValueError):
            pass
    out = dict(merged)
    out["week_ret_peak"] = peak_w
    if float(out.get("week_ret") or -999) < peak_w - 0.0005 and peak and peak.get("knobs"):
        # Peak week_ret may have been clobbered; restore knobs from champion_peak if better
        try:
            if float(peak.get("week_ret_peak") or peak.get("week_ret") or -999) >= peak_w - 0.0005:
                out["knobs"] = dict(peak.get("knobs") or out.get("knobs") or {})
                out["week_ret"] = float(peak.get("week_ret_peak") or peak.get("week_ret") or peak_w)
        except (TypeError, ValueError):
            pass
    return out


def _sync_champion_peak(state: dict, peak: dict | None) -> None:
    """Write champion_peak and keep legacy ``champion`` alias in sync for UI."""
    if not peak or not peak.get("knobs"):
        return
    blob = dict(peak)
    blob["week_ret_peak"] = float(
        blob.get("week_ret_peak") or blob.get("week_ret") or 0
    )
    state["champion_peak"] = blob
    state["champion"] = dict(blob)


def _raise_champion_peak(
    state: dict,
    *,
    knobs: dict,
    week_ret: float,
    score: int | float | None,
    week: dict | None,
    ts: str,
    generation: int,
) -> dict:
    """Raise-only update to peak champion (never from silence_rotate)."""
    prev = _resolve_champion_peak(state)
    old_peak = float((prev or {}).get("week_ret_peak") or (prev or {}).get("week_ret") or -999)
    new_w = float(week_ret)
    if prev and new_w < old_peak - 0.0005:
        _sync_champion_peak(state, prev)
        return prev
    blob = {
        "ts": ts,
        "generation": generation,
        "week_ret": new_w,
        "week_ret_peak": max(old_peak, new_w),
        "score": score,
        "knobs": knobs,
        "week": week,
    }
    _sync_champion_peak(state, blob)
    return blob


def _density_polish_cfgs(peak_cfg: Config) -> list:
    """One-notch looser filters around peak — prefer densifying before family swap."""
    out = []
    adx = float(peak_cfg.adx_enter)
    vol = float(peak_cfg.volume_multiplier)
    for adx_n, sq, vol_n in (
        (max(10.0, adx - 2.0), peak_cfg.require_squeeze, vol),
        (adx, False, vol),
        (adx, peak_cfg.require_squeeze, max(1.0, round(vol * 0.9, 3))),
        (max(10.0, adx - 2.0), False, max(1.0, round(vol * 0.9, 3))),
    ):
        out.append(
            replace(
                peak_cfg,
                adx_enter=float(adx_n),
                adx_exit=min(float(peak_cfg.adx_exit), float(adx_n) - 2.0),
                require_squeeze=bool(sq),
                volume_multiplier=float(vol_n),
            )
        )
    return _dedupe_cfgs(peak_cfg, out)


def _material_research_edge(
    *,
    best_cfg: Config,
    base_cfg: Config,
    best_grade: dict,
    base_grade: dict,
    best_week: dict,
    base_week: dict,
    best_o: BacktestResult,
    base_o: BacktestResult,
    best_soft: Optional[BacktestResult],
    base_soft: Optional[BacktestResult],
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
        >= int(base_grade.get("score_raw", base_grade.get("score", 0))) + RESEARCH_MIN_SCORE_EDGE
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


def _hunt_search_mode(
    generation: int,
    held_streak: int,
    hunt: bool,
    *,
    has_champion: bool = False,
) -> str:
    """
    With champion: ~40% polish / 40% explore / 20% breakout (matrix covers TF×lev).
    Stuck: prefer breakout so cross-strategy elite seeds actually run.
    """
    if not hunt:
        return "polish"
    stuck = int(held_streak) >= int(HUNT_STUCK_HELD)
    # Stuck: 60% breakout / 40% explore — polish can't escape a dead family.
    if stuck:
        return "breakout" if (int(generation) % 5) in (0, 1, 3) else "explore"
    if has_champion:
        mod = int(generation) % 5
        if mod == 0:
            return "breakout"  # 20%
        if mod in (1, 3):
            return "explore"  # 40%
        return "polish"  # 40%
    return "explore"


def _cfg_dedupe_key(c: Config) -> tuple:
    return (
        c.strategy,
        c.timeframe,
        c.adx_enter,
        c.adx_exit,
        c.volume_multiplier,
        c.breakout_buffer_atr,
        c.breakout_lookback,
        c.trail_atr_multiple,
        c.trail_atr_loser,
        round(c.initial_stop_atr, 2),
        c.cooldown_bars,
        c.require_rising_adx,
        c.require_squeeze,
        c.require_rsi_veto,
        c.require_btc_regime,
        c.use_asymmetric_trail,
        round(c.h4_adx_min_short, 1),
        c.rsi_long_max,
        c.rsi_short_min,
        c.squeeze_lookback,
        c.enable_partial,
        c.partial_at_r,
        c.partial_pct,
        c.enable_time_stop,
        c.time_stop_bars,
        round(c.risk_pct, 3),
        round(c.initial_margin_pct, 3),
        int(c.mom_lookback),
        round(c.mom_min_atr, 3),
        int(c.leverage),
        int(c.max_open_positions),
        round(float(c.max_net_leverage), 2),
        int(c.h4_ema_fast),
        int(c.h4_ema_slow),
        int(c.ema_entry_period),
        int(c.pullback_lookback),
        round(float(c.btc_adx_min), 1),
        round(float(c.breadth_min), 2),
    )


def _dedupe_cfgs(base: Config, seeds: list) -> list:
    seen = set()
    uniq = []
    for c in seeds:
        c = _freeze_risk(base, c)
        key = _cfg_dedupe_key(c)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def _dedupe_cfgs_matrix(base: Config, seeds: list) -> list:
    """Dedupe while allowing TF×lev from candidates (risk rails still from base)."""
    seen = set()
    uniq = []
    for c in seeds:
        c = _freeze_risk_matrix(base, c)
        key = _cfg_dedupe_key(c)
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
        seeds.append(
            replace(
                base,
                volume_multiplier=max(1.1, min(3.0, round(base.volume_multiplier + d, 1))),
            )
        )
    for d in (-3, -2, -1, 1, 2, 3):
        seeds.append(replace(base, mom_lookback=max(3, min(24, int(base.mom_lookback + d)))))
    for d in (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3):
        seeds.append(
            replace(
                base,
                mom_min_atr=max(0.25, min(2.0, round(base.mom_min_atr + d, 2))),
            )
        )
    for d in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
        seeds.append(
            replace(
                base,
                trail_atr_multiple=max(1.25, min(4.0, round(base.trail_atr_multiple + d, 2))),
            )
        )
    for d in (-0.75, -0.5, -0.25, 0.25, 0.5, 0.75):
        seeds.append(
            replace(
                base,
                initial_stop_atr=max(1.5, min(4.0, round(base.initial_stop_atr + d, 2))),
            )
        )
    for d in (-6, -4, -2, 2, 4, 6):
        seeds.append(replace(base, time_stop_bars=max(4, min(48, int(base.time_stop_bars + d)))))
    for d in (-2, -1, 1, 2, 3):
        seeds.append(replace(base, cooldown_bars=max(0, min(12, int(base.cooldown_bars + d)))))
    for d in (-0.02, -0.01, 0.01, 0.02, 0.03):
        rp = max(0.02, min(0.05, round(base.risk_pct + d, 3)))
        seeds.append(replace(base, risk_pct=rp, initial_margin_pct=max(0.16, min(0.80, rp * 8))))
    for d in (-0.5, -0.25, 0.25, 0.5):
        seeds.append(
            replace(base, partial_at_r=max(0.5, min(3.0, round(base.partial_at_r + d, 2))))
        )
    # Filter flips — often the only way out of a stuck peak
    seeds.append(replace(base, require_btc_regime=not bool(base.require_btc_regime)))
    seeds.append(replace(base, require_rising_adx=not bool(base.require_rising_adx)))
    seeds.append(replace(base, use_asymmetric_trail=not bool(base.use_asymmetric_trail)))
    # HTF gate flips (polish seeds on — hunt can turn them off)
    seeds.append(replace(base, require_h4_bias=True))
    seeds.append(replace(base, require_daily_align=True))
    seeds.append(replace(base, require_weekly_bias=True))
    seeds.append(
        replace(base, require_h4_bias=True, require_daily_align=True, require_weekly_bias=True)
    )
    seeds.append(
        replace(
            base,
            require_h4_bias=not bool(getattr(base, "require_h4_bias", False)),
        )
    )
    # Leverage dial (aggressive only) — matrix-constrained
    if bool(getattr(base, "aggressive_mode", False)):
        for lev in HUNT_LEV_POLISH:
            if lev == int(base.leverage):
                continue
            tf = _tf_for_leverage(int(lev), str(base.timeframe))
            if not _lev_tf_allowed(int(lev), str(tf)):
                continue
            seeds.append(
                replace(
                    base,
                    leverage=lev,
                    timeframe=str(tf),
                    max_account_leverage=float(lev),
                    max_net_leverage=float(lev),
                )
            )
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
            seeds.append(
                replace(
                    base,
                    mom_lookback=max(3, min(24, int(base.mom_lookback + lb))),
                    mom_min_atr=max(0.25, min(2.0, round(base.mom_min_atr + thr, 2))),
                )
            )
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
    lev_opts = (
        list(HUNT_LEV_BREAKOUT if wide else HUNT_LEV_POLISH) if aggressive else [int(base.leverage)]
    )
    open_opts = [4, 5, 6, 7, 8] if aggressive else [int(base.max_open_positions)]
    strat_opts = (
        list(HUNT_STRATEGIES)
        if (aggressive and wide)
        else (
            list(HUNT_STRATEGIES)
            if (aggressive and rng.random() < 0.45)
            else [(base.strategy or "trend_mtf").strip().lower()]
        )
    )
    adx = float(rng.choice(adx_opts))
    risk = float(rng.choice(risk_opts))
    lev = int(rng.choice(lev_opts))
    n_open = int(rng.choice(open_opts))
    strat = str(rng.choice(strat_opts))
    # Prefer faster clocks; wide samples full HUNT_TF_ALLOW under lev×tf matrix.
    tf = _tf_for_leverage(int(lev), str(base.timeframe))
    if wide and rng.random() < 0.55:
        tf = str(rng.choice(list(HUNT_TF_ALLOW)))
        if not _lev_tf_allowed(int(lev), tf):
            tf = _tf_for_leverage(int(lev), tf)
    elif (not wide) and rng.random() < 0.25:
        tf = str(rng.choice(list(HUNT_TF_ALLOW)))
        if not _lev_tf_allowed(int(lev), tf):
            tf = _tf_for_leverage(int(lev), tf)
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
        require_rising_adx=not rng.random() < 0.85,
        require_squeeze=not rng.random() < 0.9,
        require_rsi_veto=not rng.random() < 0.85,
        require_btc_regime=bool(rng.choice([True, False, False])),
        require_h4_bias=bool(rng.choice([False, False, True])),
        require_daily_align=bool(rng.choice([False, False, True])),
        require_weekly_bias=bool(rng.choice([False, False, True])),
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
        max_account_leverage=float(lev),
        max_net_leverage=float(lev),
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
                seeds.append(
                    replace(
                        base,
                        require_btc_regime=btc,
                        require_rising_adx=rise,
                        use_asymmetric_trail=asym,
                    )
                )
    seeds.extend(_polish_cfgs(base)[:8])
    return seeds


def _lev_tf_allowed(lev: int, tf: str) -> bool:
    """Fixed leverage × timeframe matrix (before density gates)."""
    lev = int(lev)
    tf = str(tf or "").strip().lower()
    if tf == "5m":
        return lev in (5, 7, 10, 15, 25)
    if tf == "15m":
        return lev in (5, 7, 10, 15)  # no 25@15m
    return lev in (5, 7, 10)


def _tf_for_leverage(lev: int, base_tf: str) -> str:
    """Map leverage onto an allowed TF in the fixed matrix.

    25× always 5m; 15× may stay on 15m (density-gated later) or use 5m.
    """
    lev = int(lev)
    bt = str(base_tf or "15m").strip().lower()
    if bt not in HUNT_TF_ALLOW:
        bt = "15m"
    if lev >= 25:
        return "5m"
    if _lev_tf_allowed(lev, bt):
        return bt
    return "5m" if lev >= 15 else bt


def _lev_density_ok(
    lev: int,
    week: dict | None,
    *,
    dual_ok: bool = False,
    tf: str | None = None,
) -> bool:
    """High leverage requires setup density (holdout week + soft trade counts)."""
    lev = int(lev)
    week = week or {}
    wn = int(week.get("trades") or 0)
    wsn = int(week.get("soft_trades") or 0)
    if lev >= 25:
        return wn >= int(HUNT_LEV25_MIN_WEEK_TRADES) or wsn >= int(HUNT_LEV25_MIN_SOFT_TRADES)
    if lev >= 15:
        dense = wn >= int(HUNT_LEV15_MIN_WEEK_TRADES) or wsn >= int(HUNT_LEV15_MIN_SOFT_TRADES)
        # 15x@15m only with high density + dual-week OK
        if str(tf or "").strip().lower() == "15m":
            return dense and bool(dual_ok)
        return dense
    return True


def _tf_minutes(tf: str) -> float:
    """Candle size in minutes for bar-knob scaling."""
    t = str(tf or "").strip().lower()
    if t.endswith("m") and t[:-1].isdigit():
        return float(t[:-1])
    if t.endswith("h") and t[:-1].isdigit():
        return float(t[:-1]) * 60.0
    if t.endswith("d") and t[:-1].isdigit():
        return float(t[:-1]) * 1440.0
    return 15.0


def _scale_bar_knobs_for_tf(base: Config, new_tf: str) -> Config:
    """Scale bar-denominated knobs so 5m→1h is closer to like-for-like."""
    old_m = _tf_minutes(str(base.timeframe))
    new_m = _tf_minutes(new_tf)
    if old_m <= 0 or abs(new_m - old_m) < 1e-9:
        return replace(base, timeframe=str(new_tf).strip().lower())
    ratio = float(new_m) / float(old_m)
    # Coarser TF → fewer bars for same wall-clock; finer → more bars.
    # Scale lookbacks so wall-clock horizon stays roughly constant.
    def _sc(val: int, lo: int, hi: int) -> int:
        return int(max(lo, min(hi, round(int(val) / ratio if ratio >= 1 else int(val) * (1.0 / ratio)))))

    # When going 5m → 1h, ratio=12: same wall-clock needs FEWER bars → divide.
    # When going 1h → 5m, ratio=1/12: need MORE bars → multiply via 1/ratio branch above.
    if ratio >= 1.0:
        # coarser: divide bar counts
        mom = int(max(3, min(24, round(int(base.mom_lookback) / ratio))))
        cool = int(max(0, min(12, round(int(base.cooldown_bars) / ratio))))
        bo = int(max(8, min(48, round(int(base.breakout_lookback) / ratio))))
        ts = int(max(4, min(48, round(int(base.time_stop_bars) / ratio))))
        sq = int(max(4, min(24, round(int(base.squeeze_lookback) / ratio))))
        pb = int(max(2, min(12, round(int(base.pullback_lookback) / ratio))))
    else:
        inv = 1.0 / ratio
        mom = int(max(3, min(24, round(int(base.mom_lookback) * inv))))
        cool = int(max(0, min(12, round(int(base.cooldown_bars) * inv))))
        bo = int(max(8, min(48, round(int(base.breakout_lookback) * inv))))
        ts = int(max(4, min(48, round(int(base.time_stop_bars) * inv))))
        sq = int(max(4, min(24, round(int(base.squeeze_lookback) * inv))))
        pb = int(max(2, min(12, round(int(base.pullback_lookback) * inv))))
    return replace(
        base,
        timeframe=str(new_tf).strip().lower(),
        mom_lookback=mom,
        cooldown_bars=cool,
        breakout_lookback=bo,
        time_stop_bars=ts,
        squeeze_lookback=sq,
        pullback_lookback=pb,
    )


def _freeze_risk_matrix(base: Config, c: Config) -> Config:
    """Like `_freeze_risk` but allows strategy + TF + leverage from the candidate.

    Kill rails / risk_pct / daily / peak DD stay pinned to ``base``.
    Plain ``_freeze_risk`` pins strat/TF/lev to Live when aggressive_mode is off —
    that silently turned champion rescue into a no-op. Matrix freeze restores the
    tradable identity (S/T/L) from ``c``.
    """
    # Start from A.1 freeze (pins lev/strat when non-aggressive), then restore
    # candidate strategy + TF + lev so rescue/elite/pending deploy the real book.
    pinned = _freeze_risk(base, c)
    strat = str(getattr(c, "strategy", base.strategy) or base.strategy).strip().lower()
    if strat not in HUNT_STRATEGIES and strat not in (
        "momentum_rule",
        "combo_and",
        "regime_breakout",
    ):
        strat = (base.strategy or "trend_mtf").strip().lower()
    tf = str(getattr(c, "timeframe", base.timeframe) or base.timeframe).strip().lower()
    if tf not in HUNT_TF_ALLOW:
        tf = str(base.timeframe) if str(base.timeframe) in HUNT_TF_ALLOW else "15m"
    lev = int(getattr(c, "leverage", base.leverage) or base.leverage)
    if not _lev_tf_allowed(lev, tf):
        # Snap to nearest allowed lev at this TF
        for cand_lev in (5, 7, 10, 15, 25):
            if _lev_tf_allowed(cand_lev, tf):
                lev = cand_lev
                break
    return replace(
        pinned,
        strategy=strat,
        leverage=int(lev),
        timeframe=tf,
        max_account_leverage=float(lev),
        max_net_leverage=float(lev),
        risk_pct=float(base.risk_pct),
        max_daily_loss_pct=float(base.max_daily_loss_pct),
        max_peak_drawdown_pct=float(base.max_peak_drawdown_pct),
    )


def _tf_lev_matrix_cfgs(base: Config) -> list:
    """Full like-for-like TF×lev grid (~15 cells) with bar knobs scaled per TF."""
    seeds = []
    for tf in HUNT_MATRIX_TFS:
        tf = str(tf).strip().lower()
        scaled = _scale_bar_knobs_for_tf(base, tf)
        for lev in (5, 7, 10, 15, 25):
            if not _lev_tf_allowed(int(lev), tf):
                continue
            seeds.append(
                replace(
                    scaled,
                    leverage=int(lev),
                    timeframe=tf,
                    max_account_leverage=float(lev),
                    max_net_leverage=float(lev),
                )
            )
    return seeds


def _leverage_ladder_cfgs(base: Config) -> list:
    """Legacy book-TF ladder (+ 5m high-lev probes). Prefer `_tf_lev_matrix_cfgs`."""
    seeds = []
    bt = str(base.timeframe or "15m").strip().lower()
    if bt not in HUNT_TF_ALLOW:
        bt = "15m"
    # Allowed levs at the book's TF
    allowed = (5, 7, 10, 15, 25) if bt == "5m" else (5, 7, 10, 15)
    for lev in allowed:
        if not _lev_tf_allowed(int(lev), bt):
            continue
        seeds.append(
            replace(
                base,
                leverage=int(lev),
                timeframe=bt,
                max_account_leverage=float(lev),
                max_net_leverage=float(lev),
            )
        )
    # If book is 15m, also probe 5m high-lev slots from the matrix
    if bt == "15m":
        for lev in (15, 25):
            seeds.append(
                replace(
                    base,
                    leverage=int(lev),
                    timeframe="5m",
                    max_account_leverage=float(lev),
                    max_net_leverage=float(lev),
                )
            )
    return seeds


def _holdout_span_ok(days: float, *, min_days: float | None = None) -> bool:
    """True if holdout calendar span is long enough to score."""
    need = float(WEEK_DAYS - 1) if min_days is None else float(min_days)
    try:
        return float(days) + 1e-9 >= need
    except (TypeError, ValueError):
        return False


def _rank_matrix_rows(rows: list[dict] | None) -> dict | None:
    """Best robust, non-skipped row by holdout week_ret."""
    best = None
    best_w = -999.0
    for row in rows or []:
        if not row or row.get("skip") or not row.get("robust_ok"):
            continue
        try:
            # Do not use `or -999` — week_ret 0.0 is a valid flat week.
            raw = row.get("week_ret")
            wr = float(raw) if raw is not None else -999.0
        except (TypeError, ValueError):
            continue
        if wr > best_w + 1e-12:
            best_w = wr
            best = row
    return best


def _rank_matrix_rows_escape(
    rows: list[dict] | None, *, base_week_ret: float
) -> dict | None:
    """When no robust cell exists and Live is red, pick the least-bad matrix week.

    Requires the cell week to beat the deployed book by ≥2pp (or be green while
    base is red). Robust_ok is NOT required — that gate was freezing Live on a
    death TF×lev while 4h sat at +1%.
    """
    best = None
    best_w = -999.0
    base_w = float(base_week_ret)
    for row in rows or []:
        if not row or row.get("skip"):
            continue
        try:
            raw = row.get("week_ret")
            wr = float(raw) if raw is not None else -999.0
        except (TypeError, ValueError):
            continue
        beats = wr >= base_w + 0.02
        green_escape = base_w < 0.0 and wr > 0.0
        if not (beats or green_escape):
            continue
        if wr > best_w + 1e-12:
            best_w = wr
            best = row
    return best


def _best_by_tf_raise(
    state: dict,
    *,
    tf: str,
    knobs: dict,
    week_ret: float,
    oos_ret: float,
    generation: int,
    ts: str,
    prior_ret: float | None = None,
) -> None:
    """Raise-only per-TF memory."""
    tf = str(tf).strip().lower()
    mem = state.setdefault("best_by_tf", {})
    if not isinstance(mem, dict):
        mem = {}
        state["best_by_tf"] = mem
    prev = mem.get(tf) if isinstance(mem.get(tf), dict) else None
    old_w = float((prev or {}).get("week_ret") or -999)
    new_w = float(week_ret)
    if prev and new_w < old_w - 0.0005:
        return
    blob = {
        "ts": ts,
        "generation": generation,
        "week_ret": new_w,
        "oos_ret": float(oos_ret),
        "knobs": knobs,
    }
    if prior_ret is not None:
        blob["prior_ret"] = float(prior_ret)
    mem[tf] = blob


def _best_by_tf_seed_ok(
    blob: dict | None,
    *,
    prior_ret: float | None = None,
    oos_ret: float | None = None,
    soft_ret: float | None = None,
) -> bool:
    """Revalidate stored best_by_tf before injecting as a polish seed."""
    if not isinstance(blob, dict) or not blob.get("knobs"):
        return False
    pr = prior_ret if prior_ret is not None else blob.get("prior_ret")
    oos = oos_ret if oos_ret is not None else blob.get("oos_ret")
    if oos is None:
        return False
    week = {"prior_ret": pr, "ret": blob.get("week_ret")}
    if pr is None:
        return False
    soft = soft_ret if soft_ret is not None else blob.get("soft_ret")
    ok, _ = _period_robust_ok(week, float(oos), float(soft) if soft is not None else float(oos))
    return bool(ok)


def _select_polish_base(
    *,
    live_base: Config,
    matrix_winner_cfg: Config | None,
    best_by_tf_cfg: Config | None,
    champion_cfg: Config | None,
) -> Config:
    """Precedence: robust matrix winner > best_by_tf > champion > Live."""
    if matrix_winner_cfg is not None:
        return matrix_winner_cfg
    if best_by_tf_cfg is not None:
        return best_by_tf_cfg
    if champion_cfg is not None:
        return champion_cfg
    return live_base


def _run_tf_lev_matrix_from_rows(
    rows: list[dict],
    *,
    live_base: Config,
    cell_cfgs: dict[tuple[str, int], Config],
) -> tuple[Config | None, dict | None]:
    """Pick polish cfg from precomputed matrix rows (testable without network)."""
    winner = _rank_matrix_rows(rows)
    if not winner:
        return None, None
    key = (str(winner.get("tf")), int(winner.get("leverage") or 0))
    cfg = cell_cfgs.get(key)
    if cfg is None:
        return None, winner
    return cfg, winner


def _candidate_cfgs(
    base: Config,
    generation: int = 0,
    hunt: bool = True,
    search_mode: str = "polish",
    *,
    allow_lev_tf: bool = False,
) -> list:
    """
    polish: nudges around current best.
    explore: wider random + local ring.
    breakout: large jumps across filters/exits when held forever.

    When allow_lev_tf=True (matrix hunt), dedupe uses `_freeze_risk_matrix`.
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
    dedupe = _dedupe_cfgs_matrix if allow_lev_tf else _dedupe_cfgs
    return dedupe(base, seeds)


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
        int(getattr(cfg, "ema_weekly_period", 10) or 10),
    )


def _prepare_cached(raw: dict, cfg: Config, cache: dict) -> dict:
    """Feature prepare with hit/miss counters on cache['_stats']."""
    stats = cache.setdefault("_stats", {"hits": 0, "misses": 0, "prepare_calls": 0})
    key = _feature_cache_key(cfg)
    hit = cache.get(key)
    if hit is not None:
        stats["hits"] = int(stats.get("hits") or 0) + 1
        return hit
    stats["misses"] = int(stats.get("misses") or 0) + 1
    stats["prepare_calls"] = int(stats.get("prepare_calls") or 0) + 1
    data = prepare_data(raw, cfg)
    cache[key] = data
    return data


def _hard_promising(
    ci: BacktestResult,
    co: BacktestResult,
    best_i: BacktestResult,
    best_o: BacktestResult,
    best_score: int,
    *,
    slack: int = 2,
) -> bool:
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
    return _rank_key(ci, co, None) > _rank_key(best_i, best_o, None)


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
WF_WARMUP_DAYS = 60  # skip indicator warm-up at the very start
WF_MIN_FOLD_TRADES = 8
WF_MIN_FOLDS_USED = 2
WF_WORST_FOLD_FLOOR = -0.15  # no unseen fold may lose more than 15%
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
        folds.append(
            {
                "from": f"{a:%Y-%m-%d}",
                "to": f"{b:%Y-%m-%d}",
                "trades": int(res.trades),
                "exp": round(float(res.expectancy_r), 3),
                "ret": round(float(res.total_return), 3),
                "win": round(float(res.win_rate), 3),
            }
        )
    used = [f for f in folds if f["trades"] >= WF_MIN_FOLD_TRADES]
    n_tot = sum(f["trades"] for f in used)
    w_exp = (sum(f["exp"] * f["trades"] for f in used) / n_tot) if n_tot else 0.0
    green = sum(1 for f in used if f["ret"] > 0)
    worst = min((f["ret"] for f in used), default=0.0)
    ok = bool(
        len(used) >= WF_MIN_FOLDS_USED
        and w_exp > 0
        and green * 3 >= len(used) * 2  # ≥ 2/3 of used folds green
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
        "ok": ok,
        "why": why,
        "folds": folds,
        "trades": int(n_tot),
        "exp": round(float(w_exp), 3),
        "green": f"{green}/{len(used)}",
        "worst": round(float(worst), 3),
        "span": f"{a0:%Y-%m-%d} → {pd.Timestamp(hunt_start):%Y-%m-%d}",
    }


FWD_MIN_TRADES = 20  # closed forward paper trades required before the score can exceed 85
FWD_SCORE_CAP = 85

# Recent-window signal (not a hard promote veto while the book is climbing)
RECENT_DAYS = 90
RECENT_MIN_TRADES = 12
RECENT_EXP_FLOOR = -0.25  # climb-friendly; deep bleed still flagged, mild drag OK

# Primary money holdout (calendar days frozen out of fit / scored for promote).
# 14d > old 7d — more trades / power before research promote.
WEEK_DAYS = 14


def _window_eval(
    data: dict,
    cfg: Config,
    days: int,
    end: Optional[pd.Timestamp] = None,
    *,
    soft: bool = True,
    min_trades: int = 0,
    exp_floor: float = -99.0,
) -> dict:
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
            "ok": False,
            "why": f"{days}d eval failed: {e}",
            "trades": 0,
            "exp": 0.0,
            "ret": 0.0,
            "final": 0.0,
            "span": f"{start:%Y-%m-%d} → {pd.Timestamp(end):%Y-%m-%d}",
            "days": int(days),
            "soft": bool(soft),
        }
    thin = int(min_trades) > 0 and res.trades < int(min_trades)
    ok = (not thin) and float(res.expectancy_r) >= float(exp_floor)
    why = (
        "passed"
        if ok
        else (
            f"thin {days}d sample n={res.trades}"
            if thin
            else f"{days}d exp {res.expectancy_r:+.2f}R < {exp_floor}"
        )
    )
    return {
        "ok": ok,
        "why": why,
        "trades": int(res.trades),
        "exp": round(float(res.expectancy_r), 3),
        "ret": round(float(res.total_return), 3),
        "final": round(float(res.final_equity), 2),
        "win": round(float(res.win_rate), 3),
        "span": f"{start:%Y-%m-%d} → {pd.Timestamp(end):%Y-%m-%d}",
        "days": int(days),
        "soft": bool(soft),
    }


def _recent_eval(data: dict, cfg: Config, end: Optional[pd.Timestamp] = None) -> dict:
    """Soft (kills off) expectancy over the last RECENT_DAYS ending at `end`."""
    return _window_eval(
        data,
        cfg,
        RECENT_DAYS,
        end=end,
        soft=True,
        min_trades=RECENT_MIN_TRADES,
        exp_floor=RECENT_EXP_FLOOR,
    )


def _week_eval(data: dict, cfg: Config, end: Optional[pd.Timestamp] = None) -> dict:
    """PRIMARY: hard + soft P&L over the last WEEK_DAYS calendar days."""
    week_min = int(HUNT_TARGETS.get("week_min_trades", 2))
    hard = _window_eval(
        data,
        cfg,
        WEEK_DAYS,
        end=end,
        soft=False,
        min_trades=week_min,
        exp_floor=-0.50,
    )
    soft = _window_eval(
        data,
        cfg,
        WEEK_DAYS,
        end=end,
        soft=True,
        min_trades=0,
        exp_floor=-99.0,
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


def _dual_week_eval(data: dict, cfg: Config, end: Optional[pd.Timestamp] = None) -> dict:
    """Primary WEEK_DAYS + prior WEEK_DAYS (non-overlapping)."""
    if end is None:
        end = max(sd.bars[-1].Index for sd in data.values())
    end = pd.Timestamp(end)
    primary = _week_eval(data, cfg, end=end)
    prior_end = end - pd.Timedelta(days=int(WEEK_DAYS))
    week_min = int(HUNT_TARGETS.get("week_min_trades", 2))
    prior_hard = _window_eval(
        data,
        cfg,
        WEEK_DAYS,
        end=prior_end,
        soft=False,
        min_trades=week_min,
        exp_floor=-0.50,
    )
    primary["prior_ret"] = float(prior_hard.get("ret") or 0)
    primary["prior_trades"] = int(prior_hard.get("trades") or 0)
    primary["prior_ok"] = bool(prior_hard.get("ok"))
    primary["prior_span"] = prior_hard.get("span")
    primary["dual_ok"] = _dual_week_ok(primary)
    return primary


def _dual_week_ok(week: dict | None) -> bool:
    """Promote/rescue: prior ≥ 0 OR both ≥ week_hard_floor; primary drives +100% stretch."""
    if not week:
        return False
    hard_floor = float(HUNT_TARGETS.get("week_hard_floor", 0.25))
    week_tgt = float(HUNT_TARGETS.get("week_return", 1.0))
    try:
        primary = float(week.get("ret") or 0)
        prior = float(week.get("prior_ret") or 0)
    except (TypeError, ValueError):
        return False
    if primary >= week_tgt and prior >= 0.0:
        return True
    return primary >= hard_floor and prior >= hard_floor


def _period_robust_ok(
    week: dict | None,
    oos_ret: float,
    soft_ret: float | None = None,
) -> tuple[bool, str]:
    """Hard period floors: recent week alone must not crown a multi-window loser.

    When require_period_robust: long OOS, prior 14d, and soft OOS must clear floors
    (default 0 — no red periods). Year@coarse backtests can still look ugly; this
    stops hunt from calling a one-window spike "best".
    """
    if not HUNT_TARGETS.get("require_period_robust", True):
        return True, "robust_off"
    try:
        oos = float(oos_ret)
    except (TypeError, ValueError):
        return False, "oos_missing"
    oos_floor = float(HUNT_TARGETS.get("oos_floor", 0.0))
    if oos < oos_floor:
        return False, f"oos {oos:+.0%} < floor {oos_floor:+.0%}"
    if week is None or week.get("prior_ret") is None:
        return False, "prior_week_missing"
    try:
        prior = float(week.get("prior_ret"))
    except (TypeError, ValueError):
        return False, "prior_week_bad"
    prior_floor = float(HUNT_TARGETS.get("prior_week_floor", 0.0))
    if prior < prior_floor:
        return False, f"prior {prior:+.0%} < floor {prior_floor:+.0%}"
    if soft_ret is not None:
        try:
            soft = float(soft_ret)
        except (TypeError, ValueError):
            return False, "soft_missing"
        soft_floor = float(HUNT_TARGETS.get("soft_promote_floor", 0.0))
        if soft < soft_floor:
            return False, f"soft {soft:+.0%} < floor {soft_floor:+.0%}"
    return True, "period_robust"


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
        "n": n,
        "pnl": round(pnl, 2),
        "win": round(wins / n, 3) if n else 0.0,
        "need": FWD_MIN_TRADES,
        "since": since_iso,
        "ok": bool(n >= FWD_MIN_TRADES and pnl > 0),
    }


def _wf_msg(wf: Optional[dict]) -> str:
    if not wf:
        return "unseen-history check not run"
    verdict = "PASS" if wf.get("ok") else "FAIL"
    return (
        f"unseen {wf.get('span', '')} {verdict} ({wf.get('green', '?')} periods green, "
        f"{wf.get('exp', 0):+.2f}R avg, {wf.get('trades', 0)} trades"
        + ("" if wf.get("ok") else f" — {wf.get('why', '')}")
        + ")"
    )


def _score(
    ins: BacktestResult, oos: BacktestResult, soft_oos: Optional[BacktestResult] = None
) -> float:
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


def _rank_key(
    ins: BacktestResult,
    oos: BacktestResult,
    soft_oos: Optional[BacktestResult] = None,
    week: Optional[dict] = None,
) -> tuple:
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


def _beats(
    a_i: BacktestResult,
    a_o: BacktestResult,
    b_i: BacktestResult,
    b_o: BacktestResult,
    a_soft: Optional[BacktestResult] = None,
    b_soft: Optional[BacktestResult] = None,
    a_week: Optional[dict] = None,
    b_week: Optional[dict] = None,
) -> bool:
    """True if A is strictly better than B for hunting."""
    return _rank_key(a_i, a_o, a_soft, a_week) > _rank_key(b_i, b_o, b_soft, b_week)


def _grade(
    ins: BacktestResult,
    oos: BacktestResult,
    soft_oos: Optional[BacktestResult] = None,
    wf: Optional[dict] = None,
    wf_required: bool = False,
    fwd: Optional[dict] = None,
    fwd_required: bool = False,
    week: Optional[dict] = None,
) -> dict:
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
    kill_illusion = hard_halted and soft_oos is not None and soft_ret < hard_ret - 0.05
    over_killed = hard_halted and soft_oos is not None and soft_ret > hard_ret + 0.10
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
            ins.expectancy_r > 0 and oos.expectancy_r > 0 and ins.total_return > 0 and hard_ret > 0
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
    return not cand_o.max_drawdown > 0.85


def _is_valid_candidate(
    cand_i: BacktestResult, cand_o: BacktestResult, soft_oos: Optional[BacktestResult] = None
) -> bool:
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
    return not (
        HUNT_TARGETS.get("block_kill_illusion", True)
        and soft_oos is not None
        and (cand_o.halted or "").strip()
        and soft_oos.total_return < cand_o.total_return - 0.05
    )


def _is_climb_ok(
    cand_i: BacktestResult,
    cand_o: BacktestResult,
    soft_oos: Optional[BacktestResult],
    base_i: Optional[BacktestResult] = None,
    base_o: Optional[BacktestResult] = None,
) -> bool:
    """Research-promote gate: allow relative climb from a losing baseline.

    Soft floor loosens while baseline is underwater so soft-check can feed
    _consider. Kill-illusion still blocked. Does NOT demand exp>0 while ≤0.
    """
    if not _is_screenable(cand_i, cand_o):
        return False
    baseline_losing = (
        base_i is not None
        and base_o is not None
        and (base_i.expectancy_r <= 0 or base_o.expectancy_r <= 0 or base_o.total_return <= 0)
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
        seeds.append(
            replace(
                base,
                strategy="momentum_rule" if i % 2 == 0 else "combo_or",
                timeframe=str(base.timeframe) if str(base.timeframe) in HUNT_TF_ALLOW else "15m",
            )
        )
    log.info("discover inject: %d seed(s) from %s", len(seeds), path.name)
    return seeds
