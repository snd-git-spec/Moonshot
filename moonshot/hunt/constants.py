from __future__ import annotations

import fcntl
import json
import logging
import os
import uuid
from dataclasses import replace
from datetime import datetime, timezone
from typing import Callable, Optional

import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    ConfigError,
)
from moonshot.config import (
    Config,
)
from moonshot.data import (
    RANGE_PRICE_SOURCES,
    RANGE_SOURCE_BINANCE_SPOT,
    RANGE_SOURCE_YAHOO,
    _MAX_RANGE_BARS,
    _as_utc_ts,
    _normalize_range_source,
    estimate_range_bars,
    fetch_historical_range,
    range_warmup_bars,
    timeframe_ms,
)

from moonshot.paths import (
    ENV_PATH,
    AUTOPILOT_STATE,
    LIVE_FEED,
)

_SAFE_ENV_KEYS = {
    "ADX_ENTER",
    "ADX_EXIT",
    "VOLUME_MULTIPLIER",
    "BREAKOUT_BUFFER_ATR",
    "BREAKOUT_LOOKBACK",
    "TRAIL_ATR",
    "TRAIL_ATR_LOSER",
    "USE_ASYMMETRIC_TRAIL",
    "COOLDOWN_BARS",
    "REQUIRE_RISING_ADX",
    "ENABLE_PYRAMIDING",
    "ENABLE_PARTIAL",
    "PARTIAL_AT_R",
    "PARTIAL_PCT",
    "LEVERAGE",
    "MAX_ACCOUNT_LEVERAGE",
    "MAX_NET_LEVERAGE",
    "MAX_OPEN_POSITIONS",
    "TIMEFRAME",
    "STRATEGY",
    "MOM_LOOKBACK",
    "MOM_MIN_ATR",
    "INITIAL_STOP_ATR",
    "FEE_TAKER",
    "SLIPPAGE_BPS",
    "H4_EMA_FAST",
    "H4_EMA_SLOW",
    "H4_ADX_MIN_SHORT",
    "EMA_ENTRY_PERIOD",
    "PULLBACK_LOOKBACK",
    "BTC_ADX_MIN",
    "BREADTH_MIN",
    "RISK_PCT",
    "INITIAL_MARGIN_PCT",
    "AGGRESSIVE_MARGIN_PCT",
    "MAX_DAILY_LOSS_PCT",
    "MAX_PEAK_DRAWDOWN_PCT",
    "STREAK_LOSSES",
    "STREAK_PAUSE_HOURS",
    "EXPECTANCY_FLOOR",
    "REQUIRE_RSI_VETO",
    "RSI_LONG_MAX",
    "RSI_SHORT_MIN",
    "RSI_PERIOD",
    "REQUIRE_SQUEEZE",
    "SQUEEZE_LOOKBACK",
    "BB_PERIOD",
    "BB_STD",
    "KC_PERIOD",
    "KC_ATR_MULT",
    "REQUIRE_BTC_REGIME",
    "REQUIRE_H4_BIAS",
    "REQUIRE_DAILY_ALIGN",
    "REQUIRE_WEEKLY_BIAS",
    "EMA_WEEKLY_PERIOD",
    "ENABLE_TIME_STOP",
    "TIME_STOP_BARS",
    "TIME_STOP_MIN_R",
}

# Autopilot hunt targets.
# Hunt *search* primary = p25 fold Sortino (see fold_eval / _rank_key).
# Holdout week_return remains the Live dashboard / +100% *goal* diagnostic — not Stage A sort.
HUNT_TARGETS = {
    # Fold-primary search floor (p25 Sortino across folds)
    "fold_sortino_floor": 0.0,
    "hunt_primary_metric": "p25_fold_sortino",
    "train_exp": 0.10,
    "test_exp": 0.08,
    "train_n": 20,
    "test_n": 20,
    "max_dd": 0.55,
    "win_rate": 0.60,  # PRIMARY quality bar — aim for consistent winners, not coin-flip
    "test_return": 0.50,  # secondary: long OOS context (NOT the weekly claim)
    "soft_return_floor": -0.30,  # soft-check screen (pool entry); promote uses floors below
    "week_return": 1.00,  # PRIMARY: +100% in holdout window on leveraged book
    # Consider/promote gate — denser than old 6/7d so noise books don't promote.
    "week_min_trades": 10,
    "week_target_trades": 40,
    "weekly_return_goal": 1.00,
    "week_stretch": 1.00,
    # Hard week floor before "targets met" (soft-only fireworks blocked)
    "week_hard_floor": 0.25,
    # Period-robust promote/champion: no red prior 14d / long OOS / soft OOS.
    # Stops crowning a +week book that bleeds every other window (year chart death).
    "require_period_robust": True,
    "oos_floor": 0.0,
    "prior_week_floor": 0.0,
    "soft_promote_floor": 0.0,
    # Escape promote (leave a death book): absolute wipeout floors + relative
    # vs Live. Week-led-only escape is forbidden — holdout green / OOS red
    # must not patch .env.
    "escape_oos_floor": -0.10,
    "escape_soft_floor": -0.10,
    "escape_rel_oos_pp": 0.02,
    "escape_rel_soft_pp": 0.02,
    "escape_week_slack_pp": 0.05,  # cand week may be up to 5pp below Live
    # Arm escape only when Live is a death book (see live_death_book).
    "escape_arm_oos": -0.10,
    "escape_arm_soft": -0.10,
    # Sealed holdout (blind exam after selection) — calendar days after selection end.
    "sealed_holdout_days": 14,
    "sealed_week_floor": 0.0,
    "sealed_vs_live_edge": 0.008,  # match RESEARCH_MIN_WEEK_EDGE default spirit
    "escape_sealed_floor": -0.10,
    # When Live sealed is already below escape_sealed_floor, allow escape if
    # cand sealed is within this many pp of Live sealed (survive-first).
    "escape_sealed_vs_live_tol_pp": 0.02,
    # Reject classic kill illusion: hard quit-while-ahead while soft bleeds
    "block_kill_illusion": True,
    # Refuse .env patch when research venue ≠ MEXC and MEXC holdout ≤ 0 while research > 0
    "require_mexc_transfer_ok": True,
    # Research promote requires walk-forward green (silence_rotate may bypass)
    "require_walk_forward": True,
    # DSR promote floor (research); escape records but does not require
    "require_dsr": True,
    "dsr_threshold": 0.0,
}

# Fast hunt knobs — more passes/hour beat giant slow batches for self-improve.
# Funnel: scout many ideas on BTC only → confirm survivors on multi-coin →
# leverage ladder only on books that cleared multi-coin.
# 3m is OUT of hunt (1m-resample cost).
#
# Calendar-normalized depth: same wall-clock for every TF; bars follow TF under
# MAX_RANGE_BARS. Research prices default to Binance spot (deeper than MEXC).
HUNT_SCOUT_DAYS = 90  # reserved scout-only slice
HUNT_CONFIRM_DAYS = 180  # search / multi-coin confirm window
HUNT_FINALIST_DAYS = 540  # promote recheck target (clamped by bar cap)
HUNT_WF_DAYS = 180  # walk-forward history before fit start
HUNT_SEARCH_SYMBOLS = 5  # multi-coin confirm book
HUNT_SCOUT_TOP = 16  # BTC survivors that get multi-coin soft/week
HUNT_EXPLORE_N = 48  # aggressive scout (cheap on 1 coin)
HUNT_BREAKOUT_N = 56
HUNT_SOFT_BUDGET = {"polish": 10, "explore": 14, "breakout": 16}
# After this many HOLD passes, loosen floor / widen confirm so overnight climb continues
HUNT_STUCK_HELD = 6
# Pass liveness (MemoryMax / OOM-restart loops): finish signal, not activity.
# Stuck = no heartbeat progress (not wall-clock alone — full matrix passes are long).
HUNT_HEARTBEAT_STALE_MIN = 40.0  # busy + pass_heartbeat_at older → HUNT_PASS_STUCK
HUNT_STALE_FINISH_HOURS = 6.0  # no last_improve_at move → HUNT_NO_FINISH
HUNT_TERMINAL_GRACE_MIN = 15.0  # ignore stale right after hold/upgrade/etc.
# Back-compat alias (old wall-clock stuck); health uses heartbeat stale instead.
HUNT_PASS_MAX_HOURS = HUNT_HEARTBEAT_STALE_MIN / 60.0
HUNT_CHAMPION_STALE_PP = 0.08  # stored peak − live reeval ≥ 8pp → floor uses live
HUNT_STRATEGIES = ("trend_mtf", "momentum_15", "combo_or")
# A.4 — research allow-list includes 1h/4h; keep 5m/15m for continuity.
HUNT_TF_ALLOW = ("5m", "15m", "1h", "4h")
# Full bake-off TFs for like-for-like TF×lev matrix (same as allow list).
HUNT_MATRIX_TFS = HUNT_TF_ALLOW
HUNT_MATRIX_SOFT_CAP = 5  # soft+robust top-N (TF-diverse pick; see _matrix_soft_pick)
# Denser knob grid on Live TF only: 3 strat × 3 lev × 3 adx × 1 cooldown = 27.
# 81 (×3 cooldowns) OOMed Stage A on the 2G droplet — never rebuild that bomb.
HUNT_DENSER_MAX = 27
HUNT_EXHAUSTED_DENSER_MAX = 36  # cross-TF denser when hunt_exhausted
HUNT_RETIRE_GRACE_PASSES = 10  # paper retire without robust → regime_reassess
HUNT_DENSER_COOLDOWNS = (1,)  # single CD → n ≤ 27
HUNT_LEV_POLISH = (5, 7, 10)
HUNT_LEV_BREAKOUT = (5, 7, 10, 15, 25)
HUNT_LEV_FORCE = (5, 10, 15, 25)

# Successive-halving + curriculum + banlist + champion rescue
HUNT_HALVING_A_DAYS = 45  # cheap Stage A window on BTC
HUNT_HALVING_A_KEEP = 16  # thinner Stage A→B keep (was 24; stuck fat was churn)
HUNT_HALVING_B_KEEP = 12
# When stuck: raise soft_budget to at least this (was 20 — fat funnel diluted escapes).
HUNT_STUCK_SOFT_BUDGET = 14
HUNT_CURRICULUM_SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT")
HUNT_BANLIST_CAP = 5000
HUNT_BAN_WEEK_FLOOR = -0.20  # week_ret below this → ban
HUNT_CHAMP_TRAIL_PP = 0.30  # champ − live week ≥ 30pp
HUNT_CHAMP_TRAIL_PASSES = 2  # consecutive passes before rescue
# Silence rotate may not deploy a book worse than peak same-window reeval by more than this.
HUNT_SILENCE_MAX_PEAK_GAP_PP = 0.20
# Silence rotate: escape hatch when paper is idle OR bleeding while hunt is stuck.
# Without this, a stale +week champion freezes Live on a losing book forever.
HUNT_SILENCE_ROTATE_ENABLED = True
# Paper bleed: last N closed paper trades all losers → force rotate / pause path
PAPER_BLEED_N = 5
PAPER_BLEED_FORCE_ROTATE = True
# Hard circuit: while bleeding, paper opens no new entries until hunt deploys
# a different book (cleared on research/silence/manual promote).
PAPER_BLEED_HALT_ENTRIES = True
# After this many consecutive HOLDs, absolute period-robust floors soften to
# "beat the deployed book" so hunt can leave a death setup.
HUNT_STUCK_SOFTEN_HELD = 12
# High-lev density floors (holdout week trades)
HUNT_LEV15_MIN_WEEK_TRADES = 12
HUNT_LEV25_MIN_WEEK_TRADES = 20
HUNT_LEV15_MIN_SOFT_TRADES = 20
HUNT_LEV25_MIN_SOFT_TRADES = 35


def autopilot_frozen() -> bool:
    """When True: no .env promote / paper restart from hunt (measurement window)."""
    import os

    return os.getenv("AUTOPILOT_FROZEN", "0").strip() in ("1", "true", "TRUE", "yes", "YES")


def _hunt_soft_workers() -> int:
    """Parallel soft-check workers (1–4). Default 1 (2G droplet); override via HUNT_SOFT_WORKERS."""
    raw = os.getenv("HUNT_SOFT_WORKERS", "").strip()
    if not raw:
        # deploy_do sets MOONSHOT_SOFT_WORKERS=1; default safe for shared MemoryMax=2000M
        ms = os.getenv("MOONSHOT_SOFT_WORKERS", "1").strip().lower()
        raw = "1" if ms in ("", "1", "true", "yes") else "2"
    try:
        n = int(raw)
    except (TypeError, ValueError):
        n = 1
    return max(1, min(4, n))


# Legacy names kept for Health / external imports (now derived from calendar).
HUNT_SEARCH_BARS_15M = 4500
HUNT_FULL_BARS_15M = 9000


def _hunt_price_source() -> str:
    """Research venue for hunt — live/paper stay on MEXC. Yahoo blocked (daily-only)."""
    raw = str(os.getenv("HUNT_PRICE_SOURCE", RANGE_SOURCE_BINANCE_SPOT)).strip().lower()
    if raw not in RANGE_PRICE_SOURCES or raw == RANGE_SOURCE_YAHOO:
        return RANGE_SOURCE_BINANCE_SPOT
    return raw


def _max_calendar_days_for_tf(tf: str, *, max_bars: int = _MAX_RANGE_BARS) -> int:
    """Longest calendar span whose bar estimate (incl. warmup) fits ``max_bars``."""
    tf = str(tf).strip().lower()
    warm = range_warmup_bars(tf)
    step_ms = timeframe_ms(tf)
    usable = max(1, int(max_bars) - int(warm) - 2)
    days = int(usable * step_ms / 86_400_000)
    return max(14, days)


def _hunt_calendar_days(tier: str, tf: str) -> int:
    """Target calendar days for a hunt tier, clamped to the bar cap at ``tf``."""
    want = {
        "scout": int(HUNT_SCOUT_DAYS),
        "confirm": int(HUNT_CONFIRM_DAYS),
        "finalist": int(HUNT_FINALIST_DAYS),
        "wf": int(HUNT_WF_DAYS),
    }.get(str(tier).strip().lower(), int(HUNT_CONFIRM_DAYS))
    return int(min(want, _max_calendar_days_for_tf(tf)))


def _hunt_window_bounds(
    tier: str,
    tf: str,
    *,
    end=None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """``[start, end)`` UTC window for a hunt tier at ``tf``."""
    t1 = _as_utc_ts(end) if end is not None else pd.Timestamp.now(tz="UTC")
    days = _hunt_calendar_days(tier, tf)
    t0 = t1 - pd.Timedelta(days=int(days))
    return t0, t1


def _hunt_fetch_raw(
    symbols: list,
    tf: str,
    tier: str,
    *,
    refresh: bool = False,
    end=None,
    source: Optional[str] = None,
) -> dict:
    """Load OHLCV for hunt via calendar range (default Binance spot)."""
    src = _normalize_range_source(source or _hunt_price_source())
    t0, t1 = _hunt_window_bounds(tier, tf, end=end)
    warm = range_warmup_bars(tf)
    raw: dict = {}
    for s in symbols:
        try:
            raw[s] = fetch_historical_range(
                s,
                tf,
                t0,
                t1,
                warmup_bars=warm,
                refresh=bool(refresh),
                source=src,
            )
        except Exception as e:
            log.warning("hunt fetch %s %s %s failed: %s", s, tf, tier, e)
    log.info(
        "hunt data · source=%s tier=%s tf=%s days≈%d · %s→%s · %d/%d symbols",
        src,
        tier,
        tf,
        _hunt_calendar_days(tier, tf),
        t0.date(),
        t1.date(),
        len(raw),
        len(symbols),
    )
    return raw


def _hunt_fetch_wf_raw(
    symbols: list,
    tf: str,
    hunt_start,
    *,
    refresh: bool = False,
    source: Optional[str] = None,
) -> dict:
    """History ending at fit start for walk-forward folds the hunt never trains on."""
    src = _normalize_range_source(source or _hunt_price_source())
    hs = _as_utc_ts(hunt_start)
    days = _hunt_calendar_days("wf", tf)
    # Extra calendar warmup must still fit the bar cap (5m +60d was blowing
    # past MAX_RANGE_BARS → WF disabled every pass → hunt stuck on HOLD).
    max_d = _max_calendar_days_for_tf(tf)
    extra = min(60, max(0, int(max_d) - int(days)))
    t0 = hs - pd.Timedelta(days=int(days) + int(extra))
    warm = range_warmup_bars(tf)
    raw: dict = {}
    for s in symbols:
        try:
            raw[s] = fetch_historical_range(
                s,
                tf,
                t0,
                hs,
                warmup_bars=warm,
                refresh=bool(refresh),
                source=src,
            )
        except Exception as e:
            log.warning("hunt WF fetch %s %s failed: %s", s, tf, e)
    return raw


def _hunt_bars_for_tf(tf: str, *, full: bool = False) -> int:
    """Approx bar count for Health / legacy callers (calendar tier → estimate)."""
    tier = "finalist" if full else "confirm"
    t0, t1 = _hunt_window_bounds(tier, tf)
    return int(
        estimate_range_bars(
            tf,
            t0,
            t1,
            warmup_bars=range_warmup_bars(tf),
        )
    )


def _week_delta_needed(incumbent_ret: float, week_tgt: float = 1.0) -> float:
    """Epsilon only — +100%/week is the SCORE target, never a per-pass promote bar.

    Any strictly better holdout week (or OOS/soft rescue) may promote. Requiring
    multi-pp steps here was holding the hunt in a forever-HOLD loop.
    """
    _ = (incumbent_ret, week_tgt)
    return 0.001


def _lev_slip_stress(cfg: Config) -> Config:
    """Widen slippage for ≥10x so high-lev books must survive honest costs."""
    lev = int(cfg.leverage)
    if lev < 10:
        return cfg
    mult = 2.0 if lev >= 25 else (1.75 if lev >= 15 else 1.5)
    return replace(cfg, slippage_bps=float(cfg.slippage_bps) * mult)


def _pct_of_risk_msg(exp: float) -> str:
    """+0.22 → '+22% of risk' for logs / Live feed."""
    return f"{exp:+.0%} of risk"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _load_autopilot_state() -> dict:
    if not AUTOPILOT_STATE.exists():
        return {}
    try:
        raw = AUTOPILOT_STATE.read_text()
        # Tolerate a torn double-write (two JSON docs concatenated)
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(AUTOPILOT_STATE.read_text())
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}
    except Exception:
        return {}


def _save_autopilot_state(state: dict) -> None:
    """Atomic replace so a paper_tick cannot clobber a half-written hunt save.

    Unique tmp names matter: paper child + hunt parent both save; a shared
    `.json.tmp` lets one process unlink the other's file mid-replace.
    """
    payload = json.dumps(state, indent=2, default=str)
    tmp = AUTOPILOT_STATE.with_name(f".moonshot_autopilot.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(payload)
        os.replace(str(tmp), str(AUTOPILOT_STATE))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _update_autopilot_state(mutator: Callable[[dict], None]) -> dict:
    """Load-modify-save under flock so live_feed / hunt / watchdog cannot clobber."""
    lock_path = AUTOPILOT_STATE.with_name(".moonshot_autopilot.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "a+", encoding="utf-8") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
        st = _load_autopilot_state()
        mutator(st)
        _save_autopilot_state(st)
        return st


def _live_feed(kind: str, message: str, **extra) -> None:
    """Append one clean line for the dashboard Live page."""
    row = {"ts": _utc_now().isoformat(), "kind": kind, "message": message, **extra}
    try:
        with LIVE_FEED.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        # keep file from growing forever
        if LIVE_FEED.stat().st_size > 2_000_000:
            lines = LIVE_FEED.read_text().splitlines()[-500:]
            LIVE_FEED.write_text("\n".join(lines) + "\n")
    except Exception:
        pass

    def _mutate(state: dict) -> None:
        # Paper heartbeat stays under state["live"]. Hunt/search events must NOT
        # overwrite it (that blanked bars_seen/setups and made Live look idle/wrong).
        if kind in ("paper_tick", "paper_restart"):
            state["live"] = row
        elif kind in ("open", "close", "setup", "reject"):
            live = dict(state.get("live") or {})
            live["last_trade_event"] = row
            live["ts"] = row.get("ts") or live.get("ts")
            state["live"] = live
        else:
            state["hunt_event"] = row
        # paper_tick must NOT clear busy during an active hunt pass
        if kind in ("improve_start", "paper_restart"):
            state["busy"] = True
        elif (
            kind in ("hold", "upgrade", "targets_met", "error", "autopilot_on")
            # Do not clear busy while a pass_started_at is still open (abort/HOLD clears it).
            and not state.get("pass_started_at")
        ):
            state["busy"] = False

    _update_autopilot_state(_mutate)


def _env_book_fingerprint() -> tuple[str, str, str]:
    """STRATEGY / TIMEFRAME / LEVERAGE as currently written in .env (+ process)."""
    vals = {"STRATEGY": "", "TIMEFRAME": "", "LEVERAGE": ""}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, _, val = stripped.partition("=")
            key = key.strip()
            if key in vals:
                vals[key] = val.strip().strip('"').strip("'")
    for key in vals:
        if not vals[key]:
            vals[key] = str(os.environ.get(key, "") or "")
    return (vals["STRATEGY"], vals["TIMEFRAME"], vals["LEVERAGE"])


def _verify_env_updates(expected: dict) -> None:
    """Fail loud if .env did not receive the promote keys (apply bugs)."""
    if not ENV_PATH.exists():
        raise ConfigError("promote verify: .env missing after patch")
    got: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, val = stripped.partition("=")
        got[key.strip()] = val.strip()
    bad = []
    for key, want in expected.items():
        if key not in _SAFE_ENV_KEYS:
            continue
        have = got.get(key)
        if have is None or str(have) != str(want):
            bad.append(f"{key}: want={want!r} have={have!r}")
    if bad:
        raise ConfigError("promote verify failed: " + "; ".join(bad[:8]))


def _clear_paper_bleed_circuit(state: dict, *, ts: str | None = None) -> None:
    """New book may trade; old losing closes must not re-arm the halt."""
    state["paper_bleed_halt"] = False
    state.pop("paper_bleed_halt_at", None)
    state.pop("paper_bleed_pnl", None)
    state["paper_bleed_window_start"] = ts or _utc_now().isoformat()
    state["deployed_book"] = {
        "strategy": os.environ.get("STRATEGY"),
        "timeframe": os.environ.get("TIMEFRAME"),
        "leverage": os.environ.get("LEVERAGE"),
        "at": state["paper_bleed_window_start"],
    }


def _patch_env(updates: dict) -> None:
    """Update only allow-listed knobs in .env. Never touches live/API keys."""
    clean = {k: v for k, v in updates.items() if k in _SAFE_ENV_KEYS}
    if "LIVE_TRADING_ENABLED" in updates:
        raise ConfigError("autopilot refuses to change LIVE_TRADING_ENABLED")
    if not clean:
        return
    lines: list[str] = []
    if ENV_PATH.exists():
        lines = ENV_PATH.read_text().splitlines()
    found = set()
    out = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in clean:
            out.append(f"{key}={clean[key]}")
            found.add(key)
        else:
            out.append(line)
    for key, val in clean.items():
        if key not in found:
            out.append(f"{key}={val}")
    ENV_PATH.write_text("\n".join(out) + "\n")
    for key, val in clean.items():
        os.environ[key] = str(val)
    _verify_env_updates(clean)


def _cfg_to_knobs(cfg: Config) -> dict:
    """Full tradable knob dump — history / pending_promote must keep every field."""
    return {
        "strategy": cfg.strategy,
        "timeframe": cfg.timeframe,
        "leverage": int(cfg.leverage),
        "adx_enter": cfg.adx_enter,
        "adx_exit": cfg.adx_exit,
        "volume_multiplier": cfg.volume_multiplier,
        "breakout_buffer_atr": cfg.breakout_buffer_atr,
        "breakout_lookback": cfg.breakout_lookback,
        "trail_atr": cfg.trail_atr_multiple,
        "initial_stop_atr": cfg.initial_stop_atr,
        "cooldown_bars": cfg.cooldown_bars,
        "require_rising_adx": cfg.require_rising_adx,
        "require_squeeze": cfg.require_squeeze,
        "require_rsi_veto": cfg.require_rsi_veto,
        "require_btc_regime": cfg.require_btc_regime,
        "require_h4_bias": bool(getattr(cfg, "require_h4_bias", False)),
        "require_daily_align": bool(getattr(cfg, "require_daily_align", False)),
        "require_weekly_bias": bool(getattr(cfg, "require_weekly_bias", False)),
        "ema_weekly_period": int(getattr(cfg, "ema_weekly_period", 10) or 10),
        "rsi_long_max": cfg.rsi_long_max,
        "rsi_short_min": cfg.rsi_short_min,
        "squeeze_lookback": cfg.squeeze_lookback,
        "enable_partial": cfg.enable_partial,
        "partial_at_r": cfg.partial_at_r,
        "partial_pct": cfg.partial_pct,
        "enable_time_stop": cfg.enable_time_stop,
        "time_stop_bars": cfg.time_stop_bars,
        "time_stop_min_r": cfg.time_stop_min_r,
        "trail_atr_loser": cfg.trail_atr_loser,
        "use_asymmetric_trail": cfg.use_asymmetric_trail,
        "risk_pct": cfg.risk_pct,
        "initial_margin_pct": cfg.initial_margin_pct,
        "aggressive_margin_pct": cfg.aggressive_margin_pct,
        "max_daily_loss_pct": cfg.max_daily_loss_pct,
        "max_peak_drawdown_pct": cfg.max_peak_drawdown_pct,
        "mom_lookback": cfg.mom_lookback,
        "mom_min_atr": cfg.mom_min_atr,
        "max_account_leverage": cfg.max_account_leverage,
        "max_net_leverage": cfg.max_net_leverage,
        "max_open_positions": cfg.max_open_positions,
        "h4_ema_fast": cfg.h4_ema_fast,
        "h4_ema_slow": cfg.h4_ema_slow,
        "ema_entry_period": cfg.ema_entry_period,
        "pullback_lookback": cfg.pullback_lookback,
        "btc_adx_min": cfg.btc_adx_min,
        "breadth_min": cfg.breadth_min,
    }


def _cfg_to_env_updates(cfg: Config) -> dict:
    return {
        "LEVERAGE": str(int(cfg.leverage)),
        "MAX_ACCOUNT_LEVERAGE": str(int(cfg.max_account_leverage)),
        "MAX_NET_LEVERAGE": str(cfg.max_net_leverage),
        "MAX_OPEN_POSITIONS": str(int(cfg.max_open_positions)),
        "RISK_PCT": str(cfg.risk_pct),
        "INITIAL_MARGIN_PCT": str(cfg.initial_margin_pct),
        "AGGRESSIVE_MARGIN_PCT": str(cfg.aggressive_margin_pct),
        "MAX_DAILY_LOSS_PCT": str(cfg.max_daily_loss_pct),
        "MAX_PEAK_DRAWDOWN_PCT": str(cfg.max_peak_drawdown_pct),
        "STREAK_LOSSES": str(int(cfg.streak_losses)),
        "STREAK_PAUSE_HOURS": str(cfg.streak_pause_hours),
        "EXPECTANCY_FLOOR": str(cfg.expectancy_floor),
        "ENABLE_PYRAMIDING": "1" if cfg.enable_pyramiding else "0",
        "ENABLE_PARTIAL": "1" if cfg.enable_partial else "0",
        "PARTIAL_AT_R": str(cfg.partial_at_r),
        "PARTIAL_PCT": str(cfg.partial_pct),
        "ADX_ENTER": str(cfg.adx_enter),
        "ADX_EXIT": str(cfg.adx_exit),
        "VOLUME_MULTIPLIER": str(cfg.volume_multiplier),
        "BREAKOUT_BUFFER_ATR": str(cfg.breakout_buffer_atr),
        "BREAKOUT_LOOKBACK": str(int(cfg.breakout_lookback)),
        "TRAIL_ATR": str(cfg.trail_atr_multiple),
        "TRAIL_ATR_LOSER": str(cfg.trail_atr_loser),
        "USE_ASYMMETRIC_TRAIL": "1" if cfg.use_asymmetric_trail else "0",
        "COOLDOWN_BARS": str(int(cfg.cooldown_bars)),
        "REQUIRE_RISING_ADX": "1" if cfg.require_rising_adx else "0",
        "STRATEGY": str(cfg.strategy),
        # Write actual engine TF (3m/5m/15m). 4H bias is always resampled from it.
        "TIMEFRAME": str(cfg.timeframe) if str(cfg.timeframe) in HUNT_TF_ALLOW else "15m",
        "MOM_LOOKBACK": str(int(cfg.mom_lookback)),
        "MOM_MIN_ATR": str(cfg.mom_min_atr),
        "INITIAL_STOP_ATR": str(cfg.initial_stop_atr),
        "H4_EMA_FAST": str(int(cfg.h4_ema_fast)),
        "H4_EMA_SLOW": str(int(cfg.h4_ema_slow)),
        "H4_ADX_MIN_SHORT": str(cfg.h4_adx_min_short),
        "EMA_ENTRY_PERIOD": str(int(cfg.ema_entry_period)),
        "PULLBACK_LOOKBACK": str(int(cfg.pullback_lookback)),
        "BTC_ADX_MIN": str(cfg.btc_adx_min),
        "BREADTH_MIN": str(cfg.breadth_min),
        "REQUIRE_RSI_VETO": "1" if cfg.require_rsi_veto else "0",
        "RSI_PERIOD": str(int(cfg.rsi_period)),
        "RSI_LONG_MAX": str(cfg.rsi_long_max),
        "RSI_SHORT_MIN": str(cfg.rsi_short_min),
        "REQUIRE_SQUEEZE": "1" if cfg.require_squeeze else "0",
        "SQUEEZE_LOOKBACK": str(int(cfg.squeeze_lookback)),
        "BB_PERIOD": str(int(cfg.bb_period)),
        "BB_STD": str(cfg.bb_std),
        "KC_PERIOD": str(int(cfg.kc_period)),
        "KC_ATR_MULT": str(cfg.kc_atr_mult),
        "REQUIRE_BTC_REGIME": "1" if cfg.require_btc_regime else "0",
        "REQUIRE_H4_BIAS": "1" if getattr(cfg, "require_h4_bias", False) else "0",
        "REQUIRE_DAILY_ALIGN": "1" if getattr(cfg, "require_daily_align", False) else "0",
        "REQUIRE_WEEKLY_BIAS": "1" if getattr(cfg, "require_weekly_bias", False) else "0",
        "EMA_WEEKLY_PERIOD": str(int(getattr(cfg, "ema_weekly_period", 10) or 10)),
        "ENABLE_TIME_STOP": "1" if cfg.enable_time_stop else "0",
        "TIME_STOP_BARS": str(int(cfg.time_stop_bars)),
        "TIME_STOP_MIN_R": str(cfg.time_stop_min_r),
    }
