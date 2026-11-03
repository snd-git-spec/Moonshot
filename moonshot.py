"""
moonshot.py (v3) - Regime-filtered, conditional-pyramiding trend bot for MEXC futures.

One file. One decision engine shared by backtest, paper and live, so what you
test is what you run.

What changed versus v2 (all of these were backtest-validity bugs):
  * No lookahead: indicators are precomputed once; the daily filter only ever
    uses COMPLETED daily bars; signals fire on bar close and fill at the NEXT
    bar's open.
  * Per-lot accounting: pyramided positions track an average entry, so adds are
    no longer credited at the original entry price.
  * R-multiples use the real initial dollar risk (qty * stop distance).
  * One chronological event loop across all symbols (shared equity, shared
    drawdown / daily-loss / kill-switch state).
  * Regime is re-evaluated every bar, so adds and regime exits can actually fire.
  * Gap-aware stop fills, slippage, taker fees and funding are all modelled.
  * Separate entry trigger (Donchian breakout + volume) from the regime filter,
    ADX hysteresis to stop regime flicker, cooldown after exits.
  * Quality filters for higher win rate: Bollinger/Keltner squeeze, RSI exhaustion
    veto, and BTC regime gate on altcoin entries.
  * Pyramiding is sized by a "free-roll" rule: after the add, the position must
    still be net-profitable if the tightened stop is hit.
  * Metrics: Sharpe, Sortino, bootstrap CI on expectancy, in-sample vs
    out-of-sample split.
  * Real paper loop, restart-safe position state, reconcile against the broker.

USAGE
-----
    python moonshot.py selftest
    python moonshot.py backtest --synthetic          # smoke test only
    python moonshot.py backtest                      # real MEXC data (needs ccxt)
    python moonshot.py paper
    python moonshot.py autopilot                     # keep paper running + safe self-improve
    LIVE_TRADING_ENABLED=1 python moonshot.py live   # keys required

ENV
---
    MEXC_API_KEY, MEXC_API_SECRET, LIVE_TRADING_ENABLED
    STARTING_EQUITY (50), LEVERAGE (5), AGGRESSIVE_MODE (0), MAX_ACCOUNT_LEVERAGE (5)
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, LOG_LEVEL

DISCLAIMER
----------
Educational software. Leveraged futures can lose your entire balance. The live
broker adapter has NOT been verified against the live MEXC API; test it with
the smallest possible size first. Nothing here is financial advice.
"""

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

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

log = logging.getLogger("moonshot")

# ============================================================
# 1. CORE TYPES
# ============================================================


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class RegimeState(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class ExitReason(str, Enum):
    STOP = "stop"
    TRAILING = "trailing"
    LIQUIDATION = "liquidation"
    REGIME_INVALID = "regime_invalid"
    TIME_STOP = "time_stop"
    KILL_SWITCH = "kill_switch"
    END_OF_DATA = "end_of_data"


class ActionKind(str, Enum):
    EXIT = "exit"
    PARTIAL = "partial"
    ADD = "add"
    MOVE_STOP = "move_stop"


class AddDecision(str, Enum):
    ALLOW = "allow"
    BLOCK_MAX_ADDS = "block_max_adds"
    BLOCK_R_THRESHOLD = "block_r_threshold"
    BLOCK_REGIME_INVALID = "block_regime_invalid"
    BLOCK_LEVERAGE_CAP = "block_leverage_cap"
    BLOCK_NO_FLOATING_PROFIT = "block_no_floating_profit"
    BLOCK_TRADING_HALTED = "block_trading_halted"
    BLOCK_STOP_WIDEN = "block_stop_widen"
    BLOCK_SIZE_TOO_SMALL = "block_size_too_small"


class BotError(Exception):
    """Base class for bot errors."""


class RiskError(BotError):
    pass


class ExecutionError(BotError):
    pass


class ConfigError(BotError):
    pass


class DataError(BotError):
    pass


@dataclass(frozen=True)
class InstrumentSpec:
    """Exchange constraints for one symbol. Quantities are in BASE units."""

    symbol: str = ""
    qty_step: float = 0.0  # 0 = continuous (backtests)
    min_qty: float = 0.0
    contract_size: float = 1.0

    def round_qty(self, qty: float) -> float:
        if qty <= 0 or not math.isfinite(qty):
            return 0.0
        if self.qty_step > 0:
            qty = math.floor(qty / self.qty_step + 1e-9) * self.qty_step
        return qty if qty >= self.min_qty and qty > 0 else 0.0


@dataclass
class Signal:
    symbol: str
    direction: Direction
    ref_price: float  # bar close that produced the signal
    atr: float
    regime: RegimeState
    reason: str = ""


@dataclass
class Action:
    kind: ActionKind
    qty: float = 0.0
    price: float = 0.0  # new stop for ADD / MOVE_STOP
    reason: Optional[ExitReason] = None


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    timestamp: datetime


@dataclass
class BrokerPosition:
    direction: Direction
    qty: float
    entry_price: float


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    qty: float = 0.0
    margin: float = 0.0
    leverage: int = 0
    risk_dollars: float = 0.0


@dataclass
class Lot:
    price: float
    qty: float
    time: datetime


@dataclass
class AddRecord:
    price: float
    qty: float
    stop_at_add: float
    r_at_add: float
    floating_profit_used: float
    time: datetime


@dataclass
class TradeResult:
    symbol: str
    direction: Direction
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    initial_qty: float
    pnl: float
    r_multiple: float
    liquidated: bool
    exit_reason: ExitReason
    max_adds: int
    partial_taken: bool
    fees: float
    funding: float

    @property
    def duration_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600.0


def _pandas_tf(tf: str) -> str:
    """Map exchange timeframes to pandas offsets (15m → 15min; 1h stays 1h)."""
    s = str(tf).strip().lower()
    if s.endswith("m") and not s.endswith("min") and not s.endswith("ms"):
        # 'm' alone means month in modern pandas — minutes need 'min'
        num = s[:-1] or "1"
        if num.isdigit():
            return f"{num}min"
    return s


# ============================================================
# 2. CONFIG
# ============================================================


@dataclass
class Config:
    # account
    starting_equity: float = 50.0
    aggressive_mode: bool = False
    leverage: int = 5  # 5x passed extensive IS/OOS; 10x failed OOS on same sample
    max_account_leverage: float = 5.0
    max_open_positions: int = 4
    # Dollar growth needs risk_pct to actually bind. Old defaults (2% risk + 10% margin
    # × 5x) capped notional so risk above ~2% did nothing → $50→$52 over months.
    risk_pct: float = 0.02  # start conservative; earn higher risk after soft+hard prove edge
    initial_margin_pct: float = 0.25  # must be high enough that q_margin ≥ q_risk
    aggressive_margin_pct: float = 0.40
    # |sum(signed notional)| / equity may not exceed this (correlated-book brake)
    max_net_leverage: float = 3.0
    min_notional: float = 5.0  # USDT; check MEXC's real minimum per contract
    maintenance_margin_rate: float = 0.005

    # risk / kill switches
    max_daily_loss_pct: float = 0.15  # halts NEW risk until next UTC day
    max_peak_drawdown_pct: float = 0.45  # permanent kill + flatten
    streak_losses: int = 5  # consecutive losses -> temporary pause
    streak_pause_hours: float = 48.0
    expectancy_window: int = 30
    # 0.0 kills 15m books on normal noise (creates "kill illusion" OOS). Soft floor.
    expectancy_floor: float = -0.15  # rolling expectancy below this -> kill

    # strategy: trend_mtf (4H bias + 15m timing), momentum_15, or regime_breakout
    strategy: str = "trend_mtf"
    timeframe: str = "15m"
    adx_period: int = 14
    adx_enter: float = 20.0
    adx_exit: float = 14.0
    ema_htf_period: int = 200
    ema_daily_period: int = 50
    # 4H trend bias (resampled from the engine timeframe; completed bars only)
    h4_ema_fast: int = 20
    h4_ema_slow: int = 50
    h4_adx_min_short: float = 20.0  # shorts need stronger 4H trend confirmation
    ema_entry_period: int = 20      # 15m EMA for pullback / resume timing
    rsi_rule_min: float = 70.0      # momentum_rule: RSI must exceed this
    btc_adx_min: float = 0.0        # chop filter: skip all entries when BTC ADX below this (0 = off)
    breadth_min: float = 0.0        # chop filter: need |share of coins trending| >= this, same side
    pullback_lookback: int = 4      # bars to look back for a touch of ema_entry
    breakout_lookback: int = 18
    volume_lookback: int = 20
    volume_multiplier: float = 1.4
    breakout_buffer_atr: float = 0.15
    atr_period: int = 14
    initial_stop_atr: float = 1.5
    trail_atr_multiple: float = 2.0
    trail_atr_loser: float = 1.25  # tighter trail while trade is underwater
    use_asymmetric_trail: bool = False
    cooldown_bars: int = 2
    exit_on_regime_invalid: bool = True
    require_adx_for_entry: bool = True
    require_rising_adx: bool = False
    # momentum continuation (15m): move over N bars in ATR units + volume
    mom_lookback: int = 8
    mom_min_atr: float = 0.5

    # quality filters (raise win rate: fewer fake breakouts)
    rsi_period: int = 14
    rsi_long_max: float = 70.0   # block longs when RSI already extended
    rsi_short_min: float = 30.0  # block shorts when RSI already extended
    require_rsi_veto: bool = False  # hunt-optional; stacking with squeeze/BTC starved samples
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_atr_mult: float = 1.5
    squeeze_lookback: int = 8    # squeeze must have fired within this many bars
    require_squeeze: bool = False
    require_btc_regime: bool = True

    # time stop — dump stagnant trades that never reach +min_r (hunt-optional)
    enable_time_stop: bool = False
    time_stop_bars: int = 12
    time_stop_min_r: float = 1.0

    # pyramiding
    enable_pyramiding: bool = True
    max_adds_per_trade: int = 2
    add_trigger_r: tuple = (2.0, 3.0)
    min_add_qty_pct: float = 0.25
    addon_giveback_r: float = 0.0  # 0 = adds must be "free-roll" at the tightened stop

    # partial profit — hunt-optional (forced early partial hurt some windows)
    enable_partial: bool = False
    partial_at_r: float = 1.0
    partial_pct: float = 0.50

    # costs (VERIFY against MEXC's current futures fee schedule)
    fee_taker: float = 0.0002  # MEXC USDT-M taker per ccxt (maker is 0); re-verify before live
    slippage_bps: float = 3.0
    funding_rate: float = 0.0001  # per interval; positive = longs pay shorts
    funding_interval_hours: int = 8

    # live loop
    poll_seconds: int = 30
    # 0 = auto-scale from timeframe (≥~70d so daily EMA warms on 15m/5m/3m)
    live_history_bars: int = 0
    db_path: str = "moonshot.db"
    log_level: str = "INFO"

    def __post_init__(self) -> None:
        if self.leverage < 1:
            raise ConfigError("leverage must be >= 1")
        if not 0 < self.risk_pct <= 0.12:
            raise ConfigError("risk_pct must be in (0, 0.12]")
        if not 0.05 <= self.initial_margin_pct <= 0.80:
            raise ConfigError("initial_margin_pct must be in [0.05, 0.80]")
        if len(self.add_trigger_r) < self.max_adds_per_trade:
            raise ConfigError("add_trigger_r needs one entry per allowed add")
        if self.adx_exit > self.adx_enter:
            raise ConfigError("adx_exit must be <= adx_enter")
        if int(self.live_history_bars) <= 0:
            object.__setattr__(
                self, "live_history_bars", live_history_bars_for_tf(self.timeframe),
            )

    @property
    def slip(self) -> float:
        return self.slippage_bps / 10_000.0

    @property
    def pandas_tf(self) -> str:
        """Exchange TF → pandas offset ('15m' is minutes; bare 'm' means month-end)."""
        return _pandas_tf(self.timeframe)

    @property
    def bar_td(self) -> pd.Timedelta:
        return pd.Timedelta(_pandas_tf(self.timeframe))

    @classmethod
    def from_env(cls) -> "Config":
        lev = int(os.getenv("LEVERAGE", "5"))
        return cls(
            starting_equity=float(os.getenv("STARTING_EQUITY", "50.0")),
            aggressive_mode=os.getenv("AGGRESSIVE_MODE", "0") == "1",
            leverage=lev,
            max_account_leverage=float(os.getenv("MAX_ACCOUNT_LEVERAGE", str(lev))),
            risk_pct=float(os.getenv("RISK_PCT", "0.02")),
            max_open_positions=int(os.getenv("MAX_OPEN_POSITIONS", "4")),
            max_net_leverage=float(os.getenv("MAX_NET_LEVERAGE", "3.0")),
            initial_margin_pct=float(os.getenv("INITIAL_MARGIN_PCT", "0.25")),
            aggressive_margin_pct=float(os.getenv("AGGRESSIVE_MARGIN_PCT", "0.40")),
            max_daily_loss_pct=float(os.getenv("MAX_DAILY_LOSS_PCT", "0.15")),
            max_peak_drawdown_pct=float(os.getenv("MAX_PEAK_DRAWDOWN_PCT", "0.45")),
            streak_losses=int(os.getenv("STREAK_LOSSES", "5")),
            streak_pause_hours=float(os.getenv("STREAK_PAUSE_HOURS", "48")),
            expectancy_floor=float(os.getenv("EXPECTANCY_FLOOR", "-0.15")),
            enable_pyramiding=os.getenv("ENABLE_PYRAMIDING", "1") == "1",
            enable_partial=os.getenv("ENABLE_PARTIAL", "1") == "1",
            partial_at_r=float(os.getenv("PARTIAL_AT_R", "1.0")),
            partial_pct=float(os.getenv("PARTIAL_PCT", "0.50")),
            adx_enter=float(os.getenv("ADX_ENTER", "20")),
            adx_exit=float(os.getenv("ADX_EXIT", "14")),
            volume_multiplier=float(os.getenv("VOLUME_MULTIPLIER", "1.4")),
            breakout_buffer_atr=float(os.getenv("BREAKOUT_BUFFER_ATR", "0.15")),
            breakout_lookback=int(os.getenv("BREAKOUT_LOOKBACK", "18")),
            trail_atr_multiple=float(os.getenv("TRAIL_ATR", "2.0")),
            trail_atr_loser=float(os.getenv("TRAIL_ATR_LOSER", "1.25")),
            use_asymmetric_trail=os.getenv("USE_ASYMMETRIC_TRAIL", "0") == "1",
            cooldown_bars=int(os.getenv("COOLDOWN_BARS", "2")),
            require_rising_adx=os.getenv("REQUIRE_RISING_ADX", "0") == "1",
            rsi_period=int(os.getenv("RSI_PERIOD", "14")),
            rsi_long_max=float(os.getenv("RSI_LONG_MAX", "70")),
            rsi_short_min=float(os.getenv("RSI_SHORT_MIN", "30")),
            require_rsi_veto=os.getenv("REQUIRE_RSI_VETO", "0") == "1",
            bb_period=int(os.getenv("BB_PERIOD", "20")),
            bb_std=float(os.getenv("BB_STD", "2.0")),
            kc_period=int(os.getenv("KC_PERIOD", "20")),
            kc_atr_mult=float(os.getenv("KC_ATR_MULT", "1.5")),
            squeeze_lookback=int(os.getenv("SQUEEZE_LOOKBACK", "8")),
            require_squeeze=os.getenv("REQUIRE_SQUEEZE", "0") == "1",
            require_btc_regime=os.getenv("REQUIRE_BTC_REGIME", "1") == "1",
            enable_time_stop=os.getenv("ENABLE_TIME_STOP", "1") == "1",
            time_stop_bars=int(os.getenv("TIME_STOP_BARS", "8")),
            time_stop_min_r=float(os.getenv("TIME_STOP_MIN_R", "1.0")),
            strategy=os.getenv("STRATEGY", "trend_mtf").strip().lower(),
            timeframe=os.getenv("TIMEFRAME", "15m"),
            mom_lookback=int(os.getenv("MOM_LOOKBACK", "8")),
            mom_min_atr=float(os.getenv("MOM_MIN_ATR", "0.5")),
            initial_stop_atr=float(os.getenv("INITIAL_STOP_ATR", "1.5")),
            fee_taker=float(os.getenv("FEE_TAKER", "0.0002")),  # MEXC USDT-M taker 0.02%
            slippage_bps=float(os.getenv("SLIPPAGE_BPS", "3")),
            h4_ema_fast=int(os.getenv("H4_EMA_FAST", "20")),
            h4_ema_slow=int(os.getenv("H4_EMA_SLOW", "50")),
            h4_adx_min_short=float(os.getenv("H4_ADX_MIN_SHORT", "20")),
            ema_entry_period=int(os.getenv("EMA_ENTRY_PERIOD", "20")),
            rsi_rule_min=float(os.getenv("RSI_RULE_MIN", "70")),
            btc_adx_min=float(os.getenv("BTC_ADX_MIN", "0")),
            breadth_min=float(os.getenv("BREADTH_MIN", "0")),
            pullback_lookback=int(os.getenv("PULLBACK_LOOKBACK", "4")),
            live_history_bars=int(os.getenv("LIVE_HISTORY_BARS", "0")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )


# ============================================================
# 3. INDICATORS + FEATURES (precomputed once, no lookahead)
# ============================================================


def ema(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(span=period, adjust=False, min_periods=period).mean()


def wilder(s: pd.Series, period: int) -> pd.Series:
    return s.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    a = atr(df, period)
    plus_di = 100 * wilder(plus_dm, period) / a
    minus_di = 100 * wilder(minus_dm, period) / a
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, period)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0.0), period)
    loss = wilder((-delta).clip(lower=0.0), period)
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def bollinger(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple:
    mid = close.rolling(period, min_periods=period).mean()
    sd = close.rolling(period, min_periods=period).std(ddof=0)
    return mid, mid + num_std * sd, mid - num_std * sd


def keltner(df: pd.DataFrame, period: int = 20, atr_mult: float = 1.5) -> tuple:
    mid = ema(df["close"], period)
    band = atr_mult * atr(df, period)
    return mid, mid + band, mid - band


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Aggregate OHLCV to a higher timeframe (pandas offset, e.g. '4h' or '1D')."""
    return (
        df.resample(rule)
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
    )


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    return resample_ohlcv(df, "1D")


def hysteresis(values: np.ndarray, on: float, off: float) -> np.ndarray:
    out = np.zeros(len(values), dtype=bool)
    state = False
    for i, v in enumerate(values):
        if v == v:  # not NaN
            if v >= on:
                state = True
            elif v < off:
                state = False
        out[i] = state
    return out


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """
    Every column at row t uses only information available at the CLOSE of bar t.
    Daily and 4H filters use the previous COMPLETED higher-TF bar (shift(1) before ffill).
    """
    f = df[["open", "high", "low", "close", "volume"]].astype(float).copy()
    f["ema_htf"] = ema(f["close"], cfg.ema_htf_period)
    f["ema_entry"] = ema(f["close"], max(2, int(cfg.ema_entry_period)))
    f["ema_50"] = ema(f["close"], 50)
    f["ema_200"] = ema(f["close"], 200)
    f["atr"] = atr(f, cfg.atr_period)
    f["adx"] = adx(f, cfg.adx_period)
    f["vol_ratio"] = f["volume"] / f["volume"].rolling(cfg.volume_lookback).mean().shift(1)
    f["don_high"] = f["high"].rolling(cfg.breakout_lookback).max().shift(1)
    f["don_low"] = f["low"].rolling(cfg.breakout_lookback).min().shift(1)
    # Momentum in ATR units over mom_lookback bars (available at bar close)
    mom_n = max(2, int(cfg.mom_lookback))
    f["mom_atr"] = (f["close"] - f["close"].shift(mom_n)) / f["atr"].replace(0, np.nan)

    # RSI exhaustion + Bollinger/Keltner squeeze (TTM-style compression)
    f["rsi"] = rsi(f["close"], cfg.rsi_period)
    _, bb_up, bb_lo = bollinger(f["close"], cfg.bb_period, cfg.bb_std)
    _, kc_up, kc_lo = keltner(f, cfg.kc_period, cfg.kc_atr_mult)
    f["bb_upper"] = bb_up
    f["bb_lower"] = bb_lo
    f["kc_upper"] = kc_up
    f["kc_lower"] = kc_lo
    squeeze_now = (bb_up < kc_up) & (bb_lo > kc_lo)
    f["squeeze"] = squeeze_now.fillna(False)
    # Allow entry if we were compressed recently (breakout after coil)
    f["squeeze_ok"] = (
        f["squeeze"].astype(float)
        .rolling(cfg.squeeze_lookback, min_periods=1).max()
        .fillna(0.0) > 0.0
    )

    # 15m pullback timing: touched EMA_entry recently (within ~0.75 ATR)
    atr_safe = f["atr"].replace(0, np.nan)
    pb_n = max(2, int(cfg.pullback_lookback))
    near_long = (f["low"] <= f["ema_entry"] + 0.35 * atr_safe) & (
        f["low"] >= f["ema_entry"] - 0.75 * atr_safe
    )
    near_short = (f["high"] >= f["ema_entry"] - 0.35 * atr_safe) & (
        f["high"] <= f["ema_entry"] + 0.75 * atr_safe
    )
    f["pullback_long"] = near_long.fillna(False).astype(float).rolling(pb_n, min_periods=1).max() > 0
    f["pullback_short"] = near_short.fillna(False).astype(float).rolling(pb_n, min_periods=1).max() > 0

    daily = resample_to_daily(f)
    daily["ema_d"] = ema(daily["close"], cfg.ema_daily_period)
    prev = daily[["close", "ema_d"]].shift(1).rename(columns={"close": "d_close"})
    prev = prev.reindex(f.index, method="ffill")
    f["d_close"] = prev["d_close"]
    f["d_ema"] = prev["ema_d"]

    # Completed 4H trend bias (no lookahead): EMA stack on closed 4H bars only
    h4 = resample_ohlcv(f, "4h")
    h4["h4_ema_fast"] = ema(h4["close"], max(2, int(cfg.h4_ema_fast)))
    h4["h4_ema_slow"] = ema(h4["close"], max(3, int(cfg.h4_ema_slow)))
    h4["h4_adx"] = adx(h4, cfg.adx_period)
    stack_up = (h4["close"] > h4["h4_ema_slow"]) & (h4["h4_ema_fast"] > h4["h4_ema_slow"])
    stack_dn = (h4["close"] < h4["h4_ema_slow"]) & (h4["h4_ema_fast"] < h4["h4_ema_slow"])
    h4["h4_bias"] = np.where(stack_up, 1.0, np.where(stack_dn, -1.0, 0.0))
    h4_prev = h4[["h4_ema_fast", "h4_ema_slow", "h4_adx", "h4_bias"]].shift(1)
    h4_prev = h4_prev.reindex(f.index, method="ffill")
    f["h4_ema_fast"] = h4_prev["h4_ema_fast"]
    f["h4_ema_slow"] = h4_prev["h4_ema_slow"]
    f["h4_adx"] = h4_prev["h4_adx"]
    f["h4_bias"] = h4_prev["h4_bias"].fillna(0.0)

    valid = f[["ema_htf", "atr", "adx", "d_ema"]].notna().all(axis=1).to_numpy()
    adx_on = hysteresis(f["adx"].to_numpy(), cfg.adx_enter, cfg.adx_exit)
    up = adx_on & (f["close"] > f["ema_htf"]).to_numpy() & (f["d_close"] > f["d_ema"]).to_numpy()
    dn = adx_on & (f["close"] < f["ema_htf"]).to_numpy() & (f["d_close"] < f["d_ema"]).to_numpy()
    f["regime"] = np.where(
        ~valid,
        RegimeState.UNKNOWN.value,
        np.where(up, RegimeState.TRENDING_UP.value,
                 np.where(dn, RegimeState.TRENDING_DOWN.value, RegimeState.RANGING.value)),
    )
    f["adx_slope"] = f["adx"].diff()
    # Filled later by prepare_data when BTC is available; 0 = neutral/unknown
    f["btc_trend"] = 0.0
    return f


def _is_btc_symbol(symbol: str) -> bool:
    return str(symbol).upper().startswith("BTC/")


def _btc_symbol_in(symbols) -> Optional[str]:
    for s in symbols:
        if _is_btc_symbol(s):
            return s
    return None


def _regime_to_trend(regime: Any) -> float:
    if regime == RegimeState.TRENDING_UP.value or regime == RegimeState.TRENDING_UP:
        return 1.0
    if regime == RegimeState.TRENDING_DOWN.value or regime == RegimeState.TRENDING_DOWN:
        return -1.0
    return 0.0


@dataclass
class SymbolData:
    symbol: str
    bars: list  # list of Bar namedtuples (Index, open, high, low, close, ...)
    index: dict  # Timestamp -> position in bars
    spec: InstrumentSpec


def prepare_symbol(symbol: str, df: pd.DataFrame, cfg: Config,
                   spec: Optional[InstrumentSpec] = None,
                   feats: Optional[pd.DataFrame] = None) -> SymbolData:
    frame = feats if feats is not None else build_features(df, cfg)
    bars = list(frame.itertuples(name="Bar"))
    return SymbolData(symbol, bars, {b.Index: i for i, b in enumerate(bars)},
                      spec or InstrumentSpec(symbol))


def prepare_data(raw: dict, cfg: Config) -> dict:
    """Build per-symbol features; stamp BTC trend onto alts when enabled."""
    feats = {s: build_features(df, cfg) for s, df in raw.items()}
    btc = _btc_symbol_in(feats.keys())
    if cfg.require_btc_regime and btc is not None:
        # Prefer completed 4H bias for MTF; fall back to 15m/daily regime map
        if "h4_bias" in feats[btc].columns:
            btc_trend = feats[btc]["h4_bias"].astype(float).fillna(0.0)
        else:
            btc_trend = feats[btc]["regime"].map(_regime_to_trend).astype(float)
        for s, f in feats.items():
            if _is_btc_symbol(s):
                f["btc_trend"] = btc_trend
            else:
                # Align on timestamp; forward-fill so missing BTC bars don't flash 0 mid-trend
                f["btc_trend"] = btc_trend.reindex(f.index).ffill().fillna(0.0)
    # Market-wide context (no lookahead: values at bar close). Used by chop filters.
    if btc is not None:
        btc_adx = feats[btc]["adx"].astype(float)
    else:
        btc_adx = pd.concat([f["adx"] for f in feats.values()], axis=1).mean(axis=1)
    breadth = pd.concat([f["h4_bias"].astype(float) for f in feats.values()], axis=1).mean(axis=1)
    for f in feats.values():
        f["btc_adx"] = btc_adx.reindex(f.index).ffill().fillna(0.0)
        f["breadth"] = breadth.reindex(f.index).ffill().fillna(0.0)
    return {s: prepare_symbol(s, raw[s], cfg, feats=f) for s, f in feats.items()}


# ============================================================
# 4. DATA LOADERS
# ============================================================

_TF_UNITS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def timeframe_ms(tf: str) -> int:
    return int(tf[:-1]) * _TF_UNITS[tf[-1]]


def live_history_bars_for_tf(tf: str, *, days: float = 70.0) -> int:
    """Bars needed so daily EMA50 (+ buffer) can warm on this TF.

    Comment historically assumed 4h bars (1200 ≈ 200d). On 15m that is only ~12d
    and leaves regime stuck at ``unknown`` → every paper exit is regime_invalid.
    """
    minutes = max(1, int(timeframe_ms(str(tf).strip().lower()) // 60_000))
    n = int(max(2000, (float(days) * 24.0 * 60.0) / minutes))
    return int(min(n, 20_000))


# TFs some venues lack natively → build from a finer base TF.
# MEXC swap: no 3m, no 2h. Binance spot: has native 2h (see _resample_base_tf).
_RESAMPLE_BASE = {"3m": "1m", "2h": "1h"}
_RESAMPLE_FROM_1M = frozenset(tf for tf, base in _RESAMPLE_BASE.items() if base == "1m")


def _resample_base_tf(tf: str, *, source: str = "mexc") -> Optional[str]:
    """Return base TF to resample from, or None if ``tf`` is fetched natively."""
    tf = str(tf).strip().lower()
    src = str(source or "mexc").strip().lower()
    # Binance spot exposes native 2h klines.
    if tf == "2h" and src == "binance_spot":
        return None
    return _RESAMPLE_BASE.get(tf)


def _ohlcv_resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate finer OHLCV up to `tf` (e.g. 1m→3m, 1h→2h)."""
    if df is None or df.empty:
        return df
    rule = _pandas_tf(tf)
    out = df.resample(rule, label="left", closed="left").agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna(subset=["open", "high", "low", "close"])
    return out


def paged_ohlcv(ex: Any, symbol: str, tf: str, since_ms: int, until_ms: int,
                page: int = 1000) -> list:
    """Paginate fetch_ohlcv forward from since_ms (exchanges cap one call).

    If ``since_ms`` is before the market listed, MEXC returns ``[]`` on the
    first page — probe forward (and finally with no ``since``) so newer coins
    (SUI, WIF, …) still load whatever history exists.
    """
    rows: list = []
    step = timeframe_ms(tf)
    since = int(since_ms)
    last_seen = -1
    empty_streak = 0
    while since < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=page)
        if not batch:
            empty_streak += 1
            # Pre-listing / gap: jump forward instead of giving up immediately.
            if not rows and empty_streak <= 12:
                # Exponential jump toward now (covers multi-year pre-list windows).
                jump = max(step * page, (until_ms - since) // 4)
                since = min(until_ms - step, since + max(jump, step * 24))
                time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
                continue
            if not rows:
                # Last resort: latest page with no since, then walk… we only
                # need a foothold; caller trims to the requested window.
                batch = ex.fetch_ohlcv(symbol, timeframe=tf, limit=page)
                if not batch:
                    break
                # Restart pagination from the earliest bar in this foothold
                # by walking backward isn't supported — take foothold and
                # continue forward from there (newer side of history).
                rows.extend(batch)
                last_seen = batch[-1][0]
                since = last_seen + step
                empty_streak = 0
                time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
                continue
            break
        empty_streak = 0
        rows.extend(batch)
        newest = batch[-1][0]
        if newest <= last_seen:
            break
        last_seen = newest
        since = newest + step
        time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
    return rows


def ohlcv_to_frame(rows: list, tf: str) -> pd.DataFrame:
    if not rows:
        raise DataError("no OHLCV rows returned")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index().astype(float)
    # drop a still-forming candle
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(milliseconds=timeframe_ms(tf)) <= now]


# Backtest-only price venues (live/paper/hunt always use MEXC via public_exchange).
RANGE_SOURCE_MEXC = "mexc"
RANGE_SOURCE_BINANCE_SPOT = "binance_spot"
RANGE_SOURCE_YAHOO = "yahoo"
RANGE_PRICE_SOURCES = (
    RANGE_SOURCE_MEXC,
    RANGE_SOURCE_BINANCE_SPOT,
    RANGE_SOURCE_YAHOO,
)
RANGE_SOURCE_LABELS = {
    RANGE_SOURCE_MEXC: "MEXC swap (live venue)",
    RANGE_SOURCE_BINANCE_SPOT: "Binance spot",
    RANGE_SOURCE_YAHOO: "Yahoo Finance (daily)",
}


def _normalize_range_source(source: Optional[str]) -> str:
    s = str(source or RANGE_SOURCE_MEXC).strip().lower()
    if s not in RANGE_PRICE_SOURCES:
        raise DataError(
            f"unknown price source {source!r}; "
            f"expected one of {', '.join(RANGE_PRICE_SOURCES)}"
        )
    return s


def _book_base(symbol: str) -> str:
    """BTC from BTC/USDT:USDT or BTC/USDT."""
    return str(symbol).split("/")[0].strip().upper()


def to_binance_spot_symbol(symbol: str) -> str:
    """Map book swap symbol → Binance spot USDT pair."""
    base = _book_base(symbol)
    return f"{base}/USDT"


def to_yahoo_ticker(symbol: str) -> str:
    """Map book symbol → Yahoo crypto ticker (spot USD)."""
    base = _book_base(symbol)
    # Yahoo uses standard BASE-USD for majors and most alts we trade.
    return f"{base}-USD"


def public_exchange() -> Any:
    """MEXC USDT-M — live, paper, hunt, and default Backtest source."""
    try:
        import ccxt  # type: ignore
    except ImportError as e:
        raise DataError("ccxt not installed: pip install ccxt") from e
    return ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def exchange_for_source(source: str = RANGE_SOURCE_MEXC) -> Any:
    """ccxt public client for a Backtest range source (MEXC / Binance spot)."""
    src = _normalize_range_source(source)
    if src == RANGE_SOURCE_YAHOO:
        raise DataError("Yahoo Finance is not a ccxt exchange — use fetch_yahoo_ohlcv_range")
    try:
        import ccxt  # type: ignore
    except ImportError as e:
        raise DataError("ccxt not installed: pip install ccxt") from e
    if src == RANGE_SOURCE_BINANCE_SPOT:
        return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    return ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def _ccxt_symbol_for_source(symbol: str, source: str) -> str:
    src = _normalize_range_source(source)
    if src == RANGE_SOURCE_BINANCE_SPOT:
        return to_binance_spot_symbol(symbol)
    return symbol


# Serialize OHLCV fetches so hunt + paper + backtest do not stampede exchanges.
_OHLCV_LOCK = threading.Lock()


def _cache_stem(symbol: str, tf: str, *, source: str = RANGE_SOURCE_MEXC) -> str:
    src = _normalize_range_source(source)
    return f"{src}_{symbol.replace('/', '_').replace(':', '_')}_{tf}_"


def _read_ohlcv_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _ohlcv_is_stale(df: pd.DataFrame, tf: str, *, max_age_bars: float = 1.0) -> bool:
    """True if last closed bar is behind the wall-clock expected closed bar.

    Default max_age_bars=1 means anything older than the latest closed candle is
    stale (forces tail refresh). A looser value was leaving paper on 21:45 while
    22:00 had already closed — bars froze and setups never fired.
    """
    if df is None or df.empty:
        return True
    step = pd.Timedelta(milliseconds=timeframe_ms(tf))
    last = pd.Timestamp(df.index[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    expected = pd.Timestamp.now(tz="UTC").floor(_pandas_tf(tf)) - step
    slack = step * max(0.0, float(max_age_bars) - 1.0)
    return last < expected - slack


def _tail_merge_ohlcv(df: pd.DataFrame, symbol: str, tf: str, bars: int,
                      ex: Optional[Any] = None) -> pd.DataFrame:
    """Fetch candles after the cached last bar and merge; drop forming candle."""
    if df is None or df.empty:
        return df
    step_ms = timeframe_ms(tf)
    last_ms = int(pd.Timestamp(df.index[-1]).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    since_ms = last_ms + step_ms
    if since_ms >= now_ms:
        return df
    exchange = ex or public_exchange()
    with _OHLCV_LOCK:
        rows = paged_ohlcv(exchange, symbol, tf, since_ms, now_ms)
        if not rows:
            time.sleep(0.35)
            rows = paged_ohlcv(exchange, symbol, tf, since_ms, now_ms)
    if not rows:
        return df
    fresh = ohlcv_to_frame(rows, tf)
    if fresh.empty:
        return df
    out = pd.concat([df, fresh]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    now = pd.Timestamp.now(tz="UTC")
    out = out[out.index + pd.Timedelta(milliseconds=step_ms) <= now]
    return out.tail(max(int(bars), len(out)))


def clear_stale_live_caches(cache_dir: str = ".moonshot_cache",
                            tf: str = "15m", bars: int = 1200) -> int:
    """Delete live-sized CSV caches whose last bar is stale (unblocks paper)."""
    cache = Path(cache_dir)
    if not cache.exists():
        return 0
    removed = 0
    for p in cache.glob(f"*_{tf}_{int(bars)}.csv"):
        try:
            df = _read_ohlcv_csv(p)
            if _ohlcv_is_stale(df, tf):
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    if removed:
        log.info("cleared %d stale %s×%d cache file(s)", removed, tf, bars)
    return removed


def fetch_historical(symbol: str, tf: str = "4h", bars: int = 3000,
                     cache_dir: str = ".moonshot_cache", refresh: bool = False) -> pd.DataFrame:
    """Paginated history with a CSV cache + incremental tail refresh when stale.

    TFs missing on MEXC contracts (``3m``, ``2h``) are resampled from a finer base.
    """
    tf = str(tf).strip().lower()
    base_tf = _resample_base_tf(tf, source="mexc")
    if base_tf is not None:
        ratio = max(1, int(timeframe_ms(tf) // timeframe_ms(base_tf)))
        src_bars = int(bars) * ratio + ratio  # +1 bucket of slack
        raw = fetch_historical(
            symbol, base_tf, src_bars, cache_dir=cache_dir, refresh=refresh,
        )
        if raw is None or raw.empty:
            raise DataError(f"no {base_tf} history to build {tf} for {symbol}")
        built = _ohlcv_resample(raw, tf)
        if built.empty:
            raise DataError(f"resample to {tf} produced empty frame for {symbol}")
        cache = Path(cache_dir)
        cache.mkdir(exist_ok=True)
        path = cache / f"{_cache_stem(symbol, tf)}{int(bars)}.csv"
        out = built.tail(int(bars)) if len(built) > int(bars) else built
        try:
            out.to_csv(path)
        except Exception:
            pass
        log.info(
            "%s %s built from %s (%d → %d bars)",
            symbol, tf, base_tf, len(raw), len(out),
        )
        return out

    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    stem = _cache_stem(symbol, tf)
    path = cache / f"{stem}{bars}.csv"

    def _sibling_frames(min_n: int = 0) -> list[tuple[int, Path, pd.Timestamp]]:
        """Return (n, path, last_ts) sorted by newest last bar, then larger n."""
        found: list[tuple[int, Path, pd.Timestamp]] = []
        for p in cache.glob(f"{stem}*.csv"):
            try:
                n = int(p.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if n < int(min_n):
                continue
            try:
                df_s = _read_ohlcv_csv(p)
                if df_s.empty:
                    continue
                last = pd.Timestamp(df_s.index[-1])
                found.append((n, p, last))
            except Exception:
                continue
        found.sort(key=lambda t: (t[2], t[0]), reverse=True)
        return found

    def _persist(df: pd.DataFrame) -> pd.DataFrame:
        out = df.tail(int(bars)) if len(df) > int(bars) else df
        try:
            out.to_csv(path)
        except Exception:
            pass
        return out

    def _ensure_fresh(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if not _ohlcv_is_stale(df, tf):
            return df.tail(int(bars)) if len(df) > int(bars) else df
        try:
            merged = _tail_merge_ohlcv(df, symbol, tf, bars)
            if merged is not None and not merged.empty:
                return _persist(merged)
        except Exception as e:
            log.warning("%s %s tail refresh failed: %s", symbol, tf, e)
        return df.tail(int(bars)) if len(df) > int(bars) else df

    # Exact cache hit — still tail-refresh if stale so paper never freezes.
    if path.exists() and not refresh:
        try:
            cached = _read_ohlcv_csv(path)
            if len(cached) > 0:
                return _ensure_fresh(cached)
        except Exception:
            pass

    # Prefer newest same-TF sibling (not smallest n) then tail-refresh.
    if not refresh:
        siblings = _sibling_frames(min_n=max(200, int(bars) // 4))
        for n, p, _last in siblings:
            try:
                df = _read_ohlcv_csv(p)
                if df.empty:
                    continue
                if n >= int(bars):
                    df = df.tail(int(bars))
                return _ensure_fresh(df if n >= int(bars) else _persist(_ensure_fresh(df)))
            except Exception:
                continue

    with _OHLCV_LOCK:
        ex = public_exchange()
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - int(bars) * timeframe_ms(tf)
        rows = paged_ohlcv(ex, symbol, tf, since_ms, now_ms)
        if not rows:
            time.sleep(0.5)
            rows = paged_ohlcv(ex, symbol, tf, since_ms, now_ms)

    if not rows:
        shorter = _sibling_frames(min_n=200)
        if shorter:
            n, p, _ = shorter[0]
            df = _read_ohlcv_csv(p)
            if len(df) > 0:
                log.warning(
                    "%s %s: only %d cached bars (asked %d) — using available history",
                    symbol, tf, len(df), bars,
                )
                return _ensure_fresh(df)
        raise DataError(
            f"no OHLCV rows returned ({symbol} {tf}×{bars}). "
            "Exchange returned empty candles — retry, or check symbol/TF."
        )

    df = ohlcv_to_frame(rows, tf)
    if df.empty:
        raise DataError(f"OHLCV empty after candle filter ({symbol} {tf})")
    if len(df) < int(bars) * 0.5:
        log.warning(
            "%s %s: exchange only returned %d/%d bars (coin listed later than window)",
            symbol, tf, len(df), bars,
        )
    return _persist(df)


# Hard cap for calendar-range backtests (per symbol, including warmup).
_MAX_RANGE_BARS = 25_000
MAX_RANGE_BARS = _MAX_RANGE_BARS

# Fine → coarse. Auto picks the densest TF whose estimate fits under the bar cap.
RANGE_TF_LADDER = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d")


def _as_utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def range_warmup_bars(tf: str) -> int:
    """Warmup bar count used by calendar-range fetches for ``tf``."""
    return max(500, min(int(live_history_bars_for_tf(tf)), 3_000))


def estimate_range_bars(tf: str, start, end, *, warmup_bars: int = 0) -> int:
    """Approximate OHLCV rows for ``[start - warmup, end)`` at timeframe ``tf``."""
    t0 = _as_utc_ts(start)
    t1 = _as_utc_ts(end)
    if not (t0 < t1):
        return 0
    warm = int(warmup_bars) if int(warmup_bars) > 0 else range_warmup_bars(tf)
    step_ms = timeframe_ms(str(tf).strip().lower())
    fetch_start = t0 - pd.Timedelta(milliseconds=step_ms * warm)
    fetch_end = min(t1, pd.Timestamp.now(tz="UTC"))
    if fetch_start >= fetch_end:
        return 0
    return int((fetch_end - fetch_start).total_seconds() * 1000 // step_ms) + 2


def pick_range_timeframe(
    start,
    end,
    preferred: str,
    *,
    max_bars: int = _MAX_RANGE_BARS,
) -> tuple[str, int, str]:
    """Pick the densest candle size that still fits ``max_bars`` for the window.

    Walks ``RANGE_TF_LADDER`` fine→coarse and returns the first TF whose
    estimated bar count is ≤ ``max_bars`` (maximizes candles in that calendar
    span). ``preferred`` is the book TF — used in the note when Auto differs.
    Returns ``(timeframe, estimated_bars, note)``.
    """
    preferred = str(preferred).strip().lower() or "15m"
    ladder = list(RANGE_TF_LADDER)
    if preferred not in ladder:
        ladder = sorted(set(ladder + [preferred]), key=lambda t: timeframe_ms(t))
    # Entire ladder fine→coarse — maximize candle count under the cap.
    candidates = sorted(set(ladder), key=lambda t: timeframe_ms(t))

    rejected: list[tuple[str, int]] = []
    for tf in candidates:
        warm = range_warmup_bars(tf)
        n = estimate_range_bars(tf, start, end, warmup_bars=warm)
        if 0 < n <= int(max_bars):
            note = ""
            if tf != preferred:
                why = ""
                if rejected:
                    prev_tf, prev_n = rejected[-1]
                    why = f" {prev_tf} would be ~{prev_n:,} bars (over cap)."
                # Finer than book = packing more candles; coarser = forced by cap.
                if timeframe_ms(tf) < timeframe_ms(preferred):
                    note = (
                        f"Auto packed denser candles {preferred} → {tf} "
                        f"(~{n:,} bars in this window, cap {int(max_bars):,})."
                        f"{why}"
                    )
                else:
                    note = (
                        f"Auto raised candles {preferred} → {tf} "
                        f"(~{n:,} bars in this window, cap {int(max_bars):,})."
                        f"{why}"
                    )
            return tf, n, note
        if n > int(max_bars):
            rejected.append((tf, n))

    tf = candidates[-1]
    warm = range_warmup_bars(tf)
    n = estimate_range_bars(tf, start, end, warmup_bars=warm)
    note = (
        f"Even {tf} needs ~{n:,} bars (cap {int(max_bars):,}). "
        "Narrow the date range."
    )
    return tf, n, note


def fetch_yahoo_ohlcv_range(
    symbol: str,
    fetch_start: pd.Timestamp,
    fetch_end: pd.Timestamp,
) -> pd.DataFrame:
    """Download daily spot OHLCV from Yahoo Finance for a book symbol.

    Returns the same columns/index shape as ``ohlcv_to_frame`` (UTC daily bars).
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as e:
        raise DataError("yfinance not installed: pip install yfinance") from e

    ticker = to_yahoo_ticker(symbol)
    # yfinance end is exclusive-ish; pad one day so the last calendar day arrives.
    start_s = fetch_start.tz_convert("UTC").strftime("%Y-%m-%d")
    end_s = (fetch_end.tz_convert("UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            ticker,
            start=start_s,
            end=end_s,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise DataError(f"Yahoo download failed for {ticker} ({symbol}): {e}") from e

    if raw is None or raw.empty:
        raise DataError(
            f"no Yahoo daily bars for {symbol} (ticker {ticker}). "
            "This coin may not list on Yahoo Finance."
        )

    # yfinance may return MultiIndex columns when ticker is a single string.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).strip().lower() for c in raw.columns]
    else:
        raw.columns = [str(c).strip().lower() for c in raw.columns]

    need = {"open", "high", "low", "close"}
    missing = need - set(raw.columns)
    if missing:
        raise DataError(f"Yahoo response for {ticker} missing columns: {sorted(missing)}")

    vol_col = "volume" if "volume" in raw.columns else None
    df = pd.DataFrame({
        "open": pd.to_numeric(raw["open"], errors="coerce"),
        "high": pd.to_numeric(raw["high"], errors="coerce"),
        "low": pd.to_numeric(raw["low"], errors="coerce"),
        "close": pd.to_numeric(raw["close"], errors="coerce"),
        "volume": (
            pd.to_numeric(raw[vol_col], errors="coerce")
            if vol_col else 0.0
        ),
    })
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    # Drop still-forming daily candle (same rule as ohlcv_to_frame).
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(days=1) <= now]
    df = df[(df.index >= fetch_start) & (df.index < fetch_end)]
    if df.empty:
        raise DataError(
            f"no Yahoo daily bars for {symbol} (ticker {ticker}) "
            f"in {fetch_start:%Y-%m-%d}→{fetch_end:%Y-%m-%d}"
        )
    return df.astype(float)


def probe_earliest_ohlcv(
    symbol: str,
    tf: str,
    *,
    probe_from: str = "2014-01-01",
    source: str = RANGE_SOURCE_MEXC,
) -> Optional[pd.Timestamp]:
    """Find the earliest completed candle the active source returns for ``symbol``/``tf``."""
    src = _normalize_range_source(source)
    tf = str(tf).strip().lower()

    if src == RANGE_SOURCE_YAHOO:
        if tf != "1d":
            return None
        try:
            df = fetch_yahoo_ohlcv_range(
                symbol,
                _as_utc_ts(probe_from),
                pd.Timestamp.now(tz="UTC"),
            )
        except Exception:
            return None
        if df is None or df.empty:
            return None
        return pd.Timestamp(df.index[0])

    step_ms = timeframe_ms(tf)
    lo = int(_as_utc_ts(probe_from).timestamp() * 1000)
    hi = int(time.time() * 1000)
    market = _ccxt_symbol_for_source(symbol, src)
    with _OHLCV_LOCK:
        ex = exchange_for_source(src)
        # Seed an upper bound: any recent page proves the market exists.
        seed = ex.fetch_ohlcv(market, timeframe=tf, limit=5)
        if not seed:
            return None
        hi = int(seed[0][0])

        # Binary search the earliest since that still returns a candle.
        best: Optional[int] = hi
        for _ in range(24):
            if hi - lo <= step_ms * 2:
                break
            mid = lo + (hi - lo) // 2
            batch = ex.fetch_ohlcv(market, timeframe=tf, since=mid, limit=5)
            time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
            if batch:
                best = int(batch[0][0])
                hi = mid
            else:
                lo = mid
        # One final pull from the refined lower bound.
        final = ex.fetch_ohlcv(market, timeframe=tf, since=max(0, (best or hi) - step_ms * 50), limit=50)
        time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
    if final:
        return pd.Timestamp(int(final[0][0]), unit="ms", tz="UTC")
    if best is not None:
        return pd.Timestamp(int(best), unit="ms", tz="UTC")
    return None


def _range_source_blurb(source: str) -> str:
    src = _normalize_range_source(source)
    if src == RANGE_SOURCE_BINANCE_SPOT:
        return "Binance spot"
    if src == RANGE_SOURCE_YAHOO:
        return "Yahoo Finance daily (spot USD)"
    return "MEXC USDT-M"


def _range_no_history_error(
    symbol: str,
    tf: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    *,
    got: Optional[pd.DataFrame] = None,
    source: str = RANGE_SOURCE_MEXC,
) -> DataError:
    """Clear error when the requested calendar window has no bars on ``source``."""
    src = _normalize_range_source(source)
    label = _range_source_blurb(src)
    earliest = None
    try:
        earliest = probe_earliest_ohlcv(symbol, tf, source=src)
    except Exception:
        earliest = None
    if earliest is None and got is not None and not got.empty:
        earliest = pd.Timestamp(got.index[0])
    if earliest is not None:
        extra = ""
        if src == RANGE_SOURCE_MEXC:
            extra = " (swap/futures, not spot). Try Binance spot or Yahoo for earlier years."
        elif src == RANGE_SOURCE_BINANCE_SPOT:
            extra = " (spot USDT). Try Yahoo daily for pre-listing years."
        if earliest >= t1:
            return DataError(
                f"no {tf} bars for {symbol} in {t0:%Y-%m-%d}→{t1:%Y-%m-%d}. "
                f"That window is entirely before {label} listing "
                f"(~{earliest:%Y-%m-%d}).{extra} "
                f"Set start to {earliest:%Y-%m-%d} or later."
            )
        return DataError(
            f"no {tf} bars for {symbol} in {t0:%Y-%m-%d}→{t1:%Y-%m-%d}. "
            f"{label} history for this market starts ~{earliest:%Y-%m-%d}"
            f"{extra} Set start to {earliest:%Y-%m-%d} or later."
        )
    hint = {
        RANGE_SOURCE_MEXC: (
            "MEXC returned no candles — USDT-M markets often list years after spot. "
            "Try Binance spot, Yahoo daily, or a more recent start."
        ),
        RANGE_SOURCE_BINANCE_SPOT: (
            "Binance spot returned no candles — pair may list later or lack this TF. "
            "Try Yahoo daily or a more recent start."
        ),
        RANGE_SOURCE_YAHOO: (
            "Yahoo returned no daily bars — ticker may be missing. "
            "Try another coin or a more recent start."
        ),
    }.get(src, "No candles in that window.")
    return DataError(
        f"no {tf} bars for {symbol} in {t0:%Y-%m-%d}→{t1:%Y-%m-%d}. {hint}"
    )


def fetch_historical_range(
    symbol: str,
    tf: str,
    start,
    end,
    *,
    warmup_bars: int = 0,
    cache_dir: str = ".moonshot_cache",
    refresh: bool = False,
    source: str = RANGE_SOURCE_MEXC,
) -> pd.DataFrame:
    """Fetch OHLCV covering ``[start - warmup, end)`` for calendar backtests.

    Warmup bars sit *before* ``start`` so indicators are valid when the
    BacktestEngine window opens; callers should still pass ``start``/``end``
    into the engine so only the chosen window is traded.

    ``source`` is Backtest-only: ``mexc`` (default), ``binance_spot``, or ``yahoo``.
    Live/paper paths never pass this and stay on MEXC.
    """
    src = _normalize_range_source(source)
    tf = str(tf).strip().lower()
    t0 = _as_utc_ts(start)
    t1 = _as_utc_ts(end)
    if not (t0 < t1):
        raise DataError(f"invalid range: start {t0} must be before end {t1}")

    if src == RANGE_SOURCE_YAHOO and tf != "1d":
        raise DataError(
            "Yahoo Finance Backtest source only supports 1d candles "
            f"(got {tf}). Switch candle size to 1d or pick MEXC / Binance spot."
        )

    warm = int(warmup_bars)
    if warm <= 0:
        warm = max(500, min(int(live_history_bars_for_tf(tf)), 3_000))

    step_ms = timeframe_ms(tf)
    fetch_start = t0 - pd.Timedelta(milliseconds=step_ms * warm)
    # Never request future candles past "now"
    now = pd.Timestamp.now(tz="UTC")
    fetch_end = min(t1, now)
    if fetch_start >= fetch_end:
        raise DataError(
            f"range is empty after clamping to now ({symbol} {tf}: "
            f"{fetch_start} → {fetch_end})"
        )

    est_bars = int((fetch_end - fetch_start).total_seconds() * 1000 // step_ms) + 2
    if est_bars > _MAX_RANGE_BARS:
        raise DataError(
            f"range too large for {symbol} {tf}: ~{est_bars:,} bars "
            f"(max {_MAX_RANGE_BARS:,}). Narrow the window or use a higher timeframe."
        )

    # Unsupported native TF → build from a finer base over the same window.
    base_tf = _resample_base_tf(tf, source=src)
    if base_tf is not None:
        if src == RANGE_SOURCE_YAHOO:
            raise DataError(
                f"Yahoo cannot build {tf} — use 1d or pick MEXC / Binance spot."
            )
        base_ms = timeframe_ms(base_tf)
        ratio = max(1, int(step_ms // base_ms))
        raw = fetch_historical_range(
            symbol, base_tf, t0, t1,
            warmup_bars=warm * ratio,
            cache_dir=cache_dir,
            refresh=refresh,
            source=src,
        )
        if raw is None or raw.empty:
            raise DataError(f"no {base_tf} history to build {tf} for {symbol}")
        built = _ohlcv_resample(raw, tf)
        if built.empty:
            raise DataError(f"resample to {tf} produced empty frame for {symbol}")
        out = built[(built.index >= fetch_start) & (built.index < fetch_end)]
        if out.empty:
            raise _range_no_history_error(symbol, tf, t0, t1, got=built, source=src)
        return out

    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    key = (
        f"{_cache_stem(symbol, tf, source=src)}range_"
        f"{int(fetch_start.timestamp())}_{int(fetch_end.timestamp())}_w{warm}.csv"
    )
    path = cache / key

    if path.exists() and not refresh:
        try:
            cached = _read_ohlcv_csv(path)
            if len(cached) > 0:
                # Re-apply window in case an older cache key was reused.
                hit = cached[(cached.index >= fetch_start) & (cached.index < fetch_end)]
                if not hit.empty:
                    return hit
        except Exception:
            pass

    if src == RANGE_SOURCE_YAHOO:
        try:
            df = fetch_yahoo_ohlcv_range(symbol, fetch_start, fetch_end)
        except DataError:
            raise
        except Exception as e:
            raise DataError(f"Yahoo fetch failed for {symbol}: {e}") from e
        if df.empty:
            raise _range_no_history_error(symbol, tf, t0, t1, source=src)
        try:
            df.to_csv(path)
        except Exception:
            pass
        log.info(
            "%s yahoo 1d range %s→%s (+%d warmup) → %d bars",
            symbol, t0.date(), t1.date(), warm, len(df),
        )
        return df

    since_ms = int(fetch_start.timestamp() * 1000)
    until_ms = int(fetch_end.timestamp() * 1000)
    market = _ccxt_symbol_for_source(symbol, src)
    with _OHLCV_LOCK:
        ex = exchange_for_source(src)
        rows = paged_ohlcv(ex, market, tf, since_ms, until_ms)
        if not rows:
            time.sleep(0.5)
            rows = paged_ohlcv(ex, market, tf, since_ms, until_ms)

    if not rows:
        raise _range_no_history_error(symbol, tf, t0, t1, source=src)

    # Drop foothold bars that landed entirely after the requested window
    # (paged_ohlcv jumps forward when since is pre-listing).
    rows = [r for r in rows if since_ms <= int(r[0]) < until_ms]
    if not rows:
        raise _range_no_history_error(symbol, tf, t0, t1, source=src)

    df = ohlcv_to_frame(rows, tf)
    if df.empty:
        raise DataError(f"OHLCV empty after candle filter ({symbol} {tf})")
    # Keep warmup + window; drop anything strictly at/after fetch_end
    df = df[(df.index >= fetch_start) & (df.index < fetch_end)]
    if df.empty:
        raise _range_no_history_error(symbol, tf, t0, t1, got=None, source=src)
    try:
        df.to_csv(path)
    except Exception:
        pass
    log.info(
        "%s %s %s range %s→%s (+%d warmup) → %d bars",
        symbol, src, tf, t0.date(), t1.date(), warm, len(df),
    )
    return df


def synthetic_ohlcv(symbol: str, n: int, tf: str = "4h") -> pd.DataFrame:
    """Deterministic (crc32, not hash()) regime-switching data. SMOKE TESTS ONLY."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    drift = np.zeros(n)
    i = 0
    while i < n:
        length = int(rng.integers(60, 400))
        drift[i:i + length] = rng.choice([0.0015, -0.0015, 0.0003, 0.0])
        i += length
    sigma = 0.008 * np.exp(0.25 * np.cumsum(rng.normal(0, 0.1, n)) * 0.1)
    rets = drift + rng.normal(0, 1, n) * sigma
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = 2000 * (1 + 40 * np.abs(rets)) * rng.lognormal(0, 0.3, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=_pandas_tf(tf), tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


# ============================================================
# 5. STRATEGY (regime filter + separate entry trigger)
# ============================================================


class RegimeBreakoutStrategy:
    """
    Legacy: regime + Donchian breakout. Fragile under soft kills — kept for A/B only.
    Evaluated at bar close; executed at the next open.
    """

    name = "regime_breakout_v7"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def on_bar(self, symbol: str, bar: Any) -> Optional[Signal]:
        regime = bar.regime
        if regime not in (RegimeState.TRENDING_UP, RegimeState.TRENDING_DOWN):
            return None
        vals = (bar.atr, bar.don_high, bar.don_low, bar.vol_ratio, bar.adx)
        if any(v != v for v in vals) or bar.atr <= 0:
            return None
        if self.cfg.require_adx_for_entry and bar.adx < self.cfg.adx_enter:
            return None
        if self.cfg.require_rising_adx:
            slope = getattr(bar, "adx_slope", float("nan"))
            if slope != slope or slope <= 0:
                return None
        if bar.vol_ratio < self.cfg.volume_multiplier:
            return None
        if self.cfg.require_squeeze and not bool(getattr(bar, "squeeze_ok", False)):
            return None

        want_long = regime == RegimeState.TRENDING_UP
        rsi_v = getattr(bar, "rsi", float("nan"))
        if self.cfg.require_rsi_veto and rsi_v == rsi_v:
            if want_long and rsi_v > self.cfg.rsi_long_max:
                return None
            if (not want_long) and rsi_v < self.cfg.rsi_short_min:
                return None

        if self.cfg.require_btc_regime and not _is_btc_symbol(symbol):
            btc_t = float(getattr(bar, "btc_trend", 0.0) or 0.0)
            if want_long and btc_t <= 0:
                return None
            if (not want_long) and btc_t >= 0:
                return None

        buf = self.cfg.breakout_buffer_atr * bar.atr
        tags = []
        if bool(getattr(bar, "squeeze_ok", False)):
            tags.append("sqz")
        if rsi_v == rsi_v:
            tags.append(f"RSI={rsi_v:.0f}")
        btc_t = float(getattr(bar, "btc_trend", 0.0) or 0.0)
        if btc_t:
            tags.append("BTCup" if btc_t > 0 else "BTCdn")
        tag = (" " + " ".join(tags)) if tags else ""

        if want_long and bar.close > bar.don_high + buf:
            return Signal(symbol, Direction.LONG, bar.close, bar.atr, RegimeState.TRENDING_UP,
                          f"breakout>{bar.don_high:.4g}+buf ADX={bar.adx:.1f}↑ vol x{bar.vol_ratio:.1f}{tag}")
        if (not want_long) and bar.close < bar.don_low - buf:
            return Signal(symbol, Direction.SHORT, bar.close, bar.atr, RegimeState.TRENDING_DOWN,
                          f"breakdown<{bar.don_low:.4g}-buf ADX={bar.adx:.1f}↑ vol x{bar.vol_ratio:.1f}{tag}")
        return None


class Momentum15Strategy:
    """
    Dense 15m momentum continuation: recent move in ATR units + volume spike,
    optional ADX floor. No Donchian/regime gate as the sole trigger.
    Evaluated at bar close; executed at the next open.
    """

    name = "momentum_15"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def on_bar(self, symbol: str, bar: Any) -> Optional[Signal]:
        mom = float(getattr(bar, "mom_atr", float("nan")))
        vals = (bar.atr, bar.vol_ratio, bar.adx, mom)
        if any(v != v for v in vals) or bar.atr <= 0:
            return None
        if self.cfg.require_adx_for_entry and bar.adx < self.cfg.adx_enter:
            return None
        if self.cfg.require_rising_adx:
            slope = getattr(bar, "adx_slope", float("nan"))
            if slope != slope or slope <= 0:
                return None
        if bar.vol_ratio < self.cfg.volume_multiplier:
            return None
        if self.cfg.require_squeeze and not bool(getattr(bar, "squeeze_ok", False)):
            return None

        thr = float(self.cfg.mom_min_atr)
        want_long = mom >= thr
        want_short = mom <= -thr
        if not want_long and not want_short:
            return None

        rsi_v = getattr(bar, "rsi", float("nan"))
        if self.cfg.require_rsi_veto and rsi_v == rsi_v:
            if want_long and rsi_v > self.cfg.rsi_long_max:
                return None
            if want_short and rsi_v < self.cfg.rsi_short_min:
                return None

        if self.cfg.require_btc_regime and not _is_btc_symbol(symbol):
            btc_t = float(getattr(bar, "btc_trend", 0.0) or 0.0)
            if want_long and btc_t <= 0:
                return None
            if want_short and btc_t >= 0:
                return None

        # Soft HTF bias: prefer trades aligned with EMA200 when available
        ema_htf = getattr(bar, "ema_htf", float("nan"))
        if ema_htf == ema_htf:
            if want_long and bar.close < ema_htf:
                return None
            if want_short and bar.close > ema_htf:
                return None

        tag = f"mom={mom:+.2f}ATR ADX={bar.adx:.1f} vol x{bar.vol_ratio:.1f}"
        if want_long:
            return Signal(symbol, Direction.LONG, bar.close, bar.atr,
                          RegimeState.TRENDING_UP, f"mom_up {tag}")
        return Signal(symbol, Direction.SHORT, bar.close, bar.atr,
                      RegimeState.TRENDING_DOWN, f"mom_dn {tag}")


class TrendMTFStrategy:
    """
    4H decides direction; 15m times the entry.
    Long only with completed 4H up-bias; short only with 4H down-bias + stronger ADX.
    15m trigger: momentum continuation or pullback-to-EMA20 resume.
    """

    name = "trend_mtf"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def on_bar(self, symbol: str, bar: Any) -> Optional[Signal]:
        mom = float(getattr(bar, "mom_atr", float("nan")))
        h4_bias = float(getattr(bar, "h4_bias", 0.0) or 0.0)
        h4_adx = float(getattr(bar, "h4_adx", float("nan")))
        ema_e = float(getattr(bar, "ema_entry", float("nan")))
        vals = (bar.atr, bar.vol_ratio, bar.adx, mom, ema_e)
        if any(v != v for v in vals) or bar.atr <= 0:
            return None
        if h4_bias == 0.0:
            return None
        if self.cfg.require_adx_for_entry and bar.adx < self.cfg.adx_enter:
            return None
        if self.cfg.require_rising_adx:
            slope = getattr(bar, "adx_slope", float("nan"))
            if slope != slope or slope <= 0:
                return None
        if self.cfg.require_squeeze and not bool(getattr(bar, "squeeze_ok", False)):
            return None

        thr = float(self.cfg.mom_min_atr)
        vol_ok = bar.vol_ratio >= self.cfg.volume_multiplier
        pb_long = bool(getattr(bar, "pullback_long", False))
        pb_short = bool(getattr(bar, "pullback_short", False))

        # Continuation OR pullback-resume in bias direction
        long_trigger = (mom >= thr and vol_ok) or (
            pb_long and bar.close > ema_e and mom > 0 and vol_ok
        )
        short_trigger = (mom <= -thr and vol_ok) or (
            pb_short and bar.close < ema_e and mom < 0 and vol_ok
        )

        # Market-wide chop filters (hunt-searchable; 0 = off)
        if self.cfg.btc_adx_min > 0 and float(getattr(bar, "btc_adx", 0.0)) < self.cfg.btc_adx_min:
            return None
        brd = float(getattr(bar, "breadth", 0.0))
        if self.cfg.breadth_min > 0:
            long_trigger = long_trigger and brd >= self.cfg.breadth_min
            short_trigger = short_trigger and brd <= -self.cfg.breadth_min

        want_long = h4_bias > 0 and long_trigger
        # Shorts: stricter — need 4H down stack AND 4H ADX strength
        short_adx_ok = h4_adx == h4_adx and h4_adx >= float(self.cfg.h4_adx_min_short)
        want_short = h4_bias < 0 and short_adx_ok and short_trigger
        if not want_long and not want_short:
            return None

        rsi_v = getattr(bar, "rsi", float("nan"))
        if self.cfg.require_rsi_veto and rsi_v == rsi_v:
            if want_long and rsi_v > self.cfg.rsi_long_max:
                return None
            if want_short and rsi_v < self.cfg.rsi_short_min:
                return None

        if self.cfg.require_btc_regime and not _is_btc_symbol(symbol):
            btc_t = float(getattr(bar, "btc_trend", 0.0) or 0.0)
            if want_long and btc_t <= 0:
                return None
            if want_short and btc_t >= 0:
                return None

        how = "mom" if abs(mom) >= thr else "pullback"
        tag = (
            f"{how} mom={mom:+.2f}ATR h4={h4_bias:+.0f} "
            f"h4ADX={h4_adx if h4_adx == h4_adx else 0:.0f} vol x{bar.vol_ratio:.1f}"
        )
        if want_long:
            return Signal(symbol, Direction.LONG, bar.close, bar.atr,
                          RegimeState.TRENDING_UP, f"mtf_long {tag}")
        return Signal(symbol, Direction.SHORT, bar.close, bar.atr,
                      RegimeState.TRENDING_DOWN, f"mtf_short {tag}")


class MomentumRuleStrategy:
    """
    Rule found by discover.py (automatic search, validated on unseen data):
    LONG when RSI is hot (> rsi_rule_min), EMA50 > EMA200 (long-term uptrend) and ADX shows a trend.
    Long-only: shorts were ~0 on unseen data. Needs forward paper proof before any trust.
    """

    name = "momentum_rule"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def on_bar(self, symbol: str, bar: Any) -> Optional[Signal]:
        rsi_v = float(getattr(bar, "rsi", float("nan")))
        e50 = float(getattr(bar, "ema_50", float("nan")))
        e200 = float(getattr(bar, "ema_200", float("nan")))
        if any(v != v for v in (rsi_v, e50, e200, bar.atr, bar.adx)) or bar.atr <= 0:
            return None
        if not (rsi_v > self.cfg.rsi_rule_min and e50 > e200 and bar.adx >= self.cfg.adx_enter):
            return None
        if self.cfg.require_btc_regime and not _is_btc_symbol(symbol):
            if float(getattr(bar, "btc_trend", 0.0) or 0.0) <= 0:
                return None
        return Signal(symbol, Direction.LONG, bar.close, bar.atr, RegimeState.TRENDING_UP,
                      f"rule_long rsi={rsi_v:.0f} adx={bar.adx:.0f} ema50>200")


class ComboStrategy:
    """trend_mtf + momentum_rule. mode 'and' = both must agree (confluence); 'or' = either."""

    def __init__(self, cfg: Config, mode: str):
        self.cfg, self.mode = cfg, mode
        self.name = f"combo_{mode}"
        self.a = TrendMTFStrategy(cfg)
        self.b = MomentumRuleStrategy(cfg)

    def on_bar(self, symbol: str, bar: Any) -> Optional[Signal]:
        sa = self.a.on_bar(symbol, bar)
        sb = self.b.on_bar(symbol, bar)
        if self.mode == "and":
            if sa is not None and sb is not None and sa.direction == sb.direction:
                return sa
            return None
        return sa or sb


def make_strategy(cfg: Config):
    """Select entry strategy from cfg.strategy."""
    name = (cfg.strategy or "trend_mtf").strip().lower()
    if name in ("combo_and", "combo_or"):
        return ComboStrategy(cfg, name.split("_")[1])
    if name in ("momentum_rule", "rule"):
        return MomentumRuleStrategy(cfg)
    if name in ("regime_breakout", "regime_breakout_v7", "breakout"):
        return RegimeBreakoutStrategy(cfg)
    if name in ("momentum_15", "momentum", "mom"):
        return Momentum15Strategy(cfg)
    return TrendMTFStrategy(cfg)


def _regime_value(regime: Any) -> str:
    if isinstance(regime, RegimeState):
        return regime.value
    return str(regime)


def regime_aligned(direction: Direction, regime: Any) -> bool:
    """True when regime actively agrees with the position (not unknown/ranging)."""
    want = RegimeState.TRENDING_UP if direction is Direction.LONG else RegimeState.TRENDING_DOWN
    return _regime_value(regime) == want.value


def regime_hostile(direction: Direction, regime: Any) -> bool:
    """True only when regime is the *opposite* trend.

    UNKNOWN (warmup / missing daily EMA) and RANGING must not force regime_invalid
    exits — that was killing every paper hold after one bar.
    """
    r = _regime_value(regime)
    if r in (RegimeState.UNKNOWN.value, RegimeState.RANGING.value, ""):
        return False
    if direction is Direction.LONG:
        return r == RegimeState.TRENDING_DOWN.value
    return r == RegimeState.TRENDING_UP.value


# ============================================================
# 6. POSITION (per-lot accounting)
# ============================================================


@dataclass
class Position:
    symbol: str
    direction: Direction
    leverage: int
    entry_time: datetime
    initial_entry: float
    initial_qty: float
    initial_stop: float
    avg_entry: float
    current_qty: float
    current_stop: float
    lots: list = field(default_factory=list)
    adds: list = field(default_factory=list)
    realized_gross: float = 0.0  # price P&L of already-closed partial quantity
    fees: float = 0.0
    funding: float = 0.0
    partial_done: bool = False

    # ---- construction ----
    @classmethod
    def open(cls, symbol: str, direction: Direction, leverage: int, price: float,
             qty: float, stop: float, ts: datetime, fee: float) -> "Position":
        return cls(symbol, direction, leverage, ts, price, qty, stop, price, qty, stop,
                   lots=[Lot(price, qty, ts)], fees=fee)

    # ---- derived ----
    @property
    def sign(self) -> int:
        return self.direction.sign

    @property
    def initial_risk_per_unit(self) -> float:
        return abs(self.initial_entry - self.initial_stop)

    @property
    def initial_risk_dollars(self) -> float:
        return self.initial_qty * self.initial_risk_per_unit

    @property
    def adds_count(self) -> int:
        return len(self.adds)

    def margin(self) -> float:
        return self.avg_entry * self.current_qty / self.leverage

    def unrealized(self, price: float) -> float:
        return self.sign * (price - self.avg_entry) * self.current_qty

    def net_pnl_at(self, price: float) -> float:
        """Everything booked or floating if the remainder closed at `price` (before exit fee)."""
        return self.realized_gross + self.unrealized(price) - self.fees - self.funding

    def r_multiple(self, price: float) -> float:
        if self.initial_risk_per_unit <= 0:
            return 0.0
        return self.sign * (price - self.initial_entry) / self.initial_risk_per_unit

    def bars_held(self, bar_ts, bar_td: pd.Timedelta) -> int:
        """Whole bars since entry fill (0 on the entry bar)."""
        try:
            delta = pd.Timestamp(bar_ts) - pd.Timestamp(self.entry_time)
            if bar_td.total_seconds() <= 0:
                return 0
            return max(0, int(delta / bar_td))
        except Exception:
            return 0

    def liq_price(self, mmr: float) -> float:
        return self.avg_entry * (1 - self.sign * (1.0 / self.leverage - mmr))

    # ---- mutation ----
    def is_tighter(self, stop: float) -> bool:
        return stop >= self.current_stop if self.sign > 0 else stop <= self.current_stop

    def tighten_stop(self, stop: float) -> bool:
        if self.is_tighter(stop):
            self.current_stop = stop
            return True
        return False

    def add_lot(self, price: float, qty: float, ts: datetime, fee: float,
                new_stop: float, floating_used: float) -> None:
        if not self.is_tighter(new_stop):
            raise RiskError(f"refusing to widen stop {self.current_stop} -> {new_stop}")
        r_at_add = self.r_multiple(price)
        cost = self.avg_entry * self.current_qty + price * qty
        self.current_qty += qty
        self.avg_entry = cost / self.current_qty  # average entry, not original entry
        self.fees += fee
        self.lots.append(Lot(price, qty, ts))
        self.adds.append(AddRecord(price, qty, new_stop, r_at_add, floating_used, ts))
        self.current_stop = new_stop

    def reduce(self, price: float, qty: float, fee: float) -> float:
        qty = min(qty, self.current_qty)
        gross = self.sign * (price - self.avg_entry) * qty
        self.realized_gross += gross
        self.fees += fee
        self.current_qty -= qty
        return gross

    def close_out(self, price: float, fee: float, ts: datetime, reason: ExitReason,
                  liquidated: bool = False) -> tuple:
        """Close the remainder. Returns (TradeResult, cash_delta)."""
        gross = self.sign * (price - self.avg_entry) * self.current_qty
        net = self.realized_gross + gross - self.fees - fee - self.funding
        risk = self.initial_risk_dollars
        res = TradeResult(
            symbol=self.symbol, direction=self.direction, entry_time=self.entry_time,
            exit_time=ts, entry_price=self.initial_entry, exit_price=price,
            initial_qty=self.initial_qty, pnl=net, r_multiple=net / risk if risk > 0 else 0.0,
            liquidated=liquidated, exit_reason=reason, max_adds=self.adds_count,
            partial_taken=self.partial_done, fees=self.fees + fee, funding=self.funding,
        )
        self.current_qty = 0.0
        return res, gross - fee

    # ---- persistence ----
    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["entry_time"] = pd.Timestamp(self.entry_time).isoformat()
        for item in d["lots"] + d["adds"]:
            item["time"] = pd.Timestamp(item["time"]).isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["direction"] = Direction(d["direction"])
        d["entry_time"] = pd.Timestamp(d["entry_time"])
        d["lots"] = [Lot(l["price"], l["qty"], pd.Timestamp(l["time"])) for l in d["lots"]]
        d["adds"] = [AddRecord(a["price"], a["qty"], a["stop_at_add"], a["r_at_add"],
                               a["floating_profit_used"], pd.Timestamp(a["time"]))
                     for a in d["adds"]]
        return cls(**d)


# ============================================================
# 7. RISK ENGINE
# ============================================================


class RiskEngine:
    """Single gatekeeper for new risk. Time always comes from the caller (backtest-safe)."""

    def __init__(self, cfg: Config, silent: bool = False):
        self.cfg = cfg
        self.silent = silent  # True in backtests — avoid flooding logs with historical halts
        self.reset()

    def reset(self, equity: Optional[float] = None) -> None:
        self.equity = equity if equity is not None else self.cfg.starting_equity
        self.peak_equity = self.equity
        self.daily_start_equity = self.equity
        self.current_day = None
        self.daily_halt = False
        self.pause_until: Optional[pd.Timestamp] = None
        self.consecutive_losses = 0
        self.rolling_r: list = []
        self.kill_switch_active = False
        self.halt_reason = ""

    def update_equity(self, equity: float, ts: pd.Timestamp) -> None:
        equity = max(equity, 0.0)
        day = ts.date()
        if self.current_day is None:
            self.daily_start_equity = equity
        elif day != self.current_day:
            self.daily_start_equity = self.equity  # last mark of the previous day
            self.daily_halt = False  # a daily halt only lasts until the next UTC day
        self.current_day = day
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd >= self.cfg.max_peak_drawdown_pct:
            self._kill(f"peak drawdown {dd:.1%} >= {self.cfg.max_peak_drawdown_pct:.0%}")
        if self.daily_start_equity > 0:
            loss = (self.daily_start_equity - equity) / self.daily_start_equity
            if loss >= self.cfg.max_daily_loss_pct and not self.daily_halt:
                self.daily_halt = True
                if not self.silent:
                    log.warning("DAILY HALT: loss %.1f%% (resumes next UTC day)", loss * 100)

    def register_trade_result(self, r: float, ts: pd.Timestamp) -> None:
        self.rolling_r.append(r)
        self.consecutive_losses = self.consecutive_losses + 1 if r < 0 else 0
        if self.consecutive_losses >= self.cfg.streak_losses:
            self.pause_until = ts + pd.Timedelta(hours=self.cfg.streak_pause_hours)
            self.consecutive_losses = 0
            if not self.silent:
                log.warning("LOSS-STREAK PAUSE until %s", self.pause_until)
        w = self.cfg.expectancy_window
        if len(self.rolling_r) >= w:
            exp = float(np.mean(self.rolling_r[-w:]))
            if exp < self.cfg.expectancy_floor:
                self._kill(f"rolling {w}-trade expectancy {exp:.2f}R < {self.cfg.expectancy_floor}")

    def trading_allowed(self, ts: pd.Timestamp) -> bool:
        if self.kill_switch_active or self.daily_halt:
            return False
        return self.pause_until is None or ts >= self.pause_until

    def evaluate(self, direction: Direction, entry: float, stop: float, *, equity: float,
                 open_notional: float, used_margin: float, open_positions: int,
                 spec: InstrumentSpec, ts: pd.Timestamp,
                 net_notional: float = 0.0) -> RiskDecision:
        cfg = self.cfg
        no = lambda why: RiskDecision(False, why)  # noqa: E731
        if not self.trading_allowed(ts):
            return no(f"trading halted ({self.halt_reason or 'daily/pause'})")
        if open_positions >= cfg.max_open_positions:
            return no("max open positions")
        if equity <= 0:
            return no("zero equity")
        stop_dist = (entry - stop) * direction.sign
        if stop_dist <= 0:
            return no("invalid stop")

        lev = int(cfg.leverage)
        margin_pct = cfg.aggressive_margin_pct if cfg.aggressive_mode else cfg.initial_margin_pct
        q_risk = equity * cfg.risk_pct / stop_dist
        q_margin = equity * margin_pct * lev / entry
        q_free = max(0.0, equity - used_margin) * lev / entry
        q_lev = max(0.0, equity * cfg.max_account_leverage - open_notional) / entry
        # Directional book brake: room left before |net signed notional| hits max_net_leverage
        net_cap = max(0.0, equity * float(cfg.max_net_leverage))
        room = max(0.0, net_cap - direction.sign * net_notional)
        q_net = room / entry if entry > 0 else 0.0
        qty = spec.round_qty(min(q_risk, q_margin, q_free, q_lev, q_net))
        if qty <= 0:
            return no("size below exchange minimum (or net-directional cap)")
        if qty * entry < cfg.min_notional:
            return no(f"notional {qty * entry:.2f} < min {cfg.min_notional}")
        proposed_net = abs(net_notional + direction.sign * qty * entry)
        if equity > 0 and proposed_net / equity > float(cfg.max_net_leverage) + 1e-9:
            return no(
                f"net directional {proposed_net / equity:.2f}x > max_net_leverage "
                f"{cfg.max_net_leverage:.2f}x"
            )

        liq_dist_pct = 1.0 / lev - cfg.maintenance_margin_rate
        stop_pct = stop_dist / entry
        thr = 0.65 if cfg.aggressive_mode else 0.5
        if stop_pct > liq_dist_pct * thr:
            return no(f"stop {stop_pct:.2%} > {thr:.0%} of liq distance {liq_dist_pct:.2%}")
        return RiskDecision(True, "ok", qty, qty * entry / lev, lev, qty * stop_dist)

    def _kill(self, reason: str) -> None:
        if not self.kill_switch_active:
            self.kill_switch_active = True
            self.halt_reason = reason
            if not self.silent:
                log.warning("KILL SWITCH: %s", reason)


# ============================================================
# 8. ROLL MANAGER + POSITION MANAGER (shared decision logic)
# ============================================================


class RollManager:
    """Conditional pyramiding. Sizing is 'free-roll': an add may never turn a locked-in
    winner into a net loser if the tightened stop is hit."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def size_add(self, pos: Position, price: float, new_stop: float, equity: float,
                 used_margin: float, spec: InstrumentSpec) -> tuple:
        cfg = self.cfg
        floating = pos.unrealized(price)
        if floating <= 0:
            return None, AddDecision.BLOCK_NO_FLOATING_PROFIT
        loss_per_unit = pos.sign * (price - new_stop)
        if loss_per_unit <= 0:
            return None, AddDecision.BLOCK_STOP_WIDEN

        q_house = floating * pos.leverage / price  # add notional <= floating profit x leverage
        locked = pos.net_pnl_at(new_stop)  # net P&L if existing qty stops out at new_stop
        allowed_loss = locked + cfg.addon_giveback_r * pos.initial_risk_dollars
        if allowed_loss <= 0:
            return None, AddDecision.BLOCK_NO_FLOATING_PROFIT
        q_free_roll = allowed_loss / loss_per_unit
        q_lev = max(0.0, equity * cfg.max_account_leverage - pos.current_qty * price) / price
        q_margin = max(0.0, equity - used_margin) * pos.leverage / price

        min_qty = pos.initial_qty * cfg.min_add_qty_pct
        if min(q_lev, q_margin) < min_qty:
            return None, AddDecision.BLOCK_LEVERAGE_CAP
        qty = spec.round_qty(min(q_house, q_free_roll, q_lev, q_margin))
        if qty < min_qty or qty <= 0:
            return None, AddDecision.BLOCK_SIZE_TOO_SMALL
        return qty, AddDecision.ALLOW

    def evaluate_add(self, pos: Position, price: float, stop_candidate: float, regime: Any,
                     allowed: bool, equity: float, used_margin: float,
                     spec: InstrumentSpec) -> tuple:
        cfg = self.cfg
        if not allowed:
            return AddDecision.BLOCK_TRADING_HALTED, None
        if not regime_aligned(pos.direction, regime):
            return AddDecision.BLOCK_REGIME_INVALID, None
        if pos.adds_count >= cfg.max_adds_per_trade:
            return AddDecision.BLOCK_MAX_ADDS, None
        if pos.r_multiple(price) < cfg.add_trigger_r[pos.adds_count]:
            return AddDecision.BLOCK_R_THRESHOLD, None
        if not pos.is_tighter(stop_candidate):
            return AddDecision.BLOCK_STOP_WIDEN, None
        qty, decision = self.size_add(pos, price, stop_candidate, equity, used_margin, spec)
        return decision, qty


class PositionManager:
    """Pure decision logic evaluated at each bar CLOSE. Returns actions to execute."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.roll = RollManager(cfg)

    def decide(self, pos: Position, bar: Any, *, allowed: bool, kill: bool, equity: float,
               used_margin: float, spec: InstrumentSpec) -> list:
        cfg = self.cfg
        if kill:
            return [Action(ActionKind.EXIT, reason=ExitReason.KILL_SWITCH)]
        regime = bar.regime
        # Grace ≥1 closed bar. Exit only on *hostile* regime (opposite trend) —
        # never on unknown/ranging (warmup + MTF/mom entries ignore bar.regime).
        held = pos.bars_held(getattr(bar, "Index", None), cfg.bar_td)
        if (
            cfg.exit_on_regime_invalid
            and held >= 1
            and regime_hostile(pos.direction, regime)
        ):
            return [Action(ActionKind.EXIT, reason=ExitReason.REGIME_INVALID)]

        close = bar.close
        r_now = pos.r_multiple(close)

        # Time stop: if the trade never reaches +min_r within N bars, dump it
        if cfg.enable_time_stop and cfg.time_stop_bars > 0:
            if held >= cfg.time_stop_bars and r_now < cfg.time_stop_min_r:
                return [Action(ActionKind.EXIT, reason=ExitReason.TIME_STOP)]

        stop = pos.current_stop
        if bar.atr == bar.atr and bar.atr > 0:
            trail_mult = cfg.trail_atr_multiple
            if cfg.use_asymmetric_trail and r_now < 0:
                trail_mult = min(trail_mult, cfg.trail_atr_loser)
            trail = close - pos.sign * trail_mult * bar.atr
            stop = max(stop, trail) if pos.sign > 0 else min(stop, trail)

        acts: list = []
        if cfg.enable_partial and not pos.partial_done and r_now >= cfg.partial_at_r:
            q = spec.round_qty(pos.current_qty * cfg.partial_pct)
            if 0 < q < pos.current_qty:
                acts.append(Action(ActionKind.PARTIAL, qty=q))
                # Lock runner at breakeven after banking the partial
                stop = max(stop, pos.avg_entry) if pos.sign > 0 else min(stop, pos.avg_entry)

        if cfg.enable_pyramiding:
            decision, _ = self.roll.evaluate_add(pos, close, stop, regime, allowed, equity,
                                                 used_margin, spec)
            if decision is AddDecision.ALLOW:
                acts.append(Action(ActionKind.ADD, price=stop))

        if stop != pos.current_stop:
            acts.append(Action(ActionKind.MOVE_STOP, price=stop))
        return acts


# ============================================================
# 9. BROKERS
# ============================================================


class BaseBroker:
    """Quantities are BASE units; brokers convert to exchange contracts internally."""

    def set_leverage(self, symbol: str, leverage: int) -> None: ...
    def market_order(self, symbol: str, side: str, qty: float, ref_price: float,
                     reduce_only: bool = False) -> Fill: raise NotImplementedError
    def set_stop(self, symbol: str, direction: Direction, qty: float, stop: float) -> None:
        raise NotImplementedError
    def cancel_stop(self, symbol: str) -> None: raise NotImplementedError
    def get_position(self, symbol: str) -> Optional[BrokerPosition]: raise NotImplementedError
    def get_equity(self) -> float: raise NotImplementedError
    def on_price(self, symbol: str, price: float) -> None: ...
    def pop_stop_fill(self, symbol: str) -> Optional[Fill]: return None


class PaperBroker(BaseBroker):
    """In-memory broker: fees, slippage, reduce-only, exchange-side stops triggered by on_price()."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cash = cfg.starting_equity
        self.signed: dict = {}  # symbol -> signed qty
        self.avg: dict = {}
        self.stops: dict = {}  # symbol -> stop price
        self.last: dict = {}
        self._stop_fills: dict = {}
        self._n = 0

    def _fill(self, symbol: str, side: str, qty: float, price: float) -> Fill:
        self._n += 1
        buy = side == "buy"
        px = price * (1 + self.cfg.slip if buy else 1 - self.cfg.slip)
        fee = px * qty * self.cfg.fee_taker
        delta = qty if buy else -qty
        s0 = self.signed.get(symbol, 0.0)
        if s0 != 0 and (s0 > 0) != buy:  # reducing -> realize P&L
            closing = min(abs(s0), qty)
            self.cash += (px - self.avg[symbol]) * closing * (1 if s0 > 0 else -1)
        elif s0 == 0 or (s0 > 0) == buy:  # opening / adding -> new average
            tot = abs(s0) + qty
            self.avg[symbol] = (self.avg.get(symbol, 0.0) * abs(s0) + px * qty) / tot
        s1 = s0 + delta
        self.signed[symbol] = 0.0 if abs(s1) < 1e-12 else s1
        if self.signed[symbol] == 0:
            self.avg.pop(symbol, None)
            self.stops.pop(symbol, None)
        self.cash -= fee
        return Fill(f"paper-{self._n}", symbol, side, qty, px, fee, pd.Timestamp.now(tz="UTC"))

    def market_order(self, symbol, side, qty, ref_price, reduce_only=False) -> Fill:
        s0 = self.signed.get(symbol, 0.0)
        if reduce_only:
            if s0 == 0 or (s0 > 0) == (side == "buy"):
                raise ExecutionError("reduce-only order would not reduce")
            qty = min(qty, abs(s0))
        self.last[symbol] = ref_price
        return self._fill(symbol, side, qty, ref_price)

    def set_stop(self, symbol, direction, qty, stop) -> None:
        self.stops[symbol] = stop

    def cancel_stop(self, symbol) -> None:
        self.stops.pop(symbol, None)

    def get_position(self, symbol) -> Optional[BrokerPosition]:
        s = self.signed.get(symbol, 0.0)
        if s == 0:
            return None
        return BrokerPosition(Direction.LONG if s > 0 else Direction.SHORT, abs(s), self.avg[symbol])

    def get_equity(self) -> float:
        eq = self.cash
        for sym, s in self.signed.items():
            if s != 0 and sym in self.avg:
                eq += (self.last.get(sym, self.avg[sym]) - self.avg[sym]) * s
        return eq

    def on_price(self, symbol, price) -> None:
        self.last[symbol] = price
        s, stop = self.signed.get(symbol, 0.0), self.stops.get(symbol)
        if not s or stop is None:
            return
        if s > 0 and price <= stop:
            self._stop_fills[symbol] = self._fill(symbol, "sell", abs(s), min(stop, price))
        elif s < 0 and price >= stop:
            self._stop_fills[symbol] = self._fill(symbol, "buy", abs(s), max(stop, price))

    def pop_stop_fill(self, symbol) -> Optional[Fill]:
        return self._stop_fills.pop(symbol, None)

    def rehydrate(self, positions: dict, closed_pnl: float) -> None:
        """
        Restore broker state after a process restart.
        Without this, LiveRunner.load_positions() leaves PaperBroker empty, and the
        next _reconcile treats every restored position as 'exchange stop fired' and
        fabricates a -1R close. Cash = starting + closed trade PnL − open fees/funding.
        """
        self.cash = float(self.cfg.starting_equity) + float(closed_pnl)
        self.signed.clear()
        self.avg.clear()
        self.stops.clear()
        self._stop_fills.clear()
        for sym, pos in positions.items():
            self.cash -= float(pos.fees) + float(pos.funding)
            self.signed[sym] = pos.sign * float(pos.current_qty)
            self.avg[sym] = float(pos.avg_entry)
            self.stops[sym] = float(pos.current_stop)
            self.last[sym] = float(pos.avg_entry)


class MexcBroker(BaseBroker):
    """
    Live MEXC USDT-perp adapter via CCXT.  *** UNVERIFIED AGAINST THE LIVE API ***
    Before trusting it: run with the minimum size, and check each order type, the
    leverage/openType params, trigger-order cancellation and balance field against
    the current ccxt `mexc` docs. Points to verify are marked VERIFY.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise ConfigError("ccxt is required for MexcBroker") from e
        key, secret = os.getenv("MEXC_API_KEY"), os.getenv("MEXC_API_SECRET")
        if not key or not secret:
            raise ConfigError("MEXC_API_KEY / MEXC_API_SECRET not set")
        self.ex = ccxt.mexc({"apiKey": key, "secret": secret, "enableRateLimit": True,
                             "options": {"defaultType": "swap"}})
        self.ex.load_markets()
        self._lev: dict = {}
        self._stop_id: dict = {}

    def _contracts(self, symbol: str, qty: float) -> float:
        m = self.ex.market(symbol)
        size = float(m.get("contractSize") or 1.0)
        return float(self.ex.amount_to_precision(symbol, qty / size))

    def set_leverage(self, symbol, leverage) -> None:
        self._lev[symbol] = leverage
        try:
            self.ex.set_leverage(leverage, symbol, params={"openType": 1})  # VERIFY isolated
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"set_leverage failed: {e}") from e

    def market_order(self, symbol, side, qty, ref_price, reduce_only=False) -> Fill:
        amount = self._contracts(symbol, qty)
        if amount <= 0:
            raise ExecutionError("order size rounds to zero contracts")
        params: dict = {"reduceOnly": reduce_only}
        if not reduce_only:  # VERIFY: MEXC swap requires leverage + openType on opening orders
            params.update({"leverage": self._lev.get(symbol, self.cfg.leverage), "openType": 1})
        try:
            raw = self.ex.create_order(symbol, "market", side, amount, None, params)
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"market order failed: {e}") from e
        size = float(self.ex.market(symbol).get("contractSize") or 1.0)
        px = float(raw.get("average") or raw.get("price") or ref_price)
        fee = float((raw.get("fee") or {}).get("cost") or px * qty * self.cfg.fee_taker)
        return Fill(str(raw["id"]), symbol, side, float(raw.get("filled") or amount) * size,
                    px, fee, pd.Timestamp.now(tz="UTC"))

    def set_stop(self, symbol, direction, qty, stop) -> None:
        self.cancel_stop(symbol)
        side = "sell" if direction is Direction.LONG else "buy"
        amount = self._contracts(symbol, qty)
        try:  # VERIFY: stopLossPrice trigger-market params on MEXC swap
            raw = self.ex.create_order(symbol, "market", side, amount, None,
                                       {"stopLossPrice": float(self.ex.price_to_precision(symbol, stop)),
                                        "reduceOnly": True})
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"stop placement failed: {e}") from e
        self._stop_id[symbol] = raw.get("id")

    def cancel_stop(self, symbol) -> None:
        oid = self._stop_id.pop(symbol, None)
        if not oid:
            return
        try:  # VERIFY: MEXC can return success with no body; ccxt may raise anyway
            self.ex.cancel_order(oid, symbol, params={"trigger": True})
        except Exception as e:  # noqa: BLE001
            log.warning("cancel_stop(%s) raised %s - check for orphaned stop orders!", symbol, e)

    def get_position(self, symbol) -> Optional[BrokerPosition]:
        try:
            raws = self.ex.fetch_positions([symbol])
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"fetch_positions failed: {e}") from e
        size = float(self.ex.market(symbol).get("contractSize") or 1.0)
        for r in raws:
            c = float(r.get("contracts") or 0)
            if c > 0:
                d = Direction.LONG if r["side"] == "long" else Direction.SHORT
                return BrokerPosition(d, c * size, float(r.get("entryPrice") or 0))
        return None

    def get_equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            return float(bal["total"]["USDT"])  # VERIFY: should include unrealized PnL
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"fetch_balance failed: {e}") from e


class MarketData:
    """Public market data for paper/live loops."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ex = public_exchange()
        self.ex.load_markets()

    def instrument(self, symbol: str) -> InstrumentSpec:
        m = self.ex.market(symbol)
        size = float(m.get("contractSize") or 1.0)
        min_c = float(((m.get("limits") or {}).get("amount") or {}).get("min") or 1.0)
        # ASSUMPTION: 1 contract is the lot step. VERIFY per symbol.
        return InstrumentSpec(symbol, qty_step=size, min_qty=size * min_c, contract_size=size)

    def last_price(self, symbol: str) -> float:
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                return float(self.ex.fetch_ticker(symbol)["last"])
            except Exception as e:  # noqa: BLE001 — network blips must not kill paper
                last_err = e
                time.sleep(0.25 * (attempt + 1))
        raise DataError(f"ticker failed after retries ({symbol}): {last_err}") from last_err

    def recent(self, symbol: str) -> pd.DataFrame:
        tf, n = self.cfg.timeframe, self.cfg.live_history_bars
        # Shared cache with automatic tail refresh when stale (Layer 0).
        try:
            return fetch_historical(symbol, tf, int(n), refresh=False)
        except DataError:
            now_ms = int(time.time() * 1000)
            with _OHLCV_LOCK:
                rows = paged_ohlcv(self.ex, symbol, tf, now_ms - int(n) * timeframe_ms(tf), now_ms)
            if not rows:
                raise DataError(
                    f"no OHLCV rows returned ({symbol} {tf}×{n}). "
                    "Exchange empty — usually rate-limit while hunt is fetching."
                )
            return ohlcv_to_frame(rows, tf)


# ============================================================
# 10. PERSISTENCE + NOTIFIER
# ============================================================


class Persistence:
    """SQLite log for paper/live. WAL + named columns so the dashboard can read safely."""

    def __init__(self, path: str, mode: str = "paper"):
        self.path = path
        self.mode = mode if mode in ("paper", "live") else "paper"
        self.conn = sqlite3.connect(path, timeout=30.0)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS trades (
                ts TEXT, symbol TEXT, direction TEXT, entry REAL, exit REAL, qty REAL,
                pnl REAL, r_multiple REAL, liquidated INTEGER, exit_reason TEXT, adds INTEGER,
                fees REAL, funding REAL, entry_time TEXT, partial_taken INTEGER, mode TEXT);
            CREATE TABLE IF NOT EXISTS equity (
                ts TEXT, equity REAL, peak REAL, drawdown REAL, mode TEXT);
            CREATE TABLE IF NOT EXISTS positions (symbol TEXT PRIMARY KEY, state TEXT);
            CREATE TABLE IF NOT EXISTS risk_snapshot (
                ts TEXT, mode TEXT, equity REAL, peak REAL, drawdown REAL,
                halt_reason TEXT, kill INTEGER, streak INTEGER, daily_halt INTEGER,
                pause_until TEXT);
            CREATE TABLE IF NOT EXISTS events (
                ts TEXT, mode TEXT, symbol TEXT, kind TEXT, payload TEXT);
            """
        )
        self._migrate()
        self.conn.commit()

    def _cols(self, table: str) -> set:
        return {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})").fetchall()}

    def _ensure_col(self, table: str, col: str, decl: str) -> None:
        if col not in self._cols(table):
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")

    def _migrate(self) -> None:
        for col, decl in (("entry_time", "TEXT"), ("partial_taken", "INTEGER"), ("mode", "TEXT")):
            self._ensure_col("trades", col, decl)
        self._ensure_col("equity", "mode", "TEXT")

    def log_trade(self, t: TradeResult) -> None:
        self.conn.execute(
            """INSERT INTO trades
               (ts, symbol, direction, entry, exit, qty, pnl, r_multiple, liquidated,
                exit_reason, adds, fees, funding, entry_time, partial_taken, mode)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (pd.Timestamp(t.exit_time).isoformat(), t.symbol, t.direction.value, t.entry_price,
             t.exit_price, t.initial_qty, t.pnl, t.r_multiple, int(t.liquidated),
             t.exit_reason.value, t.max_adds, t.fees, t.funding,
             pd.Timestamp(t.entry_time).isoformat(), int(t.partial_taken), self.mode))
        self.conn.commit()

    def log_equity(self, ts: datetime, equity: float, peak: float) -> None:
        dd = (peak - equity) / peak if peak > 0 else 0.0
        self.conn.execute(
            "INSERT INTO equity (ts, equity, peak, drawdown, mode) VALUES (?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), equity, peak, dd, self.mode))
        self.conn.commit()

    def log_risk(self, ts: datetime, risk: "RiskEngine") -> None:
        peak = risk.peak_equity
        eq = risk.equity
        dd = (peak - eq) / peak if peak > 0 else 0.0
        pause = risk.pause_until.isoformat() if risk.pause_until is not None else None
        self.conn.execute(
            """INSERT INTO risk_snapshot
               (ts, mode, equity, peak, drawdown, halt_reason, kill, streak, daily_halt, pause_until)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (pd.Timestamp(ts).isoformat(), self.mode, eq, peak, dd, risk.halt_reason or "",
             int(risk.kill_switch_active), int(risk.consecutive_losses), int(risk.daily_halt), pause))
        self.conn.commit()

    def log_event(self, ts: datetime, symbol: str, kind: str, payload: Optional[dict] = None) -> None:
        self.conn.execute(
            "INSERT INTO events (ts, mode, symbol, kind, payload) VALUES (?,?,?,?,?)",
            (pd.Timestamp(ts).isoformat(), self.mode, symbol, kind,
             json.dumps(payload or {}, default=str)))
        self.conn.commit()

    def save_position(self, symbol: str, pos: Optional[Position]) -> None:
        if pos is None:
            self.conn.execute("DELETE FROM positions WHERE symbol=?", (symbol,))
        else:
            self.conn.execute("INSERT OR REPLACE INTO positions VALUES (?,?)",
                              (symbol, json.dumps(pos.to_dict())))
        self.conn.commit()

    def load_positions(self) -> dict:
        rows = self.conn.execute("SELECT symbol, state FROM positions").fetchall()
        return {s: Position.from_dict(json.loads(j)) for s, j in rows}

    def closed_pnl_sum(self, mode: Optional[str] = None) -> float:
        """Sum of closed-trade PnL for cash reconstruction after a paper restart."""
        q = "SELECT COALESCE(SUM(pnl), 0) FROM trades"
        args: tuple = ()
        if mode:
            q += " WHERE mode=?"
            args = (mode,)
        row = self.conn.execute(q, args).fetchone()
        return float(row[0] if row and row[0] is not None else 0.0)

    def load_equity(self, mode: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT ts, equity, peak, drawdown, mode FROM equity"
        args: tuple = ()
        if mode:
            q += " WHERE mode=? OR mode IS NULL"
            args = (mode,)
        q += " ORDER BY ts"
        df = pd.read_sql_query(q, self.conn, params=args)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def load_trades(self, mode: Optional[str] = None) -> pd.DataFrame:
        q = """SELECT ts, symbol, direction, entry, exit, qty, pnl, r_multiple, liquidated,
                      exit_reason, adds, fees, funding, entry_time, partial_taken, mode
               FROM trades"""
        args: tuple = ()
        if mode:
            q += " WHERE mode=? OR mode IS NULL"
            args = (mode,)
        q += " ORDER BY ts"
        df = pd.read_sql_query(q, self.conn, params=args)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
            if "entry_time" in df.columns:
                df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True, errors="coerce")
        return df

    def load_events(self, mode: Optional[str] = None, limit: int = 500) -> pd.DataFrame:
        q = "SELECT ts, mode, symbol, kind, payload FROM events"
        args: list = []
        if mode:
            q += " WHERE mode=?"
            args.append(mode)
        q += " ORDER BY ts DESC LIMIT ?"
        args.append(limit)
        df = pd.read_sql_query(q, self.conn, params=args)
        if not df.empty:
            df["ts"] = pd.to_datetime(df["ts"], utc=True)
        return df

    def load_risk(self, mode: Optional[str] = None) -> Optional[dict]:
        q = """SELECT ts, mode, equity, peak, drawdown, halt_reason, kill, streak,
                      daily_halt, pause_until FROM risk_snapshot"""
        args: tuple = ()
        if mode:
            q += " WHERE mode=?"
            args = (mode,)
        q += " ORDER BY ts DESC LIMIT 1"
        row = self.conn.execute(q, args).fetchone()
        if not row:
            return None
        cols = ["ts", "mode", "equity", "peak", "drawdown", "halt_reason", "kill",
                "streak", "daily_halt", "pause_until"]
        return dict(zip(cols, row))

    def close(self) -> None:
        self.conn.close()

    @staticmethod
    def open_readonly(path: str) -> sqlite3.Connection:
        """Dashboard-safe reader (WAL-friendly). Raises if the file is missing."""
        uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=10.0)
        conn.row_factory = sqlite3.Row
        return conn


class Notifier:
    def __init__(self) -> None:
        self.token = os.getenv("TELEGRAM_TOKEN")
        self.chat_id = os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = bool(self.token and self.chat_id)

    def send(self, msg: str) -> None:
        log.info("NOTIFY: %s", msg)
        if not self.enabled:
            return
        try:
            import urllib.request
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{self.token}/sendMessage",
                data=json.dumps({"chat_id": self.chat_id, "text": msg}).encode(),
                headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:  # noqa: BLE001
            log.warning("Notifier failed: %s", e)


# ============================================================
# 11. BACKTEST ENGINE (chronological, multi-symbol, shared equity)
# ============================================================


@dataclass
class BacktestResult:
    pyramiding: bool
    partial: bool
    leverage: int
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    expectancy_r: float
    exp_ci: tuple
    profit_factor: float
    max_drawdown: float
    liquidations: int
    trades: int
    win_rate: float
    top5_pct: float
    avg_adds: float
    halted: str
    equity_curve: pd.Series
    trade_list: list


class BacktestEngine:
    def __init__(self, data: dict, cfg: Config, start: Optional[pd.Timestamp] = None,
                 end: Optional[pd.Timestamp] = None):
        self.data, self.cfg, self.start, self.end = data, cfg, start, end
        self.strategy = make_strategy(cfg)
        self.pm = PositionManager(cfg)

    # ---- helpers ----
    def _slip(self, px: float, buy: bool) -> float:
        return px * (1 + self.cfg.slip) if buy else px * (1 - self.cfg.slip)

    def _equity(self) -> float:
        eq = self.cash
        for s, p in self.positions.items():
            eq += p.unrealized(self.last_close.get(s, p.avg_entry))
        return eq

    def _used_margin(self) -> float:
        return sum(p.margin() for p in self.positions.values())

    def _open_notional(self) -> float:
        return sum(p.current_qty * self.last_close.get(s, p.avg_entry)
                   for s, p in self.positions.items())

    def _net_notional(self) -> float:
        """Signed notional: long positive, short negative."""
        return sum(
            p.sign * p.current_qty * self.last_close.get(s, p.avg_entry)
            for s, p in self.positions.items()
        )

    # ---- main loop ----
    def run(self, label: Optional[tuple] = None) -> BacktestResult:
        cfg = self.cfg
        self.risk = RiskEngine(cfg, silent=True)
        self.cash = cfg.starting_equity
        self.positions: dict = {}
        self.pend_entry: dict = {}
        self.pend_act: dict = {}
        self.last_close: dict = {}
        self.cooldown: dict = {}
        self.trades: list = []
        curve: list = []

        stamps = sorted(set().union(*[set(sd.index) for sd in self.data.values()]))
        stamps = [t for t in stamps
                  if (self.start is None or t >= self.start) and (self.end is None or t < self.end)]
        if not stamps:
            raise DataError("no timestamps in the requested window")

        for ts in stamps:
            for sym, sd in self.data.items():
                i = sd.index.get(ts)
                if i is None:
                    continue
                self._step_symbol(sym, sd.bars[i], sd, ts)
            eq = self._equity()
            self.risk.update_equity(eq, ts)
            curve.append((ts, eq))

        # close whatever is still open at the last known close
        for sym in list(self.positions):
            px = self._slip(self.last_close[sym], buy=self.positions[sym].sign < 0)
            self._close(sym, px, stamps[-1], ExitReason.END_OF_DATA)
        curve[-1] = (stamps[-1], self.cash)
        return self._summarize(curve, label)

    def _step_symbol(self, sym: str, bar: Any, sd: SymbolData, ts: pd.Timestamp) -> None:
        cfg = self.cfg
        # 1) funding at the bar open for positions already held
        pos = self.positions.get(sym)
        if pos and ts.hour % cfg.funding_interval_hours == 0 and ts.minute == 0:
            cost = pos.sign * pos.current_qty * bar.open * cfg.funding_rate
            pos.funding += cost
            self.cash -= cost

        # 2) execute orders decided at the previous close, at this open
        if pos:
            acts = self.pend_act.pop(sym, [])
            if acts and self._apply_actions(sym, pos, acts, bar, sd, ts):
                return  # closed at the open
        else:
            sig = self.pend_entry.pop(sym, None)
            if sig and self.risk.trading_allowed(ts):
                self._open(sym, sig, bar, sd, ts)

        # 3) intrabar stop / liquidation check
        pos = self.positions.get(sym)
        if pos and self._check_stop(sym, pos, bar, ts):
            self.last_close[sym] = bar.close
            return

        # 4) decisions at this close (executed next open)
        self.last_close[sym] = bar.close
        pos = self.positions.get(sym)
        if pos:
            self.pend_act[sym] = self.pm.decide(
                pos, bar, allowed=self.risk.trading_allowed(ts), kill=self.risk.kill_switch_active,
                equity=self._equity(), used_margin=self._used_margin(), spec=sd.spec)
        elif (self.risk.trading_allowed(ts) and ts >= self.cooldown.get(sym, ts)
              and len(self.positions) + len(self.pend_entry) < cfg.max_open_positions):
            sig = self.strategy.on_bar(sym, bar)
            if sig:
                self.pend_entry[sym] = sig

    # ---- order execution ----
    def _open(self, sym: str, sig: Signal, bar: Any, sd: SymbolData, ts: pd.Timestamp) -> None:
        cfg = self.cfg
        buy = sig.direction is Direction.LONG
        entry = self._slip(bar.open, buy)
        stop = entry - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        dec = self.risk.evaluate(
            sig.direction, entry, stop, equity=self._equity(),
            open_notional=self._open_notional(), used_margin=self._used_margin(),
            open_positions=len(self.positions), spec=sd.spec, ts=ts,
            net_notional=self._net_notional(),
        )
        if not dec.approved:
            return
        fee = entry * dec.qty * cfg.fee_taker
        self.cash -= fee
        self.positions[sym] = Position.open(sym, sig.direction, dec.leverage, entry, dec.qty,
                                            stop, ts, fee)

    def _apply_actions(self, sym: str, pos: Position, acts: list, bar: Any, sd: SymbolData,
                       ts: pd.Timestamp) -> bool:
        """Returns True if the position was closed."""
        cfg = self.cfg
        buy_to_close = pos.sign < 0
        for a in acts:
            if a.kind is ActionKind.EXIT:
                self._close(sym, self._slip(bar.open, buy_to_close), ts, a.reason)
                return True
        for a in acts:
            if a.kind is ActionKind.PARTIAL:
                px = self._slip(bar.open, buy_to_close)
                qty = min(a.qty, pos.current_qty)
                fee = px * qty * cfg.fee_taker
                self.cash += pos.reduce(px, qty, fee) - fee
                pos.partial_done = True
            elif a.kind is ActionKind.ADD:
                px = self._slip(bar.open, buy=pos.sign > 0)
                qty, dec = self.pm.roll.size_add(pos, px, a.price, self._equity(),
                                                 self._used_margin(), sd.spec)
                if dec is AddDecision.ALLOW and qty:
                    fee = px * qty * cfg.fee_taker
                    self.cash -= fee
                    pos.add_lot(px, qty, ts, fee, a.price, pos.unrealized(px))
            elif a.kind is ActionKind.MOVE_STOP:
                pos.tighten_stop(a.price)
        return False

    def _check_stop(self, sym: str, pos: Position, bar: Any, ts: pd.Timestamp) -> bool:
        cfg = self.cfg
        liq = pos.liq_price(cfg.maintenance_margin_rate)
        long = pos.sign > 0
        hit_stop = bar.low <= pos.current_stop if long else bar.high >= pos.current_stop
        hit_liq = bar.low <= liq if long else bar.high >= liq
        if hit_stop and ((pos.current_stop > liq) if long else (pos.current_stop < liq)):
            raw = min(pos.current_stop, bar.open) if long else max(pos.current_stop, bar.open)
            if (raw <= liq) if long else (raw >= liq):  # gapped through the liquidation price
                self._close(sym, liq, ts, ExitReason.LIQUIDATION, liquidated=True)
                return True
            reason = ExitReason.TRAILING if pos.is_tighter(pos.initial_stop) and \
                pos.current_stop != pos.initial_stop else ExitReason.STOP
            self._close(sym, self._slip(raw, buy=not long), ts, reason)
            return True
        if hit_liq:
            self._close(sym, liq, ts, ExitReason.LIQUIDATION, liquidated=True)
            return True
        return False

    def _close(self, sym: str, px: float, ts: pd.Timestamp, reason: ExitReason,
               liquidated: bool = False) -> None:
        pos = self.positions.pop(sym)
        fee = 0.0 if liquidated else px * pos.current_qty * self.cfg.fee_taker
        res, cash_delta = pos.close_out(px, fee, ts, reason, liquidated)
        self.cash += cash_delta
        self.trades.append(res)
        self.risk.register_trade_result(res.r_multiple, ts)
        self.cooldown[sym] = ts + self.cfg.cooldown_bars * self.cfg.bar_td
        self.pend_act.pop(sym, None)

    # ---- metrics ----
    def _summarize(self, curve: list, label: Optional[tuple]) -> BacktestResult:
        cfg = self.cfg
        lab = label or (cfg.enable_pyramiding, cfg.enable_partial, cfg.leverage)
        # Defensive: never crash on a short/odd label (older call sites used ("recent",))
        if not isinstance(lab, (tuple, list)) or len(lab) < 3:
            lab = (cfg.enable_pyramiding, cfg.enable_partial, cfg.leverage)
        eq = pd.Series([e for _, e in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
        final = float(eq.iloc[-1])
        days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0, 1e-9)
        total_ret = final / cfg.starting_equity - 1.0
        cagr = (final / cfg.starting_equity) ** (365.0 / days) - 1.0 if days >= 90 and final > 0 else float("nan")

        dr = eq.resample("1D").last().dropna().pct_change().dropna()
        sharpe = float(dr.mean() / dr.std() * math.sqrt(365)) if len(dr) > 2 and dr.std() > 0 else float("nan")
        dd_dev = math.sqrt(float(np.mean(np.minimum(dr.to_numpy(), 0.0) ** 2))) if len(dr) else 0.0
        sortino = float(dr.mean() / dd_dev * math.sqrt(365)) if dd_dev > 0 else float("nan")
        max_dd = float(((eq.cummax() - eq) / eq.cummax().replace(0, np.nan)).max()) if len(eq) else 0.0

        t = self.trades
        r = np.array([x.r_multiple for x in t]) if t else np.array([])
        pnl = [x.pnl for x in t]
        gw, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
        wins = sorted((p for p in pnl if p > 0), reverse=True)
        ci = (float("nan"), float("nan"))
        if len(r) >= 10:
            rng = np.random.default_rng(0)
            means = rng.choice(r, size=(2000, len(r)), replace=True).mean(axis=1)
            ci = (float(np.percentile(means, 5)), float(np.percentile(means, 95)))
        return BacktestResult(
            pyramiding=bool(lab[0]), partial=bool(lab[1]), leverage=int(lab[2]), final_equity=final,
            total_return=total_ret, cagr=cagr, sharpe=sharpe, sortino=sortino,
            expectancy_r=float(r.mean()) if len(r) else 0.0, exp_ci=ci,
            profit_factor=gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0),
            max_drawdown=max_dd, liquidations=sum(x.liquidated for x in t), trades=len(t),
            win_rate=float((r > 0).mean()) if len(r) else 0.0,
            top5_pct=100.0 * sum(wins[:5]) / gw if gw > 0 else 0.0,
            avg_adds=float(np.mean([x.max_adds for x in t])) if t else 0.0,
            halted=self.risk.halt_reason, equity_curve=eq, trade_list=t)


def run_grid(data: dict, cfg: Config, start=None, end=None) -> list:
    out = []
    for p, q, lev in product([False, True], [False, True], [3, 5]):
        c = replace(cfg, enable_pyramiding=p, enable_partial=q, leverage=lev,
                    max_account_leverage=float(min(cfg.max_account_leverage, lev)))
        out.append(BacktestEngine(data, c, start, end).run((p, q, lev)))
    return out


def results_as_frame(results: list) -> pd.DataFrame:
    """Tabular view of BacktestResult list for CLI / Streamlit."""
    rows = []
    for r in results:
        lo, hi = r.exp_ci
        rows.append({
            "pyr": r.pyramiding, "part": r.partial, "lev": r.leverage,
            "final": round(r.final_equity, 2), "ret": r.total_return,
            "cagr": r.cagr, "sharpe": r.sharpe, "sortino": r.sortino,
            "exp_r": r.expectancy_r, "ci_lo": lo, "ci_hi": hi,
            "pf": r.profit_factor if math.isfinite(r.profit_factor) else float("nan"),
            "max_dd": r.max_drawdown, "liq": r.liquidations, "n": r.trades,
            "win": r.win_rate, "top5_pct": r.top5_pct, "avg_adds": r.avg_adds,
            "halt": (r.halted or "")[:48],
        })
    return pd.DataFrame(rows)


def results_table(results: list) -> str:
    df = results_as_frame(results)
    if df.empty:
        return "(no results)"
    show = df.copy()
    show["lev"] = show["lev"].map(lambda x: f"{x}x")
    show["ret"] = show["ret"].map(lambda x: f"{x:+.0%}")
    show["cagr"] = show["cagr"].map(lambda x: "n/a" if math.isnan(x) else f"{x:.0%}")
    show["sharpe"] = show["sharpe"].map(lambda x: "n/a" if math.isnan(x) else round(x, 2))
    show["exp_r"] = show["exp_r"].map(lambda x: round(x, 3))
    show["ci"] = [
        "n/a" if math.isnan(lo) else f"[{lo:.2f},{hi:.2f}]"
        for lo, hi in zip(show["ci_lo"], show["ci_hi"])
    ]
    show["pf"] = show["pf"].map(lambda x: round(x, 2) if x == x else "n/a")
    show["max_dd"] = show["max_dd"].map(lambda x: f"{x:.0%}")
    show["win"] = show["win"].map(lambda x: f"{x:.0%}")
    show["top5_pct"] = show["top5_pct"].map(lambda x: f"{x:.0f}")
    show = show.rename(columns={"exp_r": "exp(R)", "ci": "90%CI", "max_dd": "maxDD",
                                "top5_pct": "top5%", "cagr": "CAGR", "sharpe": "Sharpe"})
    out = show[["pyr", "part", "lev", "final", "ret", "CAGR", "Sharpe", "exp(R)", "90%CI",
                "pf", "maxDD", "liq", "n", "win", "top5%", "halt"]]
    try:
        return out.to_markdown(index=False)
    except ImportError:
        return out.to_string(index=False)


def verdicts(results: list) -> list:
    """KEEP only if the candidate is better AND still acceptable (non-negative expectancy)."""
    idx = {(r.pyramiding, r.partial, r.leverage): r for r in results}
    out = []
    for part, lev in product([False, True], [3, 5]):
        a, b = idx[(False, part, lev)], idx[(True, part, lev)]
        exp_ok = b.expectancy_r > a.expectancy_r * 1.10 if a.expectancy_r > 0 else b.expectancy_r > a.expectancy_r
        liq_a = a.liquidations / a.trades if a.trades else 0.0
        liq_b = b.liquidations / b.trades if b.trades else 0.0
        better = (exp_ok or b.top5_pct > a.top5_pct) and (b.max_drawdown - a.max_drawdown <= 0.05) \
            and (liq_b - liq_a <= 0.02)
        keep = better and b.expectancy_r > 0 and b.trades >= 10
        tag = "KEEP" if keep else ("DROP (still losing)" if better and b.expectancy_r <= 0 else "DROP")
        out.append(f"pyramiding @ partial={part}, {lev}x: {tag} "
                   f"(exp {a.expectancy_r:+.2f}R -> {b.expectancy_r:+.2f}R, "
                   f"DD {a.max_drawdown:.0%} -> {b.max_drawdown:.0%})")
    for pyr, lev in product([False, True], [3, 5]):
        a, b = idx[(pyr, False, lev)], idx[(pyr, True, lev)]
        better = b.final_equity >= a.final_equity * 1.15 and (
            not math.isnan(b.sharpe) and not math.isnan(a.sharpe) and b.sharpe > a.sharpe
        )
        keep = better and b.expectancy_r > 0 and b.trades >= 10
        tag = "KEEP" if keep else "DROP"
        out.append(f"partial@6R @ pyramiding={pyr}, {lev}x: {tag} "
                   f"(final {a.final_equity:.2f} -> {b.final_equity:.2f})")
    hi = idx[(True, True, 5)]
    base3 = idx[(True, False, 3)]
    liq_rate = hi.liquidations / hi.trades if hi.trades else 0.0
    if liq_rate > 0.10 or hi.max_drawdown > 0.40 or base3.expectancy_r > hi.expectancy_r:
        out.append("leverage: PREFER 3x (safer; 5x not clearly better on this sample)")
    else:
        out.append("leverage: 5x did not breach liq/DD limits — still prefer 3x until OOS n is large")
    return out


def promotion_reading(ins: "BacktestResult", oos: "BacktestResult") -> str:
    """Honest label for train vs test — never celebrate a losing train as success."""
    if ins.trades < 10 or oos.trades < 10:
        return "thin sample"
    if ins.expectancy_r <= 0 and oos.expectancy_r > 0:
        return "train failed · test lucky?"
    if ins.expectancy_r <= 0 and oos.expectancy_r <= 0:
        return "train failed"
    good_is = ins.expectancy_r > 0 and (math.isnan(ins.sharpe) or ins.sharpe > 0.3)
    weak_oos = oos.expectancy_r < ins.expectancy_r * 0.5 or (
        not math.isnan(oos.sharpe) and not math.isnan(ins.sharpe) and oos.sharpe < ins.sharpe * 0.5
    )
    if good_is and weak_oos:
        return "overfit risk"
    if oos.expectancy_r > 0 and oos.trades >= 30 and ins.expectancy_r > 0:
        return "both halves OK"
    if oos.expectancy_r > 0 and ins.expectancy_r > 0:
        return "both positive · thin"
    return "check manually"


# ============================================================
# 12. LIVE / PAPER RUNNER (same decision logic as the backtest)
# ============================================================


class LiveRunner:
    def __init__(self, cfg: Config, broker: BaseBroker, market: MarketData, symbols: list,
                 live: bool):
        self.cfg, self.broker, self.market, self.symbols, self.live = cfg, broker, market, symbols, live
        self.risk = RiskEngine(cfg)
        self.strategy = make_strategy(cfg)
        self.pm = PositionManager(cfg)
        self.db = Persistence(cfg.db_path, mode="live" if live else "paper")
        self.notify = Notifier()
        self.positions: dict = self.db.load_positions()
        self.specs = {s: market.instrument(s) for s in symbols}
        self.last_bar: dict = {}
        self.last_price: dict = {}
        self.cooldown: dict = {}
        self.blocked: set = set()
        self.pend_entry: dict = {}  # sym -> Signal; fill once next bar opens
        self.pend_after: dict = {}  # sym -> Timestamp (earliest fill time = signal bar + 1)
        self.pend_exit: dict = {}  # sym -> (ExitReason, after_ts); fill at next open
        self._seen_broker_pos: set = set()  # symbols confirmed on broker this session
        self._last_heartbeat = 0.0
        self._last_funding_key: Optional[pd.Timestamp] = None
        self._bars_seen = 0
        self._signals_seen = 0
        self._btc_trend = 0.0
        self._raw: dict = {}  # symbol -> OHLCV for prepare_data parity with hunt
        self._book_feats: dict = {}  # symbol -> feature frame from prepare_data
        self._stale_log_at: dict = {}  # symbol -> last stale warning monotonic time
        self._cache_age_sec: Optional[float] = None
        # Paper must rebuild cash from closed trades even with 0 open positions.
        # Old gate `and self.positions` left wallet stuck at $50 after flat restarts
        # while Live still counted closed trades → "19 closed, balance $50".
        if not live and isinstance(broker, PaperBroker):
            closed = self.db.closed_pnl_sum(mode="paper")
            if self.positions or abs(closed) > 1e-12:
                broker.rehydrate(self.positions or {}, closed)
                log.info(
                    "paper rehydrated %d open · closed_pnl=%+.2f · cash=%.2f",
                    len(self.positions), closed, broker.cash,
                )
        # Poll BTC first so alt entries see a fresh regime stamp
        if cfg.require_btc_regime:
            btc = _btc_symbol_in(symbols)
            if btc and symbols[0] != btc:
                self.symbols = [btc] + [s for s in symbols if s != btc]

    def _runtime_path(self) -> Path:
        return Path(self.cfg.db_path).resolve().parent / PAPER_RUNTIME_PATH.name

    def _save_runtime(self) -> None:
        """Persist risk / cooldown / pending queues across paper restarts."""
        if self.live:
            return
        try:
            cool = {
                s: pd.Timestamp(t).isoformat()
                for s, t in self.cooldown.items()
                if t is not None
            }
            pend_after = {
                s: pd.Timestamp(t).isoformat()
                for s, t in self.pend_after.items()
                if t is not None
            }
            pend_exit = {
                s: {
                    "reason": (r.value if hasattr(r, "value") else str(r)),
                    "after": pd.Timestamp(a).isoformat(),
                }
                for s, (r, a) in self.pend_exit.items()
            }
            payload = {
                "ts": _utc_now().isoformat(),
                "peak_equity": float(self.risk.peak_equity),
                "equity": float(self.risk.equity),
                "kill": bool(self.risk.kill_switch_active),
                "halt_reason": str(self.risk.halt_reason or ""),
                "daily_halt": bool(self.risk.daily_halt),
                "streak": int(self.risk.consecutive_losses),
                "pause_until": (
                    None if self.risk.pause_until is None
                    else pd.Timestamp(self.risk.pause_until).isoformat()
                ),
                "rolling_r": [float(x) for x in list(self.risk.rolling_r)[-50:]],
                "cooldown": cool,
                "pend_after": pend_after,
                "pend_exit": pend_exit,
                "blocked": sorted(self.blocked),
            }
            path = self._runtime_path()
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("paper runtime save failed: %s", e)

    def _restore_runtime(self) -> None:
        path = self._runtime_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("paper runtime load failed: %s", e)
            return
        try:
            peak = float(data.get("peak_equity") or 0)
            if peak > 0:
                self.risk.peak_equity = max(float(self.risk.peak_equity), peak)
            if data.get("kill"):
                self.risk.kill_switch_active = True
                self.risk.halt_reason = str(data.get("halt_reason") or "restored")
            self.risk.daily_halt = bool(data.get("daily_halt"))
            self.risk.consecutive_losses = int(data.get("streak") or 0)
            pu = data.get("pause_until")
            if pu:
                self.risk.pause_until = pd.Timestamp(pu)
            rr = data.get("rolling_r") or []
            if isinstance(rr, list) and rr:
                self.risk.rolling_r = [float(x) for x in rr]
            for s, t in (data.get("cooldown") or {}).items():
                self.cooldown[str(s)] = pd.Timestamp(t)
            for s, t in (data.get("pend_after") or {}).items():
                # Signals themselves aren't restored — drop stale queue keys
                self.pend_after.pop(str(s), None)
            for s, blob in (data.get("pend_exit") or {}).items():
                try:
                    reason = ExitReason(str(blob.get("reason")))
                    after = pd.Timestamp(blob.get("after"))
                    self.pend_exit[str(s)] = (reason, after)
                except Exception:
                    pass
            for s in data.get("blocked") or []:
                self.blocked.add(str(s))
            log.info(
                "paper runtime restored (peak=%.2f kill=%s cooldown=%d)",
                self.risk.peak_equity, self.risk.kill_switch_active, len(self.cooldown),
            )
        except Exception as e:
            log.warning("paper runtime apply failed: %s", e)

    def run_forever(self) -> None:
        mode = "LIVE" if self.live else "PAPER"
        # Unblock paper if live-sized CSVs froze behind wall clock
        try:
            clear_stale_live_caches(
                tf=str(self.cfg.timeframe), bars=int(self.cfg.live_history_bars),
            )
        except Exception as e:
            log.warning("stale cache clear failed: %s", e)
        self.notify.send(
            f"moonshot started ({mode}) symbols={self.symbols} "
            f"tf={self.cfg.timeframe} lev={self.cfg.leverage}x "
            f"hist={self.cfg.live_history_bars}"
        )
        eq0 = self.broker.get_equity()
        snap = None if self.live else self.db.load_risk(mode="paper")
        self.risk.reset(eq0)
        if snap and not self.live:
            try:
                peak = float(snap.get("peak") or 0)
                if peak > eq0:
                    self.risk.peak_equity = peak
                if int(snap.get("kill") or 0):
                    self.risk.kill_switch_active = True
                    self.risk.halt_reason = str(snap.get("halt_reason") or "restored")
                if int(snap.get("daily_halt") or 0):
                    self.risk.daily_halt = True
                self.risk.consecutive_losses = int(snap.get("streak") or 0)
                if snap.get("pause_until"):
                    self.risk.pause_until = pd.Timestamp(snap["pause_until"])
            except Exception as e:
                log.warning("risk snapshot restore failed: %s", e)
        if not self.live:
            self._restore_runtime()
        try:
            while True:
                if self.cfg.require_btc_regime and not _btc_symbol_in(self.symbols):
                    try:
                        self._refresh_btc_regime()
                    except Exception as e:
                        log.warning("BTC regime refresh failed: %s", e)
                for sym in self.symbols:
                    try:
                        self._poll(sym)
                    except BotError as e:
                        log.error("%s: %s", sym, e)
                        self.notify.send(f"ERROR {sym}: {e}")
                    except Exception as e:  # noqa: BLE001 — keep paper alive on MEXC timeouts
                        log.warning("%s poll failed (continuing): %s", sym, e)
                self._heartbeat()
                time.sleep(self.cfg.poll_seconds)
        except KeyboardInterrupt:
            log.info("stopped by user (positions keep their exchange-side stops)")
        finally:
            self.db.close()

    def _heartbeat(self) -> None:
        """Wall-clock pulse so the Live page does not look frozen between candles."""
        now = time.time()
        if now - self._last_heartbeat < 60:
            return
        self._last_heartbeat = now
        ts = pd.Timestamp.now(tz="UTC")
        equity = self.broker.get_equity()
        self.db.log_equity(ts, equity, self.risk.peak_equity)
        self.db.log_risk(ts, self.risk)
        # next candle close (approximate)
        floored = ts.floor(self.cfg.pandas_tf)
        next_close = floored + self.cfg.bar_td
        mins = max(0, int((next_close - ts).total_seconds() // 60))
        age = self._cache_age_sec
        age_s = f" · cache {int(age)}s" if age is not None else ""
        msg = (
            f"Paper alive · {self.cfg.timeframe} · {self.cfg.strategy} · "
            f"{self.cfg.leverage}x · {len(self.positions)} open · "
            f"{len(self.symbols)} coins · next candle decision in ~{mins}m · "
            f"bars={self._bars_seen} setups={self._signals_seen}{age_s}"
        )
        log.info(msg)
        try:
            _live_feed(
                "paper_tick", msg,
                open=len(self.positions), next_mins=mins,
                timeframe=self.cfg.timeframe, equity=round(equity, 2),
                bars_seen=int(self._bars_seen), setups_seen=int(self._signals_seen),
                adx_enter=float(self.cfg.adx_enter),
                vol_mult=float(self.cfg.volume_multiplier),
                require_rising_adx=bool(self.cfg.require_rising_adx),
                leverage=int(self.cfg.leverage),
                cache_age_sec=None if age is None else round(float(age), 1),
            )
        except Exception:
            pass

    def _refresh_book_features(self) -> None:
        """Rebuild hunt-parity features (btc_adx / breadth) across the paper book."""
        if len(self._raw) < max(1, min(3, len(self.symbols))):
            return
        try:
            # Mirror prepare_data feature stamping (SymbolData has bars, not a feats frame)
            feats = {s: build_features(df, self.cfg) for s, df in self._raw.items()}
            btc = _btc_symbol_in(feats.keys())
            cfg = self.cfg
            if cfg.require_btc_regime and btc is not None:
                if "h4_bias" in feats[btc].columns:
                    btc_trend = feats[btc]["h4_bias"].astype(float).fillna(0.0)
                else:
                    btc_trend = feats[btc]["regime"].map(_regime_to_trend).astype(float)
                for s, f in feats.items():
                    if _is_btc_symbol(s):
                        f["btc_trend"] = btc_trend
                    else:
                        f["btc_trend"] = btc_trend.reindex(f.index).ffill().fillna(0.0)
            if btc is not None:
                btc_adx = feats[btc]["adx"].astype(float)
            else:
                btc_adx = pd.concat([f["adx"] for f in feats.values()], axis=1).mean(axis=1)
            breadth = pd.concat(
                [f["h4_bias"].astype(float) for f in feats.values()], axis=1,
            ).mean(axis=1)
            for f in feats.values():
                f["btc_adx"] = btc_adx.reindex(f.index).ffill().fillna(0.0)
                f["breadth"] = breadth.reindex(f.index).ffill().fillna(0.0)
            self._book_feats = feats
            if btc and btc in feats:
                self._btc_trend = self._btc_bias_from_feats(feats[btc])
        except Exception as e:
            log.warning("prepare_data book refresh failed: %s", e)

    def _bar_open_price(self, sym: str, fallback: float) -> float:
        """Fill at the open of the newest bar (backtest parity), else ticker."""
        df = self._raw.get(sym)
        if df is not None and not df.empty and "open" in df.columns:
            try:
                return float(df["open"].iloc[-1])
            except Exception:
                pass
        return float(fallback)

    def _bar_aligned_ts(self, sym: str, target: pd.Timestamp,
                        fallback: Optional[pd.Timestamp] = None) -> pd.Timestamp:
        """Snap a timestamp to an OHLCV bar open so charts/DB match candles."""
        ts = pd.Timestamp(target)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        df = self._raw.get(sym)
        if df is not None and not df.empty:
            idx = df.index
            # Exact bar open, else nearest at-or-before target
            if ts in idx:
                return pd.Timestamp(ts)
            pos = int(idx.searchsorted(ts, side="right")) - 1
            if pos >= 0:
                return pd.Timestamp(idx[pos])
            return pd.Timestamp(idx[0])
        floored = ts.floor(self.cfg.pandas_tf)
        return floored if fallback is None else pd.Timestamp(fallback)

    # ---- per-symbol cycle ----
    def _poll(self, sym: str) -> None:
        price = self.market.last_price(sym)
        self.last_price[sym] = price
        self.broker.on_price(sym, price)

        now = pd.Timestamp.now(tz="UTC")
        self._accrue_funding(now)
        expected = now.floor(self.cfg.pandas_tf) - self.cfg.bar_td  # open of latest closed bar

        # Refresh OHLCV before fills/exits so entry_time snaps to real candle opens
        # (wall-clock fills were stacking every marker on the chart's last bar).
        need_bars = (
            self.last_bar.get(sym, pd.Timestamp(0, tz="UTC")) < expected
            or sym not in self._raw
            or self._raw.get(sym) is None
            or getattr(self._raw.get(sym), "empty", True)
        )
        df = self._raw.get(sym)
        if need_bars:
            df = self.market.recent(sym)
            if df.empty or df.index[-1] < expected:
                try:
                    df = fetch_historical(
                        sym, self.cfg.timeframe, int(self.cfg.live_history_bars),
                        refresh=True,
                    )
                except Exception as e:
                    log.warning("%s forced OHLCV refresh failed: %s", sym, e)
            if not df.empty:
                self._raw[sym] = df
                try:
                    self._cache_age_sec = (now - pd.Timestamp(df.index[-1])).total_seconds()
                except Exception:
                    self._cache_age_sec = 0.0

        # Stop fills use the last closed bar time (not wall clock)
        self._reconcile(sym)

        # Fill queued entries at the *candle* open they belong to (pend_after),
        # never wall-clock `now` — that broke entry/exit markers on price charts.
        if sym in self.pend_entry and sym not in self.positions:
            after = self.pend_after.get(sym)
            if after is not None and now >= after:
                # Expire late queues — don't fill hours-old setups at a past open
                max_age = self.cfg.bar_td * int(PEND_ENTRY_MAX_BARS)
                if now - pd.Timestamp(after) > max_age:
                    self.pend_entry.pop(sym, None)
                    self.pend_after.pop(sym, None)
                    log.info("%s dropped stale queued entry (>%s late)", sym, max_age)
                    self._save_runtime()
                else:
                    raw = self._raw.get(sym)
                    # Wait until we have the fill bar (or later) so price+time match candles
                    if raw is None or raw.empty or pd.Timestamp(raw.index[-1]) < pd.Timestamp(after) - self.cfg.bar_td:
                        pass  # keep queued; next poll after refresh
                    else:
                        sig = self.pend_entry.pop(sym)
                        self.pend_after.pop(sym, None)
                        fill_ts = self._bar_aligned_ts(sym, after)
                        # Prefer the open of the fill bar when present
                        fill_px = price
                        try:
                            if fill_ts in raw.index:
                                fill_px = float(raw.loc[fill_ts, "open"])
                            else:
                                fill_px = self._bar_open_price(sym, price)
                        except Exception:
                            fill_px = self._bar_open_price(sym, price)
                        self._open_from_signal(sym, sig, fill_px, fill_ts, self.broker.get_equity())
                        self._save_runtime()
        elif sym in self.pend_entry and sym in self.positions:
            self.pend_entry.pop(sym, None)
            self.pend_after.pop(sym, None)

        # Queued exits fill at next bar open (backtest parity)
        if sym in self.pend_exit and sym in self.positions:
            reason, after = self.pend_exit[sym]
            if now >= after:
                raw = self._raw.get(sym)
                if raw is not None and not raw.empty and pd.Timestamp(raw.index[-1]) >= pd.Timestamp(after) - self.cfg.bar_td:
                    self.pend_exit.pop(sym, None)
                    fill_ts = self._bar_aligned_ts(sym, after)
                    fill_px = price
                    try:
                        if fill_ts in raw.index:
                            fill_px = float(raw.loc[fill_ts, "open"])
                    except Exception:
                        pass
                    pos = self.positions[sym]
                    self.broker.cancel_stop(sym)
                    side = "sell" if pos.sign > 0 else "buy"
                    fill = self.broker.market_order(sym, side, pos.current_qty, fill_px, reduce_only=True)
                    self._finish(sym, pos, fill.price, fill.fee, fill_ts, reason)

        if df is None or df.empty or df.index[-1] < expected:
            last = None if df is None or df.empty else pd.Timestamp(df.index[-1])
            age = None if last is None else (now - last).total_seconds()
            self._cache_age_sec = age
            tmon = time.monotonic()
            if tmon - float(self._stale_log_at.get(sym, 0.0)) >= 300.0:
                self._stale_log_at[sym] = tmon
                log.warning(
                    "%s OHLCV behind closed bar: last=%s expected=%s age=%ss tf=%s",
                    sym, last, expected, None if age is None else int(age), self.cfg.timeframe,
                )
            return
        if self.last_bar.get(sym, pd.Timestamp(0, tz="UTC")) >= expected:
            return
        self.last_bar[sym] = df.index[-1]
        self._bars_seen += 1
        # Prefer full-book prepare_data features (same as hunt); fall back to local build
        self._refresh_book_features()
        if sym in self._book_feats and not self._book_feats[sym].empty:
            feats = self._book_feats[sym]
        else:
            feats = build_features(df, self.cfg)
            if self.cfg.require_btc_regime:
                if _is_btc_symbol(sym):
                    self._btc_trend = self._btc_bias_from_feats(feats)
                feats["btc_trend"] = float(self._btc_trend)
                if "btc_adx" not in feats.columns:
                    feats["btc_adx"] = feats["adx"]
                if "breadth" not in feats.columns:
                    feats["breadth"] = feats.get("h4_bias", 0.0)
        bar = next(feats.tail(1).itertuples(name="Bar"))
        self._on_closed_bar(sym, bar, price, bar.Index + self.cfg.bar_td)

    @staticmethod
    def _btc_bias_from_feats(feats: pd.DataFrame) -> float:
        """Prefer completed 4H bias; fall back to 15m/daily regime."""
        if "h4_bias" in feats.columns and len(feats):
            v = float(feats["h4_bias"].iloc[-1])
            if v == v:
                return v
        return _regime_to_trend(feats["regime"].iloc[-1])

    def _refresh_btc_regime(self) -> None:
        """When BTC is not in the trade universe, still read it for the alt gate."""
        btc = "BTC/USDT:USDT"
        df = self.market.recent(btc)
        if df.empty:
            return
        feats = build_features(df, self.cfg)
        self._btc_trend = self._btc_bias_from_feats(feats)

    def _used_margin(self) -> float:
        return sum(p.margin() for p in self.positions.values())

    def _open_notional(self) -> float:
        return sum(p.current_qty * self.last_price.get(s, p.avg_entry) for s, p in self.positions.items())

    def _net_notional(self) -> float:
        return sum(
            p.sign * p.current_qty * self.last_price.get(s, p.avg_entry)
            for s, p in self.positions.items()
        )

    def _accrue_funding(self, ts: pd.Timestamp) -> None:
        """Match backtest: charge constant funding at settlement hours (paper cash)."""
        cfg = self.cfg
        if not self.positions:
            return
        if ts.minute > 2:  # only near the hour
            return
        if ts.hour % max(1, int(cfg.funding_interval_hours)) != 0:
            return
        key = ts.floor("h")
        if self._last_funding_key is not None and key <= self._last_funding_key:
            return
        self._last_funding_key = key
        for sym, pos in list(self.positions.items()):
            px = float(self.last_price.get(sym, pos.avg_entry))
            cost = pos.sign * pos.current_qty * px * cfg.funding_rate
            pos.funding += cost
            if isinstance(self.broker, PaperBroker):
                self.broker.cash -= cost
            self.db.save_position(sym, pos)
            self.db.log_event(ts, sym, "funding", {"cost": cost, "rate": cfg.funding_rate, "price": px})

    def _check_bar_stop_liq(self, sym: str, pos: Position, bar: Any, ts: pd.Timestamp) -> bool:
        """Paper OHLC stop + liquidation (ticker-only stops miss wicks)."""
        if not isinstance(self.broker, PaperBroker):
            return False
        cfg = self.cfg
        liq = pos.liq_price(cfg.maintenance_margin_rate)
        long = pos.sign > 0
        hit_stop = bar.low <= pos.current_stop if long else bar.high >= pos.current_stop
        hit_liq = bar.low <= liq if long else bar.high >= liq
        if hit_stop and ((pos.current_stop > liq) if long else (pos.current_stop < liq)):
            raw = min(pos.current_stop, bar.open) if long else max(pos.current_stop, bar.open)
            if (raw <= liq) if long else (raw >= liq):
                self.broker.cancel_stop(sym)
                # Flatten paper book at liq
                side = "sell" if long else "buy"
                fill = self.broker.market_order(sym, side, pos.current_qty, liq, reduce_only=True)
                self._finish(sym, pos, liq, 0.0, ts, ExitReason.LIQUIDATION)
                return True
            reason = (
                ExitReason.TRAILING
                if pos.is_tighter(pos.initial_stop) and pos.current_stop != pos.initial_stop
                else ExitReason.STOP
            )
            px = self.broker._slip(raw, buy=not long) if hasattr(self.broker, "_slip") else raw
            self.broker.cancel_stop(sym)
            side = "sell" if long else "buy"
            fill = self.broker.market_order(sym, side, pos.current_qty, px, reduce_only=True)
            self._finish(sym, pos, fill.price, fill.fee, ts, reason)
            return True
        if hit_liq:
            self.broker.cancel_stop(sym)
            side = "sell" if long else "buy"
            self.broker.market_order(sym, side, pos.current_qty, liq, reduce_only=True)
            self._finish(sym, pos, liq, 0.0, ts, ExitReason.LIQUIDATION)
            return True
        return False

    def _on_closed_bar(self, sym: str, bar: Any, price: float, ts: pd.Timestamp) -> None:
        # Decisions at bar close; entries/exits fill in _poll once the next bar opens.
        equity = self.broker.get_equity()
        self.risk.update_equity(equity, ts)
        self.db.log_equity(ts, equity, self.risk.peak_equity)
        self.db.log_risk(ts, self.risk)

        pos = self.positions.get(sym)
        if pos:
            # Wick-aware stops before discretionary exits
            if self._check_bar_stop_liq(sym, pos, bar, ts):
                self._save_runtime()
                return
            if sym in self.pend_exit:
                return  # already queued for next open
            acts = self.pm.decide(pos, bar, allowed=self.risk.trading_allowed(ts),
                                  kill=self.risk.kill_switch_active, equity=equity,
                                  used_margin=self._used_margin(), spec=self.specs[sym])
            # Queue EXIT for next open (match backtest); other acts still immediate
            exit_acts = [a for a in acts if a.kind is ActionKind.EXIT]
            other = [a for a in acts if a.kind is not ActionKind.EXIT]
            if exit_acts:
                self.pend_exit[sym] = (exit_acts[0].reason, pd.Timestamp(ts))
                self._save_runtime()
            if other:
                self._execute(sym, pos, other, price, ts, equity)
        elif sym not in self.blocked:
            self._queue_entry(sym, bar, ts)
        self._save_runtime()

    def _queue_entry(self, sym: str, bar: Any, ts: pd.Timestamp) -> None:
        """Queue a fill for the next bar open (ts already equals bar.Index + bar_td)."""
        cfg = self.cfg
        if ts < self.cooldown.get(sym, ts) or not self.risk.trading_allowed(ts):
            return
        if len(self.positions) + len(self.pend_entry) >= cfg.max_open_positions:
            return
        # Refuse setups on a stale book — fills against old opens then die as regime_invalid
        now = pd.Timestamp.now(tz="UTC")
        expected = now.floor(cfg.pandas_tf) - cfg.bar_td
        raw = self._raw.get(sym)
        if raw is None or raw.empty or pd.Timestamp(raw.index[-1]) < expected - cfg.bar_td:
            return
        sig = self.strategy.on_bar(sym, bar)
        if not sig:
            return
        self._signals_seen += 1
        self.pend_entry[sym] = sig
        self.pend_after[sym] = pd.Timestamp(ts)  # earliest fill = open of the new bar
        log.info("%s SETUP %s — queued for next open (%s)", sym, sig.direction.value, sig.reason)
        try:
            _live_feed(
                "setup",
                f"{sym.split('/')[0]} {sig.direction.value} setup — queued for next candle open",
                symbol=sym, direction=sig.direction.value, reason=sig.reason,
            )
        except Exception:
            pass

    def _open_from_signal(self, sym: str, sig: Signal, price: float, ts: pd.Timestamp,
                          equity: float) -> None:
        cfg = self.cfg
        if sym in self.positions or not self.risk.trading_allowed(ts):
            return
        stop = price - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        dec = self.risk.evaluate(
            sig.direction, price, stop, equity=equity,
            open_notional=self._open_notional(), used_margin=self._used_margin(),
            open_positions=len(self.positions), spec=self.specs[sym], ts=ts,
            net_notional=self._net_notional(),
        )
        if not dec.approved:
            log.info("%s queued signal rejected at open: %s", sym, dec.reason)
            try:
                _live_feed("reject", f"{sym.split('/')[0]} blocked at open: {dec.reason}",
                           symbol=sym, reason=dec.reason)
            except Exception:
                pass
            return
        side = "buy" if sig.direction is Direction.LONG else "sell"
        self.broker.set_leverage(sym, dec.leverage)
        fill = self.broker.market_order(sym, side, dec.qty, price)
        stop = fill.price - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        pos = Position.open(sym, sig.direction, dec.leverage, fill.price, fill.qty, stop, ts, fill.fee)
        try:
            self.broker.set_stop(sym, pos.direction, pos.current_qty, pos.current_stop)
        except ExecutionError:
            self.broker.market_order(sym, "sell" if side == "buy" else "buy", fill.qty, price, reduce_only=True)
            self.blocked.add(sym)
            raise
        self.positions[sym] = pos
        self.db.save_position(sym, pos)
        self.db.log_event(ts, sym, "open", {
            "direction": sig.direction.value, "qty": fill.qty, "price": fill.price,
            "stop": stop, "reason": sig.reason, "fill_timing": "next_open",
        })
        try:
            _live_feed(
                "open",
                f"OPEN {sym.split('/')[0]} {sig.direction.value} @ {fill.price:.6g}",
                symbol=sym, direction=sig.direction.value, price=fill.price, qty=fill.qty,
            )
        except Exception:
            pass
        self.notify.send(f"OPEN {sym} {sig.direction.value} qty={fill.qty:.6g} @ {fill.price:.6g} "
                         f"stop={stop:.6g} | {sig.reason}")

    def _execute(self, sym: str, pos: Position, acts: list, price: float, ts: pd.Timestamp,
                 equity: float) -> None:
        close_side = "sell" if pos.sign > 0 else "buy"
        open_side = "buy" if pos.sign > 0 else "sell"
        for a in acts:
            if a.kind is ActionKind.EXIT:
                self.broker.cancel_stop(sym)
                fill = self.broker.market_order(sym, close_side, pos.current_qty, price, reduce_only=True)
                self._finish(sym, pos, fill.price, fill.fee, ts, a.reason)
                return
        for a in acts:
            if a.kind is ActionKind.PARTIAL:
                fill = self.broker.market_order(sym, close_side, a.qty, price, reduce_only=True)
                pos.reduce(fill.price, fill.qty, fill.fee)
                pos.partial_done = True
                self.db.log_event(ts, sym, "partial", {
                    "qty": fill.qty, "price": fill.price, "remaining": pos.current_qty,
                })
            elif a.kind is ActionKind.ADD:
                qty, dec = self.pm.roll.size_add(pos, price, a.price, equity, self._used_margin(),
                                                 self.specs[sym])
                if dec is AddDecision.ALLOW and qty:
                    fill = self.broker.market_order(sym, open_side, qty, price)
                    pos.add_lot(fill.price, fill.qty, ts, fill.fee, a.price, pos.unrealized(fill.price))
                    self.db.log_event(ts, sym, "add", {
                        "qty": fill.qty, "price": fill.price, "avg": pos.avg_entry,
                        "stop": pos.current_stop,
                    })
                    self.notify.send(f"ADD {sym} +{fill.qty:.6g} @ {fill.price:.6g} avg={pos.avg_entry:.6g}")
            elif a.kind is ActionKind.MOVE_STOP:
                old = pos.current_stop
                if pos.tighten_stop(a.price):
                    self.db.log_event(ts, sym, "move_stop", {"from": old, "to": pos.current_stop})
        # keep the exchange-side stop in sync with final qty and stop
        self.broker.set_stop(sym, pos.direction, pos.current_qty, pos.current_stop)
        self.db.save_position(sym, pos)

    def _push_local_to_broker(self, sym: str, pos: Position) -> None:
        """Re-register a DB-restored position on the paper broker (never invent a stop-out)."""
        if not isinstance(self.broker, PaperBroker):
            return
        self.broker.signed[sym] = pos.sign * float(pos.current_qty)
        self.broker.avg[sym] = float(pos.avg_entry)
        self.broker.stops[sym] = float(pos.current_stop)
        self.broker.last[sym] = self.last_price.get(sym, float(pos.avg_entry))
        log.warning("%s: local position had no broker mirror — re-pushed (no fabricated exit)", sym)

    def _reconcile(self, sym: str) -> None:
        pos, bp = self.positions.get(sym), self.broker.get_position(sym)
        if pos and bp is None:
            fill = self.broker.pop_stop_fill(sym)
            # Only accept a real stop fill, or a disappearance after we had seen the
            # broker position this session. Cold-start desync must NOT invent -1R exits.
            if fill is None and sym not in self._seen_broker_pos:
                self._push_local_to_broker(sym, pos)
                self._seen_broker_pos.add(sym)
                return
            px = fill.price if fill else pos.current_stop
            fee = fill.fee if fill else px * pos.current_qty * self.cfg.fee_taker
            reason = ExitReason.TRAILING if pos.current_stop != pos.initial_stop else ExitReason.STOP
            # Stamp exit on the last closed candle — wall clock piled markers on chart edge
            exit_ts = self._bar_aligned_ts(
                sym,
                self.last_bar.get(sym) or pd.Timestamp.now(tz="UTC"),
            )
            self._finish(sym, pos, px, fee, exit_ts, reason)
        elif bp and not pos and sym not in self.blocked:
            self.blocked.add(sym)
            self.notify.send(f"UNKNOWN position on {sym}; bot will not trade it. Resolve manually.")
        elif bp and pos:
            self._seen_broker_pos.add(sym)
            if abs(bp.qty - pos.current_qty) > 0.01 * pos.current_qty:
                log.warning("%s qty mismatch local=%.6g broker=%.6g; syncing to broker", sym,
                            pos.current_qty, bp.qty)
                pos.current_qty = bp.qty

    def _finish(self, sym: str, pos: Position, px: float, fee: float, ts: pd.Timestamp,
                reason: ExitReason) -> None:
        # Never stamp exit before entry (chart / hold-time integrity)
        try:
            et = pd.Timestamp(pos.entry_time)
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            if ts < et:
                ts = et + self.cfg.bar_td
        except Exception:
            pass
        res, _ = pos.close_out(px, fee, ts, reason)
        self.positions.pop(sym, None)
        self.pend_exit.pop(sym, None)
        self.db.save_position(sym, None)
        self.db.log_trade(res)
        self.db.log_event(ts, sym, "exit", {
            "reason": reason.value, "pnl": res.pnl, "r": res.r_multiple, "price": px,
        })
        self.risk.register_trade_result(res.r_multiple, ts)
        # Longer cool-off after regime kills — stops open→kill→requeue loops
        cool_bars = int(self.cfg.cooldown_bars)
        if reason is ExitReason.REGIME_INVALID:
            cool_bars = max(cool_bars, 6)
        self.cooldown[sym] = ts + cool_bars * self.cfg.bar_td
        self._save_runtime()
        try:
            _live_feed(
                "close",
                f"CLOSE {sym.split('/')[0]} {reason.value} pnl={res.pnl:+.2f}",
                symbol=sym, reason=reason.value, pnl=res.pnl, r=res.r_multiple,
            )
        except Exception:
            pass
        self.notify.send(f"CLOSE {sym} {reason.value} pnl={res.pnl:+.2f} ({res.r_multiple:+.2f}R)")


# ============================================================
# 13. SELF-TESTS
# ============================================================


def selftest() -> bool:
    cfg = Config()
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    print("selftest")
    # 1) no lookahead: features at row k-1 identical with or without future data
    df = synthetic_ohlcv("TEST", 1800)
    k = 1500
    full, trunc = build_features(df, cfg), build_features(df.iloc[:k], cfg)
    cols = ["ema_htf", "atr", "adx", "vol_ratio", "don_high", "d_close", "d_ema"]
    same = np.allclose(full[cols].iloc[:k].to_numpy(), trunc[cols].to_numpy(), equal_nan=True)
    check("features have no lookahead (numeric)", same)
    check("regime has no lookahead", (full["regime"].iloc[:k].to_numpy() == trunc["regime"].to_numpy()).all())

    # 2) average-entry accounting
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    p = Position.open("X", Direction.LONG, 5, 100.0, 1.0, 98.0, t0, 0.0)
    p.add_lot(110.0, 0.5, t0, 0.0, 106.0, 10.0)
    check("avg entry after add", abs(p.avg_entry - (100 + 55) / 1.5) < 1e-9)
    check("P&L uses average entry", abs(p.unrealized(120.0) - (120 - 155 / 1.5) * 1.5) < 1e-9)

    # paper restart must NOT fabricate stop-outs from restored positions
    pb = PaperBroker(cfg)
    t_open = pd.Timestamp("2025-01-01", tz="UTC")
    restored = {
        "BTC/USDT:USDT": Position.open(
            "BTC/USDT:USDT", Direction.LONG, 5, 100.0, 1.0, 98.0, t_open, 0.02,
        ),
    }
    pb.rehydrate(restored, closed_pnl=-1.5)
    check("paper rehydrate restores signed qty", abs(pb.signed["BTC/USDT:USDT"] - 1.0) < 1e-12)
    check("paper rehydrate restores stop", abs(pb.stops["BTC/USDT:USDT"] - 98.0) < 1e-12)
    check(
        "paper rehydrate cash = start + closed − open fees",
        abs(pb.cash - (cfg.starting_equity - 1.5 - 0.02)) < 1e-9,
    )
    # price still above stop → on_price must NOT fill
    pb.on_price("BTC/USDT:USDT", 99.0)
    check("paper rehydrate: no false stop above stop", pb.pop_stop_fill("BTC/USDT:USDT") is None)
    pb.on_price("BTC/USDT:USDT", 97.5)
    fill = pb.pop_stop_fill("BTC/USDT:USDT")
    check("paper rehydrate: real stop still fires", fill is not None and fill.side == "sell")
    try:
        p.add_lot(111.0, 0.1, t0, 0.0, 90.0, 0.0)
        check("stop widening rejected", False)
    except RiskError:
        check("stop widening rejected", True)

    # 3) free-roll sizing: after the add, stopping out is still not a net loss
    q = Position.open("X", Direction.LONG, 5, 100.0, 1.0, 98.0, t0, 0.0)
    qty, dec = RollManager(cfg).size_add(q, 110.0, 106.0, 50.0, q.margin(), InstrumentSpec("X"))
    if qty:
        q.add_lot(110.0, qty, t0, 0.0, 106.0, 10.0)
    check("free-roll add allowed and non-losing at stop", dec is AddDecision.ALLOW and q.net_pnl_at(106.0) >= -1e-9)

    # 4) daily halt resets next UTC day; kill switch does not
    r = RiskEngine(Config(max_daily_loss_pct=0.05))
    r.update_equity(50.0, pd.Timestamp("2025-01-01 04:00", tz="UTC"))
    r.update_equity(46.0, pd.Timestamp("2025-01-01 08:00", tz="UTC"))
    halted = not r.trading_allowed(pd.Timestamp("2025-01-01 09:00", tz="UTC"))
    r.update_equity(46.0, pd.Timestamp("2025-01-02 00:00", tz="UTC"))
    check("daily halt trips then resets next day", halted and r.trading_allowed(pd.Timestamp("2025-01-02 01:00", tz="UTC")))

    # 5) R-multiple uses real initial risk, not 2% of equity
    t = Position.open("X", Direction.LONG, 5, 100.0, 2.0, 98.0, t0, 0.0)
    res, _ = t.close_out(104.0, 0.0, t0, ExitReason.STOP)
    check("R = pnl / (qty * stop distance)", abs(res.r_multiple - 2.0) < 1e-9)

    # 5b) time-stop fires when stagnant past N bars without +min_r
    ts_cfg = Config(enable_time_stop=True, time_stop_bars=3, time_stop_min_r=1.0,
                    enable_partial=False, enable_pyramiding=False)
    pos_ts = Position.open("X", Direction.LONG, 5, 100.0, 1.0, 98.0,
                           pd.Timestamp("2025-01-01 00:00", tz="UTC"), 0.0)
    # fake bar 4h later still flat
    class _B:
        Index = pd.Timestamp("2025-01-01 04:00", tz="UTC")
        close = 100.2
        atr = 1.0
        regime = RegimeState.TRENDING_UP.value
    acts = PositionManager(ts_cfg).decide(
        pos_ts, _B(), allowed=True, kill=False, equity=50.0, used_margin=0.0,
        spec=InstrumentSpec("X"),
    )
    check("time stop exits stagnant trade",
          any(a.kind is ActionKind.EXIT and a.reason is ExitReason.TIME_STOP for a in acts))

    # 6) feature columns for quality filters + momentum + 4H MTF
    demo = build_features(synthetic_ohlcv("BTC/USDT:USDT", 400), cfg)
    for col in (
        "rsi", "squeeze", "squeeze_ok", "bb_upper", "kc_upper", "btc_trend", "mom_atr",
        "h4_bias", "h4_adx", "ema_entry", "pullback_long", "pullback_short",
    ):
        check(f"feature {col} present", col in demo.columns)
    check("h4_bias has values", demo["h4_bias"].notna().any())

    # 7) end-to-end smoke on synthetic data (include BTC for regime stamp)
    raw = {s: synthetic_ohlcv(s, 2500) for s in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT")}
    data = prepare_data(raw, cfg)
    eth_bar = data["ETH/USDT:USDT"].bars[-1]
    check("btc_trend stamped onto alts", hasattr(eth_bar, "btc_trend"))
    res = BacktestEngine(data, cfg).run()
    finite = math.isfinite(res.final_equity) and res.final_equity >= 0
    check(f"backtest smoke run ({res.trades} trades, final {res.final_equity:.2f})", finite)
    check("equity curve is time-ordered", res.equity_curve.index.is_monotonic_increasing)
    mom_cfg = replace(cfg, strategy="momentum_15", timeframe="15m", require_btc_regime=False)
    mom_res = BacktestEngine(prepare_data(raw, mom_cfg), mom_cfg).run()
    check(
        f"momentum_15 smoke ({mom_res.trades} trades, final {mom_res.final_equity:.2f})",
        math.isfinite(mom_res.final_equity) and mom_res.final_equity >= 0,
    )
    check("make_strategy momentum", make_strategy(mom_cfg).name == "momentum_15")
    mtf_cfg = replace(cfg, strategy="trend_mtf", timeframe="15m", require_btc_regime=True)
    mtf_res = BacktestEngine(prepare_data(raw, mtf_cfg), mtf_cfg).run()
    check(
        f"trend_mtf smoke ({mtf_res.trades} trades, final {mtf_res.final_equity:.2f})",
        math.isfinite(mtf_res.final_equity) and mtf_res.final_equity >= 0,
    )
    check("make_strategy trend_mtf", make_strategy(mtf_cfg).name == "trend_mtf")
    rule_cfg = replace(cfg, strategy="momentum_rule", require_btc_regime=True)
    rule_res = BacktestEngine(prepare_data(raw, rule_cfg), rule_cfg).run()
    check(
        f"momentum_rule smoke ({rule_res.trades} trades, final {rule_res.final_equity:.2f})",
        math.isfinite(rule_res.final_equity) and rule_res.final_equity >= 0,
    )
    check("make_strategy momentum_rule", make_strategy(rule_cfg).name == "momentum_rule")
    # score_raw vs capped display
    from types import SimpleNamespace
    fake_ins = SimpleNamespace(expectancy_r=0.5, trades=80, total_return=1.0, max_drawdown=0.1,
                               win_rate=0.6, final_equity=100.0, halted="")
    fake_oos = SimpleNamespace(expectancy_r=0.4, trades=50, total_return=0.5, max_drawdown=0.15,
                               win_rate=0.55, final_equity=75.0, halted="")
    g_cap = _grade(fake_ins, fake_oos, fake_oos, wf={"ok": True}, wf_required=True,
                   fwd={"ok": False}, fwd_required=True)
    check("score_raw present", "score_raw" in g_cap)
    check("forward cap applies to display only",
          g_cap["score"] <= FWD_SCORE_CAP and g_cap["score_raw"] >= g_cap["score"])
    # net directional brake
    rsk = RiskEngine(replace(cfg, max_net_leverage=1.0, max_open_positions=8), silent=True)
    rsk.reset(100.0)
    spec = InstrumentSpec(symbol="X", contract_size=1.0, qty_step=0.01, min_qty=0.01)
    # Net long already at the 1.0x cap → no more longs; shorts still OK
    d_long = rsk.evaluate(Direction.LONG, 100.0, 98.0, equity=100.0, open_notional=100.0,
                          used_margin=20.0, open_positions=1, spec=spec,
                          ts=pd.Timestamp("2025-01-01", tz="UTC"), net_notional=100.0)
    d_short = rsk.evaluate(Direction.SHORT, 100.0, 102.0, equity=100.0, open_notional=100.0,
                           used_margin=20.0, open_positions=1, spec=spec,
                           ts=pd.Timestamp("2025-01-01", tz="UTC"), net_notional=100.0)
    check("net-leverage blocks adding with the book", not d_long.approved)
    check("net-leverage still allows opposite side", d_short.approved)
    check(
        "make_strategy breakout",
        make_strategy(replace(cfg, strategy="regime_breakout")).name == "regime_breakout_v7",
    )
    print("ALL PASSED" if ok else "SOME TESTS FAILED")
    return ok


# ============================================================
# 14. CLI
# ============================================================

# Core majors (stable sample) + hot movers (ZEC-style explosions). Hunt/paper share the set.
# Hot names can thin samples and spike DD — kept because dollar upside is the goal.
DEFAULT_SYMBOLS = [
    "BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT", "XRP/USDT:USDT",
    "DOGE/USDT:USDT", "ZEC/USDT:USDT", "SUI/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT",
]
HOT_SYMBOLS = [
    "ZEC/USDT:USDT", "SUI/USDT:USDT", "PEPE/USDT:USDT", "WIF/USDT:USDT",
    "SOL/USDT:USDT", "DOGE/USDT:USDT",
]
PAPER_SYMBOLS = list(DEFAULT_SYMBOLS)
HUNT_SYMBOLS = list(DEFAULT_SYMBOLS)
# Extra perps available in the dashboard; not default paper universe.
OPTIONAL_SYMBOLS = [
    "ADA/USDT:USDT", "LINK/USDT:USDT", "AVAX/USDT:USDT", "LTC/USDT:USDT",
    "DOT/USDT:USDT", "ATOM/USDT:USDT", "NEAR/USDT:USDT", "APT/USDT:USDT",
    "ARB/USDT:USDT", "OP/USDT:USDT", "INJ/USDT:USDT", "UNI/USDT:USDT", "AAVE/USDT:USDT",
    "TRX/USDT:USDT", "BCH/USDT:USDT", "ETC/USDT:USDT",
    "RENDER/USDT:USDT", "FET/USDT:USDT", "SEI/USDT:USDT", "TIA/USDT:USDT", "WLD/USDT:USDT",
    "POL/USDT:USDT",
]


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = Config.from_env()
    symbols = args.symbols or DEFAULT_SYMBOLS
    if args.synthetic:
        print("!! SYNTHETIC DATA - plumbing test only, results mean nothing !!")
        raw = {s: synthetic_ohlcv(s, args.bars) for s in symbols}
    else:
        raw = {s: fetch_historical(s, cfg.timeframe, args.bars, refresh=args.refresh) for s in symbols}
    data = prepare_data(raw, cfg)
    t0 = min(sd.bars[0].Index for sd in data.values())
    t1 = max(sd.bars[-1].Index for sd in data.values())
    split = t0 + (t1 - t0) * (1 - args.oos)
    print(f"data {t0:%Y-%m-%d} -> {t1:%Y-%m-%d} | in-sample < {split:%Y-%m-%d} <= out-of-sample")

    ins = run_grid(data, cfg, end=split)
    oos = run_grid(data, cfg, start=split)
    print("\n=== IN-SAMPLE (choose settings here) ===\n" + results_table(ins))
    print("\n=== OUT-OF-SAMPLE (judge them here, do not tune on it) ===\n" + results_table(oos))
    print("\n=== Decisions from in-sample ===")
    for line in verdicts(ins):
        print(" -", line)
    few = [r for r in oos if r.trades < 30]
    if few:
        print(f"\nWARNING: {len(few)}/{len(oos)} out-of-sample runs have <30 trades; "
              "differences between configs are noise at this sample size.")
    print("WARNING: 8 configs on one dataset invites overfitting; prefer configs that are "
          "good in BOTH halves, and confirm in paper trading.")


def _live_runner(live: bool, symbols: list) -> LiveRunner:
    cfg = Config.from_env()
    market = MarketData(cfg)
    broker: BaseBroker = MexcBroker(cfg) if live else PaperBroker(cfg)
    return LiveRunner(cfg, broker, market, symbols, live)


# ============================================================
# AUTOPILOT — paper forever + cautious self-improve (never live)
# ============================================================


_ROOT = Path(__file__).resolve().parent
_ENV_PATH = _ROOT / ".env"
_AUTOPILOT_STATE = _ROOT / ".moonshot_autopilot.json"
_LIVE_FEED = _ROOT / ".moonshot_live.jsonl"
_SAFE_ENV_KEYS = {
    "ADX_ENTER", "ADX_EXIT", "VOLUME_MULTIPLIER", "BREAKOUT_BUFFER_ATR",
    "BREAKOUT_LOOKBACK", "TRAIL_ATR", "TRAIL_ATR_LOSER", "USE_ASYMMETRIC_TRAIL",
    "COOLDOWN_BARS", "REQUIRE_RISING_ADX",
    "ENABLE_PYRAMIDING", "ENABLE_PARTIAL", "PARTIAL_AT_R", "PARTIAL_PCT",
    "LEVERAGE", "MAX_ACCOUNT_LEVERAGE", "MAX_NET_LEVERAGE", "MAX_OPEN_POSITIONS",
    "TIMEFRAME", "STRATEGY",
    "MOM_LOOKBACK", "MOM_MIN_ATR",
    "INITIAL_STOP_ATR", "FEE_TAKER", "SLIPPAGE_BPS",
    "H4_EMA_FAST", "H4_EMA_SLOW", "H4_ADX_MIN_SHORT",
    "EMA_ENTRY_PERIOD", "PULLBACK_LOOKBACK", "BTC_ADX_MIN", "BREADTH_MIN",
    "RISK_PCT", "INITIAL_MARGIN_PCT", "AGGRESSIVE_MARGIN_PCT",
    "MAX_DAILY_LOSS_PCT", "MAX_PEAK_DRAWDOWN_PCT",
    "STREAK_LOSSES", "STREAK_PAUSE_HOURS", "EXPECTANCY_FLOOR",
    "REQUIRE_RSI_VETO", "RSI_LONG_MAX", "RSI_SHORT_MIN", "RSI_PERIOD",
    "REQUIRE_SQUEEZE", "SQUEEZE_LOOKBACK", "BB_PERIOD", "BB_STD",
    "KC_PERIOD", "KC_ATR_MULT", "REQUIRE_BTC_REGIME",
    "ENABLE_TIME_STOP", "TIME_STOP_BARS", "TIME_STOP_MIN_R",
}

# Autopilot hunt targets — PRIMARY bar is 7-day return with leverage.
# +100%/week is the real goal (compounded leveraged edge), not a stretch label.
HUNT_TARGETS = {
    "train_exp": 0.10,
    "test_exp": 0.08,
    "train_n": 20,
    "test_n": 20,
    "max_dd": 0.55,
    "win_rate": 0.60,     # PRIMARY quality bar — aim for consistent winners, not coin-flip
    "test_return": 0.50,     # secondary: long OOS context (NOT the weekly claim)
    "soft_return_floor": -0.30,
    "week_return": 1.00,     # PRIMARY: +100% in last 7d on leveraged book
    # Consider/promote gate — must be reachable on current books (live ~8–30/week).
    # Score still prefers denser weeks via week_target_trades.
    "week_min_trades": 6,
    "week_target_trades": 40,
    "weekly_return_goal": 1.00,
    "week_stretch": 1.00,
    # Hard week floor before "targets met" (soft-only fireworks blocked)
    "week_hard_floor": 0.25,
    # Reject classic kill illusion: hard quit-while-ahead while soft bleeds
    "block_kill_illusion": True,
}

# Fast hunt knobs — more passes/hour beat giant slow batches for self-improve.
# Funnel: scout many ideas on BTC only → confirm survivors on multi-coin →
# leverage ladder only on books that cleared multi-coin.
# 3m is OUT of hunt (1m-resample cost).
#
# Calendar-normalized depth: same wall-clock for every TF; bars follow TF under
# MAX_RANGE_BARS. Research prices default to Binance spot (deeper than MEXC).
HUNT_SCOUT_DAYS = 90              # reserved scout-only slice
HUNT_CONFIRM_DAYS = 180           # search / multi-coin confirm window
HUNT_FINALIST_DAYS = 540          # promote recheck target (clamped by bar cap)
HUNT_WF_DAYS = 180                # walk-forward history before fit start
HUNT_SEARCH_SYMBOLS = 5           # multi-coin confirm book
HUNT_SCOUT_TOP = 16               # BTC survivors that get multi-coin soft/week
HUNT_EXPLORE_N = 48               # aggressive scout (cheap on 1 coin)
HUNT_BREAKOUT_N = 56
HUNT_SOFT_BUDGET = {"polish": 10, "explore": 14, "breakout": 16}
# After this many HOLD passes, loosen floor / widen confirm so overnight climb continues
HUNT_STUCK_HELD = 6
HUNT_CHAMPION_STALE_PP = 0.08     # stored peak − live reeval ≥ 8pp → floor uses live
HUNT_STRATEGIES = ("trend_mtf", "momentum_15", "combo_or")
HUNT_TF_ALLOW = ("5m", "15m")
HUNT_LEV_POLISH = (5, 7, 10)
HUNT_LEV_BREAKOUT = (5, 7, 10, 15, 25)
HUNT_LEV_FORCE = (5, 10, 15, 25)

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
                s, tf, t0, t1,
                warmup_bars=warm,
                refresh=bool(refresh),
                source=src,
            )
        except Exception as e:
            log.warning("hunt fetch %s %s %s failed: %s", s, tf, tier, e)
    log.info(
        "hunt data · source=%s tier=%s tf=%s days≈%d · %s→%s · %d/%d symbols",
        src, tier, tf, _hunt_calendar_days(tier, tf),
        t0.date(), t1.date(), len(raw), len(symbols),
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
    # Match WF_WARMUP_DAYS (60) — defined later with walk-forward constants.
    t0 = hs - pd.Timedelta(days=int(days) + 60)
    warm = range_warmup_bars(tf)
    raw: dict = {}
    for s in symbols:
        try:
            raw[s] = fetch_historical_range(
                s, tf, t0, hs,
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
    return int(estimate_range_bars(
        tf, t0, t1, warmup_bars=range_warmup_bars(tf),
    ))


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
    if not _AUTOPILOT_STATE.exists():
        return {}
    try:
        raw = _AUTOPILOT_STATE.read_text()
        # Tolerate a torn double-write (two JSON docs concatenated)
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            obj, _ = json.JSONDecoder().raw_decode(_AUTOPILOT_STATE.read_text())
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
    tmp = _AUTOPILOT_STATE.with_name(
        f".moonshot_autopilot.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    )
    try:
        tmp.write_text(payload)
        os.replace(str(tmp), str(_AUTOPILOT_STATE))
    finally:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass


def _live_feed(kind: str, message: str, **extra) -> None:
    """Append one clean line for the dashboard Live page."""
    row = {"ts": _utc_now().isoformat(), "kind": kind, "message": message, **extra}
    try:
        with _LIVE_FEED.open("a") as f:
            f.write(json.dumps(row, default=str) + "\n")
        # keep file from growing forever
        if _LIVE_FEED.stat().st_size > 2_000_000:
            lines = _LIVE_FEED.read_text().splitlines()[-500:]
            _LIVE_FEED.write_text("\n".join(lines) + "\n")
    except Exception:
        pass
    state = _load_autopilot_state()
    # Paper heartbeat stays under state["live"]. Hunt/search events must NOT
    # overwrite it (that blanked bars_seen/setups and made Live look idle/wrong).
    if kind in ("paper_tick", "paper_restart"):
        state["live"] = row
    elif kind in ("open", "close", "setup", "reject"):
        # Keep last paper snapshot; only refresh ts/message hints
        live = dict(state.get("live") or {})
        live["last_trade_event"] = row
        live["ts"] = row.get("ts") or live.get("ts")
        state["live"] = live
    else:
        state["hunt_event"] = row
    # paper_tick must NOT clear busy during an active hunt pass
    if kind in ("improve_start", "paper_restart"):
        state["busy"] = True
    elif kind in ("hold", "upgrade", "targets_met", "error", "autopilot_on"):
        state["busy"] = False
    _save_autopilot_state(state)


def _patch_env(updates: dict) -> None:
    """Update only allow-listed knobs in .env. Never touches live/API keys."""
    clean = {k: v for k, v in updates.items() if k in _SAFE_ENV_KEYS}
    if "LIVE_TRADING_ENABLED" in updates:
        raise ConfigError("autopilot refuses to change LIVE_TRADING_ENABLED")
    if not clean:
        return
    lines: list[str] = []
    if _ENV_PATH.exists():
        lines = _ENV_PATH.read_text().splitlines()
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
    _ENV_PATH.write_text("\n".join(out) + "\n")
    for key, val in clean.items():
        os.environ[key] = str(val)


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
        "ENABLE_TIME_STOP": "1" if cfg.enable_time_stop else "0",
        "TIME_STOP_BARS": str(int(cfg.time_stop_bars)),
        "TIME_STOP_MIN_R": str(cfg.time_stop_min_r),
    }


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
PAPER_RUNTIME_PATH = Path(".moonshot_paper_runtime.json")


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

# Primary money window the user cares about
WEEK_DAYS = 7


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
    """PRIMARY: hard + soft P&L over the last 7 calendar days."""
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
        conn = sqlite3.connect(str(_ROOT / "moonshot.db"))
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
    path = _ROOT / ".moonshot_rules.json"
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


def improve_once(symbols: list, bars: int = 5000, oos_frac: float = 0.3,
                 refresh: bool = False, generation: int = 0, hunt: bool = True) -> dict:
    """
    One hunt/improve pass. Keeps the global best-ever; aims at HUNT_TARGETS.
    Never enables live trading.
    Fit window excludes last WEEK_DAYS (holdout). Research promote may patch .env;
    elite / targets_met require forward paper proof.
    """
    try:
        from dotenv import load_dotenv as _ld
        _ld(_ENV_PATH, override=True)
    except ImportError:
        pass
    base = Config.from_env()
    mode = "HUNT" if hunt else "TUNE"
    state = _load_autopilot_state()
    held = _held_streak(state)
    search_mode = _hunt_search_mode(generation, held, hunt)
    full_symbols = list(symbols)
    # BTC must be present for the altcoin regime gate
    if base.require_btc_regime and _btc_symbol_in(symbols) is None:
        symbols = ["BTC/USDT:USDT"] + list(symbols)
        full_symbols = list(symbols)
    _mode_blurb = {
        "polish": "nudges around current best…",
        "explore": "wider explore to escape a local peak…",
        "breakout": "BREAKOUT search — large jumps across filters/exits…",
    }.get(search_mode, "searching…")
    _live_feed(
        "improve_start",
        f"{mode} pass #{generation} · {search_mode} — {_mode_blurb}",
        generation=generation, bars=bars, symbols=len(symbols), mode=mode.lower(),
        search_mode=search_mode, held_streak=held,
    )
    # Funnel hunt: confirm book (multi-coin) + scout book (BTC only).
    tf_s = str(base.timeframe)
    hunt_tier = "confirm" if hunt else "finalist"
    hunt_src = _hunt_price_source()
    hunt_days = _hunt_calendar_days(hunt_tier, tf_s)
    bars = int(_hunt_bars_for_tf(tf_s, full=not hunt))  # for status / legacy feeds
    n_confirm = int(HUNT_SEARCH_SYMBOLS)
    hunt_syms = list(symbols)
    btc = _btc_symbol_in(hunt_syms)
    rest = [s for s in hunt_syms if s != btc]
    core = ([btc] if btc else []) + rest
    if hunt and len(core) > n_confirm:
        core = core[:n_confirm]
    symbols = core  # multi-coin confirm + promote / Live week
    scout_sym = btc or (symbols[0] if symbols else None)
    scout_symbols = [scout_sym] if scout_sym else list(symbols[:1])
    log.info(
        "hunt funnel · scout=%s · confirm ~%dd (%d bars est) × %d symbols tf=%s source=%s",
        ",".join(s.split("/")[0] for s in scout_symbols),
        hunt_days, bars, len(symbols), tf_s, hunt_src,
    )
    raw = _hunt_fetch_raw(
        symbols, base.timeframe, hunt_tier, refresh=refresh, source=hunt_src,
    )
    if not raw:
        raise DataError(
            f"hunt could not load any OHLCV (source={hunt_src} tier={hunt_tier} tf={tf_s})"
        )
    feat_cache: dict = {}
    # Per-TF maps for confirm (multi-coin) and scout (BTC)
    raw_by_tf: dict = {str(base.timeframe): raw}
    scout_raw_by_tf: dict = {
        str(base.timeframe): {s: raw[s] for s in scout_symbols if s in raw},
    }

    def _raw_for_tf(tf: str) -> dict:
        key = str(tf)
        hit = raw_by_tf.get(key)
        if hit is not None:
            return hit
        # Same calendar tier for every TF so 5m vs 15m compare equal wall-clock.
        loaded = _hunt_fetch_raw(
            symbols, key, hunt_tier, refresh=False, source=hunt_src,
        )
        raw_by_tf[key] = loaded
        return loaded

    def _scout_raw_for_tf(tf: str) -> dict:
        key = str(tf)
        hit = scout_raw_by_tf.get(key)
        if hit is not None:
            return hit
        full = _raw_for_tf(key)
        slim = {s: full[s] for s in scout_symbols if s in full}
        scout_raw_by_tf[key] = slim
        return slim

    base_data = _prepare_cached(raw, base, feat_cache)
    t0 = min(sd.bars[0].Index for sd in base_data.values())
    t1 = max(sd.bars[-1].Index for sd in base_data.values())
    # F1: freeze last WEEK_DAYS — fit/IS/OOS never train on holdout week
    holdout_start = pd.Timestamp(t1) - pd.Timedelta(days=int(WEEK_DAYS))
    fit_end = holdout_start
    if fit_end <= t0 + pd.Timedelta(days=14):
        log.warning("holdout window too tight (t0=%s t1=%s) — widening fit", t0, t1)
        fit_end = t0 + (t1 - t0) * 0.85
        holdout_start = fit_end
    split = t0 + (fit_end - t0) * (1 - oos_frac)
    log.info(
        "holdout week %s → %s (fit ends %s); split %s",
        holdout_start.date(), pd.Timestamp(t1).date(), fit_end.date(), split.date(),
    )

    # Walk-forward on history the hunt never sees (before t0). Fail closed on grade.
    wf_raw: Optional[dict] = None
    wf_cache: dict = {}
    try:
        wf_raw = _hunt_fetch_wf_raw(
            symbols, base.timeframe, t0, refresh=False, source=hunt_src,
        )
        if not wf_raw:
            raise RuntimeError("no walk-forward history for any symbol")
        min_len = 800 if tf_s in ("3m", "5m") else 1500
        wf_raw = {
            s: df for s, df in wf_raw.items()
            if df is not None and len(df) > min_len
        }
        if not wf_raw:
            raise RuntimeError("walk-forward series too short after filter")
        need_days = 40 if tf_s in ("3m", "5m") else int(HUNT_WF_DAYS) // 2
        if min(df.index[0] for df in wf_raw.values()) >= t0 - pd.Timedelta(days=need_days):
            log.warning("walk-forward: not enough history before %s", t0)
            wf_raw = None
    except Exception as e:
        log.warning("walk-forward data unavailable: %s", e)
        wf_raw = None
    wf_enabled = wf_raw is not None
    if not wf_enabled:
        log.warning(
            "walk-forward DISABLED for %s this pass — research promote still needs other gates",
            tf_s,
        )

    def _wf_for(cfg: Config) -> Optional[dict]:
        if not wf_enabled:
            return None
        try:
            return _wf_eval(_prepare_cached(wf_raw, cfg, wf_cache), cfg, t0)
        except Exception as e:
            log.warning("walk-forward eval failed: %s", e)
            return {"ok": False, "why": f"error: {e}", "folds": [], "trades": 0,
                    "exp": 0.0, "green": "0/0", "worst": 0.0, "span": ""}

    # Forward (paper) record: clock starts when this strategy first ran in paper
    if not state.get("forward_since"):
        state["forward_since"] = _utc_now().isoformat()
        _save_autopilot_state(state)
    fwd = _forward_record(list(symbols), str(state["forward_since"]))

    base_i, base_o = _eval_cfg(base_data, base, split, end=fit_end)
    base_soft = _eval_soft_oos(base_data, base, split, end=fit_end)
    base_wf = _wf_for(base)
    base_recent = _recent_eval(base_data, base, end=t1)
    base_week = _week_eval(base_data, base, end=t1)  # holdout last 7d
    base_grade = _grade(
        base_i, base_o, base_soft, base_wf, wf_required=True,
        fwd=fwd, fwd_required=True, week=base_week,
    )
    base_grade["recent"] = base_recent
    base_grade["week"] = base_week

    elite = state.get("elite")  # best-ever across all generations (forward-confirmed)
    # Champion = best holdout-week book ever kept (independent of forward proof).
    # Live / research promote must not fall below this week. Rebuild from history
    # if a reseed/wipe left a weaker floor (that is how −26% got applied).
    champion = _champion_from_history(
        list(state.get("history") or []),
        state.get("champion") if isinstance(state.get("champion"), dict) else None,
    )
    if champion:
        state["champion"] = champion

    best_cfg = base
    best_i, best_o, best_soft = base_i, base_o, base_soft
    best_wf = base_wf
    best_recent = dict(base_recent)
    best_week = dict(base_week)
    best_grade = dict(base_grade)
    base_wf_ok = (not wf_enabled) or bool(base_wf and base_wf.get("ok"))
    promote_reason = "baseline"

    def _consider(cand_cfg: Config, ci_, co_, cs_, label_: str = "") -> bool:
        """Promote cand in-memory if it beats incumbent on fit rank or holdout week."""
        nonlocal best_cfg, best_i, best_o, best_soft, best_wf, best_grade
        nonlocal best_recent, best_week, promote_reason
        # ≥10x must arrive with soft OOS already evaluated
        if int(cand_cfg.leverage) >= 10 and cs_ is None:
            return False
        best_ok = (not wf_enabled) or bool(best_wf and best_wf.get("ok"))
        best_recent_ok = bool(best_recent.get("ok"))
        cwf = _wf_for(cand_cfg)
        cand_ok = (not wf_enabled) or bool(cwf and cwf.get("ok"))
        if wf_enabled and not cand_ok:
            return False
        # Always score week/recent on the multi-coin confirm book (correct TF).
        c_raw = _raw_for_tf(str(cand_cfg.timeframe)) or raw
        cdata = _prepare_cached(c_raw, cand_cfg, feat_cache)
        crecent = _recent_eval(cdata, cand_cfg, end=t1)
        cweek = _week_eval(cdata, cand_cfg, end=t1)  # holdout only (last 7d)
        week_min = int(HUNT_TARGETS.get("week_min_trades", 4))
        if int(cweek.get("trades") or 0) < week_min and int(cweek.get("soft_trades") or 0) < week_min:
            return False
        # Never replace a healthy recent with a worse one
        if not crecent.get("ok") and best_recent_ok:
            return False
        # Relative climb when both fail the 90d floor
        recent_relative = (
            not best_recent_ok
            and float(crecent.get("exp") or -99) >= float(best_recent.get("exp") or -99)
            and float(crecent.get("ret") or -99) >= float(best_recent.get("ret") or -99)
        )
        week_tgt = float(HUNT_TARGETS.get("week_return", 1.0))
        min_delta = _week_delta_needed(float(best_week.get("ret") or 0), week_tgt)
        week_active = (
            int(cweek.get("trades") or 0) >= week_min
            or int(cweek.get("soft_trades") or 0) >= week_min
        )
        # Any better holdout week while active — do NOT require week.ok (that was
        # a silent veto while every book is underwater on the same regime).
        week_edge = (
            week_active
            and float(cweek.get("ret") or 0)
            > float(best_week.get("ret") or 0) + min_delta
        )
        # Less-bad OOS / soft when incumbent is losing — lets hunt climb before week flips
        losing_book = float(best_o.total_return) <= 0 or float(best_o.expectancy_r) <= 0
        oos_edge = (
            losing_book
            and float(co_.total_return) > float(best_o.total_return) + 0.005
            and float(co_.expectancy_r) >= float(best_o.expectancy_r) - 0.05
        )
        soft_edge = (
            losing_book
            and cs_ is not None and best_soft is not None
            and float(cs_.total_return) > float(best_soft.total_return) + 0.005
        )
        recent_edge = (
            float(crecent.get("ret") or 0) > float(best_recent.get("ret") or 0) + 0.01
            and float(crecent.get("exp") or 0) >= float(best_recent.get("exp") or 0) - 0.02
        )
        cand_beats = _beats(
            ci_, co_, best_i, best_o, cs_, best_soft, cweek, best_week,
        )
        rescue = (
            (wf_enabled and not best_ok and cand_ok)
            or (not best_recent_ok and bool(crecent.get("ok")))
            or recent_edge or recent_relative or week_edge
            or oos_edge or soft_edge
        )
        if not cand_beats and not rescue:
            return False
        # PRIMARY floor: never replace in-memory best with a worse holdout week.
        # OOS/soft "rescues" were demoting +16% champions down to -21% on Live.
        cand_week_ret = float(cweek.get("ret") or 0)
        best_week_ret = float(best_week.get("ret") or 0)
        if week_active and cand_week_ret < best_week_ret - 0.001:
            return False
        reason = label_ or (
            "week_ladder" if week_edge else
            "recent_rescue" if (recent_edge or recent_relative) else
            "rank_beat" if cand_beats else "rescue"
        )
        best_cfg, best_i, best_o, best_soft, best_wf = cand_cfg, ci_, co_, cs_, cwf
        best_recent = crecent
        best_week = cweek
        best_grade = _grade(
            ci_, co_, cs_, cwf, wf_required=True, fwd=fwd, fwd_required=True, week=cweek,
        )
        best_grade["recent"] = crecent
        best_grade["week"] = cweek
        promote_reason = reason
        return True

    def _knobs_to_cfg(kn: dict) -> Config:
        return _freeze_risk(base, replace(
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
            aggressive_margin_pct=float(kn.get("aggressive_margin_pct", base.aggressive_margin_pct)),
            max_daily_loss_pct=float(kn.get("max_daily_loss_pct", base.max_daily_loss_pct)),
            max_peak_drawdown_pct=float(kn.get("max_peak_drawdown_pct", base.max_peak_drawdown_pct)),
            mom_lookback=int(kn.get("mom_lookback", base.mom_lookback)),
            mom_min_atr=float(kn.get("mom_min_atr", base.mom_min_atr)),
            leverage=int(kn.get("leverage", base.leverage)),
            max_account_leverage=float(kn.get("max_account_leverage", base.max_account_leverage)),
            max_net_leverage=float(kn.get("max_net_leverage", base.max_net_leverage)),
            max_open_positions=int(kn.get("max_open_positions", base.max_open_positions)),
            h4_ema_fast=int(kn.get("h4_ema_fast", base.h4_ema_fast)),
            h4_ema_slow=int(kn.get("h4_ema_slow", base.h4_ema_slow)),
            ema_entry_period=int(kn.get("ema_entry_period", base.ema_entry_period)),
            pullback_lookback=int(kn.get("pullback_lookback", base.pullback_lookback)),
            btc_adx_min=float(kn.get("btc_adx_min", base.btc_adx_min)),
            breadth_min=float(kn.get("breadth_min", base.breadth_min)),
        ))

    # Seed from elite — allow cross-family in breakout mode
    if elite and elite.get("knobs"):
        try:
            kn = elite["knobs"]
            elite_strat = str(kn.get("strategy", "") or "").strip().lower()
            if (
                elite_strat
                and elite_strat != str(base.strategy).strip().lower()
                and search_mode != "breakout"
            ):
                raise ValueError(f"elite strategy {elite_strat} != {base.strategy} (wait for breakout)")
            elite_cfg = _knobs_to_cfg(kn)
            e_raw = _raw_for_tf(elite_cfg.timeframe)
            ed = _prepare_cached(e_raw, elite_cfg, feat_cache)
            ei, eo = _eval_cfg(ed, elite_cfg, split, end=fit_end)
            es = _eval_soft_oos(ed, elite_cfg, split, end=fit_end)
            if _is_climb_ok(ei, eo, es, base_i, base_o) and _consider(elite_cfg, ei, eo, es, "elite"):
                log.info(
                    "restored elite seed score=%d win=%.0f%% test_exp=%+.2f soft=%+.0f%% n=%d",
                    best_grade["score"], eo.win_rate * 100, eo.expectancy_r,
                    es.total_return * 100, eo.trades,
                )
        except Exception as e:
            log.warning("elite re-eval failed: %s", e)

    # Seed champion so hunt cannot forget a better week book
    if champion and champion.get("knobs"):
        try:
            ch_week_ret = float(champion.get("week_ret") or 0)
            if ch_week_ret > float(best_week.get("ret") or 0) + 0.001:
                ch_cfg = _knobs_to_cfg(champion["knobs"])
                ch_raw = _raw_for_tf(ch_cfg.timeframe)
                ch_data = _prepare_cached(ch_raw, ch_cfg, feat_cache)
                chi, cho = _eval_cfg(ch_data, ch_cfg, split, end=fit_end)
                chs = _eval_soft_oos(ch_data, ch_cfg, split, end=fit_end)
                if _is_climb_ok(chi, cho, chs, base_i, base_o) and _consider(
                    ch_cfg, chi, cho, chs, "champion"
                ):
                    log.info(
                        "restored champion week=%+.0f%% score=%s",
                        ch_week_ret * 100,
                        champion.get("score"),
                    )
        except Exception as e:
            log.warning("champion restore failed: %s", e)

    # Re-apply a climb that was found earlier but blocked by the settle timer
    pending = state.get("pending_promote") or {}
    if pending.get("knobs"):
        try:
            pend_week = float(pending.get("week_ret") or 0)
            champ_floor = float((champion or {}).get("week_ret") or base_week.get("ret") or -999)
            # Never restore a pending book worse than champion / deployed week
            if pend_week + 0.001 < champ_floor:
                log.info(
                    "dropping stale pending week=%+.0f%% < champion/base %+0.f%%",
                    pend_week * 100, champ_floor * 100,
                )
                state.pop("pending_promote", None)
                _save_autopilot_state(state)
            else:
                p_cfg = _knobs_to_cfg(pending["knobs"])
                p_raw = _raw_for_tf(p_cfg.timeframe)
                pd_ = _prepare_cached(p_raw, p_cfg, feat_cache)
                pi, po = _eval_cfg(pd_, p_cfg, split, end=fit_end)
                ps = _eval_soft_oos(pd_, p_cfg, split, end=fit_end)
                if _is_climb_ok(pi, po, ps, base_i, base_o) and _consider(
                    p_cfg, pi, po, ps, "pending"
                ):
                    log.info(
                        "restored pending climb week=%+.0f%% score=%s",
                        pend_week * 100,
                        pending.get("score"),
                    )
        except Exception as e:
            log.warning("pending promote restore failed: %s", e)

    polish_base = best_cfg
    tested = 0
    soft_runs = 0
    stuck = held >= 2 or search_mode in ("explore", "breakout")
    soft_budget = int(HUNT_SOFT_BUDGET.get(search_mode, 6))
    scout_top = int(HUNT_SCOUT_TOP)
    if held >= int(HUNT_STUCK_HELD):
        soft_budget = max(soft_budget, 20)
        scout_top = max(scout_top, 24)
    hard_pool: list = []
    thin_pool: list = []
    cand_list = _candidate_cfgs(
        polish_base, generation=generation, hunt=hunt, search_mode=search_mode,
    )
    cand_list = list(cand_list) + _discover_seed_cfgs(polish_base, generation, held)
    # Density seeds every pass — ADX28/squeeze books starve week samples
    denser = [
        replace(
            polish_base, strategy=str(st), leverage=int(lev), timeframe="15m",
            adx_enter=float(adx), adx_exit=float(max(10.0, adx - 5)),
            require_squeeze=False, require_rising_adx=False, cooldown_bars=int(cd),
            volume_multiplier=1.1, mom_lookback=8, mom_min_atr=0.35,
            max_account_leverage=float(max(lev, float(polish_base.max_account_leverage))),
            max_net_leverage=float(max(lev, float(polish_base.max_net_leverage))),
        )
        for st in HUNT_STRATEGIES
        for lev in (5, 7, 10)
        for adx in (14.0, 16.0, 18.0)
        for cd in (0, 1, 2)
    ]
    cand_list = _dedupe_cfgs(polish_base, list(cand_list) + denser)

    # ---- Stage 1: scout hard IS/OOS + week on BTC only (rank by week first) ----
    for cand in cand_list:
        tested += 1
        try:
            use_raw = _scout_raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            ci, co = _eval_cfg(data, cand, split, end=fit_end)
            cweek_s = _week_eval(data, cand, end=t1)
        except Exception as e:
            log.warning("scout candidate failed: %s", e)
            continue
        # Sort key: holdout week first (what we climb on), then fit rank
        row = (
            float(cweek_s.get("ret") or -999),
            int(cweek_s.get("trades") or 0),
            _rank_key(ci, co, None),
            cand, ci, co,
        )
        if _is_screenable(ci, co):
            hard_pool.append(row)
        else:
            thin_pool.append(row)
    if not hard_pool and thin_pool:
        log.warning("scout: no screenable books — promoting thin pool to confirm")
        hard_pool = thin_pool
    hard_pool.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    shortlist = hard_pool[: max(scout_top, soft_budget)]
    top_w = float(shortlist[0][0]) if shortlist else float("nan")
    log.info(
        "funnel scout: tried %d on %s · %d screenable · %d → confirm (best scout week %+0.f%%)",
        tested,
        scout_symbols[0].split("/")[0] if scout_symbols else "?",
        len(hard_pool),
        len(shortlist),
        top_w * 100 if top_w == top_w else 0,
    )

    # ---- Stage 2: multi-coin soft + week; only survivors may enter _consider ----
    leverage_ladder: list = []
    confirm_hits = 0
    confirm_rejects = {"climb": 0, "consider": 0, "err": 0}
    seen_soft = set()
    for _wr, _wn, _rk, cand, _sci, _sco in shortlist:
        if soft_runs >= soft_budget:
            break
        ck = id(cand)
        if ck in seen_soft:
            continue
        seen_soft.add(ck)
        try:
            use_raw = _raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            ci, co = _eval_cfg(data, cand, split, end=fit_end)
            cs = _eval_soft_oos(data, cand, split, end=fit_end)
            soft_runs += 1
        except Exception as e:
            log.warning("confirm soft failed: %s", e)
            confirm_rejects["err"] += 1
            continue
        if not _is_climb_ok(ci, co, cs, base_i, base_o):
            confirm_rejects["climb"] += 1
            continue
        if _consider(cand, ci, co, cs, "confirm"):
            confirm_hits += 1
        else:
            confirm_rejects["consider"] += 1

    log.info(
        "funnel confirm: soft-checked %d/%d · climbs accepted=%d · rejects %s",
        soft_runs, soft_budget, confirm_hits, confirm_rejects,
    )

    # ---- Stage 3: leverage bake-off on multi-coin (always try _consider) ----
    ladder_src = best_cfg if (confirm_hits > 0 or best_cfg is not base) else base
    ladder_rows: list[dict] = []
    run_ladder_consider = True
    for cand in _leverage_ladder_cfgs(ladder_src):
        try:
            cand = _freeze_risk(base, cand)
            use_raw = _raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            ci, co = _eval_cfg(data, cand, split, end=fit_end)
            cweek = _week_eval(data, cand, end=t1)
            cs = None
            if run_ladder_consider and _is_screenable(ci, co):
                cs = _eval_soft_oos(data, cand, split, end=fit_end)
                if _is_climb_ok(ci, co, cs, base_i, base_o):
                    _consider(cand, ci, co, cs, "lev_ladder")
            grade_l = _grade(
                ci, co, cs, None, wf_required=False, fwd=fwd, fwd_required=False, week=cweek,
            )
            ladder_rows.append({
                "leverage": int(cand.leverage),
                "timeframe": str(cand.timeframe),
                "strategy": str(cand.strategy),
                "week_ret": float(cweek.get("ret") or 0),
                "week_trades": int(cweek.get("trades") or 0),
                "soft_ret": float(cs.total_return) if cs is not None else None,
                "test_ret": float(co.total_return),
                "test_n": int(co.trades),
                "win": float(co.win_rate),
                "score": int(grade_l.get("score") or 0),
                "expectancy": float(co.expectancy_r),
            })
        except Exception as e:
            log.warning("leverage ladder %sx failed: %s", getattr(cand, "leverage", "?"), e)
    by_lev: dict[int, dict] = {}
    for row in ladder_rows:
        lev = int(row["leverage"])
        prev = by_lev.get(lev)
        if prev is None or float(row["week_ret"]) > float(prev["week_ret"]):
            by_lev[lev] = row
    leverage_ladder = [by_lev[k] for k in sorted(by_lev)]
    if leverage_ladder:
        log.info(
            "leverage ladder%s: %s",
            "" if run_ladder_consider else " (chart)",
            " · ".join(
                f"{r['leverage']}x@{r['timeframe']} week={r['week_ret']:+.0%} "
                f"n={r['week_trades']} score={r['score']}"
                for r in leverage_ladder
            ),
        )

    # F4 finalist: re-score holdout week on FULL paper symbol book before research promote
    best_wf_ok = (not wf_enabled) or bool(best_wf and best_wf.get("ok"))
    week_min = int(HUNT_TARGETS.get("week_min_trades", 4))
    best_week_active = (
        int(best_week.get("trades") or 0) >= week_min
        or int(best_week.get("soft_trades") or 0) >= week_min
    )
    # Research promote only on a *material* step — not every ±1% thrash that
    # restarts paper and floods Live activity with setup/upgrade spam.
    material_ok, material_why = _material_research_edge(
        best_cfg=best_cfg, base_cfg=base,
        best_grade=best_grade, base_grade=base_grade,
        best_week=best_week, base_week=base_week,
        best_o=best_o, base_o=base_o,
        best_soft=best_soft, base_soft=base_soft,
    )
    promote_cooled = True
    last_prom = state.get("research_promote_at")
    if last_prom:
        try:
            age_s = (_utc_now() - pd.Timestamp(last_prom)).total_seconds()
            promote_cooled = age_s >= float(RESEARCH_PROMOTE_MIN_SEC)
            if not promote_cooled:
                material_why = f"promote cooldown {int(RESEARCH_PROMOTE_MIN_SEC - age_s)}s left"
        except Exception:
            promote_cooled = True
    # Never leave a big week win on the table for a short settle timer
    week_delta = float(best_week.get("ret") or 0) - float(base_week.get("ret") or 0)
    if (not promote_cooled) and week_delta >= 0.05:
        promote_cooled = True
        material_why = "big_week_bypass"
        log.info("bypassing promote lock — week edge %+0.fpp", week_delta * 100)

    # Champion floor: use live re-eval when the stored peak is stale for this
    # holdout window (ghost +16% was freezing overnight climbs while live is -3%).
    champ_stored = float((champion or {}).get("week_ret") or -999)
    champ_live = None
    if champion and champion.get("knobs"):
        try:
            ch_cfg = _knobs_to_cfg(champion["knobs"])
            ch_raw = _raw_for_tf(str(ch_cfg.timeframe)) or raw
            ch_data = _prepare_cached(ch_raw, ch_cfg, feat_cache)
            champ_live = float(_week_eval(ch_data, ch_cfg, end=t1).get("ret") or 0)
        except Exception as e:
            log.warning("champion live reeval failed: %s", e)
    base_week_ret = float(base_week.get("ret") or 0)
    if (
        champ_live is not None
        and champ_stored > -900
        and (champ_stored - champ_live) >= float(HUNT_CHAMPION_STALE_PP)
    ):
        champ_week_floor = max(champ_live, base_week_ret)
        log.info(
            "champion stale · stored %+0.f%% → live %+0.f%% · promote floor %+0.f%% (peak kept)",
            champ_stored * 100, champ_live * 100, champ_week_floor * 100,
        )
        if isinstance(champion, dict):
            champion = dict(champion)
            champion["week_ret_peak"] = max(
                float(champion.get("week_ret_peak") or champ_stored), champ_stored,
            )
            champion["week_ret_live"] = champ_live
            state["champion"] = champion
    else:
        champ_week_floor = champ_stored if champ_stored > -900 else base_week_ret
    # Stuck overnight: never require better than beating the deployed book
    if held >= int(HUNT_STUCK_HELD):
        champ_week_floor = min(champ_week_floor, base_week_ret + 0.001)
        log.info(
            "stuck ×%d — promote floor eased to deployed week %+0.f%%",
            held, champ_week_floor * 100,
        )

    best_week_ret_now = float(best_week.get("ret") or 0)
    # Never research-promote below the (possibly stale-adjusted) champion floor
    if best_cfg is not base and best_week_ret_now + 0.001 < champ_week_floor:
        material_ok = False
        material_why = (
            f"below champion week {champ_week_floor:+.0%} "
            f"(cand {best_week_ret_now:+.0%})"
        )

    research_improved = (
        best_cfg is not base
        and _is_climb_ok(best_i, best_o, best_soft, base_i, base_o)
        and (best_wf_ok or not wf_enabled)
        and not best_grade.get("kill_illusion")
        and best_week_active
        and material_ok
        and promote_cooled
        and best_week_ret_now + 0.001 >= champ_week_floor
    )
    if best_cfg is not base and not research_improved:
        log.info(
            "climb found but no research promote (%s) · week %+0.f%%→%+0.f%% score %s→%s",
            material_why,
            float(base_week.get("ret") or 0) * 100,
            float(best_week.get("ret") or 0) * 100,
            base_grade.get("score_raw", base_grade.get("score")),
            best_grade.get("score_raw", best_grade.get("score")),
        )
        # Remember only real week upgrades — never park a worse book as pending
        if (
            best_cfg is not base
            and best_week_active
            and best_week_ret_now >= champ_week_floor - 0.001
            and best_week_ret_now >= float(base_week.get("ret") or 0) + RESEARCH_MIN_WEEK_EDGE
        ):
            state["pending_promote"] = {
                "ts": _utc_now().isoformat(),
                "week_ret": best_week_ret_now,
                "score": int(best_grade.get("score_raw", best_grade.get("score") or 0)),
                "why": material_why,
                "knobs": _cfg_to_knobs(best_cfg),
            }
            _save_autopilot_state(state)
    # Finalist honesty on full book (max calendar under bar cap)
    if research_improved and len(full_symbols) > len(symbols):
        try:
            f_raw = _hunt_fetch_raw(
                full_symbols, best_cfg.timeframe, "finalist",
                refresh=False, source=hunt_src,
            )
            if not f_raw:
                raise RuntimeError("finalist fetch empty")
            f_data = prepare_data(f_raw, best_cfg)
            f_t1 = max(sd.bars[-1].Index for sd in f_data.values())
            f_week = _week_eval(f_data, best_cfg, end=f_t1)
            f_soft = _eval_soft_oos(
                f_data, best_cfg,
                min(sd.bars[0].Index for sd in f_data.values())
                + (f_t1 - min(sd.bars[0].Index for sd in f_data.values())) * (1 - oos_frac),
                end=pd.Timestamp(f_t1) - pd.Timedelta(days=int(WEEK_DAYS)),
            )
            search_week_ret = float(best_week.get("ret") or 0)
            f_week_ret = float(f_week.get("ret") or 0)
            # Never throw away a big search-week win just because the wider
            # symbol book is quieter — keep promoting the search climb.
            big_search_week = (
                search_week_ret >= float(base_week.get("ret") or 0) + 0.05
            )
            if (
                int(f_week.get("trades") or 0) < week_min
                and int(f_week.get("soft_trades") or 0) < week_min
                and not big_search_week
            ):
                log.warning("finalist silent on full book — skip research promote")
                research_improved = False
            else:
                # Soft floor: only veto when baseline is already healthy; while
                # climbing out of a hole, full-book soft is informational.
                f_floor = float(HUNT_TARGETS.get("soft_return_floor", -0.30))
                base_healthy = (
                    base_o.expectancy_r > 0 and base_o.total_return > 0
                )
                if base_healthy and f_soft.total_return < f_floor and not big_search_week:
                    log.warning("finalist soft floor fail on full book — skip research promote")
                    research_improved = False
                else:
                    # Prefer the better week number for Live (search vs full).
                    if f_week_ret >= search_week_ret or not big_search_week:
                        best_week = f_week
                        best_grade["week"] = f_week
                        best_grade["week_return"] = f_week.get("ret")
                    log.info(
                        "finalist full-book holdout week %+.0f%% n=%d soft=%+.0f%%",
                        float(f_week.get("ret") or 0) * 100,
                        int(f_week.get("trades") or 0),
                        f_soft.total_return * 100,
                    )
        except Exception as e:
            log.warning("finalist full-book recheck failed: %s — keeping search promote", e)

    improved = bool(research_improved)  # research tier → .env
    promote_tier = "research" if improved else "hold"
    # F3: confirmed / elite requires forward paper proof
    fwd_ok = bool(fwd and fwd.get("ok"))
    confirmed = bool(improved and fwd_ok and best_grade.get("targets_met"))

    knobs = _cfg_to_knobs(best_cfg)
    fail_checks = [
        k for k, v in (best_grade.get("checks") or {}).items() if not v
    ]
    decision = {
        "ts": _utc_now().isoformat(),
        "improved": improved,
        "promote_tier": promote_tier,
        "promote_reason": promote_reason,
        "material_why": material_why,
        "promote_cooled": bool(promote_cooled),
        "fail_checks": fail_checks,
        "tested": tested,
        "soft_runs": soft_runs,
        "search_mode": search_mode,
        "held_streak_before": held,
        "generation": generation,
        "mode": mode.lower(),
        "range": f"{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}",
        "holdout": f"{holdout_start:%Y-%m-%d} -> {t1:%Y-%m-%d}",
        "hunt_source": hunt_src,
        "hunt_tier": hunt_tier,
        "hunt_days": int(hunt_days),
        "hunt_bars_est": int(bars),
        "leverage_ladder": leverage_ladder,
        "baseline": {
            "train_exp": round(base_i.expectancy_r, 3),
            "test_exp": round(base_o.expectancy_r, 3),
            "train_n": base_i.trades,
            "test_n": base_o.trades,
            "test_return": round(base_o.total_return, 3),
            "score": base_grade["score"],
            "week_return": base_week.get("ret"),
            "leverage": int(base.leverage),
            "timeframe": str(base.timeframe),
        },
        "best": {
            "train_exp": round(best_i.expectancy_r, 3),
            "test_exp": round(best_o.expectancy_r, 3),
            "train_n": best_i.trades,
            "test_n": best_o.trades,
            "max_dd": round(best_o.max_drawdown, 3),
            "test_return": round(best_o.total_return, 3),
            "oos_final": round(best_o.final_equity, 2),
            "soft_return": best_grade.get("soft_return"),
            "soft_final": best_grade.get("soft_final"),
            "kill_illusion": best_grade.get("kill_illusion"),
            "score": best_grade["score"],
            "score_raw": best_grade.get("score_raw"),
            "recent": best_recent,
            "week": best_week,
            "week_return": best_week.get("ret"),
            "week_final": best_week.get("final"),
            "week_span": best_week.get("span"),
            "targets_met": best_grade["targets_met"],
            "leverage": int(best_cfg.leverage),
            "timeframe": str(best_cfg.timeframe),
            **knobs,
        },
        "grade": best_grade,
        "forward": fwd,
        "message": "",
    }

    def _score_msg(g: dict) -> str:
        raw = int(g.get("score_raw", g.get("score", 0)))
        disp = int(g.get("score", 0))
        return f"{disp}/100" if raw == disp else f"{disp}/100 (research {raw})"

    best_recent_ok = bool(best_recent.get("ok"))
    week_msg = (
        f"holdout7d ${float(best_week.get('final') or 0):.0f} "
        f"({float(best_week.get('ret') or 0):+.0%}, "
        f"soft {float(best_week.get('soft_ret') or 0):+.0%}, "
        f"n={int(best_week.get('trades') or 0)}, {best_week.get('span', '')})"
    )
    recent_msg = (
        f"90d {best_recent.get('exp', 0):+.2f}R n={best_recent.get('trades', 0)} "
        f"[{'OK' if best_recent_ok else best_recent.get('why', 'FAIL')}]"
    )
    week_tgt = float(HUNT_TARGETS.get("week_return", 1.0))
    week_gap = max(0.0, week_tgt - float(best_week.get("ret") or 0))
    if improved:
        _patch_env(_cfg_to_env_updates(best_cfg))
        # Do NOT reset forward_since on research promote — that made FWD_MIN_TRADES
        # unreachable across climbs. Only elite/confirmed resets the clock below.
        state["research_promote_at"] = decision["ts"]
        state.pop("pending_promote", None)
        decision["message"] = (
            f"Research promote ({promote_reason}) · {best_cfg.strategy} "
            f"{best_cfg.leverage}x@{best_cfg.timeframe} · {week_msg} · "
            f"score {_score_msg(base_grade)}→{_score_msg(best_grade)} · "
            f"long-OOS ${base_o.final_equity:.0f}→${best_o.final_equity:.0f} "
            f"({base_o.total_return:+.0%}→{best_o.total_return:+.0%}) · "
            f"win {base_o.win_rate:.0%}→{best_o.win_rate:.0%} ({best_o.trades} trades). "
            f"Climbing — goal +{week_tgt:.0%} holdout week "
            f"(gap {week_gap:.0%} still open). "
            f"[{_wf_msg(best_wf)}; {recent_msg}] Paper trial — real money still off."
        )
        log.info("AUTOPILOT RESEARCH UPGRADE: %s", decision["message"])
        _live_feed(
            "upgrade", decision["message"], generation=generation,
            score=best_grade["score"], score_raw=best_grade.get("score_raw"),
            targets_met=best_grade["targets_met"], promote_tier=promote_tier,
            promote_reason=promote_reason,
            week_return=best_week.get("ret"), week_final=best_week.get("final"),
            leverage=int(best_cfg.leverage), timeframe=str(best_cfg.timeframe),
        )
    else:
        hold_why = (
            f"soft-checked {soft_runs}/{soft_budget}; "
            f"holdout week {float(best_week.get('ret') or 0):+.0%} "
            f"(any better week/OOS/soft promotes; gap to +{week_tgt:.0%} is "
            f"{week_gap:.0%}); {recent_msg}"
        )
        decision["message"] = (
            f"Pass #{generation} ({search_mode}): HOLD · {week_msg} · "
            f"score {_score_msg(best_grade)} · "
            f"{best_cfg.strategy} {best_cfg.leverage}x@{best_cfg.timeframe} · "
            f"long-OOS ${best_o.final_equity:.0f} / {best_o.total_return:+.0%} · "
            f"win {best_o.win_rate:.0%}, n={best_o.trades}; "
            f"tried {tested} · {hold_why}. "
            + ("TARGETS MET — waiting forward proof." if best_grade["targets_met"]
               else "Still hunting holdout 7d toward +100%.")
            + f" [{_wf_msg(best_wf)}]"
        )
        log.info("AUTOPILOT HOLD: %s", decision["message"])
        _live_feed(
            "targets_met" if best_grade["targets_met"] else "hold",
            decision["message"],
            generation=generation, tested=tested, score=best_grade["score"],
            targets_met=best_grade["targets_met"], promote_tier="hold",
            soft_runs=soft_runs, hold_why=hold_why,
            leverage=int(best_cfg.leverage), timeframe=str(best_cfg.timeframe),
        )

    # Persist progress; elite only with forward proof (F3)
    state["last_improve"] = decision
    state["last_improve_at"] = decision["ts"]
    state["generation"] = generation
    state["improve_passes"] = int(state.get("improve_passes", 0)) + 1
    state["busy"] = False
    # Live must show the DEPLOYED book. On HOLD, never paint an unpromoted
    # candidate week (that is how +16% became -21% on the dashboard).
    state["grade"] = best_grade if improved else base_grade
    state["targets"] = HUNT_TARGETS
    state["hunt"] = hunt
    state["promote_tier"] = promote_tier
    state["leverage_ladder"] = leverage_ladder
    raw_score = int(best_grade.get("score_raw", best_grade["score"]))
    elite_raw = int((elite or {}).get("score_raw", (elite or {}).get("score", -1)))
    elite_ok = (
        _is_valid_candidate(best_i, best_o, best_soft)
        and raw_score > 0
        and not best_grade.get("kill_illusion")
        and best_wf_ok
        and fwd_ok  # F3: forward paper required for elite
    )
    if elite_ok and (elite is None or raw_score > elite_raw):
        state["elite"] = {
            "score": best_grade["score"],
            "score_raw": raw_score,
            "targets_met": best_grade["targets_met"],
            "confirmed": True,
            "knobs": knobs,
            "train_exp": decision["best"]["train_exp"],
            "test_exp": decision["best"]["test_exp"],
            "train_n": decision["best"]["train_n"],
            "test_n": decision["best"]["test_n"],
            "max_dd": decision["best"]["max_dd"],
            "win": best_grade.get("win"),
            "test_return": best_grade.get("test_return"),
            "oos_final": best_grade.get("oos_final"),
            "soft_return": best_grade.get("soft_return"),
            "soft_final": best_grade.get("soft_final"),
            "walk_forward": best_wf,
            "recent": best_recent,
            "week": best_week,
            "forward": fwd,
            "ts": decision["ts"],
            "generation": generation,
        }
        promote_tier = "confirmed"
        decision["promote_tier"] = "confirmed"
        state["forward_since"] = decision["ts"]
    elif confirmed:
        decision["promote_tier"] = "confirmed"
        state["forward_since"] = decision["ts"]

    # Champion floor: only raise, never lower. History rebuild already ran at pass start.
    if improved:
        new_week = float(best_week.get("ret") or 0)
        old_week = float((champion or {}).get("week_ret") or -999)
        if new_week >= old_week - 0.0005:
            state["champion"] = {
                "ts": decision["ts"],
                "generation": generation,
                "week_ret": new_week,
                "score": raw_score,
                "knobs": knobs,
                "week": best_week,
            }
        elif champion:
            state["champion"] = champion
    elif champion:
        state["champion"] = champion
    hist = list(state.get("history") or [])
    hist.append({
        "ts": decision["ts"],
        "generation": generation,
        "improved": improved,
        "promote_tier": decision.get("promote_tier"),
        "promote_reason": promote_reason,
        "tested": tested,
        "soft_runs": soft_runs,
        "search_mode": search_mode,
        "train_exp": decision["best"]["train_exp"],
        "test_exp": decision["best"]["test_exp"],
        "train_n": decision["best"]["train_n"],
        "test_n": decision["best"]["test_n"],
        "win": best_grade.get("win"),
        "test_return": best_grade.get("test_return"),
        "oos_final": best_grade.get("oos_final"),
        "soft_return": best_grade.get("soft_return"),
        "soft_final": best_grade.get("soft_final"),
        "kill_illusion": best_grade.get("kill_illusion"),
        "score": best_grade["score"],
        "score_raw": raw_score,
        "recent_exp": best_recent.get("exp"),
        "recent_n": best_recent.get("trades"),
        "recent_ok": best_recent_ok,
        "week_return": best_week.get("ret"),
        "targets_met": best_grade["targets_met"],
        "strategy": knobs.get("strategy"),
        "timeframe": knobs.get("timeframe"),
        "leverage": knobs.get("leverage"),
        "adx_enter": knobs["adx_enter"],
        "volume_multiplier": knobs["volume_multiplier"],
        "risk_pct": knobs.get("risk_pct"),
        "mom_lookback": knobs.get("mom_lookback"),
        "mom_min_atr": knobs.get("mom_min_atr"),
        "require_squeeze": knobs.get("require_squeeze"),
        "require_rsi_veto": knobs.get("require_rsi_veto"),
        "require_btc_regime": knobs.get("require_btc_regime"),
        # Full knobs so a blocked climb can be restored later
        "knobs": knobs,
        "leverage_ladder": leverage_ladder,
    })
    state["history"] = hist[-300:]
    _save_autopilot_state(state)
    try:
        Persistence(str(_ROOT / "moonshot.db"), mode="paper").log_event(
            _utc_now(), "*", "autopilot", decision,
        )
    except Exception:
        pass
    return decision


class PaperSupervisor:
    """Keep `python moonshot.py paper` alive as a child process."""

    def __init__(self, symbols: Optional[list] = None):
        self.symbols = symbols
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._stop_unlocked()
            cmd = [sys.executable, str(_ROOT / "moonshot.py"), "paper"]
            if self.symbols:
                cmd += ["--symbols", *self.symbols]
            log.info("starting paper child: %s", " ".join(cmd))
            self.proc = subprocess.Popen(
                cmd, cwd=str(_ROOT),
                env={**os.environ, "LIVE_TRADING_ENABLED": "0"},
            )

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def ensure(self) -> None:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                code = None if self.proc is None else self.proc.returncode
                if code not in (None, 0):
                    log.warning("paper child exited with code %s — restarting", code)
                # start without re-entering lock
                cmd = [sys.executable, str(_ROOT / "moonshot.py"), "paper"]
                if self.symbols:
                    cmd += ["--symbols", *self.symbols]
                log.info("starting paper child: %s", " ".join(cmd))
                self.proc = subprocess.Popen(
                    cmd, cwd=str(_ROOT),
                    env={**os.environ, "LIVE_TRADING_ENABLED": "0"},
                )


def cmd_autopilot(args: argparse.Namespace) -> None:
    try:
        clear_stale_live_caches()
    except Exception as e:
        log.warning("stale cache clear on autopilot start: %s", e)
    """
    Leave this running:
      - paper trading stays up
      - HUNT mode: wide search until near-perfect score targets are met
      - only keeps configs that are profitable on BOTH history halves
      - never turns on real-money trading
    """
    if os.getenv("LIVE_TRADING_ENABLED") == "1":
        raise ConfigError(
            "Autopilot refuses to start while LIVE_TRADING_ENABLED=1. "
            "Set it to 0 — autopilot is practice-only."
        )
    # Focus hunt + paper on BTC until edge is proven; --symbols overrides.
    symbols = args.symbols or list(HUNT_SYMBOLS)
    paper_symbols = args.symbols or list(PAPER_SYMBOLS)
    continuous = not bool(getattr(args, "daily", False))
    cooldown = max(5, int(getattr(args, "cooldown_seconds", 15)))
    every_h = float(getattr(args, "every_hours", 0) or 0)
    hunt = not bool(getattr(args, "tune_only", False))

    state = _load_autopilot_state()
    state["started_at"] = _utc_now().isoformat()
    state["mode"] = "hunt" if hunt else ("continuous" if continuous else f"every_{every_h}h")
    state["cooldown_seconds"] = cooldown
    state["targets"] = HUNT_TARGETS
    state["hunt"] = hunt
    state["hunt_symbols"] = list(symbols)
    state["paper_symbols"] = list(paper_symbols)
    _save_autopilot_state(state)

    if args.improve_only:
        gen = int(state.get("generation", 0))
        decision = improve_once(
            symbols, bars=args.bars, oos_frac=args.oos,
            refresh=args.refresh, generation=gen, hunt=hunt,
        )
        print(decision["message"])
        print(json.dumps({"best": decision["best"], "grade": decision.get("grade")}, indent=2))
        return

    supervisor = PaperSupervisor(paper_symbols)
    supervisor.start()
    stop = threading.Event()
    restart_paper = threading.Event()
    gen_box = {"n": int(state.get("generation", 0))}

    def improve_worker() -> None:
        while not stop.is_set():
            met = False
            just_promoted = False
            try:
                decision = improve_once(
                    symbols, bars=args.bars, oos_frac=args.oos,
                    refresh=False, generation=gen_box["n"], hunt=hunt,
                )
                gen_box["n"] += 1
                met = bool((decision.get("grade") or {}).get("targets_met"))
                if decision.get("improved"):
                    just_promoted = True
                    restart_paper.set()
                    _live_feed("paper_restart", "Restarting practice bot with new settings…")
            except Exception as e:
                log.error("improve pass failed: %s", e)
                _live_feed("error", f"Improve pass failed: {e}")
            if not continuous:
                stop.wait(max(60.0, every_h * 3600))
                continue
            # After a real promote: pause so paper can trade the new book instead
            # of immediate re-hunt → re-promote → restart thrash.
            if just_promoted:
                log.info(
                    "post-promote pause %ds — letting paper run before next hunt",
                    int(POST_PROMOTE_HUNT_PAUSE_SEC),
                )
                stop.wait(float(POST_PROMOTE_HUNT_PAUSE_SEC))
                continue
            # Not near-perfect → re-hunt immediately (tiny yield for other threads)
            if hunt and not met:
                stop.wait(2.0)
                continue
            # Targets met (or tune-only): brief pause before re-checking
            stop.wait(cooldown)

    worker = threading.Thread(target=improve_worker, name="moonshot-improve", daemon=True)
    worker.start()
    log.info(
        "AUTOPILOT ON — HUNT %d coins · %s · targets test_exp>=%.2f n>=%d · "
        "re-runs IMMEDIATELY until targets met · practice only",
        len(symbols), ",".join(symbols), HUNT_TARGETS["test_exp"], HUNT_TARGETS["test_n"],
    )
    _live_feed(
        "autopilot_on",
        f"HUNT on {len(symbols)} coins. Backtests until targets met "
        f"(avg ≥{_pct_of_risk_msg(HUNT_TARGETS['test_exp'])}, "
        f"win ≥{HUNT_TARGETS['win_rate']:.0%}, ≥{HUNT_TARGETS['test_n']} trades). "
        f"Fake money only.",
        cooldown=0, targets=HUNT_TARGETS, symbols=symbols,
    )

    try:
        while True:
            supervisor.ensure()
            if restart_paper.is_set():
                restart_paper.clear()
                log.info("restarting paper with upgraded settings")
                supervisor.start()
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("autopilot stopped by user")
    finally:
        stop.set()
        supervisor.stop()


def main() -> None:
    ap = argparse.ArgumentParser(description="moonshot trend bot")
    sub = ap.add_subparsers(dest="cmd")
    bt = sub.add_parser("backtest")
    bt.add_argument("--synthetic", action="store_true")
    bt.add_argument("--bars", type=int, default=3000)
    bt.add_argument("--symbols", nargs="*")
    bt.add_argument("--oos", type=float, default=0.3, help="out-of-sample fraction")
    bt.add_argument("--refresh", action="store_true", help="ignore the CSV cache")
    for name in ("paper", "live"):
        p = sub.add_parser(name)
        p.add_argument("--symbols", nargs="*")
    auto = sub.add_parser("autopilot", help="keep paper running and continuously self-improve")
    auto.add_argument("--symbols", nargs="*")
    auto.add_argument("--bars", type=int, default=15000)
    auto.add_argument("--oos", type=float, default=0.3)
    auto.add_argument("--cooldown-seconds", type=int, default=60,
                      help="pause only AFTER targets are met (default 60); "
                           "failed hunts re-run immediately")
    auto.add_argument("--daily", action="store_true",
                      help="old behavior: improve once per day instead of continuously")
    auto.add_argument("--every-hours", type=float, default=24.0,
                      help="only used with --daily")
    auto.add_argument("--tune-only", action="store_true",
                      help="small local nudges only (default is wide HUNT search)")
    auto.add_argument("--improve-only", action="store_true",
                      help="run one improve/hunt pass then exit (no paper child)")
    auto.add_argument("--refresh", action="store_true")
    sub.add_parser("selftest")
    args = ap.parse_args()

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    try:
        if args.cmd == "selftest":
            sys.exit(0 if selftest() else 1)
        elif args.cmd in (None, "backtest"):
            if args.cmd is None:
                args = bt.parse_args([])
            cmd_backtest(args)
        elif args.cmd == "paper":
            _live_runner(False, args.symbols or PAPER_SYMBOLS).run_forever()
        elif args.cmd == "autopilot":
            cmd_autopilot(args)
        elif args.cmd == "live":
            if os.getenv("LIVE_TRADING_ENABLED") != "1":
                raise ConfigError("Refusing live mode: set LIVE_TRADING_ENABLED=1 after 4+ weeks "
                                  "of paper results that match the backtest.")
            log.warning("LIVE TRADING - use the smallest size first")
            _live_runner(True, args.symbols or DEFAULT_SYMBOLS).run_forever()
    except BotError as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()