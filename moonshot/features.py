from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    InstrumentSpec,
    RegimeState,
)
from moonshot.config import (
    Config,
)


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
        f["squeeze"].astype(float).rolling(cfg.squeeze_lookback, min_periods=1).max().fillna(0.0)
        > 0.0
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
    f["pullback_long"] = (
        near_long.fillna(False).astype(float).rolling(pb_n, min_periods=1).max() > 0
    )
    f["pullback_short"] = (
        near_short.fillna(False).astype(float).rolling(pb_n, min_periods=1).max() > 0
    )

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

    # Completed weekly bias (no lookahead): close vs weekly EMA on closed weeks only
    weekly = resample_ohlcv(f, "1W")
    w_period = max(2, int(getattr(cfg, "ema_weekly_period", 10) or 10))
    weekly["ema_w"] = ema(weekly["close"], w_period)
    w_prev = weekly[["close", "ema_w"]].shift(1).rename(columns={"close": "w_close"})
    w_prev = w_prev.reindex(f.index, method="ffill")
    f["w_close"] = w_prev["w_close"]
    f["w_ema"] = w_prev["ema_w"]
    w_up = (f["w_close"] > f["w_ema"]).fillna(False)
    w_dn = (f["w_close"] < f["w_ema"]).fillna(False)
    f["weekly_bias"] = np.where(w_up, 1.0, np.where(w_dn, -1.0, 0.0))

    valid = f[["ema_htf", "atr", "adx", "d_ema"]].notna().all(axis=1).to_numpy()
    adx_on = hysteresis(f["adx"].to_numpy(), cfg.adx_enter, cfg.adx_exit)
    up = adx_on & (f["close"] > f["ema_htf"]).to_numpy() & (f["d_close"] > f["d_ema"]).to_numpy()
    dn = adx_on & (f["close"] < f["ema_htf"]).to_numpy() & (f["d_close"] < f["d_ema"]).to_numpy()
    f["regime"] = np.where(
        ~valid,
        RegimeState.UNKNOWN.value,
        np.where(
            up,
            RegimeState.TRENDING_UP.value,
            np.where(dn, RegimeState.TRENDING_DOWN.value, RegimeState.RANGING.value),
        ),
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


def prepare_symbol(
    symbol: str,
    df: pd.DataFrame,
    cfg: Config,
    spec: Optional[InstrumentSpec] = None,
    feats: Optional[pd.DataFrame] = None,
) -> SymbolData:
    frame = feats if feats is not None else build_features(df, cfg)
    bars = list(frame.itertuples(name="Bar"))
    return SymbolData(
        symbol, bars, {b.Index: i for i, b in enumerate(bars)}, spec or InstrumentSpec(symbol)
    )


def default_btc_symbol() -> str:
    """Canonical BTC perp used for regime context when the book has no BTC."""
    return "BTC/USDT:USDT"


def needs_btc_regime_context(symbols, cfg: Config) -> bool:
    """True when alt-only selection would silently block every entry."""
    return bool(getattr(cfg, "require_btc_regime", False)) and _btc_symbol_in(symbols) is None


def prepare_data(raw: dict, cfg: Config, context: Optional[dict] = None) -> dict:
    """Build per-symbol features; stamp BTC trend onto alts when enabled.

    ``context`` is optional OHLCV used only for market regime / breadth
    (typically BTC when the user picked alt-only). Context symbols are
    **not** returned and therefore not traded by BacktestEngine.
    """
    tradable = set(raw.keys())
    merged = dict(raw)
    if context:
        for s, df in context.items():
            if s not in merged:
                merged[s] = df
    feats = {s: build_features(df, cfg) for s, df in merged.items()}
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
    elif cfg.require_btc_regime and btc is None and tradable:
        log.warning(
            "require_btc_regime on but no BTC in book/context — alt entries will be blocked"
        )
    # Market-wide context (no lookahead: values at bar close). Used by chop filters.
    if btc is not None:
        btc_adx = feats[btc]["adx"].astype(float)
    else:
        btc_adx = pd.concat([f["adx"] for f in feats.values()], axis=1).mean(axis=1)
    breadth = pd.concat([f["h4_bias"].astype(float) for f in feats.values()], axis=1).mean(axis=1)
    for f in feats.values():
        f["btc_adx"] = btc_adx.reindex(f.index).ffill().fillna(0.0)
        f["breadth"] = breadth.reindex(f.index).ffill().fillna(0.0)
    return {
        s: prepare_symbol(s, merged[s], cfg, feats=feats[s])
        for s in tradable
        if s in feats and s in merged
    }
