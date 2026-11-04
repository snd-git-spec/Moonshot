from __future__ import annotations

import logging
from typing import Any, Optional


log = logging.getLogger("moonshot")

from moonshot.types import (
    Direction,
    RegimeState,
    Signal,
)
from moonshot.config import (
    Config,
)
from moonshot.features import (
    _is_btc_symbol,
)


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
            return Signal(
                symbol,
                Direction.LONG,
                bar.close,
                bar.atr,
                RegimeState.TRENDING_UP,
                f"breakout>{bar.don_high:.4g}+buf ADX={bar.adx:.1f}↑ vol x{bar.vol_ratio:.1f}{tag}",
            )
        if (not want_long) and bar.close < bar.don_low - buf:
            return Signal(
                symbol,
                Direction.SHORT,
                bar.close,
                bar.atr,
                RegimeState.TRENDING_DOWN,
                f"breakdown<{bar.don_low:.4g}-buf ADX={bar.adx:.1f}↑ vol x{bar.vol_ratio:.1f}{tag}",
            )
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
            return Signal(
                symbol, Direction.LONG, bar.close, bar.atr, RegimeState.TRENDING_UP, f"mom_up {tag}"
            )
        return Signal(
            symbol, Direction.SHORT, bar.close, bar.atr, RegimeState.TRENDING_DOWN, f"mom_dn {tag}"
        )


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
            return Signal(
                symbol,
                Direction.LONG,
                bar.close,
                bar.atr,
                RegimeState.TRENDING_UP,
                f"mtf_long {tag}",
            )
        return Signal(
            symbol,
            Direction.SHORT,
            bar.close,
            bar.atr,
            RegimeState.TRENDING_DOWN,
            f"mtf_short {tag}",
        )


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
        if (
            self.cfg.require_btc_regime
            and not _is_btc_symbol(symbol)
            and float(getattr(bar, "btc_trend", 0.0) or 0.0) <= 0
        ):
            return None
        return Signal(
            symbol,
            Direction.LONG,
            bar.close,
            bar.atr,
            RegimeState.TRENDING_UP,
            f"rule_long rsi={rsi_v:.0f} adx={bar.adx:.0f} ema50>200",
        )


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
