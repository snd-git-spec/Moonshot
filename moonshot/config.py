from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    ConfigError,
    _pandas_tf,
)

from moonshot.paths import ENV_PATH

try:
    from dotenv import load_dotenv

    load_dotenv(ENV_PATH)
except ImportError:
    pass


@dataclass
class Config:
    # account
    starting_equity: float = 50.0
    # Hunt/backtest research notional (paper Live uses starting_equity / STARTING_EQUITY).
    research_starting_equity: float = 50.0
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
    max_peak_drawdown_pct: float = 0.45  # live: permanent kill; backtest: circuit pause
    streak_losses: int = 5  # consecutive losses -> temporary pause
    streak_pause_hours: float = 48.0
    expectancy_window: int = 30
    # 0.0 kills 15m books on normal noise (creates "kill illusion" OOS). Soft floor.
    expectancy_floor: float = -0.15  # rolling expectancy below this -> trip
    # Backtest only: after a trip, pause this long then resume (live stays permanent).
    kill_resume_hours: float = 168.0  # 7 days

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
    ema_entry_period: int = 20  # 15m EMA for pullback / resume timing
    rsi_rule_min: float = 70.0  # momentum_rule: RSI must exceed this
    btc_adx_min: float = 0.0  # chop filter: skip all entries when BTC ADX below this (0 = off)
    breadth_min: float = 0.0  # chop filter: need |share of coins trending| >= this, same side
    pullback_lookback: int = 4  # bars to look back for a touch of ema_entry
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
    rsi_long_max: float = 70.0  # block longs when RSI already extended
    rsi_short_min: float = 30.0  # block shorts when RSI already extended
    require_rsi_veto: bool = False  # hunt-optional; stacking with squeeze/BTC starved samples
    bb_period: int = 20
    bb_std: float = 2.0
    kc_period: int = 20
    kc_atr_mult: float = 1.5
    squeeze_lookback: int = 8  # squeeze must have fired within this many bars
    require_squeeze: bool = False
    require_btc_regime: bool = True
    # Higher-TF entry gates (Momentum15 / hunt-optional; default off for back-compat)
    require_h4_bias: bool = False
    require_daily_align: bool = False
    require_weekly_bias: bool = False
    ema_weekly_period: int = 10

    # time stop — from_env defaults ENABLE_TIME_STOP=1, TIME_STOP_BARS=8
    enable_time_stop: bool = True
    time_stop_bars: int = 8
    time_stop_min_r: float = 1.0

    # pyramiding
    enable_pyramiding: bool = True
    max_adds_per_trade: int = 2
    add_trigger_r: tuple = (2.0, 3.0)
    min_add_qty_pct: float = 0.25
    addon_giveback_r: float = 0.0  # 0 = adds must be "free-roll" at the tightened stop

    # partial profit — from_env default ENABLE_PARTIAL=1
    enable_partial: bool = True
    partial_at_r: float = 1.0
    partial_pct: float = 0.50

    # costs (VERIFY against MEXC's current futures fee schedule)
    fee_taker: float = (
        0.0008  # MEXC API futures taker (web/app is 0.02%; API is 0.08% since 2026-06-01)
    )
    # Maker used by cost-stress 2× gate (fold_eval.cost_ok_for_cfg). Default = taker
    # so API-path books stay conservative when maker is unknown.
    fee_maker: float = 0.0008
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
            # Defined in data.py; lazy to avoid config↔data import cycle.
            from moonshot.data import live_history_bars_for_tf

            object.__setattr__(
                self,
                "live_history_bars",
                live_history_bars_for_tf(self.timeframe),
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
            research_starting_equity=float(os.getenv("RESEARCH_STARTING_EQUITY", "50.0")),
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
            kill_resume_hours=float(os.getenv("KILL_RESUME_HOURS", "168")),
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
            require_h4_bias=os.getenv("REQUIRE_H4_BIAS", "0") == "1",
            require_daily_align=os.getenv("REQUIRE_DAILY_ALIGN", "0") == "1",
            require_weekly_bias=os.getenv("REQUIRE_WEEKLY_BIAS", "0") == "1",
            ema_weekly_period=int(os.getenv("EMA_WEEKLY_PERIOD", "10")),
            enable_time_stop=os.getenv("ENABLE_TIME_STOP", "1") == "1",
            time_stop_bars=int(os.getenv("TIME_STOP_BARS", "8")),
            time_stop_min_r=float(os.getenv("TIME_STOP_MIN_R", "1.0")),
            strategy=os.getenv("STRATEGY", "trend_mtf").strip().lower(),
            timeframe=os.getenv("TIMEFRAME", "15m"),
            mom_lookback=int(os.getenv("MOM_LOOKBACK", "8")),
            mom_min_atr=float(os.getenv("MOM_MIN_ATR", "0.5")),
            initial_stop_atr=float(os.getenv("INITIAL_STOP_ATR", "1.5")),
            fee_taker=float(os.getenv("FEE_TAKER", "0.0008")),  # MEXC API futures taker 0.08%
            fee_maker=float(os.getenv("FEE_MAKER", os.getenv("FEE_TAKER", "0.0008"))),
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
