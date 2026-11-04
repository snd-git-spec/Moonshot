"""Moonshot trading package — public API via import moonshot as ms."""

from __future__ import annotations

from moonshot.paths import (
    AUTOPILOT_STATE,
    CACHE_DIR,
    DB_PATH,
    ENV_PATH,
    LIVE_FEED,
    PAPER_RUNTIME_PATH,
    ROOT,
    RULES_PATH,
)

from moonshot.types import BotError, DataError
from moonshot.config import Config
from moonshot.features import adx, atr, build_features, ema, prepare_data, rsi
from moonshot.data import (
    MAX_RANGE_BARS,
    RANGE_SOURCE_BINANCE_SPOT,
    RANGE_SOURCE_LABELS,
    RANGE_SOURCE_MEXC,
    RANGE_SOURCE_YAHOO,
    RANGE_TF_LADDER,
    _ohlcv_is_stale,
    estimate_range_bars,
    estimate_range_fetch_bars,
    fetch_historical,
    fetch_historical_range,
    pick_range_timeframe,
    public_exchange,
    range_warmup_bars,
    synthetic_ohlcv,
)
from moonshot.position import Position
from moonshot.backtest import (
    BacktestEngine,
    BacktestResult,
    promotion_reading,
    results_as_frame,
    run_grid,
    verdicts,
)
from moonshot.symbols import (
    DEFAULT_SYMBOLS,
    HUNT_SYMBOLS,
    OPTIONAL_SYMBOLS,
    PAPER_SYMBOLS,
)
from moonshot.selftest import selftest
from moonshot.hunt.constants import _hunt_bars_for_tf
from moonshot.hunt.scoring import RECENT_DAYS, _grade, _soft_kill_cfg
from moonshot.hunt.improve import improve_once
from moonshot.hunt.autopilot import PaperSupervisor
from moonshot.cli import cmd_autopilot, cmd_backtest, main, setup_logging

ohlcv_is_stale = _ohlcv_is_stale

__all__ = [
    "Config",
    "BotError",
    "DataError",
    "BacktestResult",
    "BacktestEngine",
    "Position",
    "DEFAULT_SYMBOLS",
    "OPTIONAL_SYMBOLS",
    "PAPER_SYMBOLS",
    "HUNT_SYMBOLS",
    "fetch_historical",
    "fetch_historical_range",
    "synthetic_ohlcv",
    "public_exchange",
    "prepare_data",
    "range_warmup_bars",
    "estimate_range_bars",
    "estimate_range_fetch_bars",
    "pick_range_timeframe",
    "MAX_RANGE_BARS",
    "RANGE_TF_LADDER",
    "RANGE_SOURCE_MEXC",
    "RANGE_SOURCE_BINANCE_SPOT",
    "RANGE_SOURCE_YAHOO",
    "RANGE_SOURCE_LABELS",
    "adx",
    "atr",
    "ema",
    "rsi",
    "build_features",
    "RECENT_DAYS",
    "_hunt_bars_for_tf",
    "_soft_kill_cfg",
    "run_grid",
    "promotion_reading",
    "results_as_frame",
    "verdicts",
    "_grade",
    "_ohlcv_is_stale",
    "ohlcv_is_stale",
    "selftest",
    "improve_once",
    "PaperSupervisor",
    "cmd_autopilot",
    "cmd_backtest",
    "main",
    "setup_logging",
    "ROOT",
    "ENV_PATH",
    "DB_PATH",
    "CACHE_DIR",
    "AUTOPILOT_STATE",
    "LIVE_FEED",
    "PAPER_RUNTIME_PATH",
    "RULES_PATH",
]
