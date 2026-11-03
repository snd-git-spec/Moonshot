"""Moonshot trading package — public API via import moonshot as ms."""
from __future__ import annotations

from moonshot._pull import pull
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

import moonshot.types as _types; pull(_types, globals())
import moonshot.config as _config; pull(_config, globals())
import moonshot.features as _features; pull(_features, globals())
import moonshot.data as _data; pull(_data, globals())
import moonshot.strategy as _strategy; pull(_strategy, globals())
import moonshot.position as _position; pull(_position, globals())
import moonshot.risk as _risk; pull(_risk, globals())
import moonshot.portfolio as _portfolio; pull(_portfolio, globals())
import moonshot.brokers as _brokers; pull(_brokers, globals())
import moonshot.persist as _persist; pull(_persist, globals())
import moonshot.backtest as _backtest; pull(_backtest, globals())
import moonshot.live as _live; pull(_live, globals())
import moonshot.selftest as _selftest; pull(_selftest, globals())
import moonshot.symbols as _symbols; pull(_symbols, globals())
import moonshot.hunt.constants as _hconst; pull(_hconst, globals())
import moonshot.hunt.scoring as _hscore; pull(_hscore, globals())
import moonshot.hunt.improve as _himprove; pull(_himprove, globals())
import moonshot.hunt.autopilot as _hauto; pull(_hauto, globals())
import moonshot.cli as _cli; pull(_cli, globals())

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
]

ohlcv_is_stale = _ohlcv_is_stale  # noqa: F405
