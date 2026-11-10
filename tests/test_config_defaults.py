"""0.1b — Config() dataclass defaults must match Config.from_env() with empty env."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from moonshot.config import Config

_ENV_KEYS = (
    "STARTING_EQUITY",
    "RESEARCH_STARTING_EQUITY",
    "AGGRESSIVE_MODE",
    "LEVERAGE",
    "MAX_ACCOUNT_LEVERAGE",
    "RISK_PCT",
    "MAX_OPEN_POSITIONS",
    "MAX_NET_LEVERAGE",
    "INITIAL_MARGIN_PCT",
    "AGGRESSIVE_MARGIN_PCT",
    "MAX_DAILY_LOSS_PCT",
    "MAX_PEAK_DRAWDOWN_PCT",
    "STREAK_LOSSES",
    "STREAK_PAUSE_HOURS",
    "EXPECTANCY_FLOOR",
    "KILL_RESUME_HOURS",
    "ENABLE_PYRAMIDING",
    "ENABLE_PARTIAL",
    "PARTIAL_AT_R",
    "PARTIAL_PCT",
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
    "RSI_PERIOD",
    "RSI_LONG_MAX",
    "RSI_SHORT_MIN",
    "REQUIRE_RSI_VETO",
    "BB_PERIOD",
    "BB_STD",
    "KC_PERIOD",
    "KC_ATR_MULT",
    "SQUEEZE_LOOKBACK",
    "REQUIRE_SQUEEZE",
    "REQUIRE_BTC_REGIME",
    "REQUIRE_H4_BIAS",
    "REQUIRE_DAILY_ALIGN",
    "REQUIRE_WEEKLY_BIAS",
    "EMA_WEEKLY_PERIOD",
    "ENABLE_TIME_STOP",
    "TIME_STOP_BARS",
    "TIME_STOP_MIN_R",
    "STRATEGY",
    "TIMEFRAME",
    "MOM_LOOKBACK",
    "MOM_MIN_ATR",
    "INITIAL_STOP_ATR",
    "FEE_TAKER",
    "SLIPPAGE_BPS",
    "H4_EMA_FAST",
    "H4_EMA_SLOW",
    "H4_ADX_MIN_SHORT",
    "EMA_ENTRY_PERIOD",
    "RSI_RULE_MIN",
    "BTC_ADX_MIN",
    "BREADTH_MIN",
    "PULLBACK_LOOKBACK",
    "LIVE_HISTORY_BARS",
    "LOG_LEVEL",
)


@pytest.fixture
def clear_config_env(monkeypatch):
    for k in _ENV_KEYS:
        monkeypatch.delenv(k, raising=False)


def test_config_equals_from_env_empty(clear_config_env):
    a = Config()
    b = Config.from_env()
    da, db = asdict(a), asdict(b)
    diffs = {k: (da[k], db[k]) for k in da if da[k] != db[k]}
    assert diffs == {}, f"Config() vs from_env mismatches: {diffs}"
