"""Shared fixtures for Moonshot unit tests (no network)."""
from __future__ import annotations

import pytest

from moonshot.config import Config
from moonshot.data import synthetic_ohlcv
from moonshot.features import prepare_data


@pytest.fixture
def cfg() -> Config:
    return Config(
        starting_equity=50.0,
        leverage=5,
        require_btc_regime=False,
        enable_pyramiding=False,
        enable_partial=False,
        strategy="momentum_15",
        timeframe="15m",
    )


@pytest.fixture
def synthetic_book(cfg: Config) -> dict:
    raw = {
        s: synthetic_ohlcv(s, 800, tf=cfg.timeframe)
        for s in ("BTC/USDT:USDT", "ETH/USDT:USDT")
    }
    return prepare_data(raw, cfg)
