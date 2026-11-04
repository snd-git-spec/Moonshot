"""Config validation rejects bad knobs."""

from __future__ import annotations

import pytest

from moonshot.config import Config
from moonshot.types import ConfigError


def test_leverage_must_be_positive() -> None:
    with pytest.raises(ConfigError, match="leverage"):
        Config(leverage=0)


def test_risk_pct_bounds() -> None:
    with pytest.raises(ConfigError, match="risk_pct"):
        Config(risk_pct=0.0)
    with pytest.raises(ConfigError, match="risk_pct"):
        Config(risk_pct=0.5)


def test_adx_exit_vs_enter() -> None:
    with pytest.raises(ConfigError, match="adx_exit"):
        Config(adx_enter=20, adx_exit=25)


def test_valid_defaults() -> None:
    cfg = Config()
    assert cfg.leverage >= 1
    assert 0 < cfg.risk_pct <= 0.12
