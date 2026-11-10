"""Champion/elite rebuild must restore strategy, timeframe, and leverage."""

from __future__ import annotations

from moonshot.config import Config
from moonshot.hunt.scoring import knobs_to_cfg


def test_knobs_to_cfg_restores_champion_stl_against_different_live_base() -> None:
    base = Config(
        strategy="trend_mtf",
        timeframe="15m",
        leverage=7,
        max_account_leverage=7.0,
        max_net_leverage=7.0,
        aggressive_mode=False,
        adx_enter=28.0,
        adx_exit=22.0,
        volume_multiplier=1.2,
    )
    kn = {
        "strategy": "momentum_15",
        "timeframe": "5m",
        "leverage": 10,
        "adx_enter": 22.0,
        "adx_exit": 14.0,
        "volume_multiplier": 1.2,
        "mom_lookback": 12,
        "mom_min_atr": 0.75,
        "require_squeeze": True,
        "require_btc_regime": True,
    }
    cfg = knobs_to_cfg(kn, base=base)
    assert cfg.strategy == "momentum_15"
    assert cfg.timeframe == "5m"
    assert int(cfg.leverage) == 10
