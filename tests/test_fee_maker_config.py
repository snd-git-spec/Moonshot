"""cost_ok_for_cfg must be able to replace(cfg, fee_maker=...)."""

from __future__ import annotations

from dataclasses import replace

from moonshot.config import Config


def test_config_accepts_fee_maker_replace():
    cfg = Config(strategy="momentum_15", timeframe="1h", leverage=5)
    c2 = replace(cfg, fee_taker=0.0016, fee_maker=0.0016, slippage_bps=6.0)
    assert c2.fee_maker == 0.0016
    assert c2.fee_taker == 0.0016


def test_from_env_has_fee_maker(monkeypatch):
    monkeypatch.setenv("FEE_TAKER", "0.0008")
    monkeypatch.setenv("FEE_MAKER", "0.0006")
    cfg = Config.from_env()
    assert abs(cfg.fee_maker - 0.0006) < 1e-12
