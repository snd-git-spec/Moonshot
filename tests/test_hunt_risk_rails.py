"""A.1 — hunt candidates never vary pinned risk rails."""

from __future__ import annotations

from moonshot.config import Config
from moonshot.hunt.scoring import (
    _candidate_cfgs,
    _freeze_risk,
    _leverage_ladder_cfgs,
    _polish_cfgs,
    _sample_cfg,
)
import numpy as np


def _rails(cfg: Config) -> tuple:
    return (
        int(cfg.leverage),
        float(cfg.risk_pct),
        float(cfg.max_daily_loss_pct),
        float(cfg.max_peak_drawdown_pct),
    )


def test_hunt_never_varies_risk_rails():
    base = Config(
        aggressive_mode=True,
        leverage=10,
        risk_pct=0.03,
        max_daily_loss_pct=0.32,
        max_peak_drawdown_pct=0.55,
        timeframe="5m",
        strategy="momentum_15",
    )
    want = _rails(base)
    rng = np.random.default_rng(0)
    pool = []
    pool.extend(_polish_cfgs(base)[:20])
    pool.extend(_candidate_cfgs(base, generation=1, hunt=True, search_mode="explore")[:20])
    pool.extend(_leverage_ladder_cfgs(base))
    for _ in range(10):
        pool.append(_sample_cfg(base, rng, wide=True))
    for cand in pool:
        out = _freeze_risk(base, cand)
        assert _rails(out) == want, (cand.leverage, out.leverage, want)
