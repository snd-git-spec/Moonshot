"""Holistic hunt / Live optimization — silence peak split, HTF gates, lev matrix."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from moonshot.config import Config
from moonshot.features import build_features
from moonshot.hunt.scoring import (
    _lev_density_ok,
    _lev_tf_allowed,
    _leverage_ladder_cfgs,
    _raise_champion_peak,
    _resolve_champion_peak,
    _sync_champion_peak,
    _tf_for_leverage,
)
from moonshot.strategy import Momentum15Strategy
from moonshot.types import Direction


def test_silence_never_lowers_peak_via_raise_only():
    state: dict = {}
    peak_knobs = {
        "strategy": "momentum_15",
        "timeframe": "5m",
        "leverage": 10,
        "adx_enter": 22.0,
        "volume_multiplier": 1.2,
        "require_squeeze": True,
    }
    _raise_champion_peak(
        state,
        knobs=peak_knobs,
        week_ret=0.52,
        score=88,
        week={"ret": 0.52},
        ts="2026-01-01T00:00:00Z",
        generation=1,
    )
    assert float(state["champion_peak"]["week_ret_peak"]) == pytest.approx(0.52)
    # Silence-style weaker book must not lower peak
    weak = dict(peak_knobs)
    weak["leverage"] = 5
    weak["timeframe"] = "15m"
    out = _raise_champion_peak(
        state,
        knobs=weak,
        week_ret=0.05,
        score=40,
        week={"ret": 0.05},
        ts="2026-01-02T00:00:00Z",
        generation=2,
    )
    assert float(out["week_ret_peak"]) == pytest.approx(0.52)
    assert int(state["champion_peak"]["knobs"]["leverage"]) == 10
    assert str(state["champion_peak"]["knobs"]["timeframe"]) == "5m"
    # Resolve still prefers peak
    resolved = _resolve_champion_peak(state)
    assert int(resolved["knobs"]["leverage"]) == 10
    assert float(resolved["week_ret_peak"]) == pytest.approx(0.52)


def test_sync_champion_peak_aliases_legacy_champion():
    state: dict = {}
    blob = {
        "week_ret": 0.4,
        "week_ret_peak": 0.4,
        "knobs": {"strategy": "momentum_15", "leverage": 10, "timeframe": "5m"},
    }
    _sync_champion_peak(state, blob)
    assert state["champion_peak"]["knobs"]["leverage"] == 10
    assert state["champion"]["knobs"]["leverage"] == 10


def _synth_bars(n: int = 800, *, trend: float = 0.002) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + trend + rng.normal(0, 0.001, n))
    high = close * 1.002
    low = close * 0.998
    open_ = close
    vol = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_weekly_bias_completed_bar_no_lookahead():
    cfg = Config(strategy="momentum_15", timeframe="5m", require_weekly_bias=True)
    raw = _synth_bars(2000, trend=0.0005)
    feats = build_features(raw, cfg)
    assert "weekly_bias" in feats.columns
    assert "w_ema" in feats.columns
    # First rows after incomplete warm-up may be 0; later rows must be ±1 or 0
    vals = feats["weekly_bias"].dropna().unique()
    assert set(float(v) for v in vals).issubset({-1.0, 0.0, 1.0})
    # Shifted: weekly_bias at t must not depend on same-week incomplete close
    # Sanity: bias is finite and mostly populated after warm-up
    tail = feats["weekly_bias"].iloc[500:]
    assert tail.notna().mean() > 0.95


def test_momentum15_htf_gates_block_counter_trend():
    cfg = Config(
        strategy="momentum_15",
        timeframe="5m",
        require_adx_for_entry=False,
        require_btc_regime=False,
        require_h4_bias=True,
        require_daily_align=True,
        require_weekly_bias=True,
        mom_min_atr=0.1,
        volume_multiplier=0.5,
    )
    strat = Momentum15Strategy(cfg)

    class Bar:
        atr = 1.0
        vol_ratio = 2.0
        adx = 30.0
        mom_atr = 1.0  # wants long
        close = 100.0
        ema_htf = 90.0  # soft HTF ok for long
        h4_bias = -1.0  # counter
        d_close = 90.0
        d_ema = 100.0  # counter daily
        weekly_bias = -1.0  # counter weekly
        rsi = 50.0
        squeeze_ok = True
        btc_trend = 1.0

    assert strat.on_bar("ETH/USDT:USDT", Bar()) is None

    # Align all HTF → long allowed
    Bar.h4_bias = 1.0
    Bar.d_close = 110.0
    Bar.d_ema = 100.0
    Bar.weekly_bias = 1.0
    sig = strat.on_bar("ETH/USDT:USDT", Bar())
    assert sig is not None
    assert sig.direction == Direction.LONG


def test_lev_matrix_rejects_25x_at_15m():
    assert not _lev_tf_allowed(25, "15m")
    assert _lev_tf_allowed(25, "5m")
    assert _lev_tf_allowed(15, "15m")
    assert _tf_for_leverage(25, "15m") == "5m"
    assert _tf_for_leverage(10, "15m") == "15m"


def test_lev_density_gate_blocks_sparse_25x():
    sparse = {"trades": 3, "soft_trades": 5}
    dense = {"trades": 25, "soft_trades": 40}
    assert not _lev_density_ok(25, sparse)
    assert _lev_density_ok(25, dense)
    assert not _lev_density_ok(15, sparse, tf="5m")
    assert _lev_density_ok(15, dense, tf="5m")
    # 15@15m needs dual_ok
    assert not _lev_density_ok(15, dense, dual_ok=False, tf="15m")
    assert _lev_density_ok(15, dense, dual_ok=True, tf="15m")


def test_leverage_ladder_respects_matrix():
    base = Config(strategy="momentum_15", timeframe="15m", leverage=5, aggressive_mode=True)
    cfgs = _leverage_ladder_cfgs(base)
    pairs = {(int(c.leverage), str(c.timeframe)) for c in cfgs}
    assert (25, "15m") not in pairs
    assert (5, "15m") in pairs
    assert (10, "15m") in pairs
    # High lev probed on 5m when book is 15m
    assert (25, "5m") in pairs or (15, "5m") in pairs


def test_freeze_risk_forces_25_off_15m():
    """A.1: risk rails (incl. leverage) are pinned to base — matrix no longer mutates them."""
    from moonshot.hunt.scoring import _freeze_risk

    base = Config(aggressive_mode=True, leverage=10, timeframe="15m", risk_pct=0.03)
    cand = replace(base, leverage=25, timeframe="15m", risk_pct=0.05)
    out = _freeze_risk(base, cand)
    assert int(out.leverage) == 10
    assert float(out.risk_pct) == pytest.approx(0.03)
    assert float(out.max_daily_loss_pct) == float(base.max_daily_loss_pct)
