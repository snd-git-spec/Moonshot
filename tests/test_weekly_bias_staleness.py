"""0.1c — weekly_stale gates Momentum15 when require_weekly_bias."""

from __future__ import annotations

import numpy as np
import pandas as pd

from moonshot.config import Config
from moonshot.features import build_features
from moonshot.strategy import Momentum15Strategy
from moonshot.types import Direction


def _bars(n: int = 2500, *, gap_days: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    if gap_days > 0:
        # Punch a hole so ffill weekly ends become stale vs later bars
        cut = idx[n // 2]
        idx = idx.delete(np.where((idx > cut) & (idx <= cut + pd.Timedelta(days=gap_days)))[0])
    close = 100.0 * np.cumprod(1.0 + 0.0003 + rng.normal(0, 0.0008, len(idx)))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "volume": rng.uniform(1e3, 5e3, len(idx)),
        },
        index=idx,
    )


def test_weekly_bias_staleness():
    cfg = Config(
        strategy="momentum_15",
        timeframe="5m",
        require_weekly_bias=True,
        require_btc_regime=False,
        require_adx_for_entry=False,
        mom_min_atr=0.1,
        volume_multiplier=0.5,
    )
    raw = _bars(3000, gap_days=14)
    feats = build_features(raw, cfg)
    assert "weekly_stale" in feats.columns
    # After a multi-week gap, some bars must be marked stale
    assert bool(feats["weekly_stale"].any())

    strat = Momentum15Strategy(cfg)

    class Bar:
        atr = 1.0
        vol_ratio = 2.0
        adx = 30.0
        mom_atr = 1.0
        close = 100.0
        ema_htf = 90.0
        h4_bias = 1.0
        d_close = 110.0
        d_ema = 100.0
        weekly_bias = 1.0
        weekly_stale = True
        rsi = 50.0
        squeeze_ok = True
        btc_trend = 1.0

    assert strat.on_bar("ETH/USDT:USDT", Bar()) is None
    Bar.weekly_stale = False
    sig = strat.on_bar("ETH/USDT:USDT", Bar())
    assert sig is not None
    assert sig.direction == Direction.LONG
