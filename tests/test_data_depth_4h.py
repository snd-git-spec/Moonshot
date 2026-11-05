"""A.4 — depth cache + 4h fold trade-count planning."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from moonshot.data_cache import (
    estimate_fold_trades_4h,
    get_depth_cache,
    put_depth_cache,
    scale_lookback,
)
from moonshot.hunt.constants import HUNT_TF_ALLOW


def test_fold_trade_count_4h(tmp_path: Path):
    assert "1h" in HUNT_TF_ALLOW and "4h" in HUNT_TF_ALLOW
    assert "5m" in HUNT_TF_ALLOW and "15m" in HUNT_TF_ALLOW

    # ~3y of 4h bars ≈ 3*365*6 = 6570
    n_bars = 6570
    counts = estimate_fold_trades_4h(n_bars, n_folds=4, signals_per_bar=0.025)
    assert len(counts) == 4
    assert min(counts) >= 30

    assert scale_lookback(40, "15m") == 40
    assert scale_lookback(40, "5m") == 40
    assert scale_lookback(40, "1h") == 10
    assert scale_lookback(40, "4h") == 2  # round(40/16)=2

    idx = pd.date_range("2020-01-01", periods=50, freq="4h", tz="UTC")
    df = pd.DataFrame(
        {"open": 1.0, "high": 1.1, "low": 0.9, "close": 1.05, "volume": 10.0},
        index=idx,
    )
    put_depth_cache(
        df,
        symbol="BTC/USDT:USDT",
        timeframe="4h",
        start="2020-01-01",
        end="2020-01-10",
        base=tmp_path,
    )
    hit = get_depth_cache(
        symbol="BTC/USDT:USDT",
        timeframe="4h",
        start="2020-01-01",
        end="2020-01-10",
        base=tmp_path,
    )
    assert hit is not None
    assert len(hit) == 50
