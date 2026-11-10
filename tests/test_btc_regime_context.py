"""Alt-only books still need BTC context when require_btc_regime is on."""

from __future__ import annotations

from moonshot.config import Config
from moonshot.data import synthetic_ohlcv
from moonshot.features import prepare_data


def test_prepare_data_context_stamps_btc_trend_without_trading_btc():
    cfg = Config(
        starting_equity=1000.0,
        leverage=10,
        require_btc_regime=True,
        enable_pyramiding=False,
        enable_partial=False,
        strategy="momentum_15",
        timeframe="15m",
        adx_enter=10.0,
        adx_exit=8.0,
        require_squeeze=False,
        volume_multiplier=1.0,
        mom_min_atr=0.1,
    )
    sol = "SOL/USDT:USDT"
    btc = "BTC/USDT:USDT"
    raw = {sol: synthetic_ohlcv(sol, 900, tf=cfg.timeframe)}
    context = {btc: synthetic_ohlcv(btc, 900, tf=cfg.timeframe)}

    # Without context: alt btc_trend stays 0 → regime gate blocks everything.
    bare = prepare_data(raw, cfg)
    assert btc not in bare
    bare_trends = [float(getattr(b, "btc_trend", 0) or 0) for b in bare[sol].bars]
    assert all(t == 0.0 for t in bare_trends)

    # With context: BTC stamps trend onto SOL but is not a tradable key.
    data = prepare_data(raw, cfg, context=context)
    assert set(data.keys()) == {sol}
    trends = [float(getattr(b, "btc_trend", 0) or 0) for b in data[sol].bars]
    assert any(t != 0.0 for t in trends), "BTC context must stamp non-zero regime onto SOL"

    # At least some bars should be eligible past the BTC gate (may still fail other filters).
    gated = 0
    for b in data[sol].bars:
        if float(getattr(b, "btc_trend", 0) or 0) == 0.0:
            continue
        gated += 1
    assert gated > 0
