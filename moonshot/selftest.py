from __future__ import annotations

import logging
import math
from dataclasses import replace

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    ActionKind,
    AddDecision,
    Direction,
    ExitReason,
    InstrumentSpec,
    RegimeState,
    RiskError,
)
from moonshot.config import (
    Config,
)
from moonshot.features import (
    build_features,
    prepare_data,
)
from moonshot.data import (
    synthetic_ohlcv,
)
from moonshot.strategy import (
    make_strategy,
)
from moonshot.position import (
    Position,
)
from moonshot.risk import (
    RiskEngine,
)
from moonshot.portfolio import (
    PositionManager,
    RollManager,
)
from moonshot.brokers import (
    PaperBroker,
)
from moonshot.backtest import (
    BacktestEngine,
)
from moonshot.hunt.scoring import (
    FWD_SCORE_CAP,
    _grade,
)


def selftest() -> bool:
    cfg = Config()
    ok = True

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
        ok &= bool(cond)

    print("selftest")
    # 1) no lookahead: features at row k-1 identical with or without future data
    df = synthetic_ohlcv("TEST", 1800)
    k = 1500
    full, trunc = build_features(df, cfg), build_features(df.iloc[:k], cfg)
    cols = ["ema_htf", "atr", "adx", "vol_ratio", "don_high", "d_close", "d_ema"]
    same = np.allclose(full[cols].iloc[:k].to_numpy(), trunc[cols].to_numpy(), equal_nan=True)
    check("features have no lookahead (numeric)", same)
    check(
        "regime has no lookahead",
        (full["regime"].iloc[:k].to_numpy() == trunc["regime"].to_numpy()).all(),
    )

    # 2) average-entry accounting
    t0 = pd.Timestamp("2025-01-01", tz="UTC")
    p = Position.open("X", Direction.LONG, 5, 100.0, 1.0, 98.0, t0, 0.0)
    p.add_lot(110.0, 0.5, t0, 0.0, 106.0, 10.0)
    check("avg entry after add", abs(p.avg_entry - (100 + 55) / 1.5) < 1e-9)
    check("P&L uses average entry", abs(p.unrealized(120.0) - (120 - 155 / 1.5) * 1.5) < 1e-9)

    # paper restart must NOT fabricate stop-outs from restored positions
    pb = PaperBroker(cfg)
    t_open = pd.Timestamp("2025-01-01", tz="UTC")
    restored = {
        "BTC/USDT:USDT": Position.open(
            "BTC/USDT:USDT",
            Direction.LONG,
            5,
            100.0,
            1.0,
            98.0,
            t_open,
            0.02,
        ),
    }
    pb.rehydrate(restored, closed_pnl=-1.5)
    check("paper rehydrate restores signed qty", abs(pb.signed["BTC/USDT:USDT"] - 1.0) < 1e-12)
    check("paper rehydrate restores stop", abs(pb.stops["BTC/USDT:USDT"] - 98.0) < 1e-12)
    check(
        "paper rehydrate cash = start + closed − open fees",
        abs(pb.cash - (cfg.starting_equity - 1.5 - 0.02)) < 1e-9,
    )
    # price still above stop → on_price must NOT fill
    pb.on_price("BTC/USDT:USDT", 99.0)
    check("paper rehydrate: no false stop above stop", pb.pop_stop_fill("BTC/USDT:USDT") is None)
    pb.on_price("BTC/USDT:USDT", 97.5)
    fill = pb.pop_stop_fill("BTC/USDT:USDT")
    check("paper rehydrate: real stop still fires", fill is not None and fill.side == "sell")
    try:
        p.add_lot(111.0, 0.1, t0, 0.0, 90.0, 0.0)
        check("stop widening rejected", False)
    except RiskError:
        check("stop widening rejected", True)

    # 3) free-roll sizing: after the add, stopping out is still not a net loss
    q = Position.open("X", Direction.LONG, 5, 100.0, 1.0, 98.0, t0, 0.0)
    qty, dec = RollManager(cfg).size_add(q, 110.0, 106.0, 50.0, q.margin(), InstrumentSpec("X"))
    if qty:
        q.add_lot(110.0, qty, t0, 0.0, 106.0, 10.0)
    check(
        "free-roll add allowed and non-losing at stop",
        dec is AddDecision.ALLOW and q.net_pnl_at(106.0) >= -1e-9,
    )

    # 4) daily halt resets next UTC day; kill switch does not
    r = RiskEngine(Config(max_daily_loss_pct=0.05))
    r.update_equity(50.0, pd.Timestamp("2025-01-01 04:00", tz="UTC"))
    r.update_equity(46.0, pd.Timestamp("2025-01-01 08:00", tz="UTC"))
    halted = not r.trading_allowed(pd.Timestamp("2025-01-01 09:00", tz="UTC"))
    r.update_equity(46.0, pd.Timestamp("2025-01-02 00:00", tz="UTC"))
    check(
        "daily halt trips then resets next day",
        halted and r.trading_allowed(pd.Timestamp("2025-01-02 01:00", tz="UTC")),
    )

    # 5) R-multiple uses real initial risk, not 2% of equity
    t = Position.open("X", Direction.LONG, 5, 100.0, 2.0, 98.0, t0, 0.0)
    res, _ = t.close_out(104.0, 0.0, t0, ExitReason.STOP)
    check("R = pnl / (qty * stop distance)", abs(res.r_multiple - 2.0) < 1e-9)

    # 5b) time-stop fires when stagnant past N bars without +min_r
    ts_cfg = Config(
        enable_time_stop=True,
        time_stop_bars=3,
        time_stop_min_r=1.0,
        enable_partial=False,
        enable_pyramiding=False,
    )
    pos_ts = Position.open(
        "X", Direction.LONG, 5, 100.0, 1.0, 98.0, pd.Timestamp("2025-01-01 00:00", tz="UTC"), 0.0
    )

    # fake bar 4h later still flat
    class _B:
        Index = pd.Timestamp("2025-01-01 04:00", tz="UTC")
        close = 100.2
        atr = 1.0
        regime = RegimeState.TRENDING_UP.value

    acts = PositionManager(ts_cfg).decide(
        pos_ts,
        _B(),
        allowed=True,
        kill=False,
        equity=50.0,
        used_margin=0.0,
        spec=InstrumentSpec("X"),
    )
    check(
        "time stop exits stagnant trade",
        any(a.kind is ActionKind.EXIT and a.reason is ExitReason.TIME_STOP for a in acts),
    )

    # 6) feature columns for quality filters + momentum + 4H MTF
    demo = build_features(synthetic_ohlcv("BTC/USDT:USDT", 400), cfg)
    for col in (
        "rsi",
        "squeeze",
        "squeeze_ok",
        "bb_upper",
        "kc_upper",
        "btc_trend",
        "mom_atr",
        "h4_bias",
        "h4_adx",
        "ema_entry",
        "pullback_long",
        "pullback_short",
    ):
        check(f"feature {col} present", col in demo.columns)
    check("h4_bias has values", demo["h4_bias"].notna().any())

    # 7) end-to-end smoke on synthetic data (include BTC for regime stamp)
    raw = {s: synthetic_ohlcv(s, 2500) for s in ("BTC/USDT:USDT", "ETH/USDT:USDT", "SOL/USDT:USDT")}
    data = prepare_data(raw, cfg)
    eth_bar = data["ETH/USDT:USDT"].bars[-1]
    check("btc_trend stamped onto alts", hasattr(eth_bar, "btc_trend"))
    res = BacktestEngine(data, cfg).run()
    finite = math.isfinite(res.final_equity) and res.final_equity >= 0
    check(f"backtest smoke run ({res.trades} trades, final {res.final_equity:.2f})", finite)
    check("equity curve is time-ordered", res.equity_curve.index.is_monotonic_increasing)
    mom_cfg = replace(cfg, strategy="momentum_15", timeframe="15m", require_btc_regime=False)
    mom_res = BacktestEngine(prepare_data(raw, mom_cfg), mom_cfg).run()
    check(
        f"momentum_15 smoke ({mom_res.trades} trades, final {mom_res.final_equity:.2f})",
        math.isfinite(mom_res.final_equity) and mom_res.final_equity >= 0,
    )
    check("make_strategy momentum", make_strategy(mom_cfg).name == "momentum_15")
    mtf_cfg = replace(cfg, strategy="trend_mtf", timeframe="15m", require_btc_regime=True)
    mtf_res = BacktestEngine(prepare_data(raw, mtf_cfg), mtf_cfg).run()
    check(
        f"trend_mtf smoke ({mtf_res.trades} trades, final {mtf_res.final_equity:.2f})",
        math.isfinite(mtf_res.final_equity) and mtf_res.final_equity >= 0,
    )
    check("make_strategy trend_mtf", make_strategy(mtf_cfg).name == "trend_mtf")
    rule_cfg = replace(cfg, strategy="momentum_rule", require_btc_regime=True)
    rule_res = BacktestEngine(prepare_data(raw, rule_cfg), rule_cfg).run()
    check(
        f"momentum_rule smoke ({rule_res.trades} trades, final {rule_res.final_equity:.2f})",
        math.isfinite(rule_res.final_equity) and rule_res.final_equity >= 0,
    )
    check("make_strategy momentum_rule", make_strategy(rule_cfg).name == "momentum_rule")
    # score_raw vs capped display
    from types import SimpleNamespace

    fake_ins = SimpleNamespace(
        expectancy_r=0.5,
        trades=80,
        total_return=1.0,
        max_drawdown=0.1,
        win_rate=0.6,
        final_equity=100.0,
        halted="",
    )
    fake_oos = SimpleNamespace(
        expectancy_r=0.4,
        trades=50,
        total_return=0.5,
        max_drawdown=0.15,
        win_rate=0.55,
        final_equity=75.0,
        halted="",
    )
    g_cap = _grade(
        fake_ins,
        fake_oos,
        fake_oos,
        wf={"ok": True},
        wf_required=True,
        fwd={"ok": False},
        fwd_required=True,
    )
    check("score_raw present", "score_raw" in g_cap)
    check(
        "forward cap applies to display only",
        g_cap["score"] <= FWD_SCORE_CAP and g_cap["score_raw"] >= g_cap["score"],
    )
    # net directional brake
    rsk = RiskEngine(replace(cfg, max_net_leverage=1.0, max_open_positions=8), silent=True)
    rsk.reset(100.0)
    spec = InstrumentSpec(symbol="X", contract_size=1.0, qty_step=0.01, min_qty=0.01)
    # Net long already at the 1.0x cap → no more longs; shorts still OK
    d_long = rsk.evaluate(
        Direction.LONG,
        100.0,
        98.0,
        equity=100.0,
        open_notional=100.0,
        used_margin=20.0,
        open_positions=1,
        spec=spec,
        ts=pd.Timestamp("2025-01-01", tz="UTC"),
        net_notional=100.0,
    )
    d_short = rsk.evaluate(
        Direction.SHORT,
        100.0,
        102.0,
        equity=100.0,
        open_notional=100.0,
        used_margin=20.0,
        open_positions=1,
        spec=spec,
        ts=pd.Timestamp("2025-01-01", tz="UTC"),
        net_notional=100.0,
    )
    check("net-leverage blocks adding with the book", not d_long.approved)
    check("net-leverage still allows opposite side", d_short.approved)
    check(
        "make_strategy breakout",
        make_strategy(replace(cfg, strategy="regime_breakout")).name == "regime_breakout_v7",
    )
    print("ALL PASSED" if ok else "SOME TESTS FAILED")
    return ok
