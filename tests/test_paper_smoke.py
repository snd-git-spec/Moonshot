"""Frozen OHLCV fixture: paper replay + mid-run restart persistence."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path

import pandas as pd

from moonshot.backtest import BacktestEngine
from moonshot.brokers import PaperBroker
from moonshot.config import Config
from moonshot.features import prepare_data
from moonshot.live import LiveRunner
from moonshot.types import InstrumentSpec

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "ohlcv_15m_btc_eth.csv"
SYMBOLS = ("BTC/USDT:USDT", "ETH/USDT:USDT")


def _load_fixture() -> dict[str, pd.DataFrame]:
    raw = pd.read_csv(FIXTURE, parse_dates=["timestamp"])
    out: dict[str, pd.DataFrame] = {}
    for sym, g in raw.groupby("symbol"):
        df = g.set_index("timestamp")[["open", "high", "low", "close", "volume"]].sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        out[str(sym)] = df
    return out


class _FixtureMarket:
    """Offline market for LiveRunner smoke (no network)."""

    def __init__(self, frames: dict[str, pd.DataFrame], cfg: Config):
        self.frames = frames
        self.cfg = cfg

    def instrument(self, symbol: str) -> InstrumentSpec:
        return InstrumentSpec(symbol=symbol, qty_step=0.0, min_qty=0.0, contract_size=1.0)

    def last_price(self, symbol: str) -> float:
        df = self.frames[symbol]
        return float(df.iloc[-1]["close"])

    def recent(self, symbol: str) -> pd.DataFrame:
        return self.frames[symbol].copy()


def _make_runner(tmp_path: Path, frames: dict[str, pd.DataFrame], cfg: Config) -> LiveRunner:
    cfg = replace(
        cfg,
        db_path=str(tmp_path / "smoke.db"),
        require_btc_regime=False,
        live_history_bars=200,
    )
    market = _FixtureMarket(frames, cfg)
    broker = PaperBroker(cfg)
    return LiveRunner(cfg, broker, market, list(SYMBOLS), live=False)


def test_fixture_file_committed() -> None:
    assert FIXTURE.is_file(), f"missing fixture {FIXTURE}"
    assert FIXTURE.stat().st_size > 1000


def test_fixture_backtest_finite(cfg: Config) -> None:
    frames = _load_fixture()
    data = prepare_data(frames, cfg)
    res = BacktestEngine(data, cfg).run()
    assert math.isfinite(res.final_equity)
    assert res.final_equity > 0
    assert len(res.equity_curve) > 10


def test_paper_replay_restart_persists_state(cfg: Config, tmp_path: Path) -> None:
    frames = _load_fixture()
    n = min(len(df) for df in frames.values())
    assert n >= 120
    mid = n // 2

    runner = _make_runner(tmp_path, frames, cfg)
    runner.risk.reset(runner.broker.get_equity())
    runner.replay_offline(frames, start_i=50, end_i=mid)

    runtime_path = runner._runtime_path()
    assert runtime_path.exists()
    rt = json.loads(runtime_path.read_text(encoding="utf-8"))
    assert rt.get("last_cycle_id"), "cycle_id must be persisted"
    assert rt.get("last_cycle_ts"), "last cycle ts must be persisted"
    eq_mid = float(runner.broker.get_equity())
    assert math.isfinite(eq_mid)
    open_syms = sorted(runner.positions)
    bars_mid = int(rt.get("bars_seen") or 0)
    assert bars_mid > 0
    cycle_mid = str(rt["last_cycle_id"])
    ts_mid = str(rt["last_cycle_ts"])
    marks_mid = dict(rt.get("last_mark") or {})

    # Restart: new runner loads DB positions + runtime snapshot
    runner2 = _make_runner(tmp_path, frames, cfg)
    eq0 = runner2.broker.get_equity()
    runner2.risk.reset(eq0)
    runner2._restore_runtime()
    assert runner2._cycle_id == cycle_mid
    assert runner2._last_cycle_ts is not None
    assert pd.Timestamp(runner2._last_cycle_ts).isoformat() == pd.Timestamp(ts_mid).isoformat()
    assert sorted(runner2.positions) == open_syms
    assert runner2._bars_seen == bars_mid

    if runner2.positions:
        assert runner2.broker.equity_stale is True
        for sym in runner2.positions:
            px = float(marks_mid.get(sym) or frames[sym].iloc[mid - 1]["close"])
            runner2.broker.on_price(sym, px)
        assert runner2.broker.equity_stale is False
        assert abs(runner2.broker.get_equity() - eq_mid) < 0.05
    else:
        # No open risk — equity is cash-only and should still match mid snapshot
        assert abs(eq0 - eq_mid) < 0.05 or abs(runner2.broker.get_equity() - eq_mid) < 0.05

    runner2.replay_offline(frames, start_i=mid, end_i=min(n, mid + 80))
    rt2 = json.loads(runner2._runtime_path().read_text(encoding="utf-8"))
    assert rt2.get("last_cycle_ts")
    assert int(rt2.get("bars_seen") or 0) >= bars_mid
    assert math.isfinite(float(runner2.broker.get_equity()))
