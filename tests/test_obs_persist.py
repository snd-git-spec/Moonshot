"""Observability spine — schema migrate + typed writers."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from moonshot.config import Config
from moonshot.obs import hunt_funnel_metrics, kill_histogram, portfolio_lev_metrics
from moonshot.persist import Persistence
from moonshot.position import Position
from moonshot.types import Direction, ExitReason, TradeResult


def test_migrate_creates_obs_tables(tmp_path: Path):
    db = Persistence(str(tmp_path / "m.db"), mode="paper")
    tables = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "cycle_snapshot" in tables
    assert "expectancy_rollup" in tables
    cols = db._cols("trades")
    assert "mae_r" in cols
    assert "open_config_hash" in cols
    risk_cols = db._cols("risk_snapshot")
    assert "effective_lev" in risk_cols
    assert "feed_age_s" in risk_cols
    # Idempotent
    db2 = Persistence(str(tmp_path / "m.db"), mode="paper")
    assert "cycle_snapshot" in {
        r[0] for r in db2.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    db.close()
    db2.close()


def test_cycle_snapshot_and_fill_event(tmp_path: Path):
    db = Persistence(str(tmp_path / "c.db"), mode="paper")
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    db.log_cycle_snapshot(
        ts,
        equity=100.0,
        peak=110.0,
        effective_lev=2.5,
        net_lev=1.0,
        open_symbols=["BTC/USDT:USDT"],
        pause_until=None,
        halt_reason="",
        kill=False,
        config_hash="abc",
        feed_age_s=12.0,
        cycle_id="1-deadbeef",
    )
    row = db.conn.execute(
        "SELECT effective_lev, config_hash, cycle_id FROM cycle_snapshot"
    ).fetchone()
    assert float(row[0]) == 2.5
    assert row[1] == "abc"
    assert row[2] == "1-deadbeef"

    db.log_fill(
        ts,
        "BTC/USDT:USDT",
        side="buy",
        reason="open",
        intended_qty=1.0,
        filled_qty=1.0,
        ref_price=100.0,
        fill_price=100.03,
        slippage_bps=3.0,
        assumed_slippage_bps=3.0,
    )
    kinds = [r[0] for r in db.conn.execute("SELECT kind FROM events").fetchall()]
    assert "fill" in kinds
    db.close()


def test_trade_obs_columns_roundtrip(tmp_path: Path):
    db = Persistence(str(tmp_path / "t.db"), mode="paper")
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    tr = TradeResult(
        symbol="BTC/USDT:USDT",
        direction=Direction.LONG,
        entry_time=ts,
        exit_time=ts,
        entry_price=100.0,
        exit_price=105.0,
        initial_qty=1.0,
        pnl=4.5,
        r_multiple=1.5,
        liquidated=False,
        exit_reason=ExitReason.STOP,
        max_adds=0,
        partial_taken=False,
        fees=0.1,
        funding=0.0,
        initial_risk_dollars=3.0,
        config_hash="closehash",
        open_config_hash="openhash",
        intended_qty=1.0,
        filled_qty=1.0,
        slippage_bps=3.0,
        assumed_slippage_bps=3.0,
        funding_rate_entry=0.0001,
        mae_r=-0.4,
        mfe_r=1.8,
        bars_held=12,
    )
    db.log_trade(tr)
    row = db.conn.execute(
        "SELECT open_config_hash, mae_r, mfe_r, bars_held, funding_rate_entry FROM trades"
    ).fetchone()
    assert row[0] == "openhash"
    assert float(row[1]) == -0.4
    assert float(row[2]) == 1.8
    assert int(row[3]) == 12
    assert float(row[4]) == 0.0001
    db.close()


def test_position_excursion_and_close_anatomy():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pos = Position.open("BTC/USDT:USDT", Direction.LONG, 5, 100.0, 1.0, 98.0, ts, 0.02)
    pos.open_config_hash = "h1"
    pos.funding_rate_entry = 0.0001
    pos.intended_qty = 1.0
    pos.entry_slippage_bps = 3.0
    pos.assumed_slippage_bps = 3.0
    pos.update_excursion(99.0)  # -0.5R
    pos.update_excursion(104.0)  # +2R
    pos.bars_held(pd_ts := __import__("pandas").Timestamp("2026-01-01 02:00", tz="UTC"), __import__("pandas").Timedelta("1h"))
    res, _ = pos.close_out(103.0, 0.02, ts, ExitReason.TRAILING)
    assert res.open_config_hash == "h1"
    assert res.mae_r == -0.5
    assert res.mfe_r == 2.0
    assert res.funding_rate_entry == 0.0001
    assert res.bars_held == 2


def test_portfolio_lev_and_funnel_helpers():
    ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pos = Position.open("ETH/USDT:USDT", Direction.LONG, 10, 2000.0, 0.1, 1900.0, ts, 0.1)
    m = portfolio_lev_metrics({"ETH/USDT:USDT": pos}, {"ETH/USDT:USDT": 2000.0}, equity=100.0)
    assert m["effective_lev"] == 2.0  # 200 notional / 100 equity
    assert m["margin_used"] == 20.0
    funnel = hunt_funnel_metrics(
        scout_in=100,
        scout_keep=20,
        confirm_b=10,
        confirm_hits=4,
        confirm_rejects={"dual": 2},
        matrix_rows=[{"skip": False, "why": ""}, {"skip": True, "why": "sparse"}],
        finalist_ok=True,
        gates_pass=True,
        promote=False,
    )
    assert funnel["scout_in"] == 100
    assert funnel["matrix_scored"] == 1
    assert funnel["promote"] == 0
    hist = kill_histogram(
        [{"skip": True, "why": "sparse"}, {"skip": False, "why": "period_robust"}],
        {"climb": 3},
        material_why="2x cost",
    )
    assert hist["sparse"] == 1
    assert hist["confirm_climb"] == 3


def test_paper_broker_fill_slippage_fields(cfg: Config):
    from moonshot.brokers import PaperBroker

    pb = PaperBroker(cfg)
    fill = pb.market_order("BTC/USDT:USDT", "buy", 0.01, 100.0)
    assert fill.intended_qty == 0.01
    assert fill.ref_price == 100.0
    assert fill.assumed_slippage_bps == cfg.slippage_bps
    assert fill.slippage_bps is not None
    assert fill.slippage_bps > 0
