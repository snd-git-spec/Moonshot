"""A.5 — expectancy buckets, legacy separation, cost_share dollar units."""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

from moonshot.paper_expectancy import (
    COST_SHARE_STOP_THRESHOLD,
    bucket_trades,
    bucket_trades_by_hash,
    cost_share_dollars,
    filter_rows_to_pinned,
    load_trade_rows,
    render_report,
)


def test_cost_share_units():
    # R is net; cost_share is dollars only
    rs = [1.0, -0.5]
    risks = [100.0, 100.0]
    fees = [5.0, 5.0]
    funding = [1.0, 1.0]
    # gross_dollar = sum(R*risk) + costs = 50 + 12 = 62; cost_share = 12/62
    cs = cost_share_dollars(rs, risks, fees, funding)
    assert abs(cs - 12.0 / 62.0) < 1e-12
    assert COST_SHARE_STOP_THRESHOLD == 0.40
    # Never treat fees as R addends
    bogus = sum(rs) + sum(fees)  # invalid construct — assert we don't use it
    assert abs(bogus - 10.5) < 1e-12
    assert abs(cs - bogus) > 0.1


def test_legacy_bucket_separate(tmp_path: Path):
    db = tmp_path / "t.db"
    con = sqlite3.connect(db)
    con.execute(
        """CREATE TABLE trades (
            ts TEXT, symbol TEXT, pnl REAL, r_multiple REAL, fees REAL, funding REAL,
            liquidated INTEGER, config_hash TEXT, config_json TEXT, initial_risk_dollars REAL
        )"""
    )
    con.executemany(
        "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?)",
        [
            (
                "2026-01-01",
                "BTC/USDT:USDT",
                10.0,
                1.0,
                1.0,
                0.0,
                0,
                None,
                None,
                100.0,
            ),
            (
                "2026-01-02",
                "BTC/USDT:USDT",
                20.0,
                2.0,
                2.0,
                0.0,
                0,
                "abcd1234",
                '{"timeframe":"5m"}',
                100.0,
            ),
            (
                "2026-01-03",
                "ETH/USDT:USDT",
                -5.0,
                -0.5,
                1.0,
                0.5,
                0,
                "abcd1234",
                '{"timeframe":"5m"}',
                100.0,
            ),
        ],
    )
    con.commit()
    con.close()
    buckets = bucket_trades(load_trade_rows(db))
    legacy = [b for b in buckets if b.legacy]
    hashed = [b for b in buckets if not b.legacy]
    assert len(legacy) == 1 and legacy[0].n == 1
    assert legacy[0].decision_label() == "LEGACY-EXCLUDED"
    assert sum(b.n for b in hashed) == 2
    assert all(b.config_hash == "abcd1234" for b in hashed)


def test_paper_expectancy_exact_values():
    rows = [
        {
            "symbol": "BTC/USDT:USDT",
            "pnl": 10.0,
            "r_multiple": 1.0,
            "fees": 2.0,
            "funding": 0.0,
            "liquidated": 0,
            "config_hash": "hash1",
            "config_json": '{"timeframe":"15m"}',
            "initial_risk_dollars": 50.0,
        },
        {
            "symbol": "BTC/USDT:USDT",
            "pnl": -5.0,
            "r_multiple": -0.5,
            "fees": 1.0,
            "funding": 1.0,
            "liquidated": 0,
            "config_hash": "hash1",
            "config_json": '{"timeframe":"15m"}',
            "initial_risk_dollars": 50.0,
        },
    ]
    b = bucket_trades(rows)[0]
    assert b.n == 2
    assert abs(b.mean_r - 0.25) < 1e-12
    assert abs(b.win_rate - 0.5) < 1e-12
    # gross = 1*50 + (-0.5)*50 + 4 = 25+4=29; cost_share=4/29
    assert abs(b.cost_share - 4.0 / 29.0) < 1e-12
    assert math.isfinite(b.ci_low)
    by_hash = bucket_trades_by_hash(rows)
    assert len(by_hash) == 1
    assert by_hash[0].config_hash == "hash1"
    assert by_hash[0].n == 2
    assert by_hash[0].symbol is None
    assert by_hash[0].decision_label() == "EXTEND-or-WAIT"


def test_filter_rows_to_pinned_excludes_old_tf():
    rows = [
        {
            "config_hash": "pinned",
            "symbol": "BTC/USDT:USDT",
            "r_multiple": 1.0,
            "pnl": 1.0,
            "fees": 0.0,
            "funding": 0.0,
            "liquidated": 0,
            "config_json": '{"timeframe":"1h"}',
            "initial_risk_dollars": 10.0,
        },
        {
            "config_hash": "old5m",
            "symbol": "BTC/USDT:USDT",
            "r_multiple": -1.0,
            "pnl": -1.0,
            "fees": 0.0,
            "funding": 0.0,
            "liquidated": 0,
            "config_json": '{"timeframe":"5m"}',
            "initial_risk_dollars": 10.0,
        },
    ]
    kept, n_ex = filter_rows_to_pinned(rows, "pinned")
    assert n_ex == 1
    assert len(kept) == 1
    assert kept[0]["config_hash"] == "pinned"
    by_hash = bucket_trades_by_hash(kept)
    assert len(by_hash) == 1 and by_hash[0].config_hash == "pinned"
    text = render_report(
        bucket_trades(kept),
        by_hash=by_hash,
        pinned_hash="pinned",
        n_excluded=n_ex,
    )
    assert "old5m" not in text
    assert "Pinned config_hash: `pinned`" in text
    assert "Excluded non-pinned closes" in text
