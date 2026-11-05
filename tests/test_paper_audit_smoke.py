"""0.1a — paper audit smoke against a tiny fixture DB."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from scripts.paper_audit import audit


def test_paper_audit_smoke(tmp_path: Path):
    db = tmp_path / "fixture.db"
    con = sqlite3.connect(str(db))
    con.executescript(
        """
        CREATE TABLE trades (
            ts TEXT, symbol TEXT, direction TEXT, entry REAL, exit REAL, qty REAL,
            pnl REAL, r_multiple REAL, liquidated INTEGER, exit_reason TEXT, adds INTEGER,
            fees REAL, funding REAL, entry_time TEXT, partial_taken INTEGER, mode TEXT
        );
        INSERT INTO trades VALUES
        ('2026-01-01T00:00:00Z','BTC/USDT:USDT','long',100,110,1,10,2.0,0,'tp',0,0.1,0.0,
         '2026-01-01T00:00:00Z',0,'paper'),
        ('2026-01-02T00:00:00Z','ETH/USDT:USDT','short',50,45,2,5,1.0,0,'tp',0,0.0,NULL,
         '2026-01-02T00:00:00Z',0,'paper'),
        ('2026-01-03T00:00:00Z','SOL/USDT:USDT','long',20,10,1,-10,-6.0,0,'sl',0,NULL,0.0,
         '2026-01-03T00:00:00Z',0,'paper');
        """
    )
    con.commit()
    con.close()

    out_dir = tmp_path / "docs"
    path = audit(db, out_dir)
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert "Total trades: **3**" in text
    assert "BTC/USDT:USDT: 1" in text
    assert "fees NULL: 1" in text
    assert "funding NULL: 1" in text
    assert "r_multiple=-6.0" in text
