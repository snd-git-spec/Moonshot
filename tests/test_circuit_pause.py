"""Circuit pause: configurable duration + survives restart snapshot."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from moonshot.config import Config
from moonshot.risk import RiskEngine


def test_kill_resume_hours_configurable() -> None:
    r = RiskEngine(
        Config(
            max_peak_drawdown_pct=0.20,
            kill_resume_hours=12,
            max_daily_loss_pct=0.99,
        ),
        silent=True,
    )
    t0 = pd.Timestamp("2025-03-01", tz="UTC")
    r.update_equity(100.0, t0)
    r.update_equity(79.0, t0 + pd.Timedelta(hours=1))
    assert r.pause_until is not None
    # 12h pause from trip time
    assert abs((r.pause_until - (t0 + pd.Timedelta(hours=1))).total_seconds() - 12 * 3600) < 1


def test_pause_survives_snapshot_reload(tmp_path: Path) -> None:
    cfg = Config(
        max_peak_drawdown_pct=0.20,
        kill_resume_hours=48,
        max_daily_loss_pct=0.99,
    )
    r = RiskEngine(cfg, silent=True)
    t0 = pd.Timestamp("2025-04-01", tz="UTC")
    r.update_equity(100.0, t0)
    r.update_equity(70.0, t0 + pd.Timedelta(hours=2))
    assert not r.kill_switch_active
    assert r.pause_until is not None
    snap = {
        "pause_until": r.pause_until.isoformat(),
        "halt_reason": r.halt_reason,
        "circuit_trips": list(r.circuit_trips),
        "peak_equity": r.peak_equity,
        "rolling_r": list(r.rolling_r),
    }
    path = tmp_path / "rt.json"
    path.write_text(json.dumps(snap))

    # New engine after "restart"
    data = json.loads(path.read_text())
    r2 = RiskEngine(cfg, silent=True)
    r2.equity = 70.0
    r2.peak_equity = float(data["peak_equity"])
    r2.pause_until = pd.Timestamp(data["pause_until"])
    r2.halt_reason = str(data["halt_reason"])
    r2.circuit_trips = list(data["circuit_trips"])

    mid = r2.pause_until - pd.Timedelta(hours=1)
    assert not r2.trading_allowed(mid)
    later = r2.pause_until + pd.Timedelta(minutes=1)
    assert r2.trading_allowed(later)
    assert r2.pause_until is None
