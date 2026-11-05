"""A.7/A.8 — discover hashes queryable; barrier parity; signal mask reuse."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from moonshot.config import Config
from moonshot.discover_fast import (
    barrier_outcomes_vectorized,
    clear_signal_mask_cache,
    data_hash_from_frame,
    signal_mask,
)
from moonshot.experiment_log import get_runs, log_run


def _tiny_ohlcv(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-01", periods=n, freq="4h", tz="UTC")
    close = 100 + np.cumsum(rng.normal(0, 0.5, size=n))
    high = close + rng.uniform(0.1, 1.0, size=n)
    low = close - rng.uniform(0.1, 1.0, size=n)
    open_ = close + rng.normal(0, 0.2, size=n)
    vol = rng.uniform(10, 100, size=n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


def test_barrier_outcomes_matches_reference():
    import discover as disc

    df = _tiny_ohlcv(120)
    cfg = Config()
    hold = 8
    ref_l, ref_s = disc.barrier_outcomes_reference(df, hold, cfg)
    fast_l, fast_s = disc.barrier_outcomes(df, hold, cfg)
    # Compare where both finite
    m = np.isfinite(ref_l) & np.isfinite(fast_l)
    assert m.sum() > 10
    assert np.allclose(ref_l[m], fast_l[m], rtol=0, atol=1e-9)
    m2 = np.isfinite(ref_s) & np.isfinite(fast_s)
    assert np.allclose(ref_s[m2], fast_s[m2], rtol=0, atol=1e-9)


def test_signal_mask_reuse_identical():
    clear_signal_mask_cache()
    a = np.array([True, False, True, True])
    b = np.array([True, True, False, True])
    atoms = {"x": ("f", a), "y": ("f", b)}
    m1 = signal_mask(atoms, ("x", "y"), cache=True)
    m2 = signal_mask(atoms, ("x", "y"), cache=True)
    assert m1 is m2
    assert np.array_equal(m1, a & b)


def test_discover_output_queryable(tmp_path: Path):
    df = _tiny_ohlcv(40)
    dh = data_hash_from_frame(df)
    assert len(dh) == 16
    db = tmp_path / "exp.db"
    log_run(
        run_id="discover_test",
        kind="discover",
        config_hash="cfghash12",
        data_hash=dh,
        metrics={"real_pass": 0},
        db_path=db,
    )
    rows = get_runs({"kind": "discover", "data_hash": dh}, db_path=db)
    assert len(rows) == 1
    assert rows[0]["config_hash"] == "cfghash12"
