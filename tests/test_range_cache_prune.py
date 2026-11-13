"""Range CSV prune is OHLCV hygiene — not search memory."""

from __future__ import annotations

import time
from pathlib import Path

from moonshot.data import prune_range_cache


def test_prune_keeps_newest_per_stem(tmp_path: Path):
    stem = "binance_spot_BTC_USDT_USDT_1h_range_"
    paths = []
    for i, mid in enumerate((100, 200, 300)):
        p = tmp_path / f"{stem}{mid}_{mid + 50}_w3000.csv"
        p.write_text(f"data{i}")
        # Stagger mtimes so newest is clear.
        time.sleep(0.02)
        paths.append(p)
    # Fixed bar cache must survive.
    keep = tmp_path / "BTC_USDT_USDT_1h_30000.csv"
    keep.write_text("bars")

    out = prune_range_cache(tmp_path, stem_prefix=stem, keep_per_stem=2, max_files=200)
    assert out["removed"] >= 1
    assert keep.exists()
    alive = sorted(tmp_path.glob(f"{stem}*.csv"))
    assert len(alive) == 2
    assert paths[-1].exists()
    assert paths[-2].exists()
    assert not paths[0].exists()


def test_prune_global_max_files(tmp_path: Path):
    for i in range(6):
        p = tmp_path / f"binance_spot_X_1h_range_{i}_{i + 1}_w1.csv"
        p.write_text("x" * 10)
        time.sleep(0.01)
    out = prune_range_cache(tmp_path, max_files=3, max_bytes=10**12, keep_per_stem=99)
    assert out["removed"] >= 3
    assert len(list(tmp_path.glob("*range_*.csv"))) <= 3
