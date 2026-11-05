"""Persistent OHLCV depth cache keyed by (symbol, timeframe, start, end) — A.4."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from moonshot.paths import ROOT

DEPTH_CACHE_DIR = ROOT / ".moonshot_cache" / "depth"


def cache_key(symbol: str, timeframe: str, start: str, end: str) -> str:
    raw = f"{symbol}|{timeframe}|{start}|{end}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def cache_paths(key: str, *, base: Path | None = None) -> tuple[Path, Path]:
    root = Path(base) if base is not None else DEPTH_CACHE_DIR
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{key}.parquet", root / f"{key}.meta.json"


def put_depth_cache(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    base: Path | None = None,
) -> Path:
    key = cache_key(symbol, timeframe, start, end)
    parquet_path, meta_path = cache_paths(key, base=base)
    try:
        df.to_parquet(parquet_path)
    except (ImportError, ValueError):
        # Engine-free fallback
        parquet_path = parquet_path.with_suffix(".pkl")
        df.to_pickle(parquet_path)
    meta_path.write_text(
        json.dumps(
            {
                "symbol": symbol,
                "timeframe": timeframe,
                "start": start,
                "end": end,
                "rows": int(len(df)),
                "key": key,
            },
            indent=2,
        )
        + "\n"
    )
    return parquet_path


def get_depth_cache(
    *,
    symbol: str,
    timeframe: str,
    start: str,
    end: str,
    base: Path | None = None,
) -> pd.DataFrame | None:
    key = cache_key(symbol, timeframe, start, end)
    parquet_path, meta_path = cache_paths(key, base=base)
    pkl = parquet_path.with_suffix(".pkl")
    path = parquet_path if parquet_path.exists() else pkl if pkl.exists() else None
    if path is None or not meta_path.exists():
        return None
    if path.suffix == ".pkl":
        return pd.read_pickle(path)
    try:
        return pd.read_parquet(path)
    except (ImportError, ValueError):
        return None


def scale_lookback(bars_at_15m: int, timeframe: str) -> int:
    """Scale feature lookbacks from a 15m baseline to 1h/4h (and leave 5m/15m)."""
    tf = str(timeframe).strip().lower()
    if tf in ("5m", "15m"):
        return max(1, int(bars_at_15m))
    # 1h = 4×15m, 4h = 16×15m → shorter lookback in bars for same wall-clock
    mult = {"1h": 4, "4h": 16}.get(tf)
    if mult is None:
        return max(1, int(bars_at_15m))
    return max(2, int(round(bars_at_15m / mult)))


def estimate_fold_trades_4h(
    n_bars_4h: int,
    *,
    n_folds: int = 4,
    signals_per_bar: float = 0.02,
) -> list[int]:
    """Rough per-fold trade count estimate for 4h depth planning (≥30 target)."""
    from moonshot.hunt.fold_runner import fold_boundaries

    folds = fold_boundaries(n_bars_4h, n_folds)
    return [max(0, int((f["end"] - f["start"]) * signals_per_bar)) for f in folds]
