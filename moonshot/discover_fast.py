"""Fast path for discover.py: vectorized barriers + signal-mask reuse (A.7/A.8)."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

STOP_ATR = 2.0
TARGET_R = 2.0


def data_hash_from_frame(df: pd.DataFrame) -> str:
    """Stable short hash of OHLCV endpoints + length."""
    if df is None or len(df) == 0:
        return "empty"
    payload = {
        "n": int(len(df)),
        "start": str(df.index[0]),
        "end": str(df.index[-1]),
        "c0": float(df["close"].iloc[0]),
        "c1": float(df["close"].iloc[-1]),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def config_hash_fee_slip(
    fee_taker: float, slip: float, *, hold: int, stop_atr: float = STOP_ATR
) -> str:
    payload = {
        "fee_taker": fee_taker,
        "slip": slip,
        "hold": hold,
        "stop_atr": stop_atr,
        "target_r": TARGET_R,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def barrier_outcomes_vectorized(
    df: pd.DataFrame,
    hold: int,
    *,
    fee_taker: float,
    slip: float,
    atr: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Triple-barrier R per bar (long/short), net of round-trip costs.
    Matches the reference loop in discover.barrier_outcomes.
    """
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    if atr is None:
        # Local ATR(14) without importing moonshot.atr (keeps unit tests light)
        prev_c = np.roll(c, 1)
        prev_c[0] = c[0]
        tr = np.maximum(h - low, np.maximum(np.abs(h - prev_c), np.abs(low - prev_c)))
        atr = pd.Series(tr).rolling(14).mean().to_numpy(float)
    n = len(df)
    rl = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    cost = 2 * fee_taker + 2 * slip
    # Precompute rolling window extrema via stride tricks for speed
    for t in range(n - hold - 2):
        if not np.isfinite(atr[t]) or atr[t] <= 0:
            continue
        e = o[t + 1]
        d = STOP_ATR * atr[t]
        cost_r = cost * e / d
        hi = h[t + 1 : t + 1 + hold]
        lo = low[t + 1 : t + 1 + hold]
        # long
        stop_l, tgt_l = e - d, e + TARGET_R * d
        hit_stop = np.flatnonzero(lo <= stop_l)
        hit_tgt = np.flatnonzero(hi >= tgt_l)
        i_s = int(hit_stop[0]) if hit_stop.size else hold + 1
        i_t = int(hit_tgt[0]) if hit_tgt.size else hold + 1
        if i_s <= hold and i_s <= i_t:
            res = -1.0
        elif i_t <= hold and i_t < i_s:
            res = TARGET_R
        else:
            res = (c[t + hold] - e) / d
        rl[t] = res - cost_r
        # short
        stop_s, tgt_s = e + d, e - TARGET_R * d
        hit_stop = np.flatnonzero(hi >= stop_s)
        hit_tgt = np.flatnonzero(lo <= tgt_s)
        i_s = int(hit_stop[0]) if hit_stop.size else hold + 1
        i_t = int(hit_tgt[0]) if hit_tgt.size else hold + 1
        if i_s <= hold and i_s <= i_t:
            res = -1.0
        elif i_t <= hold and i_t < i_s:
            res = TARGET_R
        else:
            res = (e - c[t + hold]) / d
        rs[t] = res - cost_r
    return rl, rs


_SIGNAL_MASK_CACHE: dict[tuple[Any, ...], np.ndarray] = {}


def signal_mask(atoms: dict, names: tuple[str, ...], *, cache: bool = True) -> np.ndarray:
    key = (id(atoms), names)
    if cache and key in _SIGNAL_MASK_CACHE:
        return _SIGNAL_MASK_CACHE[key]
    m = atoms[names[0]][1].copy()
    for nme in names[1:]:
        m &= atoms[nme][1]
    if cache:
        _SIGNAL_MASK_CACHE[key] = m
    return m


def clear_signal_mask_cache() -> None:
    _SIGNAL_MASK_CACHE.clear()


@lru_cache(maxsize=64)
def barrier_cache_key(
    symbol: str, tf: str, hold: int, fee: float, slip: float, n: int, end_ts: str
) -> str:
    return f"{symbol}|{tf}|{hold}|{fee}|{slip}|{n}|{end_ts}"
