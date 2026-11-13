from __future__ import annotations

import logging
import threading
import time
import zlib
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    DataError,
    _pandas_tf,
)

from moonshot.paths import CACHE_DIR
import contextlib


_TF_UNITS = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}


def timeframe_ms(tf: str) -> int:
    return int(tf[:-1]) * _TF_UNITS[tf[-1]]


def live_history_bars_for_tf(tf: str, *, days: float = 70.0) -> int:
    """Bars needed so daily EMA50 (+ buffer) can warm on this TF.

    Comment historically assumed 4h bars (1200 ≈ 200d). On 15m that is only ~12d
    and leaves regime stuck at ``unknown`` → every paper exit is regime_invalid.
    """
    minutes = max(1, int(timeframe_ms(str(tf).strip().lower()) // 60_000))
    n = int(max(2000, (float(days) * 24.0 * 60.0) / minutes))
    return int(min(n, 20_000))


# TFs some venues lack natively → build from a finer base TF.
# MEXC swap: no 3m, no 2h. Binance spot: has native 2h (see _resample_base_tf).
_RESAMPLE_BASE = {"3m": "1m", "2h": "1h"}
_RESAMPLE_FROM_1M = frozenset(tf for tf, base in _RESAMPLE_BASE.items() if base == "1m")


def _resample_base_tf(tf: str, *, source: str = "mexc") -> Optional[str]:
    """Return base TF to resample from, or None if ``tf`` is fetched natively."""
    tf = str(tf).strip().lower()
    src = str(source or "mexc").strip().lower()
    # Binance spot exposes native 2h klines.
    if tf == "2h" and src == "binance_spot":
        return None
    return _RESAMPLE_BASE.get(tf)


def _ohlcv_resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Aggregate finer OHLCV up to `tf` (e.g. 1m→3m, 1h→2h)."""
    if df is None or df.empty:
        return df
    rule = _pandas_tf(tf)
    out = (
        df.resample(rule, label="left", closed="left")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
        .dropna(subset=["open", "high", "low", "close"])
    )
    return out


def paged_ohlcv(
    ex: Any, symbol: str, tf: str, since_ms: int, until_ms: int, page: int = 1000
) -> list:
    """Paginate fetch_ohlcv forward from since_ms (exchanges cap one call).

    If ``since_ms`` is before the market listed, MEXC returns ``[]`` on the
    first page — probe forward (and finally with no ``since``) so newer coins
    (SUI, WIF, …) still load whatever history exists.
    """
    rows: list = []
    step = timeframe_ms(tf)
    since = int(since_ms)
    last_seen = -1
    empty_streak = 0
    while since < until_ms:
        batch = ex.fetch_ohlcv(symbol, timeframe=tf, since=since, limit=page)
        if not batch:
            empty_streak += 1
            # Pre-listing / gap: jump forward instead of giving up immediately.
            if not rows and empty_streak <= 12:
                # Exponential jump toward now (covers multi-year pre-list windows).
                jump = max(step * page, (until_ms - since) // 4)
                since = min(until_ms - step, since + max(jump, step * 24))
                time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
                continue
            if not rows:
                # Last resort: latest page with no since, then walk… we only
                # need a foothold; caller trims to the requested window.
                batch = ex.fetch_ohlcv(symbol, timeframe=tf, limit=page)
                if not batch:
                    break
                # Restart pagination from the earliest bar in this foothold
                # by walking backward isn't supported — take foothold and
                # continue forward from there (newer side of history).
                rows.extend(batch)
                last_seen = batch[-1][0]
                since = last_seen + step
                empty_streak = 0
                time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
                continue
            break
        empty_streak = 0
        rows.extend(batch)
        newest = batch[-1][0]
        if newest <= last_seen:
            break
        last_seen = newest
        since = newest + step
        time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
    return rows


def ohlcv_to_frame(rows: list, tf: str) -> pd.DataFrame:
    if not rows:
        raise DataError("no OHLCV rows returned")
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.drop_duplicates("timestamp").set_index("timestamp").sort_index().astype(float)
    # drop a still-forming candle
    now = pd.Timestamp.now(tz="UTC")
    return df[df.index + pd.Timedelta(milliseconds=timeframe_ms(tf)) <= now]


# Backtest-only price venues (live/paper/hunt always use MEXC via public_exchange).
RANGE_SOURCE_MEXC = "mexc"
RANGE_SOURCE_BINANCE_SPOT = "binance_spot"
RANGE_SOURCE_YAHOO = "yahoo"
RANGE_PRICE_SOURCES = (
    RANGE_SOURCE_MEXC,
    RANGE_SOURCE_BINANCE_SPOT,
    RANGE_SOURCE_YAHOO,
)
RANGE_SOURCE_LABELS = {
    RANGE_SOURCE_MEXC: "MEXC swap (live venue)",
    RANGE_SOURCE_BINANCE_SPOT: "Binance spot",
    RANGE_SOURCE_YAHOO: "Yahoo Finance (daily)",
}


def _normalize_range_source(source: Optional[str]) -> str:
    s = str(source or RANGE_SOURCE_MEXC).strip().lower()
    if s not in RANGE_PRICE_SOURCES:
        raise DataError(
            f"unknown price source {source!r}; expected one of {', '.join(RANGE_PRICE_SOURCES)}"
        )
    return s


def _book_base(symbol: str) -> str:
    """BTC from BTC/USDT:USDT or BTC/USDT."""
    return str(symbol).split("/")[0].strip().upper()


def to_binance_spot_symbol(symbol: str) -> str:
    """Map book swap symbol → Binance spot USDT pair."""
    base = _book_base(symbol)
    return f"{base}/USDT"


def to_yahoo_ticker(symbol: str) -> str:
    """Map book symbol → Yahoo crypto ticker (spot USD)."""
    base = _book_base(symbol)
    # Yahoo uses standard BASE-USD for majors and most alts we trade.
    return f"{base}-USD"


def public_exchange() -> Any:
    """MEXC USDT-M — live, paper, hunt, and default Backtest source."""
    try:
        import ccxt  # type: ignore
    except ImportError as e:
        raise DataError("ccxt not installed: pip install ccxt") from e
    return ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def exchange_for_source(source: str = RANGE_SOURCE_MEXC) -> Any:
    """ccxt public client for a Backtest range source (MEXC / Binance spot)."""
    src = _normalize_range_source(source)
    if src == RANGE_SOURCE_YAHOO:
        raise DataError("Yahoo Finance is not a ccxt exchange — use fetch_yahoo_ohlcv_range")
    try:
        import ccxt  # type: ignore
    except ImportError as e:
        raise DataError("ccxt not installed: pip install ccxt") from e
    if src == RANGE_SOURCE_BINANCE_SPOT:
        return ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "spot"}})
    return ccxt.mexc({"enableRateLimit": True, "options": {"defaultType": "swap"}})


def _ccxt_symbol_for_source(symbol: str, source: str) -> str:
    src = _normalize_range_source(source)
    if src == RANGE_SOURCE_BINANCE_SPOT:
        return to_binance_spot_symbol(symbol)
    return symbol


# Serialize OHLCV fetches so hunt + paper + backtest do not stampede exchanges.
_OHLCV_LOCK = threading.Lock()


def _cache_stem(symbol: str, tf: str, *, source: str = RANGE_SOURCE_MEXC) -> str:
    src = _normalize_range_source(source)
    return f"{src}_{symbol.replace('/', '_').replace(':', '_')}_{tf}_"


def _read_ohlcv_csv(p: Path) -> pd.DataFrame:
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


def _ohlcv_is_stale(df: pd.DataFrame, tf: str, *, max_age_bars: float = 1.0) -> bool:
    """True if last closed bar is behind the wall-clock expected closed bar.

    Default max_age_bars=1 means anything older than the latest closed candle is
    stale (forces tail refresh). A looser value was leaving paper on 21:45 while
    22:00 had already closed — bars froze and setups never fired.
    """
    if df is None or df.empty:
        return True
    step = pd.Timedelta(milliseconds=timeframe_ms(tf))
    last = pd.Timestamp(df.index[-1])
    if last.tzinfo is None:
        last = last.tz_localize("UTC")
    expected = pd.Timestamp.now(tz="UTC").floor(_pandas_tf(tf)) - step
    slack = step * max(0.0, float(max_age_bars) - 1.0)
    return last < expected - slack


def _tail_merge_ohlcv(
    df: pd.DataFrame, symbol: str, tf: str, bars: int, ex: Optional[Any] = None
) -> pd.DataFrame:
    """Fetch candles after the cached last bar and merge; drop forming candle."""
    if df is None or df.empty:
        return df
    step_ms = timeframe_ms(tf)
    last_ms = int(pd.Timestamp(df.index[-1]).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    since_ms = last_ms + step_ms
    if since_ms >= now_ms:
        return df
    exchange = ex or public_exchange()
    with _OHLCV_LOCK:
        rows = paged_ohlcv(exchange, symbol, tf, since_ms, now_ms)
        if not rows:
            time.sleep(0.35)
            rows = paged_ohlcv(exchange, symbol, tf, since_ms, now_ms)
    if not rows:
        return df
    fresh = ohlcv_to_frame(rows, tf)
    if fresh.empty:
        return df
    out = pd.concat([df, fresh]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    now = pd.Timestamp.now(tz="UTC")
    out = out[out.index + pd.Timedelta(milliseconds=step_ms) <= now]
    return out.tail(max(int(bars), len(out)))


def clear_stale_live_caches(
    cache_dir: str = str(CACHE_DIR), tf: str = "15m", bars: int = 1200
) -> int:
    """Delete live-sized CSV caches whose last bar is stale (unblocks paper)."""
    cache = Path(cache_dir)
    if not cache.exists():
        return 0
    removed = 0
    for p in cache.glob(f"*_{tf}_{int(bars)}.csv"):
        try:
            df = _read_ohlcv_csv(p)
            if _ohlcv_is_stale(df, tf):
                p.unlink(missing_ok=True)
                removed += 1
        except Exception:
            try:
                p.unlink(missing_ok=True)
                removed += 1
            except Exception:
                pass
    if removed:
        log.info("cleared %d stale %s×%d cache file(s)", removed, tf, bars)
    return removed


def fetch_historical(
    symbol: str,
    tf: str = "4h",
    bars: int = 3000,
    cache_dir: str = str(CACHE_DIR),
    refresh: bool = False,
) -> pd.DataFrame:
    """Paginated history with a CSV cache + incremental tail refresh when stale.

    TFs missing on MEXC contracts (``3m``, ``2h``) are resampled from a finer base.
    """
    tf = str(tf).strip().lower()
    base_tf = _resample_base_tf(tf, source="mexc")
    if base_tf is not None:
        ratio = max(1, int(timeframe_ms(tf) // timeframe_ms(base_tf)))
        src_bars = int(bars) * ratio + ratio  # +1 bucket of slack
        raw = fetch_historical(
            symbol,
            base_tf,
            src_bars,
            cache_dir=cache_dir,
            refresh=refresh,
        )
        if raw is None or raw.empty:
            raise DataError(f"no {base_tf} history to build {tf} for {symbol}")
        built = _ohlcv_resample(raw, tf)
        if built.empty:
            raise DataError(f"resample to {tf} produced empty frame for {symbol}")
        cache = Path(cache_dir)
        cache.mkdir(exist_ok=True)
        path = cache / f"{_cache_stem(symbol, tf)}{int(bars)}.csv"
        out = built.tail(int(bars)) if len(built) > int(bars) else built
        with contextlib.suppress(Exception):
            out.to_csv(path)
        log.info(
            "%s %s built from %s (%d → %d bars)",
            symbol,
            tf,
            base_tf,
            len(raw),
            len(out),
        )
        return out

    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    stem = _cache_stem(symbol, tf)
    path = cache / f"{stem}{bars}.csv"

    def _sibling_frames(min_n: int = 0) -> list[tuple[int, Path, pd.Timestamp]]:
        """Return (n, path, last_ts) sorted by newest last bar, then larger n."""
        found: list[tuple[int, Path, pd.Timestamp]] = []
        for p in cache.glob(f"{stem}*.csv"):
            try:
                n = int(p.stem.rsplit("_", 1)[-1])
            except ValueError:
                continue
            if n < int(min_n):
                continue
            try:
                df_s = _read_ohlcv_csv(p)
                if df_s.empty:
                    continue
                last = pd.Timestamp(df_s.index[-1])
                found.append((n, p, last))
            except Exception:
                continue
        found.sort(key=lambda t: (t[2], t[0]), reverse=True)
        return found

    def _persist(df: pd.DataFrame) -> pd.DataFrame:
        out = df.tail(int(bars)) if len(df) > int(bars) else df
        with contextlib.suppress(Exception):
            out.to_csv(path)
        return out

    def _ensure_fresh(df: pd.DataFrame) -> pd.DataFrame:
        if df is None or df.empty:
            return df
        if not _ohlcv_is_stale(df, tf):
            return df.tail(int(bars)) if len(df) > int(bars) else df
        try:
            merged = _tail_merge_ohlcv(df, symbol, tf, bars)
            if merged is not None and not merged.empty:
                return _persist(merged)
        except Exception as e:
            log.warning("%s %s tail refresh failed: %s", symbol, tf, e)
        return df.tail(int(bars)) if len(df) > int(bars) else df

    # Exact cache hit — still tail-refresh if stale so paper never freezes.
    if path.exists() and not refresh:
        try:
            cached = _read_ohlcv_csv(path)
            if len(cached) > 0:
                return _ensure_fresh(cached)
        except Exception:
            pass

    # Prefer newest same-TF sibling (not smallest n) then tail-refresh.
    if not refresh:
        siblings = _sibling_frames(min_n=max(200, int(bars) // 4))
        for n, p, _last in siblings:
            try:
                df = _read_ohlcv_csv(p)
                if df.empty:
                    continue
                if n >= int(bars):
                    df = df.tail(int(bars))
                return _ensure_fresh(df if n >= int(bars) else _persist(_ensure_fresh(df)))
            except Exception:
                continue

    with _OHLCV_LOCK:
        ex = public_exchange()
        now_ms = int(time.time() * 1000)
        since_ms = now_ms - int(bars) * timeframe_ms(tf)
        rows = paged_ohlcv(ex, symbol, tf, since_ms, now_ms)
        if not rows:
            time.sleep(0.5)
            rows = paged_ohlcv(ex, symbol, tf, since_ms, now_ms)

    if not rows:
        shorter = _sibling_frames(min_n=200)
        if shorter:
            n, p, _ = shorter[0]
            df = _read_ohlcv_csv(p)
            if len(df) > 0:
                log.warning(
                    "%s %s: only %d cached bars (asked %d) — using available history",
                    symbol,
                    tf,
                    len(df),
                    bars,
                )
                return _ensure_fresh(df)
        raise DataError(
            f"no OHLCV rows returned ({symbol} {tf}×{bars}). "
            "Exchange returned empty candles — retry, or check symbol/TF."
        )

    df = ohlcv_to_frame(rows, tf)
    if df.empty:
        raise DataError(f"OHLCV empty after candle filter ({symbol} {tf})")
    if len(df) < int(bars) * 0.5:
        log.warning(
            "%s %s: exchange only returned %d/%d bars (coin listed later than window)",
            symbol,
            tf,
            len(df),
            bars,
        )
    return _persist(df)


# Hard cap for calendar-range backtests (per symbol, including warmup).
_MAX_RANGE_BARS = 25_000
MAX_RANGE_BARS = _MAX_RANGE_BARS

# Fine → coarse. Auto picks the densest TF whose estimate fits under the bar cap.
RANGE_TF_LADDER = ("1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "1d")


def _as_utc_ts(value) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def range_warmup_bars(tf: str) -> int:
    """Warmup bar count used by calendar-range fetches for ``tf``."""
    return max(500, min(int(live_history_bars_for_tf(tf)), 3_000))


def estimate_range_bars(tf: str, start, end, *, warmup_bars: int = 0) -> int:
    """Approximate OHLCV rows for ``[start - warmup, end)`` at timeframe ``tf``."""
    t0 = _as_utc_ts(start)
    t1 = _as_utc_ts(end)
    if not (t0 < t1):
        return 0
    warm = int(warmup_bars) if int(warmup_bars) > 0 else range_warmup_bars(tf)
    step_ms = timeframe_ms(str(tf).strip().lower())
    fetch_start = t0 - pd.Timedelta(milliseconds=step_ms * warm)
    fetch_end = min(t1, pd.Timestamp.now(tz="UTC"))
    if fetch_start >= fetch_end:
        return 0
    return int((fetch_end - fetch_start).total_seconds() * 1000 // step_ms) + 2


def estimate_range_fetch_bars(
    tf: str,
    start,
    end,
    *,
    source: str = RANGE_SOURCE_MEXC,
    warmup_bars: int = 0,
) -> int:
    """Bars that must actually be downloaded (native/base TF if ``tf`` is resampled).

    MEXC has no native 3m — Auto used to accept ~17k of 3m then die fetching
    ~52k of 1m. Cap checks must use this fetch cost, not the post-resample count.
    """
    tf = str(tf).strip().lower()
    src = _normalize_range_source(source)
    warm = int(warmup_bars) if int(warmup_bars) > 0 else range_warmup_bars(tf)
    base = _resample_base_tf(tf, source=src)
    if base is None:
        return estimate_range_bars(tf, start, end, warmup_bars=warm)
    ratio = max(1, int(timeframe_ms(tf) // timeframe_ms(base)))
    return estimate_range_bars(base, start, end, warmup_bars=warm * ratio)


def pick_range_timeframe(
    start,
    end,
    preferred: str,
    *,
    max_bars: int = _MAX_RANGE_BARS,
    source: str = RANGE_SOURCE_MEXC,
) -> tuple[str, int, str]:
    """Pick densest TF at or finer than ``preferred`` that fits ``max_bars``.

    Never silently coarsens past the book TF — a coarser candle size is a
    different strategy. If the book window is over the fetch cap, returns
    ``preferred`` with an over-cap note so the UI can refuse the run.
    Returns ``(timeframe, estimated_result_bars, note)``.
    """
    preferred = str(preferred).strip().lower() or "15m"
    src = _normalize_range_source(source)
    ladder = list(RANGE_TF_LADDER)
    if preferred not in ladder:
        ladder = sorted(set(ladder + [preferred]), key=lambda t: timeframe_ms(t))
    pref_ms = timeframe_ms(preferred)
    # Fine→coarse, but never coarser than the book.
    candidates = sorted(
        {tf for tf in ladder if timeframe_ms(tf) <= pref_ms},
        key=lambda t: timeframe_ms(t),
    )
    if not candidates:
        candidates = [preferred]

    rejected: list[tuple[str, int, int]] = []
    for tf in candidates:
        warm = range_warmup_bars(tf)
        n_out = estimate_range_bars(tf, start, end, warmup_bars=warm)
        n_fetch = estimate_range_fetch_bars(
            tf,
            start,
            end,
            source=src,
            warmup_bars=warm,
        )
        if 0 < n_fetch <= int(max_bars) and n_out > 0:
            note = ""
            if tf != preferred:
                why = ""
                if rejected:
                    prev_tf, prev_out, prev_fetch = rejected[-1]
                    base = _resample_base_tf(prev_tf, source=src)
                    if base and prev_fetch > int(max_bars):
                        why = (
                            f" {prev_tf} needs ~{prev_fetch:,} {base} bars to build "
                            f"(over cap {int(max_bars):,})."
                        )
                    else:
                        why = f" {prev_tf} would be ~{prev_out:,} bars (over cap)."
                note = (
                    f"Auto packed denser candles {preferred} → {tf} "
                    f"(~{n_out:,} bars in this window, fetch cap {int(max_bars):,})."
                    f"{why}"
                )
            return tf, n_out, note
        if n_fetch > int(max_bars) or n_out > int(max_bars):
            rejected.append((tf, n_out, n_fetch))

    warm = range_warmup_bars(preferred)
    n = estimate_range_bars(preferred, start, end, warmup_bars=warm)
    n_fetch = estimate_range_fetch_bars(
        preferred,
        start,
        end,
        source=src,
        warmup_bars=warm,
    )
    note = (
        f"Book {preferred} needs ~{max(n, n_fetch):,} download bars "
        f"(cap {int(max_bars):,}). Shorten the date range — "
        "Auto will not switch to a coarser timeframe."
    )
    return preferred, n, note


def fetch_yahoo_ohlcv_range(
    symbol: str,
    fetch_start: pd.Timestamp,
    fetch_end: pd.Timestamp,
) -> pd.DataFrame:
    """Download daily spot OHLCV from Yahoo Finance for a book symbol.

    Returns the same columns/index shape as ``ohlcv_to_frame`` (UTC daily bars).
    """
    try:
        import yfinance as yf  # type: ignore
    except ImportError as e:
        raise DataError("yfinance not installed: pip install yfinance") from e

    ticker = to_yahoo_ticker(symbol)
    # yfinance end is exclusive-ish; pad one day so the last calendar day arrives.
    start_s = fetch_start.tz_convert("UTC").strftime("%Y-%m-%d")
    end_s = (fetch_end.tz_convert("UTC") + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    try:
        raw = yf.download(
            ticker,
            start=start_s,
            end=end_s,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as e:
        raise DataError(f"Yahoo download failed for {ticker} ({symbol}): {e}") from e

    if raw is None or raw.empty:
        raise DataError(
            f"no Yahoo daily bars for {symbol} (ticker {ticker}). "
            "This coin may not list on Yahoo Finance."
        )

    # yfinance may return MultiIndex columns when ticker is a single string.
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [str(c[0]).strip().lower() for c in raw.columns]
    else:
        raw.columns = [str(c).strip().lower() for c in raw.columns]

    need = {"open", "high", "low", "close"}
    missing = need - set(raw.columns)
    if missing:
        raise DataError(f"Yahoo response for {ticker} missing columns: {sorted(missing)}")

    vol_col = "volume" if "volume" in raw.columns else None
    df = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["open"], errors="coerce"),
            "high": pd.to_numeric(raw["high"], errors="coerce"),
            "low": pd.to_numeric(raw["low"], errors="coerce"),
            "close": pd.to_numeric(raw["close"], errors="coerce"),
            "volume": (pd.to_numeric(raw[vol_col], errors="coerce") if vol_col else 0.0),
        }
    )
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.dropna(subset=["open", "high", "low", "close"]).sort_index()
    # Drop still-forming daily candle (same rule as ohlcv_to_frame).
    now = pd.Timestamp.now(tz="UTC")
    df = df[df.index + pd.Timedelta(days=1) <= now]
    df = df[(df.index >= fetch_start) & (df.index < fetch_end)]
    if df.empty:
        raise DataError(
            f"no Yahoo daily bars for {symbol} (ticker {ticker}) "
            f"in {fetch_start:%Y-%m-%d}→{fetch_end:%Y-%m-%d}"
        )
    return df.astype(float)


def probe_earliest_ohlcv(
    symbol: str,
    tf: str,
    *,
    probe_from: str = "2014-01-01",
    source: str = RANGE_SOURCE_MEXC,
) -> Optional[pd.Timestamp]:
    """Find the earliest completed candle the active source returns for ``symbol``/``tf``."""
    src = _normalize_range_source(source)
    tf = str(tf).strip().lower()

    if src == RANGE_SOURCE_YAHOO:
        if tf != "1d":
            return None
        try:
            df = fetch_yahoo_ohlcv_range(
                symbol,
                _as_utc_ts(probe_from),
                pd.Timestamp.now(tz="UTC"),
            )
        except Exception:
            return None
        if df is None or df.empty:
            return None
        return pd.Timestamp(df.index[0])

    step_ms = timeframe_ms(tf)
    lo = int(_as_utc_ts(probe_from).timestamp() * 1000)
    hi = int(time.time() * 1000)
    market = _ccxt_symbol_for_source(symbol, src)
    with _OHLCV_LOCK:
        ex = exchange_for_source(src)
        # Seed an upper bound: any recent page proves the market exists.
        seed = ex.fetch_ohlcv(market, timeframe=tf, limit=5)
        if not seed:
            return None
        hi = int(seed[0][0])

        # Binary search the earliest since that still returns a candle.
        best: Optional[int] = hi
        for _ in range(24):
            if hi - lo <= step_ms * 2:
                break
            mid = lo + (hi - lo) // 2
            batch = ex.fetch_ohlcv(market, timeframe=tf, since=mid, limit=5)
            time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
            if batch:
                best = int(batch[0][0])
                hi = mid
            else:
                lo = mid
        # One final pull from the refined lower bound.
        final = ex.fetch_ohlcv(
            market, timeframe=tf, since=max(0, (best or hi) - step_ms * 50), limit=50
        )
        time.sleep(getattr(ex, "rateLimit", 100) / 1000.0)
    if final:
        return pd.Timestamp(int(final[0][0]), unit="ms", tz="UTC")
    if best is not None:
        return pd.Timestamp(int(best), unit="ms", tz="UTC")
    return None


def _range_source_blurb(source: str) -> str:
    src = _normalize_range_source(source)
    if src == RANGE_SOURCE_BINANCE_SPOT:
        return "Binance spot"
    if src == RANGE_SOURCE_YAHOO:
        return "Yahoo Finance daily (spot USD)"
    return "MEXC USDT-M"


def _range_no_history_error(
    symbol: str,
    tf: str,
    t0: pd.Timestamp,
    t1: pd.Timestamp,
    *,
    got: Optional[pd.DataFrame] = None,
    source: str = RANGE_SOURCE_MEXC,
) -> DataError:
    """Clear error when the requested calendar window has no bars on ``source``."""
    src = _normalize_range_source(source)
    label = _range_source_blurb(src)
    earliest = None
    try:
        earliest = probe_earliest_ohlcv(symbol, tf, source=src)
    except Exception:
        earliest = None
    if earliest is None and got is not None and not got.empty:
        earliest = pd.Timestamp(got.index[0])
    if earliest is not None:
        extra = ""
        if src == RANGE_SOURCE_MEXC:
            extra = " (swap/futures, not spot). Try Binance spot or Yahoo for earlier years."
        elif src == RANGE_SOURCE_BINANCE_SPOT:
            extra = " (spot USDT). Try Yahoo daily for pre-listing years."
        if earliest >= t1:
            return DataError(
                f"no {tf} bars for {symbol} in {t0:%Y-%m-%d}→{t1:%Y-%m-%d}. "
                f"That window is entirely before {label} listing "
                f"(~{earliest:%Y-%m-%d}).{extra} "
                f"Set start to {earliest:%Y-%m-%d} or later."
            )
        return DataError(
            f"no {tf} bars for {symbol} in {t0:%Y-%m-%d}→{t1:%Y-%m-%d}. "
            f"{label} history for this market starts ~{earliest:%Y-%m-%d}"
            f"{extra} Set start to {earliest:%Y-%m-%d} or later."
        )
    hint = {
        RANGE_SOURCE_MEXC: (
            "MEXC returned no candles — USDT-M markets often list years after spot. "
            "Try Binance spot, Yahoo daily, or a more recent start."
        ),
        RANGE_SOURCE_BINANCE_SPOT: (
            "Binance spot returned no candles — pair may list later or lack this TF. "
            "Try Yahoo daily or a more recent start."
        ),
        RANGE_SOURCE_YAHOO: (
            "Yahoo returned no daily bars — ticker may be missing. "
            "Try another coin or a more recent start."
        ),
    }.get(src, "No candles in that window.")
    return DataError(f"no {tf} bars for {symbol} in {t0:%Y-%m-%d}→{t1:%Y-%m-%d}. {hint}")


def fetch_historical_range(
    symbol: str,
    tf: str,
    start,
    end,
    *,
    warmup_bars: int = 0,
    cache_dir: str = str(CACHE_DIR),
    refresh: bool = False,
    source: str = RANGE_SOURCE_MEXC,
) -> pd.DataFrame:
    """Fetch OHLCV covering ``[start - warmup, end)`` for calendar backtests.

    Warmup bars sit *before* ``start`` so indicators are valid when the
    BacktestEngine window opens; callers should still pass ``start``/``end``
    into the engine so only the chosen window is traded.

    ``source`` is Backtest-only: ``mexc`` (default), ``binance_spot``, or ``yahoo``.
    Live/paper paths never pass this and stay on MEXC.
    """
    src = _normalize_range_source(source)
    tf = str(tf).strip().lower()
    t0 = _as_utc_ts(start)
    t1 = _as_utc_ts(end)
    if not (t0 < t1):
        raise DataError(f"invalid range: start {t0} must be before end {t1}")

    if src == RANGE_SOURCE_YAHOO and tf != "1d":
        raise DataError(
            "Yahoo Finance Backtest source only supports 1d candles "
            f"(got {tf}). Switch candle size to 1d or pick MEXC / Binance spot."
        )

    warm = int(warmup_bars)
    if warm <= 0:
        warm = max(500, min(int(live_history_bars_for_tf(tf)), 3_000))

    step_ms = timeframe_ms(tf)
    fetch_start = t0 - pd.Timedelta(milliseconds=step_ms * warm)
    # Never request future candles past "now"
    now = pd.Timestamp.now(tz="UTC")
    fetch_end = min(t1, now)
    if fetch_start >= fetch_end:
        raise DataError(
            f"range is empty after clamping to now ({symbol} {tf}: {fetch_start} → {fetch_end})"
        )

    est_bars = int((fetch_end - fetch_start).total_seconds() * 1000 // step_ms) + 2
    # Cap against download size (native/base TF), not post-resample candle count.
    est_fetch = estimate_range_fetch_bars(
        tf,
        t0,
        t1,
        source=src,
        warmup_bars=warm,
    )
    if est_fetch > _MAX_RANGE_BARS:
        base_tf = _resample_base_tf(tf, source=src)
        if base_tf and base_tf != tf:
            raise DataError(
                f"range too large for {symbol} {tf}: needs ~{est_fetch:,} {base_tf} bars "
                f"to build (max {_MAX_RANGE_BARS:,}). Auto should pick a coarser TF — "
                f"or choose 5m+ / narrow the dates."
            )
        raise DataError(
            f"range too large for {symbol} {tf}: ~{est_bars:,} bars "
            f"(max {_MAX_RANGE_BARS:,}). Narrow the window or use a higher timeframe."
        )

    # Unsupported native TF → build from a finer base over the same window.
    base_tf = _resample_base_tf(tf, source=src)
    if base_tf is not None:
        if src == RANGE_SOURCE_YAHOO:
            raise DataError(f"Yahoo cannot build {tf} — use 1d or pick MEXC / Binance spot.")
        base_ms = timeframe_ms(base_tf)
        ratio = max(1, int(step_ms // base_ms))
        raw = fetch_historical_range(
            symbol,
            base_tf,
            t0,
            t1,
            warmup_bars=warm * ratio,
            cache_dir=cache_dir,
            refresh=refresh,
            source=src,
        )
        if raw is None or raw.empty:
            raise DataError(f"no {base_tf} history to build {tf} for {symbol}")
        built = _ohlcv_resample(raw, tf)
        if built.empty:
            raise DataError(f"resample to {tf} produced empty frame for {symbol}")
        out = built[(built.index >= fetch_start) & (built.index < fetch_end)]
        if out.empty:
            raise _range_no_history_error(symbol, tf, t0, t1, got=built, source=src)
        return out

    cache = Path(cache_dir)
    cache.mkdir(exist_ok=True)
    stem = f"{_cache_stem(symbol, tf, source=src)}range_"
    key = f"{stem}{int(fetch_start.timestamp())}_{int(fetch_end.timestamp())}_w{warm}.csv"
    path = cache / key

    if path.exists() and not refresh:
        try:
            cached = _read_ohlcv_csv(path)
            if len(cached) > 0:
                # Re-apply window in case an older cache key was reused.
                hit = cached[(cached.index >= fetch_start) & (cached.index < fetch_end)]
                if not hit.empty:
                    return hit
        except Exception:
            pass

    if src == RANGE_SOURCE_YAHOO:
        try:
            df = fetch_yahoo_ohlcv_range(symbol, fetch_start, fetch_end)
        except DataError:
            raise
        except Exception as e:
            raise DataError(f"Yahoo fetch failed for {symbol}: {e}") from e
        if df.empty:
            raise _range_no_history_error(symbol, tf, t0, t1, source=src)
        with contextlib.suppress(Exception):
            df.to_csv(path)
            prune_range_cache(cache, stem_prefix=stem)
        log.info(
            "%s yahoo 1d range %s→%s (+%d warmup) → %d bars",
            symbol,
            t0.date(),
            t1.date(),
            warm,
            len(df),
        )
        return df

    since_ms = int(fetch_start.timestamp() * 1000)
    until_ms = int(fetch_end.timestamp() * 1000)
    market = _ccxt_symbol_for_source(symbol, src)
    with _OHLCV_LOCK:
        ex = exchange_for_source(src)
        rows = paged_ohlcv(ex, market, tf, since_ms, until_ms)
        if not rows:
            time.sleep(0.5)
            rows = paged_ohlcv(ex, market, tf, since_ms, until_ms)

    if not rows:
        raise _range_no_history_error(symbol, tf, t0, t1, source=src)

    # Drop foothold bars that landed entirely after the requested window
    # (paged_ohlcv jumps forward when since is pre-listing).
    rows = [r for r in rows if since_ms <= int(r[0]) < until_ms]
    if not rows:
        raise _range_no_history_error(symbol, tf, t0, t1, source=src)

    df = ohlcv_to_frame(rows, tf)
    if df.empty:
        raise DataError(f"OHLCV empty after candle filter ({symbol} {tf})")
    # Keep warmup + window; drop anything strictly at/after fetch_end
    df = df[(df.index >= fetch_start) & (df.index < fetch_end)]
    if df.empty:
        raise _range_no_history_error(symbol, tf, t0, t1, got=None, source=src)
    with contextlib.suppress(Exception):
        df.to_csv(path)
        prune_range_cache(cache, stem_prefix=stem)
    log.info(
        "%s %s %s range %s→%s (+%d warmup) → %d bars",
        symbol,
        src,
        tf,
        t0.date(),
        t1.date(),
        warm,
        len(df),
    )
    return df


def prune_range_cache(
    cache_dir: str | Path | None = None,
    *,
    stem_prefix: str | None = None,
    keep_per_stem: int = 2,
    max_files: int = 200,
    max_bytes: int = 2 * 1024**3,
) -> dict[str, int]:
    """Delete stale *range_*.csv OHLCV caches (not search memory).

    - Per stem prefix: keep newest ``keep_per_stem`` by mtime.
    - Global: if total range files > ``max_files`` or bytes > ``max_bytes``,
      delete oldest range files until under both caps.
    Fixed bar caches (*_{tf}_{bars}.csv) are never touched.
    """
    root = Path(cache_dir) if cache_dir is not None else Path(CACHE_DIR)
    out = {"removed": 0, "freed_bytes": 0}
    if not root.is_dir():
        return out

    def _range_files() -> list[Path]:
        return [
            p for p in root.iterdir() if p.is_file() and "range_" in p.name and p.suffix == ".csv"
        ]

    # Sibling prune for the stem that just wrote (or explicit prefix).
    if stem_prefix:
        siblings = sorted(
            [p for p in _range_files() if p.name.startswith(stem_prefix)],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for old in siblings[max(0, int(keep_per_stem)) :]:
            try:
                sz = int(old.stat().st_size)
                old.unlink(missing_ok=True)
                out["removed"] += 1
                out["freed_bytes"] += sz
            except OSError:
                pass

    files = sorted(_range_files(), key=lambda p: p.stat().st_mtime)
    total_bytes = 0
    for p in files:
        with contextlib.suppress(OSError):
            total_bytes += int(p.stat().st_size)
    # Drop oldest until under caps.
    i = 0
    while files and (len(files) - i > int(max_files) or total_bytes > int(max_bytes)):
        victim = files[i]
        i += 1
        try:
            sz = int(victim.stat().st_size)
            victim.unlink(missing_ok=True)
            out["removed"] += 1
            out["freed_bytes"] += sz
            total_bytes = max(0, total_bytes - sz)
        except OSError:
            continue
    if out["removed"]:
        log.info(
            "range_cache_prune removed=%d freed_mb=%.1f",
            out["removed"],
            out["freed_bytes"] / (1024 * 1024),
        )
    return out


def synthetic_ohlcv(symbol: str, n: int, tf: str = "4h") -> pd.DataFrame:
    """Deterministic (crc32, not hash()) regime-switching data. SMOKE TESTS ONLY."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    drift = np.zeros(n)
    i = 0
    while i < n:
        length = int(rng.integers(60, 400))
        drift[i : i + length] = rng.choice([0.0015, -0.0015, 0.0003, 0.0])
        i += length
    sigma = 0.008 * np.exp(0.25 * np.cumsum(rng.normal(0, 0.1, n)) * 0.1)
    rets = drift + rng.normal(0, 1, n) * sigma
    close = 100.0 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]]) * (1 + rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.004, n)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.004, n)))
    volume = 2000 * (1 + 40 * np.abs(rets)) * rng.lognormal(0, 0.3, n)
    idx = pd.date_range("2024-01-01", periods=n, freq=_pandas_tf(tf), tz="UTC")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume}, index=idx
    )
