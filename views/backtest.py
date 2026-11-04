"""Backtest — replay the current .env book on a user-chosen historic window."""

from __future__ import annotations

import calendar
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import moonshot as ms
from dash_lib import (
    ACCENT, GREEN, MUTED, empty_state, entries_exits_fig, inject_theme,
    load_autopilot_state, load_equity, detect_mode, load_ohlcv_range,
    meta_line, page_header, panel_header, pct_of_risk, plotly_kwargs,
    plotly_layout, polish_fig, section_header, show_autofit_price_chart,
    stat_grid, topbar,
)

inject_theme()
_eq = load_equity(detect_mode(load_equity(None)))
_bal = float(_eq["equity"].iloc[-1]) if not _eq.empty else None
_score = int((load_autopilot_state().get("grade") or {}).get("score") or 0)
topbar(balance=_bal, score=_score, mode="PAPER", active="Backtest")
page_header(
    "",
    "Backtest",
    "Pick exact historic dates and replay your current practice book",
)

import importlib
importlib.reload(ms)

ALL_SYMBOLS = list(ms.DEFAULT_SYMBOLS) + list(getattr(ms, "OPTIONAL_SYMBOLS", []) or [])
cfg0 = ms.Config.from_env()
_book = f"{cfg0.strategy} · {int(cfg0.leverage)}× · {cfg0.timeframe}"
st.caption(f"Active book (from `.env`, read-only): **{_book}**")


def _trades_frame(trades: list) -> pd.DataFrame:
    rows = []
    for t in trades:
        rows.append({
            "Coin": t.symbol.split("/")[0],
            "Side": "Buy" if t.direction.value == "long" else "Sell",
            "Opened": t.entry_time,
            "Closed": t.exit_time,
            "Profit $": round(t.pnl, 2),
            "Why": t.exit_reason.value,
        })
    return pd.DataFrame(rows)


def _trades_chart_df(trades: list) -> pd.DataFrame:
    """TradeResult list → columns expected by ``entries_exits_fig``."""
    rows = []
    for t in trades:
        direction = t.direction.value if hasattr(t.direction, "value") else str(t.direction)
        reason = t.exit_reason.value if hasattr(t.exit_reason, "value") else str(t.exit_reason)
        rows.append({
            "symbol": t.symbol,
            "direction": direction,
            "entry_time": t.entry_time,
            "ts": t.exit_time,
            "entry": float(t.entry_price),
            "exit": float(t.exit_price),
            "pnl": float(t.pnl),
            "exit_reason": reason,
            "r_multiple": float(getattr(t, "r_multiple", 0.0) or 0.0),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def _window_days(win_start, win_end) -> float:
    t0 = pd.Timestamp(win_start)
    t1 = pd.Timestamp(win_end)
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    else:
        t0 = t0.tz_convert("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    else:
        t1 = t1.tz_convert("UTC")
    days = float((t1 - t0).total_seconds() / 86_400.0)
    return max(days, 1.0 / 24.0)  # floor: 1 hour


def _trade_pace(trades: list, win_start, win_end) -> dict:
    """Average closed-trade rates over the backtest calendar window."""
    n = len(trades or [])
    days = _window_days(win_start, win_end)
    per_day = n / days
    return {
        "n": n,
        "days": days,
        "per_day": per_day,
        "per_week": per_day * 7.0,
        "per_month": per_day * (365.25 / 12.0),
        "per_year": per_day * 365.25,
    }


def _trade_period_counts(trades: list, period: str) -> pd.DataFrame:
    """Count closed trades by calendar day / week / month / year."""
    if not trades:
        return pd.DataFrame(columns=["Period", "Trades"])
    exits = pd.to_datetime([t.exit_time for t in trades], utc=True)
    # Period grouping wants tz-naive stamps (UTC calendar days).
    idx = pd.DatetimeIndex(exits).tz_convert("UTC").tz_localize(None)
    s = pd.Series(1, index=idx).sort_index()
    p = str(period).lower()
    if p == "day":
        g = s.groupby(s.index.floor("D")).sum()
        labels = [ts.strftime("%Y-%m-%d") for ts in g.index]
    elif p == "week":
        # ISO week starting Monday
        g = s.groupby(s.index.to_period("W-MON")).sum()
        labels = [str(ix) for ix in g.index]
    elif p == "month":
        g = s.groupby(s.index.to_period("M")).sum()
        labels = [str(ix) for ix in g.index]
    else:  # year
        g = s.groupby(s.index.to_period("Y")).sum()
        labels = [str(ix) for ix in g.index]
    return pd.DataFrame({"Period": labels, "Trades": [int(v) for v in g.values]})


def _show_trade_pace(trades: list, win_start, win_end, *, key_prefix: str) -> None:
    """Compact pace rates + period histogram."""
    pace = _trade_pace(trades, win_start, win_end)
    panel_header(
        "Trade pace",
        f"{pace['n']} closed · {pace['days']:.0f}d window",
    )
    stat_grid(
        [
            {"label": "Per day", "value": f"{pace['per_day']:.2f}"},
            {"label": "Per week", "value": f"{pace['per_week']:.2f}"},
            {"label": "Per month", "value": f"{pace['per_month']:.2f}"},
            {"label": "Per year", "value": f"{pace['per_year']:.1f}", "tone": "accent"},
            {"label": "Window", "value": f"{pace['days']:.0f}d"},
        ],
        cols=5,
        dense=True,
    )

    if pace["n"] <= 0:
        st.caption("No closed trades in this window.")
        return

    days = pace["days"]
    if days <= 21:
        default_bucket = "Day"
    elif days <= 120:
        default_bucket = "Week"
    elif days <= 800:
        default_bucket = "Month"
    else:
        default_bucket = "Year"

    bucket = st.radio(
        "Count by",
        options=["Day", "Week", "Month", "Year"],
        index=["Day", "Week", "Month", "Year"].index(default_bucket),
        horizontal=True,
        key=f"{key_prefix}_pace_bucket",
    )
    counts = _trade_period_counts(trades, bucket.lower())
    if counts.empty:
        return

    chart_df = counts
    max_bars_show = {"Day": 90, "Week": 52, "Month": 36, "Year": 20}.get(bucket, 40)
    if len(chart_df) > max_bars_show:
        chart_df = chart_df.tail(max_bars_show)

    c_tbl, c_bar = st.columns([1, 2], gap="small")
    with c_tbl:
        st.dataframe(
            counts,
            width="stretch",
            hide_index=True,
            height=min(280, 36 + 26 * min(len(counts), 10)),
        )
        st.caption(
            f"avg {counts['Trades'].mean():.1f}/{bucket.lower()} · "
            f"max {int(counts['Trades'].max())}"
        )
    with c_bar:
        # Categorical x avoids Plotly treating years as continuous (2017.5 ticks).
        fig = go.Figure(data=[go.Bar(
            x=chart_df["Period"].astype(str).tolist(),
            y=chart_df["Trades"].tolist(),
            marker_color=ACCENT,
            name="Trades",
            hovertemplate="%{x}<br>%{y} trades<extra></extra>",
        )])
        n_ticks = min(12, len(chart_df))
        fig.update_layout(**plotly_layout(
            kind="free",
            uirevision=f"{key_prefix}-pace-{bucket}",
            height=260,
            title=None,
            margin=dict(l=36, r=8, t=8, b=48),
            showlegend=False,
            xaxis=dict(
                type="category",
                tickangle=-35 if len(chart_df) > 8 else 0,
                nticks=n_ticks,
                title=None,
            ),
            yaxis=dict(title=None, rangemode="tozero"),
        ))
        st.plotly_chart(polish_fig(fig), **plotly_kwargs())
        if len(counts) > len(chart_df):
            st.caption(f"Chart shows last {len(chart_df)} {bucket.lower()}s")


def _show_backtest_price_chart(
    lab_state: dict,
    trades: list,
    *,
    book_tf: str,
    key_prefix: str,
) -> None:
    """Candles + entry/exit markers for the last Backtest run."""
    panel_header(
        "Price action",
        "▲ long · ▼ short · ○ exit",
    )
    chart_df = _trades_chart_df(trades)
    symbols = list(lab_state.get("symbols") or [])
    if chart_df is not None and not chart_df.empty:
        traded = (
            chart_df.groupby("symbol").size().sort_values(ascending=False).index.tolist()
        )
        # Prefer coins that actually traded; keep run universe as fallback.
        symbols = traded + [s for s in symbols if s not in traded]
    if not symbols:
        st.caption("No coins in this run to chart.")
        return

    labels = {s: s.split("/")[0] for s in symbols}
    default = symbols[0]
    c_sym, c_win = st.columns([2, 2])
    with c_sym:
        chart_sym = st.selectbox(
            "Coin to chart",
            options=symbols,
            index=0 if default not in symbols else symbols.index(default),
            format_func=lambda s: (
                f"{labels.get(s, s)}"
                + (
                    f" · {int((chart_df['symbol'] == s).sum())} trades"
                    if chart_df is not None and not chart_df.empty
                    else ""
                )
            ),
            key=f"{key_prefix}_coin",
        )
    with c_win:
        win = st.radio(
            "Time window",
            options=["auto", "1d", "7d", "30d", "all"],
            index=0,
            horizontal=True,
            key=f"{key_prefix}_win",
            help="auto = pad around this coin’s trades · all = full backtest window",
        )

    run_tf = str(lab_state.get("run_tf") or book_tf).strip().lower()
    src = str(lab_state.get("price_source") or ms.RANGE_SOURCE_MEXC)
    warmup = ms.range_warmup_bars(run_tf)
    try:
        raw = load_ohlcv_range(
            (chart_sym,),
            run_tf,
            pd.Timestamp(lab_state["win_start"]).isoformat(),
            pd.Timestamp(lab_state["win_end"]).isoformat(),
            int(warmup),
            False,
            src,
        )
        ohlcv = raw.get(chart_sym)
    except Exception as e:
        st.warning(f"Could not load {labels.get(chart_sym, chart_sym)} candles: {e}")
        return

    if ohlcv is None or ohlcv.empty:
        st.warning(f"No {run_tf} candles for {labels.get(chart_sym, chart_sym)}.")
        return

    # Clip to the traded window (drop warmup-only left pad for a cleaner chart).
    t0 = pd.Timestamp(lab_state["win_start"])
    t1 = pd.Timestamp(lab_state["win_end"])
    if t0.tzinfo is None:
        t0 = t0.tz_localize("UTC")
    if t1.tzinfo is None:
        t1 = t1.tz_localize("UTC")
    ohlcv = ohlcv[(ohlcv.index >= t0) & (ohlcv.index < t1)]
    if ohlcv.empty:
        st.warning("No candles inside the backtest window for this coin.")
        return

    sym_trades = (
        chart_df[chart_df["symbol"] == chart_sym]
        if chart_df is not None and not chart_df.empty
        else pd.DataFrame()
    )

    # Map 7d/30d via slice; entries_exits_fig knows auto/1d/all (and a few hours).
    view = str(win)
    view_ohlcv = ohlcv
    if win == "7d":
        view = "all"
        cut = ohlcv.index[-1] - pd.Timedelta(days=7)
        view_ohlcv = ohlcv[ohlcv.index >= cut]
        if view_ohlcv.empty:
            view_ohlcv = ohlcv
    elif win == "30d":
        view = "all"
        cut = ohlcv.index[-1] - pd.Timedelta(days=30)
        view_ohlcv = ohlcv[ohlcv.index >= cut]
        if view_ohlcv.empty:
            view_ohlcv = ohlcv

    max_bars = min(len(view_ohlcv), 8_000)
    show_autofit_price_chart(
        entries_exits_fig(
            view_ohlcv,
            sym_trades,
            title=f"{labels.get(chart_sym, chart_sym)} · {run_tf} · backtest",
            max_bars=max(48, int(max_bars)),
            max_trades=max(40, len(sym_trades) if not sym_trades.empty else 40),
            view=view,
        ),
        height=560,
        key=f"{key_prefix}_{str(chart_sym).split('/')[0]}_{run_tf}_{win}",
    )
    n_tr = 0 if sym_trades.empty else len(sym_trades)
    st.caption(
        f"{lab_state.get('price_source_label') or src} · {run_tf} · "
        f"{n_tr} trades on chart · ▲/▼ entry · ○ exit · "
        "scroll zooms time · drag pans"
    )


def _utc_day_start(d: date) -> pd.Timestamp:
    return pd.Timestamp(datetime(d.year, d.month, d.day, tzinfo=timezone.utc))


def _utc_day_end_exclusive(d: date) -> pd.Timestamp:
    """End of selected calendar day (exclusive next midnight UTC)."""
    return _utc_day_start(d) + pd.Timedelta(days=1)


PRESET_DAYS = {
    "Last 7 days": 7,
    "Last 30 days": 30,
    "Last 90 days": 90,
    "Last 180 days": 180,
    "Last 1 year": 365,
}
# Custom first so it's always visible / easy to click (not buried under the top nav).
PRESET_OPTIONS = ["Custom", *PRESET_DAYS.keys()]
_MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}
# Earliest year offered in custom pickers (≥12y lookback; 15y buffer).
_YEAR_LOOKBACK = 15


def _today_utc() -> date:
    return datetime.now(timezone.utc).date()


def _fmt_mdy(d: date) -> str:
    """Human date: Oct 27, 2026 — not year-first YYYY/MM/DD."""
    return f"{_MONTH_LABELS[int(d.month)]} {int(d.day)}, {int(d.year)}"


def _preset_bounds(label: str) -> tuple[date, date]:
    days = int(PRESET_DAYS[label])
    end = _today_utc()
    return end - timedelta(days=days), end


def _seed_ymd(prefix: str, d: date) -> None:
    """Write Month/Day/Year widget keys from a date (only when those widgets are not mounted)."""
    st.session_state[f"{prefix}_m"] = int(d.month)
    st.session_state[f"{prefix}_d"] = int(d.day)
    st.session_state[f"{prefix}_y"] = int(d.year)


def _read_ymd(prefix: str, fallback: date) -> date:
    try:
        y = int(st.session_state.get(f"{prefix}_y", fallback.year))
        m = int(st.session_state.get(f"{prefix}_m", fallback.month))
        d = int(st.session_state.get(f"{prefix}_d", fallback.day))
        d = min(max(1, d), calendar.monthrange(y, m)[1])
        return date(y, m, d)
    except Exception:
        return fallback


def _ymd_picker(
    label: str,
    prefix: str,
    *,
    min_d: date,
    max_d: date,
) -> date:
    """Month / Day / Year selectboxes — no calendar popover."""
    # Apply pending seed from preset→custom transition (set in on_change).
    pending = st.session_state.pop(f"{prefix}_pending", None)
    if isinstance(pending, date):
        _seed_ymd(prefix, pending)
    elif f"{prefix}_y" not in st.session_state:
        _seed_ymd(prefix, max(min_d, min(max_d, _today_utc())))

    # Clamp day into month length without fighting Month/Year widgets:
    # only adjust the Day key before the Day selectbox is created.
    y_pre = int(st.session_state.get(f"{prefix}_y", max_d.year))
    m_pre = int(st.session_state.get(f"{prefix}_m", 1))
    y_pre = min(max(y_pre, min_d.year), max_d.year)
    m_pre = min(max(m_pre, 1), 12)
    dim_pre = calendar.monthrange(y_pre, m_pre)[1]
    d_pre = int(st.session_state.get(f"{prefix}_d", 1))
    if d_pre < 1 or d_pre > dim_pre:
        st.session_state[f"{prefix}_d"] = min(max(1, d_pre), dim_pre)
    # Keep Y/M in range for the upcoming selectbox options lists.
    if int(st.session_state.get(f"{prefix}_y", y_pre)) != y_pre:
        st.session_state[f"{prefix}_y"] = y_pre
    if int(st.session_state.get(f"{prefix}_m", m_pre)) != m_pre:
        st.session_state[f"{prefix}_m"] = m_pre

    st.markdown(f"**{label}**")
    c_m, c_d, c_y = st.columns(3)
    years = list(range(min_d.year, max_d.year + 1))
    months = list(range(1, 13))
    with c_m:
        m = st.selectbox(
            "Month",
            months,
            key=f"{prefix}_m",
            format_func=lambda x: _MONTH_LABELS[int(x)],
            label_visibility="collapsed",
        )
    with c_y:
        y = st.selectbox(
            "Year",
            years,
            key=f"{prefix}_y",
            label_visibility="collapsed",
        )
    dim = calendar.monthrange(int(y), int(m))[1]
    # If day options shrink, drop the key so the selectbox remounts validly.
    if int(st.session_state.get(f"{prefix}_d", 1)) > dim:
        del st.session_state[f"{prefix}_d"]
        st.session_state[f"{prefix}_d"] = dim
    with c_d:
        d = st.selectbox(
            "Day",
            list(range(1, dim + 1)),
            key=f"{prefix}_d",
            label_visibility="collapsed",
        )

    out = date(int(y), int(m), int(d))
    if out < min_d:
        out = min_d
    if out > max_d:
        out = max_d
    st.caption(_fmt_mdy(out) + " UTC")
    return out


def _prime_custom_from_preset(label: str) -> None:
    """Queue Y/M/D values for Custom pickers (applied when those widgets mount)."""
    if label not in PRESET_DAYS:
        label = "Last 30 days"
    start, end = _preset_bounds(label)
    st.session_state["bt_lab_start_pending"] = start
    st.session_state["bt_lab_end_pending"] = end


def _on_preset_change() -> None:
    """When leaving a named preset for Custom, seed the Month/Day/Year pickers."""
    label = st.session_state.get("bt_lab_preset") or "Last 30 days"
    prev = st.session_state.get("bt_lab_preset_prev")
    # Entering Custom from a named window → keep that window as the starting point.
    if label == "Custom" and prev in PRESET_DAYS:
        _prime_custom_from_preset(prev)
    elif label in PRESET_DAYS:
        _prime_custom_from_preset(label)
    st.session_state["bt_lab_preset_prev"] = label


def _on_split_toggle() -> None:
    if not st.session_state.get("bt_lab_use_split"):
        for k in ("bt_lab_split_m", "bt_lab_split_d", "bt_lab_split_y", "bt_lab_split_pending"):
            st.session_state.pop(k, None)
        return
    today = _today_utc()
    s = _read_ymd("bt_lab_start", today - timedelta(days=30))
    e = _read_ymd("bt_lab_end", today)
    if s >= e:
        return
    mid = s + (e - s) / 2
    st.session_state["bt_lab_split_pending"] = mid


today = _today_utc()
_min_date = date(max(2010, today.year - _YEAR_LOOKBACK), 1, 1)

# One-shot defaults (v4: Custom is back in the Preset dropdown, listed first).
if st.session_state.get("bt_lab_controls_ready") != 4:
    st.session_state["bt_lab_preset"] = "Last 30 days"
    st.session_state["bt_lab_preset_prev"] = "Last 30 days"
    _prime_custom_from_preset("Last 30 days")
    _seed_ymd("bt_lab_start", st.session_state["bt_lab_start_pending"])
    _seed_ymd("bt_lab_end", st.session_state["bt_lab_end_pending"])
    st.session_state.pop("bt_lab_start_pending", None)
    st.session_state.pop("bt_lab_end_pending", None)
    st.session_state["bt_lab_symbols"] = list(ms.DEFAULT_SYMBOLS)
    st.session_state["bt_lab_refresh"] = False
    st.session_state["bt_lab_use_split"] = False
    st.session_state.pop("bt_lab_window_mode", None)
    for stale in ("bt_lab_start", "bt_lab_end", "bt_lab_split"):
        st.session_state.pop(stale, None)
    st.session_state["bt_lab_controls_ready"] = 4

section_header("Time window", "Complete control — presets or exact dates")

c_pre, c_sym = st.columns([2, 2])
with c_pre:
    preset = st.selectbox(
        "Preset",
        PRESET_OPTIONS,
        key="bt_lab_preset",
        on_change=_on_preset_change,
        help="Custom unlocks Month / Day / Year for any historic window.",
    )
with c_sym:
    symbols = st.multiselect(
        "Coins",
        ALL_SYMBOLS,
        key="bt_lab_symbols",
    )

c_ref, c_spl = st.columns(2)
with c_ref:
    refresh = st.checkbox("Re-download prices", key="bt_lab_refresh")
with c_spl:
    use_split = st.checkbox(
        "Split older / newer",
        key="bt_lab_use_split",
        on_change=_on_split_toggle,
    )

if preset == "Custom":
    d_a, d_b = st.columns(2)
    with d_a:
        start_date = _ymd_picker(
            "Start (UTC) — Month / Day / Year",
            "bt_lab_start",
            min_d=_min_date,
            max_d=today,
        )
    with d_b:
        end_date = _ymd_picker(
            "End (UTC, inclusive) — Month / Day / Year",
            "bt_lab_end",
            min_d=_min_date,
            max_d=today,
        )
else:
    start_date, end_date = _preset_bounds(str(preset))
    # Keep Custom pickers primed while browsing named presets.
    _prime_custom_from_preset(str(preset))
    d_a, d_b = st.columns(2)
    with d_a:
        st.markdown("**Start date (UTC)**")
        st.markdown(f"**{_fmt_mdy(start_date)}**")
    with d_b:
        st.markdown("**End date (UTC, inclusive)**")
        st.markdown(f"**{_fmt_mdy(end_date)}**")
    st.caption("Choose **Custom** in Preset to edit Month / Day / Year.")

st.caption(
    "MEXC USDT-M (swap) history for majors like BTC usually starts around mid-2020 — "
    "windows entirely before listing return no bars."
)

split_date = None
if use_split:
    if start_date < end_date:
        mid = start_date + (end_date - start_date) / 2
        if "bt_lab_split_y" not in st.session_state and "bt_lab_split_pending" not in st.session_state:
            st.session_state["bt_lab_split_pending"] = mid
        elif "bt_lab_split_y" in st.session_state:
            cur = _read_ymd("bt_lab_split", mid)
            if cur < start_date or cur > end_date:
                st.session_state["bt_lab_split_pending"] = mid
                for k in ("bt_lab_split_m", "bt_lab_split_d", "bt_lab_split_y"):
                    st.session_state.pop(k, None)
        split_date = _ymd_picker(
            "Split (newer starts here) — Month / Day / Year",
            "bt_lab_split",
            min_d=start_date,
            max_d=end_date,
        )
    else:
        st.warning("Fix start/end dates before choosing a split.")

# ---- Price source (Backtest only; live/paper stay on MEXC) ----
_SOURCE_OPTIONS = [
    (ms.RANGE_SOURCE_MEXC, ms.RANGE_SOURCE_LABELS[ms.RANGE_SOURCE_MEXC]),
    (ms.RANGE_SOURCE_BINANCE_SPOT, ms.RANGE_SOURCE_LABELS[ms.RANGE_SOURCE_BINANCE_SPOT]),
    (ms.RANGE_SOURCE_YAHOO, ms.RANGE_SOURCE_LABELS[ms.RANGE_SOURCE_YAHOO]),
]
_SOURCE_LABEL_TO_ID = {lab: sid for sid, lab in _SOURCE_OPTIONS}
if "bt_lab_price_source" not in st.session_state:
    st.session_state["bt_lab_price_source"] = ms.RANGE_SOURCE_LABELS[ms.RANGE_SOURCE_MEXC]

section_header("Price source", "Historic candles for this tab only — not used by live or paper")
price_source_label = st.selectbox(
    "Price source",
    [lab for _, lab in _SOURCE_OPTIONS],
    key="bt_lab_price_source",
    help=(
        "MEXC matches the live venue. Binance spot has deeper USDT history (~2017+). "
        "Yahoo daily reaches further back (spot USD) but only 1d candles."
    ),
)
price_source = _SOURCE_LABEL_TO_ID.get(str(price_source_label), ms.RANGE_SOURCE_MEXC)
if price_source != ms.RANGE_SOURCE_MEXC:
    st.caption(
        "Historic spot prices — not identical to MEXC USDT-M; for research windows only."
    )

# ---- Candle size: default is the live book; Auto never coarsens past it ----
_book_tf = str(cfg0.timeframe).strip().lower()
_book_choice = f"Book default ({_book_tf})"
_yahoo_locked = price_source == ms.RANGE_SOURCE_YAHOO
_tf_choices = (
    ["1d"]
    if _yahoo_locked
    else [
        _book_choice,
        "Auto (denser only)",
        *[tf for tf in ms.RANGE_TF_LADDER if tf != _book_tf],
    ]
)
if "bt_lab_candle_mode" not in st.session_state:
    st.session_state["bt_lab_candle_mode"] = _book_choice
if _yahoo_locked:
    st.session_state["bt_lab_candle_mode"] = "1d"
else:
    _cur = st.session_state.get("bt_lab_candle_mode")
    if _cur == "1d" and "1d" not in _tf_choices:
        st.session_state["bt_lab_candle_mode"] = _book_choice
    elif _cur == "Auto (fit window)":
        # Old label silently coarsened; pin to book so we don't lie again.
        st.session_state["bt_lab_candle_mode"] = _book_choice
    elif isinstance(_cur, str) and _cur.startswith("Book default") and _cur != _book_choice:
        st.session_state["bt_lab_candle_mode"] = _book_choice

section_header(
    "Candles",
    "Yahoo is daily-only"
    if _yahoo_locked
    else f"Default = live book ({_book_tf}). Auto may pack denser bars; it will not coarsen.",
)
candle_mode = st.selectbox(
    "Candle size",
    _tf_choices,
    key="bt_lab_candle_mode",
    help=(
        "Yahoo Finance only supplies daily bars."
        if _yahoo_locked
        else (
            f"Range fetch caps at {ms.MAX_RANGE_BARS:,} bars/symbol (incl. warmup). "
            f"Book default always uses {_book_tf}. Auto only tries finer TFs that still "
            "fit — it refuses to raise to a coarser size. Pick 15m/30m/… yourself if you "
            "want a coarse research run."
        )
    ),
)

# Preview estimate for the current window (inclusive end → exclusive next day).
_preview_start = _utc_day_start(start_date) if start_date < end_date else None
_preview_end = _utc_day_end_exclusive(end_date) if start_date < end_date else None
_run_tf = _book_tf
_tf_note = ""
_est_bars = 0
_est_fetch = 0
if _preview_start is not None and _preview_end is not None:
    if _yahoo_locked:
        _run_tf = "1d"
        _warm = ms.range_warmup_bars(_run_tf)
        _est_bars = ms.estimate_range_bars(
            _run_tf, _preview_start, _preview_end, warmup_bars=_warm,
        )
        _est_fetch = _est_bars
        _tf_note = "Yahoo Finance Backtest source is locked to 1d (spot USD)."
    elif candle_mode.startswith("Auto"):
        _run_tf, _est_bars, _tf_note = ms.pick_range_timeframe(
            _preview_start, _preview_end, _book_tf, source=price_source,
        )
        _est_fetch = ms.estimate_range_fetch_bars(
            _run_tf, _preview_start, _preview_end,
            source=price_source, warmup_bars=ms.range_warmup_bars(_run_tf),
        )
    elif candle_mode.startswith("Book default"):
        _run_tf = _book_tf
        _warm = ms.range_warmup_bars(_run_tf)
        _est_bars = ms.estimate_range_bars(
            _run_tf, _preview_start, _preview_end, warmup_bars=_warm,
        )
        _est_fetch = ms.estimate_range_fetch_bars(
            _run_tf, _preview_start, _preview_end,
            source=price_source, warmup_bars=_warm,
        )
        if _est_fetch > ms.MAX_RANGE_BARS:
            _tf_note = (
                f"Book {_book_tf} needs ~{_est_fetch:,} download bars "
                f"(cap {ms.MAX_RANGE_BARS:,}). Shorten the dates, or pick a coarser "
                "TF yourself if you want a research run (not the live book)."
            )
    else:
        _run_tf = str(candle_mode).strip().lower()
        _warm = ms.range_warmup_bars(_run_tf)
        _est_bars = ms.estimate_range_bars(
            _run_tf, _preview_start, _preview_end, warmup_bars=_warm,
        )
        _est_fetch = ms.estimate_range_fetch_bars(
            _run_tf, _preview_start, _preview_end,
            source=price_source, warmup_bars=_warm,
        )
        if _est_fetch > ms.MAX_RANGE_BARS:
            _tf_note = (
                f"{_run_tf} needs ~{_est_fetch:,} download bars "
                f"(cap {ms.MAX_RANGE_BARS:,}). Shorten the dates or pick a coarser candle."
            )
        elif _run_tf != _book_tf:
            _tf_note = f"Replaying book logic on {_run_tf} candles (book default is {_book_tf})."

    # Cap uses download size (3m is built from 1m on MEXC).
    _ok = 0 < int(_est_fetch) <= ms.MAX_RANGE_BARS
    _src_short = ms.RANGE_SOURCE_LABELS.get(price_source, price_source)
    st.caption(
        f"**{_src_short}** · ~{_est_bars:,} {_run_tf} candles "
        f"(fetch cap {ms.MAX_RANGE_BARS:,}/symbol — not a target to fill)."
        + ("" if _ok else " — too large")
    )
    if _tf_note:
        (st.warning if not _ok else st.info)(_tf_note)

run = st.button("Run backtest", type="primary", width="stretch")

if run:
    if not symbols:
        st.error("Pick at least one coin.")
        st.stop()
    if start_date >= end_date:
        st.error("Start date must be before end date.")
        st.stop()
    win_start = _utc_day_start(start_date)
    win_end = _utc_day_end_exclusive(end_date)
    if use_split and split_date is not None:
        split_ts = _utc_day_start(split_date)
        if not (win_start < split_ts < win_end):
            st.error("Split date must fall strictly inside the window.")
            st.stop()
    else:
        split_ts = None

    # Resolve candle TF for this run (same rules as the live preview).
    if price_source == ms.RANGE_SOURCE_YAHOO:
        run_tf = "1d"
        warm = ms.range_warmup_bars(run_tf)
        est_bars = ms.estimate_range_bars(run_tf, win_start, win_end, warmup_bars=warm)
        est_fetch = est_bars
        tf_note = "Yahoo Finance · 1d spot USD"
    elif candle_mode.startswith("Auto"):
        run_tf, est_bars, tf_note = ms.pick_range_timeframe(
            win_start, win_end, _book_tf, source=price_source,
        )
        est_fetch = ms.estimate_range_fetch_bars(
            run_tf, win_start, win_end,
            source=price_source, warmup_bars=ms.range_warmup_bars(run_tf),
        )
    elif candle_mode.startswith("Book default"):
        run_tf = _book_tf
        warm = ms.range_warmup_bars(run_tf)
        est_bars = ms.estimate_range_bars(run_tf, win_start, win_end, warmup_bars=warm)
        est_fetch = ms.estimate_range_fetch_bars(
            run_tf, win_start, win_end, source=price_source, warmup_bars=warm,
        )
        tf_note = ""
    else:
        run_tf = str(candle_mode).strip().lower()
        warm = ms.range_warmup_bars(run_tf)
        est_bars = ms.estimate_range_bars(run_tf, win_start, win_end, warmup_bars=warm)
        est_fetch = ms.estimate_range_fetch_bars(
            run_tf, win_start, win_end, source=price_source, warmup_bars=warm,
        )
        tf_note = ""

    if est_fetch > ms.MAX_RANGE_BARS:
        st.error(
            f"Too many download bars (~{est_fetch:,} for {run_tf}; "
            f"cap {ms.MAX_RANGE_BARS:,}/symbol). "
            "Shorten the date range, or manually pick a coarser candle for research. "
            "Auto will not leave the book timeframe. "
            "(3m is built from 1m, so it can blow the cap even when 3m candle count looks fine.)"
        )
        st.stop()

    run_cfg = replace(cfg0, timeframe=str(run_tf))
    warmup = ms.range_warmup_bars(run_tf)
    _src_label = ms.RANGE_SOURCE_LABELS.get(price_source, price_source)
    with st.spinner(
        f"Downloading {_src_label} {run_tf} prices and replaying trades… "
        "this can take a minute"
    ):
        try:
            raw = load_ohlcv_range(
                tuple(symbols),
                str(run_tf),
                win_start.isoformat(),
                win_end.isoformat(),
                int(warmup),
                bool(refresh),
                str(price_source),
            )
            data = ms.prepare_data(raw, run_cfg)
        except ms.BotError as e:
            msg = str(e)
            # Listing-date / empty-window errors already say what to do.
            if (
                "history for this market starts" in msg
                or "Set start to" in msg
                or "Yahoo" in msg
            ):
                st.error(msg)
            else:
                st.error(
                    f"Could not load prices: {msg}. "
                    "Try another price source, Auto candles, a higher TF, "
                    "narrower dates, or turn off Re-download."
                )
            st.stop()
        except Exception as e:
            st.error(f"Backtest failed while loading data: {e}")
            st.stop()

        label = (
            bool(run_cfg.enable_pyramiding),
            bool(run_cfg.enable_partial),
            int(run_cfg.leverage),
        )
        try:
            if split_ts is not None:
                older = ms.BacktestEngine(data, run_cfg, start=win_start, end=split_ts).run(label)
                newer = ms.BacktestEngine(data, run_cfg, start=split_ts, end=win_end).run(label)
                full = None
            else:
                full = ms.BacktestEngine(data, run_cfg, start=win_start, end=win_end).run(label)
                older = newer = None
        except ms.BotError as e:
            st.error(f"Replay failed: {e}")
            st.stop()
        except Exception as e:
            st.error(f"Replay failed: {e}")
            st.stop()

    _cfg_label = (
        f"{_src_label} · {run_cfg.strategy} · {int(run_cfg.leverage)}× · {run_tf}"
        + (f" (book {_book_tf})" if run_tf != _book_tf else "")
    )
    st.session_state["bt_lab"] = {
        "full": full,
        "older": older,
        "newer": newer,
        "split": split_ts,
        "win_start": win_start,
        "win_end": win_end,
        "symbols": list(symbols),
        "cfg_label": _cfg_label,
        "run_tf": run_tf,
        "book_tf": _book_tf,
        "price_source": price_source,
        "price_source_label": _src_label,
        "est_bars": est_bars,
        "tf_note": tf_note,
        "start_eq": float(run_cfg.starting_equity),
        "run_ts": datetime.now(timezone.utc).isoformat(),
    }

if "bt_lab" not in st.session_state:
    empty_state(
        "No backtest run yet",
        "Choose a time window above and click <strong>Run backtest</strong>.",
    )
    st.stop()

lab = st.session_state["bt_lab"]
_run_ts = str(lab.get("run_ts") or "")
_run_label = (_run_ts[:19].replace("T", " ") + " UTC") if _run_ts else "unknown time"
win_start = lab["win_start"]
win_end = lab["win_end"]
# Display end as inclusive calendar day
_end_disp = (pd.Timestamp(win_end) - pd.Timedelta(days=1)).date()
_window = f"{pd.Timestamp(win_start):%Y-%m-%d} → {_end_disp}"
split = lab.get("split")
if split is not None:
    _window += f" · split {pd.Timestamp(split):%Y-%m-%d}"

meta_line(
    f"{_run_label} · {lab.get('cfg_label') or _book} · {_window}"
)

section_header("Results", "Performance · equity · pace · price with entries/exits")

if lab.get("full") is not None:
    res: ms.BacktestResult = lab["full"]
    ret_tone = "good" if res.total_return >= 0 else "bad"
    dd_tone = "bad" if res.max_drawdown >= 0.25 else ""
    stat_grid(
        [
            {"label": "Return", "value": f"{res.total_return:+.0%}", "tone": ret_tone},
            {"label": "Final equity", "value": f"${res.final_equity:,.2f}", "tone": "accent"},
            {"label": "Max DD", "value": f"{res.max_drawdown:.0%}", "tone": dd_tone},
            {"label": "Win rate", "value": f"{res.win_rate:.0%}"},
            {"label": "Trades", "value": f"{int(res.trades)}"},
            {"label": "Avg profit", "value": pct_of_risk(res.expectancy_r)},
        ],
        cols=6,
        dense=True,
    )

    # Equity first (compact), then price action with trade markers
    panel_header("Practice balance", _window)
    fig = go.Figure()
    if len(res.equity_curve):
        fig.add_trace(go.Scatter(
            x=res.equity_curve.index, y=res.equity_curve.values,
            name="Equity", line=dict(color=ACCENT, width=1.5),
            fill="tozeroy",
            fillcolor="rgba(240,185,11,0.08)",
        ))
    _ys = [float(v) for v in res.equity_curve.values] if len(res.equity_curve) else []
    if _ys:
        _lo, _hi = min(_ys), max(_ys)
        _pad = max(0.5, (_hi - _lo) * 0.12)
        _yr = [_lo - _pad, _hi + _pad]
    else:
        _yr = [0, 1]
    fig.update_layout(**plotly_layout(
        kind="time",
        uirevision="bt-lab-full",
        height=280,
        title=None,
        margin=dict(l=44, r=12, t=8, b=32),
        showlegend=False,
        yaxis=dict(range=_yr, fixedrange=True, title=None),
    ))
    st.plotly_chart(polish_fig(fig), **plotly_kwargs())

    _show_backtest_price_chart(
        lab,
        list(res.trade_list or []),
        book_tf=str(lab.get("book_tf") or cfg0.timeframe),
        key_prefix="bt_pa_full",
    )

    _show_trade_pace(
        list(res.trade_list or []),
        win_start,
        win_end,
        key_prefix="bt_pace_full",
    )

    panel_header("Closed trades", f"{int(res.trades)} rows")
    if res.trade_list:
        st.dataframe(_trades_frame(res.trade_list), width="stretch", hide_index=True, height=320)
    else:
        st.caption("None in this window.")
    if res.halted:
        st.warning(f"Bot stopped itself during the window: {res.halted}")

else:
    older: ms.BacktestResult = lab["older"]
    newer: ms.BacktestResult = lab["newer"]
    c_a, c_b = st.columns(2, gap="small")
    with c_a:
        panel_header("Older period")
        stat_grid(
            [
                {
                    "label": "Return",
                    "value": f"{older.total_return:+.0%}",
                    "tone": "good" if older.total_return >= 0 else "bad",
                },
                {"label": "Win rate", "value": f"{older.win_rate:.0%}"},
                {"label": "Trades", "value": f"{int(older.trades)}"},
                {"label": "Final", "value": f"${older.final_equity:,.0f}", "tone": "accent"},
            ],
            cols=4,
            dense=True,
        )
        st.caption(
            f"max DD {older.max_drawdown:.0%} · avg {pct_of_risk(older.expectancy_r)}"
        )
    with c_b:
        panel_header("Newer period")
        stat_grid(
            [
                {
                    "label": "Return",
                    "value": f"{newer.total_return:+.0%}",
                    "tone": "good" if newer.total_return >= 0 else "bad",
                },
                {"label": "Win rate", "value": f"{newer.win_rate:.0%}"},
                {"label": "Trades", "value": f"{int(newer.trades)}"},
                {"label": "Final", "value": f"${newer.final_equity:,.0f}", "tone": "accent"},
            ],
            cols=4,
            dense=True,
        )
        st.caption(
            f"max DD {newer.max_drawdown:.0%} · avg {pct_of_risk(newer.expectancy_r)}"
        )

    _split_trades = list(older.trade_list or []) + list(newer.trade_list or [])

    fig = go.Figure()
    if len(older.equity_curve):
        fig.add_trace(go.Scatter(
            x=older.equity_curve.index, y=older.equity_curve.values,
            name="Older period", line=dict(color=ACCENT, width=1.5),
        ))
    if len(newer.equity_curve):
        fig.add_trace(go.Scatter(
            x=newer.equity_curve.index, y=newer.equity_curve.values,
            name="Newer period (starts fresh)", line=dict(color=GREEN, width=1.5),
        ))
    if split is not None:
        fig.add_vline(x=split, line_width=1, line_dash="dash", line_color=MUTED)
    _ys = []
    if len(older.equity_curve):
        _ys.extend(float(v) for v in older.equity_curve.values)
    if len(newer.equity_curve):
        _ys.extend(float(v) for v in newer.equity_curve.values)
    if _ys:
        _lo, _hi = min(_ys), max(_ys)
        _pad = max(0.5, (_hi - _lo) * 0.12)
        _yr = [_lo - _pad, _hi + _pad]
    else:
        _yr = [0, 1]
    panel_header("Practice balance", _window)
    fig.update_layout(**plotly_layout(
        kind="time",
        uirevision="bt-lab-split",
        height=280,
        title=None,
        margin=dict(l=44, r=12, t=8, b=32),
        yaxis=dict(range=_yr, fixedrange=True, title=None),
    ))
    st.plotly_chart(polish_fig(fig), **plotly_kwargs())
    st.caption(
        "Jump at the dashed line is normal — newer period restarts from starting balance."
    )

    _show_backtest_price_chart(
        lab,
        _split_trades,
        book_tf=str(lab.get("book_tf") or cfg0.timeframe),
        key_prefix="bt_pa_split",
    )

    _show_trade_pace(
        _split_trades,
        win_start,
        win_end,
        key_prefix="bt_pace_split",
    )

    t1, t2 = st.columns(2, gap="small")
    with t1:
        panel_header("Older-period trades")
        if older.trade_list:
            st.dataframe(
                _trades_frame(older.trade_list), width="stretch",
                hide_index=True, height=260,
            )
        else:
            st.caption("None.")
    with t2:
        panel_header("Newer-period trades")
        if newer.trade_list:
            st.dataframe(
                _trades_frame(newer.trade_list), width="stretch",
                hide_index=True, height=260,
            )
        else:
            st.caption("None.")
