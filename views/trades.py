"""Trades — journal + price chart with entries/exits."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import moonshot as ms
from dash_lib import (
    ACCENT, GREEN, MUTED, RED, inject_theme, db_exists, empty_state, load_trades,
    plotly_layout, plotly_kwargs, polish_fig, detect_mode, load_equity,
    page_header, section_header, pct_of_risk, entries_exits_fig,
    load_symbol_ohlcv, CHART_TIMEFRAMES, chart_bars_for_tf, topbar,
    load_autopilot_state, show_autofit_price_chart,
)

inject_theme()

_eq = load_equity(detect_mode(load_equity(None)))
_bal = float(_eq["equity"].iloc[-1]) if not _eq.empty else None
_score = int((load_autopilot_state().get("grade") or {}).get("score") or 0)
topbar(balance=_bal, score=_score, mode="PAPER", active="Trades")
page_header("", "Trades", "Closed trades · entries ▲/▼ · exits ✕")

if not db_exists():
    empty_state(
        "No trades yet",
        "Run <code>python moonshot.py paper</code> and wait for the bot to close some trades.",
    )
    st.stop()

equity_all = load_equity(None)
mode = detect_mode(equity_all)
trades = load_trades(mode)

if trades.empty:
    empty_state("No closed trades yet.", "Sitting flat for long stretches is normal for this strategy.")
    st.stop()

symbols = sorted(trades["symbol"].dropna().unique().tolist())
directions = sorted(trades["direction"].dropna().unique().tolist())
reasons = sorted(trades["exit_reason"].dropna().unique().tolist())
tmin = trades["ts"].min().to_pydatetime()
tmax = trades["ts"].max().to_pydatetime()

sym_sel = symbols
dir_sel = directions
reason_sel = reasons
dr = (tmin.date(), tmax.date())

with st.expander("Filters", expanded=False):
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        sym_sel = st.multiselect("Coin", symbols, default=symbols)
    with fc2:
        dir_sel = st.multiselect("Side", directions, default=directions)
    with fc3:
        reason_sel = st.multiselect("Exit reason", reasons, default=reasons)
    with fc4:
        dr = st.date_input("Date range", value=(tmin.date(), tmax.date()))

filt = trades[
    trades["symbol"].isin(sym_sel)
    & trades["direction"].isin(dir_sel)
    & trades["exit_reason"].isin(reason_sel)
].copy()
if isinstance(dr, tuple) and len(dr) == 2:
    start = pd.Timestamp(dr[0], tz="UTC")
    end = pd.Timestamp(dr[1], tz="UTC") + pd.Timedelta(days=1)
    filt = filt[(filt["ts"] >= start) & (filt["ts"] < end)]

if filt.empty:
    empty_state("No trades match these filters.")
    st.stop()

costs = filt["fees"].fillna(0) + filt["funding"].fillna(0)
wins = int((filt["pnl"] > 0).sum())
avg_r = float(filt["r_multiple"].mean()) if "r_multiple" in filt.columns and len(filt) else 0.0
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", len(filt))
c2.metric("Win rate", f"{wins / len(filt):.0%}", help=f"{wins} wins / {len(filt)} trades")
c3.metric("Total profit $", f"{filt['pnl'].sum():+.2f}")
c4.metric("Avg profit / trade", pct_of_risk(avg_r), help="Average result vs money risked")

# ---- Entries & exits on price ----
section_header("Entries & exits on chart", "▲ long · ▼ short · ✕ exit · dotted line = hold")
coin_labels = {s: s.split("/")[0] for s in symbols}
default_coin = max(symbols, key=lambda s: int((filt["symbol"] == s).sum()))
chart_sym = st.selectbox(
    "Coin to chart",
    options=symbols,
    index=symbols.index(default_coin) if default_coin in symbols else 0,
    format_func=lambda s: f"{coin_labels.get(s, s)} · {(filt['symbol'] == s).sum()} trades in view",
)
chart_trades = filt[filt["symbol"] == chart_sym].copy()
engine_tf = str(ms.Config.from_env().timeframe or "15m").strip().lower()
tf_opts = list(CHART_TIMEFRAMES)
if engine_tf not in tf_opts:
    tf_opts = [engine_tf] + tf_opts
default_i = tf_opts.index(engine_tf) if engine_tf in tf_opts else tf_opts.index("15m")
chart_tf = st.selectbox(
    "Candle size",
    options=tf_opts,
    index=default_i,
    key="trades_chart_tf",
    help="3m / 5m / 15m / … — chart TF is independent of the bot engine TF.",
)
n_bars = chart_bars_for_tf(str(chart_tf), hours=72)
ohlcv = load_symbol_ohlcv(chart_sym, str(chart_tf), n_bars)
if ohlcv.empty:
    st.warning(
        f"Could not load {chart_tf} candles for this coin (network / exchange). "
        "Try again in a minute."
    )
else:
    win = st.radio(
        "Time window",
        options=["auto", "4h", "12h", "1d", "all"],
        index=0,
        horizontal=True,
        key="trades_chart_window",
        help="Always shows ≥48 candles so the chart stays exchange-dense.",
    )
    max_bars = min(len(ohlcv), chart_bars_for_tf(str(chart_tf), hours=48))
    show_autofit_price_chart(
        entries_exits_fig(
            ohlcv, chart_trades,
            title=f"{coin_labels.get(chart_sym, chart_sym)} · {chart_tf}",
            max_bars=max(120, int(max_bars)),
            view=str(win),
        ),
        height=560,
        key=f"trades_{str(chart_sym).split('/')[0]}_{chart_tf}_{win}",
    )
    st.caption(
        f"{chart_tf} chart (engine {engine_tf}) · "
        "scroll zooms time · drag pans · price auto-fits visible candles · "
        "▲/▼ = entry · ○ = exit"
    )

section_header("Journal")
journal = filt.sort_values("ts", ascending=False).copy()
show = pd.DataFrame({
    "When": journal["ts"].dt.strftime("%Y-%m-%d %H:%M"),
    "Coin": journal["symbol"].map(lambda s: str(s).split("/")[0]),
    "Side": journal["direction"].map(lambda d: "Buy" if d == "long" else "Sell"),
    "Entry": journal["entry"],
    "Exit": journal["exit"] if "exit" in journal.columns else None,
    "Profit $": journal["pnl"].map(lambda x: round(float(x), 2)),
    "vs risk": journal["r_multiple"].map(lambda x: pct_of_risk(float(x))) if "r_multiple" in journal.columns else "—",
    "Why closed": journal["exit_reason"],
    "Fees $": journal["fees"].fillna(0).map(lambda x: round(float(x), 2)),
})
st.dataframe(show, width="stretch", hide_index=True, height=320)

section_header("By coin")
by_sym = filt.groupby("symbol", as_index=False)["pnl"].sum()
by_sym["Coin"] = by_sym["symbol"].map(lambda s: str(s).split("/")[0])
colors = [GREEN if v >= 0 else RED for v in by_sym["pnl"]]
fig = go.Figure(data=[go.Bar(x=by_sym["Coin"], y=by_sym["pnl"], marker_color=colors)])
fig.update_layout(**plotly_layout(
    kind="free",
    uirevision="pnl-by-coin",
    title="Profit by coin ($)", height=300, title_font_size=13,
))
st.plotly_chart(polish_fig(fig), **plotly_kwargs())

with st.expander("More charts", expanded=False):
    g1, g2 = st.columns(2)
    with g1:
        fig = go.Figure(data=[go.Histogram(
            x=filt["r_multiple"], nbinsx=30,
            marker_color=ACCENT, opacity=0.85,
        )])
        fig.update_layout(**plotly_layout(
            kind="free",
            uirevision="r-hist",
            title="Result size vs risk taken", height=280, title_font_size=13,
        ))
        st.plotly_chart(polish_fig(fig), **plotly_kwargs())
    with g2:
        by_ex = filt["exit_reason"].value_counts().reset_index()
        by_ex.columns = ["exit_reason", "n"]
        fig = go.Figure(data=[go.Bar(
            x=by_ex["exit_reason"], y=by_ex["n"], marker_color=MUTED,
        )])
        fig.update_layout(**plotly_layout(
            kind="free",
            uirevision="exit-reasons",
            title="Why trades closed", height=280, title_font_size=13,
        ))
        st.plotly_chart(polish_fig(fig), **plotly_kwargs())
