"""Status — is the bot running, what's my practice balance."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dash_lib import (
    inject_theme, db_exists, empty_state, equity_drawdown_fig, fetch_last_prices,
    format_age, heartbeat_age, load_equity, load_positions,
    load_risk, load_trades, detect_mode, paper_status_banner,
    page_header, section_header, topbar, load_autopilot_state, plotly_kwargs,
    polish_fig,
)

inject_theme()

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore
    st_autorefresh(interval=30_000, key="overview_refresh")
except ImportError:
    pass

_eq = load_equity(detect_mode(load_equity(None)))
_bal = float(_eq["equity"].iloc[-1]) if not _eq.empty else None
_score = int((load_autopilot_state().get("grade") or {}).get("score") or 0)
topbar(balance=_bal, score=_score, mode="PAPER", active="Status")
page_header("", "Status", "Balance, open trades, heartbeat — fake money only")

if not db_exists():
    empty_state(
        "Nothing to show yet",
        "Start practice trading in a terminal:<br/>"
        "<code>python -m moonshot paper</code><br/><br/>"
        "Then refresh this page.",
    )
    st.stop()

equity_all = load_equity(None)
mode = detect_mode(equity_all)
equity = load_equity(mode)
trades = load_trades(mode)
risk = load_risk(mode)
positions = load_positions()

age = heartbeat_age(equity)
last_eq = float(equity["equity"].iloc[-1]) if not equity.empty else None
halt = ""
kill = False
daily = False
if risk:
    halt = str(risk.get("halt_reason") or "")
    kill = bool(risk.get("kill"))
    daily = bool(risk.get("daily_halt"))

section_header("What should I do?", f"Last bot update · {format_age(age)}")
paper_status_banner(
    mode=mode,
    age=age,
    kill=kill,
    daily=daily,
    halt=halt,
    last_eq=last_eq,
    n_open=len(positions),
    n_closed=len(trades),
)

c1, c2 = st.columns([3, 2])
with c1:
    section_header("Practice balance", "Teal = balance · red = drop from peak")
    if equity.empty:
        empty_state("Waiting for the first update…", "Usually within a minute after paper starts.")
    else:
        st.plotly_chart(polish_fig(equity_drawdown_fig(equity)), **plotly_kwargs())

with c2:
    section_header("Open now", "Empty is normal between setups")
    if not positions:
        st.info("Flat — no open trades. Waiting for the next setup is normal.")
    else:
        symbols = tuple(positions.keys())
        marks = fetch_last_prices(symbols)
        rows = []
        for sym, pos in positions.items():
            mark = marks.get(sym)
            if mark is not None:
                u = pos.unrealized(mark)
                r = pos.r_multiple(mark)
                mark_s = f"{mark:.6g}"
            else:
                u, r, mark_s = None, None, "—"
            coin = sym.split("/")[0]
            rows.append({
                "Coin": coin,
                "Side": "Buy" if pos.direction.value == "long" else "Sell",
                "Size": round(pos.current_qty, 6),
                "Entry": pos.avg_entry,
                "Now": mark_s,
                "Unrealized $": None if u is None else round(u, 2),
                "vs risk": None if r is None else f"{r:+.0%} of risk",
            })
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

section_header("Latest closed trades")
if trades.empty:
    empty_state("No closed trades yet", "Trend bots can sit flat for hours or days. That is normal.")
else:
    recent = trades.sort_values("ts", ascending=False).head(10).copy()
    show = pd.DataFrame({
        "When": recent["ts"].dt.strftime("%Y-%m-%d %H:%M"),
        "Coin": recent["symbol"].map(lambda s: str(s).split("/")[0]),
        "Side": recent["direction"].map(lambda d: "Buy" if d == "long" else "Sell"),
        "Profit $": recent["pnl"].map(lambda x: round(float(x), 2)),
        "Why closed": recent["exit_reason"],
    })
    st.dataframe(show, width="stretch", hide_index=True)

if st.button("Refresh now"):
    st.cache_data.clear()
    st.rerun()
