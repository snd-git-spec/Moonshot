"""About — short brand page."""

from __future__ import annotations

import streamlit as st

from dash_lib import (
    inject_theme, topbar, chip_row, load_autopilot_state, load_equity,
    detect_mode, heartbeat_age, format_age,
)

inject_theme()

state = load_autopilot_state()
eq = load_equity(detect_mode(load_equity(None)))
age = heartbeat_age(eq)
score = int((state.get("grade") or state.get("elite") or {}).get("score") or 0)
bal = float(eq["equity"].iloc[-1]) if not eq.empty else None
busy = bool(state.get("busy"))

topbar(balance=bal, score=score, mode="PAPER", active="About")

st.markdown(
    '<div class="ms-page-kicker">About</div>'
    '<div class="ms-home-brand">Moonshot</div>'
    '<p class="ms-home-lead">'
    "Practice-only futures bot. It hunts better settings on paper until the numbers hold up. "
    "Real money stays off until you flip it yourself."
    "</p>",
    unsafe_allow_html=True,
)

if state:
    chip_row([
        ("Autopilot on", "ok"),
        ("Hunting…" if busy else f"Score {score}/100", "busy" if busy else "muted"),
        (f"Paper {format_age(age)}", "ok" if age is not None and age < 180 else "muted"),
    ])
else:
    chip_row([("Not running yet", "warn")])

c1, c2, c3 = st.columns(3)
with c1:
    st.markdown(
        '<div class="ms-step"><div class="ms-step-n">01</div>'
        "<h4>Leave it running</h4>"
        "<p><code>./install_keepalive.sh</code> — paper trading + continuous hunt.</p></div>",
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="ms-step"><div class="ms-step-n">02</div>'
        "<h4>Watch Live</h4>"
        "<p>Score, win rate, balance, and hunt progress — one clean grade.</p></div>",
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="ms-step"><div class="ms-step-n">03</div>'
        "<h4>Stay on practice</h4>"
        "<p>Trades · Status · Health. Live trading never enables itself.</p></div>",
        unsafe_allow_html=True,
    )
