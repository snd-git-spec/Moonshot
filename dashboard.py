"""
Moonshot — practice trading console.

    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Moonshot",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

from dash_lib import inject_theme  # noqa: E402

inject_theme()

_logo = Path(__file__).resolve().parent / "assets" / "moonshot_logo.svg"
if _logo.exists():
    st.logo(str(_logo), size="large")

pages = [
    # Default lives at "/" — keeps bookmarking simple
    st.Page("views/live.py", title="Live", default=True),
    st.Page("views/trades.py", title="Trades", url_path="trades"),
    st.Page("views/backtest.py", title="Backtest", url_path="backtest"),
    st.Page("views/status.py", title="Status", url_path="status"),
    st.Page("views/health.py", title="Health", url_path="health"),
    st.Page("views/home.py", title="About", url_path="about"),
]

# Native top nav (exchange-style) — styled in dash_lib CSS
st.navigation(pages, position="top").run()
