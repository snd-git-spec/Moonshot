"""
Moonshot — practice trading console.

    streamlit run dashboard.py
"""

from __future__ import annotations

import hmac
import os
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Moonshot",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def _require_dashboard_password() -> bool:
    """Gate cloud UI when DASHBOARD_PASSWORD is set (local unset = open)."""
    expected = (os.getenv("DASHBOARD_PASSWORD") or "").strip().strip('"').strip("'")
    if not expected:
        return True
    if st.session_state.get("dashboard_authed"):
        return True
    st.markdown("### Moonshot")
    st.caption("Private dashboard — enter password to continue.")
    with st.form("dashboard_login", clear_on_submit=False):
        pw = st.text_input("Password", type="password", autocomplete="current-password")
        ok = st.form_submit_button("Unlock", type="primary")
    if ok:
        got = (pw or "").strip()
        # compare_digest requires equal length; unequal → wrong password (no throw)
        if len(got) == len(expected) and hmac.compare_digest(
            got.encode("utf-8"), expected.encode("utf-8")
        ):
            st.session_state["dashboard_authed"] = True
            st.rerun()
        st.error("Wrong password. If you just changed it in Render, wait for deploy to finish.")
    return False


if not _require_dashboard_password():
    st.stop()

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
