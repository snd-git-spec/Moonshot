"""Shared helpers for the Moonshot Streamlit console."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

import moonshot as ms

# ---- palette (exchange terminal — MEXC-adjacent) ----
BG = "#0B0E11"
PANEL = "#12161C"
PANEL_2 = "#161A21"
BORDER = "#1E2329"
TEXT = "#EAECEF"
MUTED = "#848E9C"
ACCENT = "#F0B90B"
ACCENT_SOFT = "#F8D12F"
GREEN = "#0ECB81"
GREEN_SOFT = "#2EE6A0"
RED = "#F6465D"
RED_SOFT = "#FF7A8A"
AMBER = "#F0B90B"
AMBER_SOFT = "#F8D12F"
GRID = "#1E2329"

DB_PATH = Path(__file__).resolve().parent / "moonshot.db"
AUTOPILOT_STATE_PATH = Path(__file__).resolve().parent / ".moonshot_autopilot.json"
LIVE_FEED_PATH = Path(__file__).resolve().parent / ".moonshot_live.jsonl"

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@500;600&display=swap');

/* ---- exchange-terminal shell ---- */
html, body, [class*="css"], .stApp, .stMarkdown, .stText, label {
  font-family: 'IBM Plex Sans', system-ui, sans-serif !important;
}
.stApp { background: #0B0E11 !important; color: #EAECEF; }
[data-testid="stAppViewContainer"] { background: #0B0E11 !important; }

/* MEXC-style top header + native nav */
header[data-testid="stHeader"],
.stAppHeader,
header.stAppHeader {
  background: #0B0E11 !important;
  border-bottom: 1px solid #1E2329 !important;
  height: 48px !important;
  min-height: 48px !important;
  box-shadow: none !important;
}
[data-testid="stHeader"] * { font-family: 'IBM Plex Sans', system-ui, sans-serif !important; }
a[data-testid="stTopNavLink"] {
  color: #848E9C !important;
  font-weight: 500 !important;
  font-size: 0.9rem !important;
  text-decoration: none !important;
  padding: 0 0.85rem !important;
  height: 48px !important;
  display: inline-flex !important;
  align-items: center !important;
  border-bottom: 2px solid transparent !important;
  border-radius: 0 !important;
  background: transparent !important;
}
a[data-testid="stTopNavLink"]:hover { color: #EAECEF !important; }
a[data-testid="stTopNavLink"][aria-current="page"] {
  color: #EAECEF !important;
  font-weight: 600 !important;
  border-bottom-color: #F0B90B !important;
}
a[data-testid="stTopNavLink"] p { margin: 0 !important; color: inherit !important; }
/* Desktop: hide empty sidebar (nav is in the header). Mobile keeps sidebar nav. */
@media (min-width: 768px) {
  [data-testid="stSidebar"],
  section[data-testid="stSidebar"],
  [data-testid="stSidebarCollapsedControl"] { display: none !important; }
}
[data-testid="stDecoration"],
#MainMenu, footer { display: none !important; }
.block-container {
  padding: 0.35rem 1.1rem 1.5rem 1.1rem !important;
  max-width: 1400px !important;
}
/* Tighter vertical rhythm between Streamlit blocks */
div[data-testid="stVerticalBlock"] > div { gap: 0.35rem !important; }
div[data-testid="stMetric"] {
  padding: 0.45rem 0.65rem !important;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-size: 1.05rem !important;
}
.ms-panel {
  border: 1px solid #1E2329; border-radius: 6px; background: #12161C;
  padding: 0.65rem 0.75rem 0.7rem; margin: 0 0 0.55rem 0;
}
.ms-panel-head {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 0.75rem; margin: 0 0 0.5rem 0;
}
.ms-panel-head h4 {
  margin: 0 !important; font-size: 0.82rem !important; font-weight: 600 !important;
  color: #EAECEF !important; letter-spacing: 0.02em;
}
.ms-panel-head span { color: #848E9C; font-size: 0.72rem; }
.ms-stat-grid.cols-5 { grid-template-columns: repeat(5, minmax(0, 1fr)); }
.ms-stat-grid.cols-6 { grid-template-columns: repeat(6, minmax(0, 1fr)); }
.ms-stat-grid.dense { gap: 0.4rem; margin: 0; }
.ms-stat-grid.dense .ms-stat { padding: 0.45rem 0.55rem; }
.ms-stat-grid.dense .ms-stat-value { font-size: 1.05rem; }
.ms-stat-grid.dense .ms-stat-label { font-size: 0.62rem; margin-bottom: 0.15rem; }
@media (max-width: 1100px) {
  .ms-stat-grid.cols-5, .ms-stat-grid.cols-6 {
    grid-template-columns: repeat(3, minmax(0, 1fr));
  }
}
@media (max-width: 700px) {
  .ms-stat-grid.cols-5, .ms-stat-grid.cols-6 {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
}
.ms-meta {
  color: #848E9C; font-size: 0.75rem; line-height: 1.35;
  margin: 0 0 0.55rem 0; font-family: 'IBM Plex Mono', ui-monospace, monospace;
}
.ms-split {
  display: grid; grid-template-columns: minmax(140px, 0.85fr) 1.6fr;
  gap: 0.65rem; align-items: stretch;
}
@media (max-width: 900px) {
  .ms-split { grid-template-columns: 1fr; }
}

/* Thin status strip under native nav */
.ms-status {
  display: flex; align-items: center; gap: 0.85rem; flex-wrap: wrap;
  width: calc(100% + 2.5rem); margin: 0 -1.25rem 0.65rem -1.25rem;
  padding: 0.35rem 1.25rem; box-sizing: border-box;
  background: #12161C; border-bottom: 1px solid #1E2329;
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.75rem; color: #848E9C;
}
.ms-status b { color: #EAECEF; font-weight: 600; }
.ms-status .up { color: #0ECB81; }
.ms-status .down { color: #F6465D; }
.ms-status .mode {
  color: #F0B90B; font-weight: 600; letter-spacing: 0.04em;
}

h1, h2, h3 { letter-spacing: -0.02em !important; font-weight: 600 !important; }
div[data-testid="stMetric"] {
  background: #12161C;
  border: 1px solid #1E2329;
  padding: 0.45rem 0.65rem;
  border-radius: 6px;
}
div[data-testid="stMetric"] label {
  color: #848E9C !important; font-size: 0.65rem !important;
  text-transform: uppercase; letter-spacing: 0.06em;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
  font-family: 'IBM Plex Mono', ui-monospace, monospace !important;
  font-weight: 600 !important; font-size: 1.05rem !important;
}

.ms-chip {
  display: inline-block; padding: 0.18rem 0.55rem; border-radius: 4px;
  border: 1px solid #1E2329; font-size: 0.75rem; margin-right: 0.3rem;
  color: #EAECEF; background: #12161C; font-weight: 500;
}
.ms-chip.warn { border-color: #F6465D55; color: #F6465D; background: #F6465D12; }
.ms-chip.ok { border-color: #0ECB8155; color: #0ECB81; background: #0ECB8112; }
.ms-chip.busy { border-color: #F0B90B55; color: #F0B90B; background: #F0B90B12; }
.ms-chip.muted { color: #848E9C; }
.ms-empty {
  border: 1px dashed #1E2329; padding: 1.5rem; border-radius: 6px;
  color: #848E9C; background: #12161C;
}
.ms-empty code { color: #EAECEF; }
.ms-page-kicker {
  color: #848E9C; font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.1rem;
}
.ms-page-title {
  font-size: 1.1rem; font-weight: 600; letter-spacing: -0.02em;
  color: #EAECEF; line-height: 1.2; margin: 0 0 0.15rem 0;
}
.ms-page-sub { color: #848E9C; font-size: 0.82rem; margin: 0 0 0.55rem 0; max-width: 48rem; line-height: 1.4; }
.ms-hero {
  border: 1px solid #1E2329; border-radius: 8px; padding: 1rem 1.15rem 1.1rem;
  background: #12161C; margin-bottom: 0.85rem;
}
.ms-hero.ok { border-color: #0ECB8144; }
.ms-hero.warn { border-color: #F0B90B44; }
.ms-hero.bad { border-color: #F6465D44; }
.ms-hero-label {
  color: #848E9C; font-size: 0.7rem; font-weight: 600;
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.ms-hero-headline {
  font-size: 1.2rem; font-weight: 650; color: #EAECEF;
  letter-spacing: -0.02em; line-height: 1.3; margin: 0 0 0.4rem 0;
}
.ms-hero-detail { color: #848E9C; font-size: 0.88rem; line-height: 1.45; margin: 0; max-width: 48rem; }
.ms-stat-grid {
  display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 0.55rem;
  margin: 0.75rem 0 0.15rem 0;
}
@media (max-width: 900px) {
  .ms-stat-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
.ms-stat {
  background: #0B0E11; border: 1px solid #1E2329;
  border-radius: 6px; padding: 0.7rem 0.8rem;
}
.ms-stat-label {
  color: #848E9C; font-size: 0.68rem; font-weight: 600;
  letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 0.25rem;
}
.ms-stat-value {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 1.35rem; font-weight: 600; color: #EAECEF; line-height: 1.1;
}
.ms-stat-hint { color: #848E9C; font-size: 0.75rem; margin-top: 0.25rem; }
.ms-stat.accent .ms-stat-value { color: #F0B90B; }
.ms-stat.good .ms-stat-value { color: #0ECB81; }
.ms-stat.bad .ms-stat-value { color: #F6465D; }
.ms-section {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 1rem; margin: 0.7rem 0 0.4rem 0;
  border-bottom: 1px solid #1E2329; padding-bottom: 0.3rem;
}
.ms-section h3 {
  margin: 0 !important; font-size: 0.95rem !important; font-weight: 600 !important;
  color: #EAECEF !important;
}
.ms-section span { color: #848E9C; font-size: 0.8rem; }
.ms-goal {
  display: grid; grid-template-columns: 1.4fr 2fr auto; gap: 0.65rem; align-items: center;
  padding: 0.55rem 0.7rem; border: 1px solid #1E2329; border-radius: 6px;
  background: #12161C; margin-bottom: 0.35rem;
}
@media (max-width: 800px) {
  .ms-goal { grid-template-columns: 1fr; gap: 0.3rem; }
}
.ms-goal-title { font-weight: 600; color: #EAECEF; font-size: 0.88rem; }
.ms-goal-detail { color: #848E9C; font-size: 0.75rem; margin-top: 0.1rem; }
.ms-goal-bar {
  height: 6px; border-radius: 2px; background: #0B0E11; border: 1px solid #1E2329;
  overflow: hidden;
}
.ms-goal-fill { height: 100%; border-radius: 2px; background: #F0B90B; }
.ms-goal-fill.ok { background: #0ECB81; }
.ms-goal-fill.bad { background: #F6465D; }
.ms-goal-pct {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  font-size: 0.8rem; font-weight: 600; color: #EAECEF; text-align: right; min-width: 3.2rem;
}
.ms-goal.pass { border-color: #0ECB8144; }
.ms-goal.fail { border-color: #F6465D33; }
.ms-strip {
  display: flex; flex-wrap: wrap; gap: 0.35rem; align-items: center;
  margin: 0.15rem 0 0.75rem 0;
}
.ms-home-brand {
  font-size: clamp(2.2rem, 5vw, 3.2rem); font-weight: 700; letter-spacing: -0.04em;
  color: #EAECEF; line-height: 1; margin: 0.15rem 0 0.5rem 0;
}
.ms-home-brand span { color: #F0B90B; }
.ms-home-lead { color: #848E9C; font-size: 1.05rem; max-width: 34rem; line-height: 1.45; margin-bottom: 1rem; }
.ms-step {
  border: 1px solid #1E2329; border-radius: 6px; padding: 0.9rem 1rem;
  background: #12161C; height: 100%;
}
.ms-step-n {
  font-family: 'IBM Plex Mono', ui-monospace, monospace;
  color: #F0B90B; font-weight: 600; font-size: 0.75rem; letter-spacing: 0.08em;
  margin-bottom: 0.3rem;
}
.ms-step h4 { margin: 0 0 0.3rem 0; color: #EAECEF; font-size: 0.98rem; }
.ms-step p { margin: 0; color: #848E9C; font-size: 0.85rem; line-height: 1.45; }

/* Expanders / buttons — flatter */
div[data-testid="stExpander"] {
  border: 1px solid #1E2329 !important; border-radius: 6px !important;
  background: #12161C !important;
}
.stButton > button {
  border-radius: 4px !important; border: 1px solid #1E2329 !important;
  background: #12161C !important; color: #EAECEF !important;
  font-weight: 500 !important;
}
.stButton > button:hover { border-color: #F0B90B !important; color: #F0B90B !important; }
</style>
"""


def inject_theme() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def page_setup(title: str = "") -> None:
    """Theme only. set_page_config lives in dashboard.py (once)."""
    inject_theme()
    if title:
        pass


def topbar(balance: Optional[float] = None, score: Optional[int] = None,
           mode: str = "PAPER", active: Optional[str] = None) -> None:
    """Thin status strip under native top nav (balance / score / mode)."""
    _ = active  # page highlight is handled by st.navigation(position="top")
    bits = [f'<span class="mode">{mode}</span>']
    if balance is not None:
        cls = "up" if balance >= 50 else "down"
        bits.append(f'<span>Bal <b class="{cls}">${balance:,.2f}</b></span>')
    if score is not None:
        bits.append(f'<span>Score <b>{int(score)}/100</b></span>')
    st.markdown(
        f'<div class="ms-status">{"".join(bits)}</div>',
        unsafe_allow_html=True,
    )


def page_header(kicker: str, title: str, subtitle: str = "") -> None:
    html = ""
    if kicker:
        html += f'<div class="ms-page-kicker">{kicker}</div>'
    if title:
        html += f'<div class="ms-page-title">{title}</div>'
    if subtitle:
        html += f'<p class="ms-page-sub">{subtitle}</p>'
    if html:
        st.markdown(html, unsafe_allow_html=True)


def section_header(title: str, hint: str = "") -> None:
    hint_html = f"<span>{hint}</span>" if hint else ""
    st.markdown(
        f'<div class="ms-section"><h3>{title}</h3>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def wide(fig_or_df_kwargs: Optional[dict] = None) -> dict:
    """Plotly/dataframe width helper — Streamlit 1.66 prefers width='stretch'."""
    return {"width": "stretch"}


def empty_state(msg: str, hint: str = "") -> None:
    body = f'<div class="ms-empty"><strong>{msg}</strong>'
    if hint:
        body += f"<br/><br/>{hint}"
    body += "</div>"
    st.markdown(body, unsafe_allow_html=True)


def chip_row(items: list[tuple[str, str]]) -> None:
    """items: list of (label, kind) where kind in ok|warn|muted|busy|''."""
    html = '<div class="ms-strip">' + "".join(
        f'<span class="ms-chip {kind}">{label}</span>' for label, kind in items
    ) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def panel_header(title: str, hint: str = "") -> None:
    """Compact in-panel section title (tighter than section_header)."""
    hint_html = f"<span>{hint}</span>" if hint else ""
    st.markdown(
        f'<div class="ms-panel-head"><h4>{title}</h4>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def stat_grid(
    stats: list[dict],
    *,
    cols: int = 4,
    dense: bool = True,
) -> None:
    """Exchange-terminal KPI strip. stats: [{label, value, tone?, hint?}]."""
    n = max(2, min(int(cols), 6))
    tiles = []
    for s in stats:
        tone = str(s.get("tone") or "")
        hint = f'<div class="ms-stat-hint">{s["hint"]}</div>' if s.get("hint") else ""
        tiles.append(
            f'<div class="ms-stat {tone}">'
            f'<div class="ms-stat-label">{s["label"]}</div>'
            f'<div class="ms-stat-value">{s["value"]}</div>'
            f"{hint}</div>"
        )
    klass = f"ms-stat-grid cols-{n}" + (" dense" if dense else "")
    st.markdown(
        f'<div class="{klass}">{"".join(tiles)}</div>',
        unsafe_allow_html=True,
    )


def meta_line(text: str) -> None:
    st.markdown(f'<p class="ms-meta">{text}</p>', unsafe_allow_html=True)


def pct_of_risk(exp: float) -> str:
    """+0.22 → '+22% of risk'."""
    return f"{exp:+.0%} of risk"


def hero_panel(level: str, headline: str, detail: str, stats: list[dict]) -> None:
    """
    Primary answer block.
    level: ok|warn|bad|info
    stats: [{label, value, hint?, tone?}] tone in accent|good|bad|''
    """
    klass = {"ok": "ok", "success": "ok", "warn": "warn", "warning": "warn",
             "bad": "bad", "error": "bad"}.get(level, "")
    tiles = []
    for s in stats:
        tone = s.get("tone", "")
        hint = f'<div class="ms-stat-hint">{s["hint"]}</div>' if s.get("hint") else ""
        tiles.append(
            f'<div class="ms-stat {tone}">'
            f'<div class="ms-stat-label">{s["label"]}</div>'
            f'<div class="ms-stat-value">{s["value"]}</div>'
            f'{hint}</div>'
        )
    n = max(2, min(len(stats) or 2, 6))
    grid = f"ms-stat-grid cols-{n}" if n != 4 else "ms-stat-grid"
    st.markdown(
        f'<div class="ms-hero {klass}">'
        f'<div class="ms-hero-label">Strategy quality</div>'
        f'<div class="ms-hero-headline">{headline}</div>'
        f'<p class="ms-hero-detail">{detail}</p>'
        f'<div class="{grid}">{"".join(tiles)}</div>'
        f"</div>",
        unsafe_allow_html=True,
    )


def goal_rows(items: list[dict]) -> None:
    """
    items: {title, detail, value, goal, higher_better=True, passed?}
    Renders visual progress toward each goal.
    """
    blocks = []
    for it in items:
        value = float(it.get("value") or 0)
        goal = float(it.get("goal") or 1)
        higher = bool(it.get("higher_better", True))
        if higher:
            pct = 0.0 if goal <= 0 else max(0.0, min(1.0, value / goal))
            passed = it.get("passed", value >= goal)
        else:
            # lower is better (drawdown): full bar when at/under goal
            pct = 0.0 if goal <= 0 else max(0.0, min(1.0, 1.0 - (value / goal) + 1e-9))
            if value <= goal:
                pct = 1.0
            else:
                pct = max(0.0, min(0.99, goal / value if value else 0.0))
            passed = it.get("passed", value <= goal)
        fill = "ok" if passed else ("bad" if pct < 0.7 else "")
        row_klass = "pass" if passed else "fail"
        mark = "✓" if passed else "✗"
        blocks.append(
            f'<div class="ms-goal {row_klass}">'
            f'<div><div class="ms-goal-title">{mark} {it["title"]}</div>'
            f'<div class="ms-goal-detail">{it.get("detail", "")}</div></div>'
            f'<div class="ms-goal-bar"><div class="ms-goal-fill {fill}" style="width:{pct*100:.0f}%"></div></div>'
            f'<div class="ms-goal-pct">{pct*100:.0f}%</div>'
            f"</div>"
        )
    st.markdown("".join(blocks), unsafe_allow_html=True)


def db_exists(path: Path = DB_PATH) -> bool:
    return path.exists() and path.stat().st_size > 0


def _read_sql(query: str, params: tuple = (), path: Path = DB_PATH) -> pd.DataFrame:
    if not db_exists(path):
        return pd.DataFrame()
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    for attempt in range(5):
        try:
            conn = sqlite3.connect(uri, uri=True, timeout=10.0)
            try:
                return pd.read_sql_query(query, conn, params=params)
            finally:
                conn.close()
        except sqlite3.OperationalError:
            if attempt == 4:
                raise
            time.sleep(0.15 * (attempt + 1))
    return pd.DataFrame()


def _table_cols(path: Path, table: str) -> set:
    df = _read_sql(f"PRAGMA table_info({table})", path=path)
    if df.empty:
        return set()
    return set(df["name"].tolist())


def load_equity(mode: Optional[str] = None, path: Path = DB_PATH) -> pd.DataFrame:
    cols = _table_cols(path, "equity")
    if not cols:
        return pd.DataFrame()
    select = ["ts", "equity", "peak", "drawdown"]
    if "mode" in cols:
        select.append("mode")
    q = f"SELECT {', '.join(select)} FROM equity"
    args: tuple = ()
    if mode and "mode" in cols:
        q += " WHERE COALESCE(mode, ?) = ?"
        args = (mode, mode)
    q += " ORDER BY ts"
    df = _read_sql(q, args, path=path)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601", errors="coerce")
    if "mode" not in df.columns:
        df["mode"] = None
    return df


def load_trades(mode: Optional[str] = None, path: Path = DB_PATH) -> pd.DataFrame:
    cols = _table_cols(path, "trades")
    if not cols:
        return pd.DataFrame()
    # exit may need quoting
    wanted = ["ts", "symbol", "direction", "entry", "exit", "qty", "pnl", "r_multiple",
              "liquidated", "exit_reason", "adds", "fees", "funding",
              "entry_time", "partial_taken", "mode"]
    select = []
    for c in wanted:
        if c not in cols:
            continue
        select.append(f'"{c}"' if c == "exit" else c)
    q = f"SELECT {', '.join(select)} FROM trades"
    args: tuple = ()
    if mode and "mode" in cols:
        q += " WHERE COALESCE(mode, ?) = ?"
        args = (mode, mode)
    q += " ORDER BY ts"
    df = _read_sql(q, args, path=path)
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601", errors="coerce")
    if "entry_time" in df.columns:
        df["entry_time"] = pd.to_datetime(
            df["entry_time"], utc=True, format="ISO8601", errors="coerce",
        )
    else:
        df["entry_time"] = pd.NaT
    if "partial_taken" not in df.columns:
        df["partial_taken"] = 0
    if "mode" not in df.columns:
        df["mode"] = None
    for c in list(df.columns):
        if c.lower() == "exit" and c != "exit":
            df = df.rename(columns={c: "exit"})
    return df


def load_positions() -> dict:
    df = _read_sql("SELECT symbol, state FROM positions")
    out = {}
    for _, row in df.iterrows():
        try:
            out[row["symbol"]] = ms.Position.from_dict(json.loads(row["state"]))
        except Exception:
            continue
    return out


def load_risk(mode: Optional[str] = None) -> Optional[dict]:
    q = """SELECT ts, mode, equity, peak, drawdown, halt_reason, kill, streak,
                  daily_halt, pause_until FROM risk_snapshot"""
    args: list = []
    if mode:
        q += " WHERE mode=?"
        args.append(mode)
    q += " ORDER BY ts DESC LIMIT 1"
    try:
        df = _read_sql(q, tuple(args))
    except Exception:
        return None
    if df.empty:
        return None
    return df.iloc[0].to_dict()


def load_events(mode: Optional[str] = None, limit: int = 200) -> pd.DataFrame:
    q = "SELECT ts, mode, symbol, kind, payload FROM events"
    args: list = []
    if mode:
        q += " WHERE mode=?"
        args.append(mode)
    q += " ORDER BY ts DESC LIMIT ?"
    args.append(limit)
    try:
        df = _read_sql(q, tuple(args))
    except Exception:
        return pd.DataFrame()
    if not df.empty:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601", errors="coerce")
    return df


def detect_mode(equity: pd.DataFrame) -> str:
    if equity.empty or "mode" not in equity.columns:
        return "paper"
    modes = equity["mode"].dropna().astype(str)
    if modes.empty:
        return "paper"
    return str(modes.iloc[-1])


def heartbeat_age(equity: pd.DataFrame) -> Optional[float]:
    """Seconds since last equity write, or None."""
    if equity.empty:
        return None
    last = equity["ts"].iloc[-1]
    now = pd.Timestamp.now(tz="UTC")
    return float((now - last).total_seconds())


def format_age(seconds: Optional[float]) -> str:
    if seconds is None:
        return "no heartbeat"
    if seconds < 90:
        return f"{int(seconds)}s ago"
    if seconds < 3600:
        return f"{int(seconds / 60)}m ago"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h ago"
    return f"{seconds / 86400:.1f}d ago"


def heartbeat_kind(seconds: Optional[float], poll: float = 30.0) -> str:
    if seconds is None:
        return "muted"
    if seconds <= poll * 4:
        return "ok"
    if seconds <= poll * 20:
        return "muted"
    return "warn"


@st.cache_data(ttl=20, show_spinner=False)
def fetch_last_prices(symbols: tuple) -> dict:
    if not symbols:
        return {}
    try:
        ex = ms.public_exchange()
        out = {}
        for s in symbols:
            try:
                out[s] = float(ex.fetch_ticker(s)["last"])
            except Exception:
                continue
        return out
    except Exception:
        return {}


# Exchange-grade interaction:
#   scroll / box-zoom → time only · drag → pan · no bottom rangeslider · no photo pinch
PLOTLY_CONFIG: dict = {
    "displayModeBar": True,
    "displaylogo": False,
    "responsive": True,
    "scrollZoom": True,
    "doubleClick": "reset",
    "modeBarButtonsToRemove": [
        "lasso2d", "select2d", "autoScale2d",
        "toggleSpikelines", "hoverClosestCartesian", "hoverCompareCartesian",
    ],
}


def plotly_kwargs(**extra) -> dict:
    """Canonical st.plotly_chart kwargs."""
    cfg = dict(PLOTLY_CONFIG)
    user_cfg = extra.pop("config", None)
    if isinstance(user_cfg, dict):
        cfg.update(user_cfg)
    out = dict(width="stretch", theme=None, config=cfg)
    out.update(extra)
    return out


def polish_fig(fig: go.Figure) -> go.Figure:
    """Hard-disable Plotly's bottom rangeslider (candlesticks re-enable it by default)."""
    fig.update_xaxes(rangeslider_visible=False, rangeselector=None)
    fig.update_layout(xaxis_rangeslider_visible=False)
    # Belt-and-suspenders for multi-axis figures
    try:
        if fig.layout.xaxis is not None:
            fig.layout.xaxis.rangeslider = dict(visible=False)
    except Exception:
        pass
    return fig


def show_plotly(fig: go.Figure, **extra) -> None:
    st.plotly_chart(polish_fig(fig), **plotly_kwargs(**extra))


def _plotly_plain_json(fig: go.Figure) -> str:
    """Serialize a figure with real arrays — Plotly's default uses binary `bdata`
    blobs that break custom JS reading `trace.high[i]` / `trace.low[i]`."""
    import base64
    import json

    import numpy as np

    _DTYPE = {
        "f8": "<f8", "f4": "<f4",
        "i1": "i1", "i2": "<i2", "i4": "<i4",
        "u1": "u1", "u2": "<u2", "u4": "<u4",
    }

    def _expand(obj):
        if isinstance(obj, dict) and "bdata" in obj and "dtype" in obj:
            raw = base64.b64decode(obj["bdata"])
            dt = _DTYPE.get(str(obj.get("dtype") or "f8"), "<f8")
            return np.frombuffer(raw, dtype=dt).tolist()
        if isinstance(obj, dict):
            return {k: _expand(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_expand(v) for v in obj]
        return obj

    # to_json already stringifies timestamps; we only need to explode bdata blobs
    return json.dumps(_expand(json.loads(fig.to_json())), separators=(",", ":"))


def show_autofit_price_chart(fig: go.Figure, *, height: int = 560, key: str = "price") -> None:
    """Candlestick chart: when you zoom/pan time, price axis re-fits the visible candles.

    Streamlit's native plotly_chart keeps a fixed Y after X zoom (clips wicks). This
    renderer listens for time-axis changes and reframes Y like MEXC / TradingView.
    """
    import json
    import streamlit.components.v1 as components

    fig = polish_fig(fig)
    # Drop uirevision so programmatic Y refits aren't fought by Plotly state restore
    fig.update_layout(uirevision=None)
    fig.update_xaxes(uirevision=None)
    fig.update_yaxes(uirevision=None)
    # User scrolls/pans time only; JS owns the price scale.
    fig.update_yaxes(fixedrange=True, autorange=False, row=1, col=1)
    cfg = dict(PLOTLY_CONFIG)
    remove = [b for b in (cfg.get("modeBarButtonsToRemove") or []) if b != "autoScale2d"]
    cfg["modeBarButtonsToRemove"] = remove

    fig_json = _plotly_plain_json(fig)
    cfg_json = json.dumps(cfg)
    safe_key = "".join(ch if ch.isalnum() else "_" for ch in str(key))[:48]
    div_id = f"ms_price_{safe_key}"

    html = f"""
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8"/>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body {{ margin:0; padding:0; background:#0B0E11; overflow:hidden; }}
    #{div_id} {{ width:100%; height:{int(height)}px; }}
  </style>
</head>
<body>
  <div id="{div_id}"></div>
  <script>
  (function() {{
    const fig = {fig_json};
    const cfg = {cfg_json};
    const gd = document.getElementById("{div_id}");
    if (!gd || typeof Plotly === "undefined") return;

    function toMs(v) {{
      if (v == null) return NaN;
      if (typeof v === "number") return v;
      if (v instanceof Date) return v.getTime();
      const t = Date.parse(v);
      return Number.isFinite(t) ? t : NaN;
    }}

    function asNumArr(v) {{
      if (!v) return [];
      if (Array.isArray(v)) return v;
      // Defensive: if binary slipped through, ignore (Python side expands bdata)
      return [];
    }}

    function visibleXRange() {{
      const xa = (gd._fullLayout && gd._fullLayout.xaxis) || {{}};
      const r = xa.range;
      if (!r || r.length < 2) return null;
      const a = toMs(r[0]), b = toMs(r[1]);
      if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
      return a <= b ? [a, b] : [b, a];
    }}

    function priceAxisKey(tr) {{
      // candlestick on subplot row1 → "y" → layout key "yaxis"
      const y = (tr && tr.yaxis) ? String(tr.yaxis) : "y";
      if (y === "y" || y === "y1") return "yaxis";
      return "yaxis" + y.replace(/^y/, "");
    }}

    function fitPriceToVisible() {{
      const range = visibleXRange();
      if (!range) return;
      const [x0, x1] = range;
      let lo = Infinity, hi = -Infinity, n = 0;
      let axisKey = "yaxis";
      const data = gd.data || [];
      for (const tr of data) {{
        if (!tr || !tr.x) continue;
        const xs = asNumArr(tr.x).length ? tr.x : tr.x;
        if (tr.type === "candlestick") {{
          axisKey = priceAxisKey(tr);
          const lows = asNumArr(tr.low);
          const highs = asNumArr(tr.high);
          const nBar = Math.min(xs.length, lows.length, highs.length);
          for (let i = 0; i < nBar; i++) {{
            const xi = toMs(xs[i]);
            if (!(xi >= x0 && xi <= x1)) continue;
            const l = Number(lows[i]), h = Number(highs[i]);
            if (!Number.isFinite(l) || !Number.isFinite(h)) continue;
            if (l < lo) lo = l;
            if (h > hi) hi = h;
            n++;
          }}
        }} else if (tr.type === "scatter" && tr.y) {{
          const ys = asNumArr(tr.y);
          const nPt = Math.min(xs.length, ys.length);
          for (let i = 0; i < nPt; i++) {{
            const xi = toMs(xs[i]);
            if (!(xi >= x0 && xi <= x1)) continue;
            const y = Number(ys[i]);
            if (!Number.isFinite(y)) continue;
            if (y < lo) lo = y;
            if (y > hi) hi = y;
            n++;
          }}
        }}
      }}
      if (!n || !(lo <= hi) || !Number.isFinite(lo) || !Number.isFinite(hi)) return;
      if (lo === hi) {{
        const eps = Math.max(Math.abs(lo) * 1e-4, 1.0);
        lo -= eps; hi += eps;
      }}
      const pad = Math.max((hi - lo) * 0.12, Math.abs(hi) * 2e-4, 1e-6);
      const next = [lo - pad, hi + pad];
      const patch = {{}};
      patch[axisKey + ".autorange"] = false;
      patch[axisKey + ".range"] = next;
      Plotly.relayout(gd, patch);
    }}

    Plotly.newPlot(gd, fig.data, fig.layout, cfg).then(function() {{
      fitPriceToVisible();
      setTimeout(fitPriceToVisible, 50);
      setTimeout(fitPriceToVisible, 200);
      let timer = null;
      const schedule = function() {{
        clearTimeout(timer);
        timer = setTimeout(fitPriceToVisible, 30);
      }};
      gd.on("plotly_relayout", schedule);
      gd.on("plotly_relayouting", schedule);
      gd.on("plotly_afterplot", schedule);
      window.addEventListener("resize", schedule);
    }});
  }})();
  </script>
</body>
</html>
"""
    components.html(html, height=int(height) + 8, scrolling=False)


def plotly_layout(*, kind: str = "time", **kwargs) -> dict:
    """Exchange-style layout: pan/scroll time only; never show a bottom scrubber."""
    kind = str(kwargs.pop("kind", kind) or "time")
    uirev = kwargs.pop("uirevision", "moonshot")
    time_mode = kind != "free"
    xaxis = dict(
        gridcolor=GRID, zeroline=False, showline=True, linecolor=BORDER,
        fixedrange=False,
        rangeslider=dict(visible=False),
        rangeselector=None,
    )
    yaxis = dict(
        gridcolor=GRID, zeroline=False, showline=True, linecolor=BORDER,
        fixedrange=bool(time_mode),
        autorange=False if time_mode else True,
    )
    if "xaxis" in kwargs and isinstance(kwargs["xaxis"], dict):
        user_x = kwargs.pop("xaxis")
        xaxis.update(user_x)
        # Never allow a caller to turn the bottom scrubber back on
        xaxis["rangeslider"] = dict(visible=False)
        xaxis["rangeselector"] = None
    if "yaxis" in kwargs and isinstance(kwargs["yaxis"], dict):
        yaxis.update(kwargs.pop("yaxis"))
    base = dict(
        paper_bgcolor=BG,
        plot_bgcolor=PANEL,
        font=dict(color=TEXT, size=12),
        margin=dict(l=48, r=48, t=36, b=36),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=MUTED)),
        xaxis=xaxis,
        yaxis=yaxis,
        hovermode="x unified",
        dragmode="pan" if time_mode else "zoom",
        uirevision=uirev,
    )
    base.update(kwargs)
    return base


def _y_range_for_window(
    lows: list[float] | tuple[float, ...] | pd.Series,
    highs: list[float] | tuple[float, ...] | pd.Series,
    *,
    pad_frac: float = 0.08,
) -> list[float]:
    """Price-axis framing for the candles currently in view (platform-style)."""
    lo = float(pd.Series(lows).min())
    hi = float(pd.Series(highs).max())
    if not (lo == lo and hi == hi):  # NaN
        return [0.0, 1.0]
    span = max(hi - lo, abs(hi) * 1e-6, 1e-9)
    pad = span * float(pad_frac)
    return [lo - pad, hi + pad]


def equity_drawdown_fig(equity: pd.DataFrame) -> go.Figure:
    """Single clean equity line. Peak as a quiet dotted guide — no dual axis, no DD scribble."""
    fig = go.Figure()
    if equity.empty:
        return fig
    # Paper ticks often reuse a bar timestamp → vertical scribble unless we collapse.
    clean = (
        equity.dropna(subset=["ts", "equity"])
        .sort_values("ts")
        .drop_duplicates(subset=["ts"], keep="last")
    )
    if clean.empty:
        return fig
    # Downsample very dense series so the line stays readable
    if len(clean) > 1200:
        step = max(1, len(clean) // 900)
        clean = pd.concat([clean.iloc[::step], clean.iloc[[-1]]]).drop_duplicates(subset=["ts"])
    ts = [pd.Timestamp(t).isoformat() for t in clean["ts"]]
    vals = [float(v) for v in clean["equity"]]
    peak_vals: list[float] = []
    running = vals[0]
    for v in vals:
        running = max(running, v)
        peak_vals.append(running)
    last, start = vals[-1], vals[0]
    color = GREEN if last >= start else RED
    fig.add_trace(go.Scatter(
        x=ts, y=peak_vals, name="Peak",
        line=dict(color="rgba(139,147,167,0.35)", width=1, dash="dot"),
        hovertemplate="Peak $%{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=ts, y=vals, name="Balance",
        line=dict(color=color, width=2.4),
        fill="tonexty",
        fillcolor="rgba(184,92,92,0.10)",
        hovertemplate="Balance $%{y:.2f}<extra></extra>",
        showlegend=False,
    ))
    y_range = _y_range_for_window(vals, peak_vals, pad_frac=0.12)
    # Time-only zoom (pan/scroll); price axis framed to series — not photo-pinch
    fig.update_layout(**plotly_layout(
        kind="time",
        uirevision="equity-balance",
        title=None,
        margin=dict(l=48, r=16, t=12, b=40),
        yaxis=dict(title="Equity $", range=y_range, fixedrange=True),
        xaxis=dict(title="Time (UTC)", fixedrange=False),
        height=340,
        showlegend=False,
    ))
    return polish_fig(fig)


def _ohlcv_frame(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Normalize OHLCV to a UTC DatetimeIndex with open/high/low/close."""
    df = ohlcv.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        elif "ts" in df.columns:
            df = df.set_index("ts")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    else:
        df.index = df.index.tz_convert("UTC")
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def _bar_delta(df: pd.DataFrame) -> pd.Timedelta:
    if len(df) >= 3:
        deltas = pd.Series(df.index).diff().dropna()
        bar = deltas.median() if len(deltas) else pd.Timedelta(minutes=15)
    else:
        bar = pd.Timedelta(minutes=15)
    if not isinstance(bar, pd.Timedelta) or pd.isna(bar) or bar <= pd.Timedelta(0):
        return pd.Timedelta(minutes=15)
    return bar


def _snap_to_index(ts: pd.Timestamp, idx: pd.DatetimeIndex) -> pd.Timestamp:
    if pd.isna(ts) or idx is None or len(idx) == 0:
        return pd.NaT
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    if ts in idx:
        return ts
    pos = int(idx.searchsorted(ts, side="right")) - 1
    if pos < 0:
        return pd.Timestamp(idx[0])
    if pos >= len(idx):
        return pd.Timestamp(idx[-1])
    return pd.Timestamp(idx[pos])


def _align_trades_to_candles(trades: pd.DataFrame, df: pd.DataFrame,
                             bar: pd.Timedelta) -> pd.DataFrame:
    """Snap each trade to candle bars (no visual x-stagger — that caused the spider mess)."""
    if trades is None or trades.empty or df.empty:
        return pd.DataFrame()

    t = trades.copy()
    if "entry_time" in t.columns:
        t["entry_time"] = pd.to_datetime(t["entry_time"], utc=True, errors="coerce")
    else:
        t["entry_time"] = pd.NaT
    t["ts"] = pd.to_datetime(t["ts"], utc=True, errors="coerce")
    t = t.dropna(subset=["ts"]).sort_values("ts")
    idx = df.index

    rows = []
    for _, row in t.iterrows():
        exit_ts = pd.Timestamp(row["ts"])
        entry_ts = row.get("entry_time")
        ep = row.get("entry")
        xp = row.get("exit")
        if ep is None or xp is None or pd.isna(ep) or pd.isna(xp):
            continue

        exit_bar = _snap_to_index(exit_ts, idx)
        entry_bar = pd.NaT
        if pd.notna(entry_ts):
            et = pd.Timestamp(entry_ts)
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            else:
                et = et.tz_convert("UTC")
            # Wall-clock after exit (paper stamp bug) → treat as same-bar
            if et <= exit_ts + bar:
                entry_bar = _snap_to_index(et, idx)
                if pd.notna(entry_bar) and entry_bar > exit_bar:
                    entry_bar = exit_bar

        if pd.isna(entry_bar):
            pos_x = int(idx.searchsorted(exit_bar, side="left"))
            lo = max(0, pos_x - 4)
            window = df.iloc[lo: pos_x + 1]
            entry_bar = exit_bar
            if not window.empty and "open" in window.columns:
                err = (window["open"].astype(float) - float(ep)).abs()
                rel = err / max(abs(float(ep)), 1e-12)
                best_i = rel.idxmin()
                if float(rel.loc[best_i]) <= 5e-4:
                    entry_bar = pd.Timestamp(best_i)

        same = bool(entry_bar == exit_bar)
        try:
            bars_held = max(0, int((exit_bar - entry_bar) / bar))
        except Exception:
            bars_held = 0

        d = str(row.get("direction", "")).lower()
        is_long = d in ("long", "buy")

        rows.append({
            **row.to_dict(),
            "entry_bar": entry_bar,
            "exit_bar": exit_bar,
            "bars_held": bars_held,
            "same_bar": same,
            "is_long": is_long,
        })
    return pd.DataFrame(rows)


def _resolve_chart_window(
    df: pd.DataFrame,
    t: pd.DataFrame,
    bar: pd.Timedelta,
    view: str,
    *,
    min_bars: int = 48,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Time window that always shows enough candles (never 4 fat bars)."""
    last_bar = pd.Timestamp(df.index[-1])
    end = last_bar + bar
    start = pd.Timestamp(df.index[0])
    min_span = bar * max(12, int(min_bars))
    v = str(view or "auto").strip().lower()
    hours = {"1h": 1, "2h": 2, "4h": 4, "6h": 6, "12h": 12, "1d": 24, "all": None}.get(v, "auto")

    def _clamp(a: pd.Timestamp, b: pd.Timestamp) -> tuple[pd.Timestamp, pd.Timestamp]:
        # Clip to data first, THEN enforce min candle count (avoids fat 4-bar windows)
        a = max(start, pd.Timestamp(a))
        b = min(end, pd.Timestamp(b))
        if b < a:
            a, b = start, end
        if b - a < min_span:
            b = min(end, a + min_span)
            if b - a < min_span:
                a = max(start, b - min_span)
        return a, b

    if hours is None:
        return start, end
    if isinstance(hours, int):
        return _clamp(end - pd.Timedelta(hours=hours), end)
    # auto → context around trades, else last ~12h — always ≥ min_bars
    if not t.empty:
        t0 = min(t["entry_bar"].min(), t["exit_bar"].min())
        t1 = max(t["entry_bar"].max(), t["exit_bar"].max())
        pad = max(bar * 24, pd.Timedelta(hours=2))
        return _clamp(pd.Timestamp(t0) - pad, pd.Timestamp(t1) + pad)
    return _clamp(end - pd.Timedelta(hours=12), end)


def _marker_stacks(
    t: pd.DataFrame,
    df: pd.DataFrame,
    bar: pd.Timedelta,
    pad: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TradingView-style: entries/exits sit outside wicks; stack on shared bars."""
    if t.empty:
        return pd.DataFrame(), pd.DataFrame()

    # Mid-bar x keeps markers centered on the candle (no float-past-last-bar)
    half = bar * 0.5
    entry_counts: dict[pd.Timestamp, int] = {}
    exit_counts: dict[pd.Timestamp, int] = {}
    entries, exits = [], []

    for _, row in t.iterrows():
        eb = pd.Timestamp(row["entry_bar"])
        xb = pd.Timestamp(row["exit_bar"])
        is_long = bool(row["is_long"])
        pnl = float(row["pnl"]) if "pnl" in row and pd.notna(row["pnl"]) else 0.0
        reason = str(row.get("exit_reason") or "")
        ep = float(row["entry"])
        xp = float(row["exit"])

        # Entry: buy below low / sell above high
        e_slot = entry_counts.get(eb, 0)
        entry_counts[eb] = e_slot + 1
        if eb in df.index:
            hi = float(df.loc[eb, "high"])
            lo = float(df.loc[eb, "low"])
        else:
            hi = lo = ep
        if is_long:
            ey = lo - pad * (1.15 + 0.85 * e_slot)
            esym = "triangle-up"
            ecol = GREEN
            ename = "Long"
        else:
            ey = hi + pad * (1.15 + 0.85 * e_slot)
            esym = "triangle-down"
            ecol = RED
            ename = "Short"
        entries.append({
            "x": eb + half, "y": ey, "px": ep, "sym": esym, "color": ecol,
            "name": ename, "bars": int(row.get("bars_held") or 0),
            "same": bool(row.get("same_bar")),
        })

        # Exit: opposite side of the bar (closes the position)
        x_slot = exit_counts.get(xb, 0)
        exit_counts[xb] = x_slot + 1
        if xb in df.index:
            hi = float(df.loc[xb, "high"])
            lo = float(df.loc[xb, "low"])
        else:
            hi = lo = xp
        if is_long:
            # cover long → sell marker above
            xy = hi + pad * (1.15 + 0.85 * x_slot)
        else:
            # cover short → buy marker below
            xy = lo - pad * (1.15 + 0.85 * x_slot)
        exits.append({
            "x": xb + half, "y": xy, "px": xp, "pnl": pnl, "reason": reason,
            "win": pnl >= 0, "bars": int(row.get("bars_held") or 0),
            "same": bool(row.get("same_bar")),
            "name": "Exit",
        })

    return pd.DataFrame(entries), pd.DataFrame(exits)


def entries_exits_fig(
    ohlcv: pd.DataFrame,
    trades: pd.DataFrame,
    *,
    title: str = "Entries & exits",
    max_trades: int = 40,
    max_bars: int = 600,
    view: str = "auto",
) -> go.Figure:
    """Exchange-style candles: wick-side markers, volume pane, no spider lines.

    Zoom: scroll/box = time only · drag = pan · price framed to window.
    """
    empty = go.Figure()
    if ohlcv is None or ohlcv.empty:
        empty.update_layout(**plotly_layout(
            kind="time", height=480, title="No price data", uirevision="entries-empty",
        ))
        return empty

    df = _ohlcv_frame(ohlcv).tail(int(max_bars))
    bar = _bar_delta(df)
    t_all = trades.copy() if trades is not None else pd.DataFrame()
    if not t_all.empty:
        t_all = t_all.tail(int(max_trades))
    t = _align_trades_to_candles(t_all, df, bar)
    x0, x1 = _resolve_chart_window(df, t, bar, view, min_bars=48)

    has_vol = "volume" in df.columns and float(df["volume"].fillna(0).sum()) > 0
    fig = make_subplots(
        rows=2 if has_vol else 1,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.78, 0.22] if has_vol else [1.0],
    )

    # Slightly translucent bodies read more like TV/MEXC than solid blocks
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="Price",
        increasing_line_color=GREEN, decreasing_line_color=RED,
        increasing_fillcolor="rgba(14,203,129,0.85)",
        decreasing_fillcolor="rgba(246,70,93,0.85)",
        whiskerwidth=0.35,
        line=dict(width=1),
        showlegend=False,
    ), row=1, col=1)

    view_df = df.loc[(df.index >= x0 - bar) & (df.index <= x1)]
    if view_df.empty:
        view_df = df
    span = float(view_df["high"].max() - view_df["low"].min())
    pad = max(span * 0.035, float(view_df["close"].iloc[-1]) * 1e-4)

    entries_m, exits_m = _marker_stacks(t, df, bar, pad) if not t.empty else (pd.DataFrame(), pd.DataFrame())

    # Legend: one trace per kind (no per-trade lines)
    if not entries_m.empty:
        for name, color, sym in (
            ("Long", GREEN, "triangle-up"),
            ("Short", RED, "triangle-down"),
        ):
            sub = entries_m[entries_m["name"] == name]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["x"], y=sub["y"],
                mode="markers",
                name=f"{name} entry",
                marker=dict(
                    symbol=sym, size=11, color=color,
                    line=dict(width=1, color="#EAECEF"),
                ),
                customdata=sub[["px", "bars"]].to_numpy(),
                hovertemplate=(
                    f"<b>{name} entry</b><br>"
                    "Fill %{customdata[0]:.6g}<br>"
                    "Bars held %{customdata[1]}"
                    "<extra></extra>"
                ),
            ), row=1, col=1)

    if not exits_m.empty:
        for win, name, color in (
            (True, "Take profit / win", GREEN),
            (False, "Stop / loss", RED),
        ):
            part = exits_m[exits_m["win"] == win]
            if part.empty:
                continue
            fig.add_trace(go.Scatter(
                x=part["x"], y=part["y"],
                mode="markers",
                name=name,
                marker=dict(
                    symbol="circle", size=8, color=color,
                    line=dict(width=1.5, color="#EAECEF"),
                ),
                customdata=part[["px", "pnl", "reason", "bars"]].to_numpy(),
                hovertemplate=(
                    "<b>Exit</b> %{customdata[0]:.6g}<br>"
                    "P&L %{customdata[1]:+.2f}<br>"
                    "%{customdata[2]}<br>"
                    "Bars %{customdata[3]}"
                    "<extra></extra>"
                ),
            ), row=1, col=1)

    if has_vol:
        colors = [
            GREEN if float(c) >= float(o) else RED
            for o, c in zip(df["open"], df["close"])
        ]
        fig.add_trace(go.Bar(
            x=df.index,
            y=df["volume"],
            name="Volume",
            marker=dict(color=colors, line=dict(width=0)),
            opacity=0.45,
            showlegend=False,
            hovertemplate="Vol %{y:,.0f}<extra></extra>",
        ), row=2, col=1)

    lows = list(view_df["low"].astype(float))
    highs = list(view_df["high"].astype(float))
    if not entries_m.empty:
        in_e = entries_m[(entries_m["x"] >= x0) & (entries_m["x"] <= x1)]
        if not in_e.empty:
            lows += list(in_e["y"].astype(float))
            highs += list(in_e["y"].astype(float))
    if not exits_m.empty:
        in_x = exits_m[(exits_m["x"] >= x0) & (exits_m["x"] <= x1)]
        if not in_x.empty:
            lows += list(in_x["y"].astype(float))
            highs += list(in_x["y"].astype(float))
    y_range = _y_range_for_window(lows, highs, pad_frac=0.12)

    n_tr = 0 if t.empty else len(t)
    head = title if not n_tr else f"{title} · {n_tr}"

    layout_kwargs = plotly_layout(
        kind="time",
        uirevision=f"entries:{title}:{view}",
        height=560 if has_vol else 500,
        title=dict(text=head, font=dict(size=13, color=MUTED)),
        margin=dict(l=16, r=64, t=48, b=24),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=MUTED),
        ),
        hovermode="x unified",
        barmode="overlay",
    )
    fig.update_layout(**layout_kwargs)

    # Shared time axis — rangeslider off on every x
    x_range = [pd.Timestamp(x0).isoformat(), pd.Timestamp(x1).isoformat()]
    fig.update_xaxes(
        range=x_range, type="date", fixedrange=False,
        rangeslider_visible=False, showgrid=True, gridcolor=GRID,
        showticklabels=not has_vol, row=1, col=1,
    )
    if has_vol:
        fig.update_xaxes(
            range=x_range, type="date", fixedrange=False,
            rangeslider_visible=False, showgrid=True, gridcolor=GRID,
            row=2, col=1,
        )
        fig.update_yaxes(
            fixedrange=True, side="right", showgrid=False,
            title=None, row=2, col=1,
        )

    fig.update_yaxes(
        range=y_range, fixedrange=True, side="right",
        showgrid=True, gridcolor=GRID, title=None,
        tickfont=dict(size=11, color=MUTED),
        row=1, col=1,
    )
    fig.update_traces(selector=dict(type="candlestick"), xaxis="x")
    return polish_fig(fig)


# 3m is resampled from 1m on MEXC (exchange has no native 3m contract kline).
CHART_TIMEFRAMES = ("1m", "3m", "5m", "15m", "1h", "4h")


def chart_bars_for_tf(timeframe: str, *, hours: float = 48.0) -> int:
    """Enough bars to cover ~`hours` of calendar time on the chosen TF."""
    tf = str(timeframe).strip().lower()
    minutes = {"1m": 1, "3m": 3, "5m": 5, "15m": 15, "1h": 60, "4h": 240}.get(tf, 15)
    n = int(max(120, (float(hours) * 60.0) / minutes))
    return min(n, 5000)


@st.cache_data(ttl=30, show_spinner="Loading price chart…")
def load_symbol_ohlcv(symbol: str, timeframe: str, bars: int = 500) -> pd.DataFrame:
    """Candles for entry/exit charts — exact TF requested (3m/5m/15m/…).

    Does NOT silently fall back to 4h — that hid 3m/5m charts from the UI.
    """
    tf = str(timeframe).strip().lower() or "15m"
    n = int(bars) if bars else chart_bars_for_tf(tf)

    def _load(refresh: bool) -> pd.DataFrame:
        try:
            df = ms.fetch_historical(symbol, tf, n, refresh=refresh)
            if df is not None and not df.empty:
                return df
        except Exception:
            pass
        return pd.DataFrame()

    df = _load(False)
    try:
        if df.empty or ms._ohlcv_is_stale(df, tf, max_age_bars=1.0):
            fresh = _load(True)
            if not fresh.empty:
                df = fresh
    except Exception:
        if df.empty:
            df = _load(True)
    return df if df is not None else pd.DataFrame()


def pnl_color(v: float) -> str:
    return GREEN if v >= 0 else RED


def fmt_money(v: float) -> str:
    return f"{v:+.2f}" if v != 0 else "0.00"


def fmt_pct(v: float) -> str:
    return f"{v:.1%}"


def config_key(r: ms.BacktestResult) -> str:
    return f"pyr={r.pyramiding} part={r.partial} lev={r.leverage}x"


def results_display(df: pd.DataFrame) -> pd.DataFrame:
    """Human-friendly frame from results_as_frame."""
    if df.empty:
        return df
    out = df.copy()
    out["lev"] = out["lev"].map(lambda x: f"{int(x)}x")
    out["ret"] = out["ret"].map(lambda x: f"{x:+.0%}")
    out["cagr"] = out["cagr"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.0%}")
    out["sharpe"] = out["sharpe"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
    out["exp_r"] = out["exp_r"].map(lambda x: f"{x:.3f}")
    out["max_dd"] = out["max_dd"].map(lambda x: f"{x:.0%}")
    out["win"] = out["win"].map(lambda x: f"{x:.0%}")
    out["pf"] = out["pf"].map(lambda x: "n/a" if pd.isna(x) else f"{x:.2f}")
    return out.rename(columns={
        "exp_r": "exp(R)", "max_dd": "maxDD", "cagr": "CAGR", "sharpe": "Sharpe",
    })[
        ["pyr", "part", "lev", "final", "ret", "CAGR", "Sharpe", "exp(R)",
         "pf", "maxDD", "liq", "n", "win", "halt"]
    ]


def flag_overfit(ins: ms.BacktestResult, oos: ms.BacktestResult) -> str:
    """Honest train/test reading (delegates to moonshot.promotion_reading)."""
    return ms.promotion_reading(ins, oos)


# Plain-English labels for non-traders
_READING_PLAIN = {
    "thin sample": "Not enough trades yet to trust the result",
    "train failed · test lucky?": "Older period lost money — the recent win is probably luck",
    "train failed": "Lost money on the older period — needs work",
    "overfit risk": "Looks okay, but results may be fragile — keep practicing only",
    "both halves OK": "Passed the practice test on both time periods",
    "both positive · thin": "Slightly profitable but still not enough trades",
    "check manually": "Mixed signal — keep practicing, do not use real money",
}


def plain_reading(code: str) -> str:
    return _READING_PLAIN.get(code, code)


def what_to_do(ins: ms.BacktestResult, oos: ms.BacktestResult) -> dict:
    """
    One clear next step for a non-trader.

    Returns dict with keys: level, headline, detail, allow_live
    level ∈ success|info|warning|error
    """
    reading = ms.promotion_reading(ins, oos)
    plain = plain_reading(reading)
    train_ok = ins.expectancy_r > 0
    test_ok = oos.expectancy_r > 0
    enough = oos.trades >= 30

    # Average $ result per trade, relative to starting risk unit
    avg_train = f"{ins.expectancy_r:+.2f}"
    avg_test = f"{oos.expectancy_r:+.2f}"

    if not train_ok and test_ok:
        return {
            "level": "error",
            "headline": "Do not use real money",
            "detail": (
                f"{plain}. Older period lost on average ({avg_train} per trade); "
                f"newer period made money ({avg_test}) — that mismatch is a red flag. "
                "Keep fake-money practice only."
            ),
            "allow_live": False,
        }
    if not train_ok:
        return {
            "level": "error",
            "headline": "Bot needs more work",
            "detail": (
                f"{plain}. Average result per trade was negative on the older period "
                f"({avg_train}). Leave real money off."
            ),
            "allow_live": False,
        }
    if train_ok and test_ok and enough:
        return {
            "level": "success",
            "headline": "Keep practicing with fake money",
            "detail": (
                f"{plain}. Made money on average in both periods "
                f"(older {avg_train} / newer {avg_test} per trade, {oos.trades} newer trades). "
                "Still not ready for real money — watch paper trades for a while."
            ),
            "allow_live": False,
        }
    if train_ok and test_ok:
        return {
            "level": "info",
            "headline": "Promising — keep practicing",
            "detail": (
                f"{plain}. Both periods looked profitable, but only {oos.trades} newer trades "
                "(want 30+). Keep fake money running; do not go live yet."
            ),
            "allow_live": False,
        }
    return {
        "level": "warning",
        "headline": "Keep practicing — do not go live",
        "detail": f"{plain}. Leave real-money trading off.",
        "allow_live": False,
    }


def render_action(action: dict) -> None:
    """Show a big plain-English next step."""
    level = action.get("level", "info")
    headline = action.get("headline", "")
    detail = action.get("detail", "")
    fn = {
        "success": st.success,
        "info": st.info,
        "warning": st.warning,
        "error": st.error,
    }.get(level, st.info)
    fn(f"**{headline}**\n\n{detail}")


def paper_status_banner(
    mode: str,
    age: Optional[float],
    kill: bool,
    daily: bool,
    halt: str,
    last_eq: Optional[float],
    n_open: int,
    n_closed: int,
) -> None:
    """Overview: what is happening right now, in plain English."""
    if mode == "live":
        st.error(
            "**Real-money mode is ON.** Every open trade uses real funds. "
            "Only keep this if you fully understand the risk."
        )
        return

    # Paper can take a few minutes per cycle when watching many coins
    fresh = age is not None and age <= 180
    quiet = age is not None and age <= 900
    if kill:
        st.error(
            f"**Bot stopped itself for safety** ({halt or 'kill switch'}). "
            "It will not open new trades until you reset / restart after reviewing."
        )
    elif daily:
        st.warning(
            "**Paused for today** after hitting the daily loss limit. "
            "It resumes automatically on the next UTC day."
        )
    elif fresh:
        st.success(
            "**Practice mode is running** (fake money). Real money stays off. "
            "You do not need to change any settings."
        )
    elif quiet:
        st.info(
            "**Practice bot is alive but quiet** (waiting on the next market update). "
            "Real money stays off — no action needed."
        )
    else:
        st.warning(
            "**Practice bot does not look like it is running.** "
            "In a terminal run: `python -m moonshot paper`"
        )

    bits = []
    if last_eq is not None:
        bits.append(f"Practice balance: **${last_eq:,.2f}**")
    bits.append(f"Open trades: **{n_open}**")
    bits.append(f"Closed trades logged: **{n_closed}**")
    st.markdown(" · ".join(bits))
    st.caption("Real money stays off. Watch Live for win rate and profit goals.")


def load_autopilot_state() -> dict:
    if not AUTOPILOT_STATE_PATH.exists():
        return {}
    try:
        return json.loads(AUTOPILOT_STATE_PATH.read_text())
    except Exception:
        return {}


def load_live_feed(limit: int = 40) -> list:
    if not LIVE_FEED_PATH.exists():
        return []
    try:
        lines = LIVE_FEED_PATH.read_text().splitlines()
    except Exception:
        return []
    out = []
    for line in lines[-limit:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    out.reverse()  # newest first
    return out


def load_practice_activity(limit: int = 80, scan_lines: int = 4000) -> list:
    """Opens/closes/setups for the activity strip — skips hunt spam.

    Scans farther back than load_live_feed because improve_start/hold/paper_tick
    flood the tail and would make the chart look empty after hundreds of hunts.
    """
    want = {"setup", "reject", "open", "close"}
    out: list = []
    if LIVE_FEED_PATH.exists():
        try:
            lines = LIVE_FEED_PATH.read_text().splitlines()
        except Exception:
            lines = []
        for line in reversed(lines[-int(scan_lines):]):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            if str(row.get("kind") or "") not in want:
                continue
            out.append(row)
            if len(out) >= int(limit):
                break
    have = {
        (str(r.get("kind")), str(r.get("ts")), str(r.get("message") or "")[:40])
        for r in out
    }
    # Backfill opens/closes from DB — feed historically omitted them
    if len(out) < int(limit):
        try:
            trades = load_trades("paper")
        except Exception:
            trades = pd.DataFrame()
        if trades is not None and not trades.empty:
            tail = trades.tail(max(30, int(limit)))
            for _, tr in tail.iloc[::-1].iterrows():
                sym = str(tr.get("symbol") or "").split("/")[0]
                # close
                ts = tr.get("ts")
                pnl = tr.get("pnl")
                try:
                    pnl_s = f"{float(pnl):+.2f}"
                except (TypeError, ValueError):
                    pnl_s = "?"
                msg = f"CLOSE {sym} {tr.get('exit_reason') or 'exit'} pnl={pnl_s}"
                key = ("close", str(ts), msg[:40])
                if key not in have:
                    out.append({"ts": ts, "kind": "close", "message": msg})
                    have.add(key)
                # open (entry_time)
                ets = tr.get("entry_time")
                if pd.notna(ets):
                    side = str(tr.get("direction") or "")
                    omsg = f"OPEN {sym} {side}"
                    okey = ("open", str(ets), omsg[:40])
                    if okey not in have:
                        out.append({"ts": ets, "kind": "open", "message": omsg})
                        have.add(okey)
                if len(out) >= int(limit):
                    break
    # Newest first
    def _ts_key(row: dict):
        try:
            return pd.Timestamp(row.get("ts"))
        except Exception:
            return pd.Timestamp.min.tz_localize("UTC")
    out.sort(key=_ts_key, reverse=True)
    return out[: int(limit)]


@st.cache_data(show_spinner=False)
def load_ohlcv(symbols: tuple, bars: int, timeframe: str, synthetic: bool, refresh: bool) -> dict:
    raw = {}
    for s in symbols:
        if synthetic:
            raw[s] = ms.synthetic_ohlcv(s, bars, timeframe)
        else:
            raw[s] = ms.fetch_historical(s, timeframe, bars, refresh=refresh)
    return raw


@st.cache_data(show_spinner=False)
def load_ohlcv_range(
    symbols: tuple,
    timeframe: str,
    start_iso: str,
    end_iso: str,
    warmup_bars: int,
    refresh: bool,
    source: str = "mexc",
) -> dict:
    """Calendar-range OHLCV (warmup included) for the Backtest tab.

    ``source``: mexc (default), binance_spot, or yahoo — Backtest-only.
    """
    start = pd.Timestamp(start_iso)
    end = pd.Timestamp(end_iso)
    raw = {}
    for s in symbols:
        raw[s] = ms.fetch_historical_range(
            s, timeframe, start, end,
            warmup_bars=int(warmup_bars),
            refresh=bool(refresh),
            source=str(source or "mexc"),
        )
    return raw
