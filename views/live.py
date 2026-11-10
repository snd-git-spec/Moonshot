"""Live — strategy quality monitor in plain English."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import moonshot as ms
from dash_lib import (
    ACCENT, AMBER, GREEN, GREEN_SOFT, MUTED, RED, inject_theme, empty_state, page_header,
    section_header, chip_row, hero_panel, goal_rows, pct_of_risk, topbar,
    load_autopilot_state, load_live_feed, load_practice_activity,
    load_equity, load_trades, load_risk,
    detect_mode, heartbeat_age, format_age, plotly_layout, plotly_kwargs,
    polish_fig, fit_fig_for_viewport, equity_drawdown_fig, entries_exits_fig,
    load_symbol_ohlcv, CHART_TIMEFRAMES, chart_bars_for_tf, show_autofit_price_chart,
)

inject_theme()

try:
    from moonshot.hunt.scoring import WEEK_DAYS as _HOLD_DAYS
except Exception:
    _HOLD_DAYS = 7
_HOLDOUT = f"holdout {_HOLD_DAYS}d"

# Compact terminal header — balance filled after state loads inside the fragment
_boot_eq = load_equity(detect_mode(load_equity(None)))
_boot_bal = float(_boot_eq["equity"].iloc[-1]) if not _boot_eq.empty else None
_boot_score = int((load_autopilot_state().get("grade") or {}).get("score") or 0)
topbar(balance=_boot_bal, score=_boot_score, mode="PAPER", active="Live")
page_header(
    "",
    "Live",
    "Practice trading only · always searching for a better setup · real money stays off",
)


def _parse_ts(value):
    return pd.to_datetime(value, utc=True, format="ISO8601", errors="coerce")


def _history_week_ret(row: dict) -> float | None:
    if not isinstance(row, dict):
        return None
    wr = row.get("week_return")
    if wr is None:
        wr = (row.get("grade") or {}).get("week_return")
    if wr is None:
        week = row.get("week") or (row.get("grade") or {}).get("week") or {}
        if isinstance(week, dict) and week.get("ret") is not None:
            wr = week.get("ret")
    try:
        return float(wr) if wr is not None else None
    except (TypeError, ValueError):
        return None


def _peak_champion_knobs(state: dict) -> dict | None:
    """Knobs for the best holdout book — prefer champion_peak; recover if clobbered."""
    peak_blob = state.get("champion_peak") if isinstance(state.get("champion_peak"), dict) else None
    ch = peak_blob or (state.get("champion") if isinstance(state.get("champion"), dict) else {})
    if not ch:
        return None
    knobs = dict(ch.get("knobs") or {})
    if not knobs:
        return None
    peak = ch.get("week_ret_peak")
    try:
        peak_f = float(peak) if peak is not None else None
    except (TypeError, ValueError):
        peak_f = None
    cur = None
    try:
        cur = float(ch.get("week_ret")) if ch.get("week_ret") is not None else None
    except (TypeError, ValueError):
        cur = None
    need_recover = (not peak_blob) and (
        bool(ch.get("silence_rotate"))
        or (peak_f is not None and cur is not None and abs(cur - peak_f) > 0.005)
    )
    if need_recover and peak_f is not None:
        best_row = None
        best_gap = 1e9
        for h in state.get("history") or []:
            wr = _history_week_ret(h)
            if wr is None or not (h.get("knobs") or {}):
                continue
            gap = abs(float(wr) - peak_f)
            if gap < best_gap:
                best_gap = gap
                best_row = h
        if best_row is not None and best_gap < 0.02:
            knobs = dict(best_row.get("knobs") or knobs)
    return knobs or None


def _week_end_from_span(span: str | None) -> pd.Timestamp:
    """Parse 'YYYY-MM-DD → YYYY-MM-DD' end date; fall back to now UTC.

    Span end is the same ``end`` stamp ``_week_eval`` prints — do not +1 day.
    """
    try:
        if span and "→" in str(span):
            end_s = str(span).split("→")[-1].strip()
            return pd.Timestamp(end_s, tz="UTC")
    except Exception:
        pass
    return pd.Timestamp.now(tz="UTC")


@st.cache_data(ttl=600, show_spinner="Scoring champion on the same week…")
def _champion_same_week(
    knobs_json: str,
    end_iso: str,
    start_eq: float,
) -> dict:
    """Replay champion knobs on the same holdout window as the Live setup tile."""
    from moonshot.hunt.scoring import WEEK_DAYS, _week_eval, knobs_to_cfg
    from moonshot.features import default_btc_symbol, needs_btc_regime_context

    from dataclasses import replace as _dc_replace

    knobs = json.loads(knobs_json)
    cfg = _dc_replace(knobs_to_cfg(knobs), starting_equity=float(start_eq))
    end = pd.Timestamp(end_iso)
    if end.tzinfo is None:
        end = end.tz_localize("UTC")
    else:
        end = end.tz_convert("UTC")
    start = end - pd.Timedelta(days=int(WEEK_DAYS) + 2)
    tf = str(cfg.timeframe)
    warm = int(ms.range_warmup_bars(tf))
    symbols = list(ms.DEFAULT_SYMBOLS)
    # Hunt grades week on Binance spot — match that so Live vs champion is fair.
    src = str(getattr(ms, "RANGE_SOURCE_BINANCE_SPOT", "binance_spot"))
    raw = {}
    for s in symbols:
        raw[s] = ms.fetch_historical_range(
            s, tf, start, end, warmup_bars=warm, source=src,
        )
    context = None
    if needs_btc_regime_context(symbols, cfg):
        btc = default_btc_symbol()
        context = {
            btc: ms.fetch_historical_range(
                btc, tf, start, end, warmup_bars=warm, source=src,
            )
        }
    data = ms.prepare_data(raw, cfg, context=context)
    week = _week_eval(data, cfg, end=end)
    return {
        "week_ret": float(week.get("ret") or 0),
        "trades": int(week.get("trades") or 0),
        "span": str(week.get("span") or ""),
        "knobs": knobs,
    }


def _verdict(
    score: int, met: bool, test_exp: float, win: float, win_goal: float,
    test_return: float = 0.0, ret_goal: float = 0.25, start_eq: float = 50.0,
) -> tuple[str, str, str]:
    end_eq = start_eq * (1.0 + test_return)
    dollar = f"${start_eq:.0f}→${end_eq:.0f} ({test_return:+.0%})"
    avg = pct_of_risk(test_exp)
    if met:
        return (
            "good",
            "Targets met on paper history",
            f"Score {score}/100. Recent: win {win:.0%}, {dollar}. Past results ≠ future.",
        )
    if test_return < ret_goal and score >= 70:
        return (
            "warn",
            f"Close, but not there yet ({score}/100)",
            f"Score {score}/100. Goal ≥ {ret_goal:.0%} on the recent half "
            f"(≥ ${start_eq * (1 + ret_goal):.0f}). Autopilot is sizing up for real $.",
        )
    if win < win_goal and score >= 70:
        return (
            "warn",
            f"Close, but not there yet ({score}/100)",
            f"Score {score}/100. {dollar}. Average profit {avg}. "
            f"Win {win:.0%} vs ≥ {win_goal:.0%}.",
        )
    if score >= 80:
        return (
            "warn",
            f"Close, but not there yet ({score}/100)",
            f"{dollar}. Win {win:.0%} vs ≥ {win_goal:.0%}.",
        )
    if score >= 50:
        return (
            "warn",
            f"Only okay so far ({score}/100)",
            f"{dollar}. Win {win:.0%}. Still hunting.",
        )
    return (
        "bad",
        f"Too weak right now ({score}/100)",
        f"{dollar}. Win {win:.0%}. Keep hunting.",
    )


def _history_frame(history: list) -> pd.DataFrame:
    """One row per attempt #, time order — avoids Plotly scribbling on unsorted/dup gens."""
    df = pd.DataFrame(history)
    if df.empty or "generation" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["generation"]).copy()
    df["generation"] = pd.to_numeric(df["generation"], errors="coerce")
    df = df.dropna(subset=["generation"])
    if df.empty:
        return df
    df["generation"] = df["generation"].astype(int)
    if "ts" in df.columns:
        df["_ord"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601", errors="coerce")
    else:
        df["_ord"] = pd.NaT
    df = df.sort_values(["generation", "_ord"], kind="mergesort")
    df = df.drop_duplicates(subset=["generation"], keep="last")
    return df.drop(columns=["_ord"], errors="ignore").reset_index(drop=True)


def _week_return_fig(history: list) -> go.Figure:
    """Holdout-week hard return for the current hunt era only (same reseed cut as score)."""
    fig = go.Figure()
    df = _history_frame(history)
    if df.empty:
        fig.update_layout(**plotly_layout(
            height=260, title=f"{_HOLDOUT} path — waiting", uirevision="week-path",
        ))
        return polish_fig(fig)

    # Align era cut with score chart when scores exist
    era_start = 0
    if "score" in df.columns:
        scs = pd.to_numeric(df["score"], errors="coerce")
        imp = df["improved"].fillna(False).astype(bool) if "improved" in df.columns else pd.Series(False, index=df.index)
        for i in range(1, len(df)):
            if pd.isna(scs.iloc[i]) or pd.isna(scs.iloc[i - 1]):
                continue
            if (not bool(imp.iloc[i])) and (float(scs.iloc[i - 1]) - float(scs.iloc[i])) >= 15.0:
                era_start = i
        df = df.iloc[era_start:].reset_index(drop=True)

    weeks = []
    gens = []
    marker_colors = []
    hover_lev = []
    for _, row in df.iterrows():
        wr = row.get("week_return")
        if wr is None or (isinstance(wr, float) and wr != wr):
            g = row.get("grade") if isinstance(row.get("grade"), dict) else {}
            b = row.get("best") if isinstance(row.get("best"), dict) else {}
            wr = (g or {}).get("week_return")
            if wr is None:
                wr = ((g or {}).get("week") or {}).get("ret")
            if wr is None:
                wr = (b or {}).get("week_return")
        try:
            wr_f = float(wr)
        except (TypeError, ValueError):
            continue
        gens.append(int(row.get("generation") or 0))
        weeks.append(wr_f * 100.0)
        try:
            li = int(
                row.get("deployed_leverage")
                if row.get("deployed_leverage") is not None
                else (row.get("leverage") or 0)
            )
        except (TypeError, ValueError):
            li = 0
        marker_colors.append(_LEV_COLORS.get(li, ACCENT))
        hover_lev.append(f"{li}×" if li else "—")
    if not weeks:
        fig.update_layout(**plotly_layout(
            kind="time", height=260, title="Holdout path — no week stats yet",
            uirevision="week-path",
        ))
        return polish_fig(fig)
    fig.add_trace(go.Scatter(
        x=gens, y=weeks, name="holdout %",
        mode="lines+markers",
        line=dict(color=ACCENT, width=2),
        marker=dict(size=8, color=marker_colors, line=dict(width=1, color="#EAECEF")),
        customdata=hover_lev,
        hovertemplate="Attempt %{x}<br>Holdout %{y:+.0f}%<br>Deployed lev %{customdata}<extra></extra>",
    ))
    fig.add_hline(y=100, line_dash="dot", line_color=GREEN, annotation_text="+100% goal")
    fig.add_hline(y=0, line_width=1, line_color=MUTED)
    y0, y1 = min(weeks + [0.0, 100.0]), max(weeks + [0.0, 100.0])
    pad = max(5.0, (y1 - y0) * 0.12)
    fig.update_layout(**plotly_layout(
        kind="time",
        uirevision="week-path",
        height=280, title=None, margin=dict(l=48, r=24, t=12, b=40),
        xaxis=dict(title="Attempt #"),
        yaxis=dict(title="Holdout return %", range=[y0 - pad, y1 + pad], fixedrange=True),
        showlegend=False,
    ))
    return polish_fig(fig)


def _activity_fig(feed: list) -> go.Figure:
    """Clean trade-activity strip: opens / closes / setups only — no hunt upgrade spam."""
    fig = go.Figure()
    rows = []
    for ev in feed or []:
        kind = str(ev.get("kind") or "")
        msg = str(ev.get("message") or "")
        # Hunt upgrades/holds are NOT trading activity — they made this chart unreadable
        if kind in ("upgrade", "hold", "improve_start", "paper_restart", "autopilot_on",
                    "paper_tick", "error"):
            continue
        if kind == "setup":
            pass
        elif kind == "reject":
            pass
        elif kind in ("open", "close") or "OPEN " in msg:
            kind = "open" if (kind == "open" or "OPEN " in msg) else kind
            if "CLOSE " in msg:
                kind = "close"
        elif "CLOSE " in msg:
            kind = "close"
        else:
            continue
        rows.append({"ts": ev.get("ts"), "kind": kind, "msg": msg[:80]})
    if not rows:
        fig.update_layout(**plotly_layout(
            kind="time", height=160,
            title="No opens/closes yet", uirevision="activity",
        ))
        return polish_fig(fig)

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    df = df.dropna(subset=["ts"]).sort_values("ts")
    df = df.drop_duplicates(subset=["ts", "kind"], keep="last")

    # Single baseline lane — exchange tape style, not categorical spaghetti
    y_map = {"setup": 0.35, "reject": 0.15, "open": 0.7, "close": 0.7}
    style = {
        "setup": (ACCENT, "circle", 7),
        "reject": (MUTED, "x", 7),
        "open": (GREEN, "triangle-up", 12),
        "close": (RED, "triangle-down", 12),
    }
    for kind in ("setup", "reject", "open", "close"):
        g = df[df["kind"] == kind]
        if g.empty:
            continue
        color, symbol, size = style[kind]
        fig.add_trace(go.Scatter(
            x=g["ts"], y=[y_map[kind]] * len(g),
            mode="markers", name=kind,
            marker=dict(size=size, color=color, symbol=symbol,
                        line=dict(width=1, color="#E6E8EC")),
            customdata=g["msg"],
            hovertemplate="%{x}<br>%{customdata}<extra>" + kind + "</extra>",
        ))
    fig.update_layout(**plotly_layout(
        kind="time",
        uirevision="activity",
        height=180, title=None,
        margin=dict(l=48, r=24, t=28, b=36),
        xaxis=dict(title=None),
        yaxis=dict(
            title=None, range=[0, 1], fixedrange=True,
            showticklabels=False, showgrid=False, zeroline=False,
        ),
        showlegend=True,
        legend=dict(orientation="h", y=1.15, x=0),
    ))
    return polish_fig(fig)


_LEV_COLORS = {
    5: "#0ECB81",
    7: "#F0B90B",
    10: "#3B82F6",
    15: "#A855F7",
    25: "#F6465D",
}


def _leverage_ladder_fig(ladder: list, current_lev: int | None = None) -> go.Figure:
    """Bar chart: this pass’s forced 5/10/15/25× bake-off on last-7d return."""
    fig = go.Figure()
    if not ladder:
        fig.update_layout(**plotly_layout(
            kind="free", height=280,
            title="Leverage bake-off — waiting for next hunt pass",
            uirevision="lev-ladder",
        ))
        return polish_fig(fig)

    rows = sorted(ladder, key=lambda r: int(r.get("leverage") or 0))
    xs = [f"{int(r.get('leverage') or 0)}×" for r in rows]
    weeks = [float(r.get("week_ret") or 0) * 100.0 for r in rows]
    colors = []
    for r in rows:
        lev = int(r.get("leverage") or 0)
        colors.append(_LEV_COLORS.get(lev, ACCENT))
    custom = [
        [
            str(r.get("timeframe") or ""),
            str(r.get("strategy") or ""),
            int(r.get("week_trades") or 0),
            int(r.get("score") or 0),
            float(r.get("win") or 0) * 100.0,
        ]
        for r in rows
    ]
    fig.add_trace(go.Bar(
        x=xs, y=weeks,
        marker=dict(color=colors, line=dict(width=0)),
        customdata=custom,
        hovertemplate=(
            "<b>%{x}</b> · %{customdata[0]} · %{customdata[1]}<br>"
            "Holdout history %{y:+.1f}%<br>"
            "%{customdata[2]} week trades · score %{customdata[3]}/100<br>"
            "win %{customdata[4]:.0f}%"
            "<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.add_hline(y=0, line_width=1, line_color=MUTED)
    fig.add_hline(
        y=100, line_width=1, line_dash="dot", line_color=GREEN,
        annotation_text="+100% goal", annotation_font_color=GREEN,
        annotation_position="top left",
    )
    annotations = []
    if current_lev:
        annotations.append(dict(
            text=f"Live bot is on {int(current_lev)}× right now",
            xref="paper", yref="paper", x=0, y=1.12,
            showarrow=False, font=dict(size=12, color=MUTED),
        ))
    y0 = min(weeks + [0.0, -10.0])
    y1 = max(weeks + [0.0, 100.0])
    pad = max(8.0, (y1 - y0) * 0.12)
    fig.update_layout(**plotly_layout(
        kind="free",
        uirevision="lev-ladder",
        height=300,
        title=None,
        margin=dict(l=48, r=24, t=36, b=40),
        xaxis=dict(title="Leverage (forced every hunt pass)", fixedrange=True),
        yaxis=dict(
            title=f"{_HOLDOUT} %",
            range=[y0 - pad, y1 + pad],
            fixedrange=True,
            ticksuffix="%",
        ),
        annotations=annotations,
    ))
    return polish_fig(fig)


def _hunt_progress_fig(history: list) -> go.Figure:
    """
    Step chart of research quality for the *current* hunt era only.

    Reseeds start a new baseline — prior eras are omitted so a reseed never
    draws as a score crash on the same axis.
    """
    fig = go.Figure()
    df = _history_frame(history)
    if df.empty or "score" not in df.columns:
        fig.update_layout(**plotly_layout(
            kind="time", height=300, title="Hunt progress — waiting for results",
            uirevision="hunt-progress",
        ))
        return polish_fig(fig)

    plot_score = pd.to_numeric(df.get("score"), errors="coerce")
    if "score_raw" in df.columns:
        raw = pd.to_numeric(df["score_raw"], errors="coerce")
        plot_score = raw.fillna(plot_score)
    df = df.assign(_plot_score=plot_score).dropna(subset=["_plot_score"])
    if df.empty:
        fig.update_layout(**plotly_layout(
            kind="time", height=300, title="Hunt progress — no scores yet",
            uirevision="hunt-progress",
        ))
        return polish_fig(fig)
    df = df.reset_index(drop=True)

    improved = (
        df["improved"].fillna(False).astype(bool)
        if "improved" in df.columns
        else pd.Series(False, index=df.index)
    )

    # Last intentional reseed = last big drop without upgrade
    era_start = 0
    for i in range(1, len(df)):
        prev = float(df["_plot_score"].iloc[i - 1])
        sc = float(df["_plot_score"].iloc[i])
        if (not bool(improved.iloc[i])) and (prev - sc) >= 15.0:
            era_start = i

    chunk = df.iloc[era_start:].reset_index(drop=True)
    prior_eras = era_start > 0

    keep = [0]
    best = float(chunk["_plot_score"].iloc[0])
    for i in range(1, len(chunk)):
        sc = float(chunk["_plot_score"].iloc[i])
        if sc > best + 1e-9:
            keep.append(i)
            best = sc
    if keep[-1] != len(chunk) - 1:
        keep.append(len(chunk) - 1)
    step = chunk.iloc[keep]
    xs = [int(v) for v in step["generation"].tolist()]
    ys = [float(v) for v in step["_plot_score"].tolist()]
    n_up = sum(1 for i in range(1, len(ys)) if ys[i] > ys[i - 1] + 1e-9)

    # Color markers by *deployed* leverage (not rejected candidate lev on HOLD)
    marker_colors = []
    hover_lev = []
    for _, row in step.iterrows():
        lev = row.get("deployed_leverage")
        if lev is None:
            lev = row.get("leverage")
        try:
            li = int(lev) if lev is not None else 0
        except (TypeError, ValueError):
            li = 0
        marker_colors.append(_LEV_COLORS.get(li, GREEN))
        hover_lev.append(f"{li}×" if li else "—")
    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        name="Score since reseed",
        mode="lines+markers",
        line=dict(color=GREEN, width=2.5, shape="hv"),
        marker=dict(size=9, color=marker_colors, line=dict(width=1, color="#EAECEF")),
        customdata=hover_lev,
        hovertemplate="Attempt %{x}<br>Score %{y:.0f}/100<br>Deployed lev %{customdata}<extra></extra>",
    ))
    fig.add_hline(
        y=85, line_width=1, line_dash="dot", line_color=AMBER,
        annotation_text="Forward cap 85", annotation_font_color=AMBER,
        annotation_position="top left",
    )
    caption = f"{n_up} upgrades since current baseline"
    if prior_eras:
        caption += " · earlier reseeds hidden (not a crash)"
    fig.update_layout(**plotly_layout(
        kind="time",
        uirevision="hunt-progress",
        height=300,
        title=None,
        margin=dict(l=52, r=20, t=20, b=48),
        xaxis=dict(title=dict(text="Attempt #", standoff=8)),
        yaxis=dict(
            title=dict(text="Score / 100", standoff=6),
            range=[0, 105],
            gridcolor="#2A2F3A", zeroline=False, fixedrange=True,
        ),
        showlegend=False,
        annotations=[
            dict(
                text=caption,
                xref="paper", yref="paper", x=0, y=1.12,
                showarrow=False, font=dict(size=12, color=MUTED),
            )
        ],
    ))
    return polish_fig(fig)


def _trades_pnl_fig(trades: pd.DataFrame, focus: list | None = None,
                    start_eq: float = 50.0) -> go.Figure:
    fig = go.Figure()
    if focus and not trades.empty:
        trades = trades[trades["symbol"].isin(list(focus))]
    if trades.empty:
        fig.update_layout(**plotly_layout(
            kind="time", height=240,
            title="Realized equity — no closed fills yet",
            uirevision="closed-equity",
        ))
        return polish_fig(fig)
    t = trades.sort_values("ts").copy()
    t["equity"] = float(start_eq) + t["pnl"].cumsum()
    first_ts = pd.to_datetime(t["ts"].iloc[0], utc=True, format="ISO8601", errors="coerce")
    if pd.notna(first_ts):
        t = pd.concat(
            [pd.DataFrame({"ts": [first_ts - pd.Timedelta(minutes=1)], "equity": [float(start_eq)]}), t],
            ignore_index=True,
        )
    now_ts = pd.Timestamp.now(tz="UTC")
    last_ts = pd.to_datetime(t["ts"].iloc[-1], utc=True, format="ISO8601", errors="coerce")
    if pd.notna(last_ts) and now_ts > last_ts:
        t = pd.concat(
            [t, pd.DataFrame({"ts": [now_ts], "equity": [float(t["equity"].iloc[-1])]})],
            ignore_index=True,
        )
    up = float(t["equity"].iloc[-1]) >= float(start_eq)
    fig.add_trace(go.Scatter(
        x=t["ts"], y=t["equity"], name="Realized (closes only)",
        line=dict(color=GREEN if up else RED, width=2.2),
        fill="tozeroy",
        fillcolor="rgba(61,139,110,0.12)" if up else "rgba(184,92,92,0.12)",
        hovertemplate="Realized $%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=float(start_eq), line_width=1, line_dash="dot", line_color=MUTED)
    ys = [float(v) for v in t["equity"].tolist()] + [float(start_eq)]
    y0, y1 = min(ys), max(ys)
    pad = max(0.5, (y1 - y0) * 0.12)
    # No vertical y-axis title — section header names the chart; a rotated
    # "Equity $" at l≈48 overlaps 4-digit tick labels (1100/1050/…).
    fig.update_layout(**plotly_layout(
        kind="time",
        uirevision="closed-equity",
        height=280,
        title=None,
        margin=dict(l=64, r=16, t=12, b=40),
        yaxis=dict(
            title=None,
            range=[y0 - pad, y1 + pad],
            fixedrange=True,
            tickfont=dict(size=11),
            automargin=False,
        ),
        showlegend=False,
    ))
    return polish_fig(fig)


def _goals(grade: dict, targets: dict) -> list[dict]:
    te = float(grade.get("test_exp") or 0)
    tr = float(grade.get("train_exp") or 0)
    tn = int(grade.get("test_n") or 0)
    trn = int(grade.get("train_n") or 0)
    dd = float(grade.get("max_dd") or 0)
    win = float(grade.get("win") or 0)
    win_goal = float(targets.get("win_rate", 0.50))
    te_goal = float(targets.get("test_exp", 0.3))
    tr_goal = float(targets.get("train_exp", 0.4))
    tn_goal = int(targets.get("test_n", 40))
    trn_goal = int(targets.get("train_n", 40))
    dd_goal = float(targets.get("max_dd", 0.3))
    checks = grade.get("checks") or {}

    tret = float(grade.get("test_return") or 0)
    tret_goal = float(targets.get("test_return", 0.25))
    soft_ret = float(grade.get("soft_return") if grade.get("soft_return") is not None else tret)
    soft_floor = float(targets.get("soft_return_floor", -0.10))
    start_eq = 50.0
    try:
        start_eq = float(ms.Config.from_env().starting_equity)
    except Exception:
        pass
    # Always scale $ off current STARTING_EQUITY — stored *_final values were
    # computed at whatever start the hunt used (often $50) and go stale on reset.
    end_eq = start_eq * (1.0 + tret)
    soft_eq = start_eq * (1.0 + soft_ret)

    week = grade.get("week") or {}
    week_ret = float(week.get("ret") if week.get("ret") is not None else (grade.get("week_return") or 0))
    week_goal = float(targets.get("week_return", targets.get("weekly_return_goal", 1.0)))
    week_final = start_eq * (1.0 + week_ret)
    week_span = str(week.get("span") or grade.get("week_span") or _HOLDOUT)
    week_n = int(week.get("trades") or 0)

    return [
        {
            "title": "Last 7 days (practice history)",
            "detail": (
                f"${start_eq:.0f} → ${week_final:.0f} ({week_ret:+.0%}) · "
                f"goal ≥ {week_goal:.0%} · {week_n} trades"
            ),
            "value": week_ret, "goal": week_goal,
            "passed": checks.get("week_return", week_ret >= week_goal and week_n > 0),
        },
        {
            "title": "Longer backtest return",
            "detail": (
                f"${start_eq:.0f} → ${end_eq:.0f} ({tret:+.0%}) · "
                f"aim ≥ {tret_goal:.0%}"
            ),
            "value": tret, "goal": tret_goal,
            "passed": checks.get("test_return", tret >= tret_goal),
        },
        {
            "title": "Backtest without early quit",
            "detail": (
                f"${start_eq:.0f} → ${soft_eq:.0f} ({soft_ret:+.0%}) · "
                f"don’t fall below {soft_floor:.0%}"
            ),
            "value": soft_ret, "goal": soft_floor,
            "passed": checks.get("soft_ok", soft_ret >= soft_floor),
        },
        {
            "title": "Honest results (not quit-while-ahead)",
            "detail": (
                "Looks like it only won by quitting early"
                if not checks.get("no_kill_illusion", True)
                else "Results look consistent either way"
            ),
            "value": 1.0 if checks.get("no_kill_illusion", not grade.get("kill_illusion")) else 0.0,
            "goal": 1.0,
            "passed": bool(checks.get("no_kill_illusion", not grade.get("kill_illusion"))),
        },
        {
            "title": "Solid week before celebrating",
            "detail": (
                f"Week {week_ret:+.0%} · want at least "
                f"{float(targets.get('week_hard_floor', 0.25)):.0%}"
            ),
            "value": week_ret,
            "goal": float(targets.get("week_hard_floor", 0.25)),
            "passed": bool(checks.get(
                "week_hard_ok",
                week_ret >= float(targets.get("week_hard_floor", 0.25)),
            )),
        },
        {
            "title": "Win rate",
            "detail": f"{win:.0%} · aim ≥ {win_goal:.0%}",
            "value": win, "goal": win_goal,
            "passed": checks.get("win_rate", win >= win_goal),
        },
        {
            "title": "Average profit per trade",
            "detail": f"{pct_of_risk(te)} · aim ≥ {pct_of_risk(te_goal)}",
            "value": te, "goal": te_goal,
            "passed": checks.get("test_exp", te >= te_goal),
        },
        {
            "title": "Average profit (older period)",
            "detail": f"{pct_of_risk(tr)} · aim ≥ {pct_of_risk(tr_goal)}",
            "value": tr, "goal": tr_goal,
            "passed": checks.get("train_exp", tr >= tr_goal),
        },
        {
            "title": "Enough recent trades",
            "detail": f"{tn} trades · need ≥ {tn_goal}",
            "value": tn, "goal": tn_goal,
            "passed": checks.get("test_n", tn >= tn_goal),
        },
        {
            "title": "Enough older trades",
            "detail": f"{trn} trades · need ≥ {trn_goal}",
            "value": trn, "goal": trn_goal,
            "passed": checks.get("train_n", trn >= trn_goal),
        },
        {
            "title": "Drawdown under control",
            "detail": f"Worst drop in the recent test period {dd:.0%} · keep ≤ {dd_goal:.0%}",
            "value": dd, "goal": dd_goal, "higher_better": False,
            "passed": checks.get("max_dd", dd <= dd_goal),
        },
    ]


def _validation_summary(grade: dict, elite: dict, start_eq: float) -> None:
    """Short plain-English health chips — no internal jargon."""
    lines = []
    tones = []

    soft_final = grade.get("soft_final") or elite.get("soft_final")
    soft_return = grade.get("soft_return")
    if soft_return is None:
        soft_return = elite.get("soft_return")
    oos_final = grade.get("oos_final") or elite.get("oos_final")
    # Only shout when soft path is meaningfully worse (≥15% of starting equity)
    if soft_final is not None and oos_final is not None:
        gap = float(oos_final) - float(soft_final)
        if gap > start_eq * 0.15:
            lines.append(
                f"Results look better only because of early quit "
                f"(${float(oos_final):.0f} with quit vs ${float(soft_final):.0f} without)"
            )
            tones.append("warn")

    wf = grade.get("walk_forward") or elite.get("walk_forward")
    if wf:
        if wf.get("ok"):
            lines.append(
                f"Held up on older unseen periods "
                f"({wf.get('green', '?')} green · {wf.get('trades', 0)} trades)"
            )
            tones.append("ok")
        else:
            why = str(wf.get("why") or "failed")
            lines.append(f"Failed older unseen-period check — {why}")
            tones.append("warn")

    fwd = grade.get("forward")
    if fwd is not None:
        need = int(fwd.get("need", 20))
        n_f = int(fwd.get("n", 0))
        pnl = float(fwd.get("pnl") or 0)
        if fwd.get("ok"):
            lines.append(f"Live paper proof {n_f}/{need} trades · ${pnl:+.2f}")
            tones.append("ok")
        else:
            lines.append(
                f"Still collecting live paper proof ({n_f}/{need} trades · ${pnl:+.2f})"
            )
            tones.append("muted")

    recent = grade.get("recent") or elite.get("recent")
    if recent and not str(recent.get("why", "")).startswith("recent eval failed"):
        n_r = int(recent.get("trades") or 0)
        if recent.get("ok"):
            lines.append(f"Last {ms.RECENT_DAYS} days look OK ({n_r} trades)")
            tones.append("ok")
        else:
            lines.append(
                f"Last {ms.RECENT_DAYS} days look weak — {recent.get('why', 'not enough edge')}"
            )
            tones.append("warn")

    if not lines:
        return
    chip_row([(txt, tone) for txt, tone in zip(lines, tones)])
    if wf and wf.get("folds"):
        with st.expander("Older-period detail", expanded=False):
            folds = wf.get("folds") or []
            st.caption(
                " · ".join(
                    f"{f['from'][:7]}→{f['to'][:7]}: {f['ret']:+.0%} ({f['trades']} trades)"
                    for f in folds
                )
            )


def _plot(fig: go.Figure) -> None:
    """Render every Live chart polished (no bottom scrubber)."""
    fig = fit_fig_for_viewport(polish_fig(fig))
    # Match Streamlit iframe height to the figure so axis ticks aren't clipped.
    try:
        h = int(fig.layout.height or 360)
    except Exception:
        h = 360
    st.plotly_chart(fig, **plotly_kwargs(height=h))


_PLOT = plotly_kwargs()  # kept for any leftover **_PLOT usage


@st.fragment(run_every=timedelta(seconds=12))
def live_panel() -> None:
    try:
        state = load_autopilot_state()
        feed = load_live_feed(60)
        equity_all = load_equity(None)
        mode = detect_mode(equity_all)
        eq = load_equity(mode)
        trades = load_trades(mode)
        age = heartbeat_age(eq)
        last_eq = float(eq["equity"].iloc[-1]) if not eq.empty else None
        now = datetime.now(timezone.utc)

        if not state and not feed:
            empty_state(
                "Nothing running yet",
                "Start with <code>./install_keepalive.sh</code> or "
                "<code>python -m moonshot autopilot</code>",
            )
            return

        last = state.get("last_improve") or {}
        live = state.get("live") or {}
        history = list(state.get("history") or [])
        grade = state.get("grade") or last.get("grade") or {}
        targets = state.get("targets") or grade.get("targets") or {}
        elite = state.get("elite") or {}
        score = int(grade.get("score") or elite.get("score") or 0)
        met = bool(grade.get("targets_met") or elite.get("targets_met"))
        test_exp = float(grade.get("test_exp") or elite.get("test_exp") or 0)
        win = float(grade.get("win") if grade.get("win") is not None else (elite.get("win") or 0))
        win_goal = float(targets.get("win_rate", 0.50))
        test_return = float(
            grade.get("test_return")
            if grade.get("test_return") is not None
            else (elite.get("test_return") or 0)
        )
        ret_goal = float(targets.get("test_return", 0.25))
        start_eq = 50.0
        tf = "4h"
        try:
            cfg_env = ms.Config.from_env()
            start_eq = float(cfg_env.starting_equity)
            tf = str(cfg_env.timeframe or "4h")
        except Exception:
            cfg_env = None
        # Deployed book wins (.env / Config). Elite is often a stale forward-locked
        # snapshot and must not paint the wrong TF/leverage on Live.
        for knobs in (
            ({"timeframe": getattr(cfg_env, "timeframe", None)} if cfg_env else {}),
            (live.get("knobs") or {}),
            ((state.get("champion") or {}).get("knobs") or {}),
            (last.get("knobs") or {}),
            (elite.get("knobs") or {}),
        ):
            if knobs.get("timeframe"):
                tf = str(knobs["timeframe"])
                break
        # Scale dollars from current STARTING_EQUITY; ignore hunt-era absolute finals.
        oos_final = start_eq * (1.0 + test_return)
        week = grade.get("week") or elite.get("week") or {}
        week_ret = float(
            week.get("ret") if week.get("ret") is not None
            else (grade.get("week_return") or elite.get("week_return") or 0)
        )
        week_goal = float(targets.get("week_return", 1.0))
        week_final = start_eq * (1.0 + week_ret)
        week_span = str(week.get("span") or grade.get("week_span") or _HOLDOUT)
        week_n = int(week.get("trades") or 0)
        # Actual paper wallet change over holdout window (if equity history exists)
        paper_week_ret = None
        paper_week_pnl = None
        if not eq.empty and "ts" in eq.columns and "equity" in eq.columns:
            eq2 = eq.copy()
            eq2["_t"] = pd.to_datetime(eq2["ts"], utc=True, errors="coerce")
            eq2 = eq2.dropna(subset=["_t"]).sort_values("_t")
            if len(eq2) >= 2:
                cut = eq2["_t"].iloc[-1] - pd.Timedelta(days=int(_HOLD_DAYS))
                past = eq2[eq2["_t"] <= cut]
                if not past.empty:
                    e0 = float(past["equity"].iloc[-1])
                    e1 = float(eq2["equity"].iloc[-1])
                    if e0 > 0:
                        paper_week_pnl = e1 - e0
                        paper_week_ret = (e1 / e0) - 1.0
        _wd = state.get("scoreboard_watchdog") or {}
        if _wd.get("critical"):
            _alerts = _wd.get("alerts") or ["champ / .env mismatch"]
            st.error(
                "**SCOREBOARD CRITICAL** — Live may be stranded below champion.  \n"
                + "  \n".join(f"- {a}" for a in _alerts[:3])
            )

        busy = bool(state.get("busy"))
        passes = int(state.get("improve_passes", 0))
        promote_tier = str(
            last.get("promote_tier") or state.get("promote_tier") or "hold"
        )
        lev = int(
            (getattr(cfg_env, "leverage", 0) if cfg_env else 0)
            or (live.get("knobs") or {}).get("leverage")
            or ((state.get("champion") or {}).get("knobs") or {}).get("leverage")
            or (elite.get("knobs") or {}).get("leverage")
            or 0
        )

        # Plain-English status — never dump internal jargon on the user
        paper_ok = age is not None and age < 180
        just_upgraded = bool(last.get("improved")) and promote_tier in ("research", "confirmed")
        cool_left = None
        try:
            last_prom = state.get("research_promote_at")
            if last_prom:
                age_s = (now - pd.Timestamp(last_prom).to_pydatetime().replace(tzinfo=timezone.utc)).total_seconds()
                cool_s = float(getattr(ms, "RESEARCH_PROMOTE_MIN_SEC", 300))
                if age_s < cool_s:
                    cool_left = int(cool_s - age_s)
        except Exception:
            cool_left = None

        if busy:
            _gen = int(state.get("generation") or passes or 0)
            status_label = (
                f"Searching · pass #{_gen}" if _gen else "Searching for a better setup…"
            )
            status_tone = "busy"
        elif just_upgraded:
            status_label, status_tone = "Just switched to a better setup", "ok"
        elif met:
            status_label, status_tone = "Goals looking good — keep proving it", "ok"
        else:
            status_label, status_tone = "Running · always looking for better", "ok"

        _chips = [
            (status_label, status_tone),
            (
                (f"Practice wallet updated {format_age(age)}" if paper_ok
                 else "Practice wallet starting up"),
                "ok" if paper_ok else "warn",
            ),
            (
                f"Live book {lev}× · {tf}" if lev else f"Live book {tf}",
                "muted",
            ),
        ]
        try:
            from moonshot.hunt.constants import autopilot_frozen as _ap_frozen

            if _ap_frozen():
                _chips.insert(0, ("FROZEN — no promotions", "warn"))
        except Exception:
            pass
        if just_upgraded:
            _chips.append(("New setup just applied", "ok"))
        chip_row(_chips)

        # One human sentence — what it's doing next
        if cool_left and cool_left > 0:
            st.caption(
                f"Waiting ~{cool_left}s after the last upgrade so the practice wallet can settle."
            )
        elif just_upgraded:
            st.caption(
                "New settings are on the practice wallet. "
                "Watching real paper P&L before changing again."
            )
        elif week_ret >= 0:
            st.caption(
                f"On historical data, this setup’s {_HOLDOUT}: "
                f"${start_eq:.0f} → ${week_final:.0f} ({week_ret:+.0%}). "
                "Still searching for something that beats that."
            )
        else:
            st.caption(
                f"On historical data, this setup’s {_HOLDOUT} is down ({week_ret:+.0%}). "
                "Searching for a better setup — practice money only."
            )

        week_met = week_ret >= week_goal and week_n > 0
        if week_met:
            level, headline, detail = (
                "good",
                f"Strong holdout on paper history ({week_ret:+.0%})",
                f"${start_eq:.0f} → ${week_final:.0f} over the {_HOLDOUT} history window. "
                f"Practice wallet is separate.",
            )
        elif week_n == 0:
            level, headline, detail = (
                "warn",
                "Not enough recent trades to score the holdout yet",
                "Bot is still searching. Practice account will fill in as candles close.",
            )
        else:
            level, headline, detail = (
                "bad" if week_ret < 0 else "warn",
                f"Holdout history: {week_ret:+.0%}",
                f"${start_eq:.0f} → ${week_final:.0f} · {week_n} trades in {_HOLDOUT}. "
                f"Always hunting for a better setup.",
            )
        win_tone = "good" if win >= win_goal else "bad"
        exp_goal = float(targets.get("test_exp", 0.3))
        exp_tone = "good" if test_exp >= exp_goal else ("accent" if test_exp > 0 else "bad")
        week_tone = "good" if week_met else ("accent" if week_ret > 0 else "bad")
        soft_final = grade.get("soft_final") or elite.get("soft_final")
        bal_tone = (
            "good" if last_eq is not None and last_eq >= start_eq
            else ("bad" if last_eq is not None else "")
        )
        paper_tone = (
            "good" if paper_week_ret is not None and paper_week_ret >= 0
            else ("bad" if paper_week_ret is not None else "")
        )

        _validation_summary(grade, elite, start_eq)

        elite_score = int(elite.get("score") or 0)
        held_streak = 0
        try:
            hist_tmp = list(state.get("history") or [])
            for h in reversed(hist_tmp):
                if h.get("improved"):
                    break
                held_streak += 1
        except Exception:
            pass
        # Score on Live = held elite (never looks like a demotion on a hold pass)
        show_score = max(score, elite_score) if elite_score else score
        bars_seen = int(live.get("bars_seen") or 0)
        setups_seen = int(live.get("setups_seen") or 0)
        bal_pnl = (float(last_eq) - float(start_eq)) if last_eq is not None else None
        n_closed = len(trades)

        if n_closed == 0 and setups_seen == 0 and bars_seen > 20:
            st.warning(
                f"The practice bot has watched {bars_seen} new candles since it last restarted "
                "and still has not found a trade to take — filters are too tight right now."
            )
        elif n_closed == 0:
            st.info(
                f"Practice wallet starts at ${start_eq:.0f}. "
                "It will move once the bot closes a real practice trade."
            )

        # Four tiles — every number has a plain-English meaning
        if paper_week_ret is not None:
            paper_tile = {
                "label": f"Practice wallet · {_HOLDOUT}",
                "value": f"{paper_week_ret:+.0%}",
                "hint": f"${paper_week_pnl:+.2f} actual paper P&L over {_HOLDOUT}",
                "tone": paper_tone,
            }
        else:
            paper_tile = {
                "label": f"Practice wallet · {_HOLDOUT}",
                "value": "—",
                "hint": f"Needs a full {_HOLD_DAYS}d of wallet history before this % can show",
                "tone": "",
            }

        if held_streak >= 2:
            score_hint = (
                f"Setup quality grade · no better setup found for {held_streak} searches"
            )
        else:
            score_hint = "Setup quality grade (0–100) · higher = closer to your goals"

        if last_eq is None:
            bal_hint = "Waiting for the practice bot to report a balance"
        elif n_closed == 0:
            bal_hint = f"Started at ${start_eq:.0f} · no closed practice trades yet"
        elif bal_pnl is not None and abs(bal_pnl) < 0.005:
            bal_hint = f"{n_closed} closed trades · net about flat vs ${start_eq:.0f} start"
        else:
            bal_hint = (
                f"{n_closed} closed trades · "
                f"{bal_pnl:+.2f} vs ${start_eq:.0f} start"
            )

        # Prefer stored hunt metrics — live re-sim blocks the whole page on cloud.
        champ_tile = None
        champ_knobs = _peak_champion_knobs(state)
        if champ_knobs:
            peak_blob = (
                state.get("champion_peak")
                if isinstance(state.get("champion_peak"), dict)
                else None
            ) or (
                state.get("champion")
                if isinstance(state.get("champion"), dict)
                else {}
            )
            c_ret = None
            for key in ("week_ret_peak", "week_ret"):
                try:
                    if peak_blob.get(key) is not None:
                        c_ret = float(peak_blob[key])
                        break
                except (TypeError, ValueError):
                    pass
            if c_ret is not None:
                c_final = start_eq * (1.0 + c_ret)
                c_lev = int(champ_knobs.get("leverage") or 0)
                c_tf = str(champ_knobs.get("timeframe") or "")
                c_strat = str(champ_knobs.get("strategy") or "")
                c_span = str(peak_blob.get("week_span") or week_span or "")
                c_book = (
                    f"{c_strat} {c_lev}×@{c_tf}".strip()
                    if (c_strat or c_lev or c_tf)
                    else "champion book"
                )
                champ_tile = {
                    "label": f"Champion · {_HOLDOUT}",
                    "value": f"${c_final:,.0f}",
                    "hint": (
                        f"{c_book} · ${start_eq:,.0f} → ${c_final:,.0f} ({c_ret:+.0%})"
                        + (f" · {c_span}" if c_span else "")
                    ),
                    "tone": (
                        "good" if c_ret >= week_goal
                        else ("accent" if c_ret > 0 else "bad")
                    ),
                }

        hero_stats = [
            {
                "label": f"Live setup · {_HOLDOUT}",
                "value": f"${week_final:,.0f}",
                "hint": (
                    f"${start_eq:,.0f} → ${week_final:,.0f} ({week_ret:+.0%}) "
                    f"simulated from past candles · goal ≥ {week_goal:.0%} · "
                    f"{week_n} trades in that holdout"
                ),
                "tone": week_tone,
            },
        ]
        if champ_tile is not None:
            hero_stats.append(champ_tile)
        hero_stats.extend(
            [
                paper_tile,
                {
                    "label": "Quality score",
                    "value": f"{show_score}/100",
                    "hint": score_hint,
                    "tone": "good" if show_score >= 70 else "",
                },
                {
                    "label": "Live wallet now",
                    "value": f"${last_eq:,.2f}" if last_eq is not None else "—",
                    "hint": bal_hint,
                    "tone": bal_tone,
                },
            ]
        )
        hero_panel(level, headline, detail, hero_stats)

        # Only mention candle progress when the wallet is still quiet
        if n_closed == 0 and bars_seen > 0:
            st.caption(
                f"Since last restart: watched {bars_seen} new candle"
                f"{'s' if bars_seen != 1 else ''}"
                + (
                    f", found {setups_seen} trade signal"
                    f"{'s' if setups_seen != 1 else ''}."
                    if setups_seen
                    else ", no trade signals yet."
                )
            )

        goals = _goals(grade or elite, targets)
        n_pass = sum(1 for g in goals if g.get("passed"))
        with st.expander(f"Goals · {n_pass}/{len(goals)} passed", expanded=False):
            goal_rows(goals)
            st.caption(
                "These goals score the current setup on historical data. "
                "Live wallet / realized equity charts below are paper money — different numbers."
            )

        st.caption(
            f"Practice money only · {tf} candles · "
            f"{passes} search batches run · real exchange money stays off"
        )

        # ---- equity pair: wallet (MTM) vs realized (closes only) ----
        section_header(
            "1 · Live paper wallet",
            "Includes open positions marked to market · jumps/resets when the bot restarts after a setup change",
        )
        if eq.empty:
            st.caption("Wallet chart fills in as the practice bot runs.")
        else:
            _plot(equity_drawdown_fig(eq))
            st.caption(
                "This is the running paper account. It is not the same as closed-trade P&L below — "
                "open winners/losers move this line before a fill closes."
            )

        focus = list(state.get("paper_symbols") or state.get("hunt_symbols") or [])
        section_header(
            "2 · Realized equity (closed fills only)",
            f"Starting ${float(start_eq):.0f} + sum of closed trade P&L · ignores open marks · survives restarts",
        )
        if trades.empty:
            st.caption("No closed paper trades yet — this line appears after the first exit.")
        else:
            _plot(_trades_pnl_fig(trades, focus, start_eq=float(start_eq)))
            st.caption(
                "Only locked-in closes. If this disagrees with the wallet above, the gap is "
                "open-position mark-to-market and/or a wallet restart after promote."
            )

        hunt_fig = _hunt_progress_fig(history)
        n_up = 0
        if hunt_fig.data and len(hunt_fig.data[0].y or []) >= 2:
            ys = list(hunt_fig.data[0].y)
            n_up = sum(1 for i in range(1, len(ys)) if float(ys[i]) > float(ys[i - 1]) + 1e-9)
        section_header(
            "Search progress",
            (
                f"{n_up} upgrade{'s' if n_up != 1 else ''} found so far · "
                "score should climb when a better setup wins"
            ),
        )
        _plot(hunt_fig)

        section_header(
            f"{_HOLDOUT} history return by search",
            f"Each point = how the then-current setup did on the {_HOLDOUT} history window · "
            "marker color = deployed leverage · goal line at +100%",
        )
        _plot(_week_return_fig(history))

        ladder = list(state.get("leverage_ladder") or [])
        if not ladder and history:
            # Fall back to newest history row that carried a ladder snapshot
            for h in reversed(history):
                snap = h.get("leverage_ladder")
                if isinstance(snap, list) and snap:
                    ladder = snap
                    break
        cur_lev = None
        for src in (
            ({"leverage": getattr(cfg_env, "leverage", None)} if cfg_env else {}),
            (live.get("knobs") or {}),
            ((state.get("champion") or {}).get("knobs") or {}),
            (last.get("knobs") or {}),
            (elite.get("knobs") or {}),
        ):
            if src.get("leverage") is not None:
                try:
                    cur_lev = int(src["leverage"])
                    break
                except (TypeError, ValueError):
                    pass
        section_header(
            "Leverage bake-off (this hunt pass)",
            f"Fixed lev×TF matrix on the current book — bars = {_HOLDOUT} history return · "
            "green=5× · gold=7× · blue=10× · purple=15× · red=25× · sparse high-lev skipped for promote",
        )
        _plot(_leverage_ladder_fig(ladder, current_lev=cur_lev))
        if ladder:
            ladder_tbl = [{
                "Leverage": f"{int(r.get('leverage') or 0)}×",
                "Candle size": r.get("timeframe") or "—",
                "Style": r.get("strategy") or "—",
                f"{_HOLDOUT} history": f"{float(r.get('week_ret') or 0):+.0%}",
                "Holdout trades": int(r.get("week_trades") or 0),
                "Score": int(r.get("score") or 0),
                "Win rate": f"{float(r.get('win') or 0):.0%}",
                "On live bot": (
                    "yes" if cur_lev and int(r.get("leverage") or 0) == int(cur_lev) else ""
                ),
            } for r in sorted(ladder, key=lambda x: int(x.get("leverage") or 0))]
            st.dataframe(pd.DataFrame(ladder_tbl), width="stretch", hide_index=True, height=180)

        section_header(
            "Recent practice activity",
            "Paper opens/closes/setups only — hunt search attempts are a different counter and do not belong here",
        )
        activity = load_practice_activity(80)
        _plot(_activity_fig(activity))
        if not activity:
            st.caption(
                "No practice fills in the activity feed yet. Hunt can run hundreds of searches "
                "while paper waits for the next candle setup — check Realized equity above "
                "for older closes."
            )
        else:
            n_oc = sum(1 for e in activity if e.get("kind") in ("open", "close"))
            n_su = sum(1 for e in activity if e.get("kind") == "setup")
            st.caption(
                f"Showing {len(activity)} practice events ({n_oc} open/close · {n_su} setups). "
                f"Search attempts on Live are hunt passes, not trades."
            )

        section_header(
            "Recent searches",
            "Newest first — did this search find a better setup?",
        )
        if held_streak >= 3:
            st.caption(
                f"Last {held_streak} searches kept the same setup (score {score}/100). "
                "That means nothing better was found — not a frozen screen."
            )
        # Dedupe by generation (history can log explore+polish same gen)
        seen_gen: set[int] = set()
        rows = []
        # In-progress pass first when history has not logged it yet
        try:
            cur_gen = int(state.get("generation") or 0)
        except (TypeError, ValueError):
            cur_gen = 0
        he = state.get("hunt_event") or {}
        he_kind = str(he.get("kind") or "")
        try:
            he_gen = int(he.get("generation") or 0)
        except (TypeError, ValueError):
            he_gen = 0
        show_running = bool(busy) or he_kind == "improve_start"
        running_gen = he_gen or cur_gen
        if show_running and running_gen > 0:
            mode = str(he.get("search_mode") or "search")
            rows.append({
                "Search #": running_gen,
                "What happened": f"In progress · {mode}",
                "Leverage": "—",
                "Candle": "—",
                "Score": "—",
                "Win rate": "—",
                f"{_HOLDOUT} history": "—",
                "Trades in test": "—",
                "Avg profit": "—",
            })
            seen_gen.add(int(running_gen))
        for h in reversed(history):
            g = h.get("generation")
            try:
                gi = int(g) if g is not None else None
            except (TypeError, ValueError):
                gi = None
            if gi is not None:
                if gi in seen_gen:
                    continue
                seen_gen.add(gi)
            te = h.get("test_exp")
            w = h.get("win")
            wr = h.get("week_return")
            tested = h.get("tested")
            why = str(h.get("promote_reason") or "")
            if h.get("improved"):
                result = f"Applied to live bot ({why})" if why else "Applied to live bot"
                week_note = (
                    f"{float(wr):+.0%}" if wr is not None else "—"
                )
            elif tested is not None:
                # Money truth: not applied ⇒ current setup stayed. Do not
                # frame an internal path name (matrix_escape, etc.) as a
                # "rejected best" — that read as if something beat Live.
                gate = str(h.get("material_why") or "").strip()
                if gate and why not in ("", "baseline", "hold"):
                    result = (
                        f"Kept current setup · checked {tested} ideas · "
                        f"no promote ({gate})"
                    )
                else:
                    result = (
                        f"Kept current setup · checked {tested} ideas · "
                        "nothing better to deploy"
                    )
                # HOLD rows re-report the held book's week — not a new find.
                week_note = (
                    f"{float(wr):+.0%} (current setup)"
                    if wr is not None else "—"
                )
            else:
                result = "Kept current setup"
                week_note = (
                    f"{float(wr):+.0%} (current setup)"
                    if wr is not None else "—"
                )
            sc = h.get("score")
            lev = h.get("leverage")
            try:
                lev_s = f"{int(lev)}×" if lev is not None else "—"
            except (TypeError, ValueError):
                lev_s = "—"
            rows.append({
                "Search #": h.get("generation"),
                "What happened": result,
                "Leverage": lev_s,
                "Candle": h.get("timeframe") or "—",
                "Score": sc if sc is not None else "—",
                "Win rate": f"{float(w):.0%}" if w is not None else "—",
                f"{_HOLDOUT} history": week_note,
                "Trades in test": h.get("test_n"),
                "Avg profit": pct_of_risk(float(te)) if te is not None else "—",
            })
            if len(rows) >= 24:
                break
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=420)
        else:
            st.caption(
                "No completed hunt passes in the log yet — list fills as passes finish."
            )

        with st.expander("Activity log", expanded=False):
            interesting = {
                "upgrade", "error", "paper_restart", "autopilot_on", "targets_met",
                "setup", "reject", "open", "close",
            }
            log_src = list(activity) if activity else load_practice_activity(40)
            for row in feed[:80]:
                if row.get("kind") in ("upgrade", "error", "paper_restart", "targets_met"):
                    log_src.append(row)
            rows = []
            seen: set[tuple] = set()
            for row in log_src:
                kind = row.get("kind", "")
                if kind not in interesting:
                    continue
                ts = str(row.get("ts", ""))[:19].replace("T", " ")
                msg = row.get("message", kind)
                key = (ts, str(msg)[:60])
                if key in seen:
                    continue
                seen.add(key)
                rows.append({"When (UTC)": ts, "What": msg})
                if len(rows) >= 16:
                    break
            if rows:
                st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=200)
            else:
                st.caption("Quiet for now — practice fills and upgrades will show here.")

        with st.expander("Raw tech details", expanded=False):
            st.code(json.dumps(state, indent=2, default=str)[:4000])

    except Exception as e:
        st.error(f"Live panel error (will retry): {type(e).__name__}: {e}")


live_panel()


def entries_chart_panel() -> None:
    """Price chart lives outside the 12s refresh so zoom/pan + auto price-fit stick."""
    try:
        state = load_autopilot_state()
        equity_all = load_equity(None)
        mode = detect_mode(equity_all)
        trades = load_trades(mode)
        if trades.empty:
            return

        focus = list(state.get("paper_symbols") or state.get("hunt_symbols") or [])
        live = state.get("live") or {}
        elite = state.get("elite") or {}
        last = state.get("last_improve") or {}
        engine_tf = "15m"
        try:
            engine_tf = str(ms.Config.from_env().timeframe or "15m").strip().lower()
        except Exception:
            pass
        for knobs in (
            (live.get("knobs") or {}),
            ((state.get("champion") or {}).get("knobs") or {}),
            (last.get("knobs") or {}),
            (elite.get("knobs") or {}),
        ):
            if knobs.get("timeframe"):
                engine_tf = str(knobs["timeframe"]).strip().lower()
                break

        section_header(
            "Entries & exits",
            "Shows real practice fills only — candle size is a zoom lens, not a new backtest",
        )
        counts = trades["symbol"].value_counts()
        options = [s for s in focus if s in counts.index] + [
            s for s in counts.index if s not in focus
        ]
        if not options:
            options = list(counts.index)
        labels = {
            s: f"{str(s).split('/')[0]} · {int(counts.get(s, 0))} closed"
            for s in options
        }
        c_coin, c_tf = st.columns([2, 1])
        with c_coin:
            top_sym = st.selectbox(
                "Coin",
                options=options,
                index=0,
                format_func=lambda s: labels.get(s, str(s)),
                key="live_entries_coin",
            )
        tf_opts = list(CHART_TIMEFRAMES)
        if engine_tf not in tf_opts:
            tf_opts = [engine_tf] + tf_opts
        default_i = tf_opts.index(engine_tf) if engine_tf in tf_opts else (
            tf_opts.index("15m") if "15m" in tf_opts else 0
        )
        with c_tf:
            chart_tf = st.selectbox(
                "Chart candle size (view only)",
                options=tf_opts,
                index=default_i,
                key="live_chart_tf",
                help=(
                    "Only changes how candles are drawn. Does NOT re-run the strategy on 1m/3m/… "
                    f"The bot currently trades on {engine_tf} candles."
                ),
            )
        n_closed_sym = int(counts.get(top_sym, 0))
        st.caption(
            f"Bot is trading **{engine_tf}** candles right now. "
            f"{str(top_sym).split('/')[0]} has **{n_closed_sym}** closed practice trades in total — "
            "markers only appear where those fills happened (not a 1m simulation)."
        )
        n_bars = chart_bars_for_tf(str(chart_tf), hours=72)
        ohlcv = load_symbol_ohlcv(str(top_sym), str(chart_tf), n_bars)
        if ohlcv.empty:
            st.warning(
                f"No {chart_tf} candles for {str(top_sym).split('/')[0]} yet "
                "(cache warming / exchange). Try again in a minute."
            )
            return

        win = st.radio(
            "Time window",
            options=["auto", "4h", "12h", "1d", "all"],
            index=0,
            horizontal=True,
            key="live_chart_window",
            help="Starting window. After that, scroll/pan freely — price rescales to fit.",
        )
        max_bars = min(len(ohlcv), chart_bars_for_tf(str(chart_tf), hours=48))
        fig = entries_exits_fig(
            ohlcv,
            trades[trades["symbol"] == top_sym],
            title=(
                f"{str(top_sym).split('/')[0]} · view {chart_tf} · "
                f"bot {engine_tf} · {n_closed_sym} fills"
            ),
            max_bars=max(120, int(max_bars)),
            view=str(win),
        )
        show_autofit_price_chart(
            fig,
            height=560,
            key=f"live_{str(top_sym).split('/')[0]}_{chart_tf}_{win}",
        )
        st.caption(
            "Scroll zooms time · drag pans · price auto-fits · ▲/▼ entry · ○ exit"
        )
    except Exception as e:
        st.error(f"Chart error: {type(e).__name__}: {e}")


entries_chart_panel()

if st.button("Force refresh now", type="secondary"):
    st.rerun()
