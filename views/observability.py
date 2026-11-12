"""Pulse — command center: practice honesty, hunt progress, ops health."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dash_lib import (
    ACCENT,
    GREEN,
    MUTED,
    RED,
    db_exists,
    detect_mode,
    empty_state,
    equity_drawdown_fig,
    format_age,
    heartbeat_age,
    inject_theme,
    load_audit_events,
    load_autopilot_state,
    load_cycle_snapshots,
    load_equity,
    load_expectancy_rollups,
    load_fill_events,
    load_hunt_pass_runs,
    load_rejected_entries,
    load_risk,
    load_trades,
    page_header,
    plotly_kwargs,
    plotly_layout,
    polish_fig,
    section_header,
    topbar,
)

inject_theme()

try:
    from streamlit_autorefresh import st_autorefresh  # type: ignore

    st_autorefresh(interval=30_000, key="obs_refresh")
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite(x) -> bool:
    try:
        return x is not None and math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def _edge_label(raw: str | None) -> str:
    return {
        "KEEP-VALIDATED": "Edge looks real",
        "EXTEND-or-WAIT": "Need more trades",
        "STOP-no-edge": "No edge yet",
        "STOP-execution": "Costs eating profits",
        "LEGACY-EXCLUDED": "Old trades (ignore)",
        "EMPTY": "No closes yet",
    }.get(str(raw or ""), str(raw or "—"))


def _kill_label(raw: str) -> str:
    s = str(raw or "")
    mapping = {
        "ok": "Passed",
        "sparse": "Too few trades",
        "period_robust": "Not consistent",
        "truncated_holdout": "Holdout too short",
        "eval_fail": "Could not evaluate",
        "soft_fail": "Failed soft check",
        "confirm_climb": "Failed climb confirm",
        "confirm_consider": "Failed consider",
        "confirm_err": "Confirm error",
        "confirm_dual": "Failed dual-week",
    }
    if s in mapping:
        return mapping[s]
    if s.startswith("material:"):
        return s.split(":", 1)[1][:48]
    return s.replace("_", " ")[:48]


def _event_story(kind: str, reason: str | None) -> str:
    k, r = str(kind or ""), str(reason or "")
    if k == "promote":
        return f"Deployed better book ({r or 'research'})"
    if k == "rotate":
        return f"Rotated while stuck ({r or 'idle'})"
    if k == "freeze":
        return "Promote blocked — frozen"
    if k == "restart":
        return "Practice bot restarted" + (f" · {r}" if r else "")
    if k == "rehydrate":
        return "Restored positions after restart"
    if k == "equity_stale_cleared":
        return "Prices refreshed — equity OK again"
    return k


def _tone(ok) -> str:
    if ok is True:
        return "up"
    if ok is False:
        return "down"
    return "warn"


def _dark(fig: go.Figure, *, height: int = 300, kind: str = "time", **extra) -> go.Figure:
    # Callers may pass showlegend/margin/legend — pop so they don't collide with defaults.
    showlegend = bool(extra.pop("showlegend", True))
    margin = extra.pop("margin", dict(l=56, r=16, t=12, b=40))
    legend = extra.pop(
        "legend",
        dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=MUTED, size=11),
        ),
    )
    fig.update_layout(
        **plotly_layout(
            kind=kind,
            height=height,
            margin=margin,
            showlegend=showlegend,
            legend=legend,
            **extra,
        )
    )
    return polish_fig(fig)


def _panel(title: str, hint: str = "") -> None:
    hint_html = f"<span>{hint}</span>" if hint else ""
    st.markdown(
        f'<div class="ms-panel-head"><h4>{title}</h4>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def _stat_tile(label: str, value: str, hint: str = "", tone: str = "") -> str:
    cls = f"ms-stat {tone}".strip()
    hint_html = f'<div class="ms-stat-hint">{hint}</div>' if hint else ""
    return (
        f'<div class="{cls}">'
        f'<div class="ms-stat-label">{label}</div>'
        f'<div class="ms-stat-value">{value}</div>'
        f"{hint_html}</div>"
    )


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

_eq = load_equity(detect_mode(load_equity(None)))
_bal = float(_eq["equity"].iloc[-1]) if not _eq.empty else None
_ap = load_autopilot_state()
_score = int((_ap.get("grade") or {}).get("score") or 0)
topbar(balance=_bal, score=_score, mode="PAPER", active="Pulse")
page_header(
    "",
    "Pulse",
    "Command center — is practice honest, is the hunt finding real books, is the bot healthy?",
)

if not db_exists():
    empty_state(
        "Nothing to judge yet",
        "Start practice trading first. This page needs a running bot and closed trades.",
    )
    st.stop()

mode = detect_mode(load_equity(None))
equity = load_equity(mode)
risk = load_risk(mode) or {}
cycles = load_cycle_snapshots(mode, limit=400)
rollups = load_expectancy_rollups(mode, limit=200)
fills = load_fill_events(mode, limit=200)
audits = load_audit_events(limit=60)
hunts = load_hunt_pass_runs(limit=40)
trades = load_trades(mode)
rejects = load_rejected_entries(mode=mode, limit=80)
age = heartbeat_age(equity)

frozen = bool(_ap.get("frozen") or False)
try:
    from moonshot.hunt.constants import autopilot_frozen

    frozen = frozen or autopilot_frozen()
except Exception:
    pass

# Primary expectancy bucket (prefer most trades, skip legacy)
primary = None
snap = pd.DataFrame()
contaminated = False
if not rollups.empty:
    latest_ts = rollups["ts"].iloc[0]
    snap = rollups[rollups["ts"] == latest_ts].copy()
    if "label" in snap.columns:
        ranked = snap[snap["label"].astype(str) != "LEGACY-EXCLUDED"]
    else:
        ranked = snap
    if ranked.empty:
        ranked = snap
    if "n" in ranked.columns and not ranked.empty:
        ranked = ranked.sort_values("n", ascending=False)
    primary = ranked.iloc[0] if not ranked.empty else None
    if not audits.empty:
        dirty = audits[
            audits["kind"].isin(["restart", "rehydrate", "promote", "rotate"])
            & (audits["ts"] >= latest_ts - pd.Timedelta(hours=24))
        ]
        contaminated = not dirty.empty

promotes = (
    audits[audits["kind"].isin(["promote", "rotate", "freeze"])]
    if not audits.empty
    else pd.DataFrame()
)
latest_hunt = hunts.iloc[0] if not hunts.empty else None
feed_age = risk.get("feed_age_s")
halt = str(risk.get("halt_reason") or "").strip()
pause = risk.get("pause_until")
eff_lev = float(risk.get("effective_lev") or 0)

# Verdicts
if primary is None:
    edge_v, edge_ok, edge_h = "—", None, "Waiting for scorecard"
else:
    mr = primary.get("mean_r")
    n = int(primary.get("n") or 0)
    edge_v = f"{float(mr):+.2f}R" if _finite(mr) else "—"
    lab = str(primary.get("label") or "")
    if lab == "KEEP-VALIDATED":
        edge_ok = True
    elif _finite(mr) and float(mr) <= 0:
        edge_ok = False
    else:
        edge_ok = None
    edge_h = f"{_edge_label(lab)} · {n} closes" + (" · mixed window" if contaminated else "")

if promotes.empty:
    book_v, book_ok, book_h = "No switch yet", None, "Hunt has not deployed a new book"
else:
    lp = promotes.iloc[0]
    book_v = str(lp.get("kind") or "—").title()
    book_ok = True if lp.get("kind") in ("promote", "rotate") else None
    book_h = _event_story(str(lp.get("kind")), lp.get("reason"))

if latest_hunt is None:
    hunt_v, hunt_ok, hunt_h = "Waiting", None, "Next search still running"
else:
    promoted = int(latest_hunt.get("promote") or 0)
    scored = int(latest_hunt.get("matrix_scored") or 0)
    reason = latest_hunt.get("promote_reason") or latest_hunt.get("material_why") or "hold"
    hunt_v = "Deployed" if promoted else "No deploy"
    hunt_ok = True if promoted else None
    hunt_h = f"{scored} combos scored · {reason}"

if halt:
    ops_v, ops_ok, ops_h = "Paused", False, halt[:80]
elif pause:
    ops_v, ops_ok, ops_h = "Timed pause", None, str(pause)[:32]
elif feed_age is not None and float(feed_age) > 600:
    ops_v, ops_ok, ops_h = "Stale prices", False, f"Feed {int(float(feed_age))}s old"
elif age is not None and age > 180:
    ops_v, ops_ok, ops_h = "Quiet", False, f"Heartbeat {format_age(age)}"
else:
    ops_v, ops_ok, ops_h = "Healthy", True, f"Heartbeat {format_age(age)}"

# ===========================================================================
# 1. Command strip
# ===========================================================================
section_header("Command strip", "Four answers — green good · yellow watch · red problem")
st.markdown(
    f'<div class="ms-stat-grid cols-4 dense">'
    f'{_stat_tile("PRACTICE EDGE", edge_v, edge_h, _tone(edge_ok))}'
    f'{_stat_tile("LAST BOOK CHANGE", book_v, book_h, _tone(book_ok))}'
    f'{_stat_tile("LATEST HUNT", hunt_v, hunt_h, _tone(hunt_ok))}'
    f'{_stat_tile("BOT HEALTH", ops_v, ops_h, _tone(ops_ok))}'
    f"</div>",
    unsafe_allow_html=True,
)

# Gate honesty strip
_exh = bool(_ap.get("hunt_exhausted"))
_nq = int(_ap.get("no_qualify_streak") or 0)
_reassess = bool(_ap.get("regime_reassess_needed"))
_grade = _ap.get("grade") if isinstance(_ap.get("grade"), dict) else {}
_week = _grade.get("week") if isinstance(_grade.get("week"), dict) else {}
try:
    _oos = float(_grade["test_return"]) if _grade.get("test_return") is not None else None
except (TypeError, ValueError):
    _oos = None
try:
    _soft = float(_grade["soft_return"]) if _grade.get("soft_return") is not None else None
except (TypeError, ValueError):
    _soft = None
try:
    _prior = float(_week["prior_ret"]) if _week.get("prior_ret") is not None else None
except (TypeError, ValueError):
    _prior = None
_period_ok = None
if _oos is not None and _prior is not None and _soft is not None:
    _period_ok = _oos >= 0 and _prior >= 0 and _soft >= 0
_last_h = None
for _row in reversed(list(_ap.get("history") or [])):
    if isinstance(_row, dict):
        _last_h = _row
        break
_sealed = (_last_h or {}).get("sealed_holdout") if isinstance(_last_h, dict) else {}
_dsr = (_last_h or {}).get("dsr") if isinstance(_last_h, dict) else {}
_nm = (_last_h or {}).get("nearest_miss") if isinstance(_last_h, dict) else {}
_prov = str((_last_h or {}).get("promote_provenance") or "")
section_header("Gates", "Period-robust · sealed · DSR · exhaustion")
st.markdown(
    f'<div class="ms-stat-grid cols-4 dense">'
    f'{_stat_tile("PERIOD-ROBUST", "OK" if _period_ok else ("FAIL" if _period_ok is False else "—"), f"prior={_prior} oos={_oos} soft={_soft}", _tone(_period_ok))}'
    f'{_stat_tile("SEALED", ("pass" if (_sealed or {}).get("passed") else ("fail" if (_sealed or {}).get("evaluated") else "—")), str((_sealed or {}).get("why") or "")[:48], _tone((_sealed or {}).get("passed") if (_sealed or {}).get("evaluated") else None))}'
    f'{_stat_tile("DSR", ("pass" if (_dsr or {}).get("passed") else ("fail" if (_dsr or {}).get("evaluated") else "—")), str((_dsr or {}).get("why") or "")[:48], _tone((_dsr or {}).get("passed") if (_dsr or {}).get("evaluated") else None))}'
    f'{_stat_tile("EXHAUSTED", "YES" if _exh else "no", f"streak {_nq}" + (" · reassess" if _reassess else ""), "down" if _exh else "up")}'
    f"</div>",
    unsafe_allow_html=True,
)
if isinstance(_nm, dict) and (_nm.get("why") or _nm.get("week_ret") is not None):
    st.caption(
        f"Nearest miss · week={_nm.get('week_ret')} prior={_nm.get('prior_ret')} "
        f"oos={_nm.get('oos_ret')} soft={_nm.get('soft_ret')} · {_nm.get('why') or ''} "
        f"· mode={_nm.get('search_mode') or ''}"
    )
if _prov:
    st.caption(f"Last promote provenance: {_prov}")
if frozen:
    st.warning("Promotions frozen — hunt can find books but will not rewrite .env.")
if contaminated:
    st.caption("⚠ Restart or book change in the last 24h — treat the practice edge carefully.")

# ===========================================================================
# 2. Practice money — wallet + edge over time
# ===========================================================================
st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
_panel("Practice money", "Left: wallet. Right: average trade result (R) over time.")
c_left, c_right = st.columns(2)
with c_left:
    st.caption("Practice wallet ($)")
    if equity.empty:
        empty_state("No equity yet", "Appears after the first bot heartbeat.")
    else:
        # Show recent window so the chart is readable
        eq = equity.tail(800).copy()
        st.plotly_chart(
            polish_fig(equity_drawdown_fig(eq)),
            **plotly_kwargs(height=320),
        )
with c_right:
    st.caption("Average result per trade (R) — latest scorecard each run")
    # Build time series: for each rollup timestamp, take the biggest non-legacy bucket
    edge_ts = pd.DataFrame()
    if not rollups.empty:
        rows = []
        for _ts, g in rollups.groupby("ts", sort=True):
            if "label" in g.columns:
                gg = g[g["label"].astype(str) != "LEGACY-EXCLUDED"]
            else:
                gg = g
            if gg.empty:
                gg = g
            if "n" in gg.columns:
                row = gg.sort_values("n", ascending=False).iloc[0]
            else:
                row = gg.iloc[0]
            rows.append(row)
        edge_ts = pd.DataFrame(rows)
    if edge_ts.empty or "mean_r" not in edge_ts.columns:
        empty_state("No practice scorecard yet", "Written about every 15 minutes while paper runs.")
    else:
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color=MUTED, width=1, dash="dot"))
        if "ci_low" in edge_ts.columns and "ci_high" in edge_ts.columns:
            fig.add_trace(
                go.Scatter(
                    x=edge_ts["ts"],
                    y=edge_ts["ci_high"],
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=edge_ts["ts"],
                    y=edge_ts["ci_low"],
                    fill="tonexty",
                    fillcolor="rgba(240,185,11,0.12)",
                    line=dict(width=0),
                    name="95% confidence band",
                    hovertemplate="Low %{y:.2f}R<extra></extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=edge_ts["ts"],
                y=edge_ts["mean_r"],
                mode="lines+markers",
                name="Avg R per trade",
                line=dict(color=ACCENT, width=2.2),
                marker=dict(size=6),
                hovertemplate="Avg %{y:.2f}R<extra></extra>",
            )
        )
        st.plotly_chart(_dark(fig, height=320), **plotly_kwargs(height=320))
        if primary is not None:
            cost = primary.get("cost_share")
            cost_s = f"{float(cost):.0%}" if _finite(cost) else "—"
            st.caption(
                f"Latest: **{_edge_label(primary.get('label'))}** · "
                f"{int(primary.get('n') or 0)} closes · "
                f"fees+funding took **{cost_s}** of pre-cost profit "
                f"(MEXC-style paper costs)."
            )
st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# 3. Closed trades anatomy
# ===========================================================================
st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
_panel(
    "Closed trades",
    "Each bar is one close. Green = win · red = loss. Hover for coin and reason.",
)
if trades.empty:
    empty_state("No closed trades yet", "Trend books can sit flat for hours. That is normal.")
else:
    t = trades.sort_values("ts").tail(40).copy()
    colors = [GREEN if float(r) > 0 else RED for r in t["r_multiple"].fillna(0)]
    coins = t["symbol"].map(lambda s: str(s).split("/")[0])
    reasons = t["exit_reason"].astype(str) if "exit_reason" in t.columns else ["—"] * len(t)
    fig_r = go.Figure(
        go.Bar(
            x=t["ts"],
            y=t["r_multiple"],
            marker_color=colors,
            name="Result (R)",
            customdata=list(zip(coins, reasons, t["pnl"].fillna(0))),
            hovertemplate=(
                "%{customdata[0]} · %{y:.2f}R · $%{customdata[2]:.2f}<br>"
                "%{customdata[1]}<extra></extra>"
            ),
        )
    )
    fig_r.add_hline(y=0, line=dict(color=MUTED, width=1))
    st.plotly_chart(
        _dark(fig_r, height=280, kind="time", showlegend=False),
        **plotly_kwargs(height=280),
    )
    # MAE/MFE if present
    if "mae_r" in t.columns and t["mae_r"].notna().any():
        a1, a2 = st.columns(2)
        with a1:
            st.caption("Worst dip while open (MAE, R)")
            fig_m = go.Figure(
                go.Scatter(
                    x=t["ts"],
                    y=t["mae_r"],
                    mode="lines+markers",
                    name="MAE",
                    line=dict(color=RED, width=1.6),
                )
            )
            st.plotly_chart(_dark(fig_m, height=220, showlegend=False), **plotly_kwargs(height=220))
        with a2:
            st.caption("Best run-up while open (MFE, R)")
            fig_f = go.Figure(
                go.Scatter(
                    x=t["ts"],
                    y=t["mfe_r"],
                    mode="lines+markers",
                    name="MFE",
                    line=dict(color=GREEN, width=1.6),
                )
            )
            st.plotly_chart(_dark(fig_f, height=220, showlegend=False), **plotly_kwargs(height=220))
st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# 4. Hunt command — funnel + kill reasons
# ===========================================================================
st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
_panel(
    "Hunt search",
    "Primary rank: p25 fold Sortino (week is diagnostic). Left: funnel. Right: kill reasons.",
)
if latest_hunt is None:
    empty_state("No finished hunt pass yet", "Shows up when the current search completes.")
else:
    tried = int(latest_hunt.get("scout_in") or 0)
    kept = int(latest_hunt.get("scout_keep") or 0)
    confirmed = int(latest_hunt.get("confirm_soft") or 0)
    scored = int(latest_hunt.get("matrix_scored") or 0)
    fold_pass = int(latest_hunt.get("fold_pass") or 0)
    cost_pass = int(latest_hunt.get("cost_pass") or 0)
    promoted = int(latest_hunt.get("promote") or 0)
    primary = str(latest_hunt.get("primary_metric") or "p25_fold_sortino")
    n_s = latest_hunt.get("n_unique_strat")
    n_t = latest_hunt.get("n_unique_tf")
    n_l = latest_hunt.get("n_unique_lev")
    cov = ""
    if n_s is not None or n_t is not None or n_l is not None:
        cov = f" · unique S/T/L={n_s}/{n_t}/{n_l}"
    st.caption(
        f"Hunt primary metric: `{primary}` · week = Live goal diagnostic only{cov}"
    )
    h1, h2, h3, h4, h5, h6 = st.columns(6)
    h1.metric("Ideas screened", f"{tried}")
    h2.metric("Survived A", f"{kept}")
    h3.metric("Fold pass", f"{fold_pass}")
    h4.metric("Cost 2× pass", f"{cost_pass}")
    h5.metric("Survived C", f"{confirmed}")
    h6.metric("Deployed?", "Yes" if promoted else "No")

    left, right = st.columns(2)
    with left:
        st.caption("Funnel — Stage A → promote (fold-primary)")
        stages = ["A in", "A keep", "Fold pass", "Cost pass", "C hits", "Deployed"]
        vals = [tried, kept, fold_pass, cost_pass, confirmed, promoted]
        # If scout was skipped (champion short-circuit), show scored path only
        if tried == 0 and scored > 0:
            stages = ["Fully scored", "Deployed"]
            vals = [scored, promoted]
        _bar_colors = [ACCENT] * max(0, len(stages) - 1) + [
            GREEN if promoted else MUTED
        ]
        fig_f = go.Figure(
            go.Bar(
                x=stages,
                y=vals,
                marker_color=_bar_colors[: len(stages)],
                text=vals,
                textposition="outside",
                hovertemplate="%{x}: %{y}<extra></extra>",
            )
        )
        st.plotly_chart(
            _dark(fig_f, height=280, kind="free", showlegend=False),
            **plotly_kwargs(height=280),
        )

    with right:
        st.caption("Why candidates died — latest pass")
        hist = latest_hunt.get("kill_histogram") or {}
        if isinstance(hist, str):
            try:
                hist = json.loads(hist)
            except Exception:
                hist = {}
        hist_plot = {k: int(v) for k, v in hist.items() if str(k) != "ok" and int(v or 0) > 0}
        if not hist_plot:
            st.info("No kill reasons on this pass (or only ‘passed’ cells).")
        else:
            hdf = (
                pd.DataFrame(
                    {"Why": [_kill_label(k) for k in hist_plot], "Count": list(hist_plot.values())}
                )
                .sort_values("Count", ascending=True)
                .tail(12)
            )
            fig_k = go.Figure(
                go.Bar(
                    x=hdf["Count"],
                    y=hdf["Why"],
                    orientation="h",
                    marker_color=ACCENT,
                    hovertemplate="%{y}: %{x}<extra></extra>",
                    cliponaxis=False,
                )
            )
            # Horizontal labels need room; _dark's default l=56 clips "Failed dual-week" etc.
            longest = max((len(str(w)) for w in hdf["Why"]), default=12)
            left_m = max(120, min(280, 8 * longest + 24))
            kill_h = max(280, 28 * len(hdf) + 80)
            st.plotly_chart(
                _dark(
                    fig_k,
                    height=kill_h,
                    kind="free",
                    showlegend=False,
                    margin=dict(l=left_m, r=24, t=12, b=40),
                    yaxis=dict(
                        automargin=True,
                        tickfont=dict(size=12),
                        categoryorder="array",
                        categoryarray=list(hdf["Why"]),
                    ),
                    xaxis=dict(title=None, automargin=True, fixedrange=True),
                    hovermode="closest",
                ),
                **plotly_kwargs(height=kill_h),
            )

    # Hunt trend across recent passes
    if len(hunts) >= 2:
        st.caption("Recent searches — fully scored combos vs deploys")
        hh = hunts.sort_values("ts") if "ts" in hunts.columns else hunts.iloc[::-1]
        fig_t = go.Figure()
        fig_t.add_trace(
            go.Scatter(
                x=hh["ts"] if "ts" in hh.columns else list(range(len(hh))),
                y=hh["matrix_scored"].fillna(0),
                name="Fully scored",
                mode="lines+markers",
                line=dict(color=ACCENT, width=2),
            )
        )
        fig_t.add_trace(
            go.Scatter(
                x=hh["ts"] if "ts" in hh.columns else list(range(len(hh))),
                y=hh["promote"].fillna(0),
                name="Deployed (1=yes)",
                mode="lines+markers",
                line=dict(color=GREEN, width=2),
            )
        )
        st.plotly_chart(_dark(fig_t, height=240), **plotly_kwargs(height=240))

    reason = latest_hunt.get("promote_reason") or "—"
    material = latest_hunt.get("material_why") or "—"
    st.caption(f"Latest outcome: **{reason}** · {material}")
st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# 5. Ops health — only useful charts
# ===========================================================================
st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
_panel(
    "Bot health",
    "Price freshness and account lean. Leverage chart only when positions are open.",
)
o1, o2, o3, o4 = st.columns(4)
o1.metric("Heartbeat", format_age(age))
o2.metric(
    "Price data age",
    f"{int(float(feed_age))}s" if feed_age is not None else "—",
)
o3.metric("Open leverage", f"{eff_lev:.2f}×")
o4.metric(
    "Entries",
    "Paused" if halt else ("Timed pause" if pause else "Allowed"),
)

ops_l, ops_r = st.columns(2)
with ops_l:
    st.caption("Price data age (seconds) — spikes mean stale cache / slow feed")
    if cycles.empty or "feed_age_s" not in cycles.columns:
        st.info("No cycle snapshots yet.")
    else:
        cyc = cycles.sort_values("ts").tail(200)
        fig_age = go.Figure(
            go.Scatter(
                x=cyc["ts"],
                y=cyc["feed_age_s"],
                mode="lines",
                name="Feed age (s)",
                line=dict(color=ACCENT, width=1.8),
                fill="tozeroy",
                fillcolor="rgba(240,185,11,0.08)",
            )
        )
        fig_age.add_hline(
            y=600,
            line=dict(color=RED, width=1, dash="dot"),
            annotation_text="stale >10m",
            annotation_font_color=RED,
        )
        st.plotly_chart(_dark(fig_age, height=260), **plotly_kwargs(height=260))

with ops_r:
    if eff_lev > 0.05 and not cycles.empty and "effective_lev" in cycles.columns:
        st.caption("Account leverage while positions are open")
        cyc = cycles.sort_values("ts").tail(200)
        fig_lev = go.Figure()
        fig_lev.add_trace(
            go.Scatter(
                x=cyc["ts"],
                y=cyc["effective_lev"],
                name="Gross leverage",
                line=dict(color=ACCENT, width=2),
            )
        )
        if "net_lev" in cyc.columns:
            fig_lev.add_trace(
                go.Scatter(
                    x=cyc["ts"],
                    y=cyc["net_lev"],
                    name="Net long/short",
                    line=dict(color=GREEN, width=1.6),
                )
            )
        st.plotly_chart(_dark(fig_lev, height=260), **plotly_kwargs(height=260))
    elif not fills.empty and "slippage_bps" in fills.columns:
        st.caption("Slippage paid vs assumed (basis points)")
        fig_s = go.Figure()
        fig_s.add_trace(
            go.Scatter(
                x=fills["ts"],
                y=fills["slippage_bps"],
                mode="markers+lines",
                name="Actual",
                line=dict(color=ACCENT, width=1.5),
                marker=dict(size=7),
            )
        )
        if "assumed_slippage_bps" in fills.columns:
            fig_s.add_trace(
                go.Scatter(
                    x=fills["ts"],
                    y=fills["assumed_slippage_bps"],
                    mode="lines",
                    name="Assumed in research",
                    line=dict(color=MUTED, width=1.5, dash="dot"),
                )
            )
        st.plotly_chart(_dark(fig_s, height=260), **plotly_kwargs(height=260))
    else:
        st.info(
            "Flat right now — no open leverage to chart. "
            "Slippage chart appears after the next fill."
        )
st.markdown("</div>", unsafe_allow_html=True)

# ===========================================================================
# 6. Compact logs
# ===========================================================================
log_l, log_r = st.columns(2)
with log_l:
    st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
    _panel("Book changes & restarts", "What contaminated the measurement window")
    if audits.empty:
        st.caption("None yet.")
    else:
        marks = audits[
            audits["kind"].isin(
                ["promote", "rotate", "freeze", "restart", "rehydrate", "equity_stale_cleared"]
            )
        ].head(10)
        rows = [
            {
                "When": pd.Timestamp(r["ts"]).strftime("%m-%d %H:%M")
                if pd.notna(r.get("ts"))
                else "—",
                "What happened": _event_story(str(r.get("kind")), r.get("reason")),
            }
            for _, r in marks.iterrows()
        ]
        if rows:
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
        else:
            st.caption("None yet.")
    st.markdown("</div>", unsafe_allow_html=True)

with log_r:
    st.markdown('<div class="ms-panel">', unsafe_allow_html=True)
    _panel("Blocked setups", "Risk engine said no")
    if rejects is None or rejects.empty:
        st.caption("No recent blocks.")
    else:
        show_r = rejects.head(10)
        st.dataframe(
            pd.DataFrame(
                {
                    "When": pd.to_datetime(show_r["ts"], utc=True).dt.strftime("%m-%d %H:%M"),
                    "Coin": show_r["symbol"].map(lambda s: str(s).split("/")[0]),
                    "Side": show_r["direction"].map(
                        lambda d: "Buy" if str(d) == "long" else "Sell"
                    ),
                    "Why": show_r["reason"],
                }
            ),
            width="stretch",
            hide_index=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

# Footer sizes
db_path = Path(__file__).resolve().parent.parent / "moonshot.db"
exp_path = Path(__file__).resolve().parent.parent / "runs" / "experiments.db"
db_mb = db_path.stat().st_size / 1e6 if db_path.exists() else 0.0
exp_mb = exp_path.stat().st_size / 1e6 if exp_path.exists() else 0.0
st.caption(f"Logs · practice {db_mb:.1f} MB · hunt {exp_mb:.1f} MB")
