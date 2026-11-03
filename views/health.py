"""Health check — one button: is my bot good enough to keep practicing?"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import moonshot as ms
from dash_lib import (
    ACCENT, GREEN, MUTED, empty_state, flag_overfit, load_ohlcv,
    inject_theme, plotly_layout, plotly_kwargs, polish_fig, results_display,
    what_to_do, render_action, plain_reading, page_header, section_header,
    pct_of_risk, topbar, load_autopilot_state, load_equity, detect_mode,
)

inject_theme()
_eq = load_equity(detect_mode(load_equity(None)))
_bal = float(_eq["equity"].iloc[-1]) if not _eq.empty else None
_score = int((load_autopilot_state().get("grade") or {}).get("score") or 0)
topbar(balance=_bal, score=_score, mode="PAPER", active="Health")
page_header(
    "",
    "Health",
    "Long-history stress test of whatever is in .env right now — not the Live 7-day climb",
)

import importlib
importlib.reload(ms)

ALL_SYMBOLS = list(ms.DEFAULT_SYMBOLS) + list(getattr(ms, "OPTIONAL_SYMBOLS", []) or [])
cfg0 = ms.Config.from_env()
_ap = load_autopilot_state()
_grade = _ap.get("grade") or {}
_targets = _ap.get("targets") or getattr(ms, "HUNT_TARGETS", {})


def _cfg_get(name: str, default):
    return getattr(cfg0, name, default)


def _env_fingerprint(cfg: ms.Config) -> tuple:
    """Invalidate sticky Health results when hunt promotes a new book."""
    return (
        str(cfg.strategy), str(cfg.timeframe), int(cfg.leverage),
        round(float(cfg.adx_enter), 3), round(float(cfg.adx_exit), 3),
        round(float(cfg.volume_multiplier), 3),
        bool(cfg.require_squeeze), bool(cfg.require_rising_adx),
        bool(cfg.require_btc_regime), int(cfg.cooldown_bars),
        int(cfg.mom_lookback), round(float(cfg.mom_min_atr), 3),
        bool(cfg.enable_pyramiding), bool(cfg.enable_partial),
    )


_env_fp = _env_fingerprint(cfg0)
_prev_fp = st.session_state.get("bt_env_fp")
if "bt" in st.session_state and _prev_fp is not None and _prev_fp != _env_fp:
    # Hunt promoted overnight — old Health numbers are for a different book
    del st.session_state["bt"]
    st.session_state.pop("bt_env_fp", None)
    st.session_state.pop("bt_run_ts", None)

# Live shows the hunt's best grade; Health re-runs a fresh backtest. They diverge on purpose.
_live_score = int(_grade.get("score") or 0)
_live_ret = float(_grade.get("test_return") or 0)
_live_win = float(_grade.get("win") or 0)
_live_n = int(_grade.get("test_n") or 0)
_live_week = _grade.get("week") if isinstance(_grade.get("week"), dict) else {}
_live_week_ret = float(
    _live_week.get("ret") if _live_week.get("ret") is not None
    else (_grade.get("week_return") or 0)
)
st.info(
    f"**Live climb** (what overnight hunt optimizes): score {_live_score}/100 · "
    f"7-day history {_live_week_ret:+.0%} · long OOS {_live_ret:+.0%} · "
    f"win {_live_win:.0%} · n={_live_n}.  \n"
    f"**Health** is a different test: multi-month older/newer replay of current "
    f"`{cfg0.strategy}` {int(cfg0.leverage)}×@{cfg0.timeframe}. "
    f"It will not jump just because the 7-day hunt score climbed — click "
    f"**Check my bot** after a promote to refresh. Numbers stick on this page "
    f"until you re-run (or `.env` changes)."
)


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


def _setting_label(r: ms.BacktestResult) -> str:
    adds = "adds to winners" if r.pyramiding else "no adds"
    early = "early profit take" if r.partial else "let winners run"
    return f"{adds} · {early} · {r.leverage}x size"


st.caption(
    "Splits history into an older period and a newer period. "
    "Your bot must look okay on both — if only the newer half looks good, that is usually luck."
)

section_header("Run check")
run_simple = st.button("Check my bot", type="primary", width="stretch")

# Locked defaults for the simple path — full current .env book, long history
# Bars scale with TF so 15m/5m get a meaningful multi-month window.
_tf = str(_cfg_get("timeframe", "15m"))
_bars_default = int(getattr(ms, "_hunt_bars_for_tf", lambda *_a, **_k: 9000)(_tf, full=True))
_bars_default = max(3000, min(20000, _bars_default))
simple_cfg = {
    "symbols": list(ms.DEFAULT_SYMBOLS),
    "bars": _bars_default,
    "oos": 0.3,
    "grid_mode": False,
    "synthetic": False,
    # Pull a fresh tail so Health isn't frozen on yesterday's CSV cache
    "refresh": True,
    "leverage": int(_cfg_get("leverage", 5)),
    "risk_pct": float(_cfg_get("risk_pct", 0.02)),
    "adx_enter": float(_cfg_get("adx_enter", 32.0)),
    "adx_exit": float(_cfg_get("adx_exit", 24.0)),
    "vol_mult": float(_cfg_get("volume_multiplier", 1.8)),
    "buf": float(_cfg_get("breakout_buffer_atr", 0.15)),
    "enable_pyr": bool(_cfg_get("enable_pyramiding", True)),
    "enable_part": bool(_cfg_get("enable_partial", False)),
}

run_cfg = None
if run_simple:
    run_cfg = dict(simple_cfg)

with st.expander("Advanced options (you can ignore these)", expanded=False):
    with st.form("bt_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            adv_symbols = st.multiselect("Coins", ALL_SYMBOLS, default=list(ms.DEFAULT_SYMBOLS))
            adv_bars = st.number_input("History length (4h bars)", 200, 15_000, 15000, 100)
        with c2:
            adv_oos = st.slider("Newer-period share", 0.1, 0.5, 0.3, 0.05)
            adv_mode = st.radio(
                "What to test",
                ["Only my current settings", "Compare several setting combos"],
            )
        with c3:
            adv_synthetic = st.checkbox("Use fake prices (ignore results)", value=False)
            adv_refresh = st.checkbox("Re-download prices", value=False)
            lev0 = simple_cfg["leverage"]
            adv_leverage = st.selectbox(
                "Leverage", [3, 5, 10],
                index=[3, 5, 10].index(lev0) if lev0 in (3, 5, 10) else 1,
            )
            adv_pyr = st.checkbox("Add to winners", value=simple_cfg["enable_pyr"])
            adv_part = st.checkbox("Take profit early", value=simple_cfg["enable_part"])
        if st.form_submit_button("Run advanced check", width="stretch"):
            run_cfg = {
                "symbols": list(adv_symbols),
                "bars": int(adv_bars),
                "oos": float(adv_oos),
                "grid_mode": str(adv_mode).startswith("Compare"),
                "synthetic": bool(adv_synthetic),
                "refresh": bool(adv_refresh),
                "leverage": int(adv_leverage),
                "risk_pct": simple_cfg["risk_pct"],
                "adx_enter": simple_cfg["adx_enter"],
                "adx_exit": simple_cfg["adx_exit"],
                "vol_mult": simple_cfg["vol_mult"],
                "buf": simple_cfg["buf"],
                "enable_pyr": bool(adv_pyr),
                "enable_part": bool(adv_part),
            }

if run_cfg is not None:
    symbols = run_cfg["symbols"]
    if not symbols:
        st.error("Pick at least one coin.")
        st.stop()
    if run_cfg["adx_exit"] > run_cfg["adx_enter"]:
        st.error("Internal setting conflict. Reset .env defaults.")
        st.stop()

    cfg = replace(
        cfg0,
        leverage=int(run_cfg["leverage"]),
        max_account_leverage=float(min(cfg0.max_account_leverage, run_cfg["leverage"])),
        risk_pct=float(run_cfg["risk_pct"]),
        adx_enter=float(run_cfg["adx_enter"]),
        adx_exit=float(run_cfg["adx_exit"]),
        volume_multiplier=float(run_cfg["vol_mult"]),
        breakout_buffer_atr=float(run_cfg["buf"]),
        enable_pyramiding=bool(run_cfg["enable_pyr"]),
        enable_partial=bool(run_cfg["enable_part"]),
    )

    with st.spinner("Downloading prices and replaying trades… this can take a minute"):
        try:
            raw = load_ohlcv(
                tuple(symbols), int(run_cfg["bars"]), cfg.timeframe,
                bool(run_cfg["synthetic"]), bool(run_cfg["refresh"]),
            )
            data = ms.prepare_data(raw, cfg)
        except ms.BotError as e:
            st.error(
                f"Could not load prices: {e}. "
                "Usually MEXC returned empty candles (rate-limit / TF cache miss). "
                "Retry without Force refresh, or wait for hunt to finish warming cache."
            )
            st.stop()
        except Exception as e:
            st.error(f"Could not load prices: {e}")
            st.stop()

        t0 = min(sd.bars[0].Index for sd in data.values())
        t1 = max(sd.bars[-1].Index for sd in data.values())
        split = t0 + (t1 - t0) * (1 - float(run_cfg["oos"]))

        if run_cfg["grid_mode"]:
            ins = ms.run_grid(data, cfg, end=split)
            oos_res = ms.run_grid(data, cfg, start=split)
        else:
            label = (cfg.enable_pyramiding, cfg.enable_partial, cfg.leverage)
            ins = [ms.BacktestEngine(data, cfg, end=split).run(label)]
            oos_res = [ms.BacktestEngine(data, cfg, start=split).run(label)]
        soft_cfg = ms._soft_kill_cfg(cfg)
        soft_label = (soft_cfg.enable_pyramiding, soft_cfg.enable_partial, soft_cfg.leverage)
        soft_oos = ms.BacktestEngine(data, soft_cfg, start=split).run(soft_label)

    first_sym = symbols[0]
    feats = ms.build_features(raw[first_sym], cfg)
    st.session_state["bt"] = {
        "ins": ins, "oos": oos_res, "split": split, "t0": t0, "t1": t1,
        "synthetic": bool(run_cfg["synthetic"]),
        "verdicts": ms.verdicts(ins) if run_cfg["grid_mode"] else [],
        "feats": feats, "symbol": first_sym, "grid": bool(run_cfg["grid_mode"]),
        "soft_oos": soft_oos,
        "start_eq": float(cfg.starting_equity),
        "cfg_label": (
            f"{cfg.strategy} {int(cfg.leverage)}x@{cfg.timeframe} "
            f"adx={cfg.adx_enter:g}"
        ),
    }
    st.session_state["bt_env_fp"] = _env_fp
    st.session_state["bt_run_ts"] = datetime.now(timezone.utc).isoformat()

if "bt" not in st.session_state:
    empty_state(
        "No check run yet",
        "Click <strong>Check my bot</strong> above. Uses your current practice settings automatically.",
    )
    st.stop()

bt = st.session_state["bt"]
ins: list = bt["ins"]
oos_res: list = bt["oos"]
split = bt["split"]
oos_idx = {(r.pyramiding, r.partial, r.leverage): r for r in oos_res}
_run_ts = str(st.session_state.get("bt_run_ts") or "")
_run_label = (_run_ts[:19].replace("T", " ") + " UTC") if _run_ts else "unknown time"
_cfg_label = bt.get("cfg_label") or f"{cfg0.strategy} {int(cfg0.leverage)}x@{cfg0.timeframe}"
st.caption(
    f"Showing Health run from **{_run_label}** · {_cfg_label}. "
    f"Live 7-day climb is {_live_week_ret:+.0%} — that number updates on Live as hunt promotes; "
    f"this page only updates when you click Check (or after a promote clears a stale run)."
)

if bt["synthetic"]:
    st.error("This used fake prices. Ignore every number and re-run with real prices.")
    st.stop()

paper = ms.Config.from_env()
grid_levs = sorted({r.leverage for r in ins}) or [int(paper.leverage)]
near_lev = min(grid_levs, key=lambda L: abs(L - int(paper.leverage)))
key = (paper.enable_pyramiding, paper.enable_partial, near_lev)
fav_i = next((r for r in ins if (r.pyramiding, r.partial, r.leverage) == key), ins[0])
fav_o = oos_idx.get((fav_i.pyramiding, fav_i.partial, fav_i.leverage), oos_res[0])
soft_o = bt.get("soft_oos")
_health_grade = ms._grade(fav_i, fav_o, soft_o)
_h_score = int(_health_grade.get("score") or 0)

section_header("What should I do?")
action = what_to_do(fav_i, fav_o)
render_action(action)

c_w, c_l, c_h = st.columns(3)
c_w.metric("Live 7-day climb", f"{_live_week_ret:+.0%}", "hunt holdout week")
c_l.metric("Live hunt score", f"{_live_score}/100", f"OOS {_live_ret:+.0%}")
c_h.metric("This Health replay", f"{_h_score}/100", f"OOS {fav_o.total_return:+.0%}")
if abs(_live_score - _h_score) >= 5 or abs(_live_ret - float(fav_o.total_return)) >= 0.15:
    st.caption(
        "Gap is normal: Live optimizes last-7-day history under leverage; "
        "Health scores months of older vs newer data. A better week on Live "
        "does not automatically rewrite this long replay."
    )

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Win rate (newer)", f"{fav_o.win_rate:.0%}")
m2.metric("Avg profit (newer)", pct_of_risk(fav_o.expectancy_r))
m3.metric("Avg profit (older)", pct_of_risk(fav_i.expectancy_r))
m4.metric("Newer trades", fav_o.trades)
m5.metric("Older trades", fav_i.trades)
st.caption(
    f"History: {bt['t0']:%Y-%m-%d} → {bt['t1']:%Y-%m-%d} · "
    f"newer period starts {split:%Y-%m-%d} · "
    f"{plain_reading(flag_overfit(fav_i, fav_o))}"
)

section_header("Practice balance in the replay")
fig = go.Figure()
if len(fav_i.equity_curve):
    fig.add_trace(go.Scatter(
        x=fav_i.equity_curve.index, y=fav_i.equity_curve.values,
        name="Older period", line=dict(color=ACCENT, width=1.5),
    ))
if len(fav_o.equity_curve):
    fig.add_trace(go.Scatter(
        x=fav_o.equity_curve.index, y=fav_o.equity_curve.values,
        name="Newer period (starts fresh)", line=dict(color=GREEN, width=1.5),
    ))
fig.add_vline(x=split, line_width=1, line_dash="dash", line_color=MUTED)
# Frame Y to the series; scroll/pan only moves time (not photo-zoom)
_ys = []
if len(fav_i.equity_curve):
    _ys.extend([float(v) for v in fav_i.equity_curve.values])
if len(fav_o.equity_curve):
    _ys.extend([float(v) for v in fav_o.equity_curve.values])
if _ys:
    _lo, _hi = min(_ys), max(_ys)
    _pad = max(0.5, (_hi - _lo) * 0.12)
    _yr = [_lo - _pad, _hi + _pad]
else:
    _yr = [0, 1]
fig.update_layout(**plotly_layout(
    kind="time",
    uirevision="health-replay",
    height=380,
    title="Fake-money balance during the replay",
    margin=dict(l=48, r=16, t=40, b=40),
    yaxis=dict(range=_yr, fixedrange=True),
))
st.plotly_chart(polish_fig(fig), **plotly_kwargs())


def _flat_note(res, label: str) -> str:
    """Explain long flat stretches: usually a risk kill, not missing data."""
    eq = res.equity_curve
    if eq is None or len(eq) < 2:
        return ""
    ch = eq[eq.diff().fillna(0).abs() > 1e-9]
    if ch.empty:
        return f"{label}: never moved off the start."
    last = ch.index[-1]
    flat_days = (eq.index[-1] - last).total_seconds() / 86400.0
    halt = (res.halted or "").strip()
    if flat_days < 3:
        return ""
    if halt:
        return (
            f"{label}: flat ~{flat_days:.0f} days after {last:%Y-%m-%d} — "
            f"bot stopped itself ({halt})."
        )
    return (
        f"{label}: flat ~{flat_days:.0f} days after {last:%Y-%m-%d} — "
        f"no more trades (no setups or already flat)."
    )


notes = [
    "The jump at the dashed line is normal — the newer period starts from the starting balance again.",
    _flat_note(fav_i, "Older"),
    _flat_note(fav_o, "Newer"),
]
st.caption(" ".join(n for n in notes if n))

# Honest hard vs soft (kills off) — quit-while-ahead vs over-tight kills
if soft_o is not None:
    hard_f = float(fav_o.final_equity)
    soft_f = float(soft_o.final_equity)
    start_eq = float(bt.get("start_eq") or paper.starting_equity)
    gap = hard_f - soft_f
    halted = bool((fav_o.halted or "").strip())
    m_h, m_s, m_d = st.columns(3)
    m_h.metric("Newer · with kills", f"${hard_f:.2f}", f"{fav_o.total_return:+.0%}")
    m_s.metric("Newer · kills off", f"${soft_f:.2f}", f"{soft_o.total_return:+.0%}")
    if gap > start_eq * 0.05 and halted:
        m_d.metric("Gap · quit while ahead", f"${gap:.2f}")
        st.warning(
            f"Kill illusion — stopped early while ahead. Hard ${hard_f:.0f} vs soft "
            f"${soft_f:.0f} ({fav_o.halted}). Edge not proven if trading continues."
        )
    elif gap < -start_eq * 0.05 and halted:
        m_d.metric("Gap · kills cut the run", f"${gap:.2f}")
        st.info(
            f"Kills cut the run short. Soft ${soft_f:.0f} > hard ${hard_f:.0f} "
            f"({fav_o.halted}). Rails may be too tight for this book — edge may be real."
        )
    else:
        m_d.metric("Gap (hard − soft)", f"${gap:.2f}")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Older-period trades**")
    if fav_i.trade_list:
        st.dataframe(_trades_frame(fav_i.trade_list), width="stretch", hide_index=True, height=280)
    else:
        st.caption("None.")
with col_b:
    st.markdown("**Newer-period trades**")
    if fav_o.trade_list:
        st.dataframe(_trades_frame(fav_o.trade_list), width="stretch", hide_index=True, height=280)
    else:
        st.caption("None.")

with st.expander("Details for power users", expanded=False):
    st.markdown(f"Setting judged: `{_setting_label(fav_i)}`")
    left, right = st.columns(2)
    with left:
        st.markdown("#### Older period")
        st.dataframe(results_display(ms.results_as_frame(ins)), width="stretch", hide_index=True)
    with right:
        st.markdown("#### Newer period")
        st.dataframe(results_display(ms.results_as_frame(oos_res)), width="stretch", hide_index=True)

    rows = []
    for a in ins:
        b = oos_idx.get((a.pyramiding, a.partial, a.leverage))
        if not b:
            continue
        rows.append({
            "setting": _setting_label(a),
            "older avg": round(a.expectancy_r, 3),
            "newer avg": round(b.expectancy_r, 3),
            "older n": a.trades,
            "newer n": b.trades,
            "reading": plain_reading(flag_overfit(a, b)),
        })
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    if bt["verdicts"]:
        st.markdown("#### Auto hints")
        for line in bt["verdicts"]:
            st.markdown(f"- {line}")
