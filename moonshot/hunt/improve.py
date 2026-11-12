from __future__ import annotations

import gc
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from typing import Optional

import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    ConfigError,
    DataError,
)
from moonshot.config import (
    Config,
)
from moonshot.features import (
    _btc_symbol_in,
    prepare_data,
)
from moonshot.data import (
    RANGE_SOURCE_MEXC,
)
from moonshot.persist import (
    Persistence,
)
from moonshot.hunt.fold_eval import cost_ok_for_cfg, eval_cfg_folds
from moonshot.hunt.promote_gates import (
    assert_selection_before_sealed,
    escape_sealed_floor_ok,
    live_death_book,
    nearest_miss_payload,
    record_pass_trials,
    research_dsr_gate,
    sealed_holdout_gate,
    selection_end_for_sealed,
)
from moonshot.hunt.constants import (
    HUNT_CHAMPION_STALE_PP,
    HUNT_CHAMP_TRAIL_PASSES,
    HUNT_CHAMP_TRAIL_PP,
    HUNT_DENSER_COOLDOWNS,
    HUNT_DENSER_MAX,
    HUNT_EXHAUSTED_DENSER_MAX,
    HUNT_MATRIX_SOFT_CAP,
    HUNT_MATRIX_TFS,
    HUNT_RETIRE_GRACE_PASSES,
    HUNT_SILENCE_MAX_PEAK_GAP_PP,
    HUNT_SILENCE_ROTATE_ENABLED,
    HUNT_STUCK_SOFTEN_HELD,
    HUNT_STUCK_SOFT_BUDGET,
    PAPER_BLEED_FORCE_ROTATE,
    HUNT_CURRICULUM_SYMBOLS,
    HUNT_HALVING_A_DAYS,
    HUNT_HALVING_A_KEEP,
    HUNT_HALVING_B_KEEP,
    HUNT_SCOUT_TOP,
    HUNT_SEARCH_SYMBOLS,
    HUNT_SOFT_BUDGET,
    HUNT_STRATEGIES,
    HUNT_STUCK_HELD,
    HUNT_TARGETS,
    HUNT_WF_DAYS,
    _cfg_to_env_updates,
    _cfg_to_knobs,
    _hunt_bars_for_tf,
    _hunt_calendar_days,
    _hunt_fetch_raw,
    _hunt_fetch_wf_raw,
    _hunt_price_source,
    _hunt_soft_workers,
    _clear_paper_bleed_circuit,
    _live_feed,
    _load_autopilot_state,
    _patch_env,
    _save_autopilot_state,
    _update_autopilot_state,
    _utc_now,
    _week_delta_needed,
    autopilot_frozen,
)
from moonshot.hunt.scoring import (
    HUNT_EXHAUSTED_GENS,
    PAPER_RETIRE_MIN_N,
    PAPER_RETIRE_REARM_SEC,
    PAPER_SILENCE_ROTATE_HOURS,
    RESEARCH_HYSTERESIS_PASSES,
    RESEARCH_HYSTERESIS_WEEK_PP,
    RESEARCH_LEV_UP_WEEK_PP,
    RESEARCH_MIN_WEEK_EDGE,
    RESEARCH_PROMOTE_DWELL_SEC,
    RESEARCH_PROMOTE_MIN_SEC,
    WEEK_DAYS,
    _paper_bleed,
    _banlist_add,
    _banlist_set,
    _beats,
    _candidate_cfgs,
    _cfg_fingerprint,
    _density_polish_cfgs,
    _raise_champion_peak,
    _resolve_champion_peak,
    _sync_champion_peak,
    _dedupe_cfgs_matrix,
    _discover_seed_cfgs,
    _best_by_tf_raise,
    _best_by_tf_seed_ok,
    _dual_week_eval,
    _dual_week_ok,
    _period_robust_ok,
    _escape_window_ok,
    _eval_cfg,
    _eval_soft_oos,
    _filter_banned,
    _forward_record,
    _freeze_risk_matrix,
    _grade,
    knobs_to_cfg,
    _held_streak,
    _holdout_span_ok,
    _hunt_search_mode,
    _is_climb_ok,
    _is_screenable,
    _is_valid_candidate,
    _lev_density_ok,
    _material_research_edge,
    _paper_idle_hours,
    _prepare_cached,
    _rank_key,
    _rank_matrix_rows,
    _rank_matrix_rows_escape,
    _matrix_soft_pick,
    _book_leverage_bakeoff,
    _recent_eval,
    _matrix_tfs_for_pass,
    _select_polish_base,
    _should_ban_week,
    _should_ban_candidate,
    _tf_lev_matrix_cfgs,
    _week_eval,
    _wf_eval,
    _wf_msg,
    _window_eval,
)

from moonshot.paths import (
    ENV_PATH,
    DB_PATH,
)
import contextlib


def improve_once(
    symbols: list,
    bars: int = 5000,
    oos_frac: float = 0.3,
    refresh: bool = False,
    generation: int = 0,
    hunt: bool = True,
) -> dict:
    """One hunt/improve pass — always writes a finish record (HOLD or abort)."""
    try:
        return _improve_once_impl(
            symbols,
            bars=bars,
            oos_frac=oos_frac,
            refresh=refresh,
            generation=generation,
            hunt=hunt,
        )
    except Exception as e:
        from moonshot.hunt.health import record_pass_abort

        log.exception("improve_once failed mid-pass: %s", e)
        return record_pass_abort(generation=generation, error=e, recover=True)


def _improve_once_impl(
    symbols: list,
    bars: int = 5000,
    oos_frac: float = 0.3,
    refresh: bool = False,
    generation: int = 0,
    hunt: bool = True,
) -> dict:
    """
    One hunt/improve pass. Keeps the global best-ever; aims at HUNT_TARGETS.
    Never enables live trading.
    Fit window excludes last WEEK_DAYS (holdout). Research promote may patch .env;
    elite / targets_met require forward paper proof.
    """
    try:
        from dotenv import load_dotenv as _ld

        _ld(ENV_PATH, override=True)
    except ImportError:
        pass
    base = Config.from_env()
    # Hunt evals use research notional; paper Live keeps STARTING_EQUITY.
    research_eq = float(getattr(base, "research_starting_equity", 50.0) or 50.0)
    base = replace(base, starting_equity=research_eq)
    mode = "HUNT" if hunt else "TUNE"
    state = _load_autopilot_state()
    held = _held_streak(state)
    champion_preview = _resolve_champion_peak(state)
    if champion_preview:
        _sync_champion_peak(state, champion_preview)
    has_champion = bool(champion_preview and champion_preview.get("knobs"))
    hunt_exhausted = bool(state.get("hunt_exhausted"))
    search_mode = _hunt_search_mode(
        generation,
        held,
        hunt,
        has_champion=has_champion,
        hunt_exhausted=hunt_exhausted,
    )
    full_symbols = list(symbols)
    # BTC must be present for the altcoin regime gate
    if base.require_btc_regime and _btc_symbol_in(symbols) is None:
        symbols = ["BTC/USDT:USDT"] + list(symbols)
        full_symbols = list(symbols)
    _mode_blurb = {
        "polish": "nudges around current best…",
        "explore": "wider explore to escape a local peak…",
        "breakout": "BREAKOUT search — large jumps across filters/exits…",
    }.get(search_mode, "searching…")
    _live_feed(
        "improve_start",
        f"{mode} pass #{generation} · {search_mode} — {_mode_blurb}",
        generation=generation,
        bars=bars,
        symbols=len(symbols),
        mode=mode.lower(),
        search_mode=search_mode,
        held_streak=held,
    )
    # Pass lifecycle — finish signal for MemoryMax / OOM-restart detection.
    # Flock update so _live_feed / watchdog cannot clobber pass_started_at.
    # force_skip_denser = one-shot denser skip only — matrix always FULL TFs.
    _pass_box: dict = {"skip_denser": False, "recover_reason": None, "ts": ""}

    def _start_pass(st: dict) -> None:
        # Migrate legacy TF-slim flag → denser skip (never slim matrix TFs).
        legacy_slim = bool(st.pop("force_matrix_slim", False))
        skip = bool(st.pop("force_skip_denser", False)) or legacy_slim
        _pass_box["skip_denser"] = skip
        _pass_box["recover_reason"] = st.pop("hunt_recover_reason", None)
        if skip and not _pass_box["recover_reason"]:
            _pass_box["recover_reason"] = "pass_stuck"
        ts = _utc_now().isoformat()
        _pass_box["ts"] = ts
        st["pass_started_at"] = ts
        st["pass_heartbeat_at"] = ts
        st["busy"] = True

    _update_autopilot_state(_start_pass)
    skip_denser = bool(_pass_box["skip_denser"])
    recover_reason = _pass_box["recover_reason"]
    state["pass_started_at"] = _pass_box["ts"]
    state["pass_heartbeat_at"] = _pass_box["ts"]
    state["busy"] = True
    state.pop("force_matrix_slim", None)
    state.pop("force_skip_denser", None)
    state.pop("hunt_recover_reason", None)
    if skip_denser:
        log.info(
            "force_skip_denser · reason=%s · denser skipped this pass · matrix stays FULL",
            recover_reason,
        )
    matrix_failed = False

    def _pass_heartbeat() -> None:
        ts = _utc_now().isoformat()
        state["pass_heartbeat_at"] = ts

        def _hb(st: dict) -> None:
            st["pass_heartbeat_at"] = ts
            st["pass_started_at"] = state.get("pass_started_at") or st.get("pass_started_at")
            st["busy"] = True

        with contextlib.suppress(Exception):
            _update_autopilot_state(_hb)

    # Funnel hunt: confirm book (multi-coin) + scout book (BTC only).
    tf_s = str(base.timeframe)
    hunt_tier = "confirm" if hunt else "finalist"
    hunt_src = _hunt_price_source()
    hunt_days = _hunt_calendar_days(hunt_tier, tf_s)
    bars = int(_hunt_bars_for_tf(tf_s, full=not hunt))  # for status / legacy feeds
    n_confirm = int(HUNT_SEARCH_SYMBOLS)
    hunt_syms = list(symbols)
    btc = _btc_symbol_in(hunt_syms)
    rest = [s for s in hunt_syms if s != btc]
    core = ([btc] if btc else []) + rest
    if hunt and len(core) > n_confirm:
        core = core[:n_confirm]
    symbols = core  # multi-coin confirm + promote / Live week
    scout_sym = btc or (symbols[0] if symbols else None)
    scout_symbols = [scout_sym] if scout_sym else list(symbols[:1])
    log.info(
        "hunt funnel · scout=%s · confirm ~%dd (%d bars est) × %d symbols tf=%s source=%s",
        ",".join(s.split("/")[0] for s in scout_symbols),
        hunt_days,
        bars,
        len(symbols),
        tf_s,
        hunt_src,
    )
    raw = _hunt_fetch_raw(
        symbols,
        base.timeframe,
        hunt_tier,
        refresh=refresh,
        source=hunt_src,
    )
    if not raw:
        raise DataError(
            f"hunt could not load any OHLCV (source={hunt_src} tier={hunt_tier} tf={tf_s})"
        )
    feat_cache: dict = {}
    # Per-TF maps for confirm (multi-coin) and scout (BTC)
    raw_by_tf: dict = {str(base.timeframe): raw}
    scout_raw_by_tf: dict = {
        str(base.timeframe): {s: raw[s] for s in scout_symbols if s in raw},
    }

    def _raw_for_tf(tf: str) -> dict:
        key = str(tf)
        hit = raw_by_tf.get(key)
        if hit is not None:
            return hit
        # Same calendar tier for every TF so 5m vs 15m compare equal wall-clock.
        loaded = _hunt_fetch_raw(
            symbols,
            key,
            hunt_tier,
            refresh=False,
            source=hunt_src,
        )
        raw_by_tf[key] = loaded
        return loaded

    def _scout_raw_for_tf(tf: str) -> dict:
        key = str(tf)
        hit = scout_raw_by_tf.get(key)
        if hit is not None:
            return hit
        full = _raw_for_tf(key)
        slim = {s: full[s] for s in scout_symbols if s in full}
        scout_raw_by_tf[key] = slim
        return slim

    base_data = _prepare_cached(raw, base, feat_cache)
    t0 = min(sd.bars[0].Index for sd in base_data.values())
    t1 = max(sd.bars[-1].Index for sd in base_data.values())
    data_end = pd.Timestamp(t1)
    # Blind sealed exam: last SEALED days unseen by selection ranking.
    selection_end = selection_end_for_sealed(data_end)
    assert_selection_before_sealed(selection_end, data_end)
    # Selection holdout = WEEK_DAYS ending at selection_end (before sealed).
    holdout_start = selection_end - pd.Timedelta(days=int(WEEK_DAYS))
    fit_end = holdout_start
    if fit_end <= t0 + pd.Timedelta(days=14):
        log.warning("holdout window too tight (t0=%s t1=%s) — widening fit", t0, t1)
        fit_end = t0 + (selection_end - t0) * 0.85
        holdout_start = fit_end
    split = t0 + (fit_end - t0) * (1 - oos_frac)
    week_end = selection_end  # ranking / period-robust use selection holdout only
    log.info(
        "selection holdout %s → %s · sealed %s → %s (fit ends %s); split %s",
        holdout_start.date(),
        selection_end.date(),
        selection_end.date(),
        data_end.date(),
        fit_end.date(),
        split.date(),
    )

    # Walk-forward on history the hunt never sees (before t0). Fail closed on grade.
    wf_raw: Optional[dict] = None
    wf_cache: dict = {}
    try:
        wf_raw = _hunt_fetch_wf_raw(
            symbols,
            base.timeframe,
            t0,
            refresh=False,
            source=hunt_src,
        )
        if not wf_raw:
            raise RuntimeError("no walk-forward history for any symbol")
        min_len = 800 if tf_s in ("3m", "5m") else 1500
        wf_raw = {s: df for s, df in wf_raw.items() if df is not None and len(df) > min_len}
        if not wf_raw:
            raise RuntimeError("walk-forward series too short after filter")
        need_days = 40 if tf_s in ("3m", "5m") else int(HUNT_WF_DAYS) // 2
        if min(df.index[0] for df in wf_raw.values()) >= t0 - pd.Timedelta(days=need_days):
            log.warning("walk-forward: not enough history before %s", t0)
            wf_raw = None
    except Exception as e:
        log.warning("walk-forward data unavailable: %s", e)
        wf_raw = None
    wf_enabled = wf_raw is not None
    if not wf_enabled:
        log.warning(
            "walk-forward DISABLED for %s this pass — research promote still needs other gates",
            tf_s,
        )

    def _wf_for(cfg: Config) -> Optional[dict]:
        if not wf_enabled:
            return None
        try:
            return _wf_eval(_prepare_cached(wf_raw, cfg, wf_cache), cfg, t0)
        except Exception as e:
            log.warning("walk-forward eval failed: %s", e)
            return {
                "ok": False,
                "why": f"error: {e}",
                "folds": [],
                "trades": 0,
                "exp": 0.0,
                "green": "0/0",
                "worst": 0.0,
                "span": "",
            }

    # Forward (paper) record: clock starts when this strategy first ran in paper
    if not state.get("forward_since"):
        state["forward_since"] = _utc_now().isoformat()
        _save_autopilot_state(state)
    fwd = _forward_record(list(symbols), str(state["forward_since"]))

    base_i, base_o = _eval_cfg(base_data, base, split, end=fit_end)
    base_soft = _eval_soft_oos(base_data, base, split, end=fit_end)
    base_wf = _wf_for(base)
    base_recent = _recent_eval(base_data, base, end=week_end)
    base_week = _dual_week_eval(base_data, base, end=week_end)
    base_sealed_week = _week_eval(base_data, base, end=data_end)
    base_grade = _grade(
        base_i,
        base_o,
        base_soft,
        base_wf,
        wf_required=True,
        fwd=fwd,
        fwd_required=True,
        week=base_week,
    )
    base_grade["recent"] = base_recent
    base_grade["week"] = base_week

    elite = state.get("elite")  # best-ever across all generations (forward-confirmed)
    # Champion = best holdout-week book ever kept (independent of forward proof).
    # Live / research promote must not fall below this week. Rebuild from history
    # if a reseed/wipe left a weaker floor (that is how −26% got applied).
    champion = _resolve_champion_peak(state)
    if champion:
        _sync_champion_peak(state, champion)

    best_cfg = base
    best_i, best_o, best_soft = base_i, base_o, base_soft
    best_wf = base_wf
    best_recent = dict(base_recent)
    best_week = dict(base_week)
    best_grade = dict(base_grade)
    try:
        best_fold = eval_cfg_folds(base_data, base, end=week_end, smoke=False)
    except Exception:
        best_fold = {"ok": False, "p25_sortino": float("nan"), "roi": float(base_o.total_return)}
    try:
        best_cost_ok, _, _ = cost_ok_for_cfg(base_data, base, end=week_end)
    except Exception:
        best_cost_ok = True
    best_roi = float(best_fold.get("roi") if best_fold.get("roi") is not None else base_o.total_return)
    (not wf_enabled) or bool(base_wf and base_wf.get("ok"))
    promote_reason = "baseline"
    silence_rotate = False
    promote_provenance: str | None = None  # "robust" | "escape"
    escape_why: str | None = None

    def _consider(
        cand_cfg: Config,
        ci_,
        co_,
        cs_,
        label_: str = "",
        *,
        fold: dict | None = None,
        cost_ok: bool | None = None,
    ) -> bool:
        """Promote cand in-memory if it beats incumbent on fold-primary rank."""
        nonlocal best_cfg, best_i, best_o, best_soft, best_wf, best_grade
        nonlocal best_recent, best_week, best_fold, best_cost_ok, best_roi, promote_reason
        # ≥10x must arrive with soft OOS already evaluated
        if int(cand_cfg.leverage) >= 10 and cs_ is None:
            return False
        best_ok = (not wf_enabled) or bool(best_wf and best_wf.get("ok"))
        best_recent_ok = bool(best_recent.get("ok"))
        cwf = _wf_for(cand_cfg)
        cand_ok = (not wf_enabled) or bool(cwf and cwf.get("ok"))
        if wf_enabled and not cand_ok:
            return False
        # Always score week/recent/folds on the multi-coin confirm book (correct TF).
        c_raw = _raw_for_tf(str(cand_cfg.timeframe)) or raw
        cdata = _prepare_cached(c_raw, cand_cfg, feat_cache)
        crecent = _recent_eval(cdata, cand_cfg, end=week_end)
        cweek = _dual_week_eval(cdata, cand_cfg, end=week_end)
        cfold = fold if isinstance(fold, dict) else eval_cfg_folds(cdata, cand_cfg, end=week_end, smoke=False)
        if cost_ok is None:
            try:
                c_cost_ok, _, _ = cost_ok_for_cfg(cdata, cand_cfg, end=week_end)
            except Exception:
                c_cost_ok = True
        else:
            c_cost_ok = bool(cost_ok)
        c_roi = float(cfold.get("roi") if cfold.get("roi") is not None else co_.total_return)
        week_min = int(HUNT_TARGETS.get("week_min_trades", 4))
        if (
            int(cweek.get("trades") or 0) < week_min
            and int(cweek.get("soft_trades") or 0) < week_min
        ):
            return False
        # Never replace a healthy recent with a worse one
        if not crecent.get("ok") and best_recent_ok:
            return False
        # Relative climb when both fail the 90d floor
        recent_relative = (
            not best_recent_ok
            and float(crecent.get("exp") or -99) >= float(best_recent.get("exp") or -99)
            and float(crecent.get("ret") or -99) >= float(best_recent.get("ret") or -99)
        )
        week_tgt = float(HUNT_TARGETS.get("week_return", 1.0))
        min_delta = _week_delta_needed(float(best_week.get("ret") or 0), week_tgt)
        week_active = (
            int(cweek.get("trades") or 0) >= week_min
            or int(cweek.get("soft_trades") or 0) >= week_min
        )
        # Week edge is diagnostic rescue only — must not demote fold primary.
        try:
            cand_p25 = float(cfold.get("p25_sortino"))
        except (TypeError, ValueError):
            cand_p25 = float("nan")
        try:
            best_p25 = float(best_fold.get("p25_sortino"))
        except (TypeError, ValueError):
            best_p25 = float("nan")
        if cand_p25 != cand_p25:
            cand_p25 = -999.0
        if best_p25 != best_p25:
            best_p25 = -999.0
        week_edge = (
            week_active
            and float(cweek.get("ret") or 0) > float(best_week.get("ret") or 0) + min_delta
            and cand_p25 >= best_p25 - 0.05
        )
        # Less-bad OOS / soft when incumbent is losing — lets hunt climb before week flips
        losing_book = float(best_o.total_return) <= 0 or float(best_o.expectancy_r) <= 0
        oos_edge = (
            losing_book
            and float(co_.total_return) > float(best_o.total_return) + 0.005
            and float(co_.expectancy_r) >= float(best_o.expectancy_r) - 0.05
            and cand_p25 >= best_p25 - 0.05
        )
        soft_edge = (
            losing_book
            and cs_ is not None
            and best_soft is not None
            and float(cs_.total_return) > float(best_soft.total_return) + 0.005
            and cand_p25 >= best_p25 - 0.05
        )
        recent_edge = (
            float(crecent.get("ret") or 0) > float(best_recent.get("ret") or 0) + 0.01
            and float(crecent.get("exp") or 0) >= float(best_recent.get("exp") or 0) - 0.02
            and cand_p25 >= best_p25 - 0.05
        )
        cand_beats = _beats(
            ci_,
            co_,
            best_i,
            best_o,
            cs_,
            best_soft,
            cweek,
            best_week,
            a_fold=cfold,
            b_fold=best_fold,
            a_cost_ok=c_cost_ok,
            b_cost_ok=bool(best_cost_ok),
            a_roi=c_roi,
            b_roi=float(best_roi),
        )
        rescue = (
            (wf_enabled and not best_ok and cand_ok)
            or (not best_recent_ok and bool(crecent.get("ok")))
            or recent_edge
            or recent_relative
            or week_edge
            or oos_edge
            or soft_edge
        )
        if not cand_beats and not rescue:
            return False
        # PRIMARY floor: never replace in-memory best with a worse p25 fold Sortino.
        if cand_p25 < best_p25 - 0.001:
            return False
        # Period-robust: never let a red prior/OOS/soft book become in-memory best.
        if HUNT_TARGETS.get("require_period_robust", True):
            _soft_v = float(cs_.total_return) if cs_ is not None else float(co_.total_return)
            _rok, _rwhy = _period_robust_ok(cweek, float(co_.total_return), _soft_v)
            if not _rok:
                return False
        reason = label_ or (
            "fold_rank"
            if cand_beats
            else "week_ladder"
            if week_edge
            else "recent_rescue"
            if (recent_edge or recent_relative)
            else "rescue"
        )
        best_cfg, best_i, best_o, best_soft, best_wf = cand_cfg, ci_, co_, cs_, cwf
        best_recent = crecent
        best_week = cweek
        best_fold = dict(cfold)
        best_cost_ok = bool(c_cost_ok)
        best_roi = float(c_roi)
        best_grade = _grade(
            ci_,
            co_,
            cs_,
            cwf,
            wf_required=True,
            fwd=fwd,
            fwd_required=True,
            week=cweek,
        )
        best_grade["recent"] = crecent
        best_grade["week"] = cweek
        best_grade["fold"] = best_fold
        best_grade["p25_sortino"] = cand_p25
        best_grade["cost_ok"] = best_cost_ok
        promote_reason = reason
        return True

    def _knobs_to_cfg(kn: dict) -> Config:
        """Restore full champion/elite identity (strategy + TF + lev), not Live pins."""
        return knobs_to_cfg(kn, base=base)

    def _knobs_to_cfg_matrix(kn: dict) -> Config:
        """Same as _knobs_to_cfg — matrix freeze already restores S/T/L."""
        return knobs_to_cfg(kn, base=base)

    # Seed from elite — cross-family allowed on breakout OR when hunt is stuck
    if elite and elite.get("knobs"):
        try:
            kn = elite["knobs"]
            elite_strat = str(kn.get("strategy", "") or "").strip().lower()
            stuck_now = int(held) >= int(HUNT_STUCK_HELD)
            if (
                elite_strat
                and elite_strat != str(base.strategy).strip().lower()
                and search_mode != "breakout"
                and not stuck_now
                and not hunt_exhausted
            ):
                raise ValueError(
                    f"elite strategy {elite_strat} != {base.strategy} (wait for breakout)"
                )
            elite_cfg = _knobs_to_cfg(kn)
            e_raw = _raw_for_tf(elite_cfg.timeframe)
            ed = _prepare_cached(e_raw, elite_cfg, feat_cache)
            ei, eo = _eval_cfg(ed, elite_cfg, split, end=fit_end)
            es = _eval_soft_oos(ed, elite_cfg, split, end=fit_end)
            if _is_climb_ok(ei, eo, es, base_i, base_o) and _consider(
                elite_cfg, ei, eo, es, "elite"
            ):
                log.info(
                    "restored elite seed score=%d win=%.0f%% test_exp=%+.2f soft=%+.0f%% n=%d",
                    best_grade["score"],
                    eo.win_rate * 100,
                    eo.expectancy_r,
                    es.total_return * 100,
                    eo.trades,
                )
        except Exception as e:
            log.warning("elite re-eval failed: %s", e)

    # Seed champion so hunt cannot forget a better week book
    if champion and champion.get("knobs"):
        try:
            ch_week_ret = float(champion.get("week_ret") or 0)
            if ch_week_ret > float(best_week.get("ret") or 0) + 0.001:
                ch_cfg = _knobs_to_cfg(champion["knobs"])
                ch_raw = _raw_for_tf(ch_cfg.timeframe)
                ch_data = _prepare_cached(ch_raw, ch_cfg, feat_cache)
                chi, cho = _eval_cfg(ch_data, ch_cfg, split, end=fit_end)
                chs = _eval_soft_oos(ch_data, ch_cfg, split, end=fit_end)
                if _is_climb_ok(chi, cho, chs, base_i, base_o) and _consider(
                    ch_cfg, chi, cho, chs, "champion"
                ):
                    log.info(
                        "restored champion week=%+.0f%% score=%s",
                        ch_week_ret * 100,
                        champion.get("score"),
                    )
        except Exception as e:
            log.warning("champion restore failed: %s", e)

    # Re-apply a climb that was found earlier but blocked by the settle timer
    pending = state.get("pending_promote") or {}
    if pending.get("knobs"):
        try:
            pend_week = float(pending.get("week_ret") or 0)
            champ_floor = float((champion or {}).get("week_ret") or base_week.get("ret") or -999)
            # Never restore a pending book worse than champion / deployed week
            if pend_week + 0.001 < champ_floor:
                log.info(
                    "dropping stale pending week=%+.0f%% < champion/base %+0.f%%",
                    pend_week * 100,
                    champ_floor * 100,
                )
                state.pop("pending_promote", None)
                _save_autopilot_state(state)
            else:
                p_cfg = _knobs_to_cfg(pending["knobs"])
                p_raw = _raw_for_tf(p_cfg.timeframe)
                pd_ = _prepare_cached(p_raw, p_cfg, feat_cache)
                pi, po = _eval_cfg(pd_, p_cfg, split, end=fit_end)
                ps = _eval_soft_oos(pd_, p_cfg, split, end=fit_end)
                if _is_climb_ok(pi, po, ps, base_i, base_o) and _consider(
                    p_cfg, pi, po, ps, "pending"
                ):
                    log.info(
                        "restored pending climb week=%+.0f%% score=%s",
                        pend_week * 100,
                        pending.get("score"),
                    )
        except Exception as e:
            log.warning("pending promote restore failed: %s", e)

    # ---- Paper retire: deployed book M≥40 mean R≤0 → rescue once + cooldown ----
    # Do not pin skip_heavy forever: arm once, clear after window, re-arm only after cooldown.
    try:
        from moonshot.config_hash import config_hash as _retire_hash
        from moonshot.paper_expectancy import bucket_trades, load_trade_rows
        from moonshot.paths import DB_PATH as _retire_db

        _bch = _retire_hash(base)
        _hit = next(
            (
                b
                for b in bucket_trades(load_trade_rows(_retire_db))
                if b.config_hash == _bch and not b.legacy
            ),
            None,
        )
        if (
            _hit is not None
            and int(_hit.n) >= int(PAPER_RETIRE_MIN_N)
            and float(_hit.mean_r) <= 0.0
        ):
            now_r = _utc_now()
            last_at = state.get("paper_retire_at")
            cooldown_until = state.get("paper_retire_cooldown_until")
            in_cooldown = False
            if cooldown_until:
                try:
                    in_cooldown = now_r < pd.Timestamp(cooldown_until)
                except Exception:
                    in_cooldown = False
            if (
                state.get("pending_champion_rescue")
                and state.get("paper_retire_source") == "paper_retire"
                and last_at
            ):
                try:
                    age_r = (now_r - pd.Timestamp(last_at)).total_seconds()
                except Exception:
                    age_r = 0.0
                if age_r >= float(PAPER_RETIRE_REARM_SEC):
                    state["pending_champion_rescue"] = False
                    state["paper_retire_source"] = None
                    state["paper_retire_cooldown_until"] = (
                        now_r + pd.Timedelta(seconds=float(PAPER_RETIRE_REARM_SEC))
                    ).isoformat()
                    in_cooldown = True
                    log.info(
                        "paper retire window ended · resume heavy search · "
                        "re-arm after %dh cooldown",
                        int(PAPER_RETIRE_REARM_SEC // 3600),
                    )
                    _save_autopilot_state(state)
            if not in_cooldown and not state.get("pending_champion_rescue"):
                state["pending_champion_rescue"] = True
                state["paper_retire_at"] = now_r.isoformat()
                state["paper_retire_hash"] = _bch
                state["paper_retire_source"] = "paper_retire"
                state["paper_retire_grace_passes"] = 0
                state.pop("paper_retire_cooldown_until", None)
                _save_autopilot_state(state)
                log.warning(
                    "paper RETIRE · hash=%s n=%d mean_R=%+.3f → pending_champion_rescue",
                    _bch[:8],
                    int(_hit.n),
                    float(_hit.mean_r),
                )
                _live_feed(
                    "scoreboard_critical",
                    f"paper_retire · n={int(_hit.n)} mean_R={float(_hit.mean_r):+.3f} "
                    "— champion rescue armed",
                    config_hash=_bch[:8],
                )
            elif state.get("pending_champion_rescue") and state.get("paper_retire_source") == "paper_retire":
                grace = int(state.get("paper_retire_grace_passes") or 0) + 1
                state["paper_retire_grace_passes"] = grace
                if grace >= int(HUNT_RETIRE_GRACE_PASSES) and not state.get(
                    "regime_reassess_needed"
                ):
                    state["regime_reassess_needed"] = True
                    state["regime_reassess_at"] = now_r.isoformat()
                    log.warning(
                        "paper retire grace exhausted · %d passes without robust fix → regime_reassess",
                        grace,
                    )
                    _live_feed(
                        "scoreboard_critical",
                        f"paper_retire_grace · {grace} passes — regime reassess needed",
                        config_hash=_bch[:8],
                    )
                _save_autopilot_state(state)
    except Exception as e:
        log.info("paper retire check skipped: %s", e)

    # ---- TF×lev matrix (before polish) — like-for-like bake-off ----
    matrix_winner_cfg: Config | None = None
    matrix_winner_row: dict | None = None
    tf_lev_matrix_rows: list[dict] = []
    matrix_escape = False
    matrix_tfs: list[str] = []
    _pass_t0 = time.monotonic()
    _timing: dict[str, float] = {
        "matrix_hard_s": 0.0,
        "matrix_soft_s": 0.0,
        "stage_a_s": 0.0,
        "stage_b_s": 0.0,
        "stage_c_s": 0.0,
    }
    # Champ short-circuit: when trail/pending is hot, skip heavy search later.
    skip_heavy_search = bool(state.get("pending_champion_rescue")) or (
        int(state.get("champ_trail_streak") or 0) >= int(HUNT_CHAMP_TRAIL_PASSES)
    )
    # Full TF matrix when stuck/explore/breakout — do NOT slim just because
    # Live week ≥15% (that locked hunt onto 5m-only for dozens of HOLDs).
    try:
        matrix_tfs = _matrix_tfs_for_pass(
            base_tf=str(base.timeframe),
            held=int(held),
            search_mode=str(search_mode),
            skip_heavy_search=bool(skip_heavy_search),
            live_week_ret=float(base_week.get("ret") or 0),
            hunt_exhausted=bool(state.get("hunt_exhausted")),
        )
        log.info(
            "tf×lev matrix FULL · tfs=%s · held=%s mode=%s live_week=%+.0f%%",
            matrix_tfs,
            held,
            search_mode,
            float(base_week.get("ret") or 0) * 100,
        )
        # One TF at a time — never hold 5m+15m+1h+4h OHLCV in RAM together on
        # the 2G droplet (that OOMed mid-pass and froze Recent searches).
        _matrix_cfgs = [
            c for c in _tf_lev_matrix_cfgs(base) if str(c.timeframe) in {str(t) for t in matrix_tfs}
        ]
        matrix_cells = [_freeze_risk_matrix(base, c) for c in _matrix_cfgs]
        hard_rows: list[dict] = []
        cell_by_key: dict[tuple[str, int], Config] = {}
        cells_by_tf: dict[str, list[Config]] = {}
        for cand in matrix_cells:
            cells_by_tf.setdefault(str(cand.timeframe), []).append(cand)
            cell_by_key[(str(cand.timeframe), int(cand.leverage))] = cand

        def _as_naive_utc(ts) -> pd.Timestamp:
            out = pd.Timestamp(ts)
            if getattr(out, "tzinfo", None) is not None:
                out = out.tz_convert("UTC").tz_localize(None)
            return out

        def _prune_tf(tf_drop: str) -> None:
            if str(tf_drop) == str(base.timeframe):
                return
            raw_by_tf.pop(str(tf_drop), None)
            scout_raw_by_tf.pop(str(tf_drop), None)
            _stats = feat_cache.pop("_stats", None)
            feat_cache.clear()
            if _stats is not None:
                feat_cache["_stats"] = _stats
            gc.collect()

        _t_hard = time.monotonic()
        for tf_key in matrix_tfs:
            tf_s_m = str(tf_key)
            for cand in cells_by_tf.get(tf_s_m, []):
                key = (str(cand.timeframe), int(cand.leverage))
                row: dict = {
                    "leverage": int(cand.leverage),
                    "tf": str(cand.timeframe),
                    "strategy": str(cand.strategy),
                    "skip": False,
                    "dense_ok": False,
                    "robust_ok": False,
                    "week_ret": None,
                    "prior_ret": None,
                    "test_ret": None,
                    "soft_ret": None,
                    "week_trades": 0,
                    "week_win": None,  # holdout-window win rate (not hunt grade score)
                }
                try:
                    use_raw = _raw_for_tf(cand.timeframe)
                    if not use_raw:
                        row["skip"] = True
                        row["why"] = "no_raw"
                        hard_rows.append(row)
                        continue
                    data = _prepare_cached(use_raw, cand, feat_cache)
                    # Same calendar end as Live scorecard (like-for-like).
                    last_bar_n = _as_naive_utc(max(sd.bars[-1].Index for sd in data.values()))
                    ho_start_n = _as_naive_utc(t1) - pd.Timedelta(days=int(WEEK_DAYS))
                    if last_bar_n < ho_start_n:
                        row["skip"] = True
                        row["why"] = "truncated_holdout"
                        hard_rows.append(row)
                        continue
                    ci, co = _eval_cfg(data, cand, split, end=fit_end)
                    # Hard ranks on primary week only; soft reloads dual-week + period_robust.
                    # Overlap gate (week vs week+prior top-5) measured ≥4/5 on live matrix.
                    cweek = _week_eval(data, cand, end=week_end)
                    span_days = float(WEEK_DAYS)
                    span_s = str(cweek.get("span") or "")
                    try:
                        if "→" in span_s:
                            a, b = [x.strip() for x in span_s.split("→", 1)]
                            span_days = (
                                pd.Timestamp(b) - pd.Timestamp(a)
                            ).total_seconds() / 86400.0
                    except Exception:
                        span_days = float(WEEK_DAYS)
                    _wr = cweek.get("ret")
                    row["week_ret"] = float(_wr) if _wr is not None else 0.0
                    row["prior_ret"] = None  # filled on soft dual-week reload
                    row["week_trades"] = int(cweek.get("trades") or 0)
                    _ww = cweek.get("win")
                    row["week_win"] = float(_ww) if _ww is not None else None
                    row["test_ret"] = float(co.total_return)
                    row["span_days"] = span_days
                    if not _holdout_span_ok(span_days):
                        row["skip"] = True
                        row["why"] = "truncated_holdout"
                        hard_rows.append(row)
                        continue
                    # dual_ok deferred to soft period_robust (hard has no prior yet).
                    dense_ok = _lev_density_ok(
                        int(cand.leverage),
                        cweek,
                        dual_ok=True,
                        tf=str(cand.timeframe),
                    )
                    row["dense_ok"] = bool(dense_ok)
                    # Serializable only — soft pass reloads top-N (keeps RAM flat).
                    hard_rows.append(row)
                except Exception as e:
                    row["skip"] = True
                    row["why"] = f"eval_fail:{e}"
                    hard_rows.append(row)
                    log.warning("matrix cell %sx@%s failed: %s", cand.leverage, cand.timeframe, e)
            _prune_tf(tf_s_m)
            _pass_heartbeat()
            log.info("matrix hard done · tf=%s · rows=%d", tf_s_m, len(cells_by_tf.get(tf_s_m, [])))
        _timing["matrix_hard_s"] = time.monotonic() - _t_hard

        # Soft+robust top-N with TF diversity (raw top-5 is often one TF family).
        _t_soft = time.monotonic()
        ranked_hard = _matrix_soft_pick(hard_rows, cap=int(HUNT_MATRIX_SOFT_CAP))
        soft_keys = {(r["tf"], r["leverage"]) for r in ranked_hard}
        for row in hard_rows:
            key = (row["tf"], row["leverage"])
            if row.get("skip") or key not in soft_keys:
                if not row.get("skip") and not row.get("dense_ok"):
                    row["why"] = row.get("why") or "sparse"
                tf_lev_matrix_rows.append(row)
                continue
            try:
                cand = cell_by_key[key]
                use_raw = _raw_for_tf(cand.timeframe)
                data = _prepare_cached(use_raw, cand, feat_cache)
                ci, co = _eval_cfg(data, cand, split, end=fit_end)
                cweek = _dual_week_eval(data, cand, end=week_end)
                cs = _eval_soft_oos(data, cand, split, end=fit_end)
                soft_ret = float(cs.total_return) if cs is not None else float(co.total_return)
                row["soft_ret"] = soft_ret
                row["prior_ret"] = cweek.get("prior_ret")
                _wr = cweek.get("ret")
                if _wr is not None:
                    row["week_ret"] = float(_wr)
                row["week_trades"] = int(cweek.get("trades") or 0)
                _ww = cweek.get("win")
                row["week_win"] = float(_ww) if _ww is not None else row.get("week_win")
                row["test_ret"] = float(co.total_return)
                rok, rwhy = _period_robust_ok(cweek, float(co.total_return), soft_ret)
                row["robust_ok"] = bool(rok)
                row["why"] = rwhy if not rok else "period_robust"
                if rok and _is_screenable(ci, co) and _is_climb_ok(ci, co, cs, base_i, base_o):
                    _consider(cand, ci, co, cs, "tf_lev_matrix")
                if rok:
                    _best_by_tf_raise(
                        state,
                        tf=str(cand.timeframe),
                        knobs=_cfg_to_knobs(cand),
                        week_ret=float(cweek.get("ret") or 0),
                        oos_ret=float(co.total_return),
                        generation=generation,
                        ts=_utc_now().isoformat(),
                        prior_ret=float(cweek.get("prior_ret") or 0),
                    )
            except Exception as e:
                row["robust_ok"] = False
                row["why"] = f"soft_fail:{e}"
            finally:
                _prune_tf(str(row.get("tf") or ""))
            tf_lev_matrix_rows.append(row)
        _timing["matrix_soft_s"] = time.monotonic() - _t_soft

        matrix_winner_row = _rank_matrix_rows(tf_lev_matrix_rows)
        matrix_escape = False
        if not matrix_winner_row:
            matrix_winner_row = _rank_matrix_rows_escape(
                tf_lev_matrix_rows,
                base_week_ret=float(base_week.get("ret") or 0),
            )
            matrix_escape = bool(matrix_winner_row)
        if matrix_winner_row:
            wkey = (
                str(matrix_winner_row["tf"]),
                int(matrix_winner_row["leverage"]),
            )
            matrix_winner_cfg = cell_by_key.get(wkey)
            log.info(
                "tf×lev matrix %s %sx@%s week=%+.0f%% prior=%s oos=%+.0f%% soft=%s",
                "ESCAPE" if matrix_escape else "winner",
                matrix_winner_row.get("leverage"),
                matrix_winner_row.get("tf"),
                float(matrix_winner_row.get("week_ret") or 0) * 100,
                matrix_winner_row.get("prior_ret"),
                float(matrix_winner_row.get("test_ret") or 0) * 100,
                matrix_winner_row.get("soft_ret"),
            )
            # Defer locking best_cfg until AFTER champion rescue — otherwise a
            # weak +1% cell steals the slot and blocks deploying a +27% champ.
            if matrix_escape and matrix_winner_cfg is not None:
                promote_reason = "matrix_escape"
                log.info(
                    "matrix ESCAPE candidate · %s %sx@%s week=%+.0f%% (deferred until after champ)",
                    matrix_winner_cfg.strategy,
                    int(matrix_winner_cfg.leverage),
                    matrix_winner_cfg.timeframe,
                    float(matrix_winner_row.get("week_ret") or 0) * 100,
                )
        else:
            log.info(
                "tf×lev matrix: no robust/escape cell (%d rows, base week %+.0f%%)",
                len(tf_lev_matrix_rows),
                float(base_week.get("ret") or 0) * 100,
            )
        log.info(
            "tf×lev matrix cells: %s",
            " · ".join(
                f"{r['leverage']}x@{r['tf']} w={float(r.get('week_ret') or 0):+.0%}"
                f"{'/R' if r.get('robust_ok') else '/—'}"
                for r in sorted(
                    tf_lev_matrix_rows,
                    key=lambda x: (str(x.get("tf")), int(x.get("leverage") or 0)),
                )
            )[:500],
        )
        # Drop non-Live TF OHLCV / features before Stage A — full matrix holds
        # 5m+15m+1h+4h in RAM and was OOM-killing the 1.7–2.8G cgroup mid-search.
        keep_tfs = {str(base.timeframe)}
        if matrix_winner_cfg is not None:
            keep_tfs.add(str(matrix_winner_cfg.timeframe))
        for store in (raw_by_tf, scout_raw_by_tf):
            for k in list(store):
                if str(k) not in keep_tfs:
                    store.pop(k, None)
        _stats = feat_cache.get("_stats")
        feat_cache.clear()
        if _stats is not None:
            feat_cache["_stats"] = _stats
        gc.collect()
        log.info("tf cache pruned · keep=%s", sorted(keep_tfs))
    except Exception as e:
        matrix_failed = True
        log.critical("MATRIX FAILED · matrix_failed=1 · %s", e)
        tf_lev_matrix_rows = []
        matrix_winner_cfg = None
        matrix_winner_row = None
        matrix_escape = False

    # Polish base precedence: matrix winner > revalidated best_by_tf > champion > live
    champ_polish = None
    if has_champion:
        try:
            champ_polish = _knobs_to_cfg(champion_preview["knobs"])
            champ_polish = replace(champ_polish, starting_equity=research_eq)
        except Exception as e:
            log.warning("champion polish base failed: %s", e)
    by_tf_polish = None
    mem_tf = (state.get("best_by_tf") or {}).get(str(base.timeframe))
    if isinstance(mem_tf, dict) and _best_by_tf_seed_ok(mem_tf):
        try:
            by_tf_polish = replace(
                _knobs_to_cfg_matrix(mem_tf["knobs"]),
                starting_equity=research_eq,
            )
        except Exception as e:
            log.warning("best_by_tf seed failed: %s", e)
    if matrix_winner_cfg is not None:
        matrix_winner_cfg = replace(matrix_winner_cfg, starting_equity=research_eq)
    polish_base = _select_polish_base(
        live_base=best_cfg,
        matrix_winner_cfg=matrix_winner_cfg,
        best_by_tf_cfg=by_tf_polish,
        champion_cfg=champ_polish,
    )

    tested = 0
    soft_runs = 0
    soft_budget = int(HUNT_SOFT_BUDGET.get(search_mode, 6))
    scout_top = int(HUNT_SCOUT_TOP)
    if held >= int(HUNT_STUCK_HELD):
        # Fat soft_budget≥20 was churn: escape rate lower than thinner soft runs.
        soft_budget = max(soft_budget, int(HUNT_STUCK_SOFT_BUDGET))
        # Do not bump scout_top to 24 — keep Stage B narrower when stuck.
    keep_a = max(int(HUNT_HALVING_A_KEEP), scout_top)
    keep_b = max(int(HUNT_HALVING_B_KEEP), min(soft_budget, scout_top))

    banned = _banlist_set(state)
    cand_list = _candidate_cfgs(
        polish_base,
        generation=generation,
        hunt=hunt,
        search_mode=search_mode,
        allow_lev_tf=True,
    )
    cand_list = list(cand_list) + _discover_seed_cfgs(
        polish_base, generation, held, hunt_exhausted=hunt_exhausted
    )

    # Inject revalidated best_by_tf seeds (other TFs)
    for _tf, blob in (state.get("best_by_tf") or {}).items():
        if not isinstance(blob, dict) or not _best_by_tf_seed_ok(blob):
            continue
        try:
            seed = _knobs_to_cfg_matrix(blob["knobs"])
            seed = replace(seed, starting_equity=research_eq)
            cand_list.append(seed)
        except Exception:
            pass
    # denser when stuck / explore / breakout: Live-TF knob grid ≤ HUNT_DENSER_MAX (27).
    # Exhausted: cross-TF denser up to HUNT_EXHAUSTED_DENSER_MAX.
    # force_skip_denser (oom/stuck recover): skip denser this pass only; matrix stays FULL.
    denser: list[Config] = []
    _pass_heartbeat()
    if skip_denser:
        log.info(
            "denser skipped · reason=%s · matrix stays FULL",
            recover_reason or "force_skip_denser",
        )
    elif (not skip_heavy_search) and (
        hunt_exhausted
        or held >= int(HUNT_STUCK_HELD)
        or search_mode in ("explore", "breakout")
    ):
        _denser_tfs = (
            list(HUNT_MATRIX_TFS) if hunt_exhausted else [str(polish_base.timeframe)]
        )
        _denser_cap = (
            int(HUNT_EXHAUSTED_DENSER_MAX) if hunt_exhausted else int(HUNT_DENSER_MAX)
        )
        for _dtf in _denser_tfs:
            for st in HUNT_STRATEGIES:
                for lev in (5, 7, 10):
                    for adx in (14.0, 16.0, 18.0):
                        for cd in HUNT_DENSER_COOLDOWNS:
                            denser.append(
                                replace(
                                    polish_base,
                                    strategy=str(st),
                                    leverage=int(lev),
                                    timeframe=str(_dtf),
                                    adx_enter=float(adx),
                                    adx_exit=float(max(10.0, adx - 5)),
                                    require_squeeze=False,
                                    require_rising_adx=False,
                                    cooldown_bars=int(cd),
                                    volume_multiplier=1.1,
                                    mom_lookback=8,
                                    mom_min_atr=0.35,
                                    max_account_leverage=float(
                                        max(lev, float(polish_base.max_account_leverage))
                                    ),
                                    max_net_leverage=float(
                                        max(lev, float(polish_base.max_net_leverage))
                                    ),
                                )
                            )
        if len(denser) > int(_denser_cap):
            denser = denser[: int(_denser_cap)]
        log.info(
            "denser inject · tfs=%s · exhausted=%s · n=%d",
            ",".join(str(x) for x in _denser_tfs),
            hunt_exhausted,
            len(denser),
        )
    # Dedupe with polish_base as freeze anchor so matrix lev/tf survive
    cand_list = _dedupe_cfgs_matrix(polish_base, list(cand_list) + denser)
    cand_list, banned_skipped = _filter_banned(cand_list, banned)
    log.info(
        "hunt search · mode=%s · candidates=%d · banned_skipped=%d · polish_from=%s",
        search_mode,
        len(cand_list),
        banned_skipped,
        "champion" if polish_base is not best_cfg else "live",
    )

    # Curriculum book for Stage C soft-check (BTC+ETH ∩ confirm symbols)
    curriculum = [s for s in HUNT_CURRICULUM_SYMBOLS if s in symbols] or list(symbols[:2])

    def _curriculum_raw_for_tf(tf: str) -> dict:
        full = _raw_for_tf(tf)
        return {s: full[s] for s in curriculum if s in full}

    # ---- Stage A: cheap BTC 2-fold smoke Sortino (week = diagnostic only) ----
    stage_a: list = []
    thin_a: list = []
    funnel_counts = {
        "A_in": 0,
        "A_keep": 0,
        "B": 0,
        "C": 0,
        "cost_pass": 0,
        "fold_pass": 0,
        "promote_attempt": 0,
        "promote": 0,
    }
    _pass_heartbeat()
    _t_a = time.monotonic()
    _stage_a_hits_before = int((feat_cache.get("_stats") or {}).get("hits") or 0)
    _stage_a_prep_before = int((feat_cache.get("_stats") or {}).get("prepare_calls") or 0)
    # Live TF only in Stage A on the 2G droplet — explore TF jumps were
    # re-fetching 15m/1h/4h books and OOMing before HOLD. Other TFs are
    # already scored in the sequential matrix; B/C may still see survivors.
    _live_tf_a = str(base.timeframe)
    _stage_a_cands = (
        [] if skip_heavy_search else [c for c in cand_list if str(c.timeframe) == _live_tf_a]
    )
    funnel_counts["A_in"] = len(_stage_a_cands)
    if skip_heavy_search:
        log.info("champion short-circuit — skipping Stage A/B/C candidate search")
    elif len(_stage_a_cands) < len(cand_list):
        log.info(
            "stage A Live-TF filter · %s · %d/%d candidates",
            _live_tf_a,
            len(_stage_a_cands),
            len(cand_list),
        )
    for _ai, cand in enumerate(_stage_a_cands):
        tested += 1
        fold_a: dict = {}
        cheap: dict = {}
        try:
            use_raw = _scout_raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            fold_a = eval_cfg_folds(data, cand, end=week_end, smoke=True)
            # Week diagnostic only (not sort key).
            cheap = _window_eval(
                data,
                cand,
                int(HUNT_HALVING_A_DAYS),
                end=week_end,
                soft=False,
                min_trades=max(2, int(HUNT_TARGETS.get("week_min_trades", 10)) // 2),
                exp_floor=-0.75,
            )
        except Exception as e:
            log.warning("stage A failed: %s", e)
            continue
        finally:
            # Flush feat_cache every 5 — denser lev variants share feature keys
            # (lev/strategy not in key); every-cand flush destroyed reuse.
            _ctf = str(getattr(cand, "timeframe", "") or "")
            if _ctf and _ctf != str(base.timeframe):
                raw_by_tf.pop(_ctf, None)
                scout_raw_by_tf.pop(_ctf, None)
            if (_ai + 1) % 5 == 0:
                _stats_a = feat_cache.pop("_stats", None)
                feat_cache.clear()
                if _stats_a is not None:
                    feat_cache["_stats"] = _stats_a
                _pass_heartbeat()
                gc.collect()
        try:
            p25 = float(fold_a.get("p25_sortino"))
        except (TypeError, ValueError):
            p25 = float("nan")
        if p25 != p25:
            p25 = -999.0
        n_tr = int(fold_a.get("trades") or cheap.get("trades") or 0)
        roi_a = float(fold_a.get("roi") or cheap.get("ret") or -999)
        row = (
            p25,
            n_tr,
            roi_a,
            cand,
            None,
            None,
            cheap,
            fold_a,
        )
        # Prefer books with some trades or non-disastrous fold; keep thin as fallback.
        if n_tr >= 2 or p25 > -0.5 or float(cheap.get("ret") or 0) > -0.05:
            stage_a.append(row)
        else:
            thin_a.append(row)
        ban_w = {"ret": cheap.get("ret")}
        do_ban, why_ban = _should_ban_candidate(week=ban_w, fold=fold_a)
        if do_ban:
            _banlist_add(state, cand, week_ret=float(cheap.get("ret") or 0), reason=f"A:{why_ban}")
    if not stage_a and thin_a:
        log.warning("stage A: no screenable books — using thin pool")
        stage_a = thin_a
    stage_a.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    short_a = stage_a[:keep_a]
    funnel_counts["A_keep"] = len(short_a)
    _timing["stage_a_s"] = time.monotonic() - _t_a
    _sa_hits = int((feat_cache.get("_stats") or {}).get("hits") or 0) - _stage_a_hits_before
    _sa_prep = (
        int((feat_cache.get("_stats") or {}).get("prepare_calls") or 0) - _stage_a_prep_before
    )
    _best_p25 = float(short_a[0][0]) if short_a else float("nan")
    log.info(
        "halving A: tried %d · keep %d/%d (best p25_sortino %+0.3f) · "
        "feat_cache hits=%d prepare=%d",
        tested,
        len(short_a),
        keep_a,
        _best_p25 if _best_p25 == _best_p25 else 0.0,
        max(0, _sa_hits),
        max(0, _sa_prep),
    )

    # ---- Stage B: full folds + cost on BTC scout (week diagnostic) ----
    stage_b: list = []
    _pass_heartbeat()
    _t_b = time.monotonic()
    for _p25_a, _ntr_a, _roi_a, cand, _ci, _co, _cheap, _fold_a in short_a:
        fold_b: dict = {}
        cost_ok_b = True
        try:
            use_raw = _scout_raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            ci, co = _eval_cfg(data, cand, split, end=fit_end)
            cweek_s = _week_eval(data, cand, end=week_end)
            fold_b = eval_cfg_folds(data, cand, end=week_end, smoke=False)
            cost_ok_b, _, _ = cost_ok_for_cfg(data, cand, end=week_end)
        except Exception as e:
            log.warning("stage B failed: %s", e)
            continue
        do_ban, why_ban = _should_ban_candidate(
            week=cweek_s, fold=fold_b, cost_ok=cost_ok_b
        )
        if do_ban:
            _banlist_add(
                state, cand, week_ret=float(cweek_s.get("ret") or 0), reason=f"B:{why_ban}"
            )
            continue
        if cost_ok_b:
            funnel_counts["cost_pass"] += 1
        if fold_b.get("ok"):
            funnel_counts["fold_pass"] += 1
        rk = _rank_key(
            ci,
            co,
            None,
            cweek_s,
            fold=fold_b,
            cost_ok=bool(cost_ok_b),
            roi=float(fold_b.get("roi") or cweek_s.get("ret") or 0),
        )
        stage_b.append(
            (
                rk,
                float(fold_b.get("p25_sortino") or -999),
                int(fold_b.get("trades") or cweek_s.get("trades") or 0),
                cand,
                ci,
                co,
                cweek_s,
                fold_b,
                bool(cost_ok_b),
            )
        )
    stage_b.sort(key=lambda row: row[0], reverse=True)
    shortlist = stage_b[:keep_b]
    funnel_counts["B"] = len(shortlist)
    _timing["stage_b_s"] = time.monotonic() - _t_b
    top_p25 = float(shortlist[0][1]) if shortlist else float("nan")
    log.info(
        "halving B: %d → keep %d (best p25_sortino %+0.3f)",
        len(stage_b),
        len(shortlist),
        top_p25 if top_p25 == top_p25 else 0.0,
    )

    # ---- Stage C: curriculum soft-check (parallel) + dual-week ----
    leverage_ladder: list = []
    confirm_hits = 0
    confirm_rejects = {"climb": 0, "consider": 0, "err": 0, "dual": 0}
    rotate_pool: list = []
    _pass_heartbeat()
    _t_c = time.monotonic()
    # Stage C workers=1 by default on 2G (see _hunt_soft_workers / MOONSHOT_SOFT_WORKERS).
    workers = _hunt_soft_workers()

    _feat_lock = threading.Lock()

    def _stage_c_one(cand: Config):
        use_raw = _curriculum_raw_for_tf(cand.timeframe)
        if not use_raw:
            use_raw = _raw_for_tf(cand.timeframe)
        if not use_raw:
            raise RuntimeError("empty curriculum/confirm raw")
        with _feat_lock:
            data = _prepare_cached(use_raw, cand, feat_cache)
        ci, co = _eval_cfg(data, cand, split, end=fit_end)
        cs = _eval_soft_oos(data, cand, split, end=fit_end)
        cweek_r = _dual_week_eval(data, cand, end=week_end)
        fold_c = eval_cfg_folds(data, cand, end=week_end, smoke=False)
        cost_ok_c, _, _ = cost_ok_for_cfg(data, cand, end=week_end)
        return cand, ci, co, cs, cweek_r, fold_c, bool(cost_ok_c)

    to_soft = [row[3] for row in shortlist[:soft_budget]]
    results_c: list = []
    if workers <= 1 or len(to_soft) <= 1:
        for cand in to_soft:
            try:
                results_c.append(_stage_c_one(cand))
                soft_runs += 1
            except Exception as e:
                log.warning("confirm soft failed: %s", e)
                confirm_rejects["err"] += 1
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_stage_c_one, c): c for c in to_soft}
            for fut in as_completed(futs):
                try:
                    results_c.append(fut.result())
                    soft_runs += 1
                except Exception as e:
                    log.warning("confirm soft failed: %s", e)
                    confirm_rejects["err"] += 1

    for cand, ci, co, cs, cweek_r, fold_c, cost_ok_c in results_c:
        do_ban, why_ban = _should_ban_candidate(
            week=cweek_r,
            fold=fold_c,
            cost_ok=cost_ok_c,
            kill_illusion=False,
        )
        if do_ban:
            _banlist_add(
                state, cand, week_ret=float(cweek_r.get("ret") or 0), reason=f"C:{why_ban}"
            )
        if cost_ok_c:
            funnel_counts["cost_pass"] += 1
        if fold_c.get("ok"):
            funnel_counts["fold_pass"] += 1
        if _is_screenable(ci, co):
            rotate_pool.append(
                (
                    _rank_key(
                        ci,
                        co,
                        cs,
                        cweek_r,
                        fold=fold_c,
                        cost_ok=cost_ok_c,
                        roi=float(fold_c.get("roi") or cweek_r.get("ret") or 0),
                    ),
                    float(fold_c.get("p25_sortino") or -999),
                    int(fold_c.get("trades") or cweek_r.get("trades") or 0),
                    cand,
                    ci,
                    co,
                    cs,
                    cweek_r,
                    fold_c,
                    cost_ok_c,
                )
            )
        primary_ret = float(cweek_r.get("ret") or 0)
        week_tgt = float(HUNT_TARGETS.get("week_return", 1.0))
        hard_floor = float(HUNT_TARGETS.get("week_hard_floor", 0.25))
        if not _dual_week_ok(cweek_r) and primary_ret < week_tgt and primary_ret < hard_floor:
            confirm_rejects["dual"] += 1
            continue
        if not _is_climb_ok(ci, co, cs, base_i, base_o):
            confirm_rejects["climb"] += 1
            continue
        if _consider(cand, ci, co, cs, "confirm", fold=fold_c, cost_ok=cost_ok_c):
            confirm_hits += 1
        else:
            confirm_rejects["consider"] += 1

    funnel_counts["C"] = int(confirm_hits)
    _pass_heartbeat()
    _timing["stage_c_s"] = time.monotonic() - _t_c
    stats = feat_cache.get("_stats") or {}
    log.info(
        "halving C: soft-checked %d/%d workers=%d · climbs=%d · rejects %s · "
        "cache hits=%s prepare=%s · curriculum=%s · primary=p25_fold_sortino",
        soft_runs,
        soft_budget,
        workers,
        confirm_hits,
        confirm_rejects,
        stats.get("hits"),
        stats.get("prepare_calls"),
        ",".join(s.split("/")[0] for s in curriculum),
    )

    # ---- Leverage bake-off: same Live strategy + TF, sweep leverage only ----
    # (Not best-holdout-TF-per-lev — that mixed candle sizes and lied about lev.)
    ladder_rows = [
        {
            "leverage": int(r.get("leverage") or 0),
            "tf": str(r.get("tf") or ""),
            "timeframe": str(r.get("tf") or ""),
            "strategy": str(r.get("strategy") or ""),
            "week_ret": float(r.get("week_ret") or 0),
            "week_trades": int(r.get("week_trades") or 0),
            "week_win": r.get("week_win"),
            "prior_ret": r.get("prior_ret"),
            "soft_ret": r.get("soft_ret"),
            "test_ret": r.get("test_ret"),
            "robust_ok": bool(r.get("robust_ok")),
            "dense_ok": bool(r.get("dense_ok")),
            "skip": bool(r.get("skip")),
            "why": r.get("why"),
        }
        for r in (tf_lev_matrix_rows or [])
    ]
    leverage_ladder = _book_leverage_bakeoff(
        ladder_rows,
        strategy=str(base.strategy),
        timeframe=str(base.timeframe),
    )
    if leverage_ladder:
        log.info(
            "leverage bake-off · %s@%s: %s",
            base.strategy,
            base.timeframe,
            " · ".join(
                f"{r['leverage']}x week={float(r.get('week_ret') or 0):+.0%}"
                f" oos={r.get('test_ret')}"
                for r in leverage_ladder
            ),
        )

    week_min = int(HUNT_TARGETS.get("week_min_trades", 4))
    hard_floor = float(HUNT_TARGETS.get("week_hard_floor", 0.25))

    # Death-book predicate — escape paths may arm only when Live is dying.
    bleed_info = _paper_bleed()
    paper_bleeding = bool(PAPER_BLEED_FORCE_ROTATE and bleed_info.get("bleed"))
    if paper_bleeding:
        log.warning(
            "paper BLEED · last %d closes all losers · pnl=%+.2f — forcing rotate path",
            int(bleed_info.get("n") or 0),
            float(bleed_info.get("pnl") or 0),
        )
    _base_soft_for_death = (
        float(base_soft.total_return) if base_soft is not None else float(base_o.total_return)
    )
    live_is_death, live_death_why = live_death_book(
        week_ret=float(base_week.get("ret") or 0),
        oos_ret=float(base_o.total_return),
        soft_ret=_base_soft_for_death,
        paper_bleeding=paper_bleeding,
    )
    if not live_is_death:
        log.info("live not death · escape armed blocked · %s", live_death_why)

    # ---- Policy-3 champion rescue BEFORE silence (prefer peak over rotate junk) ----
    champion_rescue = False
    champ_week_live = None
    # Keep silence_rotate if matrix_escape already locked a better TF×lev earlier.
    idle_h = float(_paper_idle_hours())
    _deploy_ts_candidates: list[str] = []
    _man = state.get("manual_champion_deploy") or {}
    if isinstance(_man, dict) and _man.get("ts"):
        _deploy_ts_candidates.append(str(_man["ts"]))
    if state.get("research_promote_at"):
        _deploy_ts_candidates.append(str(state["research_promote_at"]))
    for _dts in _deploy_ts_candidates:
        try:
            _since = (_utc_now() - pd.Timestamp(_dts)).total_seconds() / 3600.0
            idle_h = min(idle_h, max(0.0, float(_since)))
        except Exception:
            pass

    if champion and champion.get("knobs"):
        try:
            ch_cfg = _knobs_to_cfg(champion["knobs"])
            ch_raw = _raw_for_tf(str(ch_cfg.timeframe)) or raw
            ch_data = _prepare_cached(ch_raw, ch_cfg, feat_cache)
            ch_week = _dual_week_eval(ch_data, ch_cfg, end=week_end)
            champ_week_live = float(ch_week.get("ret") or 0)
            live_week_now = float(base_week.get("ret") or 0)
            trail_pp = champ_week_live - live_week_now
            if trail_pp >= float(HUNT_CHAMP_TRAIL_PP):
                state["champ_trail_streak"] = int(state.get("champ_trail_streak") or 0) + 1
            else:
                state["champ_trail_streak"] = 0
            pending_drift = bool(state.get("pending_champion_rescue"))
            trail_ready = int(state.get("champ_trail_streak") or 0) >= int(HUNT_CHAMP_TRAIL_PASSES)
            # A.2: rescue = restore peak only (pending drift / trail), never idle swap.
            # When Live is deep red / far behind champ, accept champ ≥15% (not only
            # hard_floor 25%) — 24% champ vs 1% Live was rejected forever on a 1pp miss.
            rescue_floor = float(hard_floor)
            if live_week_now < 0.05 or trail_pp >= float(HUNT_CHAMP_TRAIL_PP):
                rescue_floor = min(rescue_floor, 0.15)
            dual_ok = _dual_week_ok(ch_week) or champ_week_live >= rescue_floor
            ch_wf = _wf_for(ch_cfg) if wf_enabled else {"ok": True}
            wf_ok_rescue = (not wf_enabled) or bool(ch_wf and ch_wf.get("ok"))
            # Far behind a still-strong champ: don't wait for trail streak / pending.
            urgent_rescue = (
                trail_pp >= float(HUNT_CHAMP_TRAIL_PP)
                and champ_week_live >= rescue_floor
                and live_week_now < 0.10
            )
            # Matrix escape may already have stolen best_cfg to a weak +1% cell.
            # Champ must still win when it beats that cell — requiring
            # `best_cfg is base` was the core "champion never deploys" bug.
            best_week_now = float(best_week.get("ret") or 0)
            champ_beats_best = champ_week_live >= best_week_now + 0.05
            slot_ok = best_cfg is base or champ_beats_best
            if (
                slot_ok
                and (pending_drift or trail_ready or urgent_rescue)
                and dual_ok
                and (wf_ok_rescue or urgent_rescue)
                and champ_week_live >= rescue_floor
                and _cfg_fingerprint(ch_cfg) != _cfg_fingerprint(base)
            ):
                ci, co = _eval_cfg(ch_data, ch_cfg, split, end=fit_end)
                cs = _eval_soft_oos(ch_data, ch_cfg, split, end=fit_end)
                _ch_soft = float(cs.total_return) if cs is not None else float(co.total_return)
                _base_soft_r = (
                    float(base_soft.total_return)
                    if base_soft is not None
                    else float(base_o.total_return)
                )
                _rok, _rwhy = _period_robust_ok(ch_week, float(co.total_return), _ch_soft)
                _eok, _ewhy = _escape_window_ok(
                    ch_week,
                    float(co.total_return),
                    _ch_soft,
                    base_week,
                    float(base_o.total_return),
                    _base_soft_r,
                )
                # Non-death Live: rescue only if champion is period-robust (no escape).
                _rescue_ok = _rok if not live_is_death else (_rok or _eok)
                if not _rescue_ok:
                    log.info(
                        "champion rescue blocked · death=%s robust=%s escape=%s",
                        live_is_death,
                        _rwhy,
                        _ewhy,
                    )
                else:
                    best_cfg, best_i, best_o, best_soft, best_wf = ch_cfg, ci, co, cs, ch_wf
                    best_week = ch_week
                    best_recent = _recent_eval(ch_data, ch_cfg, end=week_end)
                    best_grade = _grade(
                        ci,
                        co,
                        cs,
                        ch_wf,
                        wf_required=not urgent_rescue,
                        fwd=fwd,
                        fwd_required=True,
                        week=ch_week,
                    )
                    best_grade["recent"] = best_recent
                    best_grade["week"] = ch_week
                    best_grade["week_prior_ret"] = ch_week.get("prior_ret")
                    promote_reason = "champion_rescue"
                    champion_rescue = True
                    silence_rotate = False  # champ overrides matrix_escape
                    state["pending_champion_rescue"] = False
                    state["champ_trail_streak"] = 0
                    log.info(
                        "CHAMPION RESCUE · trail=%+.0fpp · pending_drift=%s · "
                        "deploying champ week=%+.0f%% (live %+0.f%% best_was %+0.f%%) · %s",
                        trail_pp * 100,
                        pending_drift,
                        champ_week_live * 100,
                        live_week_now * 100,
                        best_week_now * 100,
                        "robust" if _rok else _ewhy,
                    )
            else:
                # Already on the champion book — pending_rescue must not stick
                # forever (it forces skip_heavy and blocks real hunt).
                if pending_drift and _cfg_fingerprint(ch_cfg) == _cfg_fingerprint(base):
                    state["pending_champion_rescue"] = False
                    state["champ_trail_streak"] = 0
                    _save_autopilot_state(state)
                    log.info(
                        "champion rescue cleared — already on champ fingerprint "
                        "(live_week=%+.0f%% champ=%+.0f%%)",
                        live_week_now * 100,
                        champ_week_live * 100,
                    )
                log.info(
                    "champion trail · live_week=%+.0f%% champ=%+.0f%% trail=%+.0fpp "
                    "streak=%d pending_drift=%s",
                    live_week_now * 100,
                    champ_week_live * 100,
                    trail_pp * 100,
                    int(state.get("champ_trail_streak") or 0),
                    pending_drift,
                )
        except Exception as e:
            log.warning("champion rescue eval failed: %s", e)

    # ---- Deferred matrix escape (only if champion did not win) ----
    if (
        (not champion_rescue)
        and matrix_escape
        and matrix_winner_cfg is not None
        and best_cfg is base
        and live_is_death
    ):
        try:
            esc = replace(matrix_winner_cfg, starting_equity=research_eq)
            esc_raw = _raw_for_tf(str(esc.timeframe)) or raw
            esc_data = _prepare_cached(esc_raw, esc, feat_cache)
            ei, eo = _eval_cfg(esc_data, esc, split, end=fit_end)
            es = _eval_soft_oos(esc_data, esc, split, end=fit_end)
            eweek = _dual_week_eval(esc_data, esc, end=week_end)
            esc_week = float(eweek.get("ret") or 0)
            base_w = float(base_week.get("ret") or 0)
            # Escape never lev-ups — same/lower lev TF escape only (+2pp).
            lev_up = int(esc.leverage) > int(base.leverage)
            need_pp = float(RESEARCH_HYSTERESIS_WEEK_PP)
            clear_margin = esc_week >= base_w + need_pp
            champ_gap_ok = (
                champ_week_live is None
                or esc_week + 0.10 >= float(champ_week_live)
                or float(champ_week_live) < 0.10
            )
            dense_ok = (
                bool((matrix_winner_row or {}).get("dense_ok"))
                or int(eweek.get("soft_trades") or 0) >= 2
                or int(eweek.get("trades") or 0) >= 2
            )
            _esc_soft = float(es.total_return) if es is not None else float(eo.total_return)
            _base_soft_r = (
                float(base_soft.total_return)
                if base_soft is not None
                else float(base_o.total_return)
            )
            _eok, _ewhy = _escape_window_ok(
                eweek,
                float(eo.total_return),
                _esc_soft,
                base_week,
                float(base_o.total_return),
                _base_soft_r,
            )
            if lev_up:
                log.info(
                    "matrix escape skipped · lev-up forbidden on escape (%sx→%sx)",
                    int(base.leverage),
                    int(esc.leverage),
                )
            elif not dense_ok:
                log.info("matrix escape skipped · not dense enough")
            elif not _eok:
                log.info("matrix escape skipped · %s", _ewhy)
            elif clear_margin and champ_gap_ok:
                best_cfg, best_i, best_o, best_soft = esc, ei, eo, es
                best_week = eweek
                best_recent = _recent_eval(esc_data, esc, end=week_end)
                best_grade = _grade(
                    ei, eo, es, None, wf_required=False, fwd=fwd, fwd_required=True, week=eweek
                )
                best_grade["recent"] = best_recent
                best_grade["week"] = eweek
                promote_reason = "matrix_escape"
                silence_rotate = True
                log.warning(
                    "matrix ESCAPE locked · %s %sx@%s week=%+.0f%% (base was %+.0f%%) · %s",
                    esc.strategy,
                    int(esc.leverage),
                    esc.timeframe,
                    esc_week * 100,
                    base_w * 100,
                    _ewhy,
                )
            else:
                log.info(
                    "matrix escape skipped · week=%+.0f%% base=%+.0f%% champ=%s "
                    "(need ≥%+.0fpp vs Live; not far below champ)",
                    esc_week * 100,
                    base_w * 100,
                    f"{float(champ_week_live):+.0%}" if champ_week_live is not None else "n/a",
                    need_pp * 100,
                )
        except Exception as e:
            log.warning("matrix escape apply failed: %s", e)

    # ---- Silence / bleed rotate: escape a frozen losing book ----
    # Do NOT require held>=3 — a false "improved" history row resets the streak
    # and would permanently disable bleed escape while paper keeps losing.
    if (
        bool(HUNT_SILENCE_ROTATE_ENABLED)
        and (not champion_rescue)
        and (not silence_rotate)  # matrix escape already locked a book
        and live_is_death  # never polish-deploy on a non-death Live book
        and (held >= 3 or paper_bleeding)
        and (
            idle_h >= float(PAPER_SILENCE_ROTATE_HOURS)
            or paper_bleeding
            or held >= max(int(HUNT_STUCK_HELD) * 8, 48)
        )
    ):
        base_fp = _cfg_fingerprint(base)
        peak_fp = None
        peak_cfg_live = None
        if champion and champion.get("knobs"):
            try:
                peak_cfg_live = _knobs_to_cfg(champion["knobs"])
                peak_fp = _cfg_fingerprint(peak_cfg_live)
            except Exception:
                peak_cfg_live = None
                peak_fp = None

        # If Live IS the peak book, densify peak filters before swapping families.
        if peak_cfg_live is not None and peak_fp == base_fp:
            for cand in _density_polish_cfgs(peak_cfg_live):
                try:
                    # Matrix freeze: silence polish may change TF/lev; plain
                    # _freeze_risk would pin lev to the bleeding base book.
                    cand = _freeze_risk_matrix(base, cand)
                    if _cfg_fingerprint(cand) == base_fp:
                        continue
                    use_raw = _raw_for_tf(str(cand.timeframe)) or raw
                    if not use_raw:
                        continue
                    data = _prepare_cached(use_raw, cand, feat_cache)
                    ci, co = _eval_cfg(data, cand, split, end=fit_end)
                    cs = _eval_soft_oos(data, cand, split, end=fit_end)
                    cweek = _dual_week_eval(data, cand, end=week_end)
                    if not _is_screenable(ci, co):
                        continue
                    if not _is_climb_ok(ci, co, cs, base_i, base_o):
                        continue
                    wn = int(cweek.get("trades") or 0)
                    if wn < week_min and int(cweek.get("soft_trades") or 0) < week_min:
                        continue
                    if int(cand.leverage) > int(base.leverage):
                        continue  # escape never lev-ups
                    _cand_soft = (
                        float(cs.total_return) if cs is not None else float(co.total_return)
                    )
                    _base_soft_r = (
                        float(base_soft.total_return)
                        if base_soft is not None
                        else float(base_o.total_return)
                    )
                    _eok, _ewhy = _escape_window_ok(
                        cweek,
                        float(co.total_return),
                        _cand_soft,
                        base_week,
                        float(base_o.total_return),
                        _base_soft_r,
                    )
                    if not _eok:
                        log.info("silence density skip · %s", _ewhy)
                        continue
                    best_cfg, best_i, best_o, best_soft = cand, ci, co, cs
                    best_week = cweek
                    best_recent = _recent_eval(data, cand, end=week_end)
                    best_grade = _grade(
                        ci, co, cs, None, wf_required=False, fwd=fwd, fwd_required=True, week=cweek
                    )
                    best_grade["recent"] = best_recent
                    best_grade["week"] = cweek
                    promote_reason = "silence_density"
                    silence_rotate = True
                    log.info(
                        "silence density polish · idle %.1fh · %s %sx@%s week=%+.0f%% "
                        "(peak family kept) · %s",
                        idle_h,
                        cand.strategy,
                        int(cand.leverage),
                        cand.timeframe,
                        float(cweek.get("ret") or 0) * 100,
                        _ewhy,
                    )
                    break
                except Exception as e:
                    log.warning("density polish candidate failed: %s", e)

        if (not silence_rotate) and rotate_pool:
            # Fold-primary sort: rank_key, then p25, then trades
            rotate_pool.sort(key=lambda r: (r[0], r[1], r[2]), reverse=True)
            peak_live = float(champ_week_live) if champ_week_live is not None else None
            base_week_now = float(base_week.get("ret") or 0)
            for _rk, _p25, wn, cand, ci, co, cs, cweek, fold_r, cost_ok_r in rotate_pool:
                if _cfg_fingerprint(cand) == base_fp:
                    continue
                wsn = int(cweek.get("soft_trades") or 0)
                if int(wn) < week_min and int(wsn) < week_min:
                    continue
                if not _is_screenable(ci, co):
                    continue
                cret = float(cweek.get("ret") or 0)
                soft_edge = float(cs.total_return) > float(base_soft.total_return) + 0.02
                week_ok = (
                    _dual_week_ok(cweek)
                    or cret >= hard_floor
                    or (paper_bleeding and cret >= base_week_now - 0.05 and soft_edge)
                )
                if not week_ok:
                    continue
                if (
                    peak_live is not None
                    and not paper_bleeding
                    and cret < peak_live - float(HUNT_SILENCE_MAX_PEAK_GAP_PP)
                ):
                    continue
                if int(cand.leverage) > int(base.leverage):
                    continue  # escape never lev-ups
                cwf = _wf_for(cand) if wf_enabled else None
                if wf_enabled and not paper_bleeding and not (cwf and cwf.get("ok")):
                    continue
                _cand_soft = float(cs.total_return) if cs is not None else float(co.total_return)
                _base_soft_r = (
                    float(base_soft.total_return)
                    if base_soft is not None
                    else float(base_o.total_return)
                )
                _eok, _ewhy = _escape_window_ok(
                    cweek,
                    float(co.total_return),
                    _cand_soft,
                    base_week,
                    float(base_o.total_return),
                    _base_soft_r,
                )
                if not _eok:
                    log.info("silence rotate skip · %s", _ewhy)
                    continue
                best_cfg, best_i, best_o, best_soft, best_wf = cand, ci, co, cs, cwf
                best_week = cweek
                best_fold = dict(fold_r) if isinstance(fold_r, dict) else best_fold
                best_cost_ok = bool(cost_ok_r)
                best_roi = float(
                    (fold_r or {}).get("roi") if isinstance(fold_r, dict) else cret
                )
                best_recent = _recent_eval(
                    _prepare_cached(_raw_for_tf(str(cand.timeframe)) or raw, cand, feat_cache),
                    cand,
                    end=week_end,
                )
                best_grade = _grade(
                    ci, co, cs, cwf, wf_required=True, fwd=fwd, fwd_required=True, week=cweek
                )
                best_grade["recent"] = best_recent
                best_grade["week"] = cweek
                best_grade["fold"] = best_fold
                promote_reason = "silence_rotate"
                silence_rotate = True
                log.info(
                    "silence rotate · paper idle %.1fh · Live-only deploy %s %sx@%s "
                    "week=%+.0f%% n=%d (champion_peak untouched) · %s",
                    idle_h,
                    cand.strategy,
                    int(cand.leverage),
                    cand.timeframe,
                    cret * 100,
                    int(wn),
                    _ewhy,
                )
                break
            if not silence_rotate:
                log.info(
                    "silence rotate skipped · idle %.1fh · no candidate cleared "
                    "floor/peak-gap (pool=%d)",
                    idle_h,
                    len(rotate_pool),
                )
        elif not silence_rotate:
            log.info(
                "silence rotate armed · idle %.1fh · held=%d pool=%d",
                idle_h,
                held,
                len(rotate_pool),
            )

    # F4 finalist: re-score holdout week on FULL paper symbol book before research promote
    require_wf = bool(HUNT_TARGETS.get("require_walk_forward", True))
    best_wf_ok = bool(best_wf and best_wf.get("ok"))
    # Normal promote needs WF green; silence_rotate may proceed without it.
    wf_gate_ok = best_wf_ok if (require_wf and wf_enabled) else ((not wf_enabled) or best_wf_ok)
    best_week_active = (
        int(best_week.get("trades") or 0) >= week_min
        or int(best_week.get("soft_trades") or 0) >= week_min
    )
    # Escape hatch: 4h matrix cells often have only 2–3 holdout trades. Requiring
    # week_min (4) was rejecting every matrix_escape while Live stayed on a -9% book.
    # champion_rescue included — peak books can be sparse on the holdout window.
    if (
        not best_week_active
        and (silence_rotate or champion_rescue)
        and best_cfg is not base
        and (
            int(best_week.get("trades") or 0) >= 2
            or int(best_week.get("soft_trades") or 0) >= 2
            or float(best_week.get("ret") or 0) > float(base_week.get("ret") or 0) + 0.02
        )
    ):
        best_week_active = True
        log.info(
            "escape week_min relaxed · trades=%d soft=%d week=%+.0f%% (base %+.0f%%)",
            int(best_week.get("trades") or 0),
            int(best_week.get("soft_trades") or 0),
            float(best_week.get("ret") or 0) * 100,
            float(base_week.get("ret") or 0) * 100,
        )
    # Research promote only on a *material* step — not every ±1% thrash that
    # restarts paper and floods Live activity with setup/upgrade spam.
    material_ok, material_why = _material_research_edge(
        best_cfg=best_cfg,
        base_cfg=base,
        best_grade=best_grade,
        base_grade=base_grade,
        best_week=best_week,
        base_week=base_week,
        best_o=best_o,
        base_o=base_o,
        best_soft=best_soft,
        base_soft=base_soft,
    )
    promote_cooled = True
    last_prom = state.get("research_promote_at")
    if last_prom:
        try:
            age_s = (_utc_now() - pd.Timestamp(last_prom)).total_seconds()
            promote_cooled = age_s >= float(RESEARCH_PROMOTE_MIN_SEC)
            if not promote_cooled:
                material_why = f"promote cooldown {int(RESEARCH_PROMOTE_MIN_SEC - age_s)}s left"
        except Exception:
            promote_cooled = True
    # Never leave a big week win on the table for a short settle timer
    week_delta = float(best_week.get("ret") or 0) - float(base_week.get("ret") or 0)
    if (not promote_cooled) and week_delta >= 0.05:
        promote_cooled = True
        material_why = "big_week_bypass"
        log.info("bypassing promote lock — week edge %+0.fpp", week_delta * 100)

    # Champion floor: use live re-eval when the stored peak is stale for this
    # holdout window (ghost +16% was freezing overnight climbs while live is -3%).
    champ_stored = float((champion or {}).get("week_ret") or -999)
    champ_live = None
    if champion and champion.get("knobs"):
        try:
            ch_cfg = _knobs_to_cfg(champion["knobs"])
            ch_raw = _raw_for_tf(str(ch_cfg.timeframe)) or raw
            ch_data = _prepare_cached(ch_raw, ch_cfg, feat_cache)
            _ch_dual = _dual_week_eval(ch_data, ch_cfg, end=week_end)
            champ_live = float(_ch_dual.get("ret") or 0)
            champ_prior = _ch_dual.get("prior_ret")
        except Exception as e:
            log.warning("champion live reeval failed: %s", e)
            champ_prior = None
    else:
        champ_prior = None
    base_week_ret = float(base_week.get("ret") or 0)
    if (
        champ_live is not None
        and champ_stored > -900
        and (champ_stored - champ_live) >= float(HUNT_CHAMPION_STALE_PP)
    ):
        champ_week_floor = max(champ_live, base_week_ret)
        log.info(
            "champion stale · stored %+0.f%% → live %+0.f%% · promote floor %+0.f%% (peak kept)",
            champ_stored * 100,
            champ_live * 100,
            champ_week_floor * 100,
        )
        if isinstance(champion, dict):
            champion = dict(champion)
            champion["week_ret_peak"] = max(
                float(champion.get("week_ret_peak") or champ_stored),
                champ_stored,
            )
            champion["week_ret_live"] = champ_live
            state["champion"] = champion
    else:
        champ_week_floor = champ_stored if champ_stored > -900 else base_week_ret
    # Period-loser peak must not freeze Live: if prior 14d is red, floor = deployed week.
    if (
        HUNT_TARGETS.get("require_period_robust", True)
        and champ_prior is not None
        and float(champ_prior) < float(HUNT_TARGETS.get("prior_week_floor", 0.0))
    ):
        champ_week_floor = min(champ_week_floor, base_week_ret + 0.001)
        log.info(
            "champion prior %+0.f%% fails period-robust — promote floor eased to live %+0.f%%",
            float(champ_prior) * 100,
            champ_week_floor * 100,
        )
    # Stuck overnight / paper bleeding: never require better than beating deployed
    if held >= int(HUNT_STUCK_HELD) or paper_bleeding:
        champ_week_floor = min(champ_week_floor, base_week_ret + 0.001)
        log.info(
            "%s — promote floor eased to deployed week %+0.f%%",
            "paper bleed" if paper_bleeding else f"stuck ×{held}",
            champ_week_floor * 100,
        )

    best_week_ret_now = float(best_week.get("ret") or 0)
    # Never research-promote below the (possibly stale-adjusted) champion floor
    # (silence_rotate is the escape hatch when Live is idle under that floor).
    if (
        not silence_rotate
        and not champion_rescue
        and best_cfg is not base
        and best_week_ret_now + 0.001 < champ_week_floor
    ):
        material_ok = False
        material_why = (
            f"below champion week {champ_week_floor:+.0%} (cand {best_week_ret_now:+.0%})"
        )

    stuck_soften = held >= int(HUNT_STUCK_SOFTEN_HELD)
    # When stuck long enough, dual-week absolute bar eases: beating deployed week
    # is enough (still need climb/material + period-robust relative escape).
    # Apply path: also allow beating Live with prior ≥ 0 without hitting hard_floor
    # (hard_floor still gates targets_met / champion raise elsewhere).
    _prior_now = best_week.get("prior_ret")
    _prior_nonneg = _prior_now is None or float(_prior_now) >= 0.0
    dual_or_goal = (
        _dual_week_ok(best_week)
        or best_week_ret_now >= float(HUNT_TARGETS.get("week_return", 1.0))
        or (stuck_soften and best_week_ret_now + 0.001 >= float(base_week.get("ret") or 0))
        or (
            _prior_nonneg
            and best_week_ret_now
            >= float(base_week.get("ret") or 0) + float(RESEARCH_MIN_WEEK_EDGE)
        )
    )
    research_improved = (
        best_cfg is not base
        and best_week_active
        and promote_cooled
        # Kill-illusion blocked even on escapes (survive-first).
        and not best_grade.get("kill_illusion")
        and (
            # Idle/bleed escape may skip climb/material — but not wipeout windows
            # (checked via _escape_window_ok below).
            silence_rotate
            or champion_rescue
            or (
                wf_gate_ok
                and _is_climb_ok(best_i, best_o, best_soft, base_i, base_o)
                and material_ok
                and best_week_ret_now + 0.001 >= champ_week_floor
                and dual_or_goal
            )
        )
    )
    if research_improved:
        funnel_counts["promote_attempt"] = 1
    # Period-robust: never .env-promote a book red on prior / long OOS / soft.
    # Escape: relative OOS+soft vs Live AND absolute escape floors — never week-led alone.
    if research_improved and HUNT_TARGETS.get("require_period_robust", True):
        _soft_r = (
            float(best_soft.total_return) if best_soft is not None else float(best_o.total_return)
        )
        robust_ok, robust_why = _period_robust_ok(best_week, float(best_o.total_return), _soft_r)
        if robust_ok:
            promote_provenance = "robust"
        else:
            base_soft_r = (
                float(base_soft.total_return)
                if base_soft is not None
                else float(base_o.total_return)
            )
            stuck_soften = held >= int(HUNT_STUCK_SOFTEN_HELD)
            escape_armed = bool(
                live_is_death
                and (silence_rotate or paper_bleeding or stuck_soften or champion_rescue)
            )
            esc_ok = False
            esc_why_local = "escape_not_armed" if live_is_death else f"live_not_death:{live_death_why}"
            if escape_armed:
                esc_ok, esc_why_local = _escape_window_ok(
                    best_week,
                    float(best_o.total_return),
                    _soft_r,
                    base_week,
                    float(base_o.total_return),
                    base_soft_r,
                )
            if esc_ok:
                promote_provenance = "escape"
                escape_why = esc_why_local
                log.warning(
                    "period-robust absolute fail (%s) — relative escape OK (%s) "
                    "(week %+.0f%%→%+.0f%% soft %+.0f%%→%+.0f%% OOS %+.0f%%→%+.0f%%)",
                    robust_why,
                    esc_why_local,
                    float(base_week.get("ret") or 0) * 100,
                    float(best_week.get("ret") or 0) * 100,
                    base_soft_r * 100,
                    _soft_r * 100,
                    float(base_o.total_return) * 100,
                    float(best_o.total_return) * 100,
                )
            else:
                log.info(
                    "period-robust blocked promote: %s · escape: %s",
                    robust_why,
                    esc_why_local,
                )
                research_improved = False
                material_ok = False
                material_why = f"{robust_why}; {esc_why_local}"
                promote_reason = "hold"
                promote_provenance = None
                escape_why = esc_why_local
    elif research_improved:
        promote_provenance = "robust"

    # ---- Sealed holdout (blind exam) + DSR (research) ----
    sealed_meta: dict = {"evaluated": False, "passed": None, "why": "not_attempted"}
    dsr_meta: dict = {"evaluated": False, "passed": None, "why": "not_attempted"}
    if research_improved and best_cfg is not None:
        try:
            _seal_raw = _raw_for_tf(str(best_cfg.timeframe)) or raw
            _seal_data = _prepare_cached(_seal_raw, best_cfg, feat_cache)
            cand_sealed = _week_eval(_seal_data, best_cfg, end=data_end)
            live_sealed = base_sealed_week
            sealed_ret = float(cand_sealed.get("ret") or 0)
            live_sealed_ret = float((live_sealed or {}).get("ret") or 0)
            if promote_provenance == "escape" or silence_rotate:
                _sok, _swhy = escape_sealed_floor_ok(sealed_ret)
                sealed_meta = {
                    "evaluated": True,
                    "passed": bool(_sok),
                    "sealed_week_ret": sealed_ret,
                    "live_sealed_week_ret": live_sealed_ret,
                    "why": _swhy,
                    "path": "escape",
                }
                if not _sok:
                    research_improved = False
                    material_ok = False
                    material_why = _swhy
                    promote_reason = "hold"
                    promote_provenance = None
                    log.info("escape sealed blocked: %s", _swhy)
                    _db, _dw = _should_ban_candidate(sealed_fail=True)
                    if _db:
                        _banlist_add(
                            state,
                            best_cfg,
                            week_ret=float(best_week.get("ret") or 0),
                            reason=f"escape_sealed:{_dw}",
                        )
            else:
                _sok, _swhy, sealed_meta = sealed_holdout_gate(
                    sealed_week_ret=sealed_ret,
                    live_sealed_week_ret=live_sealed_ret,
                    kill_illusion=bool(best_grade.get("kill_illusion")),
                    research_min_week_edge=float(RESEARCH_MIN_WEEK_EDGE),
                )
                sealed_meta["path"] = "research"
                if not _sok:
                    research_improved = False
                    material_ok = False
                    material_why = _swhy
                    promote_reason = "hold"
                    promote_provenance = None
                    log.info("sealed holdout blocked: %s", _swhy)
                    _db, _dw = _should_ban_candidate(sealed_fail=True)
                    if _db:
                        _banlist_add(
                            state,
                            best_cfg,
                            week_ret=float(best_week.get("ret") or 0),
                            reason=f"sealed:{_dw}",
                        )
            # DSR: required for research; audit-only on escape
            if research_improved:
                try:
                    from moonshot.config_hash import config_hash as _dsr_hash

                    _trial_hashes = []
                    for _c in list(cand_list)[: max(1, int(tested) or 1)]:
                        try:
                            _trial_hashes.append(_dsr_hash(_c))
                        except Exception:
                            pass
                    try:
                        _trial_hashes.append(_dsr_hash(best_cfg))
                    except Exception:
                        pass
                    n_pass, n_cum = record_pass_trials(
                        _trial_hashes,
                        run_id=f"hunt_{generation}",
                        first_seen_ts=_utc_now().isoformat(),
                    )
                    # Prefer sealed trade count as n_obs proxy; SR from sealed week ret.
                    n_obs = max(2, int(cand_sealed.get("trades") or 0))
                    # Map week ret to a crude SR proxy when trade list unavailable.
                    observed_sr = float(sealed_ret) * (n_obs**0.5)
                    _dok, _dwhy, dsr_meta = research_dsr_gate(
                        observed_sr=observed_sr,
                        n_obs=n_obs,
                        n_trials=max(1, int(n_cum)),
                        threshold=float(HUNT_TARGETS.get("dsr_threshold", 0.0)),
                    )
                    dsr_meta["n_trials_pass"] = n_pass
                    dsr_meta["n_trials_cum"] = n_cum
                    if (
                        promote_provenance != "escape"
                        and not silence_rotate
                        and bool(HUNT_TARGETS.get("require_dsr", True))
                        and not _dok
                    ):
                        research_improved = False
                        material_ok = False
                        material_why = _dwhy
                        promote_reason = "hold"
                        promote_provenance = None
                        log.info("DSR blocked research promote: %s", _dwhy)
                        _db, _dw = _should_ban_candidate(dsr_fail=True)
                        if _db:
                            _banlist_add(
                                state,
                                best_cfg,
                                week_ret=float(best_week.get("ret") or 0),
                                reason=f"dsr:{_dw}",
                            )
                except Exception as _dsr_e:
                    dsr_meta = {"evaluated": True, "passed": False, "why": f"dsr_error:{_dsr_e}"}
                    if promote_provenance != "escape" and not silence_rotate:
                        research_improved = False
                        material_ok = False
                        material_why = f"dsr_error:{_dsr_e}"
                        promote_reason = "hold"
                        promote_provenance = None
                        _db, _dw = _should_ban_candidate(dsr_fail=True)
                        if _db:
                            _banlist_add(
                                state,
                                best_cfg,
                                week_ret=float(best_week.get("ret") or 0),
                                reason=f"dsr_err:{_dw}",
                            )
        except Exception as _seal_e:
            sealed_meta = {"evaluated": True, "passed": False, "why": f"sealed_error:{_seal_e}"}
            research_improved = False
            material_ok = False
            material_why = f"sealed_error:{_seal_e}"
            promote_reason = "hold"
            promote_provenance = None
            log.warning("sealed/DSR gate failed: %s", _seal_e)
    if (
        not silence_rotate
        and best_cfg is not base
        and require_wf
        and (not wf_enabled or not best_wf_ok)
        and research_improved is False
        and material_ok
    ):
        if not wf_enabled:
            material_why = "walk-forward unavailable — promote blocked"
        elif not best_wf_ok:
            material_why = f"walk-forward fail ({(best_wf or {}).get('why', 'no')})"
    if champion_rescue and research_improved:
        material_ok = True
        material_why = "champion_rescue"
        log.info(
            "CHAMPION RESCUE promote locked · %s %sx@%s week=%+.0f%%",
            best_cfg.strategy,
            int(best_cfg.leverage),
            best_cfg.timeframe,
            best_week_ret_now * 100,
        )
    if silence_rotate and research_improved:
        material_ok = True
        material_why = f"silence_rotate idle={idle_h:.1f}h"
        log.info(
            "silence rotate promote locked in · %s %sx@%s week=%+.0f%%",
            best_cfg.strategy,
            int(best_cfg.leverage),
            best_cfg.timeframe,
            best_week_ret_now * 100,
        )
    elif silence_rotate and not research_improved:
        log.warning(
            "silence rotate selected a book but promote gates still blocked it "
            "(active=%s cooled=%s wf=%s) — check week_min",
            best_week_active,
            promote_cooled,
            wf_gate_ok,
        )
    if best_cfg is not base and not research_improved:
        log.info(
            "climb found but no research promote (%s) · week %+0.f%%→%+0.f%% score %s→%s",
            material_why,
            float(base_week.get("ret") or 0) * 100,
            float(best_week.get("ret") or 0) * 100,
            base_grade.get("score_raw", base_grade.get("score")),
            best_grade.get("score_raw", best_grade.get("score")),
        )
        # Remember only real week upgrades — never park a worse book as pending
        if (
            best_cfg is not base
            and best_week_active
            and best_week_ret_now >= champ_week_floor - 0.001
            and best_week_ret_now >= float(base_week.get("ret") or 0) + RESEARCH_MIN_WEEK_EDGE
        ):
            state["pending_promote"] = {
                "ts": _utc_now().isoformat(),
                "week_ret": best_week_ret_now,
                "score": int(best_grade.get("score_raw", best_grade.get("score") or 0)),
                "why": material_why,
                "knobs": _cfg_to_knobs(best_cfg),
            }
            _save_autopilot_state(state)
    # Finalist honesty on full book (max calendar under bar cap).
    # Skip for silence_rotate / champion_rescue — escape hatches to free Live now.
    if (
        research_improved
        and len(full_symbols) > len(symbols)
        and not silence_rotate
        and not champion_rescue
    ):
        try:
            f_raw = _hunt_fetch_raw(
                full_symbols,
                best_cfg.timeframe,
                "finalist",
                refresh=False,
                source=hunt_src,
            )
            if not f_raw:
                raise RuntimeError("finalist fetch empty")
            f_data = prepare_data(f_raw, best_cfg)
            f_t1 = max(sd.bars[-1].Index for sd in f_data.values())
            f_week = _week_eval(f_data, best_cfg, end=f_t1)
            f_soft = _eval_soft_oos(
                f_data,
                best_cfg,
                min(sd.bars[0].Index for sd in f_data.values())
                + (f_t1 - min(sd.bars[0].Index for sd in f_data.values())) * (1 - oos_frac),
                end=pd.Timestamp(f_t1) - pd.Timedelta(days=int(WEEK_DAYS)),
            )
            search_week_ret = float(best_week.get("ret") or 0)
            f_week_ret = float(f_week.get("ret") or 0)
            # Never throw away a big search-week win just because the wider
            # symbol book is quieter — keep promoting the search climb.
            big_search_week = search_week_ret >= float(base_week.get("ret") or 0) + 0.05
            if (
                int(f_week.get("trades") or 0) < week_min
                and int(f_week.get("soft_trades") or 0) < week_min
                and not big_search_week
            ):
                log.warning("finalist silent on full book — skip research promote")
                research_improved = False
            else:
                # Soft floor: only veto when baseline is already healthy; while
                # climbing out of a hole, full-book soft is informational.
                f_floor = float(HUNT_TARGETS.get("soft_return_floor", -0.30))
                base_healthy = base_o.expectancy_r > 0 and base_o.total_return > 0
                if base_healthy and f_soft.total_return < f_floor and not big_search_week:
                    log.warning("finalist soft floor fail on full book — skip research promote")
                    research_improved = False
                else:
                    # Prefer the better week number for Live (search vs full).
                    if f_week_ret >= search_week_ret or not big_search_week:
                        best_week = f_week
                        best_grade["week"] = f_week
                        best_grade["week_return"] = f_week.get("ret")
                    log.info(
                        "finalist full-book holdout week %+.0f%% n=%d soft=%+.0f%%",
                        float(f_week.get("ret") or 0) * 100,
                        int(f_week.get("trades") or 0),
                        f_soft.total_return * 100,
                    )
        except Exception as e:
            # Fail closed for research — never .env-promote on finalist error.
            research_improved = False
            material_ok = False
            material_why = f"finalist_error:{e}"
            promote_reason = "hold"
            promote_provenance = None
            log.warning("finalist full-book recheck failed: %s — research promote blocked", e)

    # A1: research venue (often Binance spot) ≠ MEXC perps — gate .env on MEXC holdout.
    transfer_gap: Optional[dict] = None
    if (
        research_improved
        and bool(HUNT_TARGETS.get("require_mexc_transfer_ok", True))
        and str(hunt_src) != str(RANGE_SOURCE_MEXC)
    ):
        spot_ret = float(best_week.get("ret") or 0)
        try:
            mexc_syms = list(full_symbols) if full_symbols else list(symbols)
            mexc_raw = _hunt_fetch_raw(
                mexc_syms[: max(len(symbols), 1)],
                best_cfg.timeframe,
                hunt_tier,
                refresh=False,
                source=RANGE_SOURCE_MEXC,
            )
            if not mexc_raw:
                raise RuntimeError("MEXC transfer fetch empty")
            mexc_data = prepare_data(mexc_raw, best_cfg)
            mexc_t1 = max(sd.bars[-1].Index for sd in mexc_data.values())
            mexc_week = _week_eval(mexc_data, best_cfg, end=mexc_t1)
            mexc_ret = float(mexc_week.get("ret") or 0)
            transfer_gap = {
                "research_source": str(hunt_src),
                "research_ret": spot_ret,
                "mexc_ret": mexc_ret,
                "mexc_trades": int(mexc_week.get("trades") or 0),
                "mexc_span": mexc_week.get("span"),
            }
            log.info(
                "transfer gap · %s holdout %+.0f%% n=%d → MEXC %+.0f%% n=%d",
                hunt_src,
                spot_ret * 100,
                int(best_week.get("trades") or 0),
                mexc_ret * 100,
                int(mexc_week.get("trades") or 0),
            )
            escape_xfer = bool(
                silence_rotate
                or paper_bleeding
                or champion_rescue
                or held >= int(HUNT_STUCK_SOFTEN_HELD)
            )
            xfer_gap_pp = float(spot_ret) - float(mexc_ret)
            transfer_gap["gap_pp"] = xfer_gap_pp
            if mexc_ret <= 0.0 and spot_ret > 0.0 and not escape_xfer:
                research_improved = False
                material_why = (
                    f"MEXC holdout {mexc_ret:+.0%} ≤ 0 while research "
                    f"({hunt_src}) {spot_ret:+.0%} — refuse .env patch"
                )
                log.warning("transfer gate blocked promote: %s", material_why)
            elif (
                (not escape_xfer)
                and xfer_gap_pp >= 0.15
                and mexc_ret < float(base_week.get("ret") or 0) + 0.02
            ):
                # Large venue mismatch: research looks great, MEXC does not beat Live.
                research_improved = False
                material_why = (
                    f"MEXC-first: gap {xfer_gap_pp:+.0%} vs research; "
                    f"MEXC {mexc_ret:+.0%} does not beat Live "
                    f"{float(base_week.get('ret') or 0):+.0%}"
                )
                log.warning("transfer gap blocked promote: %s", material_why)
            elif mexc_ret <= 0.0 and spot_ret > 0.0 and escape_xfer:
                if live_is_death:
                    log.warning(
                        "transfer gate soft-fail during death escape "
                        "(MEXC %+.0f%% research %+.0f%%) — still promoting",
                        mexc_ret * 100,
                        spot_ret * 100,
                    )
                else:
                    research_improved = False
                    material_why = (
                        f"MEXC holdout {mexc_ret:+.0%} ≤ 0 while research "
                        f"{spot_ret:+.0%} — escape transfer refused (live not death)"
                    )
                    log.warning("transfer gate blocked: %s", material_why)
            if research_improved and mexc_ret == mexc_ret:
                # Prefer MEXC (paper venue) for Live messaging / decision numbers.
                best_week = {
                    **best_week,
                    "ret": mexc_week.get("ret"),
                    "soft_ret": mexc_week.get("soft_ret"),
                    "final": mexc_week.get("final"),
                    "trades": mexc_week.get("trades"),
                    "soft_trades": mexc_week.get("soft_trades"),
                    "span": mexc_week.get("span"),
                    "transfer_ok": True,
                    "research_ret": spot_ret,
                }
                best_grade["week"] = best_week
                best_grade["week_return"] = best_week.get("ret")
                best_week_ret_now = float(best_week.get("ret") or 0)
        except Exception as e:
            escape_xfer = bool(
                silence_rotate
                or paper_bleeding
                or champion_rescue
                or held >= int(HUNT_STUCK_SOFTEN_HELD)
            )
            if escape_xfer and live_is_death:
                log.warning(
                    "transfer check failed during death escape (%s) — still promoting", e
                )
                transfer_gap = {
                    "research_source": str(hunt_src),
                    "research_ret": spot_ret,
                    "error": str(e),
                    "soft_fail": True,
                }
            else:
                research_improved = False
                material_why = f"MEXC transfer check failed: {e}"
                log.warning("transfer gate blocked promote: %s", material_why)
                transfer_gap = {
                    "research_source": str(hunt_src),
                    "research_ret": spot_ret,
                    "error": str(e),
                }

    # Paper expectancy demotion: research winners that bleed on paper stay off .env
    # (escape/champ rescue still allowed — capital already stranded).
    # mean_R≤0 @ M≥40 on the *deployed* book is handled by paper retire (rescue path);
    # here we only block promoting *to* a challenger that already STOP-labels on paper.
    if research_improved and not (silence_rotate or champion_rescue or paper_bleeding):
        try:
            from moonshot.config_hash import config_hash as _cfg_hash
            from moonshot.paper_expectancy import bucket_trades, load_trade_rows
            from moonshot.paths import DB_PATH as _DB

            _ch = _cfg_hash(best_cfg)
            _buckets = bucket_trades(load_trade_rows(_DB))
            _hit = next(
                (b for b in _buckets if b.config_hash == _ch and not b.legacy),
                None,
            )
            if _hit is not None and _hit.n >= 20:
                label = _hit.decision_label()
                if label.startswith("STOP"):
                    research_improved = False
                    material_why = (
                        f"paper_expectancy demote · hash={_ch[:8]} n={_hit.n} "
                        f"mean_R={_hit.mean_r:+.3f} → {label}"
                    )
                    log.warning("%s", material_why)
        except Exception as e:
            log.info("paper expectancy check skipped: %s", e)

    # ---- Product A: lev-up / hysteresis / dwell / interim 2× cost (escape bypass) ----
    escape_bypass = bool(champion_rescue or paper_bleeding or silence_rotate)
    base_week_ret = float(base_week.get("ret") or 0)
    best_week_ret_now = float(best_week.get("ret") or 0)

    if research_improved and int(best_cfg.leverage) > int(base.leverage):
        # Escape never lev-ups — wipeout risk. Robust only, with +5pp margin.
        if promote_provenance == "escape" or silence_rotate or champion_rescue:
            research_improved = False
            material_why = f"escape cannot lev-up ({int(base.leverage)}x→{int(best_cfg.leverage)}x)"
            log.info("lev-up blocked promote: %s", material_why)
            promote_provenance = None
        else:
            lev_edge = best_week_ret_now - base_week_ret
            if lev_edge < float(RESEARCH_LEV_UP_WEEK_PP):
                research_improved = False
                material_why = (
                    f"lev_up needs +{float(RESEARCH_LEV_UP_WEEK_PP):.0%} week "
                    f"(have {lev_edge:+.1%})"
                )
                log.info("lev-up blocked promote: %s", material_why)
                promote_provenance = None

    if research_improved and not escape_bypass:
        fp = _cfg_fingerprint(best_cfg)
        prev_fp = state.get("challenger_fp")
        if prev_fp != fp:
            streak = 1
            state["challenger_fp"] = fp
        else:
            streak = int(state.get("challenger_streak") or 0) + 1
        state["challenger_streak"] = streak
        if streak < int(RESEARCH_HYSTERESIS_PASSES):
            research_improved = False
            material_why = f"hysteresis streak {streak}/{int(RESEARCH_HYSTERESIS_PASSES)} fp={fp}"
            log.info("hysteresis blocked promote: %s", material_why)
        elif best_week_ret_now < base_week_ret + float(RESEARCH_HYSTERESIS_WEEK_PP):
            research_improved = False
            material_why = (
                f"hysteresis need +{float(RESEARCH_HYSTERESIS_WEEK_PP):.0%} week "
                f"(have {best_week_ret_now - base_week_ret:+.1%})"
            )
            log.info("hysteresis blocked promote: %s", material_why)
    elif not research_improved:
        # Fingerprint changed while not promoting — reset challenger streak.
        try:
            fp_now = _cfg_fingerprint(best_cfg) if best_cfg is not base else None
        except Exception:
            fp_now = None
        if fp_now and state.get("challenger_fp") and state.get("challenger_fp") != fp_now:
            state["challenger_streak"] = 0
            state["challenger_fp"] = None

    if research_improved and not escape_bypass:
        # Dwell only after a prior *non-escape* research promote (escape may bypass).
        last_ne = state.get("research_non_escape_promote_at")
        if last_ne:
            try:
                dwell_age = (_utc_now() - pd.Timestamp(last_ne)).total_seconds()
            except Exception:
                dwell_age = float(RESEARCH_PROMOTE_DWELL_SEC)
            if dwell_age < float(RESEARCH_PROMOTE_DWELL_SEC):
                research_improved = False
                left = int(float(RESEARCH_PROMOTE_DWELL_SEC) - dwell_age)
                material_why = f"dwell {left}s left after non-escape promote"
                log.info("dwell blocked promote: %s", material_why)

    if research_improved and not escape_bypass:
        try:
            stressed = replace(
                best_cfg,
                fee_taker=float(best_cfg.fee_taker) * 2.0,
                slippage_bps=float(best_cfg.slippage_bps) * 2.0,
            )
            s_raw = _raw_for_tf(str(stressed.timeframe)) or raw
            s_data = _prepare_cached(s_raw, stressed, feat_cache)
            s_week = _week_eval(s_data, stressed, end=week_end)
            s_ret = float(s_week.get("ret") or 0)
            if s_ret < base_week_ret:
                research_improved = False
                material_why = f"2x cost week {s_ret:+.0%} < Live {base_week_ret:+.0%}"
                log.info("2x-cost blocked promote: %s", material_why)
                _db, _dw = _should_ban_candidate(cost_ok=False)
                if _db:
                    _banlist_add(
                        state,
                        best_cfg,
                        week_ret=float(best_week.get("ret") or 0),
                        reason=f"cost_2x:{_dw}",
                    )
            else:
                log.info(
                    "2x-cost OK · stressed week %+.0f%% vs Live %+.0f%%",
                    s_ret * 100,
                    base_week_ret * 100,
                )
        except Exception as e:
            log.info("2x cost gate skipped: %s", e)

    # Stop-N: flag + CRITICAL; clear exhausted only on robust research promote.
    if research_improved and promote_provenance == "robust":
        state["no_qualify_streak"] = 0
        state["hunt_exhausted"] = False
        state["regime_reassess_needed"] = False
        state.pop("regime_reassess_at", None)
        state.pop("regime_reassess_snapshot", None)
    elif research_improved:
        # Escape deploy does not clear exhaustion / reassess.
        log.info(
            "escape promote · hunt_exhausted stays %s (provenance=%s)",
            bool(state.get("hunt_exhausted")),
            promote_provenance,
        )
    else:
        nq = int(state.get("no_qualify_streak") or 0) + 1
        state["no_qualify_streak"] = nq
        if nq >= int(HUNT_EXHAUSTED_GENS):
            was = bool(state.get("hunt_exhausted"))
            state["hunt_exhausted"] = True
            state["regime_reassess_needed"] = True
            if not state.get("regime_reassess_at"):
                state["regime_reassess_at"] = _utc_now().isoformat()
            state["regime_reassess_snapshot"] = {
                "live_week": float(base_week.get("ret") or 0),
                "prior": (base_week or {}).get("prior_ret"),
                "oos": float(base_o.total_return),
                "soft": float(
                    base_soft.total_return if base_soft is not None else base_o.total_return
                ),
                "book": {
                    "strategy": str(base.strategy),
                    "timeframe": str(base.timeframe),
                    "leverage": int(base.leverage),
                },
                "no_qualify_streak": nq,
            }
            if not was:
                exh_msg = (
                    f"hunt_exhausted · {nq} passes with no qualifying promote — "
                    "reassess data/objective/universe (hunt still running)"
                )
                log.warning("%s", exh_msg)
                _live_feed(
                    "scoreboard_critical",
                    exh_msg,
                    generation=generation,
                    no_qualify_streak=nq,
                )

    improved = bool(research_improved)  # research tier → .env
    if not improved:
        promote_provenance = None
    elif promote_provenance is None:
        promote_provenance = "escape" if (silence_rotate or champion_rescue) else "robust"
    # Persist real reason — do not collapse matrix_escape into silence_rotate.
    if champion_rescue and improved:
        promote_tier = "champion_rescue"
    elif improved and promote_reason == "matrix_escape":
        promote_tier = "matrix_escape"
    elif improved and promote_reason == "silence_density":
        promote_tier = "silence_density"
    elif silence_rotate and improved:
        promote_tier = "silence_rotate"
    else:
        promote_tier = "research" if improved else "hold"
    # F3: confirmed / elite requires forward paper proof
    fwd_ok = bool(fwd and fwd.get("ok"))
    confirmed = bool(improved and fwd_ok and best_grade.get("targets_met"))

    knobs = _cfg_to_knobs(best_cfg)
    fail_checks = [k for k, v in (best_grade.get("checks") or {}).items() if not v]
    decision = {
        "ts": _utc_now().isoformat(),
        "improved": improved,
        "promote_tier": promote_tier,
        "promote_reason": promote_reason,
        "promote_provenance": promote_provenance,
        "research_promote": bool(
            improved and promote_provenance == "robust" and not silence_rotate
        ),
        "escape_why": escape_why,
        "material_why": material_why,
        "promote_cooled": bool(promote_cooled),
        "fail_checks": fail_checks,
        "tested": tested,
        "soft_runs": soft_runs,
        "search_mode": search_mode,
        "primary_metric": "p25_fold_sortino",
        "funnel": dict(funnel_counts),
        "p25_sortino": (best_fold or {}).get("p25_sortino"),
        "fold_roi": best_roi,
        "cost_ok": bool(best_cost_ok),
        "sealed_holdout": sealed_meta,
        "dsr": dsr_meta,
        "live_death": {"death": live_is_death, "why": live_death_why},
        "nearest_miss": nearest_miss_payload(
            week_ret=float(best_week.get("ret") or 0) if not improved else None,
            prior_ret=(best_week or {}).get("prior_ret") if not improved else None,
            oos_ret=float(best_o.total_return) if not improved else None,
            soft_ret=(
                float(best_soft.total_return)
                if (not improved and best_soft is not None)
                else None
            ),
            why=str(material_why or escape_why or "") if not improved else "",
            search_mode=str(search_mode),
            book={
                "strategy": str(best_cfg.strategy),
                "timeframe": str(best_cfg.timeframe),
                "leverage": int(best_cfg.leverage),
            }
            if not improved
            else {},
        ),
        "hunt_exhausted": bool(state.get("hunt_exhausted")),
        "regime_reassess_needed": bool(state.get("regime_reassess_needed")),
        "selection_span": f"{holdout_start:%Y-%m-%d} → {selection_end:%Y-%m-%d}",
        "sealed_span": f"{selection_end:%Y-%m-%d} → {data_end:%Y-%m-%d}",
        "held_streak_before": held,
        "generation": generation,
        "mode": mode.lower(),
        "range": f"{t0:%Y-%m-%d} -> {t1:%Y-%m-%d}",
        "holdout": f"{holdout_start:%Y-%m-%d} -> {t1:%Y-%m-%d}",
        "hunt_source": hunt_src,
        "hunt_tier": hunt_tier,
        "hunt_days": int(hunt_days),
        "hunt_bars_est": int(bars),
        "transfer_gap": transfer_gap,
        "leverage_ladder": leverage_ladder,
        "baseline": {
            "train_exp": round(base_i.expectancy_r, 3),
            "test_exp": round(base_o.expectancy_r, 3),
            "train_n": base_i.trades,
            "test_n": base_o.trades,
            "test_return": round(base_o.total_return, 3),
            "score": base_grade["score"],
            "week_return": base_week.get("ret"),
            "leverage": int(base.leverage),
            "timeframe": str(base.timeframe),
        },
        "best": {
            "train_exp": round(best_i.expectancy_r, 3),
            "test_exp": round(best_o.expectancy_r, 3),
            "train_n": best_i.trades,
            "test_n": best_o.trades,
            "max_dd": round(best_o.max_drawdown, 3),
            "test_return": round(best_o.total_return, 3),
            "oos_final": round(best_o.final_equity, 2),
            "soft_return": best_grade.get("soft_return"),
            "soft_final": best_grade.get("soft_final"),
            "kill_illusion": best_grade.get("kill_illusion"),
            "score": best_grade["score"],
            "score_raw": best_grade.get("score_raw"),
            "recent": best_recent,
            "week": best_week,
            "week_return": best_week.get("ret"),
            "week_final": best_week.get("final"),
            "week_span": best_week.get("span"),
            "targets_met": best_grade["targets_met"],
            "leverage": int(best_cfg.leverage),
            "timeframe": str(best_cfg.timeframe),
            **knobs,
        },
        "grade": best_grade,
        "forward": fwd,
        "message": "",
    }

    def _score_msg(g: dict) -> str:
        raw = int(g.get("score_raw", g.get("score", 0)))
        disp = int(g.get("score", 0))
        return f"{disp}/100" if raw == disp else f"{disp}/100 (research {raw})"

    best_recent_ok = bool(best_recent.get("ok"))
    week_msg = (
        f"holdout{int(WEEK_DAYS)}d ${float(best_week.get('final') or 0):.0f} "
        f"({float(best_week.get('ret') or 0):+.0%}, "
        f"soft {float(best_week.get('soft_ret') or 0):+.0%}, "
        f"n={int(best_week.get('trades') or 0)}, {best_week.get('span', '')})"
    )
    recent_msg = (
        f"90d {best_recent.get('exp', 0):+.2f}R n={best_recent.get('trades', 0)} "
        f"[{'OK' if best_recent_ok else best_recent.get('why', 'FAIL')}]"
    )
    week_tgt = float(HUNT_TARGETS.get("week_return", 1.0))
    week_gap = max(0.0, week_tgt - float(best_week.get("ret") or 0))
    if improved and autopilot_frozen():
        log.info("AUTOPILOT_FROZEN=1: promotion suppressed")
        decision["promote_suppressed"] = True
        decision["improved"] = False
        decision["promote_tier"] = "hold"
        decision["message"] = (
            f"FROZEN — would promote {best_cfg.strategy} {best_cfg.leverage}x@"
            f"{best_cfg.timeframe} but AUTOPILOT_FROZEN=1 (no .env rewrite)."
        )
        improved = False
        promote_tier = "hold"
        _live_feed("hold", decision["message"], generation=generation)
        with contextlib.suppress(Exception):
            from moonshot.config_hash import config_hash as _ch

            Persistence(str(DB_PATH), mode="paper").log_event(
                _utc_now(),
                "*",
                "freeze",
                {
                    "reason": "AUTOPILOT_FROZEN",
                    "old_hash": _ch(base),
                    "new_hash": _ch(best_cfg),
                    "frozen": True,
                    "generation": generation,
                    "would_promote": True,
                    "promote_reason": promote_reason,
                },
            )
    if improved:
        env_updates = _cfg_to_env_updates(best_cfg)
        try:
            from moonshot.config_hash import config_hash as _ch_pre

            _old_hash = _ch_pre(base)
            _new_hash = _ch_pre(best_cfg)
        except Exception:
            _old_hash, _new_hash = None, None
        try:
            _patch_env(env_updates)  # includes _verify_env_updates
        except ConfigError as e:
            log.error("promote APPLY FAIL — not marking improved: %s", e)
            improved = False
            promote_tier = "hold"
            decision["improved"] = False
            decision["promote_tier"] = "hold"
            decision["apply_fail"] = str(e)
            decision["message"] = (
                f"APPLY FAIL — gates said promote {best_cfg.strategy} "
                f"{best_cfg.leverage}x@{best_cfg.timeframe} but .env verify failed: {e}"
            )
            material_why = f"apply_fail:{e}"
            promote_reason = "apply_fail"
            decision["promote_reason"] = promote_reason
            decision["material_why"] = material_why
            _live_feed(
                "error",
                decision["message"],
                generation=generation,
                promote_reason="apply_fail",
            )
        else:
            # Do NOT reset forward_since on research promote — that made FWD_MIN_TRADES
            # unreachable across climbs. Only elite/confirmed resets the clock below.
            funnel_counts["promote"] = 1
            decision["funnel"] = dict(funnel_counts)
            decision["research_promote"] = bool(
                promote_provenance == "robust" and not silence_rotate
            )
            state["research_promote_at"] = decision["ts"]
            if not escape_bypass:
                state["research_non_escape_promote_at"] = decision["ts"]
                state["challenger_streak"] = 0
                state["challenger_fp"] = None
            state["expected_paper_book"] = {
                "strategy": str(best_cfg.strategy),
                "timeframe": str(best_cfg.timeframe),
                "leverage": str(int(best_cfg.leverage)),
                "at": decision["ts"],
                "promote_reason": promote_reason,
            }
            # Research promote attempt artifacts (escape may skip incomplete layout flag).
            with contextlib.suppress(Exception):
                from moonshot.hunt.artifacts import (
                    artifact_layout_complete,
                    new_hunt_run_dir,
                    write_artifact_layout,
                )
                from moonshot.paths import ROOT

                _run = new_hunt_run_dir(ROOT / "runs")
                write_artifact_layout(
                    _run,
                    config=_cfg_to_knobs(best_cfg),
                    folds={
                        "fold_sortinos": (best_fold or {}).get("fold_sortinos") or [],
                        "p25_sortino": (best_fold or {}).get("p25_sortino"),
                        "ok": (best_fold or {}).get("ok"),
                        "why": (best_fold or {}).get("why"),
                    },
                    metrics={
                        "primary_metric": "p25_fold_sortino",
                        "p25_fold_sortino": (best_fold or {}).get("p25_sortino"),
                        "roi": best_roi,
                        "week_ret": float(best_week.get("ret") or 0),
                        "cost_ok": bool(best_cost_ok),
                    },
                    dsr=dsr_meta,
                    cost_stress={"cost_ok": bool(best_cost_ok)},
                    sealed_holdout=sealed_meta,
                    trials={"tested": tested, "funnel": funnel_counts},
                )
                decision["hunt_run_dir"] = str(_run)
                decision["artifact_complete"] = bool(artifact_layout_complete(_run))
                if (
                    promote_provenance == "robust"
                    and not silence_rotate
                    and not decision["artifact_complete"]
                ):
                    decision["research_artifact_incomplete"] = True
                    log.warning("research promote artifact layout incomplete · %s", _run)
            _audit_kind = "rotate" if silence_rotate else "promote"
            with contextlib.suppress(Exception):
                Persistence(str(DB_PATH), mode="paper").log_event(
                    _utc_now(),
                    "*",
                    _audit_kind,
                    {
                        "reason": promote_reason,
                        "promote_tier": promote_tier,
                        "promote_provenance": promote_provenance,
                        "old_hash": _old_hash,
                        "new_hash": _new_hash,
                        "frozen": False,
                        "generation": generation,
                        "strategy": str(best_cfg.strategy),
                        "timeframe": str(best_cfg.timeframe),
                        "leverage": int(best_cfg.leverage),
                    },
                )
            state.pop("pending_promote", None)
            # New book — clear paper-retire arm so skip_heavy does not stick.
            if state.get("paper_retire_source") == "paper_retire":
                state["paper_retire_source"] = None
                state.pop("paper_retire_cooldown_until", None)
            # Bleed halt clears only on robust promote — escape keeps entries paused.
            if promote_provenance == "robust":
                _clear_paper_bleed_circuit(state, ts=decision["ts"])
            else:
                log.warning(
                    "escape promote · bleed halt stays armed until robust promote "
                    "(provenance=%s reason=%s)",
                    promote_provenance,
                    promote_reason,
                )
            _prov_tag = (
                "robust"
                if promote_provenance == "robust"
                else f"ESCAPE ({escape_why or promote_reason})"
            )
            decision["message"] = (
                f"Research promote [{_prov_tag}] ({promote_reason}) · {best_cfg.strategy} "
                f"{best_cfg.leverage}x@{best_cfg.timeframe} · {week_msg} · "
                f"score {_score_msg(base_grade)}→{_score_msg(best_grade)} · "
                f"long-OOS ${base_o.final_equity:.0f}→${best_o.final_equity:.0f} "
                f"({base_o.total_return:+.0%}→{best_o.total_return:+.0%}) · "
                f"win {base_o.win_rate:.0%}→{best_o.win_rate:.0%} ({best_o.trades} trades). "
                f"Climbing — goal +{week_tgt:.0%} holdout week "
                f"(gap {week_gap:.0%} still open). "
                f"[{_wf_msg(best_wf)}; {recent_msg}] Paper trial — real money still off."
            )
            log.info("AUTOPILOT RESEARCH UPGRADE: %s", decision["message"])
            _live_feed(
                "upgrade",
                decision["message"],
                generation=generation,
                score=best_grade["score"],
                score_raw=best_grade.get("score_raw"),
                targets_met=best_grade["targets_met"],
                promote_tier=promote_tier,
                promote_reason=promote_reason,
                week_return=best_week.get("ret"),
                week_final=best_week.get("final"),
                leverage=int(best_cfg.leverage),
                timeframe=str(best_cfg.timeframe),
            )
    if not improved and not decision.get("apply_fail"):
        hold_why = (
            f"soft-checked {soft_runs}/{soft_budget}; "
            f"holdout week {float(best_week.get('ret') or 0):+.0%} "
            f"(any better week/OOS/soft promotes; gap to +{week_tgt:.0%} is "
            f"{week_gap:.0%}); {recent_msg}"
        )
        decision["message"] = (
            f"Pass #{generation} ({search_mode}): HOLD · {week_msg} · "
            f"score {_score_msg(best_grade)} · "
            f"{best_cfg.strategy} {best_cfg.leverage}x@{best_cfg.timeframe} · "
            f"long-OOS ${best_o.final_equity:.0f} / {best_o.total_return:+.0%} · "
            f"win {best_o.win_rate:.0%}, n={best_o.trades}; "
            f"tried {tested} · {hold_why}. "
            + (
                "TARGETS MET — waiting forward proof."
                if best_grade["targets_met"]
                else f"Still hunting holdout {int(WEEK_DAYS)}d toward +100%."
            )
            + f" [{_wf_msg(best_wf)}]"
        )
        log.info("AUTOPILOT HOLD: %s", decision["message"])
        _live_feed(
            "targets_met" if best_grade["targets_met"] else "hold",
            decision["message"],
            generation=generation,
            tested=tested,
            score=best_grade["score"],
            targets_met=best_grade["targets_met"],
            promote_tier="hold",
            soft_runs=soft_runs,
            hold_why=hold_why,
            leverage=int(best_cfg.leverage),
            timeframe=str(best_cfg.timeframe),
        )

    # Phase timing — required for closed-loop "faster" claims (inputs + seconds).
    _timing["total_s"] = time.monotonic() - _pass_t0
    _denser_tag = "skip" if skip_denser else str(len(denser))
    log.info(
        "pass_timing matrix_hard_s=%.1f matrix_soft_s=%.1f stage_a_s=%.1f "
        "stage_b_s=%.1f stage_c_s=%.1f total_s=%.1f denser=%s n_cand=%d "
        "n_stage_a=%d tfs=%s week=%+.1f%%",
        float(_timing.get("matrix_hard_s") or 0),
        float(_timing.get("matrix_soft_s") or 0),
        float(_timing.get("stage_a_s") or 0),
        float(_timing.get("stage_b_s") or 0),
        float(_timing.get("stage_c_s") or 0),
        float(_timing.get("total_s") or 0),
        _denser_tag,
        int(len(cand_list)),
        int(len(_stage_a_cands)),
        ",".join(str(t) for t in matrix_tfs) or str(base.timeframe),
        float(best_week.get("ret") or 0) * 100,
    )

    # Persist progress; elite only with forward proof (F3)
    state["last_improve"] = decision
    state["last_improve_at"] = decision["ts"]
    state["generation"] = generation
    state["improve_passes"] = int(state.get("improve_passes", 0)) + 1
    state["busy"] = False
    state.pop("pass_started_at", None)
    state.pop("pass_heartbeat_at", None)
    # Live must show the DEPLOYED book. On HOLD, never paint an unpromoted
    # candidate week (that is how +16% became -21% on the dashboard).
    state["grade"] = best_grade if improved else base_grade
    state["targets"] = HUNT_TARGETS
    state["hunt"] = hunt
    state["promote_tier"] = promote_tier
    state["leverage_ladder"] = leverage_ladder
    state["tf_lev_matrix"] = list(tf_lev_matrix_rows or [])
    state["matrix_failed"] = bool(matrix_failed)
    decision["matrix_failed"] = bool(matrix_failed)
    if matrix_winner_row:
        state["tf_lev_matrix_winner"] = {
            k: matrix_winner_row.get(k)
            for k in (
                "tf",
                "leverage",
                "week_ret",
                "prior_ret",
                "test_ret",
                "soft_ret",
                "robust_ok",
            )
        }
    else:
        state.pop("tf_lev_matrix_winner", None)
    raw_score = int(best_grade.get("score_raw", best_grade["score"]))
    elite_raw = int((elite or {}).get("score_raw", (elite or {}).get("score", -1)))
    elite_ok = (
        _is_valid_candidate(best_i, best_o, best_soft)
        and raw_score > 0
        and not best_grade.get("kill_illusion")
        and best_wf_ok
        and fwd_ok  # F3: forward paper required for elite
    )
    if elite_ok and (elite is None or raw_score > elite_raw):
        state["elite"] = {
            "score": best_grade["score"],
            "score_raw": raw_score,
            "targets_met": best_grade["targets_met"],
            "confirmed": True,
            "knobs": knobs,
            "train_exp": decision["best"]["train_exp"],
            "test_exp": decision["best"]["test_exp"],
            "train_n": decision["best"]["train_n"],
            "test_n": decision["best"]["test_n"],
            "max_dd": decision["best"]["max_dd"],
            "win": best_grade.get("win"),
            "test_return": best_grade.get("test_return"),
            "oos_final": best_grade.get("oos_final"),
            "soft_return": best_grade.get("soft_return"),
            "soft_final": best_grade.get("soft_final"),
            "walk_forward": best_wf,
            "recent": best_recent,
            "week": best_week,
            "forward": fwd,
            "ts": decision["ts"],
            "generation": generation,
        }
        promote_tier = "confirmed"
        decision["promote_tier"] = "confirmed"
        state["forward_since"] = decision["ts"]
    elif confirmed:
        decision["promote_tier"] = "confirmed"
        state["forward_since"] = decision["ts"]

    # Live book vs peak champion: silence only patches Live; peak is raise-only.
    live_knobs = knobs if improved else _cfg_to_knobs(base)
    state["live_deployed"] = {
        "ts": decision["ts"],
        "generation": generation,
        "week_ret": float((best_week if improved else base_week).get("ret") or 0),
        "knobs": live_knobs,
        "promote_reason": promote_reason if improved else "hold",
        "promote_tier": decision.get("promote_tier"),
    }
    # Peak raise only on robust provenance — never escape / silence.
    if improved and promote_provenance == "robust" and not silence_rotate:
        _soft_r = (
            float(best_soft.total_return) if best_soft is not None else float(best_o.total_return)
        )
        _raise_ok, _raise_why = _period_robust_ok(best_week, float(best_o.total_return), _soft_r)
        if _raise_ok:
            try:
                _p25_raise = float(best_fold.get("p25_sortino"))
            except (TypeError, ValueError):
                _p25_raise = float("nan")
            if _p25_raise != _p25_raise:
                _p25_raise = None
            champion = _raise_champion_peak(
                state,
                knobs=knobs,
                week_ret=float(best_week.get("ret") or 0),
                score=raw_score,
                week=best_week,
                ts=decision["ts"],
                generation=generation,
                oos_ret=float(best_o.total_return),
                soft_ret=_soft_r,
                prior_ret=(
                    float(best_week.get("prior_ret"))
                    if best_week.get("prior_ret") is not None
                    else None
                ),
                p25_sortino=_p25_raise,
                roi=float(best_roi),
                cost_ok=bool(best_cost_ok),
            )
        else:
            log.info("period-robust blocked champion raise: %s", _raise_why)
            if champion:
                _sync_champion_peak(state, champion)
    elif champion:
        # Escape / silence / hold — never raise champion_peak on escape tiers.
        _sync_champion_peak(state, champion)
    hist = list(state.get("history") or [])
    hist.append(
        {
            "ts": decision["ts"],
            "generation": generation,
            "improved": improved,
            "promote_tier": decision.get("promote_tier"),
            "promote_reason": promote_reason,
            "promote_provenance": promote_provenance,
            "escape_why": escape_why,
            "material_why": material_why,
            "matrix_failed": bool(matrix_failed),
            "tested": tested,
            "soft_runs": soft_runs,
            "search_mode": search_mode,
            "train_exp": decision["best"]["train_exp"],
            "test_exp": decision["best"]["test_exp"],
            "train_n": decision["best"]["train_n"],
            "test_n": decision["best"]["test_n"],
            "win": best_grade.get("win"),
            "test_return": best_grade.get("test_return"),
            "oos_final": best_grade.get("oos_final"),
            "soft_return": best_grade.get("soft_return"),
            "soft_final": best_grade.get("soft_final"),
            "kill_illusion": best_grade.get("kill_illusion"),
            "score": best_grade["score"],
            "score_raw": raw_score,
            "recent_exp": best_recent.get("exp"),
            "recent_n": best_recent.get("trades"),
            "recent_ok": best_recent_ok,
            "week_return": best_week.get("ret"),
            "targets_met": best_grade["targets_met"],
            # Deployed book (not best rejected candidate) — chart colors use these
            "strategy": live_knobs.get("strategy"),
            "timeframe": live_knobs.get("timeframe"),
            "leverage": live_knobs.get("leverage"),
            "deployed_leverage": live_knobs.get("leverage"),
            "adx_enter": live_knobs.get("adx_enter", knobs.get("adx_enter")),
            "volume_multiplier": live_knobs.get(
                "volume_multiplier", knobs.get("volume_multiplier")
            ),
            "risk_pct": live_knobs.get("risk_pct"),
            "mom_lookback": live_knobs.get("mom_lookback"),
            "mom_min_atr": live_knobs.get("mom_min_atr"),
            "require_squeeze": live_knobs.get("require_squeeze"),
            "require_rsi_veto": live_knobs.get("require_rsi_veto"),
            "require_btc_regime": live_knobs.get("require_btc_regime"),
            # Full candidate knobs so a blocked climb can be restored later
            "knobs": knobs,
            "deployed_knobs": live_knobs,
            "leverage_ladder": leverage_ladder,
            # Full TF×lev matrix for honest Recent searches (not by-lev 5m collapse).
            "tf_lev_matrix": list(tf_lev_matrix_rows or []),
            "matrix_tfs": sorted(
                {str(r.get("tf")) for r in (tf_lev_matrix_rows or []) if r.get("tf")}
            ),
        }
    )
    state["history"] = hist[-300:]

    def _finish_save(st: dict) -> None:
        # Preserve one-shot denser skip if watchdog armed mid-pass — never TF slim.
        keep_skip = bool(st.get("force_skip_denser"))
        keep_reason = st.get("hunt_recover_reason")
        st.clear()
        st.update(state)
        st.pop("force_matrix_slim", None)  # legacy TF-slim — never re-arm
        if keep_skip:
            st["force_skip_denser"] = True
            if keep_reason:
                st["hunt_recover_reason"] = keep_reason
        st.pop("pass_started_at", None)
        st.pop("pass_heartbeat_at", None)
        st["busy"] = False

    _update_autopilot_state(_finish_save)
    with contextlib.suppress(Exception):
        Persistence(str(DB_PATH), mode="paper").log_event(
            _utc_now(),
            "*",
            "autopilot",
            decision,
        )
    # Observability: durable hunt_pass row for funnel / ladder / timing trends.
    with contextlib.suppress(Exception):
        from moonshot.config_hash import config_hash as _obs_ch
        from moonshot.obs import (
            enrich_ladder_rows,
            hunt_funnel_metrics,
            kill_histogram,
            log_hunt_pass_experiment,
        )

        _cache_stats = dict(feat_cache.get("_stats") or {})
        _funnel = hunt_funnel_metrics(
            scout_in=int(funnel_counts.get("A_in") or len(_stage_a_cands)),
            scout_keep=int(funnel_counts.get("A_keep") or len(short_a)),
            confirm_b=int(funnel_counts.get("B") or len(stage_b)),
            confirm_hits=int(funnel_counts.get("C") or confirm_hits),
            confirm_rejects=confirm_rejects,
            matrix_rows=list(tf_lev_matrix_rows or []),
            finalist_ok=bool(decision.get("promote_tier") not in (None, "hold")),
            gates_pass=bool(decision.get("improved")),
            promote=bool(decision.get("improved")),
            cost_pass=int(funnel_counts.get("cost_pass") or 0),
            fold_pass=int(funnel_counts.get("fold_pass") or 0),
            promote_attempt=int(funnel_counts.get("promote_attempt") or 0),
            primary_metric="p25_fold_sortino",
        )
        _hist = kill_histogram(
            list(tf_lev_matrix_rows or []),
            confirm_rejects,
            material_why=material_why,
        )
        _ladder = enrich_ladder_rows(
            leverage_ladder,
            book_exp=float(best_o.expectancy_r) if best_o is not None else None,
            book_dd=float(best_o.max_drawdown) if best_o is not None else None,
            cost_2x_ok=None,
        )
        log_hunt_pass_experiment(
            generation=int(generation),
            config_hash=_obs_ch(best_cfg if improved else base),
            data_hash=None,
            funnel=_funnel,
            kill_hist=_hist,
            ladder=_ladder,
            timing=_timing,
            cache=_cache_stats,
            sealed_holdout=sealed_meta,
            dsr=dsr_meta,
            meta={
                "hunt_source": hunt_src,
                "venue": hunt_src,
                "generation": generation,
                "promote_reason": promote_reason,
                "promote_tier": decision.get("promote_tier"),
                "promote_provenance": promote_provenance,
                "research_promote": bool(decision.get("research_promote")),
                "improved": bool(decision.get("improved")),
                "material_why": material_why,
                "matrix_failed": bool(matrix_failed),
                "nearest_miss": decision.get("nearest_miss"),
                "hunt_exhausted": bool(state.get("hunt_exhausted")),
                "regime_reassess_needed": bool(state.get("regime_reassess_needed")),
                "primary_metric": "p25_fold_sortino",
                "funnel_conversion": dict(funnel_counts),
                "p25_sortino": (best_fold or {}).get("p25_sortino"),
                "fold_roi": best_roi,
                "cost_ok": bool(best_cost_ok),
            },
        )
    return decision
