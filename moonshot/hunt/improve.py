from __future__ import annotations

import logging
from dataclasses import replace
from typing import Optional

import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
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
from moonshot.hunt.constants import (
    HUNT_CHAMPION_STALE_PP,
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
    _live_feed,
    _load_autopilot_state,
    _patch_env,
    _save_autopilot_state,
    _utc_now,
    _week_delta_needed,
)
from moonshot.hunt.scoring import (
    PAPER_SILENCE_ROTATE_HOURS,
    RESEARCH_MIN_WEEK_EDGE,
    RESEARCH_PROMOTE_MIN_SEC,
    WEEK_DAYS,
    _beats,
    _candidate_cfgs,
    _cfg_fingerprint,
    _champion_from_history,
    _dedupe_cfgs,
    _discover_seed_cfgs,
    _eval_cfg,
    _eval_soft_oos,
    _forward_record,
    _freeze_risk,
    _grade,
    _held_streak,
    _hunt_search_mode,
    _is_climb_ok,
    _is_screenable,
    _is_valid_candidate,
    _leverage_ladder_cfgs,
    _material_research_edge,
    _paper_idle_hours,
    _prepare_cached,
    _rank_key,
    _recent_eval,
    _week_eval,
    _wf_eval,
    _wf_msg,
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
    mode = "HUNT" if hunt else "TUNE"
    state = _load_autopilot_state()
    held = _held_streak(state)
    search_mode = _hunt_search_mode(generation, held, hunt)
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
    # F1: freeze last WEEK_DAYS — fit/IS/OOS never train on holdout week
    holdout_start = pd.Timestamp(t1) - pd.Timedelta(days=int(WEEK_DAYS))
    fit_end = holdout_start
    if fit_end <= t0 + pd.Timedelta(days=14):
        log.warning("holdout window too tight (t0=%s t1=%s) — widening fit", t0, t1)
        fit_end = t0 + (t1 - t0) * 0.85
        holdout_start = fit_end
    split = t0 + (fit_end - t0) * (1 - oos_frac)
    log.info(
        "holdout week %s → %s (fit ends %s); split %s",
        holdout_start.date(),
        pd.Timestamp(t1).date(),
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
    base_recent = _recent_eval(base_data, base, end=t1)
    base_week = _week_eval(base_data, base, end=t1)  # holdout last 7d
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
    champion = _champion_from_history(
        list(state.get("history") or []),
        state.get("champion") if isinstance(state.get("champion"), dict) else None,
    )
    if champion:
        state["champion"] = champion

    best_cfg = base
    best_i, best_o, best_soft = base_i, base_o, base_soft
    best_wf = base_wf
    best_recent = dict(base_recent)
    best_week = dict(base_week)
    best_grade = dict(base_grade)
    (not wf_enabled) or bool(base_wf and base_wf.get("ok"))
    promote_reason = "baseline"

    def _consider(cand_cfg: Config, ci_, co_, cs_, label_: str = "") -> bool:
        """Promote cand in-memory if it beats incumbent on fit rank or holdout week."""
        nonlocal best_cfg, best_i, best_o, best_soft, best_wf, best_grade
        nonlocal best_recent, best_week, promote_reason
        # ≥10x must arrive with soft OOS already evaluated
        if int(cand_cfg.leverage) >= 10 and cs_ is None:
            return False
        best_ok = (not wf_enabled) or bool(best_wf and best_wf.get("ok"))
        best_recent_ok = bool(best_recent.get("ok"))
        cwf = _wf_for(cand_cfg)
        cand_ok = (not wf_enabled) or bool(cwf and cwf.get("ok"))
        if wf_enabled and not cand_ok:
            return False
        # Always score week/recent on the multi-coin confirm book (correct TF).
        c_raw = _raw_for_tf(str(cand_cfg.timeframe)) or raw
        cdata = _prepare_cached(c_raw, cand_cfg, feat_cache)
        crecent = _recent_eval(cdata, cand_cfg, end=t1)
        cweek = _week_eval(cdata, cand_cfg, end=t1)  # holdout only (last 7d)
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
        # Any better holdout week while active — do NOT require week.ok (that was
        # a silent veto while every book is underwater on the same regime).
        week_edge = (
            week_active
            and float(cweek.get("ret") or 0) > float(best_week.get("ret") or 0) + min_delta
        )
        # Less-bad OOS / soft when incumbent is losing — lets hunt climb before week flips
        losing_book = float(best_o.total_return) <= 0 or float(best_o.expectancy_r) <= 0
        oos_edge = (
            losing_book
            and float(co_.total_return) > float(best_o.total_return) + 0.005
            and float(co_.expectancy_r) >= float(best_o.expectancy_r) - 0.05
        )
        soft_edge = (
            losing_book
            and cs_ is not None
            and best_soft is not None
            and float(cs_.total_return) > float(best_soft.total_return) + 0.005
        )
        recent_edge = (
            float(crecent.get("ret") or 0) > float(best_recent.get("ret") or 0) + 0.01
            and float(crecent.get("exp") or 0) >= float(best_recent.get("exp") or 0) - 0.02
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
        # PRIMARY floor: never replace in-memory best with a worse holdout week.
        # OOS/soft "rescues" were demoting +16% champions down to -21% on Live.
        cand_week_ret = float(cweek.get("ret") or 0)
        best_week_ret = float(best_week.get("ret") or 0)
        if week_active and cand_week_ret < best_week_ret - 0.001:
            return False
        reason = label_ or (
            "week_ladder"
            if week_edge
            else "recent_rescue"
            if (recent_edge or recent_relative)
            else "rank_beat"
            if cand_beats
            else "rescue"
        )
        best_cfg, best_i, best_o, best_soft, best_wf = cand_cfg, ci_, co_, cs_, cwf
        best_recent = crecent
        best_week = cweek
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
        promote_reason = reason
        return True

    def _knobs_to_cfg(kn: dict) -> Config:
        return _freeze_risk(
            base,
            replace(
                base,
                strategy=str(kn.get("strategy", base.strategy)),
                timeframe=str(kn.get("timeframe", base.timeframe)),
                adx_enter=float(kn["adx_enter"]),
                adx_exit=float(kn.get("adx_exit", float(kn["adx_enter"]) - 6)),
                volume_multiplier=float(kn["volume_multiplier"]),
                breakout_buffer_atr=float(kn.get("breakout_buffer_atr", base.breakout_buffer_atr)),
                breakout_lookback=int(kn.get("breakout_lookback", base.breakout_lookback)),
                trail_atr_multiple=float(kn.get("trail_atr", base.trail_atr_multiple)),
                initial_stop_atr=float(kn.get("initial_stop_atr", base.initial_stop_atr)),
                cooldown_bars=int(kn.get("cooldown_bars", base.cooldown_bars)),
                require_rising_adx=bool(kn.get("require_rising_adx", base.require_rising_adx)),
                require_squeeze=bool(kn.get("require_squeeze", base.require_squeeze)),
                require_rsi_veto=bool(kn.get("require_rsi_veto", base.require_rsi_veto)),
                require_btc_regime=bool(kn.get("require_btc_regime", base.require_btc_regime)),
                rsi_long_max=float(kn.get("rsi_long_max", base.rsi_long_max)),
                rsi_short_min=float(kn.get("rsi_short_min", base.rsi_short_min)),
                squeeze_lookback=int(kn.get("squeeze_lookback", base.squeeze_lookback)),
                enable_partial=bool(kn.get("enable_partial", base.enable_partial)),
                partial_at_r=float(kn.get("partial_at_r", base.partial_at_r)),
                partial_pct=float(kn.get("partial_pct", base.partial_pct)),
                enable_time_stop=bool(kn.get("enable_time_stop", base.enable_time_stop)),
                time_stop_bars=int(kn.get("time_stop_bars", base.time_stop_bars)),
                time_stop_min_r=float(kn.get("time_stop_min_r", base.time_stop_min_r)),
                trail_atr_loser=float(kn.get("trail_atr_loser", base.trail_atr_loser)),
                use_asymmetric_trail=bool(
                    kn.get("use_asymmetric_trail", base.use_asymmetric_trail)
                ),
                risk_pct=float(kn.get("risk_pct", base.risk_pct)),
                initial_margin_pct=float(kn.get("initial_margin_pct", base.initial_margin_pct)),
                aggressive_margin_pct=float(
                    kn.get("aggressive_margin_pct", base.aggressive_margin_pct)
                ),
                max_daily_loss_pct=float(kn.get("max_daily_loss_pct", base.max_daily_loss_pct)),
                max_peak_drawdown_pct=float(
                    kn.get("max_peak_drawdown_pct", base.max_peak_drawdown_pct)
                ),
                mom_lookback=int(kn.get("mom_lookback", base.mom_lookback)),
                mom_min_atr=float(kn.get("mom_min_atr", base.mom_min_atr)),
                leverage=int(kn.get("leverage", base.leverage)),
                max_account_leverage=float(
                    kn.get("max_account_leverage", base.max_account_leverage)
                ),
                max_net_leverage=float(kn.get("max_net_leverage", base.max_net_leverage)),
                max_open_positions=int(kn.get("max_open_positions", base.max_open_positions)),
                h4_ema_fast=int(kn.get("h4_ema_fast", base.h4_ema_fast)),
                h4_ema_slow=int(kn.get("h4_ema_slow", base.h4_ema_slow)),
                ema_entry_period=int(kn.get("ema_entry_period", base.ema_entry_period)),
                pullback_lookback=int(kn.get("pullback_lookback", base.pullback_lookback)),
                btc_adx_min=float(kn.get("btc_adx_min", base.btc_adx_min)),
                breadth_min=float(kn.get("breadth_min", base.breadth_min)),
            ),
        )

    # Seed from elite — allow cross-family in breakout mode
    if elite and elite.get("knobs"):
        try:
            kn = elite["knobs"]
            elite_strat = str(kn.get("strategy", "") or "").strip().lower()
            if (
                elite_strat
                and elite_strat != str(base.strategy).strip().lower()
                and search_mode != "breakout"
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

    polish_base = best_cfg
    tested = 0
    soft_runs = 0
    soft_budget = int(HUNT_SOFT_BUDGET.get(search_mode, 6))
    scout_top = int(HUNT_SCOUT_TOP)
    if held >= int(HUNT_STUCK_HELD):
        soft_budget = max(soft_budget, 20)
        scout_top = max(scout_top, 24)
    hard_pool: list = []
    thin_pool: list = []
    cand_list = _candidate_cfgs(
        polish_base,
        generation=generation,
        hunt=hunt,
        search_mode=search_mode,
    )
    cand_list = list(cand_list) + _discover_seed_cfgs(polish_base, generation, held)
    # Density seeds every pass — ADX28/squeeze books starve week samples
    denser = [
        replace(
            polish_base,
            strategy=str(st),
            leverage=int(lev),
            timeframe="15m",
            adx_enter=float(adx),
            adx_exit=float(max(10.0, adx - 5)),
            require_squeeze=False,
            require_rising_adx=False,
            cooldown_bars=int(cd),
            volume_multiplier=1.1,
            mom_lookback=8,
            mom_min_atr=0.35,
            max_account_leverage=float(max(lev, float(polish_base.max_account_leverage))),
            max_net_leverage=float(max(lev, float(polish_base.max_net_leverage))),
        )
        for st in HUNT_STRATEGIES
        for lev in (5, 7, 10)
        for adx in (14.0, 16.0, 18.0)
        for cd in (0, 1, 2)
    ]
    cand_list = _dedupe_cfgs(polish_base, list(cand_list) + denser)

    # ---- Stage 1: scout hard IS/OOS + week on BTC only (rank by week first) ----
    for cand in cand_list:
        tested += 1
        try:
            use_raw = _scout_raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            ci, co = _eval_cfg(data, cand, split, end=fit_end)
            cweek_s = _week_eval(data, cand, end=t1)
        except Exception as e:
            log.warning("scout candidate failed: %s", e)
            continue
        # Sort key: holdout week first (what we climb on), then fit rank
        row = (
            float(cweek_s.get("ret") or -999),
            int(cweek_s.get("trades") or 0),
            _rank_key(ci, co, None),
            cand,
            ci,
            co,
        )
        if _is_screenable(ci, co):
            hard_pool.append(row)
        else:
            thin_pool.append(row)
    if not hard_pool and thin_pool:
        log.warning("scout: no screenable books — promoting thin pool to confirm")
        hard_pool = thin_pool
    hard_pool.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    shortlist = hard_pool[: max(scout_top, soft_budget)]
    top_w = float(shortlist[0][0]) if shortlist else float("nan")
    log.info(
        "funnel scout: tried %d on %s · %d screenable · %d → confirm (best scout week %+0.f%%)",
        tested,
        scout_symbols[0].split("/")[0] if scout_symbols else "?",
        len(hard_pool),
        len(shortlist),
        top_w * 100 if top_w == top_w else 0,
    )

    # ---- Stage 2: multi-coin soft + week; only survivors may enter _consider ----
    leverage_ladder: list = []
    confirm_hits = 0
    confirm_rejects = {"climb": 0, "consider": 0, "err": 0}
    # Screenable confirm books kept for silence_rotate even when climb/_consider lose
    # to a high holdout-week champion (Live can sit idle while hunt HOLDs forever).
    rotate_pool: list = []
    seen_soft = set()
    for _wr, _wn, _rk, cand, _sci, _sco in shortlist:
        if soft_runs >= soft_budget:
            break
        ck = id(cand)
        if ck in seen_soft:
            continue
        seen_soft.add(ck)
        try:
            use_raw = _raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            ci, co = _eval_cfg(data, cand, split, end=fit_end)
            cs = _eval_soft_oos(data, cand, split, end=fit_end)
            soft_runs += 1
        except Exception as e:
            log.warning("confirm soft failed: %s", e)
            confirm_rejects["err"] += 1
            continue
        if _is_screenable(ci, co):
            try:
                cweek_r = _week_eval(data, cand, end=t1)
                rotate_pool.append(
                    (
                        float(cweek_r.get("ret") or -999),
                        int(cweek_r.get("trades") or 0),
                        int(cweek_r.get("soft_trades") or 0),
                        _rank_key(ci, co, cs),
                        cand,
                        ci,
                        co,
                        cs,
                        cweek_r,
                    )
                )
            except Exception:
                pass
        if not _is_climb_ok(ci, co, cs, base_i, base_o):
            confirm_rejects["climb"] += 1
            continue
        if _consider(cand, ci, co, cs, "confirm"):
            confirm_hits += 1
        else:
            confirm_rejects["consider"] += 1

    log.info(
        "funnel confirm: soft-checked %d/%d · climbs accepted=%d · rejects %s",
        soft_runs,
        soft_budget,
        confirm_hits,
        confirm_rejects,
    )

    # ---- Stage 3: leverage bake-off on multi-coin (always try _consider) ----
    ladder_src = best_cfg if (confirm_hits > 0 or best_cfg is not base) else base
    ladder_rows: list[dict] = []
    run_ladder_consider = True
    for cand in _leverage_ladder_cfgs(ladder_src):
        try:
            cand = _freeze_risk(base, cand)
            use_raw = _raw_for_tf(cand.timeframe)
            if not use_raw:
                continue
            data = _prepare_cached(use_raw, cand, feat_cache)
            ci, co = _eval_cfg(data, cand, split, end=fit_end)
            cweek = _week_eval(data, cand, end=t1)
            cs = None
            if run_ladder_consider and _is_screenable(ci, co):
                cs = _eval_soft_oos(data, cand, split, end=fit_end)
                if _is_climb_ok(ci, co, cs, base_i, base_o):
                    _consider(cand, ci, co, cs, "lev_ladder")
            grade_l = _grade(
                ci,
                co,
                cs,
                None,
                wf_required=False,
                fwd=fwd,
                fwd_required=False,
                week=cweek,
            )
            ladder_rows.append(
                {
                    "leverage": int(cand.leverage),
                    "timeframe": str(cand.timeframe),
                    "strategy": str(cand.strategy),
                    "week_ret": float(cweek.get("ret") or 0),
                    "week_trades": int(cweek.get("trades") or 0),
                    "soft_ret": float(cs.total_return) if cs is not None else None,
                    "test_ret": float(co.total_return),
                    "test_n": int(co.trades),
                    "win": float(co.win_rate),
                    "score": int(grade_l.get("score") or 0),
                    "expectancy": float(co.expectancy_r),
                }
            )
        except Exception as e:
            log.warning("leverage ladder %sx failed: %s", getattr(cand, "leverage", "?"), e)
    by_lev: dict[int, dict] = {}
    for row in ladder_rows:
        lev = int(row["leverage"])
        prev = by_lev.get(lev)
        if prev is None or float(row["week_ret"]) > float(prev["week_ret"]):
            by_lev[lev] = row
    leverage_ladder = [by_lev[k] for k in sorted(by_lev)]
    if leverage_ladder:
        log.info(
            "leverage ladder%s: %s",
            "" if run_ladder_consider else " (chart)",
            " · ".join(
                f"{r['leverage']}x@{r['timeframe']} week={r['week_ret']:+.0%} "
                f"n={r['week_trades']} score={r['score']}"
                for r in leverage_ladder
            ),
        )

    # ---- Silence rotate: paper idle → deploy a different tradeable book ----
    # Auto-improve must *test* approaches on paper, not only backtest against a
    # frozen high holdout-week champion that produces zero live setups.
    silence_rotate = False
    idle_h = float(_paper_idle_hours())
    week_min = int(HUNT_TARGETS.get("week_min_trades", 4))
    if (
        best_cfg is base
        and idle_h >= float(PAPER_SILENCE_ROTATE_HOURS)
        and held >= 3
        and rotate_pool
    ):
        base_fp = _cfg_fingerprint(base)
        # Prefer denser week activity, then week return, then fit rank.
        rotate_pool.sort(key=lambda r: (r[1] + r[2], r[0], r[3]), reverse=True)
        for _wr, wn, wsn, _rk, cand, ci, co, cs, cweek in rotate_pool:
            if _cfg_fingerprint(cand) == base_fp:
                continue
            if int(wn) < week_min and int(wsn) < week_min:
                continue
            if not _is_screenable(ci, co):
                continue
            cwf = _wf_for(cand) if wf_enabled else None
            if wf_enabled and not (cwf and cwf.get("ok")):
                continue
            best_cfg, best_i, best_o, best_soft, best_wf = cand, ci, co, cs, cwf
            best_week = cweek
            best_recent = _recent_eval(
                _prepare_cached(_raw_for_tf(str(cand.timeframe)) or raw, cand, feat_cache),
                cand,
                end=t1,
            )
            best_grade = _grade(
                ci,
                co,
                cs,
                cwf,
                wf_required=True,
                fwd=fwd,
                fwd_required=True,
                week=cweek,
            )
            best_grade["recent"] = best_recent
            best_grade["week"] = cweek
            promote_reason = "silence_rotate"
            silence_rotate = True
            log.info(
                "silence rotate · paper idle %.1fh · deploying %s %sx@%s week=%+.0f%% n=%d "
                "(champion holdout no longer freezes Live)",
                idle_h,
                cand.strategy,
                int(cand.leverage),
                cand.timeframe,
                float(cweek.get("ret") or 0) * 100,
                int(wn),
            )
            break
        if not silence_rotate:
            log.info(
                "silence rotate skipped · idle %.1fh but no alternate screenable book in pool (%d)",
                idle_h,
                len(rotate_pool),
            )
    elif best_cfg is base and idle_h >= float(PAPER_SILENCE_ROTATE_HOURS):
        log.info(
            "silence rotate armed · paper idle %.1fh · held=%d pool=%d (need held>=3 + pool)",
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
            champ_live = float(_week_eval(ch_data, ch_cfg, end=t1).get("ret") or 0)
        except Exception as e:
            log.warning("champion live reeval failed: %s", e)
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
    # Stuck overnight: never require better than beating the deployed book
    if held >= int(HUNT_STUCK_HELD):
        champ_week_floor = min(champ_week_floor, base_week_ret + 0.001)
        log.info(
            "stuck ×%d — promote floor eased to deployed week %+0.f%%",
            held,
            champ_week_floor * 100,
        )

    best_week_ret_now = float(best_week.get("ret") or 0)
    # Never research-promote below the (possibly stale-adjusted) champion floor
    # (silence_rotate is the escape hatch when Live is idle under that floor).
    if not silence_rotate and best_cfg is not base and best_week_ret_now + 0.001 < champ_week_floor:
        material_ok = False
        material_why = (
            f"below champion week {champ_week_floor:+.0%} (cand {best_week_ret_now:+.0%})"
        )

    research_improved = (
        best_cfg is not base
        and best_week_active
        and promote_cooled
        and (
            # Idle paper must rotate even if the alternate fails kill-illusion /
            # climb gates — otherwise a silent champion freezes Live forever.
            silence_rotate
            or (
                wf_gate_ok
                and not best_grade.get("kill_illusion")
                and _is_climb_ok(best_i, best_o, best_soft, base_i, base_o)
                and material_ok
                and best_week_ret_now + 0.001 >= champ_week_floor
            )
        )
    )
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
    # Skip for silence_rotate — that path exists to break Live free now.
    if research_improved and len(full_symbols) > len(symbols) and not silence_rotate:
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
            log.warning("finalist full-book recheck failed: %s — keeping search promote", e)

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
            if mexc_ret <= 0.0 and spot_ret > 0.0:
                research_improved = False
                material_why = (
                    f"MEXC holdout {mexc_ret:+.0%} ≤ 0 while research "
                    f"({hunt_src}) {spot_ret:+.0%} — refuse .env patch"
                )
                log.warning("transfer gate blocked promote: %s", material_why)
            else:
                # Prefer MEXC number for Live messaging when transfer clears.
                best_week = {
                    **best_week,
                    "ret": mexc_week.get("ret"),
                    "soft_ret": mexc_week.get("soft_ret"),
                    "final": mexc_week.get("final"),
                    "trades": mexc_week.get("trades"),
                    "soft_trades": mexc_week.get("soft_trades"),
                    "span": mexc_week.get("span"),
                    "transfer_ok": True,
                }
                best_grade["week"] = best_week
                best_grade["week_return"] = best_week.get("ret")
        except Exception as e:
            research_improved = False
            material_why = f"MEXC transfer check failed: {e}"
            log.warning("transfer gate blocked promote: %s", material_why)
            transfer_gap = {
                "research_source": str(hunt_src),
                "research_ret": spot_ret,
                "error": str(e),
            }

    improved = bool(research_improved)  # research tier → .env
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
        "material_why": material_why,
        "promote_cooled": bool(promote_cooled),
        "fail_checks": fail_checks,
        "tested": tested,
        "soft_runs": soft_runs,
        "search_mode": search_mode,
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
    if improved:
        _patch_env(_cfg_to_env_updates(best_cfg))
        # Do NOT reset forward_since on research promote — that made FWD_MIN_TRADES
        # unreachable across climbs. Only elite/confirmed resets the clock below.
        state["research_promote_at"] = decision["ts"]
        state.pop("pending_promote", None)
        decision["message"] = (
            f"Research promote ({promote_reason}) · {best_cfg.strategy} "
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
    else:
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

    # Persist progress; elite only with forward proof (F3)
    state["last_improve"] = decision
    state["last_improve_at"] = decision["ts"]
    state["generation"] = generation
    state["improve_passes"] = int(state.get("improve_passes", 0)) + 1
    state["busy"] = False
    # Live must show the DEPLOYED book. On HOLD, never paint an unpromoted
    # candidate week (that is how +16% became -21% on the dashboard).
    state["grade"] = best_grade if improved else base_grade
    state["targets"] = HUNT_TARGETS
    state["hunt"] = hunt
    state["promote_tier"] = promote_tier
    state["leverage_ladder"] = leverage_ladder
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

    # Champion floor: only raise, never lower — except silence_rotate, which must
    # replace a ghost high-week champion that left paper with zero live setups.
    if improved:
        new_week = float(best_week.get("ret") or 0)
        old_week = float((champion or {}).get("week_ret") or -999)
        if silence_rotate or new_week >= old_week - 0.0005:
            state["champion"] = {
                "ts": decision["ts"],
                "generation": generation,
                "week_ret": new_week,
                "score": raw_score,
                "knobs": knobs,
                "week": best_week,
                **(
                    {"week_ret_peak": max(old_week, new_week), "silence_rotate": True}
                    if silence_rotate
                    else {}
                ),
            }
        elif champion:
            state["champion"] = champion
    elif champion:
        state["champion"] = champion
    hist = list(state.get("history") or [])
    hist.append(
        {
            "ts": decision["ts"],
            "generation": generation,
            "improved": improved,
            "promote_tier": decision.get("promote_tier"),
            "promote_reason": promote_reason,
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
            "strategy": knobs.get("strategy"),
            "timeframe": knobs.get("timeframe"),
            "leverage": knobs.get("leverage"),
            "adx_enter": knobs["adx_enter"],
            "volume_multiplier": knobs["volume_multiplier"],
            "risk_pct": knobs.get("risk_pct"),
            "mom_lookback": knobs.get("mom_lookback"),
            "mom_min_atr": knobs.get("mom_min_atr"),
            "require_squeeze": knobs.get("require_squeeze"),
            "require_rsi_veto": knobs.get("require_rsi_veto"),
            "require_btc_regime": knobs.get("require_btc_regime"),
            # Full knobs so a blocked climb can be restored later
            "knobs": knobs,
            "leverage_ladder": leverage_ladder,
        }
    )
    state["history"] = hist[-300:]
    _save_autopilot_state(state)
    with contextlib.suppress(Exception):
        Persistence(str(DB_PATH), mode="paper").log_event(
            _utc_now(),
            "*",
            "autopilot",
            decision,
        )
    return decision
