from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Optional


log = logging.getLogger("moonshot")

from moonshot.types import (
    ConfigError,
)
from moonshot.data import (
    clear_stale_live_caches,
)
from moonshot.symbols import (
    HUNT_SYMBOLS,
    PAPER_SYMBOLS,
)
from moonshot.live import (
    _live_feed,
    _utc_now,
)
from moonshot.hunt.constants import (
    HUNT_CHAMP_TRAIL_PP,
    HUNT_TARGETS,
    _env_book_fingerprint,
    _load_autopilot_state,
    _pct_of_risk_msg,
    _save_autopilot_state,
    _update_autopilot_state,
    autopilot_frozen,
)
from moonshot.hunt.scoring import (
    POST_PROMOTE_HUNT_PAUSE_SEC,
)
from moonshot.hunt.improve import (
    improve_once,
)
from moonshot.hunt.health import (
    _arm_skip_denser,
    hunt_pass_health,
    recent_oom_hint,
)

from moonshot.paths import (
    ROOT,
)

# How often the champ/.env/apply watchdog re-checks (seconds).
SCOREBOARD_WATCHDOG_SEC = 180
PAPER_BOOK_ASSERT_SEC = 45


def _run_scoreboard_watchdog() -> dict:
    """CRITICAL: champ vs .env strand + apply-fail + stuck/no-finish pass recovery."""
    st = _load_autopilot_state()
    env_fp = _env_book_fingerprint()
    ch = st.get("champion") or {}
    ck = ch.get("knobs") or {}
    champ_fp = (
        str(ck.get("strategy") or ""),
        str(ck.get("timeframe") or ""),
        str(ck.get("leverage") or ""),
    )
    alerts: list[str] = []
    critical = False

    # Pass finish liveness (OOM-restart loops look "busy" forever without this).
    pass_h = hunt_pass_health(st, process_alive=True, mode=str(st.get("mode") or ""))
    for msg in pass_h.get("failures") or []:
        critical = True
        alerts.append(msg)
        log.critical("SCOREBOARD WATCHDOG · %s", msg)
        _live_feed("scoreboard_critical", msg)
        reason = "no_finish" if "HUNT_NO_FINISH" in msg else "pass_stuck"
        if recent_oom_hint():
            reason = "oom"
        _arm_skip_denser(st, reason)
        log.warning(
            "SCOREBOARD WATCHDOG · force_skip_denser=1 reason=%s · matrix stays FULL",
            reason,
        )

    champ_week = ch.get("week_ret_live")
    if champ_week is None:
        champ_week = ch.get("week_ret")
    try:
        champ_week_f = float(champ_week) if champ_week is not None else None
    except (TypeError, ValueError):
        champ_week_f = None

    live_dep = st.get("live_deployed") or {}
    try:
        live_week = (
            float(live_dep.get("week_ret")) if live_dep.get("week_ret") is not None else None
        )
    except (TypeError, ValueError):
        live_week = None
    if live_week is None:
        last = st.get("last_improve") or {}
        try:
            live_week = (
                float(last.get("best", {}).get("week_return"))
                if (last.get("best") or {}).get("week_return") is not None
                else None
            )
        except (TypeError, ValueError, AttributeError):
            live_week = None

    if champ_fp[0] and champ_fp == env_fp and st.get("pending_champion_rescue"):
        # Already deployed on champion S/T/L — stuck pending only blocks heavy hunt.
        st["pending_champion_rescue"] = False
        st["champ_trail_streak"] = 0
        log.info(
            "SCOREBOARD WATCHDOG · cleared stuck pending_champion_rescue "
            "(already on champ %s %sx@%s)",
            champ_fp[0],
            champ_fp[2],
            champ_fp[1],
        )
        _live_feed(
            "rescue_cleared",
            f"cleared stuck pending_champion_rescue — already on {champ_fp[0]} "
            f"{champ_fp[2]}x@{champ_fp[1]}",
        )

    if champ_fp[0] and champ_fp != env_fp and champ_week_f is not None:
        trail = champ_week_f - float(live_week if live_week is not None else -1.0)
        if trail >= float(HUNT_CHAMP_TRAIL_PP) and champ_week_f >= 0.10:
            critical = True
            msg = (
                f"STRANDED: champ {champ_fp[0]} {champ_fp[2]}x@{champ_fp[1]} "
                f"week={champ_week_f:+.0%} trails Live by {trail:+.0%}pp but .env is "
                f"{env_fp[0]} {env_fp[2]}x@{env_fp[1]}"
            )
            alerts.append(msg)
            if not st.get("pending_champion_rescue"):
                st["pending_champion_rescue"] = True
            log.critical("SCOREBOARD WATCHDOG · %s", msg)
            _live_feed("scoreboard_critical", msg, trail_pp=trail)

    hist = list(st.get("history") or [])
    if hist:
        last_h = hist[-1]
        if last_h.get("improved"):
            hs = str(last_h.get("strategy") or "")
            ht = str(last_h.get("timeframe") or "")
            hl = str(last_h.get("leverage") or "")
            if hs and ht and hl and (hs, ht, hl) != env_fp:
                critical = True
                msg = (
                    f"APPLY-FAIL: history improved {hs} {hl}x@{ht} "
                    f"reason={last_h.get('promote_reason')} but .env={env_fp}"
                )
                alerts.append(msg)
                log.critical("SCOREBOARD WATCHDOG · %s", msg)
                _live_feed("scoreboard_critical", msg)

    # Post-restart paper fingerprint: expected_paper_book vs process env
    expected = st.get("expected_paper_book") or {}
    if expected.get("strategy"):
        exp_fp = (
            str(expected.get("strategy") or ""),
            str(expected.get("timeframe") or ""),
            str(expected.get("leverage") or ""),
        )
        if exp_fp != env_fp:
            critical = True
            msg = f"paper expected book {exp_fp} != .env {env_fp}"
            alerts.append(msg)
            log.critical("SCOREBOARD WATCHDOG · %s", msg)

    wd = {
        "ts": _utc_now().isoformat(),
        "critical": critical,
        "alerts": alerts[:8],
        "env": {"strategy": env_fp[0], "timeframe": env_fp[1], "leverage": env_fp[2]},
        "champ": {
            "strategy": champ_fp[0],
            "timeframe": champ_fp[1],
            "leverage": champ_fp[2],
            "week_ret": champ_week_f,
        },
    }
    want_skip = bool(st.get("force_skip_denser"))
    want_reason = st.get("hunt_recover_reason")
    want_pending = st.get("pending_champion_rescue")
    want_trail = st.get("champ_trail_streak")

    def _patch(latest: dict) -> None:
        latest["scoreboard_watchdog"] = wd
        # Never keep legacy TF-slim flag.
        latest.pop("force_matrix_slim", None)
        if "pending_champion_rescue" in st:
            latest["pending_champion_rescue"] = want_pending
        if "champ_trail_streak" in st:
            latest["champ_trail_streak"] = want_trail
        if want_skip:
            latest["force_skip_denser"] = True
            if want_reason:
                latest["hunt_recover_reason"] = want_reason

    latest = _update_autopilot_state(_patch)
    return latest.get("scoreboard_watchdog") or wd


class PaperSupervisor:
    """Keep `python -m moonshot paper` alive as a child process."""

    def __init__(self, symbols: Optional[list] = None):
        self.symbols = symbols
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._stop_unlocked()
            self._reload_env_into_process()
            cmd = [sys.executable, "-m", "moonshot", "paper"]
            if self.symbols:
                cmd += ["--symbols", *self.symbols]
            log.info(
                "starting paper child: %s (book %s %sx@%s)",
                " ".join(cmd),
                os.environ.get("STRATEGY"),
                os.environ.get("LEVERAGE"),
                os.environ.get("TIMEFRAME"),
            )
            self.proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT),
                env={**os.environ, "LIVE_TRADING_ENABLED": "0"},
            )

    def stop(self) -> None:
        with self._lock:
            self._stop_unlocked()

    def _stop_unlocked(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    @staticmethod
    def _reload_env_into_process() -> None:
        """Child must inherit the .env book hunt just wrote — not a stale parent env."""
        try:
            from dotenv import load_dotenv

            from moonshot.hunt.constants import (
                _clear_paper_bleed_circuit,
                _env_book_fingerprint,
                _load_autopilot_state,
                _save_autopilot_state,
                _utc_now,
            )
            from moonshot.paths import ENV_PATH

            load_dotenv(ENV_PATH, override=True)
            fp = _env_book_fingerprint()
            st = _load_autopilot_state()
            prev = st.get("deployed_book") or {}
            prev_fp = (
                str(prev.get("strategy") or ""),
                str(prev.get("timeframe") or ""),
                str(prev.get("leverage") or ""),
            )
            if not fp[0]:
                return
            book_changed = fp != prev_fp
            stale_halt = bool(st.get("paper_bleed_halt")) and not st.get("paper_bleed_window_start")
            if book_changed or stale_halt:
                _clear_paper_bleed_circuit(st, ts=_utc_now().isoformat())
                _save_autopilot_state(st)
                log.info(
                    "paper bleed circuit cleared · book %s → %s · changed=%s stale_halt=%s",
                    prev_fp,
                    fp,
                    book_changed,
                    stale_halt,
                )
            elif not prev.get("strategy"):
                st["deployed_book"] = {
                    "strategy": fp[0],
                    "timeframe": fp[1],
                    "leverage": fp[2],
                    "at": _utc_now().isoformat(),
                }
                _save_autopilot_state(st)
        except Exception as e:
            log.warning("paper env reload failed: %s", e)

    def ensure(self) -> None:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                code = None if self.proc is None else self.proc.returncode
                if code not in (None, 0):
                    log.warning("paper child exited with code %s — restarting", code)
                self._reload_env_into_process()
                cmd = [sys.executable, "-m", "moonshot", "paper"]
                if self.symbols:
                    cmd += ["--symbols", *self.symbols]
                log.info(
                    "starting paper child: %s (book %s %sx@%s)",
                    " ".join(cmd),
                    os.environ.get("STRATEGY"),
                    os.environ.get("LEVERAGE"),
                    os.environ.get("TIMEFRAME"),
                )
                self.proc = subprocess.Popen(
                    cmd,
                    cwd=str(ROOT),
                    env={**os.environ, "LIVE_TRADING_ENABLED": "0"},
                )


def cmd_autopilot(args: argparse.Namespace) -> None:
    """
    Leave this running:
      - paper trading stays up
      - HUNT mode: wide search until near-perfect score targets are met
      - only keeps configs that are profitable on BOTH history halves
      - never turns on real-money trading
    """
    try:
        clear_stale_live_caches()
    except Exception as e:
        log.warning("stale cache clear on autopilot start: %s", e)
    # launchd KeepAlive does not maintain logs/autopilot.pid — write it ourselves
    try:
        pid_path = ROOT / "logs" / "autopilot.pid"
        pid_path.parent.mkdir(parents=True, exist_ok=True)
        pid_path.write_text(str(os.getpid()))
    except OSError as e:
        log.warning("could not write autopilot.pid: %s", e)
    if os.getenv("LIVE_TRADING_ENABLED") == "1":
        raise ConfigError(
            "Autopilot refuses to start while LIVE_TRADING_ENABLED=1. "
            "Set it to 0 — autopilot is practice-only."
        )
    if autopilot_frozen():
        log.info("AUTOPILOT_FROZEN=1: promotions and paper restarts from hunt suppressed")
    else:
        log.info("AUTOPILOT_FROZEN=0: promotions enabled")
    # Focus hunt + paper on BTC until edge is proven; --symbols overrides.
    symbols = args.symbols or list(HUNT_SYMBOLS)
    paper_symbols = args.symbols or list(PAPER_SYMBOLS)
    continuous = not bool(getattr(args, "daily", False))
    cooldown = max(5, int(getattr(args, "cooldown_seconds", 15)))
    every_h = float(getattr(args, "every_hours", 0) or 0)
    hunt = not bool(getattr(args, "tune_only", False))

    state = _load_autopilot_state()
    # Clamp corrupt generation jumps (e.g. 1559 → 9999) back to improve_passes.
    try:
        gen_now = int(state.get("generation") or 0)
        passes = int(state.get("improve_passes") or 0)
        if passes > 0 and gen_now > passes + 50:
            log.warning(
                "generation %s >> improve_passes %s — clamping to %s",
                gen_now,
                passes,
                passes,
            )
            state["generation"] = passes
            _live_feed(
                "gen_clamp",
                f"clamped corrupt generation {gen_now} → {passes}",
                generation=passes,
            )
    except (TypeError, ValueError):
        pass
    state["started_at"] = _utc_now().isoformat()
    state["mode"] = "hunt" if hunt else ("continuous" if continuous else f"every_{every_h}h")
    state["cooldown_seconds"] = cooldown
    state["targets"] = HUNT_TARGETS
    state["hunt"] = hunt
    state["hunt_symbols"] = list(symbols)
    state["paper_symbols"] = list(paper_symbols)
    state["busy"] = False
    # Kill legacy TF-slim lock forever — denser skip is one-shot only.
    state.pop("force_matrix_slim", None)
    _save_autopilot_state(state)

    if args.improve_only:
        gen = int(state.get("generation", 0))
        decision = improve_once(
            symbols,
            bars=args.bars,
            oos_frac=args.oos,
            refresh=args.refresh,
            generation=gen,
            hunt=hunt,
        )
        print(decision["message"])
        print(json.dumps({"best": decision["best"], "grade": decision.get("grade")}, indent=2))
        return

    supervisor = PaperSupervisor(paper_symbols)
    supervisor.start()
    stop = threading.Event()
    restart_paper = threading.Event()
    gen_box = {"n": int(state.get("generation", 0))}

    def improve_worker() -> None:
        while not stop.is_set():
            met = False
            just_promoted = False
            try:
                decision = improve_once(
                    symbols,
                    bars=args.bars,
                    oos_frac=args.oos,
                    refresh=False,
                    generation=gen_box["n"],
                    hunt=hunt,
                )
                gen_box["n"] += 1
                met = bool((decision.get("grade") or {}).get("targets_met"))
                if decision.get("improved"):
                    if autopilot_frozen():
                        log.info("AUTOPILOT_FROZEN=1: promotion suppressed — paper not restarted")
                    else:
                        just_promoted = True
                        restart_paper.set()
                        _live_feed("paper_restart", "Restarting practice bot with new settings…")
                        try:
                            from moonshot.paths import DB_PATH
                            from moonshot.persist import Persistence

                            Persistence(str(DB_PATH), mode="paper").log_event(
                                _utc_now(),
                                "*",
                                "restart",
                                {
                                    "reason": "promote_paper_restart",
                                    "generation": gen_box.get("n"),
                                    "promote": True,
                                },
                            )
                        except Exception:
                            pass
            except Exception as e:
                log.error("improve pass failed: %s", e)
                _live_feed("error", f"Improve pass failed: {e}")
            if not continuous:
                stop.wait(max(60.0, every_h * 3600))
                continue
            # After a real promote: pause so paper can trade the new book instead
            # of immediate re-hunt → re-promote → restart thrash.
            if just_promoted:
                log.info(
                    "post-promote pause %ds — letting paper run before next hunt",
                    int(POST_PROMOTE_HUNT_PAUSE_SEC),
                )
                stop.wait(float(POST_PROMOTE_HUNT_PAUSE_SEC))
                continue
            # Not near-perfect → re-hunt immediately (tiny yield for other threads)
            if hunt and not met:
                stop.wait(2.0)
                continue
            # Targets met (or tune-only): brief pause before re-checking
            stop.wait(cooldown)

    worker = threading.Thread(target=improve_worker, name="moonshot-improve", daemon=True)
    worker.start()
    log.info(
        "AUTOPILOT ON — HUNT %d coins · %s · targets test_exp>=%.2f n>=%d · "
        "re-runs IMMEDIATELY until targets met · practice only",
        len(symbols),
        ",".join(symbols),
        HUNT_TARGETS["test_exp"],
        HUNT_TARGETS["test_n"],
    )
    _live_feed(
        "autopilot_on",
        f"HUNT on {len(symbols)} coins. Backtests until targets met "
        f"(avg ≥{_pct_of_risk_msg(HUNT_TARGETS['test_exp'])}, "
        f"win ≥{HUNT_TARGETS['win_rate']:.0%}, ≥{HUNT_TARGETS['test_n']} trades). "
        f"Fake money only.",
        cooldown=0,
        targets=HUNT_TARGETS,
        symbols=symbols,
    )

    last_watchdog = 0.0
    try:
        while True:
            supervisor.ensure()
            if restart_paper.is_set():
                restart_paper.clear()
                log.info("restarting paper with upgraded settings")
                supervisor.start()
                # Post-restart: assert child inherited the expected S/T/L.
                deadline = time.time() + float(PAPER_BOOK_ASSERT_SEC)
                exp = _load_autopilot_state().get("expected_paper_book") or {}
                want = (
                    str(exp.get("strategy") or os.environ.get("STRATEGY") or ""),
                    str(exp.get("timeframe") or os.environ.get("TIMEFRAME") or ""),
                    str(exp.get("leverage") or os.environ.get("LEVERAGE") or ""),
                )
                while time.time() < deadline:
                    got = (
                        str(os.environ.get("STRATEGY") or ""),
                        str(os.environ.get("TIMEFRAME") or ""),
                        str(os.environ.get("LEVERAGE") or ""),
                    )
                    if want[0] and got == want:
                        log.info("paper fingerprint OK · %s %sx@%s", *got)
                        break
                    time.sleep(2.0)
                else:
                    got = (
                        str(os.environ.get("STRATEGY") or ""),
                        str(os.environ.get("TIMEFRAME") or ""),
                        str(os.environ.get("LEVERAGE") or ""),
                    )
                    if want[0] and got != want:
                        msg = (
                            f"CRITICAL paper fingerprint mismatch after restart: "
                            f"want={want} got={got}"
                        )
                        log.critical(msg)
                        _live_feed("scoreboard_critical", msg)
                        st = _load_autopilot_state()
                        st["scoreboard_watchdog"] = {
                            "ts": _utc_now().isoformat(),
                            "critical": True,
                            "alerts": [msg],
                        }
                        _save_autopilot_state(st)
            now = time.time()
            if now - last_watchdog >= float(SCOREBOARD_WATCHDOG_SEC):
                last_watchdog = now
                try:
                    _run_scoreboard_watchdog()
                except Exception as e:
                    log.warning("scoreboard watchdog failed: %s", e)
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("autopilot stopped by user")
    finally:
        stop.set()
        supervisor.stop()
