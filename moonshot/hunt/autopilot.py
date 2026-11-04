from __future__ import annotations

import argparse
import json
import logging
import math
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import uuid
import zlib
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    ConfigError,
)
from moonshot.data import (
    clear_stale_live_caches,
)
from moonshot.symbols import (
    HUNT_SYMBOLS, PAPER_SYMBOLS,
)
from moonshot.live import (
    _live_feed, _utc_now,
)
from moonshot.hunt.constants import (
    HUNT_TARGETS, _load_autopilot_state, _pct_of_risk_msg, _save_autopilot_state,
)
from moonshot.hunt.scoring import (
    POST_PROMOTE_HUNT_PAUSE_SEC,
)
from moonshot.hunt.improve import (
    improve_once,
)

from moonshot.paths import (
    ROOT, ENV_PATH, DB_PATH, AUTOPILOT_STATE, LIVE_FEED,
    PAPER_RUNTIME_PATH, RULES_PATH,
)

class PaperSupervisor:
    """Keep `python -m moonshot paper` alive as a child process."""

    def __init__(self, symbols: Optional[list] = None):
        self.symbols = symbols
        self.proc: Optional[subprocess.Popen] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            self._stop_unlocked()
            cmd = [sys.executable, "-m", "moonshot", "paper"]
            if self.symbols:
                cmd += ["--symbols", *self.symbols]
            log.info("starting paper child: %s", " ".join(cmd))
            self.proc = subprocess.Popen(
                cmd, cwd=str(ROOT),
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

    def ensure(self) -> None:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                code = None if self.proc is None else self.proc.returncode
                if code not in (None, 0):
                    log.warning("paper child exited with code %s — restarting", code)
                # start without re-entering lock
                cmd = [sys.executable, "-m", "moonshot", "paper"]
                if self.symbols:
                    cmd += ["--symbols", *self.symbols]
                log.info("starting paper child: %s", " ".join(cmd))
                self.proc = subprocess.Popen(
                    cmd, cwd=str(ROOT),
                    env={**os.environ, "LIVE_TRADING_ENABLED": "0"},
                )


def cmd_autopilot(args: argparse.Namespace) -> None:
    try:
        clear_stale_live_caches()
    except Exception as e:
        log.warning("stale cache clear on autopilot start: %s", e)
    """
    Leave this running:
      - paper trading stays up
      - HUNT mode: wide search until near-perfect score targets are met
      - only keeps configs that are profitable on BOTH history halves
      - never turns on real-money trading
    """
    if os.getenv("LIVE_TRADING_ENABLED") == "1":
        raise ConfigError(
            "Autopilot refuses to start while LIVE_TRADING_ENABLED=1. "
            "Set it to 0 — autopilot is practice-only."
        )
    # Focus hunt + paper on BTC until edge is proven; --symbols overrides.
    symbols = args.symbols or list(HUNT_SYMBOLS)
    paper_symbols = args.symbols or list(PAPER_SYMBOLS)
    continuous = not bool(getattr(args, "daily", False))
    cooldown = max(5, int(getattr(args, "cooldown_seconds", 15)))
    every_h = float(getattr(args, "every_hours", 0) or 0)
    hunt = not bool(getattr(args, "tune_only", False))

    state = _load_autopilot_state()
    state["started_at"] = _utc_now().isoformat()
    state["mode"] = "hunt" if hunt else ("continuous" if continuous else f"every_{every_h}h")
    state["cooldown_seconds"] = cooldown
    state["targets"] = HUNT_TARGETS
    state["hunt"] = hunt
    state["hunt_symbols"] = list(symbols)
    state["paper_symbols"] = list(paper_symbols)
    _save_autopilot_state(state)

    if args.improve_only:
        gen = int(state.get("generation", 0))
        decision = improve_once(
            symbols, bars=args.bars, oos_frac=args.oos,
            refresh=args.refresh, generation=gen, hunt=hunt,
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
                    symbols, bars=args.bars, oos_frac=args.oos,
                    refresh=False, generation=gen_box["n"], hunt=hunt,
                )
                gen_box["n"] += 1
                met = bool((decision.get("grade") or {}).get("targets_met"))
                if decision.get("improved"):
                    just_promoted = True
                    restart_paper.set()
                    _live_feed("paper_restart", "Restarting practice bot with new settings…")
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
        len(symbols), ",".join(symbols), HUNT_TARGETS["test_exp"], HUNT_TARGETS["test_n"],
    )
    _live_feed(
        "autopilot_on",
        f"HUNT on {len(symbols)} coins. Backtests until targets met "
        f"(avg ≥{_pct_of_risk_msg(HUNT_TARGETS['test_exp'])}, "
        f"win ≥{HUNT_TARGETS['win_rate']:.0%}, ≥{HUNT_TARGETS['test_n']} trades). "
        f"Fake money only.",
        cooldown=0, targets=HUNT_TARGETS, symbols=symbols,
    )

    try:
        while True:
            supervisor.ensure()
            if restart_paper.is_set():
                restart_paper.clear()
                log.info("restarting paper with upgraded settings")
                supervisor.start()
            time.sleep(5)
    except KeyboardInterrupt:
        log.info("autopilot stopped by user")
    finally:
        stop.set()
        supervisor.stop()


