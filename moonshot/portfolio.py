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
    Action, ActionKind, AddDecision, ExitReason, InstrumentSpec,
)
from moonshot.config import (
    Config,
)
from moonshot.strategy import (
    regime_aligned, regime_hostile,
)
from moonshot.position import (
    Position,
)

class RollManager:
    """Conditional pyramiding. Sizing is 'free-roll': an add may never turn a locked-in
    winner into a net loser if the tightened stop is hit."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def size_add(self, pos: Position, price: float, new_stop: float, equity: float,
                 used_margin: float, spec: InstrumentSpec) -> tuple:
        cfg = self.cfg
        floating = pos.unrealized(price)
        if floating <= 0:
            return None, AddDecision.BLOCK_NO_FLOATING_PROFIT
        loss_per_unit = pos.sign * (price - new_stop)
        if loss_per_unit <= 0:
            return None, AddDecision.BLOCK_STOP_WIDEN

        q_house = floating * pos.leverage / price  # add notional <= floating profit x leverage
        locked = pos.net_pnl_at(new_stop)  # net P&L if existing qty stops out at new_stop
        allowed_loss = locked + cfg.addon_giveback_r * pos.initial_risk_dollars
        if allowed_loss <= 0:
            return None, AddDecision.BLOCK_NO_FLOATING_PROFIT
        q_free_roll = allowed_loss / loss_per_unit
        q_lev = max(0.0, equity * cfg.max_account_leverage - pos.current_qty * price) / price
        q_margin = max(0.0, equity - used_margin) * pos.leverage / price

        min_qty = pos.initial_qty * cfg.min_add_qty_pct
        if min(q_lev, q_margin) < min_qty:
            return None, AddDecision.BLOCK_LEVERAGE_CAP
        qty = spec.round_qty(min(q_house, q_free_roll, q_lev, q_margin))
        if qty < min_qty or qty <= 0:
            return None, AddDecision.BLOCK_SIZE_TOO_SMALL
        return qty, AddDecision.ALLOW

    def evaluate_add(self, pos: Position, price: float, stop_candidate: float, regime: Any,
                     allowed: bool, equity: float, used_margin: float,
                     spec: InstrumentSpec) -> tuple:
        cfg = self.cfg
        if not allowed:
            return AddDecision.BLOCK_TRADING_HALTED, None
        if not regime_aligned(pos.direction, regime):
            return AddDecision.BLOCK_REGIME_INVALID, None
        if pos.adds_count >= cfg.max_adds_per_trade:
            return AddDecision.BLOCK_MAX_ADDS, None
        if pos.r_multiple(price) < cfg.add_trigger_r[pos.adds_count]:
            return AddDecision.BLOCK_R_THRESHOLD, None
        if not pos.is_tighter(stop_candidate):
            return AddDecision.BLOCK_STOP_WIDEN, None
        qty, decision = self.size_add(pos, price, stop_candidate, equity, used_margin, spec)
        return decision, qty


class PositionManager:
    """Pure decision logic evaluated at each bar CLOSE. Returns actions to execute."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.roll = RollManager(cfg)

    def decide(self, pos: Position, bar: Any, *, allowed: bool, kill: bool, equity: float,
               used_margin: float, spec: InstrumentSpec) -> list:
        cfg = self.cfg
        if kill:
            return [Action(ActionKind.EXIT, reason=ExitReason.KILL_SWITCH)]
        regime = bar.regime
        # Grace ≥1 closed bar. Exit only on *hostile* regime (opposite trend) —
        # never on unknown/ranging (warmup + MTF/mom entries ignore bar.regime).
        held = pos.bars_held(getattr(bar, "Index", None), cfg.bar_td)
        if (
            cfg.exit_on_regime_invalid
            and held >= 1
            and regime_hostile(pos.direction, regime)
        ):
            return [Action(ActionKind.EXIT, reason=ExitReason.REGIME_INVALID)]

        close = bar.close
        r_now = pos.r_multiple(close)

        # Time stop: if the trade never reaches +min_r within N bars, dump it
        if cfg.enable_time_stop and cfg.time_stop_bars > 0:
            if held >= cfg.time_stop_bars and r_now < cfg.time_stop_min_r:
                return [Action(ActionKind.EXIT, reason=ExitReason.TIME_STOP)]

        stop = pos.current_stop
        if bar.atr == bar.atr and bar.atr > 0:
            trail_mult = cfg.trail_atr_multiple
            if cfg.use_asymmetric_trail and r_now < 0:
                trail_mult = min(trail_mult, cfg.trail_atr_loser)
            trail = close - pos.sign * trail_mult * bar.atr
            stop = max(stop, trail) if pos.sign > 0 else min(stop, trail)

        acts: list = []
        if cfg.enable_partial and not pos.partial_done and r_now >= cfg.partial_at_r:
            q = spec.round_qty(pos.current_qty * cfg.partial_pct)
            if 0 < q < pos.current_qty:
                acts.append(Action(ActionKind.PARTIAL, qty=q))
                # Lock runner at breakeven after banking the partial
                stop = max(stop, pos.avg_entry) if pos.sign > 0 else min(stop, pos.avg_entry)

        if cfg.enable_pyramiding:
            decision, _ = self.roll.evaluate_add(pos, close, stop, regime, allowed, equity,
                                                 used_margin, spec)
            if decision is AddDecision.ALLOW:
                acts.append(Action(ActionKind.ADD, price=stop))

        if stop != pos.current_stop:
            acts.append(Action(ActionKind.MOVE_STOP, price=stop))
        return acts


