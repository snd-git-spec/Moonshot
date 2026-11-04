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
    AddRecord, Direction, ExitReason, Lot, RiskError, TradeResult,
)

@dataclass
class Position:
    symbol: str
    direction: Direction
    leverage: int
    entry_time: datetime
    initial_entry: float
    initial_qty: float
    initial_stop: float
    avg_entry: float
    current_qty: float
    current_stop: float
    lots: list = field(default_factory=list)
    adds: list = field(default_factory=list)
    realized_gross: float = 0.0  # price P&L of already-closed partial quantity
    fees: float = 0.0
    funding: float = 0.0
    partial_done: bool = False

    # ---- construction ----
    @classmethod
    def open(cls, symbol: str, direction: Direction, leverage: int, price: float,
             qty: float, stop: float, ts: datetime, fee: float) -> "Position":
        return cls(symbol, direction, leverage, ts, price, qty, stop, price, qty, stop,
                   lots=[Lot(price, qty, ts)], fees=fee)

    # ---- derived ----
    @property
    def sign(self) -> int:
        return self.direction.sign

    @property
    def initial_risk_per_unit(self) -> float:
        return abs(self.initial_entry - self.initial_stop)

    @property
    def initial_risk_dollars(self) -> float:
        return self.initial_qty * self.initial_risk_per_unit

    @property
    def adds_count(self) -> int:
        return len(self.adds)

    def margin(self) -> float:
        return self.avg_entry * self.current_qty / self.leverage

    def unrealized(self, price: float) -> float:
        return self.sign * (price - self.avg_entry) * self.current_qty

    def net_pnl_at(self, price: float) -> float:
        """Everything booked or floating if the remainder closed at `price` (before exit fee)."""
        return self.realized_gross + self.unrealized(price) - self.fees - self.funding

    def r_multiple(self, price: float) -> float:
        if self.initial_risk_per_unit <= 0:
            return 0.0
        return self.sign * (price - self.initial_entry) / self.initial_risk_per_unit

    def bars_held(self, bar_ts, bar_td: pd.Timedelta) -> int:
        """Whole bars since entry fill (0 on the entry bar)."""
        try:
            delta = pd.Timestamp(bar_ts) - pd.Timestamp(self.entry_time)
            if bar_td.total_seconds() <= 0:
                return 0
            return max(0, int(delta / bar_td))
        except Exception:
            return 0

    def liq_price(self, mmr: float) -> float:
        return self.avg_entry * (1 - self.sign * (1.0 / self.leverage - mmr))

    # ---- mutation ----
    def is_tighter(self, stop: float) -> bool:
        return stop >= self.current_stop if self.sign > 0 else stop <= self.current_stop

    def tighten_stop(self, stop: float) -> bool:
        if self.is_tighter(stop):
            self.current_stop = stop
            return True
        return False

    def add_lot(self, price: float, qty: float, ts: datetime, fee: float,
                new_stop: float, floating_used: float) -> None:
        if not self.is_tighter(new_stop):
            raise RiskError(f"refusing to widen stop {self.current_stop} -> {new_stop}")
        r_at_add = self.r_multiple(price)
        cost = self.avg_entry * self.current_qty + price * qty
        self.current_qty += qty
        self.avg_entry = cost / self.current_qty  # average entry, not original entry
        self.fees += fee
        self.lots.append(Lot(price, qty, ts))
        self.adds.append(AddRecord(price, qty, new_stop, r_at_add, floating_used, ts))
        self.current_stop = new_stop

    def reduce(self, price: float, qty: float, fee: float) -> float:
        qty = min(qty, self.current_qty)
        gross = self.sign * (price - self.avg_entry) * qty
        self.realized_gross += gross
        self.fees += fee
        self.current_qty -= qty
        return gross

    def close_out(self, price: float, fee: float, ts: datetime, reason: ExitReason,
                  liquidated: bool = False) -> tuple:
        """Close the remainder. Returns (TradeResult, cash_delta)."""
        gross = self.sign * (price - self.avg_entry) * self.current_qty
        net = self.realized_gross + gross - self.fees - fee - self.funding
        risk = self.initial_risk_dollars
        res = TradeResult(
            symbol=self.symbol, direction=self.direction, entry_time=self.entry_time,
            exit_time=ts, entry_price=self.initial_entry, exit_price=price,
            initial_qty=self.initial_qty, pnl=net, r_multiple=net / risk if risk > 0 else 0.0,
            liquidated=liquidated, exit_reason=reason, max_adds=self.adds_count,
            partial_taken=self.partial_done, fees=self.fees + fee, funding=self.funding,
        )
        self.current_qty = 0.0
        return res, gross - fee

    # ---- persistence ----
    def to_dict(self) -> dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["entry_time"] = pd.Timestamp(self.entry_time).isoformat()
        for item in d["lots"] + d["adds"]:
            item["time"] = pd.Timestamp(item["time"]).isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["direction"] = Direction(d["direction"])
        d["entry_time"] = pd.Timestamp(d["entry_time"])
        d["lots"] = [Lot(l["price"], l["qty"], pd.Timestamp(l["time"])) for l in d["lots"]]
        d["adds"] = [AddRecord(a["price"], a["qty"], a["stop_at_add"], a["r_at_add"],
                               a["floating_profit_used"], pd.Timestamp(a["time"]))
                     for a in d["adds"]]
        return cls(**d)


