from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


log = logging.getLogger("moonshot")


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"

    @property
    def sign(self) -> int:
        return 1 if self is Direction.LONG else -1


class RegimeState(str, Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    UNKNOWN = "unknown"


class ExitReason(str, Enum):
    STOP = "stop"
    TRAILING = "trailing"
    LIQUIDATION = "liquidation"
    REGIME_INVALID = "regime_invalid"
    TIME_STOP = "time_stop"
    KILL_SWITCH = "kill_switch"
    END_OF_DATA = "end_of_data"


class ActionKind(str, Enum):
    EXIT = "exit"
    PARTIAL = "partial"
    ADD = "add"
    MOVE_STOP = "move_stop"


class AddDecision(str, Enum):
    ALLOW = "allow"
    BLOCK_MAX_ADDS = "block_max_adds"
    BLOCK_R_THRESHOLD = "block_r_threshold"
    BLOCK_REGIME_INVALID = "block_regime_invalid"
    BLOCK_LEVERAGE_CAP = "block_leverage_cap"
    BLOCK_NO_FLOATING_PROFIT = "block_no_floating_profit"
    BLOCK_TRADING_HALTED = "block_trading_halted"
    BLOCK_STOP_WIDEN = "block_stop_widen"
    BLOCK_SIZE_TOO_SMALL = "block_size_too_small"


class BotError(Exception):
    """Base class for bot errors."""


class RiskError(BotError):
    pass


class ExecutionError(BotError):
    pass


class ConfigError(BotError):
    pass


class DataError(BotError):
    pass


@dataclass(frozen=True)
class InstrumentSpec:
    """Exchange constraints for one symbol. Quantities are in BASE units."""

    symbol: str = ""
    qty_step: float = 0.0  # 0 = continuous (backtests)
    min_qty: float = 0.0
    contract_size: float = 1.0

    def round_qty(self, qty: float) -> float:
        if qty <= 0 or not math.isfinite(qty):
            return 0.0
        if self.qty_step > 0:
            qty = math.floor(qty / self.qty_step + 1e-9) * self.qty_step
        return qty if qty >= self.min_qty and qty > 0 else 0.0


@dataclass
class Signal:
    symbol: str
    direction: Direction
    ref_price: float  # bar close that produced the signal
    atr: float
    regime: RegimeState
    reason: str = ""


@dataclass
class Action:
    kind: ActionKind
    qty: float = 0.0
    price: float = 0.0  # new stop for ADD / MOVE_STOP
    reason: Optional[ExitReason] = None


@dataclass
class Fill:
    order_id: str
    symbol: str
    side: str
    qty: float
    price: float
    fee: float
    timestamp: datetime


@dataclass
class BrokerPosition:
    direction: Direction
    qty: float
    entry_price: float


@dataclass
class RiskDecision:
    approved: bool
    reason: str
    qty: float = 0.0
    margin: float = 0.0
    leverage: int = 0
    risk_dollars: float = 0.0


@dataclass
class Lot:
    price: float
    qty: float
    time: datetime


@dataclass
class AddRecord:
    price: float
    qty: float
    stop_at_add: float
    r_at_add: float
    floating_profit_used: float
    time: datetime


@dataclass
class TradeResult:
    symbol: str
    direction: Direction
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    initial_qty: float
    pnl: float
    r_multiple: float
    liquidated: bool
    exit_reason: ExitReason
    max_adds: int
    partial_taken: bool
    fees: float
    funding: float
    initial_risk_dollars: float = 0.0
    config_hash: str | None = None
    config_json: str | None = None

    @property
    def duration_hours(self) -> float:
        return (self.exit_time - self.entry_time).total_seconds() / 3600.0


def _pandas_tf(tf: str) -> str:
    """Map exchange timeframes to pandas offsets (15m → 15min; 1h stays 1h)."""
    s = str(tf).strip().lower()
    if s.endswith("m") and not s.endswith("min") and not s.endswith("ms"):
        # 'm' alone means month in modern pandas — minutes need 'min'
        num = s[:-1] or "1"
        if num.isdigit():
            return f"{num}min"
    return s
