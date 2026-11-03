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

from moonshot._pull import pull
import moonshot.types as _moonshot_types
pull(_moonshot_types, globals())
import moonshot.config as _moonshot_config
pull(_moonshot_config, globals())

class BaseBroker:
    """Quantities are BASE units; brokers convert to exchange contracts internally."""

    def set_leverage(self, symbol: str, leverage: int) -> None: ...
    def market_order(self, symbol: str, side: str, qty: float, ref_price: float,
                     reduce_only: bool = False) -> Fill: raise NotImplementedError
    def set_stop(self, symbol: str, direction: Direction, qty: float, stop: float) -> None:
        raise NotImplementedError
    def cancel_stop(self, symbol: str) -> None: raise NotImplementedError
    def get_position(self, symbol: str) -> Optional[BrokerPosition]: raise NotImplementedError
    def get_equity(self) -> float: raise NotImplementedError
    def on_price(self, symbol: str, price: float) -> None: ...
    def pop_stop_fill(self, symbol: str) -> Optional[Fill]: return None


class PaperBroker(BaseBroker):
    """In-memory broker: fees, slippage, reduce-only, exchange-side stops triggered by on_price()."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cash = cfg.starting_equity
        self.signed: dict = {}  # symbol -> signed qty
        self.avg: dict = {}
        self.stops: dict = {}  # symbol -> stop price
        self.last: dict = {}
        self._stop_fills: dict = {}
        self._n = 0

    def _fill(self, symbol: str, side: str, qty: float, price: float) -> Fill:
        self._n += 1
        buy = side == "buy"
        px = price * (1 + self.cfg.slip if buy else 1 - self.cfg.slip)
        fee = px * qty * self.cfg.fee_taker
        delta = qty if buy else -qty
        s0 = self.signed.get(symbol, 0.0)
        if s0 != 0 and (s0 > 0) != buy:  # reducing -> realize P&L
            closing = min(abs(s0), qty)
            self.cash += (px - self.avg[symbol]) * closing * (1 if s0 > 0 else -1)
        elif s0 == 0 or (s0 > 0) == buy:  # opening / adding -> new average
            tot = abs(s0) + qty
            self.avg[symbol] = (self.avg.get(symbol, 0.0) * abs(s0) + px * qty) / tot
        s1 = s0 + delta
        self.signed[symbol] = 0.0 if abs(s1) < 1e-12 else s1
        if self.signed[symbol] == 0:
            self.avg.pop(symbol, None)
            self.stops.pop(symbol, None)
        self.cash -= fee
        return Fill(f"paper-{self._n}", symbol, side, qty, px, fee, pd.Timestamp.now(tz="UTC"))

    def market_order(self, symbol, side, qty, ref_price, reduce_only=False) -> Fill:
        s0 = self.signed.get(symbol, 0.0)
        if reduce_only:
            if s0 == 0 or (s0 > 0) == (side == "buy"):
                raise ExecutionError("reduce-only order would not reduce")
            qty = min(qty, abs(s0))
        self.last[symbol] = ref_price
        return self._fill(symbol, side, qty, ref_price)

    def set_stop(self, symbol, direction, qty, stop) -> None:
        self.stops[symbol] = stop

    def cancel_stop(self, symbol) -> None:
        self.stops.pop(symbol, None)

    def get_position(self, symbol) -> Optional[BrokerPosition]:
        s = self.signed.get(symbol, 0.0)
        if s == 0:
            return None
        return BrokerPosition(Direction.LONG if s > 0 else Direction.SHORT, abs(s), self.avg[symbol])

    def get_equity(self) -> float:
        eq = self.cash
        for sym, s in self.signed.items():
            if s != 0 and sym in self.avg:
                eq += (self.last.get(sym, self.avg[sym]) - self.avg[sym]) * s
        return eq

    def on_price(self, symbol, price) -> None:
        self.last[symbol] = price
        s, stop = self.signed.get(symbol, 0.0), self.stops.get(symbol)
        if not s or stop is None:
            return
        if s > 0 and price <= stop:
            self._stop_fills[symbol] = self._fill(symbol, "sell", abs(s), min(stop, price))
        elif s < 0 and price >= stop:
            self._stop_fills[symbol] = self._fill(symbol, "buy", abs(s), max(stop, price))

    def pop_stop_fill(self, symbol) -> Optional[Fill]:
        return self._stop_fills.pop(symbol, None)

    def rehydrate(self, positions: dict, closed_pnl: float) -> None:
        """
        Restore broker state after a process restart.
        Without this, LiveRunner.load_positions() leaves PaperBroker empty, and the
        next _reconcile treats every restored position as 'exchange stop fired' and
        fabricates a -1R close. Cash = starting + closed trade PnL − open fees/funding.
        """
        self.cash = float(self.cfg.starting_equity) + float(closed_pnl)
        self.signed.clear()
        self.avg.clear()
        self.stops.clear()
        self._stop_fills.clear()
        for sym, pos in positions.items():
            self.cash -= float(pos.fees) + float(pos.funding)
            self.signed[sym] = pos.sign * float(pos.current_qty)
            self.avg[sym] = float(pos.avg_entry)
            self.stops[sym] = float(pos.current_stop)
            self.last[sym] = float(pos.avg_entry)


class MexcBroker(BaseBroker):
    """
    Live MEXC USDT-perp adapter via CCXT.  *** UNVERIFIED AGAINST THE LIVE API ***
    Before trusting it: run with the minimum size, and check each order type, the
    leverage/openType params, trigger-order cancellation and balance field against
    the current ccxt `mexc` docs. Points to verify are marked VERIFY.
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg
        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise ConfigError("ccxt is required for MexcBroker") from e
        key, secret = os.getenv("MEXC_API_KEY"), os.getenv("MEXC_API_SECRET")
        if not key or not secret:
            raise ConfigError("MEXC_API_KEY / MEXC_API_SECRET not set")
        self.ex = ccxt.mexc({"apiKey": key, "secret": secret, "enableRateLimit": True,
                             "options": {"defaultType": "swap"}})
        self.ex.load_markets()
        self._lev: dict = {}
        self._stop_id: dict = {}

    def _contracts(self, symbol: str, qty: float) -> float:
        m = self.ex.market(symbol)
        size = float(m.get("contractSize") or 1.0)
        return float(self.ex.amount_to_precision(symbol, qty / size))

    def set_leverage(self, symbol, leverage) -> None:
        self._lev[symbol] = leverage
        try:
            self.ex.set_leverage(leverage, symbol, params={"openType": 1})  # VERIFY isolated
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"set_leverage failed: {e}") from e

    def market_order(self, symbol, side, qty, ref_price, reduce_only=False) -> Fill:
        amount = self._contracts(symbol, qty)
        if amount <= 0:
            raise ExecutionError("order size rounds to zero contracts")
        params: dict = {"reduceOnly": reduce_only}
        if not reduce_only:  # VERIFY: MEXC swap requires leverage + openType on opening orders
            params.update({"leverage": self._lev.get(symbol, self.cfg.leverage), "openType": 1})
        try:
            raw = self.ex.create_order(symbol, "market", side, amount, None, params)
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"market order failed: {e}") from e
        size = float(self.ex.market(symbol).get("contractSize") or 1.0)
        px = float(raw.get("average") or raw.get("price") or ref_price)
        fee = float((raw.get("fee") or {}).get("cost") or px * qty * self.cfg.fee_taker)
        return Fill(str(raw["id"]), symbol, side, float(raw.get("filled") or amount) * size,
                    px, fee, pd.Timestamp.now(tz="UTC"))

    def set_stop(self, symbol, direction, qty, stop) -> None:
        self.cancel_stop(symbol)
        side = "sell" if direction is Direction.LONG else "buy"
        amount = self._contracts(symbol, qty)
        try:  # VERIFY: stopLossPrice trigger-market params on MEXC swap
            raw = self.ex.create_order(symbol, "market", side, amount, None,
                                       {"stopLossPrice": float(self.ex.price_to_precision(symbol, stop)),
                                        "reduceOnly": True})
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"stop placement failed: {e}") from e
        self._stop_id[symbol] = raw.get("id")

    def cancel_stop(self, symbol) -> None:
        oid = self._stop_id.pop(symbol, None)
        if not oid:
            return
        try:  # VERIFY: MEXC can return success with no body; ccxt may raise anyway
            self.ex.cancel_order(oid, symbol, params={"trigger": True})
        except Exception as e:  # noqa: BLE001
            log.warning("cancel_stop(%s) raised %s - check for orphaned stop orders!", symbol, e)

    def get_position(self, symbol) -> Optional[BrokerPosition]:
        try:
            raws = self.ex.fetch_positions([symbol])
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"fetch_positions failed: {e}") from e
        size = float(self.ex.market(symbol).get("contractSize") or 1.0)
        for r in raws:
            c = float(r.get("contracts") or 0)
            if c > 0:
                d = Direction.LONG if r["side"] == "long" else Direction.SHORT
                return BrokerPosition(d, c * size, float(r.get("entryPrice") or 0))
        return None

    def get_equity(self) -> float:
        try:
            bal = self.ex.fetch_balance()
            return float(bal["total"]["USDT"])  # VERIFY: should include unrealized PnL
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"fetch_balance failed: {e}") from e


class MarketData:
    """Public market data for paper/live loops."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.ex = public_exchange()
        self.ex.load_markets()

    def instrument(self, symbol: str) -> InstrumentSpec:
        m = self.ex.market(symbol)
        size = float(m.get("contractSize") or 1.0)
        min_c = float(((m.get("limits") or {}).get("amount") or {}).get("min") or 1.0)
        # ASSUMPTION: 1 contract is the lot step. VERIFY per symbol.
        return InstrumentSpec(symbol, qty_step=size, min_qty=size * min_c, contract_size=size)

    def last_price(self, symbol: str) -> float:
        last_err: Optional[Exception] = None
        for attempt in range(3):
            try:
                return float(self.ex.fetch_ticker(symbol)["last"])
            except Exception as e:  # noqa: BLE001 — network blips must not kill paper
                last_err = e
                time.sleep(0.25 * (attempt + 1))
        raise DataError(f"ticker failed after retries ({symbol}): {last_err}") from last_err

    def recent(self, symbol: str) -> pd.DataFrame:
        tf, n = self.cfg.timeframe, self.cfg.live_history_bars
        # Shared cache with automatic tail refresh when stale (Layer 0).
        try:
            return fetch_historical(symbol, tf, int(n), refresh=False)
        except DataError:
            now_ms = int(time.time() * 1000)
            with _OHLCV_LOCK:
                rows = paged_ohlcv(self.ex, symbol, tf, now_ms - int(n) * timeframe_ms(tf), now_ms)
            if not rows:
                raise DataError(
                    f"no OHLCV rows returned ({symbol} {tf}×{n}). "
                    "Exchange empty — usually rate-limit while hunt is fetching."
                )
            return ohlcv_to_frame(rows, tf)


