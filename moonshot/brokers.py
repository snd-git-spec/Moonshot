from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    BrokerPosition,
    ConfigError,
    DataError,
    Direction,
    ExecutionError,
    Fill,
    InstrumentSpec,
)
from moonshot.config import (
    Config,
)
from moonshot.data import (
    _OHLCV_LOCK,
    fetch_historical,
    ohlcv_to_frame,
    paged_ohlcv,
    public_exchange,
    timeframe_ms,
)


class BaseBroker:
    """Quantities are BASE units; brokers convert to exchange contracts internally."""

    def set_leverage(self, symbol: str, leverage: int) -> None: ...
    def market_order(
        self, symbol: str, side: str, qty: float, ref_price: float, reduce_only: bool = False
    ) -> Fill:
        raise NotImplementedError

    def set_stop(self, symbol: str, direction: Direction, qty: float, stop: float) -> None:
        raise NotImplementedError

    def cancel_stop(self, symbol: str) -> None:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Optional[BrokerPosition]:
        raise NotImplementedError

    def get_equity(self) -> float:
        raise NotImplementedError

    def on_price(self, symbol: str, price: float) -> None: ...
    def pop_stop_fill(self, symbol: str) -> Optional[Fill]:
        return None


class PaperBroker(BaseBroker):
    """In-memory broker: fees, slippage, reduce-only, exchange-side stops triggered by on_price()."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.cash = cfg.starting_equity
        self.signed: dict = {}  # symbol -> signed qty
        self.avg: dict = {}
        self.stops: dict = {}  # symbol -> stop price
        self.last: dict = {}  # symbol -> last mark (for unrealized)
        self._stop_fills: dict = {}
        self._n = 0
        # After rehydrate, marks may be hours old — refuse new risk until on_price.
        self.equity_stale: bool = False

    def mark_snapshot(self) -> dict[str, float]:
        """Persisted last marks so equity is reconstructable without a live tick."""
        return {str(s): float(p) for s, p in self.last.items()}

    def _fill(
        self,
        symbol: str,
        side: str,
        qty: float,
        price: float,
        *,
        intended_qty: float | None = None,
    ) -> Fill:
        self._n += 1
        buy = side == "buy"
        assumed_bps = float(self.cfg.slippage_bps)
        px = price * (1 + self.cfg.slip if buy else 1 - self.cfg.slip)
        # Realized vs ref: buy positive = paid up; sell positive = received less.
        if price > 0:
            slip_bps = ((px - price) / price) * 10_000.0 * (1.0 if buy else -1.0)
        else:
            slip_bps = 0.0
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
        return Fill(
            f"paper-{self._n}",
            symbol,
            side,
            qty,
            px,
            fee,
            pd.Timestamp.now(tz="UTC"),
            intended_qty=float(intended_qty) if intended_qty is not None else float(qty),
            ref_price=float(price),
            slippage_bps=float(slip_bps),
            assumed_slippage_bps=assumed_bps,
        )

    def market_order(self, symbol, side, qty, ref_price, reduce_only=False) -> Fill:
        intended = float(qty)
        s0 = self.signed.get(symbol, 0.0)
        if reduce_only:
            if s0 == 0 or (s0 > 0) == (side == "buy"):
                raise ExecutionError("reduce-only order would not reduce")
            qty = min(qty, abs(s0))
        self.last[symbol] = ref_price
        return self._fill(symbol, side, qty, ref_price, intended_qty=intended)

    def set_stop(self, symbol, direction, qty, stop) -> None:
        self.stops[symbol] = stop

    def cancel_stop(self, symbol) -> None:
        self.stops.pop(symbol, None)

    def get_position(self, symbol) -> Optional[BrokerPosition]:
        s = self.signed.get(symbol, 0.0)
        if s == 0:
            return None
        return BrokerPosition(
            Direction.LONG if s > 0 else Direction.SHORT, abs(s), self.avg[symbol]
        )

    def get_equity(self) -> float:
        eq = self.cash
        for sym, s in self.signed.items():
            if s != 0 and sym in self.avg:
                eq += (self.last.get(sym, self.avg[sym]) - self.avg[sym]) * s
        return eq

    def on_price(self, symbol, price) -> None:
        self.last[symbol] = float(price)
        if self.equity_stale:
            # First fresh mark after restart — equity is trustworthy again.
            self.equity_stale = False
        s, stop = self.signed.get(symbol, 0.0), self.stops.get(symbol)
        if not s or stop is None:
            return
        if s > 0 and price <= stop:
            self._stop_fills[symbol] = self._fill(symbol, "sell", abs(s), min(stop, price))
        elif s < 0 and price >= stop:
            self._stop_fills[symbol] = self._fill(symbol, "buy", abs(s), max(stop, price))

    def pop_stop_fill(self, symbol) -> Optional[Fill]:
        return self._stop_fills.pop(symbol, None)

    def rehydrate(
        self,
        positions: dict,
        closed_pnl: float,
        *,
        last_marks: Optional[dict] = None,
    ) -> None:
        """
        Restore broker state after a process restart.
        Without this, LiveRunner.load_positions() leaves PaperBroker empty, and the
        next _reconcile treats every restored position as 'exchange stop fired' and
        fabricates a -1R close. Cash = starting + closed trade PnL − open fees/funding.

        Prefer ``last_marks`` from the runtime snapshot. Falls back to avg_entry only
        when marks are missing, and marks equity stale until the next ``on_price``.
        """
        self.cash = float(self.cfg.starting_equity) + float(closed_pnl)
        self.signed.clear()
        self.avg.clear()
        self.stops.clear()
        self.last.clear()
        self._stop_fills.clear()
        marks = {str(k): float(v) for k, v in (last_marks or {}).items()}
        for sym, pos in positions.items():
            self.cash -= float(pos.fees) + float(pos.funding)
            self.signed[sym] = pos.sign * float(pos.current_qty)
            self.avg[sym] = float(pos.avg_entry)
            self.stops[sym] = float(pos.current_stop)
            if sym in marks:
                self.last[sym] = marks[sym]
            else:
                self.last[sym] = float(pos.avg_entry)
        # Open book after restart → do not size new risk on possibly stale marks.
        self.equity_stale = bool(self.signed)


class MexcBroker(BaseBroker):
    """
    Live MEXC USDT-perp adapter via CCXT.

    Set ``MEXC_DRY_RUN=1`` (or ``dry_run=True``) to log intended API calls without
    sending orders — required for CI and local rehearsal. Live path still needs
    human verification of VERIFY notes against current CCXT/MEXC docs before
    ``LIVE_TRADING_ENABLED=1``.

    VERIFY audit (code intent — re-check before live):
    - openType=1 → isolated margin (MEXC swap convention in CCXT)
    - opening orders pass leverage + openType; reduceOnly omits them
    - stops use stopLossPrice + reduceOnly market trigger
    - cancel_order uses params trigger=True for conditional orders
    - balance total.USDT should include uPnL on swap accounts (confirm on account)
    - contractSize / amount_to_precision for lot sizing
    """

    def __init__(
        self,
        cfg: Config,
        *,
        dry_run: Optional[bool] = None,
        exchange: Any = None,
        api_fail_halt: int = 5,
    ):
        self.cfg = cfg
        self.dry_run = (
            bool(dry_run) if dry_run is not None else os.getenv("MEXC_DRY_RUN", "0").strip() == "1"
        )
        self._lev: dict = {}
        self._stop_id: dict = {}
        self._call_log: list[dict] = []
        self._api_fails = 0
        self._api_fail_halt = max(1, int(api_fail_halt))
        self._halted = False
        self._n = 0

        if exchange is not None:
            self.ex = exchange
            return

        if self.dry_run:
            # No network / no keys — parametric dry-run for CI and rehearsal.
            self.ex = None
            log.warning("MexcBroker DRY-RUN — orders are logged, not sent")
            return

        try:
            import ccxt  # type: ignore
        except ImportError as e:
            raise ConfigError("ccxt is required for MexcBroker") from e
        key, secret = os.getenv("MEXC_API_KEY"), os.getenv("MEXC_API_SECRET")
        if not key or not secret:
            raise ConfigError("MEXC_API_KEY / MEXC_API_SECRET not set")
        # Never log key/secret values.
        self.ex = ccxt.mexc(
            {
                "apiKey": key,
                "secret": secret,
                "enableRateLimit": True,
                "options": {"defaultType": "swap"},
            }
        )
        self.ex.load_markets()

    def _record(self, op: str, **payload) -> None:
        row = {"op": op, **payload}
        self._call_log.append(row)
        if self.dry_run:
            log.info("MEXC dry-run %s %s", op, {k: v for k, v in payload.items() if k != "secret"})

    def _api(self, op: str, fn):
        """Run a live CCXT call with consecutive-failure circuit breaker."""
        if self._halted:
            raise ExecutionError(
                f"MEXC API halted after {self._api_fail_halt} consecutive failures"
            )
        try:
            out = fn()
            self._api_fails = 0
            return out
        except Exception as e:  # noqa: BLE001
            self._api_fails += 1
            if self._api_fails >= self._api_fail_halt:
                self._halted = True
                log.error(
                    "MEXC API circuit open after %d failures on %s: %s",
                    self._api_fails,
                    op,
                    e,
                )
            raise

    def _contracts(self, symbol: str, qty: float) -> float:
        if self.ex is None:
            # Dry-run: treat 1 base unit = 1 contract step.
            return float(max(qty, 0.0))
        m = self.ex.market(symbol)
        size = float(m.get("contractSize") or 1.0)
        return float(self.ex.amount_to_precision(symbol, qty / size))

    def set_leverage(self, symbol, leverage) -> None:
        self._lev[symbol] = leverage
        params = {"openType": 1}  # VERIFY: isolated
        self._record("set_leverage", symbol=symbol, leverage=leverage, params=params)
        if self.dry_run:
            return
        try:
            self._api(
                "set_leverage",
                lambda: self.ex.set_leverage(leverage, symbol, params=params),
            )
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"set_leverage failed: {e}") from e

    def market_order(self, symbol, side, qty, ref_price, reduce_only=False) -> Fill:
        amount = self._contracts(symbol, qty)
        if amount <= 0:
            raise ExecutionError("order size rounds to zero contracts")
        params: dict = {"reduceOnly": reduce_only}
        if not reduce_only:  # VERIFY: leverage + openType on opening orders
            params.update(
                {
                    "leverage": self._lev.get(symbol, self.cfg.leverage),
                    "openType": 1,
                }
            )
        self._record(
            "create_order",
            symbol=symbol,
            side=side,
            amount=amount,
            ref_price=ref_price,
            params=dict(params),
        )
        if self.dry_run:
            self._n += 1
            px = float(ref_price)
            fee = px * float(qty) * self.cfg.fee_taker
            return Fill(
                f"dry-{self._n}",
                symbol,
                side,
                float(qty),
                px,
                fee,
                pd.Timestamp.now(tz="UTC"),
            )
        try:
            raw = self._api(
                "create_order",
                lambda: self.ex.create_order(symbol, "market", side, amount, None, params),
            )
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"market order failed: {e}") from e
        size = float(self.ex.market(symbol).get("contractSize") or 1.0)
        px = float(raw.get("average") or raw.get("price") or ref_price)
        fee = float((raw.get("fee") or {}).get("cost") or px * qty * self.cfg.fee_taker)
        return Fill(
            str(raw["id"]),
            symbol,
            side,
            float(raw.get("filled") or amount) * size,
            px,
            fee,
            pd.Timestamp.now(tz="UTC"),
        )

    def set_stop(self, symbol, direction, qty, stop) -> None:
        self.cancel_stop(symbol)
        side = "sell" if direction is Direction.LONG else "buy"
        amount = self._contracts(symbol, qty)
        if self.ex is None:
            stop_px = float(stop)
        else:
            stop_px = float(self.ex.price_to_precision(symbol, stop))
        params = {"stopLossPrice": stop_px, "reduceOnly": True}  # VERIFY trigger-market
        self._record(
            "set_stop",
            symbol=symbol,
            side=side,
            amount=amount,
            params=dict(params),
        )
        if self.dry_run:
            self._stop_id[symbol] = f"dry-stop-{symbol}"
            return
        try:
            raw = self._api(
                "set_stop",
                lambda: self.ex.create_order(
                    symbol,
                    "market",
                    side,
                    amount,
                    None,
                    params,
                ),
            )
        except Exception as e:  # noqa: BLE001
            raise ExecutionError(f"stop placement failed: {e}") from e
        self._stop_id[symbol] = raw.get("id")

    def cancel_stop(self, symbol) -> None:
        oid = self._stop_id.pop(symbol, None)
        if not oid:
            return
        self._record("cancel_stop", symbol=symbol, order_id=oid)
        if self.dry_run:
            return
        try:  # VERIFY: trigger=True for conditional; empty body may still raise
            self._api(
                "cancel_stop",
                lambda: self.ex.cancel_order(oid, symbol, params={"trigger": True}),
            )
        except Exception as e:  # noqa: BLE001
            log.warning("cancel_stop(%s) raised %s - check for orphaned stop orders!", symbol, e)

    def get_position(self, symbol) -> Optional[BrokerPosition]:
        self._record("fetch_positions", symbol=symbol)
        if self.dry_run:
            return None
        try:
            raws = self._api("fetch_positions", lambda: self.ex.fetch_positions([symbol]))
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
        self._record("fetch_balance")
        if self.dry_run:
            return float(self.cfg.starting_equity)
        try:
            bal = self._api("fetch_balance", lambda: self.ex.fetch_balance())
            # VERIFY: total USDT includes unrealized PnL on swap
            return float(bal["total"]["USDT"])
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
                ) from None
            return ohlcv_to_frame(rows, tf)
