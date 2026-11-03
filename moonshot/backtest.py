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
import moonshot.features as _moonshot_features
pull(_moonshot_features, globals())
import moonshot.data as _moonshot_data
pull(_moonshot_data, globals())
import moonshot.strategy as _moonshot_strategy
pull(_moonshot_strategy, globals())
import moonshot.position as _moonshot_position
pull(_moonshot_position, globals())
import moonshot.risk as _moonshot_risk
pull(_moonshot_risk, globals())
import moonshot.portfolio as _moonshot_portfolio
pull(_moonshot_portfolio, globals())

@dataclass
class BacktestResult:
    pyramiding: bool
    partial: bool
    leverage: int
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    expectancy_r: float
    exp_ci: tuple
    profit_factor: float
    max_drawdown: float
    liquidations: int
    trades: int
    win_rate: float
    top5_pct: float
    avg_adds: float
    halted: str
    equity_curve: pd.Series
    trade_list: list


class BacktestEngine:
    def __init__(self, data: dict, cfg: Config, start: Optional[pd.Timestamp] = None,
                 end: Optional[pd.Timestamp] = None):
        self.data, self.cfg, self.start, self.end = data, cfg, start, end
        self.strategy = make_strategy(cfg)
        self.pm = PositionManager(cfg)

    # ---- helpers ----
    def _slip(self, px: float, buy: bool) -> float:
        return px * (1 + self.cfg.slip) if buy else px * (1 - self.cfg.slip)

    def _equity(self) -> float:
        eq = self.cash
        for s, p in self.positions.items():
            eq += p.unrealized(self.last_close.get(s, p.avg_entry))
        return eq

    def _used_margin(self) -> float:
        return sum(p.margin() for p in self.positions.values())

    def _open_notional(self) -> float:
        return sum(p.current_qty * self.last_close.get(s, p.avg_entry)
                   for s, p in self.positions.items())

    def _net_notional(self) -> float:
        """Signed notional: long positive, short negative."""
        return sum(
            p.sign * p.current_qty * self.last_close.get(s, p.avg_entry)
            for s, p in self.positions.items()
        )

    # ---- main loop ----
    def run(self, label: Optional[tuple] = None) -> BacktestResult:
        cfg = self.cfg
        self.risk = RiskEngine(cfg, silent=True)
        self.cash = cfg.starting_equity
        self.positions: dict = {}
        self.pend_entry: dict = {}
        self.pend_act: dict = {}
        self.last_close: dict = {}
        self.cooldown: dict = {}
        self.trades: list = []
        curve: list = []

        stamps = sorted(set().union(*[set(sd.index) for sd in self.data.values()]))
        stamps = [t for t in stamps
                  if (self.start is None or t >= self.start) and (self.end is None or t < self.end)]
        if not stamps:
            raise DataError("no timestamps in the requested window")

        for ts in stamps:
            for sym, sd in self.data.items():
                i = sd.index.get(ts)
                if i is None:
                    continue
                self._step_symbol(sym, sd.bars[i], sd, ts)
            eq = self._equity()
            self.risk.update_equity(eq, ts)
            curve.append((ts, eq))

        # close whatever is still open at the last known close
        for sym in list(self.positions):
            px = self._slip(self.last_close[sym], buy=self.positions[sym].sign < 0)
            self._close(sym, px, stamps[-1], ExitReason.END_OF_DATA)
        curve[-1] = (stamps[-1], self.cash)
        return self._summarize(curve, label)

    def _step_symbol(self, sym: str, bar: Any, sd: SymbolData, ts: pd.Timestamp) -> None:
        cfg = self.cfg
        # 1) funding at the bar open for positions already held
        pos = self.positions.get(sym)
        if pos and ts.hour % cfg.funding_interval_hours == 0 and ts.minute == 0:
            cost = pos.sign * pos.current_qty * bar.open * cfg.funding_rate
            pos.funding += cost
            self.cash -= cost

        # 2) execute orders decided at the previous close, at this open
        if pos:
            acts = self.pend_act.pop(sym, [])
            if acts and self._apply_actions(sym, pos, acts, bar, sd, ts):
                return  # closed at the open
        else:
            sig = self.pend_entry.pop(sym, None)
            if sig and self.risk.trading_allowed(ts):
                self._open(sym, sig, bar, sd, ts)

        # 3) intrabar stop / liquidation check
        pos = self.positions.get(sym)
        if pos and self._check_stop(sym, pos, bar, ts):
            self.last_close[sym] = bar.close
            return

        # 4) decisions at this close (executed next open)
        self.last_close[sym] = bar.close
        pos = self.positions.get(sym)
        if pos:
            self.pend_act[sym] = self.pm.decide(
                pos, bar, allowed=self.risk.trading_allowed(ts), kill=self.risk.kill_switch_active,
                equity=self._equity(), used_margin=self._used_margin(), spec=sd.spec)
        elif (self.risk.trading_allowed(ts) and ts >= self.cooldown.get(sym, ts)
              and len(self.positions) + len(self.pend_entry) < cfg.max_open_positions):
            sig = self.strategy.on_bar(sym, bar)
            if sig:
                self.pend_entry[sym] = sig

    # ---- order execution ----
    def _open(self, sym: str, sig: Signal, bar: Any, sd: SymbolData, ts: pd.Timestamp) -> None:
        cfg = self.cfg
        buy = sig.direction is Direction.LONG
        entry = self._slip(bar.open, buy)
        stop = entry - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        dec = self.risk.evaluate(
            sig.direction, entry, stop, equity=self._equity(),
            open_notional=self._open_notional(), used_margin=self._used_margin(),
            open_positions=len(self.positions), spec=sd.spec, ts=ts,
            net_notional=self._net_notional(),
        )
        if not dec.approved:
            return
        fee = entry * dec.qty * cfg.fee_taker
        self.cash -= fee
        self.positions[sym] = Position.open(sym, sig.direction, dec.leverage, entry, dec.qty,
                                            stop, ts, fee)

    def _apply_actions(self, sym: str, pos: Position, acts: list, bar: Any, sd: SymbolData,
                       ts: pd.Timestamp) -> bool:
        """Returns True if the position was closed."""
        cfg = self.cfg
        buy_to_close = pos.sign < 0
        for a in acts:
            if a.kind is ActionKind.EXIT:
                self._close(sym, self._slip(bar.open, buy_to_close), ts, a.reason)
                return True
        for a in acts:
            if a.kind is ActionKind.PARTIAL:
                px = self._slip(bar.open, buy_to_close)
                qty = min(a.qty, pos.current_qty)
                fee = px * qty * cfg.fee_taker
                self.cash += pos.reduce(px, qty, fee) - fee
                pos.partial_done = True
            elif a.kind is ActionKind.ADD:
                px = self._slip(bar.open, buy=pos.sign > 0)
                qty, dec = self.pm.roll.size_add(pos, px, a.price, self._equity(),
                                                 self._used_margin(), sd.spec)
                if dec is AddDecision.ALLOW and qty:
                    fee = px * qty * cfg.fee_taker
                    self.cash -= fee
                    pos.add_lot(px, qty, ts, fee, a.price, pos.unrealized(px))
            elif a.kind is ActionKind.MOVE_STOP:
                pos.tighten_stop(a.price)
        return False

    def _check_stop(self, sym: str, pos: Position, bar: Any, ts: pd.Timestamp) -> bool:
        cfg = self.cfg
        liq = pos.liq_price(cfg.maintenance_margin_rate)
        long = pos.sign > 0
        hit_stop = bar.low <= pos.current_stop if long else bar.high >= pos.current_stop
        hit_liq = bar.low <= liq if long else bar.high >= liq
        if hit_stop and ((pos.current_stop > liq) if long else (pos.current_stop < liq)):
            raw = min(pos.current_stop, bar.open) if long else max(pos.current_stop, bar.open)
            if (raw <= liq) if long else (raw >= liq):  # gapped through the liquidation price
                self._close(sym, liq, ts, ExitReason.LIQUIDATION, liquidated=True)
                return True
            reason = ExitReason.TRAILING if pos.is_tighter(pos.initial_stop) and \
                pos.current_stop != pos.initial_stop else ExitReason.STOP
            self._close(sym, self._slip(raw, buy=not long), ts, reason)
            return True
        if hit_liq:
            self._close(sym, liq, ts, ExitReason.LIQUIDATION, liquidated=True)
            return True
        return False

    def _close(self, sym: str, px: float, ts: pd.Timestamp, reason: ExitReason,
               liquidated: bool = False) -> None:
        pos = self.positions.pop(sym)
        fee = 0.0 if liquidated else px * pos.current_qty * self.cfg.fee_taker
        res, cash_delta = pos.close_out(px, fee, ts, reason, liquidated)
        self.cash += cash_delta
        self.trades.append(res)
        self.risk.register_trade_result(res.r_multiple, ts)
        self.cooldown[sym] = ts + self.cfg.cooldown_bars * self.cfg.bar_td
        self.pend_act.pop(sym, None)

    # ---- metrics ----
    def _summarize(self, curve: list, label: Optional[tuple]) -> BacktestResult:
        cfg = self.cfg
        lab = label or (cfg.enable_pyramiding, cfg.enable_partial, cfg.leverage)
        # Defensive: never crash on a short/odd label (older call sites used ("recent",))
        if not isinstance(lab, (tuple, list)) or len(lab) < 3:
            lab = (cfg.enable_pyramiding, cfg.enable_partial, cfg.leverage)
        eq = pd.Series([e for _, e in curve], index=pd.DatetimeIndex([t for t, _ in curve]))
        final = float(eq.iloc[-1])
        days = max((eq.index[-1] - eq.index[0]).total_seconds() / 86400.0, 1e-9)
        total_ret = final / cfg.starting_equity - 1.0
        cagr = (final / cfg.starting_equity) ** (365.0 / days) - 1.0 if days >= 90 and final > 0 else float("nan")

        dr = eq.resample("1D").last().dropna().pct_change().dropna()
        sharpe = float(dr.mean() / dr.std() * math.sqrt(365)) if len(dr) > 2 and dr.std() > 0 else float("nan")
        dd_dev = math.sqrt(float(np.mean(np.minimum(dr.to_numpy(), 0.0) ** 2))) if len(dr) else 0.0
        sortino = float(dr.mean() / dd_dev * math.sqrt(365)) if dd_dev > 0 else float("nan")
        max_dd = float(((eq.cummax() - eq) / eq.cummax().replace(0, np.nan)).max()) if len(eq) else 0.0

        t = self.trades
        r = np.array([x.r_multiple for x in t]) if t else np.array([])
        pnl = [x.pnl for x in t]
        gw, gl = sum(p for p in pnl if p > 0), -sum(p for p in pnl if p < 0)
        wins = sorted((p for p in pnl if p > 0), reverse=True)
        ci = (float("nan"), float("nan"))
        if len(r) >= 10:
            rng = np.random.default_rng(0)
            means = rng.choice(r, size=(2000, len(r)), replace=True).mean(axis=1)
            ci = (float(np.percentile(means, 5)), float(np.percentile(means, 95)))
        return BacktestResult(
            pyramiding=bool(lab[0]), partial=bool(lab[1]), leverage=int(lab[2]), final_equity=final,
            total_return=total_ret, cagr=cagr, sharpe=sharpe, sortino=sortino,
            expectancy_r=float(r.mean()) if len(r) else 0.0, exp_ci=ci,
            profit_factor=gw / gl if gl > 0 else (float("inf") if gw > 0 else 0.0),
            max_drawdown=max_dd, liquidations=sum(x.liquidated for x in t), trades=len(t),
            win_rate=float((r > 0).mean()) if len(r) else 0.0,
            top5_pct=100.0 * sum(wins[:5]) / gw if gw > 0 else 0.0,
            avg_adds=float(np.mean([x.max_adds for x in t])) if t else 0.0,
            halted=self.risk.halt_reason, equity_curve=eq, trade_list=t)


def run_grid(data: dict, cfg: Config, start=None, end=None) -> list:
    out = []
    for p, q, lev in product([False, True], [False, True], [3, 5]):
        c = replace(cfg, enable_pyramiding=p, enable_partial=q, leverage=lev,
                    max_account_leverage=float(min(cfg.max_account_leverage, lev)))
        out.append(BacktestEngine(data, c, start, end).run((p, q, lev)))
    return out


def results_as_frame(results: list) -> pd.DataFrame:
    """Tabular view of BacktestResult list for CLI / Streamlit."""
    rows = []
    for r in results:
        lo, hi = r.exp_ci
        rows.append({
            "pyr": r.pyramiding, "part": r.partial, "lev": r.leverage,
            "final": round(r.final_equity, 2), "ret": r.total_return,
            "cagr": r.cagr, "sharpe": r.sharpe, "sortino": r.sortino,
            "exp_r": r.expectancy_r, "ci_lo": lo, "ci_hi": hi,
            "pf": r.profit_factor if math.isfinite(r.profit_factor) else float("nan"),
            "max_dd": r.max_drawdown, "liq": r.liquidations, "n": r.trades,
            "win": r.win_rate, "top5_pct": r.top5_pct, "avg_adds": r.avg_adds,
            "halt": (r.halted or "")[:48],
        })
    return pd.DataFrame(rows)


def results_table(results: list) -> str:
    df = results_as_frame(results)
    if df.empty:
        return "(no results)"
    show = df.copy()
    show["lev"] = show["lev"].map(lambda x: f"{x}x")
    show["ret"] = show["ret"].map(lambda x: f"{x:+.0%}")
    show["cagr"] = show["cagr"].map(lambda x: "n/a" if math.isnan(x) else f"{x:.0%}")
    show["sharpe"] = show["sharpe"].map(lambda x: "n/a" if math.isnan(x) else round(x, 2))
    show["exp_r"] = show["exp_r"].map(lambda x: round(x, 3))
    show["ci"] = [
        "n/a" if math.isnan(lo) else f"[{lo:.2f},{hi:.2f}]"
        for lo, hi in zip(show["ci_lo"], show["ci_hi"])
    ]
    show["pf"] = show["pf"].map(lambda x: round(x, 2) if x == x else "n/a")
    show["max_dd"] = show["max_dd"].map(lambda x: f"{x:.0%}")
    show["win"] = show["win"].map(lambda x: f"{x:.0%}")
    show["top5_pct"] = show["top5_pct"].map(lambda x: f"{x:.0f}")
    show = show.rename(columns={"exp_r": "exp(R)", "ci": "90%CI", "max_dd": "maxDD",
                                "top5_pct": "top5%", "cagr": "CAGR", "sharpe": "Sharpe"})
    out = show[["pyr", "part", "lev", "final", "ret", "CAGR", "Sharpe", "exp(R)", "90%CI",
                "pf", "maxDD", "liq", "n", "win", "top5%", "halt"]]
    try:
        return out.to_markdown(index=False)
    except ImportError:
        return out.to_string(index=False)


def verdicts(results: list) -> list:
    """KEEP only if the candidate is better AND still acceptable (non-negative expectancy)."""
    idx = {(r.pyramiding, r.partial, r.leverage): r for r in results}
    out = []
    for part, lev in product([False, True], [3, 5]):
        a, b = idx[(False, part, lev)], idx[(True, part, lev)]
        exp_ok = b.expectancy_r > a.expectancy_r * 1.10 if a.expectancy_r > 0 else b.expectancy_r > a.expectancy_r
        liq_a = a.liquidations / a.trades if a.trades else 0.0
        liq_b = b.liquidations / b.trades if b.trades else 0.0
        better = (exp_ok or b.top5_pct > a.top5_pct) and (b.max_drawdown - a.max_drawdown <= 0.05) \
            and (liq_b - liq_a <= 0.02)
        keep = better and b.expectancy_r > 0 and b.trades >= 10
        tag = "KEEP" if keep else ("DROP (still losing)" if better and b.expectancy_r <= 0 else "DROP")
        out.append(f"pyramiding @ partial={part}, {lev}x: {tag} "
                   f"(exp {a.expectancy_r:+.2f}R -> {b.expectancy_r:+.2f}R, "
                   f"DD {a.max_drawdown:.0%} -> {b.max_drawdown:.0%})")
    for pyr, lev in product([False, True], [3, 5]):
        a, b = idx[(pyr, False, lev)], idx[(pyr, True, lev)]
        better = b.final_equity >= a.final_equity * 1.15 and (
            not math.isnan(b.sharpe) and not math.isnan(a.sharpe) and b.sharpe > a.sharpe
        )
        keep = better and b.expectancy_r > 0 and b.trades >= 10
        tag = "KEEP" if keep else "DROP"
        out.append(f"partial@6R @ pyramiding={pyr}, {lev}x: {tag} "
                   f"(final {a.final_equity:.2f} -> {b.final_equity:.2f})")
    hi = idx[(True, True, 5)]
    base3 = idx[(True, False, 3)]
    liq_rate = hi.liquidations / hi.trades if hi.trades else 0.0
    if liq_rate > 0.10 or hi.max_drawdown > 0.40 or base3.expectancy_r > hi.expectancy_r:
        out.append("leverage: PREFER 3x (safer; 5x not clearly better on this sample)")
    else:
        out.append("leverage: 5x did not breach liq/DD limits — still prefer 3x until OOS n is large")
    return out


def promotion_reading(ins: "BacktestResult", oos: "BacktestResult") -> str:
    """Honest label for train vs test — never celebrate a losing train as success."""
    if ins.trades < 10 or oos.trades < 10:
        return "thin sample"
    if ins.expectancy_r <= 0 and oos.expectancy_r > 0:
        return "train failed · test lucky?"
    if ins.expectancy_r <= 0 and oos.expectancy_r <= 0:
        return "train failed"
    good_is = ins.expectancy_r > 0 and (math.isnan(ins.sharpe) or ins.sharpe > 0.3)
    weak_oos = oos.expectancy_r < ins.expectancy_r * 0.5 or (
        not math.isnan(oos.sharpe) and not math.isnan(ins.sharpe) and oos.sharpe < ins.sharpe * 0.5
    )
    if good_is and weak_oos:
        return "overfit risk"
    if oos.expectancy_r > 0 and oos.trades >= 30 and ins.expectancy_r > 0:
        return "both halves OK"
    if oos.expectancy_r > 0 and ins.expectancy_r > 0:
        return "both positive · thin"
    return "check manually"


