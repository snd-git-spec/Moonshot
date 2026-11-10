from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    Direction,
    InstrumentSpec,
    RiskDecision,
)
from moonshot.config import (
    Config,
)


class RiskEngine:
    """Single gatekeeper for new risk. Time always comes from the caller (backtest-safe).

    Live/paper (``silent=False``): peak-DD / expectancy trips are permanent kills.
    Backtest (``silent=True``): same trips become a timed circuit-breaker pause, then
    trading resumes so later regimes are still evaluated.
    """

    def __init__(self, cfg: Config, silent: bool = False):
        self.cfg = cfg
        self.silent = silent  # True in backtests
        self.reset()

    def reset(self, equity: Optional[float] = None) -> None:
        self.equity = equity if equity is not None else self.cfg.starting_equity
        self.peak_equity = self.equity
        self.daily_start_equity = self.equity
        self.current_day = None
        self.daily_halt = False
        self.pause_until: Optional[pd.Timestamp] = None
        self.consecutive_losses = 0
        self.rolling_r: list = []
        self.kill_switch_active = False
        self.halt_reason = ""
        self.circuit_trips: list[str] = []
        self._reanchor_peak_on_resume = False
        # Paper-vs-backtest drift band (set from champion / last improve)
        self.drift_min_trades = 20
        self.drift_exp_floor = -0.20
        self.drift_soft_slack = 0.30
        self.backtest_soft_ret: Optional[float] = None
        self.backtest_week_ret: Optional[float] = None
        self.drift_halt = False
        # Last-N paper closes all red — block new entries until hunt deploys
        self.bleed_halt = False

    def set_backtest_band(
        self,
        *,
        soft_ret: Optional[float] = None,
        week_ret: Optional[float] = None,
        min_trades: int = 20,
        exp_floor: float = -0.20,
        soft_slack: float = 0.30,
    ) -> None:
        """Anchor paper drift checks to the promoted book's backtest numbers."""
        self.backtest_soft_ret = None if soft_ret is None else float(soft_ret)
        self.backtest_week_ret = None if week_ret is None else float(week_ret)
        self.drift_min_trades = max(5, int(min_trades))
        self.drift_exp_floor = float(exp_floor)
        self.drift_soft_slack = float(soft_slack)

    def check_paper_drift(
        self,
        *,
        n_trades: int,
        paper_exp: float,
        paper_ret: Optional[float] = None,
    ) -> bool:
        """Permanent halt when paper drifts outside the backtest band (live/paper only)."""
        if self.drift_halt or self.kill_switch_active:
            return True
        if int(n_trades) < int(self.drift_min_trades):
            return False
        reasons: list[str] = []
        if float(paper_exp) < float(self.drift_exp_floor):
            reasons.append(
                f"paper exp {float(paper_exp):+.2f}R < floor {self.drift_exp_floor:+.2f}R"
            )
        if (
            paper_ret is not None
            and self.backtest_soft_ret is not None
            and float(paper_ret) < float(self.backtest_soft_ret) - float(self.drift_soft_slack)
        ):
            reasons.append(
                f"paper ret {float(paper_ret):+.0%} << backtest soft "
                f"{float(self.backtest_soft_ret):+.0%} "
                f"(slack {self.drift_soft_slack:.0%})"
            )
        if not reasons:
            return False
        why = "paper drift: " + "; ".join(reasons)
        self.drift_halt = True
        # Always permanent — do not soft-pause like backtest circuit breakers.
        self.kill_switch_active = True
        self.halt_reason = why
        if not self.silent:
            log.warning("KILL SWITCH: %s", why)
        return True

    def update_equity(self, equity: float, ts: pd.Timestamp) -> None:
        equity = max(equity, 0.0)
        day = ts.date()
        if self.current_day is None:
            self.daily_start_equity = equity
        elif day != self.current_day:
            self.daily_start_equity = self.equity
            self.daily_halt = False
        self.current_day = day
        self.equity = equity
        self.peak_equity = max(self.peak_equity, equity)

        dd = (self.peak_equity - equity) / self.peak_equity if self.peak_equity > 0 else 0.0
        if dd >= self.cfg.max_peak_drawdown_pct:
            self._trip(
                f"peak drawdown {dd:.1%} >= {self.cfg.max_peak_drawdown_pct:.0%}",
                ts,
            )
        if self.daily_start_equity > 0:
            loss = (self.daily_start_equity - equity) / self.daily_start_equity
            if loss >= self.cfg.max_daily_loss_pct and not self.daily_halt:
                self.daily_halt = True
                if not self.silent:
                    log.warning("DAILY HALT: loss %.1f%% (resumes next UTC day)", loss * 100)

    def register_trade_result(self, r: float, ts: pd.Timestamp) -> None:
        self.rolling_r.append(r)
        self.consecutive_losses = self.consecutive_losses + 1 if r < 0 else 0
        if self.consecutive_losses >= self.cfg.streak_losses:
            self.pause_until = ts + pd.Timedelta(hours=self.cfg.streak_pause_hours)
            self.consecutive_losses = 0
            if not self.silent:
                log.warning("LOSS-STREAK PAUSE until %s", self.pause_until)
        w = self.cfg.expectancy_window
        if len(self.rolling_r) >= w:
            exp = float(np.mean(self.rolling_r[-w:]))
            if exp < self.cfg.expectancy_floor:
                self._trip(
                    f"rolling {w}-trade expectancy {exp:.2f}R < {self.cfg.expectancy_floor}",
                    ts,
                )

    def trading_allowed(self, ts: pd.Timestamp) -> bool:
        if self.kill_switch_active or self.daily_halt or self.bleed_halt:
            return False
        if self.pause_until is not None and ts < self.pause_until:
            return False
        if self.pause_until is not None and ts >= self.pause_until:
            # Circuit closed — resume with a fresh peak / expectancy window.
            if self._reanchor_peak_on_resume:
                self.peak_equity = max(self.equity, 1e-9)
                self._reanchor_peak_on_resume = False
            self.rolling_r.clear()
            self.consecutive_losses = 0
            self.pause_until = None
        return True

    def set_bleed_halt(self, on: bool, reason: str = "") -> None:
        """Paper-only: stop new entries while last-N closes are all losers."""
        was = bool(self.bleed_halt)
        self.bleed_halt = bool(on)
        if on:
            self.halt_reason = reason or "paper bleed halt — waiting for hunt deploy"
            if not was and not self.silent:
                log.warning("BLEED HALT: %s", self.halt_reason)
        elif was:
            if str(self.halt_reason or "").startswith("paper bleed"):
                self.halt_reason = ""
            if not self.silent:
                log.info("bleed halt cleared — new entries allowed")

    def evaluate(
        self,
        direction: Direction,
        entry: float,
        stop: float,
        *,
        equity: float,
        open_notional: float,
        used_margin: float,
        open_positions: int,
        spec: InstrumentSpec,
        ts: pd.Timestamp,
        net_notional: float = 0.0,
    ) -> RiskDecision:
        cfg = self.cfg
        no = lambda why: RiskDecision(False, why)  # noqa: E731
        if not self.trading_allowed(ts):
            return no(f"trading halted ({self.halt_reason or 'daily/pause'})")
        if open_positions >= cfg.max_open_positions:
            return no("max open positions")
        if equity <= 0:
            return no("zero equity")
        stop_dist = (entry - stop) * direction.sign
        if stop_dist <= 0:
            return no("invalid stop")

        lev = int(cfg.leverage)
        margin_pct = cfg.aggressive_margin_pct if cfg.aggressive_mode else cfg.initial_margin_pct
        q_risk = equity * cfg.risk_pct / stop_dist
        q_margin = equity * margin_pct * lev / entry
        q_free = max(0.0, equity - used_margin) * lev / entry
        q_lev = max(0.0, equity * cfg.max_account_leverage - open_notional) / entry
        net_cap = max(0.0, equity * float(cfg.max_net_leverage))
        room = max(0.0, net_cap - direction.sign * net_notional)
        q_net = room / entry if entry > 0 else 0.0
        qty = spec.round_qty(min(q_risk, q_margin, q_free, q_lev, q_net))
        if qty <= 0:
            return no("size below exchange minimum (or net-directional cap)")
        if qty * entry < cfg.min_notional:
            return no(f"notional {qty * entry:.2f} < min {cfg.min_notional}")
        proposed_net = abs(net_notional + direction.sign * qty * entry)
        if equity > 0 and proposed_net / equity > float(cfg.max_net_leverage) + 1e-9:
            return no(
                f"net directional {proposed_net / equity:.2f}x > max_net_leverage "
                f"{cfg.max_net_leverage:.2f}x"
            )

        liq_dist_pct = 1.0 / lev - cfg.maintenance_margin_rate
        stop_pct = stop_dist / entry
        thr = 0.65 if cfg.aggressive_mode else 0.5
        if stop_pct > liq_dist_pct * thr:
            return no(f"stop {stop_pct:.2%} > {thr:.0%} of liq distance {liq_dist_pct:.2%}")
        return RiskDecision(True, "ok", qty, qty * entry / lev, lev, qty * stop_dist)

    def _trip(self, reason: str, ts: pd.Timestamp) -> None:
        """Live: permanent kill. Backtest: timed pause then resume."""
        if self.kill_switch_active:
            return
        if self.silent:
            hours = max(1.0, float(getattr(self.cfg, "kill_resume_hours", 168.0)))
            until = pd.Timestamp(ts) + pd.Timedelta(hours=hours)
            if self.pause_until is None or until > self.pause_until:
                self.pause_until = until
            self._reanchor_peak_on_resume = True
            self.rolling_r.clear()
            self.consecutive_losses = 0
            note = f"{reason} → backtest pause {hours / 24.0:.0f}d until {until:%Y-%m-%d}"
            self.halt_reason = note
            self.circuit_trips.append(note)
            return
        self.kill_switch_active = True
        self.halt_reason = reason
        log.warning("KILL SWITCH: %s", reason)

    # Back-compat alias used by older call sites / tests
    def _kill(self, reason: str, ts: Optional[pd.Timestamp] = None) -> None:
        if ts is None:
            ts = pd.Timestamp.now(tz="UTC")
        self._trip(reason, ts)
