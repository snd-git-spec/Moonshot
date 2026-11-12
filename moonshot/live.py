from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("moonshot")

from moonshot.types import (
    ActionKind,
    AddDecision,
    BotError,
    Direction,
    ExecutionError,
    ExitReason,
    Signal,
)
from moonshot.config import (
    Config,
)
from moonshot.features import (
    _btc_symbol_in,
    _is_btc_symbol,
    _regime_to_trend,
    build_features,
)
from moonshot.data import (
    clear_stale_live_caches,
    fetch_historical,
)
from moonshot.strategy import (
    make_strategy,
)
from moonshot.position import (
    Position,
)
from moonshot.risk import (
    RiskEngine,
)
from moonshot.portfolio import (
    PositionManager,
)
from moonshot.brokers import (
    BaseBroker,
    MarketData,
    PaperBroker,
)
from moonshot.persist import (
    Notifier,
    Persistence,
)

from moonshot.paths import PAPER_RUNTIME_PATH
import contextlib


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# Drop pending entries older than this many bars (also defined in hunt.scoring).
PEND_ENTRY_MAX_BARS = 2


def _live_feed(kind: str, message: str, **extra) -> None:
    # Shared feed lives in hunt.constants; lazy import avoids live↔hunt cycle at load.
    from moonshot.hunt.constants import _live_feed as _write_live_feed

    _write_live_feed(kind, message, **extra)


class LiveRunner:
    def __init__(
        self, cfg: Config, broker: BaseBroker, market: MarketData, symbols: list, live: bool
    ):
        self.cfg, self.broker, self.market, self.symbols, self.live = (
            cfg,
            broker,
            market,
            symbols,
            live,
        )
        self.risk = RiskEngine(cfg)
        self.strategy = make_strategy(cfg)
        self.pm = PositionManager(cfg)
        self.db = Persistence(cfg.db_path, mode="live" if live else "paper")
        self.notify = Notifier()
        self.positions: dict = self.db.load_positions()
        self.specs = {s: market.instrument(s) for s in symbols}
        self.last_bar: dict = {}
        self.last_price: dict = {}
        self.cooldown: dict = {}
        self.blocked: set = set()
        self.pend_entry: dict = {}  # sym -> Signal; fill once next bar opens
        self.pend_after: dict = {}  # sym -> Timestamp (earliest fill time = signal bar + 1)
        self.pend_exit: dict = {}  # sym -> (ExitReason, after_ts); fill at next open
        self._seen_broker_pos: set = set()  # symbols confirmed on broker this session
        self._last_heartbeat = 0.0
        self._last_funding_key: Optional[pd.Timestamp] = None
        self._bars_seen = 0
        self._signals_seen = 0
        self._cycle_n = 0
        self._cycle_id = ""
        self._last_cycle_ts: Optional[pd.Timestamp] = None
        # When set, paper/offline replay uses this instead of wall clock.
        self._clock: Optional[pd.Timestamp] = None
        self._btc_trend = 0.0
        self._raw: dict = {}  # symbol -> OHLCV for prepare_data parity with hunt
        self._book_feats: dict = {}  # symbol -> feature frame from prepare_data
        self._stale_log_at: dict = {}  # symbol -> last stale warning monotonic time
        self._cache_age_sec: Optional[float] = None
        self._last_expectancy_attempt = 0.0
        self._was_equity_stale = False
        # Paper must rebuild cash from closed trades even with 0 open positions.
        # Old gate `and self.positions` left wallet stuck at $50 after flat restarts
        # while Live still counted closed trades → "19 closed, balance $50".
        if not live and isinstance(broker, PaperBroker):
            closed = self.db.closed_pnl_sum(mode="paper")
            if self.positions or abs(closed) > 1e-12:
                marks = self._peek_runtime_last_marks()
                broker.rehydrate(self.positions or {}, closed, last_marks=marks)
                self._was_equity_stale = bool(broker.equity_stale)
                log.info(
                    "paper rehydrated %d open · closed_pnl=%+.2f · cash=%.2f · stale=%s",
                    len(self.positions),
                    closed,
                    broker.cash,
                    broker.equity_stale,
                )
                with contextlib.suppress(Exception):
                    self.db.log_event(
                        self._now(),
                        "*",
                        "rehydrate",
                        {
                            "open": len(self.positions),
                            "closed_pnl": float(closed),
                            "equity_stale": bool(broker.equity_stale),
                            "cycle_id": self._cycle_id or "",
                        },
                    )
        # Poll BTC first so alt entries see a fresh regime stamp
        if cfg.require_btc_regime:
            btc = _btc_symbol_in(symbols)
            if btc and symbols[0] != btc:
                self.symbols = [btc] + [s for s in symbols if s != btc]
        self._load_drift_band()
        with contextlib.suppress(Exception):
            self.db.log_event(
                self._now(),
                "*",
                "restart",
                {
                    "live": bool(live),
                    "open": len(self.positions),
                    "config_hash": self._config_hash_safe(),
                    "cycle_id": self._cycle_id or "",
                },
            )

    def _load_drift_band(self) -> None:
        """Pull champion / last-improve soft+week returns for paper drift halt."""
        try:
            from moonshot.hunt.constants import _load_autopilot_state

            state = _load_autopilot_state() or {}
            champ = state.get("champion") or {}
            li = state.get("last_improve") or {}
            soft = champ.get("soft_ret")
            if soft is None:
                soft = (li.get("best") or {}).get("soft_return")
            week = champ.get("week_ret")
            if week is None:
                week = (li.get("best") or {}).get("week_return")
            self.risk.set_backtest_band(
                soft_ret=None if soft is None else float(soft),
                week_ret=None if week is None else float(week),
            )
        except Exception as e:
            log.debug("drift band load skipped: %s", e)

    def _runtime_path(self) -> Path:
        return Path(self.cfg.db_path).resolve().parent / PAPER_RUNTIME_PATH.name

    def _now(self) -> pd.Timestamp:
        """Wall clock, or injected `_clock` for offline fixture replay."""
        if self._clock is not None:
            return pd.Timestamp(self._clock)
        return pd.Timestamp.now(tz="UTC")

    def _peek_runtime_last_marks(self) -> dict:
        """Read last_mark from disk before rehydrate (init order)."""
        path = self._runtime_path()
        if not path.exists():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            marks = data.get("last_mark") or {}
            return {str(k): float(v) for k, v in marks.items()} if isinstance(marks, dict) else {}
        except Exception:
            return {}

    def _save_runtime(self) -> None:
        """Persist risk / cooldown / pending queues / marks across paper restarts."""
        if self.live:
            return
        try:
            cool = {
                s: pd.Timestamp(t).isoformat() for s, t in self.cooldown.items() if t is not None
            }
            pend_after = {
                s: pd.Timestamp(t).isoformat() for s, t in self.pend_after.items() if t is not None
            }
            pend_exit = {
                s: {
                    "reason": (r.value if hasattr(r, "value") else str(r)),
                    "after": pd.Timestamp(a).isoformat(),
                }
                for s, (r, a) in self.pend_exit.items()
            }
            last_mark = (
                self.broker.mark_snapshot()
                if isinstance(self.broker, PaperBroker)
                else {s: float(p) for s, p in self.last_price.items()}
            )
            payload = {
                "ts": _utc_now().isoformat(),
                "peak_equity": float(self.risk.peak_equity),
                "equity": float(self.risk.equity),
                "kill": bool(self.risk.kill_switch_active),
                "halt_reason": str(self.risk.halt_reason or ""),
                "daily_halt": bool(self.risk.daily_halt),
                "streak": int(self.risk.consecutive_losses),
                "pause_until": (
                    None
                    if self.risk.pause_until is None
                    else pd.Timestamp(self.risk.pause_until).isoformat()
                ),
                "circuit_trips": list(getattr(self.risk, "circuit_trips", []) or [])[-10:],
                "rolling_r": [float(x) for x in list(self.risk.rolling_r)[-50:]],
                "last_mark": last_mark,
                "equity_stale": bool(getattr(self.broker, "equity_stale", False))
                if isinstance(self.broker, PaperBroker)
                else False,
                "cooldown": cool,
                "pend_after": pend_after,
                "pend_exit": pend_exit,
                "blocked": sorted(self.blocked),
                "last_cycle_id": str(self._cycle_id or ""),
                "last_cycle_ts": (
                    None
                    if self._last_cycle_ts is None
                    else pd.Timestamp(self._last_cycle_ts).isoformat()
                ),
                "bars_seen": int(self._bars_seen),
            }
            path = self._runtime_path()
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as e:
            log.warning("paper runtime save failed: %s", e)

    def _restore_runtime(self) -> None:
        path = self._runtime_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            log.warning("paper runtime load failed: %s", e)
            return
        try:
            peak = float(data.get("peak_equity") or 0)
            if peak > 0:
                self.risk.peak_equity = max(float(self.risk.peak_equity), peak)
            if data.get("kill"):
                self.risk.kill_switch_active = True
                self.risk.halt_reason = str(data.get("halt_reason") or "restored")
            self.risk.daily_halt = bool(data.get("daily_halt"))
            self.risk.consecutive_losses = int(data.get("streak") or 0)
            pu = data.get("pause_until")
            if pu:
                self.risk.pause_until = pd.Timestamp(pu)
            trips = data.get("circuit_trips") or []
            if isinstance(trips, list) and trips:
                self.risk.circuit_trips = [str(x) for x in trips]
            # Never revive a stale bleed halt_reason — trading_allowed uses bleed_halt,
            # but logs/UI treat halt_reason as the scoreboard and look "stuck".
            hr = str(data.get("halt_reason") or "")
            if hr and not self.risk.halt_reason and not hr.startswith("paper bleed"):
                self.risk.halt_reason = hr
            rr = data.get("rolling_r") or []
            if isinstance(rr, list) and rr:
                self.risk.rolling_r = [float(x) for x in rr]
            marks = data.get("last_mark") or {}
            if isinstance(self.broker, PaperBroker) and isinstance(marks, dict):
                for s, px in marks.items():
                    self.broker.last[str(s)] = float(px)
                    self.last_price[str(s)] = float(px)
                # Rehydrate already set stale if opens exist; keep stale until on_price
                if self.broker.signed and data.get("equity_stale", True):
                    self.broker.equity_stale = True
            cid = data.get("last_cycle_id")
            if cid:
                self._cycle_id = str(cid)
            cts = data.get("last_cycle_ts")
            if cts:
                self._last_cycle_ts = pd.Timestamp(cts)
            if data.get("bars_seen") is not None:
                self._bars_seen = int(data.get("bars_seen") or 0)
            for s, t in (data.get("cooldown") or {}).items():
                self.cooldown[str(s)] = pd.Timestamp(t)
            for s, _t in (data.get("pend_after") or {}).items():
                # Signals themselves aren't restored — drop stale queue keys
                self.pend_after.pop(str(s), None)
            for s, blob in (data.get("pend_exit") or {}).items():
                try:
                    reason = ExitReason(str(blob.get("reason")))
                    after = pd.Timestamp(blob.get("after"))
                    self.pend_exit[str(s)] = (reason, after)
                except Exception:
                    pass
            for s in data.get("blocked") or []:
                self.blocked.add(str(s))
            log.info(
                "paper runtime restored (peak=%.2f kill=%s pause=%s cooldown=%d)",
                self.risk.peak_equity,
                self.risk.kill_switch_active,
                self.risk.pause_until,
                len(self.cooldown),
            )
        except Exception as e:
            log.warning("paper runtime apply failed: %s", e)

    def run_forever(self) -> None:
        mode = "LIVE" if self.live else "PAPER"
        # Unblock paper if live-sized CSVs froze behind wall clock
        try:
            clear_stale_live_caches(
                tf=str(self.cfg.timeframe),
                bars=int(self.cfg.live_history_bars),
            )
        except Exception as e:
            log.warning("stale cache clear failed: %s", e)
        self.notify.send(
            f"moonshot started ({mode}) symbols={self.symbols} "
            f"tf={self.cfg.timeframe} lev={self.cfg.leverage}x "
            f"hist={self.cfg.live_history_bars}"
        )
        eq0 = self.broker.get_equity()
        snap = None if self.live else self.db.load_risk(mode="paper")
        self.risk.reset(eq0)
        if snap and not self.live:
            try:
                peak = float(snap.get("peak") or 0)
                if peak > eq0:
                    self.risk.peak_equity = peak
                if int(snap.get("kill") or 0):
                    self.risk.kill_switch_active = True
                    self.risk.halt_reason = str(snap.get("halt_reason") or "restored")
                if int(snap.get("daily_halt") or 0):
                    self.risk.daily_halt = True
                self.risk.consecutive_losses = int(snap.get("streak") or 0)
                if snap.get("pause_until"):
                    self.risk.pause_until = pd.Timestamp(snap["pause_until"])
            except Exception as e:
                log.warning("risk snapshot restore failed: %s", e)
        if not self.live:
            self._restore_runtime()
            # Re-arm bleed circuit from DB + autopilot flag (survives restart).
            with contextlib.suppress(Exception):
                from moonshot.hunt.constants import (
                    PAPER_BLEED_HALT_ENTRIES,
                    _load_autopilot_state,
                )
                from moonshot.hunt.scoring import _paper_bleed

                if PAPER_BLEED_HALT_ENTRIES:
                    st = _load_autopilot_state()
                    info = _paper_bleed()
                    if info.get("bleed") or st.get("paper_bleed_halt"):
                        why = (
                            f"paper bleed halt — last {info.get('n') or '?'} closes red "
                            f"(pnl {float(info.get('pnl') or 0):+.2f}) · waiting for hunt deploy"
                        )
                        self.risk.set_bleed_halt(True, why)
                    else:
                        self.risk.set_bleed_halt(False)
        try:
            while True:
                if self.cfg.require_btc_regime and not _btc_symbol_in(self.symbols):
                    try:
                        self._refresh_btc_regime()
                    except Exception as e:
                        log.warning("BTC regime refresh failed: %s", e)
                for sym in self.symbols:
                    try:
                        self._poll(sym)
                    except BotError as e:
                        log.error("%s: %s", sym, e)
                        self.notify.send(f"ERROR {sym}: {e}")
                    except Exception as e:  # noqa: BLE001 — keep paper alive on MEXC timeouts
                        log.warning("%s poll failed (continuing): %s", sym, e)
                self._heartbeat()
                time.sleep(self.cfg.poll_seconds)
        except KeyboardInterrupt:
            log.info("stopped by user (positions keep their exchange-side stops)")
        finally:
            self.db.close()

    def _heartbeat(self) -> None:
        """Wall-clock pulse so the Live page does not look frozen between candles."""
        now = time.time()
        if now - self._last_heartbeat < 60:
            return
        self._last_heartbeat = now
        ts = self._now()
        self._note_equity_stale_cleared(ts)
        equity = self.broker.get_equity()
        self.risk.update_equity(equity, ts)
        self.db.log_equity(ts, equity, self.risk.peak_equity)
        self._log_risk_cycle(ts, equity, snapshot=True)
        self._maybe_expectancy_rollup()
        # next candle close (approximate)
        floored = ts.floor(self.cfg.pandas_tf)
        next_close = floored + self.cfg.bar_td
        mins = max(0, int((next_close - ts).total_seconds() // 60))
        age = self._cache_age_sec
        age_s = f" · cache {int(age)}s" if age is not None else ""
        msg = (
            f"Paper alive · {self.cfg.timeframe} · {self.cfg.strategy} · "
            f"{self.cfg.leverage}x · {len(self.positions)} open · "
            f"{len(self.symbols)} coins · next candle decision in ~{mins}m · "
            f"bars={self._bars_seen} setups={self._signals_seen}{age_s}"
        )
        log.info(
            "%s cycle_id=%s equity=%.4f halt=%s",
            msg,
            self._cycle_id or "-",
            equity,
            self.risk.halt_reason or "-",
            extra={
                "cycle_id": self._cycle_id or "",
                "equity": float(equity),
                "halt": str(self.risk.halt_reason or ""),
            },
        )
        with contextlib.suppress(Exception):
            _live_feed(
                "paper_tick",
                msg,
                open=len(self.positions),
                next_mins=mins,
                timeframe=self.cfg.timeframe,
                equity=round(equity, 2),
                bars_seen=int(self._bars_seen),
                setups_seen=int(self._signals_seen),
                hist_bars=int(self.cfg.live_history_bars or 0),
                adx_enter=float(self.cfg.adx_enter),
                vol_mult=float(self.cfg.volume_multiplier),
                require_rising_adx=bool(self.cfg.require_rising_adx),
                leverage=int(self.cfg.leverage),
                cache_age_sec=None if age is None else round(float(age), 1),
                bleed_halt=bool(self.risk.bleed_halt),
            )
        # Sync bleed circuit with hunt: clear when promote flipped the flag off.
        if not self.live:
            with contextlib.suppress(Exception):
                from moonshot.hunt.constants import (
                    PAPER_BLEED_HALT_ENTRIES,
                    _load_autopilot_state,
                )
                from moonshot.hunt.scoring import _paper_bleed

                if PAPER_BLEED_HALT_ENTRIES:
                    st = _load_autopilot_state()
                    info = _paper_bleed()
                    if self.risk.bleed_halt and not st.get("paper_bleed_halt"):
                        # Hunt deployed (or flag cleared) — allow new entries.
                        self.risk.set_bleed_halt(False)
                    elif not self.risk.bleed_halt and (
                        info.get("bleed") or st.get("paper_bleed_halt")
                    ):
                        why = (
                            f"paper bleed halt — last {info.get('n') or '?'} closes red "
                            f"(pnl {float(info.get('pnl') or 0):+.2f}) · waiting for hunt deploy"
                        )
                        self.risk.set_bleed_halt(True, why)
                        if not st.get("paper_bleed_halt"):
                            from moonshot.hunt.constants import (
                                _save_autopilot_state,
                                _utc_now,
                            )

                            st["paper_bleed_halt"] = True
                            st["paper_bleed_halt_at"] = _utc_now().isoformat()
                            st["paper_bleed_pnl"] = float(info.get("pnl") or 0)
                            st["pending_champion_rescue"] = True
                            _save_autopilot_state(st)

    def _refresh_book_features(self) -> None:
        """Rebuild hunt-parity features (btc_adx / breadth) across the paper book."""
        if len(self._raw) < max(1, min(3, len(self.symbols))):
            return
        try:
            # Mirror prepare_data feature stamping (SymbolData has bars, not a feats frame)
            feats = {s: build_features(df, self.cfg) for s, df in self._raw.items()}
            btc = _btc_symbol_in(feats.keys())
            cfg = self.cfg
            if cfg.require_btc_regime and btc is not None:
                if "h4_bias" in feats[btc].columns:
                    btc_trend = feats[btc]["h4_bias"].astype(float).fillna(0.0)
                else:
                    btc_trend = feats[btc]["regime"].map(_regime_to_trend).astype(float)
                for s, f in feats.items():
                    if _is_btc_symbol(s):
                        f["btc_trend"] = btc_trend
                    else:
                        f["btc_trend"] = btc_trend.reindex(f.index).ffill().fillna(0.0)
            if btc is not None:
                btc_adx = feats[btc]["adx"].astype(float)
            else:
                btc_adx = pd.concat([f["adx"] for f in feats.values()], axis=1).mean(axis=1)
            breadth = pd.concat(
                [f["h4_bias"].astype(float) for f in feats.values()],
                axis=1,
            ).mean(axis=1)
            for f in feats.values():
                f["btc_adx"] = btc_adx.reindex(f.index).ffill().fillna(0.0)
                f["breadth"] = breadth.reindex(f.index).ffill().fillna(0.0)
            self._book_feats = feats
            if btc and btc in feats:
                self._btc_trend = self._btc_bias_from_feats(feats[btc])
        except Exception as e:
            log.warning("prepare_data book refresh failed: %s", e)

    def _bar_open_price(self, sym: str, fallback: float) -> float:
        """Fill at the open of the newest bar (backtest parity), else ticker."""
        df = self._raw.get(sym)
        if df is not None and not df.empty and "open" in df.columns:
            try:
                return float(df["open"].iloc[-1])
            except Exception:
                pass
        return float(fallback)

    def _bar_aligned_ts(
        self, sym: str, target: pd.Timestamp, fallback: Optional[pd.Timestamp] = None
    ) -> pd.Timestamp:
        """Snap a timestamp to an OHLCV bar open so charts/DB match candles."""
        ts = pd.Timestamp(target)
        ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        df = self._raw.get(sym)
        if df is not None and not df.empty:
            idx = df.index
            # Exact bar open, else nearest at-or-before target
            if ts in idx:
                return pd.Timestamp(ts)
            pos = int(idx.searchsorted(ts, side="right")) - 1
            if pos >= 0:
                return pd.Timestamp(idx[pos])
            return pd.Timestamp(idx[0])
        floored = ts.floor(self.cfg.pandas_tf)
        return floored if fallback is None else pd.Timestamp(fallback)

    # ---- per-symbol cycle ----
    def _poll(self, sym: str) -> None:
        price = self.market.last_price(sym)
        self.last_price[sym] = price
        self.broker.on_price(sym, price)

        now = self._now()
        self._accrue_funding(now)
        expected = now.floor(self.cfg.pandas_tf) - self.cfg.bar_td  # open of latest closed bar

        # Refresh OHLCV before fills/exits so entry_time snaps to real candle opens
        # (wall-clock fills were stacking every marker on the chart's last bar).
        need_bars = (
            self.last_bar.get(sym, pd.Timestamp(0, tz="UTC")) < expected
            or sym not in self._raw
            or self._raw.get(sym) is None
            or getattr(self._raw.get(sym), "empty", True)
        )
        df = self._raw.get(sym)
        if need_bars:
            df = self.market.recent(sym)
            if df.empty or df.index[-1] < expected:
                try:
                    df = fetch_historical(
                        sym,
                        self.cfg.timeframe,
                        int(self.cfg.live_history_bars),
                        refresh=True,
                    )
                except Exception as e:
                    log.warning("%s forced OHLCV refresh failed: %s", sym, e)
            if not df.empty:
                self._raw[sym] = df
                try:
                    self._cache_age_sec = (now - pd.Timestamp(df.index[-1])).total_seconds()
                except Exception:
                    self._cache_age_sec = 0.0

        # Stop fills use the last closed bar time (not wall clock)
        self._reconcile(sym)

        # Fill queued entries at the *candle* open they belong to (pend_after),
        # never wall-clock `now` — that broke entry/exit markers on price charts.
        if sym in self.pend_entry and sym not in self.positions:
            after = self.pend_after.get(sym)
            if after is not None and now >= after:
                # Expire late queues — don't fill hours-old setups at a past open
                max_age = self.cfg.bar_td * int(PEND_ENTRY_MAX_BARS)
                if now - pd.Timestamp(after) > max_age:
                    self.pend_entry.pop(sym, None)
                    self.pend_after.pop(sym, None)
                    log.info("%s dropped stale queued entry (>%s late)", sym, max_age)
                    self._save_runtime()
                else:
                    raw = self._raw.get(sym)
                    # Wait until we have the fill bar (or later) so price+time match candles
                    if (
                        raw is None
                        or raw.empty
                        or pd.Timestamp(raw.index[-1]) < pd.Timestamp(after) - self.cfg.bar_td
                    ):
                        pass  # keep queued; next poll after refresh
                    else:
                        sig = self.pend_entry.pop(sym)
                        self.pend_after.pop(sym, None)
                        fill_ts = self._bar_aligned_ts(sym, after)
                        # Prefer the open of the fill bar when present
                        fill_px = price
                        try:
                            if fill_ts in raw.index:
                                fill_px = float(raw.loc[fill_ts, "open"])
                            else:
                                fill_px = self._bar_open_price(sym, price)
                        except Exception:
                            fill_px = self._bar_open_price(sym, price)
                        self._open_from_signal(sym, sig, fill_px, fill_ts, self.broker.get_equity())
                        self._save_runtime()
        elif sym in self.pend_entry and sym in self.positions:
            self.pend_entry.pop(sym, None)
            self.pend_after.pop(sym, None)

        # Queued exits fill at next bar open (backtest parity)
        if sym in self.pend_exit and sym in self.positions:
            reason, after = self.pend_exit[sym]
            if now >= after:
                raw = self._raw.get(sym)
                if (
                    raw is not None
                    and not raw.empty
                    and pd.Timestamp(raw.index[-1]) >= pd.Timestamp(after) - self.cfg.bar_td
                ):
                    self.pend_exit.pop(sym, None)
                    fill_ts = self._bar_aligned_ts(sym, after)
                    fill_px = price
                    try:
                        if fill_ts in raw.index:
                            fill_px = float(raw.loc[fill_ts, "open"])
                    except Exception:
                        pass
                    pos = self.positions[sym]
                    self.broker.cancel_stop(sym)
                    side = "sell" if pos.sign > 0 else "buy"
                    fill = self.broker.market_order(
                        sym, side, pos.current_qty, fill_px, reduce_only=True
                    )
                    self._finish(sym, pos, fill.price, fill.fee, fill_ts, reason)

        if df is None or df.empty or df.index[-1] < expected:
            last = None if df is None or df.empty else pd.Timestamp(df.index[-1])
            age = None if last is None else (now - last).total_seconds()
            self._cache_age_sec = age
            tmon = time.monotonic()
            if tmon - float(self._stale_log_at.get(sym, 0.0)) >= 300.0:
                self._stale_log_at[sym] = tmon
                log.warning(
                    "%s OHLCV behind closed bar: last=%s expected=%s age=%ss tf=%s",
                    sym,
                    last,
                    expected,
                    None if age is None else int(age),
                    self.cfg.timeframe,
                )
            return
        if self.last_bar.get(sym, pd.Timestamp(0, tz="UTC")) >= expected:
            return
        self.last_bar[sym] = df.index[-1]
        self._bars_seen += 1
        # Prefer full-book prepare_data features (same as hunt); fall back to local build
        self._refresh_book_features()
        if sym in self._book_feats and not self._book_feats[sym].empty:
            feats = self._book_feats[sym]
        else:
            feats = build_features(df, self.cfg)
            if self.cfg.require_btc_regime:
                if _is_btc_symbol(sym):
                    self._btc_trend = self._btc_bias_from_feats(feats)
                feats["btc_trend"] = float(self._btc_trend)
                if "btc_adx" not in feats.columns:
                    feats["btc_adx"] = feats["adx"]
                if "breadth" not in feats.columns:
                    feats["breadth"] = feats.get("h4_bias", 0.0)
        bar = next(feats.tail(1).itertuples(name="Bar"))
        self._on_closed_bar(sym, bar, price, bar.Index + self.cfg.bar_td)

    @staticmethod
    def _btc_bias_from_feats(feats: pd.DataFrame) -> float:
        """Prefer completed 4H bias; fall back to 15m/daily regime."""
        if "h4_bias" in feats.columns and len(feats):
            v = float(feats["h4_bias"].iloc[-1])
            if v == v:
                return v
        return _regime_to_trend(feats["regime"].iloc[-1])

    def _refresh_btc_regime(self) -> None:
        """When BTC is not in the trade universe, still read it for the alt gate."""
        btc = "BTC/USDT:USDT"
        df = self.market.recent(btc)
        if df.empty:
            return
        feats = build_features(df, self.cfg)
        self._btc_trend = self._btc_bias_from_feats(feats)

    def _used_margin(self) -> float:
        return sum(p.margin() for p in self.positions.values())

    def _open_notional(self) -> float:
        return sum(
            p.current_qty * self.last_price.get(s, p.avg_entry) for s, p in self.positions.items()
        )

    def _net_notional(self) -> float:
        return sum(
            p.sign * p.current_qty * self.last_price.get(s, p.avg_entry)
            for s, p in self.positions.items()
        )

    def _config_hash_safe(self) -> Optional[str]:
        try:
            from moonshot.config_hash import config_hash

            return config_hash(self.cfg)
        except Exception:
            return None

    def _portfolio_obs(self, equity: float) -> dict:
        from moonshot.obs import portfolio_lev_metrics

        return portfolio_lev_metrics(
            self.positions,
            self.last_price,
            float(equity),
            mmr=float(self.cfg.maintenance_margin_rate),
        )

    def _log_risk_cycle(self, ts: pd.Timestamp, equity: float, *, snapshot: bool = True) -> None:
        """Enriched risk_snapshot + optional cycle_snapshot (fail-soft)."""
        obs = self._portfolio_obs(equity)
        pause = self.risk.pause_until.isoformat() if self.risk.pause_until is not None else None
        ch = self._config_hash_safe()
        feed_age = self._cache_age_sec
        with contextlib.suppress(Exception):
            self.db.log_risk(
                ts,
                self.risk,
                effective_lev=obs.get("effective_lev"),
                net_lev=obs.get("net_lev"),
                margin_used=obs.get("margin_used"),
                free_margin=obs.get("free_margin"),
                min_liq_distance_pct=obs.get("min_liq_distance_pct"),
                config_hash=ch,
                feed_age_s=feed_age,
            )
        if snapshot:
            with contextlib.suppress(Exception):
                self.db.log_cycle_snapshot(
                    ts,
                    equity=float(equity),
                    peak=float(self.risk.peak_equity),
                    effective_lev=float(obs.get("effective_lev") or 0.0),
                    net_lev=float(obs.get("net_lev") or 0.0),
                    open_symbols=list(self.positions.keys()),
                    pause_until=pause,
                    halt_reason=self.risk.halt_reason or "",
                    kill=bool(self.risk.kill_switch_active),
                    config_hash=ch,
                    feed_age_s=feed_age,
                    cycle_id=self._cycle_id or "",
                )

    def _maybe_expectancy_rollup(self) -> None:
        now = time.time()
        if now - self._last_expectancy_attempt < 60:
            return
        self._last_expectancy_attempt = now
        with contextlib.suppress(Exception):
            from moonshot.obs import write_expectancy_rollups

            write_expectancy_rollups(self.db)

    def _note_equity_stale_cleared(self, ts: pd.Timestamp) -> None:
        if not isinstance(self.broker, PaperBroker):
            return
        stale = bool(self.broker.equity_stale)
        if self._was_equity_stale and not stale:
            with contextlib.suppress(Exception):
                self.db.log_event(
                    ts,
                    "*",
                    "equity_stale_cleared",
                    {"cycle_id": self._cycle_id or ""},
                )
        self._was_equity_stale = stale

    def _log_fill_from(self, ts: pd.Timestamp, fill: Any, reason: str) -> None:
        with contextlib.suppress(Exception):
            self.db.log_fill(
                ts,
                fill.symbol,
                side=str(fill.side),
                reason=reason,
                intended_qty=float(
                    fill.intended_qty if fill.intended_qty is not None else fill.qty
                ),
                filled_qty=float(fill.qty),
                ref_price=getattr(fill, "ref_price", None),
                fill_price=float(fill.price),
                slippage_bps=getattr(fill, "slippage_bps", None),
                assumed_slippage_bps=getattr(fill, "assumed_slippage_bps", None),
            )

    def _accrue_funding(self, ts: pd.Timestamp) -> None:
        """Match backtest: charge constant funding at settlement hours (paper cash)."""
        cfg = self.cfg
        if not self.positions:
            return
        if ts.minute > 2:  # only near the hour
            return
        if ts.hour % max(1, int(cfg.funding_interval_hours)) != 0:
            return
        key = ts.floor("h")
        if self._last_funding_key is not None and key <= self._last_funding_key:
            return
        self._last_funding_key = key
        for sym, pos in list(self.positions.items()):
            px = float(self.last_price.get(sym, pos.avg_entry))
            cost = pos.sign * pos.current_qty * px * cfg.funding_rate
            pos.funding += cost
            if isinstance(self.broker, PaperBroker):
                self.broker.cash -= cost
            self.db.save_position(sym, pos)
            self.db.log_event(
                ts, sym, "funding", {"cost": cost, "rate": cfg.funding_rate, "price": px}
            )

    def _check_bar_stop_liq(self, sym: str, pos: Position, bar: Any, ts: pd.Timestamp) -> bool:
        """Paper OHLC stop + liquidation (ticker-only stops miss wicks)."""
        if not isinstance(self.broker, PaperBroker):
            return False
        cfg = self.cfg
        liq = pos.liq_price(cfg.maintenance_margin_rate)
        long = pos.sign > 0
        hit_stop = bar.low <= pos.current_stop if long else bar.high >= pos.current_stop
        hit_liq = bar.low <= liq if long else bar.high >= liq
        if hit_stop and ((pos.current_stop > liq) if long else (pos.current_stop < liq)):
            raw = min(pos.current_stop, bar.open) if long else max(pos.current_stop, bar.open)
            if (raw <= liq) if long else (raw >= liq):
                self.broker.cancel_stop(sym)
                # Flatten paper book at liq
                side = "sell" if long else "buy"
                fill = self.broker.market_order(sym, side, pos.current_qty, liq, reduce_only=True)
                self._log_fill_from(ts, fill, "liquidation")
                self._finish(sym, pos, liq, 0.0, ts, ExitReason.LIQUIDATION)
                return True
            reason = (
                ExitReason.TRAILING
                if pos.is_tighter(pos.initial_stop) and pos.current_stop != pos.initial_stop
                else ExitReason.STOP
            )
            px = self.broker._slip(raw, buy=not long) if hasattr(self.broker, "_slip") else raw
            self.broker.cancel_stop(sym)
            side = "sell" if long else "buy"
            fill = self.broker.market_order(sym, side, pos.current_qty, px, reduce_only=True)
            self._log_fill_from(ts, fill, "stop")
            self._finish(sym, pos, fill.price, fill.fee, ts, reason)
            return True
        if hit_liq:
            self.broker.cancel_stop(sym)
            side = "sell" if long else "buy"
            fill = self.broker.market_order(sym, side, pos.current_qty, liq, reduce_only=True)
            self._log_fill_from(ts, fill, "liquidation")
            self._finish(sym, pos, liq, 0.0, ts, ExitReason.LIQUIDATION)
            return True
        return False

    def _on_closed_bar(self, sym: str, bar: Any, price: float, ts: pd.Timestamp) -> None:
        # Decisions at bar close; entries/exits fill in _poll once the next bar opens.
        self._cycle_n += 1
        self._cycle_id = f"{self._cycle_n}-{uuid.uuid4().hex[:8]}"
        self._last_cycle_ts = pd.Timestamp(ts)
        self._note_equity_stale_cleared(ts)
        equity = self.broker.get_equity()
        self.risk.update_equity(equity, ts)
        self.db.log_equity(ts, equity, self.risk.peak_equity)
        self._log_risk_cycle(ts, equity, snapshot=True)
        log.info(
            "bar_close symbol=%s cycle_id=%s equity=%.4f halt=%s open=%d",
            sym,
            self._cycle_id,
            equity,
            self.risk.halt_reason or "-",
            len(self.positions),
            extra={
                "cycle_id": self._cycle_id,
                "symbol": sym,
                "equity": float(equity),
                "halt": str(self.risk.halt_reason or ""),
            },
        )

        pos = self.positions.get(sym)
        if pos:
            with contextlib.suppress(Exception):
                pos.update_excursion(float(price))
                pos.bars_held(ts, self.cfg.bar_td)
                self.db.save_position(sym, pos)
            # Wick-aware stops before discretionary exits
            if self._check_bar_stop_liq(sym, pos, bar, ts):
                self._save_runtime()
                return
            if sym in self.pend_exit:
                return  # already queued for next open
            acts = self.pm.decide(
                pos,
                bar,
                allowed=self.risk.trading_allowed(ts),
                kill=self.risk.kill_switch_active,
                equity=equity,
                used_margin=self._used_margin(),
                spec=self.specs[sym],
            )
            # Queue EXIT for next open (match backtest); other acts still immediate
            exit_acts = [a for a in acts if a.kind is ActionKind.EXIT]
            other = [a for a in acts if a.kind is not ActionKind.EXIT]
            if exit_acts:
                self.pend_exit[sym] = (exit_acts[0].reason, pd.Timestamp(ts))
                self._save_runtime()
            if other:
                self._execute(sym, pos, other, price, ts, equity)
        elif sym not in self.blocked:
            self._queue_entry(sym, bar, ts)
        self._save_runtime()

    def _queue_entry(self, sym: str, bar: Any, ts: pd.Timestamp) -> None:
        """Queue a fill for the next bar open (ts already equals bar.Index + bar_td)."""
        cfg = self.cfg
        if ts < self.cooldown.get(sym, ts) or not self.risk.trading_allowed(ts):
            return
        # After restart, marks may be stale — no new risk until first on_price.
        if isinstance(self.broker, PaperBroker) and self.broker.equity_stale:
            return
        if len(self.positions) + len(self.pend_entry) >= cfg.max_open_positions:
            return
        # Refuse setups on a stale book — fills against old opens then die as regime_invalid
        now = self._now()
        expected = now.floor(cfg.pandas_tf) - cfg.bar_td
        raw = self._raw.get(sym)
        if raw is None or raw.empty or pd.Timestamp(raw.index[-1]) < expected - cfg.bar_td:
            return
        sig = self.strategy.on_bar(sym, bar)
        if not sig:
            return
        self._signals_seen += 1
        self.pend_entry[sym] = sig
        self.pend_after[sym] = pd.Timestamp(ts)  # earliest fill = open of the new bar
        log.info(
            "%s SETUP %s — queued for next open (%s) cycle_id=%s",
            sym,
            sig.direction.value,
            sig.reason,
            self._cycle_id or "-",
            extra={"cycle_id": self._cycle_id or "", "symbol": sym},
        )
        with contextlib.suppress(Exception):
            _live_feed(
                "setup",
                f"{sym.split('/')[0]} {sig.direction.value} setup — queued for next candle open",
                symbol=sym,
                direction=sig.direction.value,
                reason=sig.reason,
            )

    def _open_from_signal(
        self, sym: str, sig: Signal, price: float, ts: pd.Timestamp, equity: float
    ) -> None:
        cfg = self.cfg
        if sym in self.positions or not self.risk.trading_allowed(ts):
            return
        if isinstance(self.broker, PaperBroker) and self.broker.equity_stale:
            return
        stop = price - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        dec = self.risk.evaluate(
            sig.direction,
            price,
            stop,
            equity=equity,
            open_notional=self._open_notional(),
            used_margin=self._used_margin(),
            open_positions=len(self.positions),
            spec=self.specs[sym],
            ts=ts,
            net_notional=self._net_notional(),
        )
        if not dec.approved:
            log.info("%s queued signal rejected at open: %s", sym, dec.reason)
            with contextlib.suppress(Exception):
                self.db.log_rejected_entry(
                    ts, sym, sig.direction.value, float(price), float(stop), str(dec.reason)
                )
            with contextlib.suppress(Exception):
                _live_feed(
                    "reject",
                    f"{sym.split('/')[0]} blocked at open: {dec.reason}",
                    symbol=sym,
                    reason=dec.reason,
                )
            return
        side = "buy" if sig.direction is Direction.LONG else "sell"
        self.broker.set_leverage(sym, dec.leverage)
        fill = self.broker.market_order(sym, side, dec.qty, price)
        stop = fill.price - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        pos = Position.open(
            sym, sig.direction, dec.leverage, fill.price, fill.qty, stop, ts, fill.fee
        )
        pos.intended_qty = float(
            fill.intended_qty if getattr(fill, "intended_qty", None) is not None else fill.qty
        )
        pos.entry_slippage_bps = getattr(fill, "slippage_bps", None)
        pos.assumed_slippage_bps = getattr(fill, "assumed_slippage_bps", None)
        pos.funding_rate_entry = float(cfg.funding_rate)
        pos.open_config_hash = self._config_hash_safe()
        try:
            self.broker.set_stop(sym, pos.direction, pos.current_qty, pos.current_stop)
        except ExecutionError:
            self.broker.market_order(
                sym, "sell" if side == "buy" else "buy", fill.qty, price, reduce_only=True
            )
            self.blocked.add(sym)
            raise
        self.positions[sym] = pos
        self.db.save_position(sym, pos)
        self._log_fill_from(ts, fill, "open")
        self.db.log_event(
            ts,
            sym,
            "open",
            {
                "direction": sig.direction.value,
                "qty": fill.qty,
                "intended_qty": pos.intended_qty,
                "price": fill.price,
                "ref_price": getattr(fill, "ref_price", price),
                "slippage_bps": pos.entry_slippage_bps,
                "assumed_slippage_bps": pos.assumed_slippage_bps,
                "stop": stop,
                "reason": sig.reason,
                "fill_timing": "next_open",
                "config_hash": pos.open_config_hash,
                "funding_rate_entry": pos.funding_rate_entry,
            },
        )
        with contextlib.suppress(Exception):
            _live_feed(
                "open",
                f"OPEN {sym.split('/')[0]} {sig.direction.value} @ {fill.price:.6g}",
                symbol=sym,
                direction=sig.direction.value,
                price=fill.price,
                qty=fill.qty,
            )
        self.notify.send(
            f"OPEN {sym} {sig.direction.value} qty={fill.qty:.6g} @ {fill.price:.6g} "
            f"stop={stop:.6g} | {sig.reason}"
        )

    def _execute(
        self, sym: str, pos: Position, acts: list, price: float, ts: pd.Timestamp, equity: float
    ) -> None:
        close_side = "sell" if pos.sign > 0 else "buy"
        open_side = "buy" if pos.sign > 0 else "sell"
        for a in acts:
            if a.kind is ActionKind.EXIT:
                self.broker.cancel_stop(sym)
                fill = self.broker.market_order(
                    sym, close_side, pos.current_qty, price, reduce_only=True
                )
                self._log_fill_from(ts, fill, "exit")
                self._finish(sym, pos, fill.price, fill.fee, ts, a.reason)
                return
        for a in acts:
            if a.kind is ActionKind.PARTIAL:
                fill = self.broker.market_order(sym, close_side, a.qty, price, reduce_only=True)
                pos.reduce(fill.price, fill.qty, fill.fee)
                pos.partial_done = True
                self._log_fill_from(ts, fill, "partial")
                self.db.log_event(
                    ts,
                    sym,
                    "partial",
                    {
                        "qty": fill.qty,
                        "intended_qty": getattr(fill, "intended_qty", fill.qty),
                        "price": fill.price,
                        "remaining": pos.current_qty,
                        "slippage_bps": getattr(fill, "slippage_bps", None),
                    },
                )
            elif a.kind is ActionKind.ADD:
                qty, dec = self.pm.roll.size_add(
                    pos, price, a.price, equity, self._used_margin(), self.specs[sym]
                )
                if dec is AddDecision.ALLOW and qty:
                    fill = self.broker.market_order(sym, open_side, qty, price)
                    pos.add_lot(
                        fill.price, fill.qty, ts, fill.fee, a.price, pos.unrealized(fill.price)
                    )
                    self._log_fill_from(ts, fill, "add")
                    self.db.log_event(
                        ts,
                        sym,
                        "add",
                        {
                            "qty": fill.qty,
                            "intended_qty": getattr(fill, "intended_qty", fill.qty),
                            "price": fill.price,
                            "avg": pos.avg_entry,
                            "stop": pos.current_stop,
                            "r_at_add": pos.adds[-1].r_at_add if pos.adds else None,
                            "slippage_bps": getattr(fill, "slippage_bps", None),
                        },
                    )
                    self.notify.send(
                        f"ADD {sym} +{fill.qty:.6g} @ {fill.price:.6g} avg={pos.avg_entry:.6g}"
                    )
            elif a.kind is ActionKind.MOVE_STOP:
                old = pos.current_stop
                if pos.tighten_stop(a.price):
                    self.db.log_event(ts, sym, "move_stop", {"from": old, "to": pos.current_stop})
        # keep the exchange-side stop in sync with final qty and stop
        self.broker.set_stop(sym, pos.direction, pos.current_qty, pos.current_stop)
        self.db.save_position(sym, pos)

    def _push_local_to_broker(self, sym: str, pos: Position) -> None:
        """Re-register a DB-restored position on the paper broker (never invent a stop-out)."""
        if not isinstance(self.broker, PaperBroker):
            return
        self.broker.signed[sym] = pos.sign * float(pos.current_qty)
        self.broker.avg[sym] = float(pos.avg_entry)
        self.broker.stops[sym] = float(pos.current_stop)
        self.broker.last[sym] = self.last_price.get(sym, float(pos.avg_entry))
        log.warning("%s: local position had no broker mirror — re-pushed (no fabricated exit)", sym)

    def _reconcile(self, sym: str) -> None:
        pos, bp = self.positions.get(sym), self.broker.get_position(sym)
        if pos and bp is None:
            fill = self.broker.pop_stop_fill(sym)
            # Only accept a real stop fill, or a disappearance after we had seen the
            # broker position this session. Cold-start desync must NOT invent -1R exits.
            if fill is None and sym not in self._seen_broker_pos:
                self._push_local_to_broker(sym, pos)
                self._seen_broker_pos.add(sym)
                return
            px = fill.price if fill else pos.current_stop
            fee = fill.fee if fill else px * pos.current_qty * self.cfg.fee_taker
            reason = (
                ExitReason.TRAILING if pos.current_stop != pos.initial_stop else ExitReason.STOP
            )
            # Stamp exit on the last closed candle — wall clock piled markers on chart edge
            exit_ts = self._bar_aligned_ts(
                sym,
                self.last_bar.get(sym) or self._now(),
            )
            self._finish(sym, pos, px, fee, exit_ts, reason)
        elif bp and not pos and sym not in self.blocked:
            self.blocked.add(sym)
            self.notify.send(f"UNKNOWN position on {sym}; bot will not trade it. Resolve manually.")
        elif bp and pos:
            self._seen_broker_pos.add(sym)
            if abs(bp.qty - pos.current_qty) > 0.01 * pos.current_qty:
                log.warning(
                    "%s qty mismatch local=%.6g broker=%.6g; syncing to broker",
                    sym,
                    pos.current_qty,
                    bp.qty,
                )
                pos.current_qty = bp.qty

    def _finish(
        self, sym: str, pos: Position, px: float, fee: float, ts: pd.Timestamp, reason: ExitReason
    ) -> None:
        # Never stamp exit before entry (chart / hold-time integrity)
        try:
            et = pd.Timestamp(pos.entry_time)
            ts = pd.Timestamp(ts)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            if et.tzinfo is None:
                et = et.tz_localize("UTC")
            if ts < et:
                ts = et + self.cfg.bar_td
        except Exception:
            pass
        with contextlib.suppress(Exception):
            pos.update_excursion(float(px))
            pos.bars_held(ts, self.cfg.bar_td)
        res, _ = pos.close_out(px, fee, ts, reason)
        try:
            from moonshot.config_hash import canonical_json, config_hash

            res.config_hash = config_hash(self.cfg)
            res.config_json = canonical_json(self.cfg)
        except Exception:
            pass
        if not res.open_config_hash:
            res.open_config_hash = getattr(pos, "open_config_hash", None)
        self.positions.pop(sym, None)
        self.pend_exit.pop(sym, None)
        self.db.save_position(sym, None)
        self.db.log_trade(res)
        self.db.log_event(
            ts,
            sym,
            "exit",
            {
                "reason": reason.value,
                "pnl": res.pnl,
                "r": res.r_multiple,
                "price": px,
                "mae_r": res.mae_r,
                "mfe_r": res.mfe_r,
                "bars_held": res.bars_held,
                "open_config_hash": res.open_config_hash,
                "config_hash": res.config_hash,
            },
        )
        self.risk.register_trade_result(res.r_multiple, ts)
        # Hard circuit: last N paper closes all red → no new entries until hunt deploys.
        if not self.live:
            with contextlib.suppress(Exception):
                from moonshot.hunt.constants import (
                    PAPER_BLEED_HALT_ENTRIES,
                    _load_autopilot_state,
                    _save_autopilot_state,
                    _utc_now,
                )
                from moonshot.hunt.scoring import _paper_bleed

                if PAPER_BLEED_HALT_ENTRIES:
                    info = _paper_bleed()
                    if info.get("bleed"):
                        why = (
                            f"paper bleed halt — last {info['n']} closes red "
                            f"(pnl {float(info['pnl']):+.2f}) · waiting for hunt deploy"
                        )
                        self.risk.set_bleed_halt(True, why)
                        st = _load_autopilot_state()
                        st["paper_bleed_halt"] = True
                        st["paper_bleed_halt_at"] = _utc_now().isoformat()
                        st["paper_bleed_pnl"] = float(info.get("pnl") or 0)
                        # Nudge hunt to escape this book
                        st["pending_champion_rescue"] = True
                        _save_autopilot_state(st)
                        with contextlib.suppress(Exception):
                            _live_feed(
                                "bleed_halt",
                                why,
                                n=int(info.get("n") or 0),
                                pnl=float(info.get("pnl") or 0),
                            )
        # Paper drift vs promoted backtest band — halt new entries if far outside.
        if not self.live and self.risk.rolling_r:
            n = len(self.risk.rolling_r)
            paper_exp = float(np.mean(self.risk.rolling_r))
            start_eq = float(self.cfg.starting_equity) or 1.0
            paper_ret = (float(self.risk.equity) / start_eq) - 1.0
            if self.risk.check_paper_drift(
                n_trades=n,
                paper_exp=paper_exp,
                paper_ret=paper_ret,
            ):
                with contextlib.suppress(Exception):
                    _live_feed(
                        "drift_halt",
                        f"PAPER DRIFT HALT · {self.risk.halt_reason}",
                        exp=paper_exp,
                        ret=paper_ret,
                        n=n,
                    )
                # Policy-3: ask hunt to deploy champion; ban the drifted knobs.
                with contextlib.suppress(Exception):
                    from moonshot.hunt.constants import (
                        _cfg_to_knobs,
                        _load_autopilot_state,
                        _save_autopilot_state,
                        _utc_now,
                    )
                    from moonshot.hunt.scoring import _banlist_add, _cfg_fingerprint

                    st = _load_autopilot_state()
                    st["pending_champion_rescue"] = True
                    st["last_drift"] = {
                        "ts": _utc_now().isoformat(),
                        "reason": str(self.risk.halt_reason or "drift"),
                        "knobs_hash": repr(_cfg_fingerprint(self.cfg)),
                        "paper_exp": float(paper_exp),
                        "paper_ret": float(paper_ret),
                        "n_trades": int(n),
                    }
                    _banlist_add(
                        st,
                        self.cfg,
                        week_ret=float(paper_ret),
                        reason="paper_drift",
                    )
                    # Keep knobs snapshot for debugging
                    st["last_drift"]["knobs"] = _cfg_to_knobs(self.cfg)
                    _save_autopilot_state(st)
                    log.warning(
                        "drift → pending_champion_rescue + banlist (%s)",
                        self.risk.halt_reason,
                    )
        # Longer cool-off after regime kills — stops open→kill→requeue loops
        cool_bars = int(self.cfg.cooldown_bars)
        if reason is ExitReason.REGIME_INVALID:
            cool_bars = max(cool_bars, 6)
        self.cooldown[sym] = ts + cool_bars * self.cfg.bar_td
        self._save_runtime()
        with contextlib.suppress(Exception):
            _live_feed(
                "close",
                f"CLOSE {sym.split('/')[0]} {reason.value} pnl={res.pnl:+.2f}",
                symbol=sym,
                reason=reason.value,
                pnl=res.pnl,
                r=res.r_multiple,
            )
        self.notify.send(f"CLOSE {sym} {reason.value} pnl={res.pnl:+.2f} ({res.r_multiple:+.2f}R)")

    def replay_offline(
        self,
        frames: dict[str, pd.DataFrame],
        *,
        start_i: int = 0,
        end_i: Optional[int] = None,
    ) -> None:
        """Drive paper decisions from frozen OHLCV (tests / smoke). Uses `_clock`."""
        if self.live:
            raise BotError("replay_offline is paper-only")
        if not frames:
            return
        # Align on shared timestamps (fixture books share an index)
        idx = None
        for df in frames.values():
            idx = df.index if idx is None else idx.intersection(df.index)
        if idx is None or len(idx) == 0:
            return
        idx = pd.DatetimeIndex(idx).sort_values()
        end = len(idx) if end_i is None else min(int(end_i), len(idx))
        start = max(0, int(start_i))
        for i in range(start, end):
            bar_ts = pd.Timestamp(idx[i])
            # Decision clock = open of the next bar (same as live closed-bar ts)
            self._clock = bar_ts + self.cfg.bar_td
            for sym in self.symbols:
                full = frames.get(sym)
                if full is None or full.empty:
                    continue
                df = full.loc[:bar_ts]
                if df.empty:
                    continue
                self._raw[sym] = df
                price = float(df.iloc[-1]["close"])
                self.last_price[sym] = price
                self.broker.on_price(sym, price)
                # Pick up exchange-side stop fills before discretionary bar logic
                self._reconcile(sym)
                # Pending fills at this open (mirror _poll fill branch)
                if sym in self.pend_entry and sym not in self.positions:
                    after = self.pend_after.get(sym)
                    if after is not None and self._clock >= after:
                        sig = self.pend_entry.pop(sym)
                        self.pend_after.pop(sym, None)
                        fill_ts = pd.Timestamp(after)
                        fill_px = price
                        try:
                            if fill_ts in df.index:
                                fill_px = float(df.loc[fill_ts, "open"])
                            elif bar_ts in df.index:
                                fill_px = float(df.loc[bar_ts, "open"])
                        except Exception:
                            pass
                        self._open_from_signal(sym, sig, fill_px, fill_ts, self.broker.get_equity())
                if sym in self.pend_exit and sym in self.positions:
                    reason, after = self.pend_exit[sym]
                    if self._clock >= after:
                        self.pend_exit.pop(sym, None)
                        fill_ts = pd.Timestamp(after)
                        fill_px = price
                        try:
                            if fill_ts in df.index:
                                fill_px = float(df.loc[fill_ts, "open"])
                        except Exception:
                            pass
                        pos = self.positions[sym]
                        self.broker.cancel_stop(sym)
                        side = "sell" if pos.sign > 0 else "buy"
                        fill = self.broker.market_order(
                            sym, side, pos.current_qty, fill_px, reduce_only=True
                        )
                        self._finish(sym, pos, fill.price, fill.fee, fill_ts, reason)
                if self.last_bar.get(sym, pd.Timestamp(0, tz="UTC")) >= bar_ts:
                    continue
                self.last_bar[sym] = bar_ts
                self._bars_seen += 1
                self._refresh_book_features()
                if sym in self._book_feats and not self._book_feats[sym].empty:
                    feats = self._book_feats[sym]
                else:
                    feats = build_features(df, self.cfg)
                if feats.empty:
                    continue
                bar = next(feats.tail(1).itertuples(name="Bar"))
                self._on_closed_bar(sym, bar, price, bar_ts + self.cfg.bar_td)
            self._save_runtime()
