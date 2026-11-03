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
import moonshot.brokers as _moonshot_brokers
pull(_moonshot_brokers, globals())
import moonshot.persist as _moonshot_persist
pull(_moonshot_persist, globals())
import moonshot.backtest as _moonshot_backtest
pull(_moonshot_backtest, globals())
from moonshot.paths import ROOT, DB_PATH, CACHE_DIR, PAPER_RUNTIME_PATH

class LiveRunner:
    def __init__(self, cfg: Config, broker: BaseBroker, market: MarketData, symbols: list,
                 live: bool):
        self.cfg, self.broker, self.market, self.symbols, self.live = cfg, broker, market, symbols, live
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
        self._btc_trend = 0.0
        self._raw: dict = {}  # symbol -> OHLCV for prepare_data parity with hunt
        self._book_feats: dict = {}  # symbol -> feature frame from prepare_data
        self._stale_log_at: dict = {}  # symbol -> last stale warning monotonic time
        self._cache_age_sec: Optional[float] = None
        # Paper must rebuild cash from closed trades even with 0 open positions.
        # Old gate `and self.positions` left wallet stuck at $50 after flat restarts
        # while Live still counted closed trades → "19 closed, balance $50".
        if not live and isinstance(broker, PaperBroker):
            closed = self.db.closed_pnl_sum(mode="paper")
            if self.positions or abs(closed) > 1e-12:
                broker.rehydrate(self.positions or {}, closed)
                log.info(
                    "paper rehydrated %d open · closed_pnl=%+.2f · cash=%.2f",
                    len(self.positions), closed, broker.cash,
                )
        # Poll BTC first so alt entries see a fresh regime stamp
        if cfg.require_btc_regime:
            btc = _btc_symbol_in(symbols)
            if btc and symbols[0] != btc:
                self.symbols = [btc] + [s for s in symbols if s != btc]

    def _runtime_path(self) -> Path:
        return Path(self.cfg.db_path).resolve().parent / PAPER_RUNTIME_PATH.name

    def _save_runtime(self) -> None:
        """Persist risk / cooldown / pending queues across paper restarts."""
        if self.live:
            return
        try:
            cool = {
                s: pd.Timestamp(t).isoformat()
                for s, t in self.cooldown.items()
                if t is not None
            }
            pend_after = {
                s: pd.Timestamp(t).isoformat()
                for s, t in self.pend_after.items()
                if t is not None
            }
            pend_exit = {
                s: {
                    "reason": (r.value if hasattr(r, "value") else str(r)),
                    "after": pd.Timestamp(a).isoformat(),
                }
                for s, (r, a) in self.pend_exit.items()
            }
            payload = {
                "ts": _utc_now().isoformat(),
                "peak_equity": float(self.risk.peak_equity),
                "equity": float(self.risk.equity),
                "kill": bool(self.risk.kill_switch_active),
                "halt_reason": str(self.risk.halt_reason or ""),
                "daily_halt": bool(self.risk.daily_halt),
                "streak": int(self.risk.consecutive_losses),
                "pause_until": (
                    None if self.risk.pause_until is None
                    else pd.Timestamp(self.risk.pause_until).isoformat()
                ),
                "rolling_r": [float(x) for x in list(self.risk.rolling_r)[-50:]],
                "cooldown": cool,
                "pend_after": pend_after,
                "pend_exit": pend_exit,
                "blocked": sorted(self.blocked),
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
            rr = data.get("rolling_r") or []
            if isinstance(rr, list) and rr:
                self.risk.rolling_r = [float(x) for x in rr]
            for s, t in (data.get("cooldown") or {}).items():
                self.cooldown[str(s)] = pd.Timestamp(t)
            for s, t in (data.get("pend_after") or {}).items():
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
                "paper runtime restored (peak=%.2f kill=%s cooldown=%d)",
                self.risk.peak_equity, self.risk.kill_switch_active, len(self.cooldown),
            )
        except Exception as e:
            log.warning("paper runtime apply failed: %s", e)

    def run_forever(self) -> None:
        mode = "LIVE" if self.live else "PAPER"
        # Unblock paper if live-sized CSVs froze behind wall clock
        try:
            clear_stale_live_caches(
                tf=str(self.cfg.timeframe), bars=int(self.cfg.live_history_bars),
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
        ts = pd.Timestamp.now(tz="UTC")
        equity = self.broker.get_equity()
        self.db.log_equity(ts, equity, self.risk.peak_equity)
        self.db.log_risk(ts, self.risk)
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
        log.info(msg)
        try:
            _live_feed(
                "paper_tick", msg,
                open=len(self.positions), next_mins=mins,
                timeframe=self.cfg.timeframe, equity=round(equity, 2),
                bars_seen=int(self._bars_seen), setups_seen=int(self._signals_seen),
                adx_enter=float(self.cfg.adx_enter),
                vol_mult=float(self.cfg.volume_multiplier),
                require_rising_adx=bool(self.cfg.require_rising_adx),
                leverage=int(self.cfg.leverage),
                cache_age_sec=None if age is None else round(float(age), 1),
            )
        except Exception:
            pass

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
                [f["h4_bias"].astype(float) for f in feats.values()], axis=1,
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

    def _bar_aligned_ts(self, sym: str, target: pd.Timestamp,
                        fallback: Optional[pd.Timestamp] = None) -> pd.Timestamp:
        """Snap a timestamp to an OHLCV bar open so charts/DB match candles."""
        ts = pd.Timestamp(target)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
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

        now = pd.Timestamp.now(tz="UTC")
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
                        sym, self.cfg.timeframe, int(self.cfg.live_history_bars),
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
                    if raw is None or raw.empty or pd.Timestamp(raw.index[-1]) < pd.Timestamp(after) - self.cfg.bar_td:
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
                if raw is not None and not raw.empty and pd.Timestamp(raw.index[-1]) >= pd.Timestamp(after) - self.cfg.bar_td:
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
                    fill = self.broker.market_order(sym, side, pos.current_qty, fill_px, reduce_only=True)
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
                    sym, last, expected, None if age is None else int(age), self.cfg.timeframe,
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
        return sum(p.current_qty * self.last_price.get(s, p.avg_entry) for s, p in self.positions.items())

    def _net_notional(self) -> float:
        return sum(
            p.sign * p.current_qty * self.last_price.get(s, p.avg_entry)
            for s, p in self.positions.items()
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
            self.db.log_event(ts, sym, "funding", {"cost": cost, "rate": cfg.funding_rate, "price": px})

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
            self._finish(sym, pos, fill.price, fill.fee, ts, reason)
            return True
        if hit_liq:
            self.broker.cancel_stop(sym)
            side = "sell" if long else "buy"
            self.broker.market_order(sym, side, pos.current_qty, liq, reduce_only=True)
            self._finish(sym, pos, liq, 0.0, ts, ExitReason.LIQUIDATION)
            return True
        return False

    def _on_closed_bar(self, sym: str, bar: Any, price: float, ts: pd.Timestamp) -> None:
        # Decisions at bar close; entries/exits fill in _poll once the next bar opens.
        equity = self.broker.get_equity()
        self.risk.update_equity(equity, ts)
        self.db.log_equity(ts, equity, self.risk.peak_equity)
        self.db.log_risk(ts, self.risk)

        pos = self.positions.get(sym)
        if pos:
            # Wick-aware stops before discretionary exits
            if self._check_bar_stop_liq(sym, pos, bar, ts):
                self._save_runtime()
                return
            if sym in self.pend_exit:
                return  # already queued for next open
            acts = self.pm.decide(pos, bar, allowed=self.risk.trading_allowed(ts),
                                  kill=self.risk.kill_switch_active, equity=equity,
                                  used_margin=self._used_margin(), spec=self.specs[sym])
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
        if len(self.positions) + len(self.pend_entry) >= cfg.max_open_positions:
            return
        # Refuse setups on a stale book — fills against old opens then die as regime_invalid
        now = pd.Timestamp.now(tz="UTC")
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
        log.info("%s SETUP %s — queued for next open (%s)", sym, sig.direction.value, sig.reason)
        try:
            _live_feed(
                "setup",
                f"{sym.split('/')[0]} {sig.direction.value} setup — queued for next candle open",
                symbol=sym, direction=sig.direction.value, reason=sig.reason,
            )
        except Exception:
            pass

    def _open_from_signal(self, sym: str, sig: Signal, price: float, ts: pd.Timestamp,
                          equity: float) -> None:
        cfg = self.cfg
        if sym in self.positions or not self.risk.trading_allowed(ts):
            return
        stop = price - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        dec = self.risk.evaluate(
            sig.direction, price, stop, equity=equity,
            open_notional=self._open_notional(), used_margin=self._used_margin(),
            open_positions=len(self.positions), spec=self.specs[sym], ts=ts,
            net_notional=self._net_notional(),
        )
        if not dec.approved:
            log.info("%s queued signal rejected at open: %s", sym, dec.reason)
            try:
                _live_feed("reject", f"{sym.split('/')[0]} blocked at open: {dec.reason}",
                           symbol=sym, reason=dec.reason)
            except Exception:
                pass
            return
        side = "buy" if sig.direction is Direction.LONG else "sell"
        self.broker.set_leverage(sym, dec.leverage)
        fill = self.broker.market_order(sym, side, dec.qty, price)
        stop = fill.price - sig.direction.sign * cfg.initial_stop_atr * sig.atr
        pos = Position.open(sym, sig.direction, dec.leverage, fill.price, fill.qty, stop, ts, fill.fee)
        try:
            self.broker.set_stop(sym, pos.direction, pos.current_qty, pos.current_stop)
        except ExecutionError:
            self.broker.market_order(sym, "sell" if side == "buy" else "buy", fill.qty, price, reduce_only=True)
            self.blocked.add(sym)
            raise
        self.positions[sym] = pos
        self.db.save_position(sym, pos)
        self.db.log_event(ts, sym, "open", {
            "direction": sig.direction.value, "qty": fill.qty, "price": fill.price,
            "stop": stop, "reason": sig.reason, "fill_timing": "next_open",
        })
        try:
            _live_feed(
                "open",
                f"OPEN {sym.split('/')[0]} {sig.direction.value} @ {fill.price:.6g}",
                symbol=sym, direction=sig.direction.value, price=fill.price, qty=fill.qty,
            )
        except Exception:
            pass
        self.notify.send(f"OPEN {sym} {sig.direction.value} qty={fill.qty:.6g} @ {fill.price:.6g} "
                         f"stop={stop:.6g} | {sig.reason}")

    def _execute(self, sym: str, pos: Position, acts: list, price: float, ts: pd.Timestamp,
                 equity: float) -> None:
        close_side = "sell" if pos.sign > 0 else "buy"
        open_side = "buy" if pos.sign > 0 else "sell"
        for a in acts:
            if a.kind is ActionKind.EXIT:
                self.broker.cancel_stop(sym)
                fill = self.broker.market_order(sym, close_side, pos.current_qty, price, reduce_only=True)
                self._finish(sym, pos, fill.price, fill.fee, ts, a.reason)
                return
        for a in acts:
            if a.kind is ActionKind.PARTIAL:
                fill = self.broker.market_order(sym, close_side, a.qty, price, reduce_only=True)
                pos.reduce(fill.price, fill.qty, fill.fee)
                pos.partial_done = True
                self.db.log_event(ts, sym, "partial", {
                    "qty": fill.qty, "price": fill.price, "remaining": pos.current_qty,
                })
            elif a.kind is ActionKind.ADD:
                qty, dec = self.pm.roll.size_add(pos, price, a.price, equity, self._used_margin(),
                                                 self.specs[sym])
                if dec is AddDecision.ALLOW and qty:
                    fill = self.broker.market_order(sym, open_side, qty, price)
                    pos.add_lot(fill.price, fill.qty, ts, fill.fee, a.price, pos.unrealized(fill.price))
                    self.db.log_event(ts, sym, "add", {
                        "qty": fill.qty, "price": fill.price, "avg": pos.avg_entry,
                        "stop": pos.current_stop,
                    })
                    self.notify.send(f"ADD {sym} +{fill.qty:.6g} @ {fill.price:.6g} avg={pos.avg_entry:.6g}")
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
            reason = ExitReason.TRAILING if pos.current_stop != pos.initial_stop else ExitReason.STOP
            # Stamp exit on the last closed candle — wall clock piled markers on chart edge
            exit_ts = self._bar_aligned_ts(
                sym,
                self.last_bar.get(sym) or pd.Timestamp.now(tz="UTC"),
            )
            self._finish(sym, pos, px, fee, exit_ts, reason)
        elif bp and not pos and sym not in self.blocked:
            self.blocked.add(sym)
            self.notify.send(f"UNKNOWN position on {sym}; bot will not trade it. Resolve manually.")
        elif bp and pos:
            self._seen_broker_pos.add(sym)
            if abs(bp.qty - pos.current_qty) > 0.01 * pos.current_qty:
                log.warning("%s qty mismatch local=%.6g broker=%.6g; syncing to broker", sym,
                            pos.current_qty, bp.qty)
                pos.current_qty = bp.qty

    def _finish(self, sym: str, pos: Position, px: float, fee: float, ts: pd.Timestamp,
                reason: ExitReason) -> None:
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
        res, _ = pos.close_out(px, fee, ts, reason)
        self.positions.pop(sym, None)
        self.pend_exit.pop(sym, None)
        self.db.save_position(sym, None)
        self.db.log_trade(res)
        self.db.log_event(ts, sym, "exit", {
            "reason": reason.value, "pnl": res.pnl, "r": res.r_multiple, "price": px,
        })
        self.risk.register_trade_result(res.r_multiple, ts)
        # Longer cool-off after regime kills — stops open→kill→requeue loops
        cool_bars = int(self.cfg.cooldown_bars)
        if reason is ExitReason.REGIME_INVALID:
            cool_bars = max(cool_bars, 6)
        self.cooldown[sym] = ts + cool_bars * self.cfg.bar_td
        self._save_runtime()
        try:
            _live_feed(
                "close",
                f"CLOSE {sym.split('/')[0]} {reason.value} pnl={res.pnl:+.2f}",
                symbol=sym, reason=reason.value, pnl=res.pnl, r=res.r_multiple,
            )
        except Exception:
            pass
        self.notify.send(f"CLOSE {sym} {reason.value} pnl={res.pnl:+.2f} ({res.r_multiple:+.2f}R)")


