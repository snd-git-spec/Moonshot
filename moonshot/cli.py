from __future__ import annotations

import argparse
import logging
import os
import sys


log = logging.getLogger("moonshot")

from moonshot.types import (
    BotError,
    ConfigError,
)
from moonshot.config import (
    Config,
    setup_logging,
)
from moonshot.features import (
    prepare_data,
)
from moonshot.data import (
    fetch_historical,
    synthetic_ohlcv,
)
from moonshot.brokers import (
    MarketData,
    MexcBroker,
    PaperBroker,
)
from moonshot.backtest import (
    results_table,
    run_grid,
    verdicts,
)
from moonshot.symbols import (
    DEFAULT_SYMBOLS,
    PAPER_SYMBOLS,
)
from moonshot.live import (
    LiveRunner,
)
from moonshot.selftest import (
    selftest,
)
from moonshot.hunt.autopilot import (
    cmd_autopilot,
)


def cmd_backtest(args: argparse.Namespace) -> None:
    cfg = Config.from_env()
    symbols = args.symbols or DEFAULT_SYMBOLS
    if args.synthetic:
        print("!! SYNTHETIC DATA - plumbing test only, results mean nothing !!")
        raw = {s: synthetic_ohlcv(s, args.bars) for s in symbols}
    else:
        raw = {
            s: fetch_historical(s, cfg.timeframe, args.bars, refresh=args.refresh) for s in symbols
        }
    data = prepare_data(raw, cfg)
    t0 = min(sd.bars[0].Index for sd in data.values())
    t1 = max(sd.bars[-1].Index for sd in data.values())
    split = t0 + (t1 - t0) * (1 - args.oos)
    print(f"data {t0:%Y-%m-%d} -> {t1:%Y-%m-%d} | in-sample < {split:%Y-%m-%d} <= out-of-sample")

    ins = run_grid(data, cfg, end=split)
    oos = run_grid(data, cfg, start=split)
    print("\n=== IN-SAMPLE (choose settings here) ===\n" + results_table(ins))
    print("\n=== OUT-OF-SAMPLE (judge them here, do not tune on it) ===\n" + results_table(oos))
    print("\n=== Decisions from in-sample ===")
    for line in verdicts(ins):
        print(" -", line)
    few = [r for r in oos if r.trades < 30]
    if few:
        print(
            f"\nWARNING: {len(few)}/{len(oos)} out-of-sample runs have <30 trades; "
            "differences between configs are noise at this sample size."
        )
    print(
        "WARNING: 8 configs on one dataset invites overfitting; prefer configs that are "
        "good in BOTH halves, and confirm in paper trading."
    )


def _live_runner(live: bool, symbols: list) -> LiveRunner:
    cfg = Config.from_env()
    market = MarketData(cfg)
    # Live honors MEXC_DRY_RUN=1 (default in .env.example) — no orders until verified.
    broker = MexcBroker(cfg) if live else PaperBroker(cfg)
    return LiveRunner(cfg, broker, market, symbols, live)


def main() -> None:
    ap = argparse.ArgumentParser(description="moonshot trend bot")
    sub = ap.add_subparsers(dest="cmd")
    bt = sub.add_parser("backtest")
    bt.add_argument("--synthetic", action="store_true")
    bt.add_argument("--bars", type=int, default=3000)
    bt.add_argument("--symbols", nargs="*")
    bt.add_argument("--oos", type=float, default=0.3, help="out-of-sample fraction")
    bt.add_argument("--refresh", action="store_true", help="ignore the CSV cache")
    for name in ("paper", "live"):
        p = sub.add_parser(name)
        p.add_argument("--symbols", nargs="*")
    auto = sub.add_parser("autopilot", help="keep paper running and continuously self-improve")
    auto.add_argument("--symbols", nargs="*")
    auto.add_argument("--bars", type=int, default=15000)
    auto.add_argument("--oos", type=float, default=0.3)
    auto.add_argument(
        "--cooldown-seconds",
        type=int,
        default=60,
        help="pause only AFTER targets are met (default 60); failed hunts re-run immediately",
    )
    auto.add_argument(
        "--daily",
        action="store_true",
        help="old behavior: improve once per day instead of continuously",
    )
    auto.add_argument("--every-hours", type=float, default=24.0, help="only used with --daily")
    auto.add_argument(
        "--tune-only",
        action="store_true",
        help="small local nudges only (default is wide HUNT search)",
    )
    auto.add_argument(
        "--improve-only",
        action="store_true",
        help="run one improve/hunt pass then exit (no paper child)",
    )
    auto.add_argument("--refresh", action="store_true")
    sub.add_parser("selftest")
    args = ap.parse_args()

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    try:
        if args.cmd == "selftest":
            sys.exit(0 if selftest() else 1)
        elif args.cmd in (None, "backtest"):
            if args.cmd is None:
                args = bt.parse_args([])
            cmd_backtest(args)
        elif args.cmd == "paper":
            _live_runner(False, args.symbols or PAPER_SYMBOLS).run_forever()
        elif args.cmd == "autopilot":
            cmd_autopilot(args)
        elif args.cmd == "live":
            # VERIFY checklist before LIVE_VERIFIED=1 (see MexcBroker docstring):
            # openType isolated, stops+reduceOnly, balance/uPnL, contract size/precision.
            if os.getenv("LIVE_TRADING_ENABLED") != "1":
                raise ConfigError(
                    "Refusing live mode: set LIVE_TRADING_ENABLED=1 after 4+ weeks "
                    "of paper results that match the backtest."
                )
            if os.getenv("LIVE_VERIFIED") != "1":
                raise ConfigError(
                    "Refusing live mode: set LIVE_VERIFIED=1 only after confirming "
                    "MexcBroker VERIFY checklist (openType, stops, reduceOnly, "
                    "balance, contract size) against current MEXC/CCXT docs."
                )
            log.warning("LIVE TRADING - use the smallest size first")
            _live_runner(True, args.symbols or DEFAULT_SYMBOLS).run_forever()
    except BotError as e:
        log.error("%s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
