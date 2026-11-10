#!/usr/bin/env python3
"""Money/deploy scoreboard — exit non-zero when Live is stranded vs champion.

Checks:
  1. .env STRATEGY/TIMEFRAME/LEVERAGE fingerprint
  2. Live feed book (if present)
  3. Champion knobs vs .env (CRITICAL when champ week >> live week)
  4. Last paper closes + bleed halt
  5. Last history improved without matching .env (apply-fail)

Usage:
  python -m scripts.scoreboard_check
  python scripts/scoreboard_check.py --json
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moonshot.hunt.constants import (  # noqa: E402
    _env_book_fingerprint,
    _load_autopilot_state,
)
from moonshot.hunt.scoring import _paper_bleed  # noqa: E402
from moonshot.paths import DB_PATH  # noqa: E402

# Champ week must beat Live by this much (and be at least this strong) to flag strand.
CHAMP_TRAIL_FAIL_PP = 0.15
CHAMP_MIN_WEEK = 0.10


def _paper_closes(n: int = 10) -> list[dict]:
    if not DB_PATH.exists():
        return []
    con = sqlite3.connect(str(DB_PATH))
    try:
        rows = con.execute(
            "SELECT ts, symbol, pnl, config_json FROM trades "
            "WHERE mode = 'paper' ORDER BY ts DESC LIMIT ?",
            (n,),
        ).fetchall()
    finally:
        con.close()
    out = []
    for ts, sym, pnl, cfg_json in rows:
        strat = tf = lev = None
        if cfg_json:
            try:
                cfg = json.loads(cfg_json)
                strat = cfg.get("strategy")
                tf = cfg.get("timeframe")
                lev = cfg.get("leverage")
            except Exception:
                pass
        out.append(
            {
                "ts": ts,
                "symbol": sym,
                "pnl": float(pnl or 0),
                "strategy": strat,
                "timeframe": tf,
                "leverage": lev,
            }
        )
    return out


def _risk_halt() -> dict | None:
    if not DB_PATH.exists():
        return None
    con = sqlite3.connect(str(DB_PATH))
    try:
        row = con.execute(
            "SELECT ts, halt_reason, kill, streak, daily_halt, equity, peak "
            "FROM risk_snapshot WHERE mode='paper' ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        con.close()
    if not row:
        return None
    return {
        "ts": row[0],
        "halt_reason": row[1],
        "kill": row[2],
        "streak": row[3],
        "daily_halt": row[4],
        "equity": row[5],
        "peak": row[6],
    }


def check() -> dict:
    st = _load_autopilot_state()
    env_fp = _env_book_fingerprint()
    ch = st.get("champion") or {}
    ck = ch.get("knobs") or {}
    champ_fp = (
        str(ck.get("strategy") or ""),
        str(ck.get("timeframe") or ""),
        str(ck.get("leverage") or ""),
    )
    live = st.get("live") or {}
    live_fp = (
        str(live.get("strategy") or ""),
        str(live.get("timeframe") or ""),
        str(live.get("leverage") or ""),
    )
    # Live feed may omit strategy — fill from env when TF/lev match
    if not live_fp[0] and live_fp[1] and live_fp[2]:
        live_fp = (env_fp[0], live_fp[1], str(live_fp[2]))

    champ_week = ch.get("week_ret_live")
    if champ_week is None:
        champ_week = ch.get("week_ret")
    try:
        champ_week_f = float(champ_week) if champ_week is not None else None
    except (TypeError, ValueError):
        champ_week_f = None

    hist = list(st.get("history") or [])
    last = hist[-1] if hist else {}
    closes = _paper_closes(10)
    bleed = _paper_bleed()
    risk = _risk_halt()

    failures: list[str] = []
    warnings: list[str] = []

    if live_fp[1] and live_fp[2]:
        if (live_fp[1], str(live_fp[2])) != (env_fp[1], env_fp[2]):
            failures.append(
                f"paper Live feed book {live_fp} != .env {env_fp}"
            )

    if champ_fp[0] and champ_fp != env_fp:
        trail_ok = False
        if champ_week_f is not None and champ_week_f >= CHAMP_MIN_WEEK:
            # Compare to last hist week_return as proxy for live week if present
            try:
                live_week = float(last.get("week_return")) if last.get("week_return") is not None else None
            except (TypeError, ValueError):
                live_week = None
            if live_week is None:
                warnings.append(
                    f"champ {champ_fp} != .env {env_fp} (week={champ_week_f:+.0%}) — "
                    "cannot score trail without live week"
                )
            elif champ_week_f - live_week >= CHAMP_TRAIL_FAIL_PP:
                failures.append(
                    f"STRANDED: champ {champ_fp} week={champ_week_f:+.0%} "
                    f"vs live_week={live_week:+.0%} but .env is {env_fp}"
                )
                trail_ok = True
            if not trail_ok and live_week is not None:
                warnings.append(
                    f"champ {champ_fp} != .env {env_fp} "
                    f"(champ {champ_week_f:+.0%} live_week {live_week:+.0%})"
                )

    if last.get("improved") and last.get("promote_reason"):
        # If last improve claimed a book, env should match that S/T/L when present
        hs = str(last.get("strategy") or "")
        ht = str(last.get("timeframe") or "")
        hl = str(last.get("leverage") or "")
        if hs and ht and hl:
            hist_fp = (hs, ht, hl)
            if hist_fp != env_fp:
                failures.append(
                    f"apply-fail: history improved={hist_fp} reason={last.get('promote_reason')} "
                    f"but .env={env_fp}"
                )

    # Bleed warn only for closes inside the per-book window (post-promote).
    if (
        int(bleed.get("n") or 0) >= int(bleed.get("need") or 5)
        and int(bleed.get("losses") or 0) >= int(bleed.get("need") or 5)
        and not (st.get("paper_bleed_halt") or bleed.get("bleed"))
    ):
        warnings.append(
            f"window bleed n={bleed.get('n')} all losers pnl={bleed.get('pnl')} "
            "but bleed halt not armed"
        )

    report = {
        "ok": not failures,
        "env": {"strategy": env_fp[0], "timeframe": env_fp[1], "leverage": env_fp[2]},
        "champion": {
            "strategy": champ_fp[0],
            "timeframe": champ_fp[1],
            "leverage": champ_fp[2],
            "week_ret": champ_week_f,
            "matches_env": champ_fp == env_fp if champ_fp[0] else None,
        },
        "live_feed": {
            "leverage": live.get("leverage"),
            "timeframe": live.get("timeframe"),
            "bleed_halt": live.get("bleed_halt"),
            "message": str(live.get("message") or "")[:160],
        },
        "bleed": bleed,
        "paper_bleed_halt": st.get("paper_bleed_halt"),
        "paper_bleed_window_start": st.get("paper_bleed_window_start"),
        "risk": risk,
        "last_history": {
            "generation": last.get("generation"),
            "improved": last.get("improved"),
            "promote_reason": last.get("promote_reason"),
            "strategy": last.get("strategy"),
            "timeframe": last.get("timeframe"),
            "leverage": last.get("leverage"),
            "week_return": last.get("week_return"),
        },
        "closes": closes[:5],
        "close_pnl_sum5": round(sum(c["pnl"] for c in closes[:5]), 2) if closes else 0.0,
        "failures": failures,
        "warnings": warnings,
    }
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args()
    report = check()
    if args.json:
        print(json.dumps(report, indent=2, default=str))
    else:
        env = report["env"]
        print(
            f"env {env['strategy']} {env['leverage']}x@{env['timeframe']} "
            f"ok={report['ok']}"
        )
        ch = report["champion"]
        print(
            f"champ {ch['strategy']} {ch['leverage']}x@{ch['timeframe']} "
            f"week={ch['week_ret']} match={ch['matches_env']}"
        )
        print(f"bleed {report['bleed']} halt_flag={report['paper_bleed_halt']}")
        print(f"closes5_sum {report['close_pnl_sum5']} risk={report['risk']}")
        for f in report["failures"]:
            print(f"FAIL: {f}")
        for w in report["warnings"]:
            print(f"WARN: {w}")
        if report["ok"]:
            print("SCOREBOARD OK")
        else:
            print("SCOREBOARD FAIL")
    return 0 if report["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
