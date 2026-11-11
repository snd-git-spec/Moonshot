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
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from moonshot.hunt.constants import (  # noqa: E402
    ENV_PATH,
    HUNT_STALE_FINISH_HOURS,
    _env_book_fingerprint,
    _load_autopilot_state,
)
from moonshot.hunt.health import hunt_pass_health  # noqa: E402
from moonshot.hunt.scoring import _paper_bleed  # noqa: E402
from moonshot.paths import DB_PATH  # noqa: E402

# Champ week must beat Live by this much (and be at least this strong) to flag strand.
CHAMP_TRAIL_FAIL_PP = 0.15
CHAMP_MIN_WEEK = 0.10
# Hunt mode with a dead autopilot process is never "SCOREBOARD OK".
AUTOPILOT_PID_PATH = ROOT / "logs" / "autopilot.pid"
# Back-compat alias — finish staleness lives in hunt.health / constants.
HUNT_STALE_IMPROVE_HOURS = float(HUNT_STALE_FINISH_HOURS)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def _pgrep_autopilot() -> int:
    """Find live autopilot even when launchd did not refresh logs/autopilot.pid."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["pgrep", "-f", "python -m moonshot autopilot"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 0
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return int(line.split()[0])
        except ValueError:
            continue
    return 0


def _autopilot_process() -> dict:
    raw = ""
    if AUTOPILOT_PID_PATH.exists():
        raw = AUTOPILOT_PID_PATH.read_text().strip()
    try:
        pid = int(raw) if raw else 0
    except ValueError:
        pid = 0
    alive = _pid_alive(pid)
    source = "pidfile" if alive else "none"
    if not alive:
        # launchd KeepAlive does not maintain our pidfile — fall back to pgrep
        probed = _pgrep_autopilot()
        if probed and _pid_alive(probed):
            pid = probed
            alive = True
            source = "pgrep"
            try:
                AUTOPILOT_PID_PATH.parent.mkdir(parents=True, exist_ok=True)
                AUTOPILOT_PID_PATH.write_text(str(pid))
            except OSError:
                pass
    return {
        "pid": pid,
        "alive": alive,
        "pid_path": str(AUTOPILOT_PID_PATH),
        "source": source,
    }


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _hours_since(ts: str | None) -> float | None:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


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

    proc = _autopilot_process()
    mode = str(st.get("mode") or "")
    last_improve_at = str(st.get("last_improve_at") or "")
    improve_age_h = _hours_since(last_improve_at)
    if mode == "hunt" and not proc["alive"]:
        failures.append(
            f"HUNT_DEAD: mode=hunt but autopilot pid={proc['pid'] or '?'} "
            f"not running (last_improve_at={last_improve_at or 'missing'})"
        )
    elif mode == "hunt" and proc["alive"]:
        # Finish signal only — busy / fresh started_at / improve_start do NOT carve out.
        pass_h = hunt_pass_health(st, process_alive=True, mode=mode)
        failures.extend(pass_h["failures"])
        warnings.extend(pass_h["warnings"])

    if (
        st.get("pending_champion_rescue")
        and champ_fp[0]
        and champ_fp == env_fp
    ):
        failures.append(
            "STUCK_RESCUE: pending_champion_rescue=True but already on champ "
            f"{champ_fp[0]} {champ_fp[2]}x@{champ_fp[1]} — heavy hunt blocked"
        )

    try:
        gen_now = int(st.get("generation") or 0)
        passes = int(st.get("improve_passes") or 0)
    except (TypeError, ValueError):
        gen_now, passes = 0, 0
    if passes > 0 and gen_now > passes + 50:
        failures.append(
            f"GEN_CORRUPT: generation={gen_now} >> improve_passes={passes}"
        )

    # Paper tick adx vs .env — catches partial-knob apply drift.
    env_adx = None
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text().splitlines():
            if line.strip().startswith("ADX_ENTER="):
                try:
                    env_adx = float(line.split("=", 1)[1].strip().strip('"').strip("'"))
                except ValueError:
                    env_adx = None
                break
    live_adx = live.get("adx_enter")
    try:
        live_adx_f = float(live_adx) if live_adx is not None else None
    except (TypeError, ValueError):
        live_adx_f = None
    if env_adx is not None and live_adx_f is not None and abs(env_adx - live_adx_f) >= 0.5:
        failures.append(
            f"KNOB_DRIFT: paper live adx_enter={live_adx_f} != .env ADX_ENTER={env_adx}"
        )

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
        "autopilot": {
            **proc,
            "mode": mode,
            "busy": bool(st.get("busy")),
            "generation": st.get("generation"),
            "last_improve_at": last_improve_at or None,
            "improve_age_hours": round(improve_age_h, 2) if improve_age_h is not None else None,
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
        ap = report.get("autopilot") or {}
        print(
            f"autopilot alive={ap.get('alive')} pid={ap.get('pid')} "
            f"mode={ap.get('mode')} gen={ap.get('generation')} "
            f"busy={ap.get('busy')} improve_age_h={ap.get('improve_age_hours')}"
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
