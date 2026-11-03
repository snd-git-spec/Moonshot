#!/usr/bin/env python3
"""Smoke QA for Moonshot ops — run after changes: python qa_check.py"""

from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
fails: list[str] = []


def ok(msg: str) -> None:
    print(f"  [PASS] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")
    fails.append(msg)


def main() -> int:
    print("Moonshot QA")
    # 1) processes
    out = subprocess.check_output(["pgrep", "-fl", "-m moonshot|streamlit"], text=True, stderr=subprocess.DEVNULL) if False else ""
    try:
        out = subprocess.check_output(["bash", "-lc", "pgrep -fl '-m moonshot|streamlit' || true"], text=True)
    except Exception as e:
        bad(f"pgrep failed: {e}")
        out = ""
    if "-m moonshot autopilot" in out:
        ok("autopilot process running")
    else:
        bad("autopilot process NOT running")
    if "-m moonshot paper" in out:
        ok("paper process running")
    else:
        bad("paper process NOT running")
    if "streamlit" in out and "dashboard.py" in out:
        ok("dashboard process running")
    else:
        bad("dashboard process NOT running")

    # 2) HTTP
    for path in ("/", "/live", "/status", "/health"):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:8501{path}", timeout=5) as r:
                code = r.status
            if code == 200:
                ok(f"HTTP {path} -> {code}")
            else:
                bad(f"HTTP {path} -> {code}")
        except Exception as e:
            bad(f"HTTP {path} failed: {e}")

    # 3) state / feed parse
    feed_path = ROOT / ".moonshot_live.jsonl"
    if feed_path.exists():
        import pandas as pd
        n = 0
        for line in feed_path.read_text().splitlines()[-20:]:
            if not line.strip():
                continue
            row = json.loads(line)
            ts = pd.to_datetime(row.get("ts"), utc=True, format="ISO8601", errors="coerce")
            if pd.isna(ts):
                bad(f"unparseable feed ts: {row.get('ts')}")
            else:
                n += 1
        ok(f"live feed timestamps parse ({n} recent rows)")
    else:
        bad("missing .moonshot_live.jsonl")

    state_path = ROOT / ".moonshot_autopilot.json"
    if state_path.exists():
        st = json.loads(state_path.read_text())
        grade = st.get("grade") or {}
        if "score" in grade or st.get("elite"):
            ok(f"hunt score present ({grade.get('score') or st.get('elite', {}).get('score')}/100)")
        else:
            bad("no hunt score/grade in state yet")
        live = st.get("live") or {}
        if live.get("kind"):
            ok(f"last live event: {live.get('kind')}")
        else:
            bad("no live event in state")
    else:
        bad("missing .moonshot_autopilot.json")

    # 4) Timestamp parse smoke (do NOT import views/live.py — it runs Streamlit)
    import pandas as pd

    def _parse_ts(value):
        if value is None or value == "":
            return pd.NaT
        return pd.to_datetime(value, utc=True, format="ISO8601", errors="coerce")

    samples = [
        "2026-10-28T20:25:11.373128+00:00",
        "2026-10-28T20:25:11+00:00",
        "2026-10-28T20:00:00+00:00",
    ]
    for s in samples:
        if pd.isna(_parse_ts(s)):
            bad(f"parse failed for {s}")
        else:
            ok(f"parse ok {s[:22]}…")

    # 5) DB readable
    try:
        con = sqlite3.connect(f"file:{(ROOT / 'moonshot.db').resolve()}?mode=ro", uri=True)
        n_eq = con.execute("select count(*) from equity").fetchone()[0]
        n_tr = con.execute("select count(*) from trades").fetchone()[0]
        ok(f"db ok equity={n_eq} trades={n_tr}")
    except Exception as e:
        bad(f"db read failed: {e}")

    # 6) config loads
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=True)
        import moonshot as ms
        c = ms.Config.from_env()
        ok(f"config tf={c.timeframe} lev={c.leverage} adx={c.adx_enter}")
        assert c.timeframe in ("1h", "4h", "15m", "2h")
    except Exception as e:
        bad(f"config failed: {e}")

    print()
    if fails:
        print(f"QA FAILED ({len(fails)}):")
        for f in fails:
            print(f"  - {f}")
        return 1
    print("QA PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
