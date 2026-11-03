#!/usr/bin/env bash
# Start Moonshot so it survives Cursor closing (and Cursor shell teardown).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
mkdir -p logs

WITH_DASH=0
if [[ "${1:-}" == "--with-dashboard" ]]; then
  WITH_DASH=1
fi

# Fully detach from parent (Cursor) using a new session
"$ROOT/.venv/bin/python" - <<PY
import os, subprocess, time
from pathlib import Path

root = Path(r"$ROOT")
venv_python = root / ".venv" / "bin" / "python"
streamlit = root / ".venv" / "bin" / "streamlit"
logs = root / "logs"
logs.mkdir(exist_ok=True)

def kill_patterns(*pats):
    for pat in pats:
        subprocess.run(["pkill", "-f", pat], check=False)
    time.sleep(1)

def spawn(cmd, log_name, pid_name):
    logf = open(logs / log_name, "a", buffering=1)
    proc = subprocess.Popen(
        cmd,
        cwd=str(root),
        stdout=logf,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # detach from Cursor / terminal job control
        env={**os.environ, "LIVE_TRADING_ENABLED": "0"},
    )
    (logs / pid_name).write_text(str(proc.pid))
    print(f"{pid_name}: {proc.pid} -> logs/{log_name}")
    return proc.pid

kill_patterns("-m moonshot autopilot", "-m moonshot paper")
spawn(
    [str(venv_python), "-m", "moonshot", "autopilot", "--cooldown-seconds", "60"],
    "autopilot.log",
    "autopilot.pid",
)

if $WITH_DASH:
    kill_patterns("streamlit run dashboard.py")
    spawn(
        [str(streamlit), "run", str(root / "dashboard.py"),
         "--server.headless", "true",
         "--server.address", "127.0.0.1",
         "--server.port", "8501"],
        "dashboard.log",
        "dashboard.pid",
    )
    print("Dashboard: http://127.0.0.1:8501/live")

print("Detached. Safe to close Cursor. Stop with: ./stop_moonshot.sh")
PY
