#!/usr/bin/env bash
# Render Web Service: always-on autopilot + Streamlit UI on $PORT.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

# Persist hunt/paper state across deploys when a disk is mounted.
DISK="${RENDER_DISK_PATH:-/var/data}"
if [[ -d "$DISK" ]]; then
  mkdir -p "$DISK" "$DISK/.moonshot_cache"
  for f in \
    .moonshot_autopilot.json \
    .moonshot_paper_runtime.json \
    .moonshot_rules.json \
    .moonshot_search_winner.json \
    .moonshot_live.jsonl \
    .env
  do
    if [[ -L "$ROOT/$f" || ! -e "$ROOT/$f" ]]; then
      [[ -e "$DISK/$f" ]] || : >"$DISK/$f"
      ln -sfn "$DISK/$f" "$ROOT/$f"
    fi
  done
  if [[ -L "$ROOT/.moonshot_cache" || ! -e "$ROOT/.moonshot_cache" ]]; then
    ln -sfn "$DISK/.moonshot_cache" "$ROOT/.moonshot_cache"
  fi
fi

export LIVE_TRADING_ENABLED="${LIVE_TRADING_ENABLED:-0}"
PORT="${PORT:-8501}"

# Autopilot (hunt + paper child) — background
python -m moonshot autopilot --cooldown-seconds "${AUTOPILOT_COOLDOWN_SECONDS:-60}" \
  >>logs/autopilot.log 2>&1 &
echo $! >logs/autopilot.pid

# Streamlit must stay in foreground so Render health-checks $PORT
exec python -m streamlit run dashboard.py \
  --server.port "$PORT" \
  --server.address 0.0.0.0 \
  --server.headless true
