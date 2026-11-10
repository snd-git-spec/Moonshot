#!/usr/bin/env bash
# Always-on: autopilot (hunt + paper) + Streamlit UI on $PORT.
# Autopilot death must take the service down so systemd/KeepAlive restarts hunt.
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
# Cloud nginx terminates TLS and proxies to localhost; local/dev may bind 0.0.0.0.
ADDR="${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"

python -m moonshot autopilot --cooldown-seconds "${AUTOPILOT_COOLDOWN_SECONDS:-60}" \
  >>logs/autopilot.log 2>&1 &
APID=$!
echo "$APID" >logs/autopilot.pid

python -m streamlit run dashboard.py \
  --server.port "$PORT" \
  --server.address "$ADDR" \
  --server.headless true \
  >>logs/dashboard.log 2>&1 &
SPID=$!
echo "$SPID" >logs/dashboard.pid

cleanup() {
  kill "$APID" "$SPID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# If hunt dies while UI stays up, the dashboard lies. Exit → systemd restart.
while true; do
  if ! kill -0 "$APID" 2>/dev/null; then
    echo "autopilot pid $APID exited — stopping UI so the service restarts" >&2
    exit 1
  fi
  if ! kill -0 "$SPID" 2>/dev/null; then
    echo "streamlit pid $SPID exited — stopping autopilot" >&2
    exit 1
  fi
  sleep 5
done
