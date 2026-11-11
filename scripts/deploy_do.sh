#!/usr/bin/env bash
# Deploy current HEAD to the DigitalOcean moonshot droplet and restart hunt.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${MOONSHOT_DO_HOST:-178.62.113.173}"
KEY="${MOONSHOT_DO_KEY:-$HOME/.ssh/moonshot_do}"
REMOTE="${MOONSHOT_DO_PATH:-/opt/moonshot}"

cd "$ROOT"
SHA="$(git rev-parse --short HEAD)"
echo "Deploying $SHA → root@$HOST:$REMOTE"

rsync -az --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '.mypy_cache/' \
  --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' \
  --exclude '.moonshot_cache/' \
  --exclude '.cursor/' \
  --exclude 'logs/' \
  --exclude '.env' \
  --exclude '.moonshot_autopilot.json' \
  --exclude '.moonshot_live.jsonl' \
  --exclude '.moonshot_paper_runtime.json' \
  --exclude '.moonshot_search_winner.json' \
  --exclude 'moonshot.db' \
  --exclude 'moonshot.db-*' \
  --exclude 'moonshot.db.bak*' \
  --exclude '*.pyc' \
  -e "ssh -i $KEY -o BatchMode=yes -o StrictHostKeyChecking=accept-new" \
  "$ROOT/" "root@$HOST:$REMOTE/"

ssh -i "$KEY" -o BatchMode=yes "root@$HOST" bash -s <<EOF
set -euo pipefail
cd $REMOTE
# Hunt multi-TF matrix + paper + streamlit need more than 1.7G; 1700M cgroup
# OOM-killed autopilot mid-pass every ~8–10 min (Recent searches froze).
# Droplet has ~2G RAM + 2G swap — allow swap rather than silent kills.
mkdir -p /etc/systemd/system/moonshot.service.d
cat >/etc/systemd/system/moonshot.service.d/override.conf <<'OVR'
[Service]
Environment=PORT=8501
Environment=STREAMLIT_SERVER_ADDRESS=127.0.0.1
MemoryMax=2800M
MemorySwapMax=infinity
OVR
# Clamp corrupt generation / clear stuck busy from dead hunt mid-pass.
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/opt/moonshot/.moonshot_autopilot.json")
s = json.loads(p.read_text())
passes = int(s.get("improve_passes") or 0)
gen = int(s.get("generation") or 0)
print("before", {"generation": gen, "improve_passes": passes, "busy": s.get("busy")})
if passes > 0 and gen > passes + 50:
    s["generation"] = passes
s["busy"] = False
s["pending_champion_rescue"] = False
p.write_text(json.dumps(s, indent=2))
print("after", {"generation": s.get("generation"), "busy": s.get("busy")})
PY
.venv/bin/pip install -q -e .
systemctl daemon-reload
systemctl restart moonshot.service
sleep 8
systemctl is-active moonshot.service
ps aux | grep -E 'moonshot autopilot|moonshot paper|streamlit run' | grep -v grep || true
python3 - <<'PY'
import json
from pathlib import Path
s = json.loads(Path("/opt/moonshot/.moonshot_autopilot.json").read_text())
he = s.get("hunt_event") or {}
print("remote_gen", s.get("generation"), "busy", s.get("busy"),
      "started", s.get("started_at"),
      "hunt", he.get("kind"), he.get("generation"), str(he.get("message") or "")[:100])
PY
echo "deployed_sha $SHA"
grep -n 'autopilot pid' /opt/moonshot/scripts/render_start.sh | head -3
EOF

echo "Deployed $SHA to https://178-62-113-173.sslip.io/"
