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
# Cap cgroup so hunt cannot eat the whole 2G box (2800M caused global OOM).
# Multi-TF hunt must stay sequential + prune (see improve.py) under this cap.
mkdir -p /etc/systemd/system/moonshot.service.d
cat >/etc/systemd/system/moonshot.service.d/override.conf <<'OVR'
[Service]
Environment=PORT=8501
Environment=STREAMLIT_SERVER_ADDRESS=127.0.0.1
Environment=MOONSHOT_SOFT_WORKERS=1
MemoryMax=2000M
OVR
# Clamp corrupt generation / clear stuck busy from dead hunt mid-pass.
# Deploy clears legacy TF-slim. Arm one-shot denser skip only when unfrozen
# (post-reset protection); when AUTOPILOT_FROZEN=1 skip is waste.
python3 - <<'PY'
import json
from pathlib import Path
p = Path("/opt/moonshot/.moonshot_autopilot.json")
s = json.loads(p.read_text())
passes = int(s.get("improve_passes") or 0)
gen = int(s.get("generation") or 0)
frozen = False
env_p = Path("/opt/moonshot/.env")
if env_p.exists():
    for line in env_p.read_text().splitlines():
        if line.strip().startswith("AUTOPILOT_FROZEN="):
            val = line.split("=", 1)[1].strip().strip("\"'")
            frozen = val in ("1", "true", "TRUE", "yes", "YES")
            break
print("before", {
    "generation": gen,
    "improve_passes": passes,
    "busy": s.get("busy"),
    "last_improve_at": s.get("last_improve_at"),
    "force_matrix_slim": s.get("force_matrix_slim"),
    "force_skip_denser": s.get("force_skip_denser"),
    "AUTOPILOT_FROZEN": frozen,
})
if passes > 0 and gen > passes + 50:
    s["generation"] = passes
s["busy"] = False
s["pending_champion_rescue"] = False
s.pop("force_matrix_slim", None)  # never TF-slim after deploy
if frozen:
    # Practice freeze: keep denser on; do not arm shallow post-deploy pass.
    s.pop("hunt_recover_reason", None)
else:
    s["force_skip_denser"] = True  # one-shot denser skip; matrix stays FULL
    s["hunt_recover_reason"] = "deploy"
# Do not clear last_improve_at — closed-loop verify needs the pre-deploy finish stamp.
p.write_text(json.dumps(s, indent=2))
print("after", {
    "generation": s.get("generation"),
    "busy": s.get("busy"),
    "improve_passes": s.get("improve_passes"),
    "last_improve_at": s.get("last_improve_at"),
    "force_matrix_slim": s.get("force_matrix_slim"),
    "force_skip_denser": s.get("force_skip_denser"),
    "hunt_recover_reason": s.get("hunt_recover_reason"),
    "AUTOPILOT_FROZEN": frozen,
})
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
print(
    "remote_gen", s.get("generation"),
    "busy", s.get("busy"),
    "improve_passes", s.get("improve_passes"),
    "last_improve_at", s.get("last_improve_at"),
    "pass_started_at", s.get("pass_started_at"),
    "force_slim", s.get("force_matrix_slim"),
    "force_skip_denser", s.get("force_skip_denser"),
    "recover", s.get("hunt_recover_reason"),
    "started", s.get("started_at"),
    "hunt", he.get("kind"), he.get("generation"), str(he.get("message") or "")[:100],
)
PY
echo "deployed_sha $SHA"
grep -n 'autopilot pid' /opt/moonshot/scripts/render_start.sh | head -3
EOF

echo "Deployed $SHA to https://178-62-113-173.sslip.io/"
