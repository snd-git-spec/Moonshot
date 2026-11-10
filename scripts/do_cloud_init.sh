#!/usr/bin/env bash
# DigitalOcean cloud-init / first-boot: Moonshot always-on UI + autopilot.
# Expects repo at /opt/moonshot and .env already in place.
set -euo pipefail

export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3.12 python3.12-venv python3-pip git ufw fail2ban

ufw allow OpenSSH
ufw allow 8501/tcp
ufw --force enable

install -d -o root -g root /opt/moonshot
cd /opt/moonshot

if [[ ! -d .git ]]; then
  echo "Clone Moonshot into /opt/moonshot before enabling the service." >&2
  exit 1
fi

python3.12 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
pip install -e .

install -d /var/lib/moonshot /var/log/moonshot
# Persist state next to the app (same filenames Render used on disk).
for f in \
  .moonshot_autopilot.json \
  .moonshot_paper_runtime.json \
  .moonshot_rules.json \
  .moonshot_search_winner.json \
  .moonshot_live.jsonl
do
  [[ -e "/var/lib/moonshot/$f" ]] || : >"/var/lib/moonshot/$f"
  ln -sfn "/var/lib/moonshot/$f" "/opt/moonshot/$f"
done
mkdir -p /var/lib/moonshot/.moonshot_cache
ln -sfn /var/lib/moonshot/.moonshot_cache /opt/moonshot/.moonshot_cache
mkdir -p /opt/moonshot/logs
ln -sfn /var/log/moonshot /opt/moonshot/logs

cat >/etc/systemd/system/moonshot.service <<'UNIT'
[Unit]
Description=Moonshot Streamlit UI + autopilot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/moonshot
Environment=PATH=/opt/moonshot/.venv/bin:/usr/bin
Environment=LIVE_TRADING_ENABLED=0
Environment=PORT=8501
Environment=AUTOPILOT_COOLDOWN_SECONDS=60
EnvironmentFile=-/opt/moonshot/.env
ExecStart=/opt/moonshot/.venv/bin/bash /opt/moonshot/scripts/render_start.sh
Restart=always
RestartSec=5
# Streamlit + hunt can sit near 512MB; give headroom on a 1GB+ droplet.
MemoryMax=900M

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now moonshot.service
systemctl --no-pager --full status moonshot.service || true
