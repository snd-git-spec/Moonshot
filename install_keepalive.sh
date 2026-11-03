#!/usr/bin/env bash
# Install Login Items via launchd — auto-start on login + restart if crashed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
UID_NUM="$(id -u)"
AGENTS="$HOME/Library/LaunchAgents"
mkdir -p "$AGENTS" "$ROOT/logs"

AUTO_PLIST="$AGENTS/com.cake.moonshot.autopilot.plist"
DASH_PLIST="$AGENTS/com.cake.moonshot.dashboard.plist"

cat > "$AUTO_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.cake.moonshot.autopilot</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ROOT/.venv/bin/python</string>
    <string>$ROOT/moonshot.py</string>
    <string>autopilot</string>
    <string>--cooldown-seconds</string>
    <string>60</string>
  </array>
  <key>EnvironmentVariables</key>
  <dict>
    <key>LIVE_TRADING_ENABLED</key><string>0</string>
  </dict>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$ROOT/logs/autopilot.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/autopilot.log</string>
</dict>
</plist>
EOF

cat > "$DASH_PLIST" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>com.cake.moonshot.dashboard</string>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>ProgramArguments</key>
  <array>
    <string>$ROOT/.venv/bin/streamlit</string>
    <string>run</string>
    <string>$ROOT/dashboard.py</string>
    <string>--server.headless</string>
    <string>true</string>
    <string>--server.address</string>
    <string>127.0.0.1</string>
    <string>--server.port</string>
    <string>8501</string>
  </array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$ROOT/logs/dashboard.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/dashboard.log</string>
</dict>
</plist>
EOF

launchctl bootout "gui/$UID_NUM/com.cake.moonshot.autopilot" 2>/dev/null || true
launchctl bootout "gui/$UID_NUM/com.cake.moonshot.dashboard" 2>/dev/null || true
# Also clear any orphan processes
pkill -f "moonshot.py autopilot" 2>/dev/null || true
pkill -f "moonshot.py paper" 2>/dev/null || true
pkill -f "streamlit run dashboard.py" 2>/dev/null || true
sleep 1

launchctl bootstrap "gui/$UID_NUM" "$AUTO_PLIST"
launchctl bootstrap "gui/$UID_NUM" "$DASH_PLIST"
launchctl enable "gui/$UID_NUM/com.cake.moonshot.autopilot"
launchctl enable "gui/$UID_NUM/com.cake.moonshot.dashboard"
launchctl kickstart -k "gui/$UID_NUM/com.cake.moonshot.autopilot"
launchctl kickstart -k "gui/$UID_NUM/com.cake.moonshot.dashboard"

echo "Installed keepalive LaunchAgents."
echo "Dashboard: http://127.0.0.1:8501/live"
echo "Unload later: ./uninstall_keepalive.sh"
