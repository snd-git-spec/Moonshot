#!/usr/bin/env bash
set -euo pipefail
UID_NUM="$(id -u)"
launchctl bootout "gui/$UID_NUM/com.cake.moonshot.autopilot" 2>/dev/null || true
launchctl bootout "gui/$UID_NUM/com.cake.moonshot.dashboard" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/com.cake.moonshot.autopilot.plist"
rm -f "$HOME/Library/LaunchAgents/com.cake.moonshot.dashboard.plist"
pkill -f "-m moonshot autopilot" 2>/dev/null || true
pkill -f "-m moonshot paper" 2>/dev/null || true
pkill -f "streamlit run dashboard.py" 2>/dev/null || true
echo "Removed Moonshot keepalive."
