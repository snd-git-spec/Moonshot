#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
UID_NUM="$(id -u)"

# Prefer stopping LaunchAgents if installed
launchctl bootout "gui/$UID_NUM/com.cake.moonshot.autopilot" 2>/dev/null || true
launchctl bootout "gui/$UID_NUM/com.cake.moonshot.dashboard" 2>/dev/null || true

pkill -f "moonshot.py autopilot" 2>/dev/null || true
pkill -f "moonshot.py paper" 2>/dev/null || true
pkill -f "streamlit run dashboard.py" 2>/dev/null || true
rm -f "$ROOT/logs/autopilot.pid" "$ROOT/logs/dashboard.pid" 2>/dev/null || true
echo "Stopped Moonshot processes."
