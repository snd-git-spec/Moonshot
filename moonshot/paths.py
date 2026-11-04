"""Filesystem roots for Moonshot — single source of truth."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = ROOT / ".env"
CACHE_DIR = ROOT / ".moonshot_cache"
DB_PATH = ROOT / "moonshot.db"
AUTOPILOT_STATE = ROOT / ".moonshot_autopilot.json"
LIVE_FEED = ROOT / ".moonshot_live.jsonl"
PAPER_RUNTIME_PATH = ROOT / ".moonshot_paper_runtime.json"
RULES_PATH = ROOT / ".moonshot_rules.json"

__all__ = [
    "ROOT",
    "ENV_PATH",
    "CACHE_DIR",
    "DB_PATH",
    "AUTOPILOT_STATE",
    "LIVE_FEED",
    "PAPER_RUNTIME_PATH",
    "RULES_PATH",
]
