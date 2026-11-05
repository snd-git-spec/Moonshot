"""Canonical config fingerprint for trade attribution (work unit 0.2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, is_dataclass
from typing import Any


def canonical_json(cfg: Any) -> str:
    """Stable JSON for hashing — sorted keys, default=str, no whitespace quirks."""
    if is_dataclass(cfg) and not isinstance(cfg, type):
        raw = asdict(cfg)
    elif isinstance(cfg, dict):
        raw = dict(cfg)
    else:
        raw = dict(getattr(cfg, "__dict__", {}) or {})
    return json.dumps(raw, sort_keys=True, default=str, separators=(",", ":"))


def config_hash(cfg: Any, *, n: int = 16) -> str:
    digest = hashlib.sha256(canonical_json(cfg).encode("utf-8")).hexdigest()
    return digest[: int(n)]
