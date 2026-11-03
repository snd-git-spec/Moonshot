"""Copy module names (including leading-underscore) into caller globals."""
from __future__ import annotations

from types import ModuleType
from typing import Any


def pull(mod: ModuleType, g: dict[str, Any]) -> None:
    for k, v in vars(mod).items():
        if k.startswith("__"):
            continue
        g[k] = v
