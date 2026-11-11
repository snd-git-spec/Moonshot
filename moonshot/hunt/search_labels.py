"""Honest Recent-searches labels in plain English.

Either a better setup was found and deployed, or it wasn't.
Silence/idle tweaks are not "better" — say that plainly.
"""

from __future__ import annotations

from typing import Any

_SILENCE_REASONS = frozenset({"silence_density", "silence_rotate"})
_LIVE_PATCH_REASONS = _SILENCE_REASONS | frozenset({"matrix_escape"})
_PEAK_RESTORE_REASONS = frozenset({"champion_rescue"})


def _why(h: dict[str, Any]) -> str:
    return str(h.get("promote_reason") or h.get("promote_tier") or "").strip()


def _tier(h: dict[str, Any]) -> str:
    return str(h.get("promote_tier") or "").strip()


def _is_live_patch(h: dict[str, Any]) -> bool:
    why, tier = _why(h), _tier(h)
    return why in _LIVE_PATCH_REASONS or tier in _LIVE_PATCH_REASONS


def what_happened_label(h: dict[str, Any]) -> str:
    """One plain line: better found, or not."""
    why = _why(h)
    tier = _tier(h)
    improved = bool(h.get("improved"))
    tested = h.get("tested")

    if improved:
        # Idle/escape tweak — changed paper, but did NOT beat the best known setup.
        if _is_live_patch(h):
            return "No better setup found · paper settings tweaked while waiting"
        if why in _PEAK_RESTORE_REASONS or tier in _PEAK_RESTORE_REASONS:
            return "Best known setup put back on paper"
        return "Better setup found · deployed to paper"

    if tested is not None:
        return f"No better setup found · checked {tested} ideas"
    return "No better setup found"


def holdout_week_note(h: dict[str, Any]) -> str:
    """This pass's holdout week % — no jargon suffixes."""
    wr = h.get("week_return")
    if wr is None:
        return "—"
    try:
        return f"{float(wr):+.0%}"
    except (TypeError, ValueError):
        return "—"


def matrix_scored_label(h: dict[str, Any]) -> str:
    """TF×lev cells tested this pass."""
    rows = h.get("tf_lev_matrix")
    if not isinstance(rows, list) or not rows:
        tfs = h.get("matrix_tfs")
        tested_n = h.get("tested")
        if isinstance(tfs, (list, tuple)) and tfs:
            tf_s = "/".join(str(t) for t in tfs)
            extra = f" · {tested_n} ideas" if tested_n is not None else ""
            return f"{tf_s}{extra}"
        ladder = h.get("leverage_ladder") or []
        if isinstance(ladder, list) and ladder:
            return _label_from_cells(ladder, tested_n=tested_n, legacy=True)
        return f"{tested_n} ideas" if tested_n is not None else "—"
    return _label_from_cells(rows, tested_n=h.get("tested"), legacy=False)


def _label_from_cells(
    cells_in: list,
    *,
    tested_n: Any,
    legacy: bool,
) -> str:
    cells: list[str] = []
    tfs: set[str] = set()
    levs: set[int] = set()
    for cell in cells_in:
        if not isinstance(cell, dict):
            continue
        try:
            lv = int(cell.get("leverage") or 0)
            tf = str(cell.get("timeframe") or cell.get("tf") or "")
        except (TypeError, ValueError):
            continue
        if not lv or not tf:
            continue
        levs.add(lv)
        tfs.add(tf)
        cells.append(f"{lv}×@{tf}")
    if not cells:
        return f"{tested_n} ideas" if tested_n is not None else "—"
    lev_s = "/".join(f"{x}×" for x in sorted(levs))
    tf_order = {"5m": 0, "15m": 1, "1h": 2, "4h": 3}
    tf_s = "/".join(sorted(tfs, key=lambda t: tf_order.get(t, 9)))
    extra = f" · {tested_n} ideas" if tested_n is not None else ""
    prefix = f"{len(cells)} cells"
    if legacy and tfs == {"5m"}:
        return f"{prefix} · {lev_s} × {tf_s} (old log){extra}"
    return f"{prefix} · {lev_s} × {tf_s}{extra}"


def holdout_column_name(holdout_label: str) -> str:
    return f"{holdout_label} (this pass)"
