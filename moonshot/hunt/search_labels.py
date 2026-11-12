"""Honest Recent-searches labels in plain English.

Either a better setup was found and deployed, or it wasn't.
Escape deploys are not champions — say that plainly.
"""

from __future__ import annotations

from typing import Any

_SILENCE_REASONS = frozenset({"silence_density", "silence_rotate"})
_LIVE_PATCH_REASONS = _SILENCE_REASONS | frozenset({"matrix_escape"})
_PEAK_RESTORE_REASONS = frozenset({"champion_rescue"})
_ESCAPE_COPY = "Escape deploy — left a worse Live book; did not pass period-robust"


def gate_fail_label(why: str | None) -> str | None:
    """Plain-English for sealed/DSR/reassess block reasons."""
    w = str(why or "").strip().lower()
    if not w:
        return None
    if w.startswith("sealed"):
        return "Blocked — failed sealed holdout (blind exam)"
    if w.startswith("dsr"):
        return "Blocked — DSR (too many trials for the edge)"
    if "regime_reassess" in w or w == "regime_reassess":
        return "Hunt exhausted — regime reassess needed"
    return None


def _why(h: dict[str, Any]) -> str:
    return str(h.get("promote_reason") or h.get("promote_tier") or "").strip()


def _tier(h: dict[str, Any]) -> str:
    return str(h.get("promote_tier") or "").strip()


def _provenance(h: dict[str, Any]) -> str:
    return str(h.get("promote_provenance") or "").strip().lower()


def _is_live_patch(h: dict[str, Any]) -> bool:
    why, tier = _why(h), _tier(h)
    if _provenance(h) == "escape":
        return True
    return why in _LIVE_PATCH_REASONS or tier in _LIVE_PATCH_REASONS


def what_happened_label(h: dict[str, Any]) -> str:
    """One plain line: better found, escape, or not."""
    why = _why(h)
    tier = _tier(h)
    improved = bool(h.get("improved"))
    tested = h.get("tested")

    if improved:
        if _is_live_patch(h):
            return _ESCAPE_COPY
        if why in _PEAK_RESTORE_REASONS or tier in _PEAK_RESTORE_REASONS:
            if _provenance(h) == "escape":
                return _ESCAPE_COPY
            return "Best known setup put back on paper"
        if _provenance(h) == "robust" or tier == "research":
            return "Better setup found · deployed to paper"
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


def oos_week_note(h: dict[str, Any]) -> str:
    """Long OOS % for this pass when present."""
    tr = h.get("test_return")
    if tr is None:
        return "—"
    try:
        return f"{float(tr):+.0%}"
    except (TypeError, ValueError):
        return "—"


def robust_flag_note(h: dict[str, Any]) -> str:
    """Robust / Escape / — for recent-searches table."""
    prov = _provenance(h)
    if prov == "robust":
        return "Robust"
    if prov == "escape" or _is_live_patch(h) and bool(h.get("improved")):
        return "Escape"
    if bool(h.get("improved")) and _tier(h) == "research":
        return "Robust"
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
