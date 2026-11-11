"""Honest Recent-searches / Live labels — peak champion ≠ Live/paper.

Pure helpers (no Streamlit) so tests can lock the wording.
"""

from __future__ import annotations

from typing import Any

# Promote paths that raise or restore the raise-only peak champion.
_PEAK_RAISE_REASONS = frozenset({"research", "confirmed", "baseline"})  # baseline unused
_PEAK_RESTORE_REASONS = frozenset({"champion_rescue"})
_SILENCE_REASONS = frozenset({"silence_density", "silence_rotate"})
# Live/paper patches that must NEVER say "new champion".
_LIVE_PATCH_REASONS = _SILENCE_REASONS | frozenset({"matrix_escape"})


def _why(h: dict[str, Any]) -> str:
    return str(h.get("promote_reason") or h.get("promote_tier") or "").strip()


def _tier(h: dict[str, Any]) -> str:
    return str(h.get("promote_tier") or "").strip()


def what_happened_label(h: dict[str, Any]) -> str:
    """One honest line for the Recent searches 'What happened' column."""
    why = _why(h)
    tier = _tier(h)
    improved = bool(h.get("improved"))
    tested = h.get("tested")
    gate = str(h.get("material_why") or "").strip()

    if improved:
        # Silence / Live patch — peak raise-only, never call this a new champion.
        if why in _SILENCE_REASONS or tier in _SILENCE_REASONS:
            return f"Live/paper rotated ({why or tier}) · peak champion unchanged"
        if why == "matrix_escape" or tier == "matrix_escape":
            return f"Live/paper rotated (matrix_escape) · peak champion unchanged"
        if why in _PEAK_RESTORE_REASONS or tier in _PEAK_RESTORE_REASONS:
            return "Peak champion restored to paper"
        if why:
            return f"New peak champion deployed to paper ({why})"
        return "New peak champion deployed to paper"

    # HOLD / no .env apply
    if tested is not None:
        if gate and why not in ("", "baseline", "hold"):
            return f"No deploy · peak unchanged · checked {tested} ideas ({gate})"
        return f"No deploy · peak unchanged · checked {tested} ideas"
    return "No deploy · peak unchanged"


def holdout_week_note(h: dict[str, Any]) -> str:
    """Week % for this pass's graded book — never labeled as peak champion."""
    wr = h.get("week_return")
    if wr is None:
        return "—"
    try:
        pct = f"{float(wr):+.0%}"
    except (TypeError, ValueError):
        return "—"
    if bool(h.get("improved")):
        why = _why(h)
        if why in _LIVE_PATCH_REASONS or _tier(h) in _SILENCE_REASONS:
            return f"{pct} (Live)"
        return f"{pct} (deployed)"
    return f"{pct} (no deploy)"


def matrix_scored_label(h: dict[str, Any]) -> str:
    """Full TF×lev matrix breadth — never the by-leverage 5m-only collapse."""
    # Prefer full matrix rows when present (honest multi-TF).
    rows = h.get("tf_lev_matrix")
    if not isinstance(rows, list) or not rows:
        # Fallback: matrix_tfs + tested, or legacy leverage_ladder.
        tfs = h.get("matrix_tfs")
        tested_n = h.get("tested")
        if isinstance(tfs, (list, tuple)) and tfs:
            tf_s = "/".join(str(t) for t in tfs)
            extra = f" · {tested_n} knob ideas" if tested_n is not None else ""
            return f"FULL × {tf_s}{extra}"
        ladder = h.get("leverage_ladder") or []
        if isinstance(ladder, list) and ladder:
            return _label_from_cells(ladder, tested_n=tested_n, legacy=True)
        return f"{tested_n} knob ideas" if tested_n is not None else "—"
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
        return f"{tested_n} knob ideas" if tested_n is not None else "—"
    lev_s = "/".join(f"{x}×" for x in sorted(levs))
    tf_order = {"5m": 0, "15m": 1, "1h": 2, "4h": 3}
    tf_s = "/".join(sorted(tfs, key=lambda t: tf_order.get(t, 9)))
    extra = f" · {tested_n} knob ideas" if tested_n is not None else ""
    prefix = f"{len(cells)} cells"
    if legacy and tfs == {"5m"}:
        # Mark collapsed ladder so it is not read as "only searched 5m".
        return f"{prefix} · {lev_s} × {tf_s} (ladder alias){extra}"
    return f"{prefix} · {lev_s} × {tf_s}{extra}"


def holdout_column_name(holdout_label: str) -> str:
    """Column header — this pass / Live week, never peak champion."""
    return f"{holdout_label} (this pass)"
