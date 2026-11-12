"""Honest Recent-searches labels — show what the hunt did, not Live reprints."""

from __future__ import annotations

from typing import Any

_ESCAPE_COPY = "Escape deploy — left a worse Live book; did not pass period-robust"
_SILENCE_REASONS = frozenset({"silence_density", "silence_rotate"})
_LIVE_PATCH_REASONS = _SILENCE_REASONS | frozenset({"matrix_escape"})
_PEAK_RESTORE_REASONS = frozenset({"champion_rescue"})


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


def _funnel(h: dict[str, Any]) -> dict[str, Any]:
    fc = h.get("funnel_counts") or h.get("funnel") or {}
    return fc if isinstance(fc, dict) else {}


def _intish(v: Any) -> int | None:
    if v is None:
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def funnel_label(h: dict[str, Any]) -> str:
    """A→B→soft path for this pass — unfinished when soft never ran."""
    fc = _funnel(h)
    a = _intish(fc.get("A_keep"))
    b = _intish(fc.get("B"))
    soft = _intish(h.get("soft_runs"))
    if a is None and b is None and soft is None:
        tested = _intish(h.get("tested"))
        return f"checked {tested}" if tested is not None else "—"
    parts = []
    if a is not None:
        parts.append(f"A{a}")
    if b is not None:
        parts.append(f"B{b}")
    if soft is not None:
        parts.append(f"soft{soft}")
    body = "→".join(parts) if parts else "—"
    if soft is not None and soft == 0 and (a or 0) > 0:
        return f"{body} · unfinished"
    return body


def coverage_label(h: dict[str, Any]) -> str:
    """How wide the candidate pool was this pass."""
    ns = _intish(h.get("n_unique_strat"))
    nt = _intish(h.get("n_unique_tf"))
    nl = _intish(h.get("n_unique_lev"))
    tested = _intish(h.get("tested"))
    bits = []
    if ns is not None and nt is not None and nl is not None and (ns or nt or nl):
        bits.append(f"{ns} strat · {nt} TF · {nl} lev")
    elif tested is not None:
        bits.append(f"{tested} ideas")
    # Keep matrix cells as secondary detail when present.
    matrix = matrix_scored_label(h)
    if matrix and matrix != "—" and "ideas" not in matrix:
        # Strip trailing " · N ideas" if we already show tested above.
        bits.append(matrix.split(" · ")[0] if "cells" in matrix else matrix)
    return " · ".join(bits) if bits else "—"


def nearest_miss_label(h: dict[str, Any]) -> str:
    """Best challenger that did not promote — not the Live reprint."""
    if bool(h.get("improved")):
        return "—"
    miss = h.get("nearest_miss") if isinstance(h.get("nearest_miss"), dict) else {}
    ch = h.get("challenger") if isinstance(h.get("challenger"), dict) else {}
    knobs = h.get("knobs") if isinstance(h.get("knobs"), dict) else {}
    book = miss.get("book") if isinstance(miss.get("book"), dict) else {}
    if not book:
        book = {
            "strategy": ch.get("strategy") or knobs.get("strategy"),
            "timeframe": ch.get("timeframe") or knobs.get("timeframe"),
            "leverage": ch.get("leverage") or knobs.get("leverage"),
        }
    strat = str(book.get("strategy") or "").strip()
    tf = str(book.get("timeframe") or "").strip()
    try:
        lev = int(book.get("leverage") or 0)
    except (TypeError, ValueError):
        lev = 0
    book_s = ""
    if strat and tf and lev:
        book_s = f"{strat} {lev}×@{tf}"
    elif strat:
        book_s = strat
    wr = miss.get("week_ret")
    if wr is None:
        wr = ch.get("week_return")
    week_s = ""
    if wr is not None:
        try:
            week_s = f"week {float(wr):+.0%}"
        except (TypeError, ValueError):
            week_s = ""
    why = str(miss.get("why") or h.get("material_why") or "").strip()
    # Keep why short — strip noisy identical-knobs when book already shown.
    if why.lower() in ("identical knobs", "baseline"):
        why = ""
    soft = _intish(h.get("soft_runs"))
    a = _intish(_funnel(h).get("A_keep"))
    if soft is not None and soft == 0 and (a or 0) > 0 and not why:
        why = "soft never ran"
    bits = [x for x in (book_s, week_s, why) if x]
    return " · ".join(bits)[:72] if bits else "—"


def live_holdout_label(h: dict[str, Any]) -> str:
    """Deployed Live holdout % — labeled so it is not mistaken for a challenger."""
    wr = h.get("week_return")
    if wr is None:
        # Prefer explicit live baseline if present on newer rows.
        base = h.get("baseline") if isinstance(h.get("baseline"), dict) else {}
        wr = base.get("week_return")
    if wr is None:
        return "—"
    try:
        pct = f"{float(wr):+.0%}"
    except (TypeError, ValueError):
        return "—"
    if bool(h.get("improved")):
        return pct
    return f"Live {pct}"


def what_happened_label(h: dict[str, Any]) -> str:
    """One plain line: better found, escape, or what this search pass did."""
    why = _why(h)
    tier = _tier(h)
    improved = bool(h.get("improved"))
    tested = h.get("tested")
    mode = str(h.get("search_mode") or "").strip() or "search"

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

    fc = _funnel(h)
    soft = _intish(h.get("soft_runs"))
    a = _intish(fc.get("A_keep"))
    a_in = _intish(fc.get("A_in"))
    b = _intish(fc.get("B"))
    n_cand = _intish(h.get("n_cand"))
    funnel = funnel_label(h)
    tested_s = f" · tried {tested}" if tested is not None else ""

    # tested=0 with a candidate pool = Live-TF Stage A wipe, not "did nothing"
    if tested is not None and int(tested) == 0:
        pool = n_cand if n_cand is not None else a_in
        if pool and int(pool) > 0:
            return (
                f"{mode} · funnel empty · 0 on Live TF "
                f"(pool {pool}; matrix may still have run)"
            )
        return f"{mode} · funnel empty · checked 0 ideas"
    if soft is not None and soft == 0 and (a or 0) > 0:
        return f"{mode} · unfinished search ({funnel}){tested_s}"
    if soft is not None and soft > 0:
        return f"{mode} · no promote · soft-checked {soft}{tested_s}"
    if b is not None and b == 0 and (a or 0) > 0:
        return f"{mode} · unfinished (B empty){tested_s}"
    if tested is not None:
        return f"{mode} · no promote · checked {tested} ideas"
    return f"{mode} · no promote"


def holdout_week_note(h: dict[str, Any]) -> str:
    """Holdout week % — prefer challenger miss when HOLD; else stored week."""
    if not bool(h.get("improved")):
        miss = h.get("nearest_miss")
        if isinstance(miss, dict) and miss.get("week_ret") is not None:
            try:
                return f"{float(miss['week_ret']):+.0%}"
            except (TypeError, ValueError):
                pass
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
