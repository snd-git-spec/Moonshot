"""Hunt pass liveness — finish signal under MemoryMax, not busy/activity."""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timezone
from typing import Any

from moonshot.hunt.constants import (
    HUNT_HEARTBEAT_STALE_MIN,
    HUNT_STALE_FINISH_HOURS,
    HUNT_TERMINAL_GRACE_MIN,
    _update_autopilot_state,
    _utc_now,
)

log = logging.getLogger("moonshot")

TERMINAL_HUNT_KINDS = frozenset({"hold", "upgrade", "targets_met", "improved"})


def _parse_iso(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def hours_since(ts: str | None) -> float | None:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0


def minutes_since(ts: str | None) -> float | None:
    dt = _parse_iso(ts)
    if dt is None:
        return None
    return (datetime.now(timezone.utc) - dt).total_seconds() / 60.0


def _terminal_grace(state: dict) -> bool:
    """Fresh terminal hunt_event only — never improve_start / busy churn."""
    hunt_ev = state.get("hunt_event") or {}
    kind = str(hunt_ev.get("kind") or "")
    if kind not in TERMINAL_HUNT_KINDS:
        return False
    age_h = hours_since(str(hunt_ev.get("ts") or ""))
    if age_h is None:
        return False
    return age_h < (float(HUNT_TERMINAL_GRACE_MIN) / 60.0)


def hunt_pass_health(
    state: dict,
    *,
    process_alive: bool = True,
    mode: str | None = None,
) -> dict[str, Any]:
    """Return ok/failures/warnings for hunt pass finish liveness.

    HUNT_PASS_STUCK = busy with stale pass_heartbeat (not wall-clock pass age).
    HUNT_NO_FINISH = no finished pass for HUNT_STALE_FINISH_HOURS.
    """
    failures: list[str] = []
    warnings: list[str] = []
    mode_s = str(mode if mode is not None else state.get("mode") or "")
    if mode_s != "hunt" or not process_alive:
        return {"ok": True, "failures": failures, "warnings": warnings}

    if _terminal_grace(state):
        return {"ok": True, "failures": failures, "warnings": warnings}

    last_improve_at = str(state.get("last_improve_at") or "")
    improve_age_h = hours_since(last_improve_at)
    if improve_age_h is None:
        improve_age_h = hours_since(str(state.get("pass_started_at") or ""))
        if improve_age_h is None:
            improve_age_h = hours_since(str(state.get("started_at") or ""))

    hb_ts = str(state.get("pass_heartbeat_at") or "") or str(state.get("pass_started_at") or "")
    hb_age_min = minutes_since(hb_ts)

    if (
        bool(state.get("busy"))
        and hb_age_min is not None
        and hb_age_min >= float(HUNT_HEARTBEAT_STALE_MIN)
    ):
        failures.append(
            f"HUNT_PASS_STUCK: busy=True but pass_heartbeat_at {hb_age_min:.0f}m ago "
            f"(>{float(HUNT_HEARTBEAT_STALE_MIN):.0f}m) — pass not progressing"
        )

    if improve_age_h is not None and improve_age_h >= float(HUNT_STALE_FINISH_HOURS):
        failures.append(
            f"HUNT_NO_FINISH: last_improve_at {improve_age_h:.1f}h ago "
            f"(>{float(HUNT_STALE_FINISH_HOURS):.0f}h) while mode=hunt — "
            "activity without a finished pass"
        )

    return {"ok": not failures, "failures": failures, "warnings": warnings}


def recent_oom_hint(since_minutes: float = 30.0) -> bool:
    """Best-effort journal skim for MemoryMax / OOM on moonshot (non-fatal if unavailable)."""
    try:
        out = subprocess.check_output(
            [
                "journalctl",
                "-u",
                "moonshot",
                "--since",
                f"{int(since_minutes)} min ago",
                "-q",
                "--no-pager",
            ],
            text=True,
            timeout=8,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return False
    low = (out or "").lower()
    return any(
        tok in low
        for tok in (
            "oom",
            "memorymax",
            "killed process",
            "out of memory",
            "cgroup",
        )
    )


def _arm_skip_denser(state: dict, reason: str) -> None:
    """One-shot denser skip — never TF-slim the matrix."""
    state["force_skip_denser"] = True
    state.pop("force_matrix_slim", None)  # legacy TF-slim key — never keep
    state["hunt_recover_reason"] = str(reason or "pass_stuck")


def record_pass_abort(
    *,
    generation: int,
    error: BaseException | str,
    recover: bool = True,
) -> dict:
    """Write a finish record after improve_once raises — never leave last_improve frozen."""
    ts = _utc_now().isoformat()
    err_s = str(error)
    decision = {
        "ts": ts,
        "improved": False,
        "message": f"PASS ABORT · gen={generation} · {err_s}",
        "best": {},
        "grade": {},
        "promote_reason": "abort",
        "promote_tier": "hold",
        "abort": True,
        "abort_error": err_s[:500],
    }

    def _mutate(state: dict) -> None:
        state["last_improve"] = decision
        state["last_improve_at"] = ts
        state["generation"] = int(generation)
        state["improve_passes"] = int(state.get("improve_passes", 0)) + 1
        state["busy"] = False
        state.pop("pass_started_at", None)
        state.pop("pass_heartbeat_at", None)
        if recover:
            _arm_skip_denser(state, "pass_stuck")

    _update_autopilot_state(_mutate)
    log.error("PASS ABORT recorded · gen=%s · %s", generation, err_s)
    return decision
