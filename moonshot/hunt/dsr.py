"""Deflated Sharpe Ratio (Bailey & López de Prado) + cumulative trial counter (A.3b).

# TODO: PBO via CSCV
Promote gate: DSR > DSR_THRESHOLD (start at 0).
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path
from typing import Any

DSR_THRESHOLD = 0.0
DEFAULT_TRIAL_DB = Path("runs/trial_counter.db")


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Acklam rational approximation of the standard normal quantile."""
    if p <= 0.0:
        return float("-inf")
    if p >= 1.0:
        return float("inf")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    plow = 0.02425
    phigh = 1.0 - plow
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    )


def expected_max_sharpe(n_trials: int, *, skew: float = 0.0, kurt: float = 3.0) -> float:
    """E[max SR] under null for n independent trials (approx)."""
    del skew, kurt  # reserved for higher-moment extension
    n = max(1, int(n_trials))
    gamma = 0.5772156649
    if n == 1:
        return 0.0
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return float((1.0 - gamma) * z1 + gamma * z2)


def deflated_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    *,
    skew: float = 0.0,
    kurt: float = 3.0,
) -> float:
    """
    DSR ≈ Φ( (SR* - SR0) * sqrt(T-1) / sqrt(1 - skew*SR* + (kurt-1)/4 * SR*^2) )
    where SR0 = E[max SR | n_trials].
    """
    if n_obs < 2 or not math.isfinite(observed_sr):
        return 0.0
    sr0 = expected_max_sharpe(n_trials, skew=skew, kurt=kurt)
    sr = float(observed_sr)
    denom = math.sqrt(max(1e-12, 1.0 - skew * sr + ((kurt - 1.0) / 4.0) * sr * sr))
    z = ((sr - sr0) * math.sqrt(n_obs - 1)) / denom
    return float(_norm_cdf(z))


def dsr_passes(dsr: float, threshold: float = DSR_THRESHOLD) -> bool:
    """Promote only if DSR > threshold. TODO: PBO via CSCV."""
    return bool(math.isfinite(dsr) and dsr > float(threshold))


def _connect(db_path: Path | str) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS trial_counter (
            config_hash TEXT PRIMARY KEY,
            first_seen_ts TEXT NOT NULL,
            run_ids_json TEXT NOT NULL
        )
        """
    )
    con.commit()
    return con


def record_trial(
    config_hash: str,
    run_id: str,
    *,
    first_seen_ts: str,
    db_path: Path | str = DEFAULT_TRIAL_DB,
) -> int:
    """Upsert trial; return cumulative distinct config_hash count."""
    con = _connect(db_path)
    try:
        row = con.execute(
            "SELECT run_ids_json FROM trial_counter WHERE config_hash = ?",
            (config_hash,),
        ).fetchone()
        if row is None:
            con.execute(
                "INSERT INTO trial_counter (config_hash, first_seen_ts, run_ids_json) VALUES (?,?,?)",
                (config_hash, first_seen_ts, json.dumps([run_id])),
            )
        else:
            ids = json.loads(row[0] or "[]")
            if run_id not in ids:
                ids.append(run_id)
            con.execute(
                "UPDATE trial_counter SET run_ids_json = ? WHERE config_hash = ?",
                (json.dumps(ids), config_hash),
            )
        con.commit()
        return int(con.execute("SELECT COUNT(*) FROM trial_counter").fetchone()[0])
    finally:
        con.close()


def trial_count(*, db_path: Path | str = DEFAULT_TRIAL_DB) -> int:
    con = _connect(db_path)
    try:
        return int(con.execute("SELECT COUNT(*) FROM trial_counter").fetchone()[0])
    finally:
        con.close()


def write_dsr_artifact(
    run_dir: Path | str,
    *,
    observed_sr: float,
    n_obs: int,
    n_trials: int,
    dsr: float | None = None,
) -> Path:
    root = Path(run_dir)
    root.mkdir(parents=True, exist_ok=True)
    if dsr is None:
        dsr = deflated_sharpe_ratio(observed_sr, n_obs, n_trials)
    payload: dict[str, Any] = {
        "observed_sr": observed_sr,
        "n_obs": n_obs,
        "n_trials": n_trials,
        "dsr": dsr,
        "threshold": DSR_THRESHOLD,
        "passes": dsr_passes(dsr),
        "todo": "PBO via CSCV",
    }
    path = root / "dsr.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
