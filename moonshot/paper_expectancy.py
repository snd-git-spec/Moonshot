"""Paper expectancy buckets by config_hash (A.5).

Units (LOCKED):
- r_multiple is already net — never invent gross_R from R+fees
- cost_share uses dollar gross_dollar_pnl; 40% threshold is on cost_share
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

COST_SHARE_STOP_THRESHOLD = 0.40
KEEP_MIN_N = 100


@dataclass
class BucketStats:
    key: str
    config_hash: str | None
    symbol: str | None
    timeframe: str | None
    n: int
    win_rate: float
    mean_r: float
    ci_low: float
    ci_high: float
    cost_share: float
    max_dd: float
    liquidations: int
    legacy: bool

    def decision_label(self) -> str:
        if self.legacy:
            return "LEGACY-EXCLUDED"
        if self.n < KEEP_MIN_N:
            return "EXTEND-or-WAIT"
        if self.mean_r > 0 and self.ci_low > 0:
            return "KEEP-VALIDATED"
        if self.cost_share > COST_SHARE_STOP_THRESHOLD:
            return "STOP-execution"
        return "STOP-no-edge"


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(arr.mean())
    if arr.size < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(arr, size=(n_boot, arr.size), replace=True).mean(axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cost_share_dollars(
    r_multiples: Sequence[float],
    initial_risks: Sequence[float],
    fees: Sequence[float],
    funding: Sequence[float],
) -> float:
    """Dollar cost_share = costs / gross_dollar_pnl (never mix into R)."""
    costs = float(sum(fees) + sum(funding))
    gross = (
        float(sum(r * risk for r, risk in zip(r_multiples, initial_risks, strict=False))) + costs
    )
    if abs(gross) < 1e-12:
        return float("nan")
    return costs / gross


def _tf_from_config_json(raw: str | None) -> str | None:
    if not raw:
        return None
    try:
        obj = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None
    tf = obj.get("timeframe") or obj.get("TIMEFRAME")
    return str(tf) if tf is not None else None


def _stats_for_items(
    key: str,
    items: list[dict[str, Any]],
    *,
    config_hash: str | None,
    symbol: str | None,
    timeframe: str | None,
    legacy: bool,
) -> BucketStats:
    rs = [float(x.get("r_multiple") or 0.0) for x in items]
    risks = [float(x.get("initial_risk_dollars") or 0.0) for x in items]
    fees = [float(x.get("fees") or 0.0) for x in items]
    funding = [float(x.get("funding") or 0.0) for x in items]
    mean_r, lo, hi = bootstrap_mean_ci(rs)
    wins = sum(1 for r in rs if r > 0)
    pnls = [float(x.get("pnl") or 0.0) for x in items]
    eq = 0.0
    peak = 0.0
    max_dd = 0.0
    for p in pnls:
        eq += p
        peak = max(peak, eq)
        if peak > 0:
            max_dd = max(max_dd, (peak - eq) / peak)
    return BucketStats(
        key=key,
        config_hash=config_hash,
        symbol=symbol,
        timeframe=timeframe,
        n=len(items),
        win_rate=wins / len(items) if items else 0.0,
        mean_r=mean_r,
        ci_low=lo,
        ci_high=hi,
        cost_share=cost_share_dollars(rs, risks, fees, funding),
        max_dd=max_dd,
        liquidations=sum(int(x.get("liquidated") or 0) for x in items),
        legacy=legacy,
    )


def bucket_trades_by_hash(rows: Iterable[dict[str, Any]]) -> list[BucketStats]:
    """Phase 0 KEEP/STOP scoreboard: one bucket per config_hash (all symbols)."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        ch = row.get("config_hash")
        key = "LEGACY" if ch is None or ch == "" else str(ch)
        groups.setdefault(key, []).append(row)
    out: list[BucketStats] = []
    for key, items in sorted(groups.items(), key=lambda kv: (-len(kv[1]), str(kv[0]))):
        legacy = key == "LEGACY"
        out.append(
            _stats_for_items(
                key=key,
                items=items,
                config_hash=None if legacy else key,
                symbol=None,
                timeframe=None,
                legacy=legacy,
            )
        )
    return out


def bucket_trades(rows: Iterable[dict[str, Any]]) -> list[BucketStats]:
    groups: dict[tuple, list[dict[str, Any]]] = {}
    for row in rows:
        ch = row.get("config_hash")
        legacy = ch is None or ch == ""
        sym = row.get("symbol")
        tf = _tf_from_config_json(row.get("config_json"))
        key = ("LEGACY", None, None) if legacy else (ch, sym, tf)
        groups.setdefault(key, []).append(row)

    out: list[BucketStats] = []
    for key, items in sorted(groups.items(), key=lambda kv: str(kv[0])):
        legacy = key[0] == "LEGACY"
        ch = None if legacy else str(key[0])
        out.append(
            _stats_for_items(
                key=str(key),
                items=items,
                config_hash=ch,
                symbol=None if legacy else key[1],
                timeframe=None if legacy else key[2],
                legacy=legacy,
            )
        )
    return out


def load_trade_rows(db_path: Path | str) -> list[dict[str, Any]]:
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    try:
        cols = {r[1] for r in con.execute("PRAGMA table_info(trades)").fetchall()}
        want = [
            "ts",
            "symbol",
            "pnl",
            "r_multiple",
            "fees",
            "funding",
            "liquidated",
            "config_hash",
            "config_json",
            "initial_risk_dollars",
        ]
        select = ", ".join(c for c in want if c in cols)
        rows = con.execute(f"SELECT {select} FROM trades ORDER BY ts").fetchall()
        return [dict(r) for r in rows]
    finally:
        con.close()


def resolve_pinned_config_hash() -> str | None:
    """Current Live/paper book hash from env (Phase 0 KEEP/STOP identity)."""
    try:
        from moonshot.config import Config
        from moonshot.config_hash import config_hash as _ch

        return str(_ch(Config.from_env()))
    except Exception:
        return None


def filter_rows_to_pinned(
    rows: Sequence[dict[str, Any]],
    pinned_hash: str | None,
) -> tuple[list[dict[str, Any]], int]:
    """Keep only pinned config_hash rows. Returns (kept, n_excluded)."""
    if not pinned_hash:
        return list(rows), 0
    kept = [r for r in rows if str(r.get("config_hash") or "") == str(pinned_hash)]
    return kept, max(0, len(rows) - len(kept))


def locked_criteria_text() -> str:
    return (
        "LOCKED (pinned hash only): KEEP if n>=100 and bootstrap 95% CI on mean(r_multiple) "
        "excludes 0 and positive; STOP if n>=100 and (mean<=0 or CI includes 0); "
        "if cost_share>40% → STOP-execution (Phase E); else STOP-no-edge; "
        "n<100 → EXTEND until 8w cap (then +4w EXTEND). Other hashes / old 5m excluded. "
        "Units: net r_multiple; dollar cost_share."
    )


def _render_bucket_table(buckets: Sequence[BucketStats], *, title: str) -> list[str]:
    lines = [
        f"## {title}",
        "",
        "| bucket | n | win% | mean_R | CI95 | cost_share | maxDD | liq | decision |",
        "|---|---:|---:|---:|---|---:|---:|---:|---|",
    ]
    for b in buckets:
        ci = (
            f"[{b.ci_low:.3f},{b.ci_high:.3f}]"
            if math.isfinite(b.ci_low) and math.isfinite(b.ci_high)
            else "n/a"
        )
        label = b.config_hash or "LEGACY"
        if b.symbol:
            label = f"{label} {b.symbol}"
        if b.timeframe:
            label = f"{label}@{b.timeframe}"
        cs = f"{b.cost_share:.1%}" if math.isfinite(b.cost_share) else "n/a"
        lines.append(
            f"| {label} | {b.n} | {b.win_rate:.0%} | {b.mean_r:.3f} | {ci} | "
            f"{cs} | {b.max_dd:.1%} | {b.liquidations} | {b.decision_label()} |"
        )
    lines.append("")
    return lines


def render_report(
    buckets: Sequence[BucketStats],
    *,
    by_hash: Sequence[BucketStats] | None = None,
    pinned_hash: str | None = None,
    n_excluded: int = 0,
) -> str:
    lines = [
        f"# Paper expectancy {datetime.now(timezone.utc).isoformat()}",
        "",
        locked_criteria_text(),
        "",
    ]
    if pinned_hash:
        lines.append(f"Pinned config_hash: `{pinned_hash}`")
        lines.append(f"Excluded non-pinned closes (old books / other TF): **{int(n_excluded)}**")
        lines.append("")
    if by_hash is not None:
        title = (
            "Pinned config_hash KEEP/STOP scoreboard"
            if pinned_hash
            else "By config_hash (KEEP/STOP scoreboard)"
        )
        if pinned_hash and not by_hash:
            # Explicit empty window — do not imply pollution from other hashes.
            by_hash = [
                BucketStats(
                    key=str(pinned_hash),
                    config_hash=str(pinned_hash),
                    symbol=None,
                    timeframe=None,
                    n=0,
                    win_rate=0.0,
                    mean_r=float("nan"),
                    ci_low=float("nan"),
                    ci_high=float("nan"),
                    cost_share=float("nan"),
                    max_dd=0.0,
                    liquidations=0,
                    legacy=False,
                )
            ]
        lines.extend(_render_bucket_table(by_hash, title=title))
    detail_title = (
        "Pinned hash × symbol × TF (detail)"
        if pinned_hash
        else "By config_hash × symbol × TF (detail)"
    )
    lines.extend(_render_bucket_table(buckets, title=detail_title))
    return "\n".join(lines)


def write_expectancy_report(
    db_path: Path | str,
    out_dir: Path | str = "runs",
    *,
    pinned_hash: str | None = None,
    pinned_only: bool = True,
) -> Path:
    """Write expectancy MD. Default: Phase 0 pinned-hash-only (excludes old 5m)."""
    rows = load_trade_rows(db_path)
    pin = pinned_hash if pinned_hash is not None else resolve_pinned_config_hash()
    n_excluded = 0
    if pinned_only and pin:
        rows, n_excluded = filter_rows_to_pinned(rows, pin)
    by_hash = bucket_trades_by_hash(rows)
    buckets = bucket_trades(rows)
    text = render_report(
        buckets,
        by_hash=by_hash,
        pinned_hash=pin if pinned_only else None,
        n_excluded=n_excluded,
    )
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    path = root / f"paper_expectancy_{stamp}.md"
    path.write_text(text)
    return path
