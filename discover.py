"""
Rule discovery: let the machine search combinations of candle patterns, RSI,
moving averages, volume and trend strength — then judge them without fooling ourselves.

Protocol (the part that matters):
  1. Every rule is a direction + 2-3 conditions from different families.
  2. Outcomes use one fixed exit (stop/target/time) with real fees + slippage.
  3. Time is split: TRAIN (find) -> VALIDATE (confirm) -> HOLDOUT (touched once).
  4. NULL CONTROL: the identical search on time-shifted (meaningless) outcomes shows how many
     rules "pass" by pure chance. Real rules must clearly beat that count.
  5. Survivors are re-tested on ETH (a different coin) before we believe them.

Usage:  python discover.py [--tf 4h] [--bars 16000] [--shifts 10]
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import moonshot as ms

ROOT = Path(__file__).resolve().parent
OUT = ROOT / ".moonshot_rules.json"

# ---- fixed exit scheme (triple barrier), in ATR units ----
STOP_ATR = 2.0
TARGET_R = 2.0           # target = TARGET_R x stop distance
HOLD_BARS = {"4h": 18, "1h": 36, "15m": 96}

# ---- search protocol ----
TRAIN_FRAC, VAL_FRAC = 0.50, 0.25          # remainder = holdout
MIN_TRAIN_N, MIN_VAL_N, MIN_HOLD_N = 40, 20, 15
MIN_TRAIN_R, MIN_VAL_R = 0.10, 0.05
MIN_TRAIN_T = 1.5                          # mean/(std/sqrt(n))
WARMUP = 260                               # EMA200 + ATR warm-up bars


# ------------------------------------------------------------------
# Condition library
# ------------------------------------------------------------------
def build_atoms(df: pd.DataFrame) -> dict:
    """name -> (family, bool ndarray). Every atom uses only info at bar close."""
    o, h, l, c, v = (df[k].astype(float) for k in ("open", "high", "low", "close", "volume"))
    a = ms.atr(df, 14)
    r = ms.rsi(c, 14)
    ax = ms.adx(df, 14)
    e20, e50, e200 = ms.ema(c, 20), ms.ema(c, 50), ms.ema(c, 200)
    vr = v / v.rolling(20).mean().shift(1)
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    po, pc = o.shift(1), c.shift(1)

    A: dict = {}

    def add(name, fam, s):
        A[name] = (fam, s.fillna(False).to_numpy(dtype=bool))

    # candle patterns
    add("bull_engulf", "candle", (c > o) & (pc < po) & (c >= po) & (o <= pc))
    add("bear_engulf", "candle", (c < o) & (pc > po) & (c <= po) & (o >= pc))
    add("hammer", "candle", (lower >= 2 * body) & (upper <= body) & (lower / rng > 0.55))
    add("shoot_star", "candle", (upper >= 2 * body) & (lower <= body) & (upper / rng > 0.55))
    add("big_bull", "candle", (c > o) & (body >= 1.0 * a))
    add("big_bear", "candle", (c < o) & (body >= 1.0 * a))
    add("green_3", "candle", (c > o) & (c.shift(1) > o.shift(1)) & (c.shift(2) > o.shift(2)))
    add("red_3", "candle", (c < o) & (c.shift(1) < o.shift(1)) & (c.shift(2) < o.shift(2)))
    # RSI
    add("rsi<30", "rsi", r < 30)
    add("rsi<40", "rsi", r < 40)
    add("rsi>60", "rsi", r > 60)
    add("rsi>70", "rsi", r > 70)
    add("rsi_xup50", "rsi", (r.shift(1) < 50) & (r >= 50))
    add("rsi_xdn50", "rsi", (r.shift(1) > 50) & (r <= 50))
    # moving averages
    for n, e in (("20", e20), ("50", e50), ("200", e200)):
        add(f"c>ema{n}", "ma", c > e)
        add(f"c<ema{n}", "ma", c < e)
    add("ema20>50", "ma", e20 > e50)
    add("ema20<50", "ma", e20 < e50)
    add("ema50>200", "ma", e50 > e200)
    add("ema50<200", "ma", e50 < e200)
    add("pullback_up20", "ma", (l <= e20) & (c > e20))
    add("pullback_dn20", "ma", (h >= e20) & (c < e20))
    add("pullback_up50", "ma", (l <= e50) & (c > e50))
    add("pullback_dn50", "ma", (h >= e50) & (c < e50))
    add("xup_ema20", "ma", (pc <= e20.shift(1)) & (c > e20))
    add("xdn_ema20", "ma", (pc >= e20.shift(1)) & (c < e20))
    # volume
    add("vol>1.5x", "vol", vr >= 1.5)
    add("vol>2x", "vol", vr >= 2.0)
    add("vol<0.8x", "vol", vr < 0.8)
    # trend strength
    add("adx>20", "trend", ax > 20)
    add("adx>25", "trend", ax > 25)
    add("adx<20", "trend", ax < 20)
    # ---- moon phase (synodic month; reference new moon 2000-01-06 18:14 UTC) ----
    ts = df.index
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    days = (ts - pd.Timestamp("2000-01-06 18:14", tz="UTC")).total_seconds() / 86400.0
    ph = pd.Series((np.asarray(days) / 29.530588853) % 1.0, index=df.index)
    add("new_moon", "moon", (ph < 0.05) | (ph > 0.95))      # about +-1.5 days
    add("full_moon", "moon", (ph > 0.45) & (ph < 0.55))
    add("waxing", "moon", ph < 0.5)
    add("waning", "moon", ph >= 0.5)

    # ---- Fibonacci retracements of the last 50-bar swing ----
    W = 50
    hh = h.rolling(W).max()
    ll = l.rolling(W).min()
    ih = h.rolling(W).apply(np.argmax, raw=True)
    il = l.rolling(W).apply(np.argmin, raw=True)
    span = (hh - ll)
    big = span >= 3 * a                                      # ignore tiny swings
    upswing = (ih > il) & big                                # low came first, then high
    dnswing = (il > ih) & big
    ret_up = (hh - c) / span.replace(0, np.nan)              # pullback depth after an up-swing
    ret_dn = (c - ll) / span.replace(0, np.nan)              # bounce depth after a down-swing
    for tag, lo, hi in (("382", 0.33, 0.43), ("50", 0.45, 0.55), ("618", 0.58, 0.68)):
        add(f"fib{tag}_after_up", "fib", upswing & (ret_up >= lo) & (ret_up <= hi))
        add(f"fib{tag}_after_dn", "fib", dnswing & (ret_dn >= lo) & (ret_dn <= hi))
    # ---- Ichimoku (9/26/52). Cloud at bar t = values computed 26 bars ago (no lookahead) ----
    ten = (h.rolling(9).max() + l.rolling(9).min()) / 2
    kij = (h.rolling(26).max() + l.rolling(26).min()) / 2
    sa = ((ten + kij) / 2).shift(26)
    sb = ((h.rolling(52).max() + l.rolling(52).min()) / 2).shift(26)
    ctop, cbot = np.maximum(sa, sb), np.minimum(sa, sb)
    ctop, cbot = pd.Series(ctop, index=df.index), pd.Series(cbot, index=df.index)
    add("ichi_above_cloud", "ichi_cloud", c > ctop)
    add("ichi_below_cloud", "ichi_cloud", c < cbot)
    add("ichi_in_cloud", "ichi_cloud", (c <= ctop) & (c >= cbot))
    add("ichi_cloud_green", "ichi_cloud", sa > sb)
    add("ichi_cloud_red", "ichi_cloud", sa < sb)
    add("ichi_break_up", "ichi_cloud", (c > ctop) & (pc <= ctop.shift(1)))
    add("ichi_break_dn", "ichi_cloud", (c < cbot) & (pc >= cbot.shift(1)))
    add("ichi_tk_up", "ichi_tk", (ten > kij) & (ten.shift(1) <= kij.shift(1)))
    add("ichi_tk_dn", "ichi_tk", (ten < kij) & (ten.shift(1) >= kij.shift(1)))
    add("ichi_ten>kij", "ichi_tk", ten > kij)
    add("ichi_ten<kij", "ichi_tk", ten < kij)
    add("ichi_c>kij", "ichi_tk", c > kij)
    add("ichi_c<kij", "ichi_tk", c < kij)
    add("ichi_chikou_bull", "ichi_chikou", c > c.shift(26))
    add("ichi_chikou_bear", "ichi_chikou", c < c.shift(26))
    return A


def enumerate_rules(atoms: dict) -> list:
    """All (direction, atom-name tuples) combos: 2-3 atoms, each from a different family."""
    by_fam: dict = {}
    for name, (fam, _) in atoms.items():
        by_fam.setdefault(fam, []).append(name)
    fams = sorted(by_fam)
    rules = []
    for k in (2, 3):
        for fam_combo in itertools.combinations(fams, k):
            for names in itertools.product(*(by_fam[f] for f in fam_combo)):
                for d in ("long", "short"):
                    rules.append((d, names))
    return rules


# ------------------------------------------------------------------
# Outcomes: triple-barrier R per bar, long and short (net of costs)
# ------------------------------------------------------------------
def barrier_outcomes_reference(df: pd.DataFrame, hold: int, cfg: ms.Config) -> tuple:
    """Original loop (kept for A.8 parity tests)."""
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = ms.atr(df, 14).to_numpy(float)
    n = len(df)
    rl = np.full(n, np.nan)
    rs = np.full(n, np.nan)
    cost = 2 * cfg.fee_taker + 2 * cfg.slip
    for t in range(n - hold - 2):
        if not np.isfinite(a[t]) or a[t] <= 0:
            continue
        e = o[t + 1]
        d = STOP_ATR * a[t]
        cost_r = cost * e / d
        # long
        stop, tgt = e - d, e + TARGET_R * d
        res = None
        for j in range(t + 1, t + 1 + hold):
            if l[j] <= stop:
                res = -1.0
                break
            if h[j] >= tgt:
                res = TARGET_R
                break
        if res is None:
            res = (c[t + hold] - e) / d
        rl[t] = res - cost_r
        # short
        stop, tgt = e + d, e - TARGET_R * d
        res = None
        for j in range(t + 1, t + 1 + hold):
            if h[j] >= stop:
                res = -1.0
                break
            if l[j] <= tgt:
                res = TARGET_R
                break
        if res is None:
            res = (e - c[t + hold]) / d
        rs[t] = res - cost_r
    return rl, rs


def barrier_outcomes(df: pd.DataFrame, hold: int, cfg: ms.Config) -> tuple:
    from moonshot.discover_fast import barrier_outcomes_vectorized

    atr = ms.atr(df, 14).to_numpy(float)
    return barrier_outcomes_vectorized(
        df,
        hold,
        fee_taker=float(cfg.fee_taker),
        slip=float(cfg.slip),
        atr=atr,
    )


# ------------------------------------------------------------------
# Rule evaluation
# ------------------------------------------------------------------
def _thin(idx: np.ndarray, gap: int) -> np.ndarray:
    """Greedy: keep a signal only if >= gap bars after the last kept one (no overlapping trades)."""
    if idx.size == 0:
        return idx
    keep = [idx[0]]
    for i in idx[1:]:
        if i - keep[-1] >= gap:
            keep.append(i)
    return np.asarray(keep)


def stats(R: np.ndarray) -> dict:
    n = int(R.size)
    if n == 0:
        return {"n": 0, "mean": 0.0, "win": 0.0, "t": 0.0}
    m = float(R.mean())
    sd = float(R.std(ddof=1)) if n > 1 else 0.0
    return {
        "n": n, "mean": m, "win": float((R > 0).mean()),
        "t": (m / (sd / np.sqrt(n))) if sd > 0 else 0.0,
    }


@dataclass
class Data:
    atoms: dict
    rl: np.ndarray
    rs: np.ndarray
    idx: pd.DatetimeIndex
    bounds: tuple      # (i_train_end, i_val_end)
    hold: int


def rule_mask(atoms: dict, names: tuple) -> np.ndarray:
    from moonshot.discover_fast import signal_mask

    return signal_mask(atoms, tuple(names), cache=True)


def eval_rule(D: Data, rule: tuple, shift: int = 0) -> dict:
    d, names = rule
    R_all = D.rl if d == "long" else D.rs
    if shift:
        R_all = np.roll(R_all, shift)          # break signal/outcome alignment (null)
    sig = np.flatnonzero(rule_mask(D.atoms, names))
    sig = sig[(sig >= WARMUP) & np.isfinite(R_all[sig])]
    tr_end, va_end = D.bounds
    out = {}
    for label, lo, hi in (("train", WARMUP, tr_end), ("val", tr_end, va_end),
                          ("hold", va_end, len(R_all))):
        s = sig[(sig >= lo) & (sig < hi)]
        out[label] = stats(R_all[_thin(s, D.hold)])
    return out


def passes(ev: dict) -> bool:
    tr, va = ev["train"], ev["val"]
    return (
        tr["n"] >= MIN_TRAIN_N and tr["mean"] >= MIN_TRAIN_R and tr["t"] >= MIN_TRAIN_T
        and va["n"] >= MIN_VAL_N and va["mean"] >= MIN_VAL_R
    )


def prepare(sym: str, tf: str, bars: int, cfg: ms.Config) -> tuple:
    raw = ms.fetch_historical(sym, tf, bars, refresh=False)
    atoms = build_atoms(raw)
    rl, rs = barrier_outcomes(raw, HOLD_BARS.get(tf, 18), cfg)
    return raw, atoms, rl, rs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="4h")
    ap.add_argument("--bars", type=int, default=16000)
    ap.add_argument("--shifts", type=int, default=10, help="null-control reruns")
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--symbol", default="BTC/USDT:USDT")
    ap.add_argument("--cross", default="ETH/USDT:USDT")
    args = ap.parse_args()

    cfg = ms.Config.from_env()
    t0 = time.time()
    print(f"loading {args.symbol} {args.tf} ...", flush=True)
    raw, atoms, rl, rs = prepare(args.symbol, args.tf, args.bars, cfg)
    n = len(raw)
    tr_end = WARMUP + int((n - WARMUP) * TRAIN_FRAC)
    va_end = WARMUP + int((n - WARMUP) * (TRAIN_FRAC + VAL_FRAC))
    D = Data(atoms, rl, rs, raw.index, (tr_end, va_end), HOLD_BARS.get(args.tf, 18))
    print(
        f"bars={n}  TRAIN {raw.index[WARMUP]:%Y-%m-%d}→{raw.index[tr_end]:%Y-%m-%d} | "
        f"VALIDATE →{raw.index[va_end]:%Y-%m-%d} | HOLDOUT →{raw.index[-1]:%Y-%m-%d}",
        flush=True,
    )
    base_long = stats(rl[WARMUP:][np.isfinite(rl[WARMUP:])])
    base_short = stats(rs[WARMUP:][np.isfinite(rs[WARMUP:])])
    print(f"baseline (enter on EVERY bar): long {base_long['mean']:+.3f}R  short {base_short['mean']:+.3f}R "
          f"(this is what luck + costs give you)", flush=True)

    rules = enumerate_rules(atoms)
    print(f"searching {len(rules)} rules ...", flush=True)

    found = []
    for rule in rules:
        ev = eval_rule(D, rule)
        if passes(ev):
            found.append((rule, ev))
    print(f"REAL data: {len(found)} rules pass train+validate  ({time.time()-t0:.0f}s)", flush=True)

    # ---- null control: same search, outcomes shifted in time (no real relationship) ----
    rng = np.random.default_rng(7)
    null_counts = []
    for k in range(args.shifts):
        sh = int(rng.integers(600, max(700, n - 600)))
        cnt = sum(1 for rule in rules if passes(eval_rule(D, rule, shift=sh)))
        null_counts.append(cnt)
        print(f"  null shift {k+1}/{args.shifts}: {cnt} pass", flush=True)
    null_mean = float(np.mean(null_counts)) if null_counts else 0.0
    null_max = int(max(null_counts)) if null_counts else 0
    print(f"NULL (pure chance): mean {null_mean:.1f}, worst {null_max} rules pass", flush=True)
    excess = len(found) - null_mean
    print(f"→ real {len(found)} vs chance {null_mean:.1f}  (excess {excess:+.1f})", flush=True)

    # The honest question: do rules that passed train+validate KEEP working on unseen holdout?
    hold_means = [ev["hold"]["mean"] for _, ev in found if ev["hold"]["n"] >= MIN_HOLD_N]
    pos_frac = float(np.mean([m > 0 for m in hold_means])) if hold_means else 0.0
    pooled = float(np.mean(hold_means)) if hold_means else 0.0
    print(f"HOLDOUT of ALL {len(hold_means)} passers: avg {pooled:+.3f}R/trade, "
          f"{pos_frac:.0%} positive (baseline long {base_long['mean']:+.3f}R)", flush=True)

    # ---- rank survivors, test on holdout ONCE, cross-check on ETH ----
    found.sort(key=lambda x: min(x[1]["train"]["mean"], x[1]["val"]["mean"]), reverse=True)
    top = found[: args.top]
    cross = None
    try:
        _, catoms, crl, crs = prepare(args.cross, args.tf, args.bars, cfg)
        craw = ms.fetch_historical(args.cross, args.tf, args.bars, refresh=False)
        # same calendar boundaries on the other coin
        c_tr = int(np.searchsorted(craw.index, raw.index[tr_end]))
        c_va = int(np.searchsorted(craw.index, raw.index[va_end]))
        cross = Data(catoms, crl, crs, craw.index, (c_tr, c_va), D.hold)
    except Exception as e:
        print(f"(ETH cross-check unavailable: {e})", flush=True)

    rows = []
    print("\nTOP RULES (holdout and ETH never used to choose them):", flush=True)
    for rule, ev in top:
        d, names = rule
        label = f"{d.upper():5s} " + " & ".join(names)
        cev = eval_rule(cross, rule) if cross is not None else None
        cs = (
            {"n": cev["train"]["n"] + cev["val"]["n"] + cev["hold"]["n"],
             "mean": float(np.average(
                 [cev[k]["mean"] for k in ("train", "val", "hold")],
                 weights=[max(cev[k]["n"], 1e-9) for k in ("train", "val", "hold")]))}
            if cev else None
        )
        h = ev["hold"]
        print(
            f"- {label}\n    train {ev['train']['mean']:+.2f}R (n={ev['train']['n']}) | "
            f"val {ev['val']['mean']:+.2f}R (n={ev['val']['n']}) | "
            f"HOLDOUT {h['mean']:+.2f}R (n={h['n']}, win {h['win']:.0%})"
            + (f" | ETH all {cs['mean']:+.2f}R (n={cs['n']})" if cs else ""),
            flush=True,
        )
        rows.append({"direction": d, "conditions": list(names), "train": ev["train"],
                     "val": ev["val"], "holdout": ev["hold"], "eth": cs})

    ok_hold = [r for r in rows if r["holdout"]["n"] >= MIN_HOLD_N and r["holdout"]["mean"] > 0]
    if excess < max(5.0, null_mean * 0.5):
        verdict = "NO RELIABLE EDGE FOUND — passing rules are no more common than chance."
    elif pooled < 0.05 or pos_frac < 0.6:
        verdict = (f"Rules looked good in-sample but did NOT hold up on unseen data "
                   f"(avg {pooled:+.2f}R, {pos_frac:.0%} positive) — overfitting, not edge.")
    else:
        verdict = (f"Possible edge: passers average {pooled:+.2f}R on untouched holdout "
                   f"({pos_frac:.0%} positive). Still needs forward paper proof.")
    print("\nVERDICT:", verdict, flush=True)
    from moonshot.config_hash import config_hash as cfg_hash_fn
    from moonshot.discover_fast import data_hash_from_frame
    from moonshot.experiment_log import log_run

    ch = cfg_hash_fn(cfg)
    dh = data_hash_from_frame(raw)
    payload = {
        "ts": pd.Timestamp.now(tz="UTC").isoformat(),
        "symbol": args.symbol,
        "tf": args.tf,
        "config_hash": ch,
        "data_hash": dh,
        "exit": {"stop_atr": STOP_ATR, "target_r": TARGET_R, "hold_bars": D.hold},
        "rules_searched": len(rules),
        "real_pass": len(found),
        "null_mean": null_mean,
        "null_max": null_max,
        "null_counts": null_counts,
        "baseline_long": base_long,
        "baseline_short": base_short,
        "verdict": verdict,
        "top": rows,
    }
    OUT.write_text(json.dumps(payload, indent=2, default=float))
    run_id = f"discover_{pd.Timestamp.now(tz='UTC').strftime('%Y%m%d_%H%M%S')}"
    try:
        log_run(
            run_id=run_id,
            kind="discover",
            config_hash=ch,
            data_hash=dh,
            metrics={"real_pass": len(found), "null_mean": null_mean},
            meta={"symbol": args.symbol, "tf": args.tf},
        )
    except Exception as e:
        print(f"(experiment_log skip: {e})", flush=True)
    print(f"saved {OUT.name}  config_hash={ch} data_hash={dh}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    sys.exit(main())
