# Paper measurement window

## Freeze rationale

Hunt thrash (silence_rotate into weaker books) corrupted Live measurement.
`AUTOPILOT_FROZEN=1` keeps paper on the deployed book so expectancy can be scored
honestly without mid-window config rewrites.

## paper_window_start_ts

`2026-11-05T11:12:00+00:00`

## Legacy trades

Rows with `config_hash IS NULL` are the **legacy** bucket. Excluded from overall
KEEP/STOP scoring once `paper_expectancy.py` lands (A.5). Pre-migration closes
remain auditable but do not count toward the frozen-window decision.

## Time cap

- Primary: **8 weeks** from `paper_window_start_ts`
- If `n < 100` at the cap → **EXTEND** 4 weeks; no tuning

Expected outcome at first review: **EXTEND** (low trade count likely).

## LOCKED acceptance criteria

- **KEEP-VALIDATED:** `n >= 100` AND bootstrap 95% CI on `mean(r_multiple)` excludes 0 and is positive
- **STOP:** `n >= 100` AND (`mean(r_multiple) <= 0` OR CI includes 0)
  - If `cost_share > 40%` → execution problem → Phase **E** first
  - If `cost_share <= 40%` → retire deployed book; restart research
- **EXTEND:** `n < 100` at 8-week cap → +4 weeks; no tuning

## LOCKED units note

- `r_multiple` is already **net** of fees and funding (`Position.close_out`)
- Never compute `gross_R = sum(r_multiple) + sum(fees) + sum(funding)` (invalid)
- Correct:

```text
mean_r = mean(t.r_multiple for t in trades)
ci = bootstrap_95(mean_r)
gross_dollar_pnl = sum(t.r_multiple * t.initial_risk_dollars) + sum(t.fees) + sum(t.funding)
cost_share = (sum(t.fees) + sum(t.funding)) / gross_dollar_pnl
```

- The **40%** threshold applies to **cost_share (dollars)**, not to gross R.

## Freeze ended

`2026-11-06T12:13:21+00:00` — user requested **always apply the best**. `AUTOPILOT_FROZEN=0`.
Champion peak (gen 1034, ADX22) synced onto Live; future research promotes may rewrite `.env` again.
