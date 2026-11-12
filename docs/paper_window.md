# Paper measurement window (Phase 0)

## Freeze rationale

Escape promoted `momentum_15` / `1h` / `5x` off a death `10x@5m` book.
`AUTOPILOT_FROZEN=1` keeps paper on that identity so expectancy can be scored
honestly without mid-window hunt thrash rewriting `.env` / restarting paper.

Research/hunt may still run; promote apply and paper restart from hunt are suppressed.

## Pinned book

- Strategy: `momentum_15`
- Timeframe: `1h`
- Leverage: `5`
- `LIVE_TRADING_ENABLED=0` (practice only)
- Pinned `config_hash` (from live `.env` book): `3f689065ab7153af`

Every paper close must carry `config_hash` (already wired in live/paper path).

## paper_window_start_ts

`2026-11-12T10:50:00+00:00` — Phase 0 freeze for post-escape `5x@1h` measurement.

## Decision identity (pinned hash only)

Phase 0 KEEP / STOP / EXTEND is scored **only** on closes whose `config_hash`
equals the pinned book hash above. Old `5m` / other-leverage / LEGACY rows are
**excluded** from the decision table (they must not pollute EXTEND / mean R).

`scripts/paper_expectancy.py` and autopilot refresh default to `pinned_only=True`.

## Legacy / pre-window trades

Rows with `config_hash IS NULL` are the **legacy** bucket — excluded.
Closes under other hashes (e.g. prior `10x@5m`) are excluded from this window’s
decision (not merely listed as separate buckets).

## Time cap

- Primary: **8 weeks** from `paper_window_start_ts`
- If `n < 100` at the cap → **EXTEND** 4 weeks; no tuning

## LOCKED acceptance criteria (written before looking)

Decision rule (pinned hash only):

1. **KEEP-VALIDATED:** `n >= 100` AND bootstrap 95% CI on `mean(r_multiple)` excludes 0 and is positive
2. **STOP:** `n >= 100` AND (`mean(r_multiple) <= 0` OR CI includes 0)
   - If `cost_share > 40%` → execution problem → fix fills/costs first
   - If `cost_share <= 40%` → retire this hash; do not polish it; restart research under fold-primary gates only
3. **EXTEND:** `n < 100` → keep measuring; no tuning of this book
   - At the **8-week** time cap with `n < 100` → EXTEND +4 weeks (still no tuning)

## LOCKED units note

- `r_multiple` is already **net** of fees and funding
- `cost_share` is dollar costs / gross dollar PnL (see `moonshot/paper_expectancy.py`)

## Ops vs research

- Escape / silence / bleed: while frozen, **no apply**
- When freeze lifts: escape may deploy with `research_promote=false`; **never** raise `champion_peak`

## Freeze status

- Set: `2026-11-12` — Phase 0 measure window
- End: only by explicit human env flip of `AUTOPILOT_FROZEN=0`
