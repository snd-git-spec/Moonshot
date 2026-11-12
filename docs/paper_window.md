# Paper measurement window

## Fresh start (2026-11-12)

Full reset at `2026-11-12T12:46:36+00:00`:

- Paper ledger wiped (trades / positions / equity / risk / events)
- Hunt memory wiped (history, banlist, champion, elite archive, islands)
- `STARTING_EQUITY=1000.0`
- `AUTOPILOT_FROZEN=0` — hunt may promote / restart paper again
- `LIVE_TRADING_ENABLED=0` (practice only)

Backup on droplet: `/opt/moonshot/backups/full_reset_20261112T124631Z`

## Live seed book (starting identity)

- Strategy: `momentum_15`
- Timeframe: `1h`
- Leverage: `5`

Hunt may replace this when a candidate passes gates.

## paper_window_start_ts

`2026-11-12T12:46:36+00:00` — post full-reset measurement.

## Decision identity (pinned hash only while scoring a fixed book)

KEEP / STOP / EXTEND for a fixed book uses closes whose `config_hash` equals
that book’s hash. After a promote, re-resolve the hash from `.env`.

`scripts/paper_expectancy.py` defaults to `pinned_only=True` (current `.env` hash).

## Time cap

- Primary: **8 weeks** from `paper_window_start_ts`
- If `n < 100` at the cap → **EXTEND** 4 weeks; no tuning

## LOCKED acceptance criteria

1. **KEEP-VALIDATED:** `n >= 100` AND bootstrap 95% CI on `mean(r_multiple)` excludes 0 and is positive
2. **STOP:** `n >= 100` AND (`mean(r_multiple) <= 0` OR CI includes 0)
   - If `cost_share > 40%` → execution problem first
   - Else retire hash; do not polish a dead book
3. **EXTEND:** `n < 100` → keep measuring

## LOCKED units

- `r_multiple` is already **net** of fees and funding
- `cost_share` is dollar costs / gross dollar PnL
