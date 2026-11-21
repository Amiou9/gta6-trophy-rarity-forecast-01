# GTA VI trophy rarity forecast — gta6-trophy-rarity-forecast-01

Forecasts every GTA VI trophy's Exophase earn rate at 2026-11-26 00:00 UTC
as five quantiles per trophy, scored by pinball loss on the logit scale.
`main.py` only — Python 3.13, standard library only, no network, deterministic,
~0.1 s per case.

## Model

Per trophy, on the log-time axis `x = log1p(hours since first read)`:

- **A — direct:** `logit(rate) ~ a + b·x`, fitted on distinct reads only
  (consecutive identical reads carry no information while Exophase sits
  unchanged for hours).
- **B — counts:** `log(earners) ~ aE + bE·x` minus `log(players_tracked) ~
  aN + bN·x`, so a refresh that adds tracked players dilutes the rate
  mechanically without being read as trophies being lost.

The two drifts are combined 0.3·A + 0.7·B in logit space. Slopes are fitted
on the **recent window only** (last 1.0 of log-time) — a saturating curve's
recent slope is the one that extrapolates — then shrunk toward the median
slope of the trophy's tier (weight `n/(n+6)`), and the extrapolated logit
move is capped at ±1.5.

Spread (logit sd) is calibrated from the history, not fixed:
`sqrt(0.05² + (0.30·sqrt(days))² + (se_b·dx)² + (0.5·resid_sd)²)`.
Flat histories get a narrow spread; volatile ones get a wide, honest one.

## Validation (local, `~/workspace/makefun/backtest_trophy.py`)

- Practice cases p00/p01: valid output (55 rows, non-decreasing quantiles,
  2,669 bytes), deterministic across runs (`diff` identical), 0.07 s.
- Practice backtest (forecast each next read from the reads before it):
  pinball loss **0.0608 vs persistence 0.0664** (1-step, −8.4%) and
  **0.0727 vs 0.0938** (2-step, −22.5%). Histories are flat, so the gain
  there is spread calibration; the median ties persistence by construction.
- Synthetic long-horizon backtests to day 7 (cuts at 24/36/48/60 h, two
  regimes — earners-saturating with dilution dominating, and
  rate-saturating): beats persistence by **5–17%** in the first regime and
  **41–55%** in the second, and beats a linear-in-time logit trend baseline
  by a wide margin in both. The whole-history fit lost to persistence in
  the first regime; the recent-window fit fixed that.
- Robustness: single flat read, null rates, a trophy at 0% and one at
  100%, and a 500-read case (final-case volume) all produce valid output
  in 0.14 s total.

## Run

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```
