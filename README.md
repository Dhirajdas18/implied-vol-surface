# Implied Volatility Surface

[![CI](https://github.com/Dhirajdas18/implied-vol-surface/actions/workflows/ci.yml/badge.svg)](https://github.com/Dhirajdas18/implied-vol-surface/actions/workflows/ci.yml)
[![Surface lab](https://github.com/Dhirajdas18/implied-vol-surface/actions/workflows/pages.yml/badge.svg)](https://dhirajdas18.github.io/implied-vol-surface/)

The NIFTY implied volatility surface, every strike and expiry on every trading day from 2015 to
2026, built from NSE's daily F&O bhavcopy, with stock option surfaces to follow. Only derived
results (fitted parameters and summaries) are committed; the raw NSE data stays local.

**Live surface lab: [dhirajdas18.github.io/implied-vol-surface](https://dhirajdas18.github.io/implied-vol-surface/)**,
pick a date and an expiry and see the 3D surface, the smile against the market, the term
structure of at-the-money volatility and skew, and the no-arbitrage flags; updated when the
studies are rerun locally.

## Features

- [x] Building the surface: forwards, out-of-the-money quotes, SVI per expiry
- [ ] No-arbitrage checks on quotes and fits
- [ ] Risk-neutral densities and tail probabilities against realised moves
- [ ] Surface dynamics: sticky strike against sticky delta, principal components
- [ ] Forecasting realised volatility: implied by maturity against GARCH and India VIX
- [ ] Stock option surfaces against NIFTY
- [ ] Does the skew or term structure predict returns or crashes

## Findings

### 1. Between quoted strikes, straight lines beat SVI; beyond them, SVI made half the error

<a href="docs/img/holdout.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/holdout.png"><img src="docs/img/holdout.gif" alt="Median error on held-out quotes for SVI, a quadratic and straight lines, between quotes and beyond the quoted strikes" width="720"></picture></a>

<a href="docs/img/term_structure.png"><picture><source media="(prefers-reduced-motion: reduce)" srcset="docs/img/term_structure.png"><img src="docs/img/term_structure.gif" alt="90-day minus 30-day at-the-money volatility against the 30-day level, one dot a day, with the median" width="720"></picture></a>

Each day's forward for each expiry comes from put-call parity at the strikes nearest the money;
quotes are the out-of-the-money options that traded that day, turned into Black-76 implied
volatility (closing prices, 6.5% rate). Every expiry with at least five quotes gets an SVI fit:
18,345 slices on 2,927 days, 2015–2026. For the hold-out test, every slice with at least ten quotes
(17,362) is fitted three ways without some of its quotes and scored on them: every third interior
quote (a strike between quotes) or the two lowest and two highest strikes (beyond the quotes).

- **Parity forwards matched the futures.** Against the same expiry's futures settle, the parity
  forward differed by a median 2.0 bp (90th percentile 6.4 bp); the futures settle was needed for
  only 0.5% of slices.
- **Between quotes, the quotes themselves were the best curve.** Straight lines between neighbouring
  quotes missed a held-out quote by a median 0.26 vol points, SVI by 0.35 and a quadratic by 0.70;
  lines were best for 66% of slices.
- **Beyond the quotes, SVI was the only safe guess.** It missed the outer strikes by 0.97 points
  against 1.75 for flat lines and 2.02 for a quadratic, and was best for 69% of slices; for weekly
  expiries the quadratic missed by 5.85.
- **Puts were dearer on 98.7% of slices, and the skew faded with maturity.** The at-the-money slope
  of volatility against log-moneyness went from −0.47 at 8–14 days to −0.19 beyond 200 days; within a
  day it fell with maturity to a median power of −0.30 (ten times the maturity, about half the skew).
- **The term structure tilted with the level.** 90-day volatility sat 0.70 points above 30-day on
  the calmest third of days and 0.91 below on the most volatile third, where it was inverted by over
  a point on 47% of days (thirds cut on all days; correlation −0.70). India VIX, computed from all
  strikes, was above 30-day at-the-money volatility on 97% of days, by a median 1.38 points
  (correlation 0.987).

## Quick start

```bash
uv sync
uv run python scripts/fetch_options.py   # NSE bhavcopy 2015 -> today into data/options/
                                         # (about an hour; resumable)
uv run python -m iv.build                # SVI per day and expiry -> data/derived/svi_daily.parquet
                                         # and docs/data/lab_surfaces.json
uv run iv report surface                 # a study's numbers and charts -> out/
uv run python scripts/build_site.py      # surface lab -> site/index.html (committed JSON only)
uv run pytest
```

Set `IV_LAKE_DIR=/path/to/lake` to take the trading calendar from a local data lake's
`market.parquet` instead of Yahoo's ^NSEI dates. The lab and the tests need no NSE data.

## Design

- `src/iv/options.py`: download and parse NSE's daily F&O bhavcopy, in the format before 8 July
  2024 and the UDiFF format after it (adapted from black-scholes-pricer); polite pacing and
  retries.
- `src/iv/data.py`: paths and the trading calendar (Yahoo ^NSEI, or the lake's NIFTY dates).
- `src/iv/surface.py`: the core maths in numpy and scipy: Black-76 prices and a bracketed Newton
  implied-volatility solver; forwards from put-call parity (median over the five traded strikes
  nearest the money) with the futures settlement price as the fallback; quote cleaning; SVI fits
  with box bounds that enforce b ≥ 0, |ρ| < 1, σ > 0, a non-negative minimum variance and Lee's
  wing bound, started from a quasi-explicit grid; Gatheral's butterfly function g(k) and the
  calendar check; at-the-money volatility, the 25-delta risk reversal and the skew at k = ±0.1.
- `src/iv/build.py`: every day's forwards, quotes, SVI slices and arbitrage flags in parallel,
  into `data/derived/svi_daily.parquet`, and the page's compact aggregate
  `docs/data/lab_surfaces.json`.
- `src/iv/iv.js`: the lab's browser copy of the SVI variance, implied volatility and g(k), tested
  against Python through Node to 1e-9.
- `src/iv/shape.py`: the hold-out test (SVI, a quadratic and straight lines on held-out quotes),
  the at-the-money skew from SVI parameters and the 30- and 90-day at-the-money volatility.
- `src/iv/reports.py`: README findings, reproducible with `iv report <topic>` (India VIX from the
  local lake when `IV_LAKE_DIR` is set).
- `src/iv/site.py`: the surface lab, one static Plotly page built from the committed JSON alone.
- `src/iv/plots.py`: the shared chart style and the animated README charts.
- Conventions: time to expiry is calendar days / 365; a constant 6.5% rate discounts option
  prices and enters put-call parity (the forward comes from the options, so no dividend guess);
  log-moneyness k = ln(K / F); total variance w = σ²T; volatilities in percent, skew and fit
  errors in volatility points.
- Data: NSE's F&O bhavcopy from nsearchives.nseindia.com, NIFTY options and futures every day,
  stock options and futures on each month's last trading day. Option prices are closing prices
  (the day's last trades) of contracts that traded; futures use the settlement price. The raw
  data is downloaded to `data/` and never committed.
- Charts: one validated palette; README charts are animated GIFs with a static fallback for
  reduced-motion users.

Research only, not investment advice. MIT licence.
