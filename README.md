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

- [ ] Building the surface: forwards, out-of-the-money quotes, SVI per expiry
- [ ] No-arbitrage checks on quotes and fits
- [ ] Risk-neutral densities and tail probabilities against realised moves
- [ ] Surface dynamics: sticky strike against sticky delta, principal components
- [ ] Forecasting realised volatility: implied by maturity against GARCH and India VIX
- [ ] Stock option surfaces against NIFTY
- [ ] Does the skew or term structure predict returns or crashes

## Findings

Findings are added as each study is completed.

## Quick start

```bash
uv sync
uv run python scripts/fetch_options.py   # NSE bhavcopy 2015 -> today into data/options/
                                         # (about an hour; resumable)
uv run python -m iv.build                # SVI per day and expiry -> data/derived/svi_daily.parquet
                                         # and docs/data/lab_surfaces.json
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
