"""The risk-neutral distribution of NIFTY at an expiry, read from an SVI slice, against what
NIFTY did.

For x = ln(S_T / F) (the move from the forward to the price at expiry), with total variance w(k)
from the slice and d2 = -k / sqrt(w) - sqrt(w) / 2 (Black-76 with the smile):
- cdf: Q(x < k) = N(-d2) + n(d2) w'(k) / (2 sqrt(w))  (the price of a digital put, undiscounted);
- pdf: g(k) / sqrt(2 pi w) exp(-d2^2 / 2), Gatheral's g (iv.surface.svi_g), so g >= 0 is needed
  for a density (slices flagged for butterfly arbitrage are refitted with the no-butterfly
  penalty first).
The flat alternative uses the at-the-money total variance at every strike: x ~ N(-w0/2, w0).

`origins` picks one slice per monthly expiry, the one closest to `target` days before it (within
`lo`-`hi` days), so the months don't overlap.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr

from iv import surface as s

SQ2PI = np.sqrt(2 * np.pi)


def _w_and_slope(p, k):
    a, b, rho, m, sig = (p[x] for x in s.PARAMS)
    k = np.asarray(k, dtype=float)
    root = np.sqrt((k - m) ** 2 + sig**2)
    return a + b * (rho * (k - m) + root), b * (rho + (k - m) / root)


def cdf(p, k):
    """Risk-neutral probability that ln(S_T / F) < k."""
    w, w1 = _w_and_slope(p, k)
    sw = np.sqrt(w)
    d2 = -np.asarray(k) / sw - sw / 2
    return ndtr(-d2) + np.exp(-d2 * d2 / 2) / SQ2PI * w1 / (2 * sw)


def pdf(p, k):
    """Risk-neutral density of ln(S_T / F) at k."""
    w, _ = _w_and_slope(p, k)
    sw = np.sqrt(w)
    d2 = -np.asarray(k) / sw - sw / 2
    return s.svi_g(p, k) / (SQ2PI * sw) * np.exp(-d2 * d2 / 2)


def flat_cdf(w0, k):
    return ndtr((np.asarray(k) + w0 / 2) / np.sqrt(w0))


def flat_logpdf(w0, k):
    z = (np.asarray(k) + w0 / 2) / np.sqrt(w0)
    return -0.5 * z * z - np.log(SQ2PI * np.sqrt(w0))


def origins(svi: pd.DataFrame, target: int = 30, lo: int = 20, hi: int = 45) -> pd.DataFrame:
    """One row of `svi` per monthly expiry: the slice of that expiry whose days to expiry is
    closest to `target`, within [lo, hi]."""
    m = svi[svi["monthly"] & svi["days"].between(lo, hi) & svi["converged"]].copy()
    m["gap"] = (m["days"] - target).abs()
    return (m.sort_values(["expiry", "gap", "date"]).groupby("expiry").head(1)
            .drop(columns="gap").reset_index(drop=True))


MOVES = (3, 5, 10)  # percent moves from the forward, down and up


def run(svi: pd.DataFrame, opts: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """For each origin (`origins`): the realised move x = ln(close on the expiry day / forward),
    its probability-integral transform under the smile and under the flat alternative, the log
    density of x under each, and the priced probabilities of moves beyond MOVES percent. Slices
    flagged for butterfly arbitrage are refitted with `iv.arb.fit_svi_no_butterfly` first.
    `close` is NIFTY's daily close indexed by date; expiries without a close are skipped."""
    from iv.arb import fit_svi_no_butterfly

    rows = []
    for r in origins(svi).itertuples():
        if r.expiry not in close.index:
            continue
        p = {x: getattr(r, x) for x in s.PARAMS}
        if r.butterfly:
            day = opts[opts["date"] == r.date]
            q = s.clean_quotes(day, s.forwards(day, r.date))
            e = q[q["expiry"] == r.expiry]
            f = fit_svi_no_butterfly(e["k"].to_numpy(), e["w"].to_numpy(), r.T, p)
            p = {x: f[x] for x in s.PARAMS}
        x = float(np.log(close.loc[r.expiry] / r.F))
        w0 = float(s.svi_w(p, 0.0))
        row = {"date": r.date, "expiry": r.expiry, "days": r.days, "x": x, "sd_atm": np.sqrt(w0),
               "pit": float(cdf(p, x)), "pit_flat": float(flat_cdf(w0, x)),
               "ls": float(np.log(max(float(pdf(p, x)), 1e-300))),
               "ls_flat": float(flat_logpdf(w0, x)), "refit": bool(r.butterfly)}
        for t in MOVES:
            kd, ku = np.log(1 - t / 100), np.log(1 + t / 100)
            row |= {f"q_down{t}": float(cdf(p, kd)), f"q_up{t}": float(1 - cdf(p, ku)),
                    f"flat_down{t}": float(flat_cdf(w0, kd)),
                    f"flat_up{t}": float(1 - flat_cdf(w0, ku)),
                    f"down{t}": x < kd, f"up{t}": x > ku}
        grid = np.linspace(-1, 1, 4001)
        dens = pdf(p, grid)
        row |= {"mass": float(np.trapezoid(dens, grid)),
                "mean": float(np.trapezoid(np.exp(grid) * dens, grid))}
        rows.append(row)
    return pd.DataFrame(rows)
