"""How well a smooth curve describes one expiry's quotes, and the shape of the fitted surface.

Hold-out test on a slice's cleaned out-of-the-money quotes (sorted by log-moneyness k), three
curves fitted to the remaining quotes and scored on the held-out ones, error in volatility points:
- SVI (`surface.fit_svi`);
- a quadratic in k fitted to implied volatility by least squares;
- straight lines between neighbouring quotes, flat beyond the last quote (no model at all).
Two hold-outs: "between" holds out every third interior quote (interpolation); "wings" holds out
the two lowest and two highest strikes (extrapolation beyond the quoted range).

Shape of the surface from the daily SVI table:
- `atm_slope`: d(implied vol)/dk at k = 0 from the SVI parameters (the at-the-money skew);
- `term_points`: at-the-money volatility at 30 and 90 days, by linear interpolation of
  at-the-money total variance in T across the day's slices.
"""

import numpy as np
import pandas as pd

from iv import surface as s

MIN_QUOTES = 10


def _quad(k, iv, k_new):
    c = np.polyfit(k, iv, 2)
    return np.polyval(c, k_new)


def _linear(k, iv, k_new):
    return np.interp(k_new, k, iv)  # flat beyond the end quotes


def holdout(k, w, T) -> dict:
    """Mean absolute error (volatility points) of SVI, quadratic and linear on the held-out quotes,
    for the "between" and "wings" hold-outs. NaN for a scheme that can't run."""
    order = np.argsort(k)
    k, w = np.asarray(k, float)[order], np.asarray(w, float)[order]
    iv = np.sqrt(w / T)
    n = len(k)
    out = {}
    idx = np.arange(n)
    schemes = {
        "between": (idx > 0) & (idx < n - 1) & (idx % 3 == 1),
        "wings": (idx < 2) | (idx >= n - 2),
    }
    for name, held in schemes.items():
        fit_k, fit_w, fit_iv = k[~held], w[~held], iv[~held]
        if held.sum() == 0 or len(fit_k) < 6:
            for m in ("svi", "quadratic", "linear"):
                out[f"{name}_{m}"] = np.nan
            continue
        f = s.fit_svi(fit_k, fit_w, T)
        p = {x: f[x] for x in s.PARAMS}
        preds = {
            "svi": s.svi_iv(p, k[held], T) if np.isfinite(f["a"]) else np.full(held.sum(), np.nan),
            "quadratic": _quad(fit_k, fit_iv, k[held]),
            "linear": _linear(fit_k, fit_iv, k[held]),
        }
        for m, pr in preds.items():
            out[f"{name}_{m}"] = 100 * float(np.mean(np.abs(pr - iv[held])))
    return out


def holdout_day(args) -> list[dict]:
    """Hold-out errors for every slice of one day with at least MIN_QUOTES quotes."""
    date, day = args
    date = pd.Timestamp(date)
    q = s.clean_quotes(day, s.forwards(day, date))
    rows = []
    for expiry, e in q.groupby("expiry"):
        if len(e) < MIN_QUOTES:
            continue
        T = float(e["T"].iloc[0])
        rows.append(
            {"date": date, "days": (pd.Timestamp(expiry) - date).days, "n": len(e)}
            | holdout(e["k"].to_numpy(), e["w"].to_numpy(), T)
        )
    return rows


def atm_slope(p, T) -> float:
    """d(implied vol)/dk at k = 0 for an SVI slice (negative when puts are dearer)."""
    a, b, rho, m, sig = (p[x] for x in s.PARAMS)
    w0 = float(s.svi_w(p, 0.0))
    dw = b * (rho - m / np.sqrt(m * m + sig * sig))
    return float(dw / (2 * np.sqrt(w0 * T)))


def term_points(day: pd.DataFrame, short: int = 30, long: int = 90) -> dict | None:
    """At-the-money volatility at `short` and `long` days from one day's slices (columns days,
    T, atm_vol), interpolating total variance linearly in T. None if the slices don't span both."""
    d = day.sort_values("T")
    if d["days"].min() > short or d["days"].max() < long:
        return None
    wv = d["atm_vol"] ** 2 * d["T"]
    out = {}
    for h in (short, long):
        t = h / 365
        out[h] = float(np.sqrt(np.interp(t, d["T"], wv) / t))
    return out
