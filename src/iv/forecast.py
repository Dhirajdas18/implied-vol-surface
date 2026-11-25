"""Forecasting NIFTY's realised variance over the next h trading days from the close of day t.

Everything is compared as total variance over the window (no annualising choice): the target is
the sum of the next h squared daily log returns; an implied forecast is implied variance times the
calendar time the window spans (D / 365, D = days from t to the h-th next trading day), so its
maturity matches the window exactly.

Forecasts:
- at-the-money implied: SVI total variance at k = 0, interpolated linearly in T to D days;
- full-smile implied (variance-swap): 2 * integral of out-of-the-money undiscounted prices / K^2
  dK, from the SVI smile across the slice's quoted strikes only (VIX likewise stops at the last
  quoted strikes; SVI's extrapolated wings to +-100% put it 9% above VIX), then interpolated in T;
- India VIX (30 days) scaled to D: VIX^2 D / 365 (only where D is within 20-40 days);
- GJR-GARCH with Student-t shocks, refitted each January on NIFTY returns from 2013 to the end of
  the year before, the sum of its 1..h step variance forecasts;
- 20-day: the last 20 squared returns' mean times h; EWMA (0.94) times h.
"""

import numpy as np
import pandas as pd
from scipy.special import ndtr

from iv import surface as s

HORIZONS = (5, 21, 63)
def varswap_w(p, k_lo: float = -1.0, k_hi: float = 1.0, n: int = 801) -> float:
    """Variance-swap total variance of an SVI slice over k in [k_lo, k_hi]: 2 * integral of the
    out-of-the-money undiscounted price (in units of the forward) times e^-k dk (= 2 * integral of
    OTM(K) / K^2 dK). The range is widened to include k = 0."""
    k = np.linspace(min(k_lo, 0.0), max(k_hi, 0.0), n)
    w = np.maximum(s.svi_w(p, k), 1e-12)
    sw = np.sqrt(w)
    d1 = -k / sw + sw / 2
    d2 = d1 - sw
    call = ndtr(d1) - np.exp(k) * ndtr(d2)
    put = np.exp(k) * ndtr(-d2) - ndtr(-d1)
    otm = np.where(k >= 0, call, put)
    return float(2 * np.trapezoid(otm * np.exp(-k), k))


def implied_at(day: pd.DataFrame, days: float, col: str) -> float | None:
    """Total variance `col` (per slice) interpolated linearly in T to `days`, as an annual
    variance; None if no slices lie on both sides."""
    d = day.sort_values("T")
    t = days / 365
    if d["T"].min() > t or d["T"].max() < t:
        return None
    return float(np.interp(t, d["T"], d[col]) / t)


def garch_paths(r_pct: pd.Series, horizons=HORIZONS, years=range(2015, 2027)) -> pd.DataFrame:
    """Date x horizon: GJR-t (refitted each January on returns before it) forecasts of the total
    variance (in decimal units) of the next h returns, from each day's close."""
    from arch import arch_model

    out = []
    hmax = max(horizons)
    for y in years:
        start = pd.Timestamp(y, 1, 1)
        upto = r_pct.loc[: pd.Timestamp(y, 12, 31)]
        if (upto.index < start).sum() < 500 or (upto.index >= start).sum() == 0:
            continue
        am = arch_model(upto, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t")
        first = upto.index[upto.index >= start][0]
        res = am.fit(last_obs=first, disp="off")
        fc = res.forecast(horizon=hmax, start=upto.index[upto.index < start][-1], reindex=False)
        v = fc.variance  # row t: variance of returns t+1..t+hmax given data to t, percent^2
        cum = v.cumsum(axis=1)
        out.append(pd.DataFrame({h: cum.iloc[:, h - 1] / 1e4 for h in horizons}))
    g = pd.concat(out)
    return g[~g.index.duplicated(keep="last")]


def build(svi: pd.DataFrame, close: pd.Series, vix: pd.Series | None = None,
          horizons=HORIZONS) -> pd.DataFrame:
    """One row per origin day t and horizon h: realised total variance of the next h returns and
    each forecast's total variance over the same window."""
    svi = svi[svi["converged"]].copy()
    svi["w_atm"] = [float(s.svi_w({x: r[x] for x in s.PARAMS}, 0.0)) for _, r in svi.iterrows()]
    svi["w_vs"] = [varswap_w({x: r[x] for x in s.PARAMS}, r["k_lo"], r["k_hi"])
                   for _, r in svi.iterrows()]
    by = dict(tuple(svi.groupby("date")))
    r = np.log(close).diff().dropna()
    r2 = r**2
    g = garch_paths(100 * r, horizons)
    hist20 = r2.rolling(20).mean()
    ewma = r2.ewm(alpha=0.06, adjust=False).mean()
    idx = r.index
    pos = {d: i for i, d in enumerate(idx)}
    rows = []
    for d, day in by.items():
        if d not in pos:
            continue
        i = pos[d]
        for h in horizons:
            if i + h >= len(idx):
                continue
            end = idx[i + h]
            D = (end - d).days
            atm = implied_at(day, D, "w_atm")
            vs = implied_at(day, D, "w_vs")
            row = {"date": d, "h": h, "D": D,
                   "realised": float(r2.iloc[i + 1: i + h + 1].sum()),
                   "atm": atm * D / 365 if atm is not None else np.nan,
                   "varswap": vs * D / 365 if vs is not None else np.nan,
                   "garch": float(g.at[d, h]) if d in g.index else np.nan,
                   "hist20": float(hist20.iloc[i]) * h, "ewma": float(ewma.iloc[i]) * h}
            if vix is not None and 20 <= D <= 40:
                v = vix.loc[:d].dropna()
                row["vix"] = float(v.iloc[-1]) ** 2 * D / 365 if len(v) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def qlike(y, f):
    y = np.maximum(np.asarray(y, float), 1e-10)
    f = np.asarray(f, float)
    return y / f - np.log(y / f) - 1


def nw_t(x: np.ndarray, lags: int) -> float:
    """Newey-West t-statistic of the mean of x."""
    x = np.asarray(x, float)
    m = float(np.mean(x))
    e = x - m
    n = len(e)
    v = e @ e / n
    for lag in range(1, lags + 1):
        v += 2 * (1 - lag / (lags + 1)) * (e[lag:] @ e[:-lag]) / n
    return m / np.sqrt(v / n) if v > 0 else float("nan")
