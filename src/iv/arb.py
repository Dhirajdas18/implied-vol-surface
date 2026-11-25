"""Static arbitrage in NIFTY option closing prices, model-free, and in the SVI fits.

Quotes (every traded option, calls and puts separately, one expiry at a time, by strike):
- order: a call must not cost more than a call at a lower strike; a put must not cost more than a
  put at a higher strike. Violation = the dearer price minus the cheaper, in rupees.
- spread: a call spread (buy K1, sell K2 > K1) can never pay more than e^(-rT)(K2 - K1), so its
  price can't exceed that; likewise the put spread. Violation = price minus that bound.
- butterfly: the price is convex in strike; for neighbours K1 < K2 < K3 the middle price must not
  exceed the straight line between the outer two. Violation = middle price minus that line
  (a long butterfly bought for less than zero).
- calendar: across consecutive expiries, the forward-normalised undiscounted price
  e^(rT) P / F at the same K / F must not fall with maturity (out-of-the-money quotes; the longer
  expiry's price is linearly interpolated in K / F, which can only overstate it, so the count is
  a lower bound). Violation in rupees on the shorter expiry's forward.
Every violation carries the smallest contracts traded among its legs, to tell real prices from
stale closes.

Fits: Gatheral's g(k) >= 0 (butterfly) and total variance rising with T (calendar), as flagged by
`iv.build`; `fit_svi_no_butterfly` refits a slice with a penalty on negative g across its quoted
range.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from iv import surface as s


def _one_side(e: pd.DataFrame, T: float, kind: str, r: float) -> list[dict]:
    e = e.sort_values("strike")
    K, P, n = (e[c].to_numpy(float) for c in ("strike", "price", "contracts"))
    out = []
    if len(K) < 2:
        return out
    disc = np.exp(-r * T)
    dK = np.diff(K)
    # order and spread bounds between neighbours
    gap = -np.diff(P) if kind == "PE" else np.diff(P)  # > 0: the dearer strike is the wrong one
    spread = (np.diff(P) if kind == "PE" else -np.diff(P)) - disc * dK
    legs2 = np.minimum(n[:-1], n[1:])
    for check, v in (("order", gap), ("spread", spread)):
        for i in np.flatnonzero(v > 0):
            out.append({"check": check, "amount": float(v[i]), "min_contracts": float(legs2[i]),
                        "strike": float(K[i])})
    if len(K) >= 3:
        lam = (K[2:] - K[1:-1]) / (K[2:] - K[:-2])
        line = lam * P[:-2] + (1 - lam) * P[2:]
        v = P[1:-1] - line
        legs3 = np.minimum(np.minimum(n[:-2], n[1:-1]), n[2:])
        for i in np.flatnonzero(v > 0):
            out.append({"check": "butterfly", "amount": float(v[i]),
                        "min_contracts": float(legs3[i]), "strike": float(K[i + 1])})
    return out


def quote_checks(day: pd.DataFrame, date, r: float = s.R, min_contracts: int = 1,
                 otm_only: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One day's model-free checks on options with at least `min_contracts` contracts traded
    (thinner strikes are dropped before neighbours are formed); with `otm_only`, only puts below
    and calls at or above each expiry's forward. Returns (violations, tested): violations has
    date, expiry, days, type, check, amount (rupees), min_contracts, strike; tested counts the
    comparisons made per expiry and check (so rates can be formed)."""
    date = pd.Timestamp(date)
    day = day[day["contracts"] >= min_contracts]
    fwd = s.forwards(day, date)
    t = day[(day["contracts"] > 0) & (day["price"] > 0)]
    if otm_only:
        t = t.merge(fwd[["expiry", "F"]], on="expiry", how="inner")
        t = t[((t["type"] == "PE") & (t["strike"] < t["F"]))
              | ((t["type"] == "CE") & (t["strike"] >= t["F"]))]
    t = t.groupby(["expiry", "type", "strike"], as_index=False).agg(
        price=("price", "last"), contracts=("contracts", "sum"))
    rows, tested = [], []
    for (expiry, kind), e in t.groupby(["expiry", "type"]):
        days = (pd.Timestamp(expiry) - date).days
        if days < s.MIN_DAYS or days > s.MAX_DAYS:
            continue
        T = days / 365
        n = len(e)
        tested += [{"expiry": expiry, "days": days, "type": kind, "check": c, "tests": k}
                   for c, k in (("order", n - 1), ("spread", n - 1), ("butterfly", n - 2)) if k > 0]
        for v in _one_side(e, T, kind, r):
            rows.append({"expiry": expiry, "days": days, "type": kind} | v)
    # calendar, on out-of-the-money quotes in forward-normalised units
    q = s.clean_quotes(day, fwd, r, min_contracts=max(1, min_contracts))
    if len(q):
        q = q.assign(x=q["strike"] / q["F"], c=np.exp(r * q["T"]) * q["price"] / q["F"])
        exps = sorted(q["expiry"].unique())
        for e1, e2 in zip(exps[:-1], exps[1:], strict=True):
            a, b = q[q["expiry"] == e1], q[q["expiry"] == e2]
            for kind in ("PE", "CE"):
                aa, bb = a[a["type"] == kind], b[b["type"] == kind].sort_values("x")
                if len(bb) < 2:
                    continue
                inside = aa[(aa["x"] >= bb["x"].iloc[0]) & (aa["x"] <= bb["x"].iloc[-1])]
                if inside.empty:
                    continue
                c2 = np.interp(inside["x"], bb["x"], bb["c"])
                gap = (inside["c"].to_numpy() - c2) * inside["F"].to_numpy()
                days = (pd.Timestamp(e1) - date).days
                tested.append({"expiry": e1, "days": days, "type": kind, "check": "calendar",
                               "tests": len(inside)})
                # legs: the shorter quote and the two longer quotes around it
                j = np.clip(np.searchsorted(bb["x"].to_numpy(), inside["x"].to_numpy()), 1,
                            len(bb) - 1)
                nb = np.minimum(bb["contracts"].to_numpy()[j - 1], bb["contracts"].to_numpy()[j])
                legs = np.minimum(inside["contracts"].to_numpy(), nb)
                for i in np.flatnonzero(gap > 0):
                    rows.append({"expiry": e1, "days": days, "type": kind, "check": "calendar",
                                 "amount": float(gap[i]), "min_contracts": float(legs[i]),
                                 "strike": float(inside["strike"].iloc[i])})
    cols = ["expiry", "days", "type", "check", "amount", "min_contracts", "strike"]
    v = pd.DataFrame(rows, columns=cols).assign(date=date)
    v["expiry"] = pd.to_datetime(v["expiry"])
    return v, pd.DataFrame(tested).assign(date=date)


def quote_checks_day(args):
    date, day, min_contracts, otm_only = args
    return quote_checks(day, date, min_contracts=min_contracts, otm_only=otm_only)


def fit_svi_no_butterfly(k, w, T, start: dict, penalty: float = 1.0, margin: float = 1e-4,
                         n_grid: int = 101) -> dict:
    """Refit an SVI slice from `start` with residuals penalty * (margin - g(k)) wherever g(k) <
    margin on a grid over the quoted range, added to the weighted fit residuals (which are of
    order 1e-4 to 1e-3 in total variance, so a penalty of 1 makes any dip in g dominate).
    Same bounds as `surface.fit_svi`. Returns the parameters, rmse (vol points) and min_g."""
    k, w = np.asarray(k, float), np.asarray(w, float)
    wt = 1 / np.sqrt(np.maximum(w, 1e-12))
    wt = wt / wt.mean()
    grid = np.linspace(k.min(), k.max(), n_grid)
    lo = np.array([0.0, 1e-6, 1e-6, k.min(), 1e-4])
    hi = np.array([2 * w.max() + 1e-6, s.LEE, s.LEE, k.max(), 2.0])
    x0 = np.clip(s._to_slopes(np.array([start[x] for x in s.PARAMS])), lo + 1e-9, hi - 1e-9)

    def resid(x):
        p = s._from_slopes(x)
        g = s.svi_g(p, grid)
        g = np.where(np.isfinite(g), g, -1.0)
        return np.r_[(s.svi_w(p, k) - w) * wt, penalty * np.clip(margin - g, 0, None)]

    res = least_squares(resid, x0, bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=3000)
    p = dict(zip(s.PARAMS, map(float, s._from_slopes(res.x)), strict=True))
    iv = s.svi_iv(p, k, T)
    rmse = 100 * float(np.sqrt(np.mean((iv - np.sqrt(w / T)) ** 2)))
    return p | {"rmse": rmse, "min_g": float(np.nanmin(s.svi_g(p, grid)))}


def refit_day(args) -> list[dict]:
    """For one day, every slice flagged with butterfly arbitrage in `flagged` (expiry -> SVI
    parameters) refitted with `fit_svi_no_butterfly`, plus its out-of-the-money quotes' own
    butterfly breaches (rupees) among neighbouring strikes."""
    date, day, flagged = args
    date = pd.Timestamp(date)
    q = s.clean_quotes(day, s.forwards(day, date))
    rows = []
    for expiry, start in flagged.items():
        e = q[q["expiry"] == expiry]
        if len(e) < 5:
            continue
        T = float(e["T"].iloc[0])
        k, w = e["k"].to_numpy(), e["w"].to_numpy()
        before = s.butterfly_check(start, np.linspace(k.min(), k.max(), 201))
        f = fit_svi_no_butterfly(k, w, T, start)
        after = s.butterfly_check(f, np.linspace(k.min(), k.max(), 201))
        iv0 = s.svi_iv(start, k, T)
        rmse0 = 100 * float(np.sqrt(np.mean((iv0 - np.sqrt(w / T)) ** 2)))
        own = []
        for kind in ("PE", "CE"):
            own += [v["amount"] for v in _one_side(e[e["type"] == kind], T, kind, s.R)
                    if v["check"] == "butterfly"]
        rows.append({"date": date, "expiry": expiry, "days": (pd.Timestamp(expiry) - date).days,
                     "n": len(e), "rmse_before": rmse0, "rmse_after": f["rmse"],
                     "min_g_before": before["min_g"], "min_g_after": f["min_g"],
                     "fixed": not after["violated"],
                     "quote_breach_max": max(own) if own else 0.0})
    return rows
