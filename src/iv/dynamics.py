"""How the NIFTY surface moves from one day to the next.

Constant-maturity surface: on each day, total variance at log-moneyness k and maturity T is
interpolated linearly in T between the two slices around T at the same k (`vol_at`); ln F is
interpolated the same way. Days whose slices don't span T are skipped.

Sticky rules (Derman's names), for the 30-day surface from day t to t+1, r = ln(F30_t+1 / F30_t):
- sticky moneyness (sticky delta): tomorrow's vol at k equals today's vol at k;
- sticky strike: tomorrow's vol at strike K equals today's at K, i.e. at k it is today's at k + r.
At the money, sticky strike gives d(vol) = S r with S today's slope d(vol)/dk at 0, sticky moneyness
gives 0; `atm_moves` returns the pieces to estimate beta in d(vol) = beta S r (1 = sticky strike,
0 = sticky moneyness, 2 = the local-volatility rule of thumb).

PCA: daily changes of the vol grid (moneyness x maturity), principal components by eigen-
decomposition of their covariance.
"""

import numpy as np
import pandas as pd

from iv import surface as s

K_GRID = np.log([0.95, 0.975, 1.0, 1.025, 1.05])  # 90% and 110% are often unquoted at 30-90 days
T_GRID = (30, 60, 90)


def vol_at(day: pd.DataFrame, k, days: float, quoted_only: bool = False):
    """Implied vol at log-moneyness k (array) and `days` to expiry on one day (rows of the SVI
    table, converged), or None if no slices lie on both sides of `days`. With `quoted_only`, NaN
    where k is outside the quoted range of either slice used."""
    d = day.sort_values("T")
    t = days / 365
    lo, hi = d[d["T"] <= t], d[d["T"] >= t]
    if lo.empty or hi.empty:
        return None
    a, b = lo.iloc[-1], hi.iloc[0]
    k = np.asarray(k, dtype=float)
    wa = s.svi_w({p: a[p] for p in s.PARAMS}, k)
    if b["T"] == a["T"]:
        w = wa
    else:
        wb = s.svi_w({p: b[p] for p in s.PARAMS}, k)
        u = (t - a["T"]) / (b["T"] - a["T"])
        w = (1 - u) * wa + u * wb
    v = np.sqrt(np.maximum(w, 1e-12) / t)
    if quoted_only:
        lo_k, hi_k = max(a["k_lo"], b["k_lo"]), min(a["k_hi"], b["k_hi"])
        v = np.where((k >= lo_k - 1e-9) & (k <= hi_k + 1e-9), v, np.nan)
    return v


def fwd_at(day: pd.DataFrame, days: float) -> float | None:
    d = day.sort_values("T")
    t = days / 365
    if d["T"].min() > t or d["T"].max() < t:
        return None
    return float(np.exp(np.interp(t, d["T"], np.log(d["F"]))))


def grid(svi: pd.DataFrame, ks=K_GRID, maturities=T_GRID) -> pd.DataFrame:
    """Date x (maturity, moneyness) implied vols, plus the 30-day forward, on days whose slices
    span every maturity; a point outside the quoted range of its slices is NaN."""
    rows = {}
    for date, day in svi.groupby("date"):
        vals = {}
        for m in maturities:
            v = vol_at(day, ks, m, quoted_only=True)
            if v is None:
                break
            for k, x in zip(ks, v, strict=True):
                vals[(m, round(float(np.exp(k)) * 100, 1))] = float(x)
        else:
            vals[("F", 30)] = fwd_at(day, 30)
            rows[date] = vals
    out = pd.DataFrame(rows).T
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    return out.sort_index()


BETAS = (0.0, 0.5, 1.0, 1.25, 1.5, 1.75, 2.0)
LEVELS = (0.5, 1.0, 1.5, 2.0)


def sticky(svi: pd.DataFrame, days: int = 30, ks=None, betas=BETAS, levels=LEVELS
           ) -> pd.DataFrame:
    """Day-to-day test of sticky rules on the `days` constant-maturity smile. The rule with
    multiplier beta says tomorrow's vol at k is today's at k + beta r (0 = sticky moneyness,
    1 = sticky strike). Per pair of consecutive trading days: the forward return r, the error of
    each rule (`err_<beta>`, vol points, root mean square over `ks`), the at-the-money change and
    today's at-the-money slope S = d(vol)/dk. Also `lvl_<b>`: the error of "same shape against
    the forward, whole smile shifted by b S r" (today's vol at k plus b S r)."""
    ks = np.linspace(-0.08, 0.08, 9) if ks is None else np.asarray(ks)
    by = {d: g for d, g in svi.groupby("date")}
    dates = sorted(by)
    rows = []
    for d0, d1 in zip(dates[:-1], dates[1:], strict=True):
        f0, f1 = fwd_at(by[d0], days), fwd_at(by[d1], days)
        if f0 is None or f1 is None:
            continue
        r = float(np.log(f1 / f0))
        v1 = vol_at(by[d1], ks, days)
        a0 = vol_at(by[d0], np.array([-0.005, 0.0, 0.005]), days)
        a1 = vol_at(by[d1], np.array([0.0]), days)
        if v1 is None or a0 is None or a1 is None:
            continue
        row = {"date": d1, "r": r, "d_atm": float(a1[0] - a0[1]),
               "slope": float((a0[2] - a0[0]) / 0.01), "atm": float(a0[1])}
        for b in betas:
            p0 = vol_at(by[d0], ks + b * r, days)
            row[f"err_{b}"] = 100 * float(np.sqrt(np.mean((v1 - p0) ** 2)))
        m0 = vol_at(by[d0], ks, days)
        for b in levels:
            p0 = m0 + b * row["slope"] * r
            row[f"lvl_{b}"] = 100 * float(np.sqrt(np.mean((v1 - p0) ** 2)))
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def pca(changes: pd.DataFrame, n: int = 3) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Shares of variance, loadings (columns of `changes` x n, sign set so each loading sums to a
    positive number) and component scores, from the covariance of `changes`."""
    x = changes.to_numpy() - changes.to_numpy().mean(axis=0)
    val, vec = np.linalg.eigh(np.cov(x, rowvar=False))
    order = np.argsort(val)[::-1]
    val, vec = val[order], vec[:, order]
    for j in range(vec.shape[1]):
        if vec[:, j].sum() < 0:
            vec[:, j] *= -1
    scores = pd.DataFrame(x @ vec[:, :n], index=changes.index,
                          columns=[f"PC{i + 1}" for i in range(n)])
    return val / val.sum(), vec[:, :n], scores
