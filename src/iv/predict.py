"""Do the surface's shapes predict NIFTY's next month, or rank stocks' next-month returns?

NIFTY, at each month-end (the last trading day of each month with a surface), from the
constant-maturity surface (`iv.dynamics.vol_at`):
- level: 30-day at-the-money vol; skew: minus its slope d(vol)/dk at k = 0 (higher = dearer puts);
- term: 60-day minus 30-day at-the-money vol (negative = inverted);
- premium: 30-day at-the-money variance minus the last 21 days' realised variance (annualised);
and India VIX. Outcomes over the next 21 trading days: log return, worst fall from the start,
realised volatility (annualised), and whether NIFTY fell over 5%.

Stocks, at each month-end in the stock file, from the slice nearest 30 days (`iv.stocks`):
skew (minus the at-the-money slope), risk reversal (25-delta call minus put vol), implied minus the
last 21 days' realised volatility, and implied vol. Outcome: the return to the next month-end
(split-adjusted close; a stock with no price then is skipped).
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from iv.dynamics import vol_at


def nifty_signals(svi: pd.DataFrame, close: pd.Series, vix: pd.Series | None = None,
                  h: int = 21) -> pd.DataFrame:
    r = np.log(close).diff().dropna()
    by = dict(tuple(svi.groupby("date")))
    dates = pd.DatetimeIndex(sorted(by))
    month_end = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max()
    pos = {d: i for i, d in enumerate(r.index)}
    rows = []
    for d in month_end:
        if d not in pos or pos[d] + h >= len(r) or pos[d] < h:
            continue
        a30 = vol_at(by[d], np.array([-0.005, 0.0, 0.005]), 30)
        a60 = vol_at(by[d], np.array([0.0]), 60)
        if a30 is None:
            continue
        i = pos[d]
        past = r.iloc[i - h + 1: i + 1]
        fut = r.iloc[i + 1: i + h + 1]
        path = np.exp(fut.cumsum())
        row = {"date": d, "level": float(a30[1]),
               "skew": -float((a30[2] - a30[0]) / 0.01),
               "term": float(a60[0] - a30[1]) if a60 is not None else np.nan,
               "premium": float(a30[1] ** 2 - (past**2).sum() * 252 / h),
               "ret": float(fut.sum()), "worst": float(min(path.min(), 1.0) - 1),
               "vol": float(np.sqrt((fut**2).sum() * 252 / h)),
               "fall5": bool(fut.sum() < np.log(0.95))}
        if vix is not None:
            v = vix.loc[:d].dropna()
            row["vix"] = float(v.iloc[-1]) if len(v) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def stock_signals(slices: pd.DataFrame, log_ret: pd.DataFrame, close: pd.DataFrame,
                  h: int = 21) -> pd.DataFrame:
    """One row per month-end and stock: the signals and the return to the next month-end."""
    one = (slices.assign(gap=(slices["days"] - 30).abs()).sort_values("gap")
           .groupby(["date", "symbol"]).head(1))
    ends = sorted(one["date"].unique())
    nxt = dict(zip(ends[:-1], ends[1:], strict=True))
    rv = np.sqrt((log_ret**2).rolling(h, min_periods=h - 2).mean() * 252)
    cl = close.ffill()
    rows = []
    for r in one.itertuples():
        if r.date not in nxt or r.symbol not in close or r.date not in cl.index:
            continue
        n = nxt[r.date]
        p0, p1 = cl.at[r.date, r.symbol], close[r.symbol].get(n, np.nan)
        rvol = rv.at[r.date, r.symbol] if r.date in rv.index else np.nan
        if not (np.isfinite(p0) and np.isfinite(p1) and np.isfinite(rvol)):
            continue
        rows.append({"date": r.date, "symbol": r.symbol, "skew": -r.slope,
                     "rr25": r.rr25, "gap": r.atm_vol - rvol, "level": r.atm_vol,
                     "ret": float(p1 / p0 - 1)})
    out = pd.DataFrame(rows)
    return out[out["rr25"].abs() < 1].reset_index(drop=True)


def rank_ic(d: pd.DataFrame, signal: str, target: str = "ret") -> pd.Series:
    """Spearman correlation of `signal` with `target` across stocks, per month-end."""
    return d.groupby("date").apply(
        lambda g: spearmanr(g[signal], g[target]).statistic if len(g) >= 20 else np.nan,
        include_groups=False).dropna()


def quintile_spread(d: pd.DataFrame, signal: str, target: str = "ret") -> pd.Series:
    """Per month-end: mean target of the top fifth by `signal` minus the bottom fifth."""
    def one(g):
        if len(g) < 20:
            return np.nan
        q = pd.qcut(g[signal].rank(method="first"), 5, labels=False)
        return g.loc[q == 4, target].mean() - g.loc[q == 0, target].mean()
    return d.groupby("date").apply(one, include_groups=False).dropna()


def bh_pass(pvals: dict[str, float], alpha: float = 0.05) -> list[str]:
    """Benjamini-Hochberg step-up: the names whose adjusted p-value is at most `alpha` (every
    hypothesis up to the largest rank i with p(i) <= alpha i / m)."""
    from scipy.stats import false_discovery_control

    names = list(pvals)
    adj = false_discovery_control([pvals[n] for n in names])
    return [n for n, a in zip(names, adj, strict=True) if a <= alpha]
