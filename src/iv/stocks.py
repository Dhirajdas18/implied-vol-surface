"""Stock option smiles at each month-end against NIFTY's, and stock implied volatility as a
forecast of the stock's own volatility.

Slices: on each month-end in the stock file, for every stock, the expiries 15-60 days out with
at least MIN_QUOTES traded out-of-the-money quotes get an SVI fit, with the forward from
put-call parity or, failing that, the stock future's settlement price (as for NIFTY). NSE stock
options are European.

Forecast baseline, garch-volatility's choice for stocks: GJR with Student-t shocks, common
parameters (the median alpha, gamma, beta of every sound per-stock fit on data before January)
variance-targeted to each stock's own history, refitted each January.

Option symbols are mapped to the company's current NSE symbol (`current_symbols`, from NSE's
symbol-change list) before prices are looked up, since the price archive files a company's whole
history under its latest symbol.
"""

import numpy as np
import pandas as pd

from iv import surface as s
from iv.shape import atm_slope

MIN_QUOTES = 5
DAYS = (15, 60)


def current_symbols(symbol_change_csv) -> dict[str, str]:
    """Old symbol -> the latest symbol, following chains (TATATEA -> TATAGLOBAL -> TATACONSUM),
    from NSE's symbol-change list (columns OLD SYMBOL, NEW SYMBOL, EFFECTIVE DATE)."""
    ch = pd.read_csv(symbol_change_csv)
    ch.columns = [c.strip().upper() for c in ch.columns]
    ch["date"] = pd.to_datetime(ch["EFFECTIVE DATE"].str.strip(), format="%d-%b-%Y",
                                errors="coerce")
    step = {}
    for r in ch.sort_values("date").itertuples(index=False):
        old, new = str(r[1]).strip(), str(r[2]).strip()
        if old and new and old != new:
            step[old] = new
    out = {}
    for old in step:
        cur, seen = old, set()
        while cur in step and cur not in seen:
            seen.add(cur)
            cur = step[cur]
        out[old] = cur
    return out


def map_symbols(slices: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Slices with each symbol replaced by its current one; where an old and a new symbol trade
    on the same date and expiry (around a change), the slice with more contracts is kept."""
    out = slices.assign(symbol=slices["symbol"].replace(mapping))
    return (out.sort_values("contracts", ascending=False)
            .drop_duplicates(["date", "symbol", "expiry"]).sort_index())


def fit_date(args) -> list[dict]:
    """Every stock's slices on one date. `rows` = that date's stock option and future rows."""
    date, rows = args
    date = pd.Timestamp(date)
    fut = (rows[rows["type"] == "FUT"].groupby(["symbol", "expiry"])["settle"].first()
           .rename("fut_settle"))
    opt = rows[rows["type"].isin(["CE", "PE"])].join(fut, on=["symbol", "expiry"])
    out = []
    for sym, day in opt.groupby("symbol"):
        day = day[(day["expiry"] - date).dt.days.between(*DAYS)]
        if day.empty:
            continue
        fwd = s.forwards(day, date)
        q = s.clean_quotes(day, fwd)
        for expiry, e in q.groupby("expiry"):
            if len(e) < MIN_QUOTES:
                continue
            T = float(e["T"].iloc[0])
            f = s.fit_svi(e["k"].to_numpy(), e["w"].to_numpy(), T)
            if not np.isfinite(f["a"]):
                continue
            p = {x: f[x] for x in s.PARAMS}
            k = e["k"].to_numpy()
            fw = fwd[fwd["expiry"] == expiry].iloc[0]
            out.append({
                "date": date, "symbol": sym, "expiry": pd.Timestamp(expiry),
                "days": (pd.Timestamp(expiry) - date).days, "T": T, "F": float(fw["F"]),
                "source": fw["source"], **p, "rmse": f["rmse"], "n": f["n"],
                "k_lo": float(k.min()), "k_hi": float(k.max()),
                "atm_vol": s.atm_vol(p, T), "slope": atm_slope(p, T),
                "rr25": s.risk_reversal(p, T),
                "butterfly": s.butterfly_check(p, np.linspace(k.min(), k.max(), 201))["violated"],
                "contracts": float(e["contracts"].sum()),
            })
    return out


def gjr_common(r_pct: dict[str, pd.Series], years=range(2015, 2027), min_obs: int = 750,
               horizon: int = 21) -> pd.DataFrame:
    """(date, symbol) -> GJR common-parameter forecast of the total variance (decimal) of the next
    `horizon` returns, for every day of each year Y, parameters set from data before Y."""
    from arch import arch_model

    rows = []
    for y in years:
        start = pd.Timestamp(y, 1, 1)
        fits = []
        for r in r_pct.values():
            a = r.loc[: start - pd.Timedelta(days=1)]
            if len(a) < min_obs:
                continue
            try:
                res = arch_model(a, mean="Constant", vol="GARCH", p=1, o=1, q=1,
                                 dist="t").fit(disp="off")
            except (ValueError, np.linalg.LinAlgError):
                continue
            pr = res.params
            al, ga, be, nu = pr["alpha[1]"], pr["gamma[1]"], pr["beta[1]"], pr["nu"]
            sound = be > 0.5 and al + ga / 2 + be < 0.999 and 2.1 < nu < 50
            if res.convergence_flag == 0 and sound:
                fits.append((al, ga, be))
        if not fits:
            continue
        al, ga, be = np.median(np.array(fits), axis=0)
        pers = al + ga / 2 + be
        for sym, r in r_pct.items():
            a = r.loc[: start - pd.Timedelta(days=1)]
            yr = r.loc[start: pd.Timestamp(y, 12, 31)]
            if len(a) < 250 or yr.empty:
                continue
            mu, v = float(a.mean()), float(a.var())
            om = v * (1 - pers)
            x = r.loc[: pd.Timestamp(y, 12, 31)].to_numpy() - mu
            s2 = np.empty(len(x) + 1)
            s2[0] = v
            for i, e in enumerate(x):
                s2[i + 1] = om + (al + ga * (e < 0)) * e * e + be * s2[i]
            # s2[i + 1] = variance of return i + 1 given data to i; forecast from each day's close
            # sum over the next `horizon` days of v + pers^j (next-day variance - v)
            total = horizon * v + (s2[1:] - v) * (pers ** np.arange(horizon)).sum()
            idx = r.loc[: pd.Timestamp(y, 12, 31)].index
            keep = idx >= start - pd.Timedelta(days=7)
            rows.append(pd.DataFrame({"date": idx[keep], "symbol": sym,
                                      "garch": total[keep] / 1e4}))
    return pd.concat(rows).drop_duplicates(["date", "symbol"], keep="last")


def forecast_table(slices: pd.DataFrame, log_ret: pd.DataFrame, garch: pd.DataFrame,
                   horizon: int = 21) -> pd.DataFrame:
    """One row per month-end and stock: realised total variance of the next `horizon` returns
    (at most 2 missing days; the sum is scaled up by horizon / days present) and forecasts of
    it: the slice closest to 30 days (at-the-money variance x calendar days spanned / 365),
    the 20-day and EWMA (0.94) variances x horizon, and the GJR common-parameter forecast.
    Needs 250 past returns."""
    one = (slices.assign(gap=(slices["days"] - 30).abs()).sort_values("gap")
           .groupby(["date", "symbol"]).head(1))
    idx = log_ret.index
    pos = {d: i for i, d in enumerate(idx)}
    ewma = (log_ret**2).ewm(alpha=0.06, adjust=False, ignore_na=True).mean()
    rows = []
    for r in one.itertuples():
        if r.date not in pos or r.symbol not in log_ret:
            continue
        i = pos[r.date]
        if i + horizon >= len(idx):
            continue
        fut = log_ret[r.symbol].iloc[i + 1: i + horizon + 1]
        past = log_ret[r.symbol].iloc[: i + 1].dropna()
        if fut.isna().sum() > 2 or len(past) < 250:
            continue
        D = (idx[i + horizon] - r.date).days
        rows.append({"date": r.date, "symbol": r.symbol,
                     "realised": float((fut**2).sum() * horizon / fut.notna().sum()),
                     "atm": r.atm_vol**2 * D / 365,
                     "hist20": float((past.iloc[-20:] ** 2).mean()) * horizon,
                     "ewma": float(ewma[r.symbol].iloc[i]) * horizon,
                     "contracts": r.contracts})
    out = pd.DataFrame(rows).merge(garch, on=["date", "symbol"], how="left")
    return out.dropna(subset=["garch"]).reset_index(drop=True)
