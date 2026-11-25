"""Build the daily SVI surface from the downloaded option rows, and the page's compact aggregate.

For each trading day: forwards per expiry, the cleaned out-of-the-money quotes, an SVI fit for
every expiry with at least MIN_QUOTES quotes, the butterfly check on each fitted slice over its
quoted range, the calendar check across the day's slices, and summary numbers (at-the-money
volatility, the skew at k = +-0.1, the 25-delta risk reversal).

Writes data/derived/svi_daily.parquet (not committed) and docs/data/lab_surfaces.json (committed:
SVI parameters and summaries, plus cleaned quotes as (k, iv) for the last QUOTE_DAYS days; derived
numbers only, no raw NSE rows).
"""

import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from iv import surface as s
from iv.data import LAB_JSON, NIFTY_OPTIONS, SVI_DAILY

MIN_QUOTES = 5
LAB_DAYS = 250  # every trading day for about the last year
QUOTE_DAYS = 20  # market quotes on the page for the last 20 days only
OPTION_COLS = ["date", "expiry", "strike", "type", "price", "contracts", "oi", "fut_settle"]


def fit_day(date, day: pd.DataFrame, keep_quotes: bool = False):
    """One day's slices (a list of dicts) and, if `keep_quotes`, its cleaned quotes."""
    date = pd.Timestamp(date)
    fwd = s.forwards(day, date)
    q = s.clean_quotes(day, fwd)
    rows = []
    for expiry, e in q.groupby("expiry"):
        if len(e) < MIN_QUOTES:
            continue
        T = float(e["T"].iloc[0])
        k, w = e["k"].to_numpy(), e["w"].to_numpy()
        fit = s.fit_svi(k, w, T)
        if not np.isfinite(fit["a"]):
            continue
        p = {n: fit[n] for n in s.PARAMS}
        f = fwd[fwd["expiry"] == expiry].iloc[0]
        bf = s.butterfly_check(p, np.linspace(k.min(), k.max(), 201))
        rows.append({
            "date": date, "expiry": pd.Timestamp(expiry), "days": (expiry - date).days, "T": T,
            "F": float(f["F"]), "source": f["source"], "pairs": int(f["pairs"]),
            "fut_settle": float(f["fut_settle"]), "monthly": bool(np.isfinite(f["fut_settle"])),
            **p, "rmse": fit["rmse"], "n": fit["n"], "converged": fit["ok"],
            "k_lo": float(k.min()), "k_hi": float(k.max()),
            "butterfly": bf["violated"], "butterfly_min_g": bf["min_g"],
            "butterfly_amount": bf["amount"],
            "atm_vol": s.atm_vol(p, T), "skew10": s.skew(p, T), "rr25": s.risk_reversal(p, T),
        })
    if rows:
        cal = s.calendar_check([{n: r[n] for n in s.PARAMS} for r in rows],
                               [r["T"] for r in rows], [r["k_lo"] for r in rows],
                               [r["k_hi"] for r in rows])
        for r, v, a in zip(rows, cal["violated"], cal["amount"], strict=True):
            r["calendar"], r["calendar_amount"] = bool(v), float(a)
    return rows, (q if keep_quotes else None)


def _work(args):
    date, day, keep = args
    return date, *fit_day(date, day, keep)


def build(opts: pd.DataFrame, workers: int | None = None, quote_days: int = QUOTE_DAYS):
    """Fit every day in `opts` (NIFTY option rows). Returns (slices table, {date: quotes})."""
    dates = sorted(opts["date"].unique())
    keep = set(dates[-quote_days:])
    jobs = ((d, g, d in keep) for d, g in opts.groupby("date", sort=True))
    rows, quotes = [], {}
    workers = workers or max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        for date, r, q in ex.map(_work, jobs, chunksize=8):
            rows += r
            if q is not None:
                quotes[pd.Timestamp(date)] = q
    return pd.DataFrame(rows), quotes


def _g(x: float, digits: int = 6) -> float:
    return float(f"{x:.{digits}g}")


FIELDS = ["days", "F", "a", "b", "rho", "m", "sigma", "rmse", "n", "k_lo", "k_hi", "atm",
          "skew10", "rr25", "flags"]
FLAGS = {"butterfly": 1, "calendar": 2, "monthly": 4, "futures_forward": 8}


def lab_data(svi: pd.DataFrame, quotes: dict, days: int = LAB_DAYS) -> dict:
    """The page's aggregate: every slice of the last `days` trading days and of each month's last
    trading day back to the start, as compact rows (FIELDS), plus market quotes for the dates in
    `quotes`. Rounded to 6 significant figures (parameters) or 4 decimals (volatilities)."""
    dates = pd.DatetimeIndex(sorted(svi["date"].unique()))
    month_end = pd.Series(dates, index=dates).groupby([dates.year, dates.month]).max()
    pick = sorted(set(dates[-days:]) | set(month_end))
    by_date = {}
    sub = svi[svi["date"].isin(pick)].sort_values(["date", "days"])
    for d, g in sub.groupby("date"):
        rows = []
        for r in g.itertuples(index=False):
            flags = (FLAGS["butterfly"] * r.butterfly + FLAGS["calendar"] * r.calendar
                     + FLAGS["monthly"] * r.monthly
                     + FLAGS["futures_forward"] * (r.source == "futures"))
            rows.append([int(r.days), round(r.F, 2), _g(r.a), _g(r.b), _g(r.rho), _g(r.m),
                         _g(r.sigma), round(r.rmse, 3), int(r.n), round(r.k_lo, 4),
                         round(r.k_hi, 4), round(r.atm_vol, 4), round(r.skew10, 4),
                         round(r.rr25, 4) if np.isfinite(r.rr25) else None, int(flags)])
        by_date[f"{d:%Y-%m-%d}"] = rows
    q_out = {}
    for d, q in sorted(quotes.items()):
        key = f"{d:%Y-%m-%d}"
        if key not in by_date:
            continue
        fitted = {row[0] for row in by_date[key]}
        q = q.assign(days=(q["expiry"] - d).dt.days)
        q_out[key] = {str(n): [[round(x, 4) for x in g["k"]], [round(x, 4) for x in g["iv"]]]
                      for n, g in q.groupby("days") if n in fitted}
    keys = list(by_date)
    return {"fields": FIELDS, "flags": FLAGS, "dates": keys, "latest": keys[-1],
            "first": keys[0], "rate": s.R, "slices": by_date, "quotes": q_out}


def main(write_json: bool = True) -> None:
    t0 = time.perf_counter()
    opts = pd.read_parquet(NIFTY_OPTIONS, columns=OPTION_COLS)
    svi, quotes = build(opts)
    seconds = time.perf_counter() - t0
    SVI_DAILY.parent.mkdir(parents=True, exist_ok=True)
    svi.to_parquet(SVI_DAILY, index=False)
    print(f"{SVI_DAILY}: {len(svi):,} slices over {svi['date'].nunique()} days "
          f"in {seconds:.0f} s")
    if write_json:
        data = lab_data(svi, quotes)
        LAB_JSON.parent.mkdir(parents=True, exist_ok=True)
        LAB_JSON.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
        print(f"{LAB_JSON}: {LAB_JSON.stat().st_size / 1e6:.2f} MB, {len(data['dates'])} dates")


if __name__ == "__main__":
    main()
