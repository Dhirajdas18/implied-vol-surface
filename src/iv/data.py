"""Paths and the NSE trading calendar.

Trading days come from NIFTY's daily history: the local data lake's market file when the
IV_LAKE_DIR environment variable points at it (NIFTY dates from 2013), otherwise Yahoo Finance's
^NSEI dates.
"""

import os
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OPTIONS_DIR = ROOT / "data" / "options"
NIFTY_OPTIONS = OPTIONS_DIR / "nifty_options.parquet"
NIFTY_FUTURES = OPTIONS_DIR / "nifty_futures.parquet"
STOCK_OPTIONS = OPTIONS_DIR / "stock_options.parquet"
SVI_DAILY = ROOT / "data" / "derived" / "svi_daily.parquet"
LAB_JSON = ROOT / "docs" / "data" / "lab_surfaces.json"
START = "2015-01-01"


def _lake_days(lake: Path) -> pd.DatetimeIndex:
    m = pd.read_parquet(lake / "market.parquet", columns=["date", "nifty"])
    m = m.dropna(subset=["nifty"])
    return pd.DatetimeIndex(pd.to_datetime(m["date"]).dt.normalize().unique()).sort_values()


def _yahoo_days(start: str, attempts: int = 4) -> pd.DatetimeIndex:
    import yfinance as yf

    for attempt in range(attempts):
        if attempt:
            time.sleep(5 * attempt)
        try:
            # threads=False: parallel downloads can lock yfinance's sqlite cache
            raw = yf.download("^NSEI", start=start, auto_adjust=False, progress=False,
                              threads=False)
        except Exception as exc:  # network or Yahoo error: retry
            print(f"  Yahoo attempt {attempt + 1} failed: {exc}")
            continue
        if len(raw):
            idx = pd.DatetimeIndex(raw.index.tz_localize(None)).normalize()
            return idx.unique().sort_values()
    raise RuntimeError("could not download ^NSEI dates from Yahoo")


def trading_days(start: str = START, end: str | None = None) -> tuple[pd.DatetimeIndex, str]:
    """NSE trading days from `start` to `end` (default: today) and where they came from."""
    lake = os.environ.get("IV_LAKE_DIR")
    if lake:
        days, source = _lake_days(Path(lake).expanduser()), f"lake ({lake})"
    else:
        days, source = _yahoo_days(start), "Yahoo ^NSEI"
    days = days[days >= pd.Timestamp(start)]
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    return days, source


def month_ends(days: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """The last trading day of each month in `days`, leaving out the current (unfinished) month."""
    s = pd.Series(days, index=days)
    last = s.groupby([days.year, days.month]).max()
    today = pd.Timestamp.today().normalize()
    return {d for d in last if (d.year, d.month) != (today.year, today.month)}
