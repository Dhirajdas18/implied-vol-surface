"""Download NSE's F&O bhavcopy for every trading day from 2015 into data/options/ (never committed).

Writes
  nifty_options.parquet  NIFTY index options, every strike and expiry, every day
  nifty_futures.parquet  NIFTY futures settlement prices per expiry, every day
  stock_options.parquet  stock options and futures, on each month's last trading day only
  fetch_log.csv          days NSE had no file for (404), so a rerun doesn't ask again

Resumable: days already in nifty_options.parquet (or logged as missing) are skipped. Polite: one
request every ~0.75 s plus download time. After repeated errors (blocked or rate-limited) it backs
off, and after MAX_ERRORS in a row it saves what it has and stops.

Usage: uv run python scripts/fetch_options.py [start] [end]
Set IV_LAKE_DIR=/path/to/lake to take the trading days from the local lake instead of
Yahoo.
"""

import sys
import time

import pandas as pd

from iv.data import (
    NIFTY_FUTURES,
    NIFTY_OPTIONS,
    OPTIONS_DIR,
    START,
    STOCK_OPTIONS,
    month_ends,
    trading_days,
)
from iv.options import fetch, parse_fo, split_day

LOG = OPTIONS_DIR / "fetch_log.csv"
PAUSE = 0.75  # seconds between requests
SAVE_EVERY = 100  # days
MAX_ERRORS = 5  # consecutive failed days before giving up
RECENT = 3  # a 404 this close to today may only mean the file isn't published yet


def _read(path):
    return [pd.read_parquet(path)] if path.exists() else []


def _save(tables):
    for path, frames in tables.items():
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df.to_parquet(path, index=False)
            tables[path] = [df]


def main(start: str = START, end: str | None = None) -> None:
    days, source = trading_days(start)
    ends = month_ends(days)  # from the full calendar, so a clipped `end` isn't taken as a month end
    if end is not None:
        days = days[days <= pd.Timestamp(end)]
    OPTIONS_DIR.mkdir(parents=True, exist_ok=True)
    tables = {NIFTY_OPTIONS: _read(NIFTY_OPTIONS), NIFTY_FUTURES: _read(NIFTY_FUTURES),
              STOCK_OPTIONS: _read(STOCK_OPTIONS)}
    done = set(tables[NIFTY_OPTIONS][0]["date"].unique()) if tables[NIFTY_OPTIONS] else set()
    log = pd.read_csv(LOG, parse_dates=["date"]) if LOG.exists() else pd.DataFrame(
        columns=["date", "status"])
    missing = set(log["date"])
    todo = [d for d in days if d not in done and d not in missing]
    print(f"{len(days)} trading days from {source}; {len(done)} done, {len(missing)} known "
          f"missing, {len(todo)} to fetch", flush=True)
    recent = pd.Timestamp.today().normalize() - pd.Timedelta(days=RECENT)
    errors, fetched, t0 = 0, 0, time.perf_counter()
    for i, d in enumerate(todo):
        raw, status = fetch(d.date())
        rows = parse_fo(d.date(), raw) if raw else None
        if rows is not None:
            parts = split_day(rows, d in ends)
            tables[NIFTY_OPTIONS].append(parts["nifty"])
            tables[NIFTY_FUTURES].append(parts["futures"])
            if "stocks" in parts:
                tables[STOCK_OPTIONS].append(parts["stocks"])
            errors, fetched = 0, fetched + 1
            status = f"{len(parts['nifty'])} NIFTY option rows"
        elif status == "missing":
            if d < recent:
                log.loc[len(log)] = [d, "missing"]
        else:
            errors += 1
            if errors >= MAX_ERRORS:
                print(f"{d:%Y-%m-%d}: {status}; {errors} failures in a row, stopping", flush=True)
                break
            wait = 30 * errors
            print(f"{d:%Y-%m-%d}: {status}; backing off {wait} s", flush=True)
            time.sleep(wait)
        if i % 25 == 0 or rows is None:
            print(f"[{i + 1}/{len(todo)}] {d:%Y-%m-%d}: {status} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
        if fetched and fetched % SAVE_EVERY == 0 and rows is not None:
            _save(tables)
            log.to_csv(LOG, index=False)
        time.sleep(PAUSE)
    _save(tables)
    log.to_csv(LOG, index=False)
    if tables[NIFTY_OPTIONS]:
        n = tables[NIFTY_OPTIONS][0]
        print(f"-> {NIFTY_OPTIONS}: {len(n):,} rows, {n['date'].nunique()} days "
              f"({n['date'].min():%Y-%m-%d} to {n['date'].max():%Y-%m-%d})")
    print(f"fetched {fetched} days this run in {time.perf_counter() - t0:.0f} s; "
          f"{len(log)} days logged missing")


if __name__ == "__main__":
    main(*sys.argv[1:])
