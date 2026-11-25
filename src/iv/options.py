"""NSE's daily F&O bhavcopy (nsearchives.nseindia.com): download and parse.

Each trading day has one zipped CSV listing every derivative contract. NSE changed the format on
8 July 2024 (the UDiFF format); both are read into the same columns. The download and parsing follow
black-scholes-pricer's bsp.options, extended to stock options and futures.

Options carry the closing price (the day's last trades) as `price`, as in black-scholes-pricer,
and the settlement price as `settle`. Futures use the settlement price. Raw NSE data is saved under
data/ and never committed.
"""

import io
import time
import urllib.error
import urllib.request
import zipfile
from datetime import date

import pandas as pd

BASE = "https://nsearchives.nseindia.com"
UDIFF_START = date(2024, 7, 8)  # NSE switched bhavcopy formats on this date
UA = "implied-vol-surface/0.1 (research)"

# Instrument codes in the two formats -> our kind
KINDS_OLD = {"OPTIDX": "OPTIDX", "FUTIDX": "FUTIDX", "OPTSTK": "OPTSTK", "FUTSTK": "FUTSTK"}
KINDS_UDIFF = {"IDO": "OPTIDX", "IDF": "FUTIDX", "STO": "OPTSTK", "STF": "FUTSTK"}
COLUMNS = ["date", "symbol", "kind", "expiry", "strike", "type", "price", "settle",
           "contracts", "oi"]


def url_fo(d: date) -> str:
    if d >= UDIFF_START:
        return f"{BASE}/content/fo/BhavCopy_NSE_FO_0_0_0_{d:%Y%m%d}_F_0000.csv.zip"
    mon = d.strftime("%b").upper()
    return f"{BASE}/content/historical/DERIVATIVES/{d:%Y}/{mon}/fo{d:%d}{mon}{d:%Y}bhav.csv.zip"


def fetch(d: date, retries: int = 3, pause: float = 1.0) -> tuple[bytes | None, str]:
    """The day's zipped bhavcopy and a status: "ok", "missing" (404: no file, e.g. a holiday) or
    "error: ..." (blocked, rate-limited or network trouble after `retries` tries, with a growing
    pause between them, as bsp does)."""
    req = urllib.request.Request(url_fo(d), headers={"User-Agent": UA})
    status = "error"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                raw = r.read()
            if raw.startswith(b"PK"):
                return raw, "ok"
            status = "error: not a zip"
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None, "missing"
            status = f"error: HTTP {e.code}"
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            status = f"error: {e}"
        time.sleep(pause * (attempt + 1))
    return None, status


def parse_fo(d: date, raw: bytes) -> pd.DataFrame | None:
    """Every index and stock option and future in the file, in COLUMNS.

    `type` is CE / PE for options and FUT for futures; futures have no strike. `price` is the close
    for options and the settlement price for futures."""
    if not raw or not raw.startswith(b"PK"):
        return None
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        df = pd.read_csv(z.open(z.namelist()[0]), low_memory=False)
    df.columns = df.columns.str.strip()
    if "FinInstrmTp" in df.columns:
        n = pd.DataFrame({
            "symbol": df["TckrSymb"],
            "kind": df["FinInstrmTp"].astype(str).str.strip().map(KINDS_UDIFF),
            "expiry": pd.to_datetime(df["XpryDt"].astype(str).str.strip(), format="%Y-%m-%d"),
            "strike": df["StrkPric"], "opt": df["OptnTp"], "close": df["ClsPric"],
            "settle": df["SttlmPric"], "contracts": df["TtlTradgVol"], "oi": df["OpnIntrst"]})
    else:
        n = pd.DataFrame({
            "symbol": df["SYMBOL"],
            "kind": df["INSTRUMENT"].astype(str).str.strip().map(KINDS_OLD),
            "expiry": pd.to_datetime(df["EXPIRY_DT"].astype(str).str.strip(), format="%d-%b-%Y"),
            "strike": df["STRIKE_PR"], "opt": df["OPTION_TYP"], "close": df["CLOSE"],
            "settle": df["SETTLE_PR"], "contracts": df["CONTRACTS"], "oi": df["OPEN_INT"]})
    n = n[n["kind"].notna()].copy()
    n["symbol"] = n["symbol"].astype(str).str.strip()
    for c in ("strike", "close", "settle", "contracts", "oi"):
        n[c] = pd.to_numeric(n[c], errors="coerce")
    fut = n["kind"].str.startswith("FUT")
    n["type"] = n["opt"].astype(str).str.strip().where(~fut, "FUT")
    n.loc[fut, "strike"] = float("nan")
    n["price"] = n["close"].where(~fut, n["settle"])
    n["date"] = pd.Timestamp(d)
    return n[COLUMNS].reset_index(drop=True)


def split_day(rows: pd.DataFrame, month_end: bool) -> dict[str, pd.DataFrame]:
    """One parsed day -> the three stored tables.

    nifty: NIFTY index options, with `fut_settle`, the NIFTY futures settlement price for the same
    expiry (monthly expiries only; NaN for weeklies). futures: NIFTY futures. stocks (only when
    `month_end`): every stock option and stock future."""
    nifty = rows["symbol"] == "NIFTY"
    fut = rows[nifty & (rows["kind"] == "FUTIDX")]
    fut = fut[["date", "expiry", "price", "contracts", "oi"]].rename(columns={"price": "settle"})
    opt = rows[nifty & (rows["kind"] == "OPTIDX")]
    opt = opt.merge(fut[["expiry", "settle"]].rename(columns={"settle": "fut_settle"}),
                    on="expiry", how="left")
    out = {
        "nifty": opt[["date", "expiry", "strike", "type", "price", "settle", "contracts", "oi",
                      "fut_settle"]].reset_index(drop=True),
        "futures": fut.reset_index(drop=True),
    }
    if month_end:
        stk = rows[rows["kind"].isin(["OPTSTK", "FUTSTK"])]
        out["stocks"] = stk[["date", "symbol", "expiry", "strike", "type", "price", "settle",
                             "contracts", "oi"]].reset_index(drop=True)
    return out
