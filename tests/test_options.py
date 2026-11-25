# ruff: noqa: E501  (sample bhavcopy rows keep NSE's real column layout)
import io
import zipfile
from datetime import date

import pandas as pd

from iv.data import month_ends
from iv.options import parse_fo, split_day, url_fo


def _zip(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("bhav.csv", text)
    return buf.getvalue()


OLD = """INSTRUMENT,SYMBOL,EXPIRY_DT,STRIKE_PR,OPTION_TYP,OPEN,HIGH,LOW,CLOSE,SETTLE_PR,CONTRACTS,VAL_INLAKH,OPEN_INT,CHG_IN_OI,TIMESTAMP,
FUTIDX,NIFTY,29-Jan-2015,0,XX,1,1,1,8950.1,8952.35,100,1,1000,0,29-JAN-2015,
OPTIDX,NIFTY,29-Jan-2015,9000,CE,1,1,1,12.5,0,40,1,500,0,29-JAN-2015,
OPTIDX,NIFTY,26-Feb-2015,8500,PE,1,1,1,30.0,0,7,1,90,0,29-JAN-2015,
OPTIDX,BANKNIFTY,29-Jan-2015,20000,PE,1,1,1,5.0,0,7,1,90,0,29-JAN-2015,
OPTSTK,SBIN,29-Jan-2015,300,CE,1,1,1,4.0,0,3,1,30,0,29-JAN-2015,
FUTSTK,SBIN,29-Jan-2015,0,XX,1,1,1,310.0,310.5,3,1,30,0,29-JAN-2015,
"""
UDIFF = """TradDt,BizDt,Sgmt,Src,FinInstrmTp,FinInstrmId,ISIN,TckrSymb,SctySrs,XpryDt,FininstrmActlXpryDt,StrkPric,OptnTp,FinInstrmNm,OpnPric,HghPric,LwPric,ClsPric,LastPric,PrvsClsgPric,UndrlygPric,SttlmPric,OpnIntrst,ChngInOpnIntrst,TtlTradgVol,TtlTrfVal,TtlNbOfTxsExctd,SsnId,NewBrdLotQty,Rmks,Rsvd1,Rsvd2,Rsvd3,Rsvd4
2024-07-31,2024-07-31,FO,NSE,IDF,1,,NIFTY,,2024-08-29,2024-08-29,,,,1,1,1,24990.0,1,1,24951.15,24995.5,1000,0,100,1,1,F1,25,,,,,
2024-07-31,2024-07-31,FO,NSE,IDO,2,,NIFTY,,2024-08-29,2024-08-29,25500,CE,,1,1,1,80.25,1,1,24951.15,80.25,700,0,60,1,1,F1,25,,,,,
2024-07-31,2024-07-31,FO,NSE,IDO,3,,NIFTY,,2024-08-01,2024-08-01,24800,PE,,1,1,1,20.5,1,1,24951.15,20.5,700,0,60,1,1,F1,25,,,,,
2024-07-31,2024-07-31,FO,NSE,STO,4,,SBIN,,2024-08-29,2024-08-29,850,PE,,1,1,1,12.0,1,1,860,12.0,70,0,6,1,1,F1,750,,,,,
"""


def test_old_format_parses_and_splits():
    rows = parse_fo(date(2015, 1, 29), _zip(OLD))
    assert set(rows["kind"]) == {"FUTIDX", "OPTIDX", "OPTSTK", "FUTSTK"}
    parts = split_day(rows, month_end=False)
    n = parts["nifty"]
    assert list(n["strike"]) == [9000, 8500] and list(n["type"]) == ["CE", "PE"]
    assert n["price"].tolist() == [12.5, 30.0]  # options: the close
    assert n["fut_settle"].iloc[0] == 8952.35 and pd.isna(n["fut_settle"].iloc[1])
    assert parts["futures"]["settle"].tolist() == [8952.35]  # futures: the settlement price
    assert "stocks" not in parts
    stocks = split_day(rows, month_end=True)["stocks"]
    assert set(stocks["symbol"]) == {"SBIN"} and set(stocks["type"]) == {"CE", "FUT"}


def test_udiff_format_parses_to_the_same_columns():
    rows = parse_fo(date(2024, 7, 31), _zip(UDIFF))
    parts = split_day(rows, month_end=True)
    n = parts["nifty"].sort_values("strike")
    assert n["type"].tolist() == ["PE", "CE"] and n["contracts"].tolist() == [60, 60]
    assert n["fut_settle"].isna().tolist() == [True, False]
    assert parts["stocks"]["strike"].tolist() == [850]


def test_urls_switch_format_on_8_july_2024():
    assert url_fo(date(2024, 7, 5)).endswith("fo05JUL2024bhav.csv.zip")
    assert url_fo(date(2024, 7, 8)).endswith("BhavCopy_NSE_FO_0_0_0_20240708_F_0000.csv.zip")
    assert parse_fo(date(2024, 7, 8), b"<html>") is None


def test_month_ends_take_each_months_last_trading_day():
    days = pd.DatetimeIndex(["2015-01-29", "2015-01-30", "2015-02-02", "2015-02-27"])
    assert month_ends(days) == {pd.Timestamp("2015-01-30"), pd.Timestamp("2015-02-27")}
