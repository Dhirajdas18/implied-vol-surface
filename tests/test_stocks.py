import numpy as np
import pandas as pd

from iv import stocks
from iv import surface as s


def test_fit_date_recovers_a_flat_stock_smile_from_futures_and_options():
    date, expiry = pd.Timestamp("2024-01-31"), pd.Timestamp("2024-02-29")
    T, F, vol = (expiry - date).days / 365, 1000.0, 0.3
    rows = [{"date": date, "symbol": "ABC", "expiry": expiry, "strike": np.nan, "type": "FUT",
             "price": F, "settle": F, "contracts": 100, "oi": 1}]
    for K in np.arange(850, 1160, 20.0):
        for typ in ("CE", "PE"):
            p = float(s.black76(F, K, T, vol, "call" if typ == "CE" else "put"))
            rows.append({"date": date, "symbol": "ABC", "expiry": expiry, "strike": K,
                         "type": typ, "price": round(p, 2), "settle": p, "contracts": 50,
                         "oi": 1})
    out = stocks.fit_date((date, pd.DataFrame(rows)))
    assert len(out) == 1
    assert abs(out[0]["atm_vol"] - vol) < 0.002 and abs(out[0]["slope"]) < 0.05


def test_gjr_common_uses_no_data_from_the_year_it_forecasts():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2011-01-03", "2016-12-30")
    base = {f"S{i}": pd.Series(rng.standard_t(6, len(idx)), index=idx) for i in range(4)}
    a = stocks.gjr_common(base, years=range(2015, 2016))
    changed = {k: v.copy() for k, v in base.items()}
    for v in changed.values():
        v.loc["2016-01-01":] *= 5  # only after 2015
    b = stocks.gjr_common(changed, years=range(2015, 2016))
    m = a.merge(b, on=["date", "symbol"])
    assert len(m) > 0 and np.allclose(m["garch_x"], m["garch_y"])
    assert (a["garch"] > 0).all()
