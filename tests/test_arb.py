import numpy as np
import pandas as pd

from iv import surface as s
from iv.arb import _one_side, fit_svi_no_butterfly, quote_checks


def _calls(prices, strikes=(100, 110, 120, 130)):
    return pd.DataFrame({"strike": strikes, "price": prices, "contracts": [500] * len(strikes)})


def test_clean_black76_prices_break_nothing():
    K = np.arange(80, 125, 5.0)
    c = s.black76(100, K, 0.1, 0.2, "call")
    e = pd.DataFrame({"strike": K, "price": c, "contracts": 10})
    assert _one_side(e, 0.1, "CE", s.R) == []


def test_each_breach_is_caught_with_its_size():
    checks = {v["check"]: v["amount"] for v in _one_side(_calls([20, 22, 5, 1]), 0.1, "CE", s.R)}
    assert np.isclose(checks["order"], 2)  # the 110 call dearer than the 100 call
    bfly = _one_side(_calls([20, 8, 3, 1]), 0.1, "CE", s.R)  # convex: no butterfly breach
    assert all(v["check"] != "butterfly" for v in bfly)
    bfly = _one_side(_calls([20, 14, 5, 1]), 0.1, "CE", s.R)  # 14 > 12.5: middle too dear
    assert np.isclose([v["amount"] for v in bfly if v["check"] == "butterfly"][0], 1.5)
    spread = _one_side(_calls([30, 15, 5, 1]), 0.1, "CE", s.R)  # 15 > e^(-rT) 10
    assert any(v["check"] == "spread" for v in spread)


def test_calendar_breach_when_the_longer_expiry_is_cheaper():
    date = pd.Timestamp("2024-01-01")
    rows = []
    for days, vol in ((30, 0.2), (60, 0.1)):  # total variance falls: 0.2^2*30 > 0.1^2*60
        T = days / 365
        for K in np.arange(90, 111, 2.0):
            for typ in ("CE", "PE"):
                p = float(s.black76(100, K, T, vol, typ))
                rows.append({"expiry": date + pd.Timedelta(days=days), "strike": K, "type": typ,
                             "price": round(p, 2), "contracts": 1000, "oi": 1,
                             "fut_settle": np.nan, "date": date})
    v, tested = quote_checks(pd.DataFrame(rows), date)
    assert (v["check"] == "calendar").any()
    assert (tested["check"] == "calendar").any()


def test_penalty_refit_removes_butterfly_arbitrage():
    bad = {"a": -0.0001, "b": 0.08, "rho": -0.95, "m": 0.0, "sigma": 0.005}  # sharp, arbitrageable
    T = 7 / 365
    k = np.linspace(-0.08, 0.04, 25)
    assert s.butterfly_check(bad, k)["violated"]
    w = s.svi_w(bad, k)
    out = fit_svi_no_butterfly(k, w, T, bad)
    assert not s.butterfly_check(out, np.linspace(k.min(), k.max(), 201))["violated"]
