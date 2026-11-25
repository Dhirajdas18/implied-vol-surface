"""Alignment and statistics checks added after review."""

import numpy as np
import pandas as pd

from iv import forecast as fc
from iv import predict as pr
from iv import stocks


def test_bh_is_step_up():
    p = {"a": 0.01, "b": 0.03, "c": 0.035, "d": 0.045}
    assert pr.bh_pass(p) == ["a", "b", "c", "d"]  # rank-by-rank would miss b
    assert pr.bh_pass({"a": 0.034, "b": 0.040, "c": 0.156, "d": 0.382}) == []


def test_current_symbols_follow_chains_and_map_symbols_dedupes(tmp_path):
    f = tmp_path / "symbolchange.csv"
    f.write_text("NAME OF THE COMPANY,OLD SYMBOL,NEW SYMBOL,EFFECTIVE DATE\n"
                 "X,TATATEA,TATAGLOBAL,21-JUL-2010\nX,TATAGLOBAL,TATACONSUM,27-FEB-2020\n")
    m = stocks.current_symbols(f)
    assert m == {"TATATEA": "TATACONSUM", "TATAGLOBAL": "TATACONSUM"}
    d = pd.Timestamp("2020-02-28")
    sl = pd.DataFrame({"date": d, "symbol": ["TATAGLOBAL", "TATACONSUM"], "expiry": d,
                       "contracts": [5.0, 50.0]})
    out = stocks.map_symbols(sl, m)
    assert len(out) == 1 and out["contracts"].iloc[0] == 50


def _flat_svi(dates, days=(10, 40), vol=0.2):
    rows = []
    for d in dates:
        for n in days:
            T = n / 365
            rows.append({"date": d, "expiry": d + pd.Timedelta(days=n), "T": T, "F": 100.0,
                         "a": vol**2 * T, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1,
                         "k_lo": -0.1, "k_hi": 0.1, "converged": True})
    return pd.DataFrame(rows)


def test_forecast_realised_window_starts_the_day_after():
    idx = pd.bdate_range("2014-01-01", "2016-12-30")
    r = pd.Series(np.random.default_rng(5).normal(0, 0.01, len(idx)), index=idx)
    close = 100 * np.exp(r.cumsum())
    t = pd.Timestamp("2016-03-01")
    d = fc.build(_flat_svi([t], days=(3, 40)), close, horizons=(5,))
    row = d.iloc[0]
    lr = np.log(close).diff()
    i = lr.index.get_loc(t)
    assert np.isclose(row["realised"], (lr.iloc[i + 1: i + 6] ** 2).sum())
    assert row["D"] == (lr.index[i + 5] - t).days
    assert np.isclose(row["atm"], 0.2**2 * row["D"] / 365)


def test_nifty_signals_use_past_and_next_windows_only():
    idx = pd.bdate_range("2016-01-01", "2016-06-30")
    r = pd.Series(np.random.default_rng(6).normal(0, 0.01, len(idx)), index=idx)
    close = 100 * np.exp(r.cumsum())
    svi = _flat_svi(idx[idx.is_month_end | (idx == idx[-1])], days=(20, 70))
    n = pr.nifty_signals(svi, close)
    t = n.index[0]
    lr = np.log(close).diff().dropna()
    i = lr.index.get_loc(t)
    assert np.isclose(n.loc[t, "ret"], lr.iloc[i + 1: i + 22].sum())
    past = (lr.iloc[i - 20: i + 1] ** 2).sum() * 252 / 21
    assert np.isclose(n.loc[t, "premium"], 0.2**2 - past)


def test_gjr_common_h_step_sum_matches_arch():
    from arch import arch_model

    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2010-01-01", "2015-12-31")
    r = {f"S{i}": pd.Series(rng.standard_t(6, len(idx)), index=idx) for i in range(3)}
    g = stocks.gjr_common(r, years=range(2015, 2016))
    one = g[g["symbol"] == "S0"].set_index("date")["garch"]
    # rebuild the common parameters the same way and compare with arch's analytic forecast
    fits = []
    for x in r.values():
        a = x.loc[:"2014-12-31"]
        res = arch_model(a, vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")
        pr_ = res.params
        fits.append((pr_["alpha[1]"], pr_["gamma[1]"], pr_["beta[1]"]))
    al, ga, be = np.median(np.array(fits), axis=0)
    a = r["S0"].loc[:"2014-12-31"]
    mu, v = a.mean(), a.var()
    om = v * (1 - al - ga / 2 - be)
    am = arch_model(r["S0"], vol="GARCH", p=1, o=1, q=1, dist="t")
    fixed = am.fix([mu, om, al, ga, be, 6.0])
    t = pd.Timestamp("2015-06-30")
    f = fixed.forecast(horizon=21, start=t, reindex=False).variance.loc[t].sum() / 1e4
    assert np.isclose(one.loc[t], f, rtol=1e-6)
