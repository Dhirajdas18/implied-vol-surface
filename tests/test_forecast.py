import numpy as np
import pandas as pd

from iv import forecast as fc


def test_varswap_of_a_flat_smile_is_its_variance():
    w = 0.2**2 * 0.1
    flat = {"a": w, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    assert np.isclose(fc.varswap_w(flat), w, rtol=1e-3)


def test_varswap_counts_only_the_given_range():
    w = 0.2**2 * 0.1
    flat = {"a": w, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    assert fc.varswap_w(flat, -0.02, 0.02) < fc.varswap_w(flat, -0.3, 0.3)


def test_implied_at_interpolates_total_variance():
    day = pd.DataFrame({"T": [10 / 365, 50 / 365], "w": [0.001, 0.006]})
    v = fc.implied_at(day, 30, "w")
    assert np.isclose(v * 30 / 365, 0.0035)
    assert fc.implied_at(day, 60, "w") is None


def test_qlike_is_zero_only_for_a_perfect_forecast():
    assert fc.qlike([2.0], [2.0])[0] == 0
    assert fc.qlike([2.0], [1.0])[0] > 0 and fc.qlike([2.0], [4.0])[0] > 0


def test_newey_west_t_matches_the_plain_t_without_autocorrelation():
    x = np.random.default_rng(1).normal(0.1, 1, 5000)
    plain = x.mean() / (x.std() / np.sqrt(len(x)))
    assert abs(fc.nw_t(x, 0) - plain) < 0.01


def test_garch_paths_add_up_daily_forecasts_and_use_only_past_years():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2012-01-02", "2015-12-31")
    r = pd.Series(rng.standard_t(6, len(idx)), index=idx)
    g = fc.garch_paths(r, horizons=(1, 5), years=range(2015, 2016))
    assert g.index.min() < pd.Timestamp("2015-01-01") <= g.index[1]
    assert (g[5] > g[1]).all() and (g[1] > 0).all()
