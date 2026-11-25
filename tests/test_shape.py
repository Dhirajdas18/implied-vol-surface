import numpy as np
import pandas as pd

from iv import surface as s
from iv.shape import atm_slope, holdout, term_points

P = {"a": 0.002, "b": 0.03, "rho": -0.6, "m": 0.02, "sigma": 0.08}


def test_holdout_svi_recovers_an_exact_svi_smile_beyond_the_quotes():
    T = 0.1
    k = np.linspace(-0.2, 0.12, 20)
    w = s.svi_w(P, k)
    e = holdout(k, w, T)
    assert e["wings_svi"] < 0.05  # vol points
    assert e["wings_svi"] < e["wings_linear"]
    assert e["between_svi"] < 0.05


def test_holdout_linear_is_exact_between_quotes_on_a_straight_smile():
    T = 0.25
    k = np.linspace(-0.2, 0.2, 15)
    iv = 0.15 - 0.2 * k
    e = holdout(k, iv**2 * T, T)
    assert e["between_linear"] < 1e-9


def test_atm_slope_matches_a_finite_difference():
    T, h = 0.2, 1e-6
    fd = (s.svi_iv(P, h, T) - s.svi_iv(P, -h, T)) / (2 * h)
    assert np.isclose(atm_slope(P, T), fd, rtol=1e-6)


def test_term_points_interpolate_total_variance_and_need_both_ends():
    day = pd.DataFrame({"days": [20, 120], "T": [20 / 365, 120 / 365], "atm_vol": [0.2, 0.1]})
    out = term_points(day)
    w30 = np.interp(30 / 365, day["T"], day["atm_vol"] ** 2 * day["T"])
    assert np.isclose(out[30], np.sqrt(w30 / (30 / 365)))
    assert term_points(day[day["days"] > 25]) is None
