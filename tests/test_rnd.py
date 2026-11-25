import numpy as np
import pandas as pd

from iv import rnd
from iv import surface as s

P = {"a": 0.002, "b": 0.03, "rho": -0.6, "m": 0.02, "sigma": 0.08}


def test_density_integrates_to_one_and_prices_the_forward():
    k = np.linspace(-1.5, 1.5, 20001)
    f = rnd.pdf(P, k)
    assert np.isclose(np.trapezoid(f, k), 1, atol=1e-3)
    assert np.isclose(np.trapezoid(np.exp(k) * f, k), 1, atol=1e-3)


def test_cdf_is_the_integral_of_the_density():
    k = np.linspace(-1.5, 0.05, 20001)
    assert np.isclose(np.trapezoid(rnd.pdf(P, k), k), rnd.cdf(P, 0.05), atol=1e-3)


def test_cdf_matches_a_digital_put_from_black76_prices():
    T, F, K, h = 0.1, 100.0, 95.0, 1e-3
    vol = lambda K: s.svi_iv(P, np.log(K / F), T)  # noqa: E731
    put = lambda K: s.black76(F, K, T, vol(K), "put", r=0.0)  # noqa: E731
    digital = (put(K + h) - put(K - h)) / (2 * h)
    assert np.isclose(rnd.cdf(P, np.log(K / F)), digital, atol=1e-6)


def test_flat_smile_reduces_to_lognormal():
    w0 = 0.04 * 0.1
    k = np.array([-0.1, 0.0, 0.05])
    flat = {"a": w0, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1}
    assert np.allclose(rnd.cdf(flat, k), rnd.flat_cdf(w0, k))
    assert np.allclose(np.log(rnd.pdf(flat, k)), rnd.flat_logpdf(w0, k))


def test_origins_pick_one_slice_per_monthly_expiry_nearest_30_days():
    e = pd.Timestamp("2024-02-29")
    svi = pd.DataFrame({"date": e - pd.to_timedelta([40, 31, 25], "D"), "expiry": e,
                        "days": [40, 31, 25], "monthly": True, "converged": True})
    o = rnd.origins(svi)
    assert len(o) == 1 and o["days"].iloc[0] == 31
