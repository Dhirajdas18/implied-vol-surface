import numpy as np
import pandas as pd

from iv import dynamics as dy
from iv import surface as s

P = {"a": 0.002, "b": 0.03, "rho": -0.6, "m": 0.02, "sigma": 0.08}


def _day(date, F, p_by_days):
    return pd.DataFrame([{"date": date, "T": d / 365, "F": F, **p} for d, p in p_by_days.items()])


def test_vol_at_interpolates_total_variance_in_time():
    p2 = dict(P, a=0.006)
    day = _day(pd.Timestamp("2024-01-01"), 100, {20: P, 40: p2})
    k = np.array([-0.05, 0.0])
    w = 0.5 * s.svi_w(P, k) + 0.5 * s.svi_w(p2, k)
    assert np.allclose(dy.vol_at(day, k, 30), np.sqrt(w / (30 / 365)))
    assert dy.vol_at(day, k, 50) is None


def test_sticky_strike_world_is_recognised():
    # Vol depends on the strike only: sigma(K) = 0.2 - 0.5 ln(K / 100). Each day's 30-day smile is
    # fitted exactly by moving SVI's vertex with the forward (a flat-wing smile in k is linear).
    d0, d1 = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")
    ks = np.linspace(-0.08, 0.08, 9)

    def svi_for(F):
        # total variance from sigma(K) at T = 30 days, fitted by SVI on a dense k grid
        k = np.linspace(-0.12, 0.12, 49)
        T = 30 / 365
        vol = 0.2 - 0.5 * (k + np.log(F / 100))
        f = s.fit_svi(k, vol**2 * T, T)
        return {x: f[x] for x in s.PARAMS}

    rows = []
    for d, F in ((d0, 100.0), (d1, 102.0)):
        p = svi_for(F)
        rows.append(_day(d, F, {29: p, 31: p}))
    st = dy.sticky(pd.concat(rows), ks=ks, betas=(0.0, 1.0), levels=(1.0,))
    assert st["err_1.0"].iloc[0] < 0.05 < st["err_0.0"].iloc[0]


def test_pca_finds_a_common_level_move():
    rng = np.random.default_rng(0)
    level = rng.normal(size=500)
    x = pd.DataFrame(np.outer(level, np.ones(6)) + 0.1 * rng.normal(size=(500, 6)))
    share, load, scores = dy.pca(x)
    assert share[0] > 0.95
    assert np.allclose(load[:, 0], 1 / np.sqrt(6), atol=0.02)
