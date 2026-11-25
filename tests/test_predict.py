import numpy as np
import pandas as pd

from iv import predict as pr


def _panel(months=3, n=30, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for m in range(months):
        sig = rng.normal(size=n)
        for i in range(n):
            rows.append({"date": pd.Timestamp(2024, m + 1, 28), "symbol": f"S{i}", "x": sig[i],
                         "ret": 0.01 * sig[i]})
    return pd.DataFrame(rows)


def test_rank_ic_and_quintile_spread_see_a_perfect_signal():
    d = _panel()
    assert np.allclose(pr.rank_ic(d, "x"), 1.0)
    qs = pr.quintile_spread(d, "x")
    assert (qs > 0).all() and len(qs) == 3


def test_stock_signals_pair_each_month_end_with_the_next():
    dates = pd.DatetimeIndex(["2024-01-31", "2024-02-29", "2024-03-28"])
    sl = pd.DataFrame({"date": dates, "symbol": "A", "days": 30, "slope": -0.1, "rr25": 0.0,
                       "atm_vol": 0.3})
    idx = pd.bdate_range("2023-11-01", "2024-03-29")
    close = pd.DataFrame({"A": np.linspace(100, 130, len(idx))}, index=idx)
    out = pr.stock_signals(sl, np.log(close).diff(), close)
    first = out[out["date"] == dates[0]].iloc[0]
    assert np.isclose(first["ret"], close.at[dates[1], "A"] / close.at[dates[0], "A"] - 1)
    assert dates[2] not in set(out["date"])  # no next month-end
