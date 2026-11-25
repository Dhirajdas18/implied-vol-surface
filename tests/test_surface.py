import numpy as np
import pandas as pd
import pytest

from iv import surface as s

GOOD = {"a": 0.01, "b": 0.08, "rho": -0.6, "m": 0.02, "sigma": 0.12}
# Axel Vogt's slice from Gatheral and Jacquier (2014), "Arbitrage-free SVI volatility surfaces":
# a well-known SVI fit with butterfly arbitrage (g < 0 around k = 0.4 to 0.8).
VOGT = {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153}


@pytest.mark.parametrize("days", [3, 30, 300])
def test_implied_vol_round_trips_black76(days):
    # Out-of-the-money quotes, as the surface uses: puts below the forward, calls above.
    F, T = 22000.0, days / 365
    sig = np.linspace(0.06, 0.8, 38)
    K = F * np.exp(np.repeat([-0.08, -0.02, 0.0, 0.03, 0.1], len(sig)) * np.sqrt(T * 4))
    sig = np.tile(sig, 5)
    kind = np.where(K < F, "PE", "CE")
    price = s.black76(F, K, T, sig, kind)
    got = s.implied_vol(price, F, K, T, kind)
    assert got == pytest.approx(sig, abs=1e-9)
    assert s.implied_vol(float(price[7]), F, float(K[7]), T, str(kind[7])) == pytest.approx(sig[7])
    itm = s.black76(F, 21000.0, T, 0.2, "CE")  # in the money works too where time value is real
    assert s.implied_vol(itm, F, 21000.0, T, "CE") == pytest.approx(0.2, abs=1e-8)


def test_implied_vol_is_nan_below_intrinsic_or_above_the_bound():
    F, K, T = 100.0, 90.0, 0.25
    disc = np.exp(-s.R * T)
    assert np.isnan(s.implied_vol(9.0 * disc, F, K, T, "CE"))  # exactly intrinsic
    assert np.isnan(s.implied_vol(101.0 * disc, F, K, T, "CE"))  # above the forward
    assert np.isnan(s.implied_vol(-1.0, F, K, T, "PE"))
    assert np.isfinite(s.implied_vol(10.5 * disc, F, K, T, "CE"))


def test_forward_from_parity_on_synthetic_prices():
    F_true, T = 22150.0, 21 / 365
    K = np.arange(21000, 23300, 100.0)
    calls = pd.Series(s.black76(F_true, K, T, 0.15, "CE"), index=K)
    puts = pd.Series(s.black76(F_true, K, T, 0.15, "PE"), index=K)
    puts.iloc[0] += 50  # one bad far quote: the median over near strikes ignores it
    F, n = s.parity_forward(calls, puts, T, ref=22000)
    assert n == s.PARITY_STRIKES and F == pytest.approx(F_true, abs=1e-6)
    F2, _ = s.parity_forward(calls, puts, T)  # no reference: the |C - P| minimum stands in
    assert F2 == pytest.approx(F_true, abs=1e-6)


def _day(F_true=22150.0, T_days=21, fut=22160.0, traded=True):
    date = pd.Timestamp("2026-11-02")
    expiry = date + pd.Timedelta(days=T_days)
    T = T_days / 365
    K = np.arange(20000, 24300, 100.0)
    rows = []
    for kind in ("CE", "PE"):
        p = s.black76(F_true, K, T, s.svi_iv(GOOD, np.log(K / F_true), T), kind)
        rows.append(pd.DataFrame({"expiry": expiry, "strike": K, "type": kind, "price": p,
                                  "contracts": 10 if traded else 0, "oi": 100,
                                  "fut_settle": fut}))
    return date, pd.concat(rows, ignore_index=True)


def test_forwards_use_parity_and_fall_back_to_futures():
    date, day = _day()
    f = s.forwards(day, date)
    assert f["source"].iloc[0] == "parity" and f["F"].iloc[0] == pytest.approx(22150.0, abs=1e-6)
    date, day = _day(traded=False)
    f = s.forwards(day, date)
    assert f["source"].iloc[0] == "futures" and f["F"].iloc[0] == 22160.0


def test_clean_quotes_keeps_out_of_the_money_traded_quotes_and_recovers_variance():
    date, day = _day()
    day.loc[0, "price"] = 0.05  # at the tick: dropped
    q = s.clean_quotes(day, s.forwards(day, date))
    assert ((q["type"] == "PE") == (q["strike"] < q["F"])).all()
    assert (q["price"] > s.TICK).all()
    assert q["w"].to_numpy() == pytest.approx(s.svi_w(GOOD, q["k"].to_numpy()), rel=1e-7)
    date, day = _day(T_days=1)
    short = s.clean_quotes(day, s.forwards(day, date))
    assert short.empty  # under two days to expiry


def test_fit_svi_recovers_known_parameters():
    T = 0.1
    k = np.linspace(-0.25, 0.15, 30)
    fit = s.fit_svi(k, s.svi_w(GOOD, k), T)
    got = np.array([fit[n] for n in s.PARAMS])
    assert got == pytest.approx([GOOD[n] for n in s.PARAMS], abs=1e-5)
    assert fit["rmse"] < 1e-4 and fit["n"] == 30
    rng = np.random.default_rng(1)  # noisy quotes still fit to about the noise
    iv = s.svi_iv(GOOD, k, T) + rng.normal(0, 0.002, k.size)
    noisy = s.fit_svi(k, iv**2 * T, T)
    assert noisy["rmse"] < 0.3 and noisy["b"] >= 0 and abs(noisy["rho"]) < 1


def test_butterfly_check_flags_vogt_and_passes_a_good_slice():
    k = np.linspace(-1.5, 1.5, 601)
    bad = s.butterfly_check(VOGT, k)
    assert bad["violated"] and bad["min_g"] < 0 and bad["amount"] > 0
    good = s.butterfly_check(GOOD, k)
    assert not good["violated"] and good["amount"] == 0


def test_g_matches_a_finite_difference_formula():
    k, h = np.linspace(-0.5, 0.5, 11), 1e-4
    w = s.svi_w(GOOD, k)
    w1 = (s.svi_w(GOOD, k + h) - s.svi_w(GOOD, k - h)) / (2 * h)
    w2 = (s.svi_w(GOOD, k + h) - 2 * w + s.svi_w(GOOD, k - h)) / h**2
    g = (1 - k * w1 / (2 * w)) ** 2 - w1**2 / 4 * (1 / w + 0.25) + w2 / 2
    assert s.svi_g(GOOD, k) == pytest.approx(g, rel=1e-5)


def test_calendar_check_flags_a_slice_that_dips_below_a_shorter_one():
    short = GOOD
    longer = GOOD | {"a": GOOD["a"] + 0.05}
    crossing = GOOD | {"a": GOOD["a"] + 0.005, "rho": 0.2}  # lower on the put side
    ok = s.calendar_check([short, longer], [0.1, 0.3])
    assert not ok["violated"].any()
    bad = s.calendar_check([longer, crossing, short], [0.3, 0.2, 0.1])  # any input order
    assert bad["violated"].tolist() == [False, True, False]
    assert bad["amount"].iloc[1] > 0
    narrow = s.calendar_check([short, crossing], [0.1, 0.2], k_lo=[0.05, 0.05], k_hi=[0.2, 0.2])
    assert not narrow["violated"].any()  # only the call side is quoted, where it is above


def test_slice_summaries():
    T = 0.1
    assert s.atm_vol(GOOD, T) == pytest.approx(np.sqrt(s.svi_w(GOOD, 0.0) / T))
    assert s.skew(GOOD, T) > 0  # rho < 0: puts dearer
    rr = s.risk_reversal(GOOD, T)
    assert rr < 0
    kc = s.delta_strike(GOOD, 0.25)
    w = s.svi_w(GOOD, kc)
    assert s.black76(1.0, np.exp(kc), T, np.sqrt(w / T), r=0.0) > 0
    from scipy.special import ndtr

    assert ndtr(-kc / np.sqrt(w) + np.sqrt(w) / 2) == pytest.approx(0.25, abs=1e-10)
