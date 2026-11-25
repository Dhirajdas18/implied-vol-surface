"""Core surface maths: forwards, Black-76 implied volatility, quote cleaning, SVI fits and the
no-arbitrage checks. numpy and scipy only.

Conventions
- Time to expiry T is calendar days / 365, from one settlement to another.
- Rates: a constant 6.5% a year (R), as in black-scholes-pricer, used only to discount option
  prices (Black-76) and in put-call parity. The forward itself comes from the option market, so
  no dividend estimate is needed.
- Log-moneyness k = ln(K / F). Total implied variance w = sigma^2 T.
- SVI is Gatheral's raw parameterisation, w(k) = a + b (rho (k - m) + sqrt((k - m)^2 + sigma^2)).
"""

import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares
from scipy.special import ndtr, ndtri

R = 0.065  # constant annual rate for discounting (black-scholes-pricer's choice)
TICK = 0.05  # NSE's minimum price step: a quote at the tick carries no volatility information
MIN_DAYS, MAX_DAYS = 2, 365
PARITY_STRIKES = 5  # forward: median over the nearest strikes with both a call and a put
MIN_PARITY = 3  # fewer pairs than this and the futures settlement price is used instead
LEE = 2.0  # Roger Lee's bound on the total-variance slope in each wing
PARAMS = ("a", "b", "rho", "m", "sigma")


# --- Black-76 ------------------------------------------------------------------------------------


def black76(F, K, T, sigma, kind="call", r=R):
    """Black-76 price of a European option on a forward F: e^(-rT) [F N(d1) - K N(d2)] for a call.
    `kind` is "call"/"CE" or "put"/"PE". Works on scalars or arrays."""
    F, K, T, sigma = (np.asarray(x, dtype=float) for x in (F, K, T, sigma))
    sd = sigma * np.sqrt(T)
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = np.log(F / K) / sd + sd / 2
    d2 = d1 - sd
    call = _is_call(kind)
    undisc = np.where(call, F * ndtr(d1) - K * ndtr(d2), K * ndtr(-d2) - F * ndtr(-d1))
    return np.exp(-r * T) * undisc


def _is_call(kind):
    k = np.asarray(kind)
    return np.isin(k, ["call", "CE", "C"])


def implied_vol(price, F, K, T, kind="call", r=R, lo=1e-4, hi=5.0, tol=1e-12, iters=100):
    """Black-76 implied volatility for scalars or arrays.

    A bracketed Newton solver: each step is Newton's on the price, falling back to bisection
    whenever Newton would leave the current bracket [lo, hi], so it always converges. NaN where
    no volatility in [lo, hi] matches the price (at or below intrinsic value, or above the
    forward or strike that bounds the option's value)."""
    price, F, K, T = np.broadcast_arrays(*(np.asarray(x, dtype=float) for x in (price, F, K, T)))
    kind = np.broadcast_to(np.asarray(kind), price.shape)
    scalar = price.ndim == 0
    price, F, K, T, kind = (np.atleast_1d(x).ravel() for x in (price, F, K, T, kind))
    target = price * np.exp(r * T)  # undiscounted
    call = _is_call(kind)
    intrinsic = np.where(call, np.maximum(F - K, 0), np.maximum(K - F, 0))
    upper = np.where(call, F, K)
    ok = (target > intrinsic) & (target < upper) & (T > 0) & np.isfinite(target)

    def undisc(sig):
        return black76(F, K, T, sig, kind, r=0.0)

    a = np.full(price.shape, lo)
    b = np.full(price.shape, hi)
    ok &= (undisc(a) <= target) & (undisc(b) >= target)
    x = np.full(price.shape, 0.2)
    sqrtT = np.sqrt(np.where(T > 0, T, 1.0))
    for _ in range(iters):
        f = undisc(x) - target
        a = np.where(f < 0, x, a)
        b = np.where(f >= 0, x, b)
        with np.errstate(divide="ignore", invalid="ignore"):
            d1 = np.log(F / K) / (x * sqrtT) + x * sqrtT / 2
            vega = F * np.exp(-0.5 * d1**2) / np.sqrt(2 * np.pi) * sqrtT
            step = x - f / vega
        bad = ~np.isfinite(step) | (step <= a) | (step >= b)
        new = np.where(bad, (a + b) / 2, step)
        done = np.abs(new - x) < tol
        x = new
        if np.all(done | ~ok):
            break
    out = np.where(ok, x, np.nan)
    return float(out[0]) if scalar else out


# --- Forwards ------------------------------------------------------------------------------------


def parity_forward(calls: pd.Series, puts: pd.Series, T: float, r: float = R,
                   ref: float | None = None, n: int = PARITY_STRIKES) -> tuple[float, int]:
    """Forward from put-call parity, F = K + e^(rT) (C - P), as the median over the `n` strikes
    nearest `ref` that have both a call and a put price (indexed by strike). Without `ref`, the
    strike where |C - P| is smallest stands in for the money. Returns (forward, pairs used);
    NaN and 0 when no strike has both."""
    both = pd.concat({"C": calls, "P": puts}, axis=1).dropna()
    if both.empty:
        return float("nan"), 0
    strikes = both.index.to_numpy(dtype=float)
    if ref is None or not np.isfinite(ref):
        ref = float(strikes[np.argmin(np.abs(both["C"] - both["P"]).to_numpy())])
    near = both.iloc[np.argsort(np.abs(strikes - ref), kind="stable")[:n]]
    f = near.index.to_numpy(dtype=float) + np.exp(r * T) * (near["C"] - near["P"]).to_numpy()
    return float(np.median(f)), len(near)


def forwards(day: pd.DataFrame, date, r: float = R) -> pd.DataFrame:
    """One day's forward for each expiry, from its option rows (expiry, strike, type, price,
    contracts, and fut_settle where a futures contract exists).

    Put-call parity on traded strikes nearest the money (`PARITY_STRIKES`, median) when at least
    `MIN_PARITY` strikes have a traded call and put, else the futures settlement price. Columns:
    expiry, T, F, source ("parity" or "futures"), pairs, fut_settle."""
    date = pd.Timestamp(date)
    rows = []
    traded = day[(day["contracts"] > 0) & (day["price"] > TICK)]
    for expiry, e in day.groupby("expiry"):
        T = (pd.Timestamp(expiry) - date).days / 365
        fut = e["fut_settle"].dropna()
        fut = float(fut.iloc[0]) if len(fut) else float("nan")
        t = traded[traded["expiry"] == expiry]
        calls = t[t["type"] == "CE"].groupby("strike")["price"].last()
        puts = t[t["type"] == "PE"].groupby("strike")["price"].last()
        F, pairs = parity_forward(calls, puts, T, r, ref=fut) if T > 0 else (float("nan"), 0)
        source = "parity"
        if pairs < MIN_PARITY:
            F, source = fut, "futures"
        rows.append({"expiry": pd.Timestamp(expiry), "T": T, "F": F, "source": source,
                     "pairs": pairs, "fut_settle": fut})
    out = pd.DataFrame(rows, columns=["expiry", "T", "F", "source", "pairs", "fut_settle"])
    return out[np.isfinite(out["F"])].reset_index(drop=True)


# --- Quotes --------------------------------------------------------------------------------------


def clean_quotes(day: pd.DataFrame, fwd: pd.DataFrame, r: float = R,
                 min_contracts: int = 1) -> pd.DataFrame:
    """The day's usable out-of-the-money quotes with implied volatility and total variance.

    Keeps options that traded (at least `min_contracts` contracts; an untraded row's close is
    stale), out of the money against the expiry's forward (puts below it, calls at or above),
    priced above the 0.05 tick and above intrinsic value, with 2 to 365 days to expiry.
    Columns: expiry, T, F, strike, type, price, contracts, oi, k, iv, w."""
    q = day.merge(fwd[["expiry", "T", "F"]], on="expiry", how="inner")
    q = q[(q["T"] >= MIN_DAYS / 365) & (q["T"] <= MAX_DAYS / 365)]
    q = q[(q["contracts"] >= min_contracts) & (q["price"] > TICK)]
    otm = ((q["type"] == "PE") & (q["strike"] < q["F"])) | (
        (q["type"] == "CE") & (q["strike"] >= q["F"]))
    q = q[otm].copy()
    if q.empty:
        return q.assign(k=[], iv=[], w=[])
    q["iv"] = implied_vol(q["price"].to_numpy(), q["F"].to_numpy(), q["strike"].to_numpy(),
                          q["T"].to_numpy(), q["type"].to_numpy(), r=r)
    q = q[np.isfinite(q["iv"])].copy()
    q["k"] = np.log(q["strike"] / q["F"])
    q["w"] = q["iv"] ** 2 * q["T"]
    cols = ["expiry", "T", "F", "strike", "type", "price", "contracts", "oi", "k", "iv", "w"]
    return q[cols].sort_values(["expiry", "k"]).reset_index(drop=True)


# --- SVI -----------------------------------------------------------------------------------------


def svi_w(p, k):
    """SVI total variance at log-moneyness k; `p` is a dict or sequence (a, b, rho, m, sigma)."""
    a, b, rho, m, s = _unpack(p)
    x = np.asarray(k, dtype=float) - m
    return a + b * (rho * x + np.sqrt(x * x + s * s))


def svi_iv(p, k, T):
    """Implied volatility from the SVI slice: sqrt(w / T)."""
    return np.sqrt(np.maximum(svi_w(p, k), 0.0) / T)


def _unpack(p):
    if isinstance(p, dict):
        return tuple(float(p[n]) for n in PARAMS)
    return tuple(float(v) for v in p)


def _from_slopes(x):
    """Fitting coordinates -> raw SVI. The fit works with v = a + b sigma sqrt(1 - rho^2) (the
    minimum total variance) and the wing slopes sL = b(1 - rho), sR = b(1 + rho), so plain box
    bounds give b >= 0, |rho| < 1, sigma > 0, a minimum variance >= 0 and Lee's wing bound."""
    v, sl, sr, m, s = x
    b = (sl + sr) / 2
    rho = (sr - sl) / (sr + sl)
    a = v - b * s * np.sqrt(1 - rho * rho)
    return np.array([a, b, rho, m, s])


def _to_slopes(p):
    a, b, rho, m, s = p
    return np.array([a + b * s * np.sqrt(1 - rho * rho), b * (1 - rho), b * (1 + rho), m, s])


def _quasi_explicit_starts(k, w, wt, n_best=3):
    """Starting points from Zeliade's quasi-explicit idea: for fixed (m, sigma), SVI is linear in
    (a, b rho sigma, b sigma) after substituting y = (k - m) / sigma, so a grid over (m, sigma)
    with a weighted linear least-squares solve at each point finds good regions cheaply."""
    span = max(k.max() - k.min(), 1e-3)
    ms = np.linspace(k.min(), k.max(), 15)
    ss = np.geomspace(max(span / 50, 1e-4), max(2 * span, 1e-3), 12)
    cands = []
    for m in ms:
        for s in ss:
            y = (k - m) / s
            A = np.column_stack([np.ones_like(y), y, np.sqrt(y * y + 1)]) * wt[:, None]
            coef, *_ = np.linalg.lstsq(A, w * wt, rcond=None)
            a, d, c = coef
            if c <= 0:
                continue
            b, rho = c / s, np.clip(d / c, -0.98, 0.98)
            p = np.array([a, b, rho, m, s])
            err = np.sum(((svi_w(p, k) - w) * wt) ** 2)
            cands.append((err, p))
    cands.sort(key=lambda t: t[0])
    return [p for _, p in cands[:n_best]]


def fit_svi(k, w, T, weights=None) -> dict:
    """Fit an SVI slice to total variances w at log-moneyness k for maturity T.

    Weighted least squares on w. Default weights are 1 / sqrt(w), which makes each residual
    proportional to the error in implied volatility. Constraints, as box bounds in the fitting
    coordinates of `_from_slopes`: b >= 0, |rho| < 1, sigma > 0, a + b sigma sqrt(1 - rho^2) >= 0,
    each wing slope b(1 +- rho) at most 2 (Lee), and the vertex m inside the quoted k range (a
    vertex beyond the quotes lets one wing run to the slope bound unseen). Several starts (the
    best quasi-explicit grid points and a generic one) are refined with scipy's bounded
    least_squares; the best wins.

    Returns a, b, rho, m, sigma, rmse (in volatility points, i.e. percent), n and ok."""
    k, w = np.asarray(k, dtype=float), np.asarray(w, dtype=float)
    wt = 1 / np.sqrt(np.maximum(w, 1e-12)) if weights is None else np.asarray(weights, float)
    wt = wt / wt.mean()
    lo = np.array([0.0, 1e-6, 1e-6, k.min(), 1e-4])
    hi = np.array([2 * w.max() + 1e-6, LEE, LEE, k.max(), 2.0])
    generic = np.array([w.min(), 0.1, -0.5, k[np.argmin(w)], 0.1])
    best = None
    for p0 in _quasi_explicit_starts(k, w, wt) + [generic]:
        p0 = np.array(p0, dtype=float)
        p0[1] = max(p0[1], 1e-5)
        p0[4] = max(p0[4], 1e-4)
        x0 = np.clip(_to_slopes(p0), lo + 1e-9, hi - 1e-9)
        try:
            res = least_squares(lambda x: (svi_w(_from_slopes(x), k) - w) * wt, x0,
                                bounds=(lo, hi), method="trf", x_scale="jac", max_nfev=2000)
        except ValueError:
            continue
        if best is None or res.cost < best.cost:
            best = res
    if best is None:
        return dict.fromkeys(PARAMS, float("nan")) | {"rmse": float("nan"), "n": len(k),
                                                    "ok": False}
    p = _from_slopes(best.x)
    iv_fit = svi_iv(p, k, T)
    rmse = 100 * float(np.sqrt(np.mean((iv_fit - np.sqrt(w / T)) ** 2)))
    return dict(zip(PARAMS, map(float, p), strict=True)) | {"rmse": rmse, "n": len(k),
                                                            "ok": bool(best.success)}


# --- No-arbitrage checks -------------------------------------------------------------------------


def svi_g(p, k):
    """Gatheral's butterfly function g(k) for an SVI slice. The implied density is
    g(k) / sqrt(2 pi w) exp(-d2^2 / 2), so g < 0 anywhere means butterfly arbitrage there.
    g = (1 - k w' / (2w))^2 - (w'^2 / 4)(1/w + 1/4) + w'' / 2."""
    a, b, rho, m, s = _unpack(p)
    k = np.asarray(k, dtype=float)
    x = k - m
    root = np.sqrt(x * x + s * s)
    w = a + b * (rho * x + root)
    w1 = b * (rho + x / root)
    w2 = b * s * s / root**3
    with np.errstate(divide="ignore", invalid="ignore"):
        return (1 - k * w1 / (2 * w)) ** 2 - w1 * w1 / 4 * (1 / w + 0.25) + w2 / 2


def butterfly_check(p, k_grid=None, tol=1e-10) -> dict:
    """Butterfly arbitrage in one SVI slice: g(k) on `k_grid` (default 401 points on [-1, 1]).
    Returns violated (g < -tol, or w <= 0, somewhere), min_g and amount (the area of g below 0)."""
    k = np.linspace(-1, 1, 401) if k_grid is None else np.asarray(k_grid, dtype=float)
    g = svi_g(p, k)
    w = svi_w(p, k)
    bad = (w <= 0) | ~np.isfinite(g)
    g = np.where(bad, -np.inf, g)
    neg = np.clip(-np.where(np.isfinite(g), g, 0.0), 0, None)
    amount = float(np.trapezoid(neg, k)) if len(k) > 1 else float(neg.sum())
    return {"violated": bool((g < -tol).any()), "min_g": float(g.min()), "amount": amount}


def calendar_check(params, T, k_lo=None, k_hi=None, n: int = 41, tol=1e-10) -> pd.DataFrame:
    """Calendar arbitrage across one day's slices: total variance must not fall as T grows at any
    fixed k. Slices are compared in maturity order, each against the one before it, on a grid
    over the k range both have quotes for (from `k_lo`, `k_hi`), or [-0.2, 0.2] if not given.

    Returns one row per slice, in the input order: violated (it dips below the shorter slice)
    and amount (the largest dip, in total variance). The shortest slice is never flagged."""
    T = np.asarray(T, dtype=float)
    order = np.argsort(T, kind="stable")
    viol = np.zeros(len(T), dtype=bool)
    amount = np.zeros(len(T))
    for i, j in zip(order[:-1], order[1:], strict=True):
        lo = -0.2 if k_lo is None else max(k_lo[i], k_lo[j])
        hi = 0.2 if k_hi is None else min(k_hi[i], k_hi[j])
        if hi <= lo:
            continue
        k = np.linspace(lo, hi, n)
        dip = float(np.max(svi_w(params[i], k) - svi_w(params[j], k)))
        if T[j] > T[i] and dip > tol:
            viol[j], amount[j] = True, dip
    return pd.DataFrame({"violated": viol, "amount": amount})


# --- Slice summaries -----------------------------------------------------------------------------


def atm_vol(p, T) -> float:
    """At-the-money-forward implied volatility (k = 0)."""
    return float(svi_iv(p, 0.0, T))


def skew(p, T, k=0.1) -> float:
    """Implied volatility at k = -0.1 minus at k = +0.1 (positive for a downward skew)."""
    return float(svi_iv(p, -k, T) - svi_iv(p, k, T))


def delta_strike(p, delta: float) -> float:
    """Log-moneyness of the call with undiscounted Black-76 delta N(d1) = `delta` on the slice,
    where d1 = -k / sqrt(w) + sqrt(w) / 2. A 25-delta put is the call with delta 0.75. The search
    widens from [-2, 2] to [-20, 20] for steep wings; NaN if nothing matches."""
    z = float(ndtri(delta))

    def f(k):
        w = max(float(svi_w(p, k)), 1e-12)
        return -k / np.sqrt(w) + np.sqrt(w) / 2 - z

    for lim in (2.0, 6.0, 20.0):
        if f(-lim) > 0 > f(lim):
            return float(brentq(f, -lim, lim, xtol=1e-12))
    return float("nan")


def risk_reversal(p, T, delta: float = 0.25) -> float:
    """25-delta risk reversal: the call's implied volatility minus the put's (negative when puts
    are dearer)."""
    kc, kp = delta_strike(p, delta), delta_strike(p, 1 - delta)
    return float(svi_iv(p, kc, T) - svi_iv(p, kp, T))
