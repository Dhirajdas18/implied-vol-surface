"""README findings, one function per topic: `iv report <topic>` writes charts to out/."""

import os
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from iv import plots
from iv import surface as s

MATURITY_BINS = [0, 7, 35, 120, 400]
MATURITY_LABELS = ["up to 7 days", "8-35 days", "36-120 days", "over 120 days"]


def svi_table() -> pd.DataFrame:
    """The daily SVI table from `iv.build` (converged slices only)."""
    from iv.data import SVI_DAILY

    d = pd.read_parquet(SVI_DAILY)
    return d[d["converged"]].reset_index(drop=True)


def india_vix() -> pd.Series | None:
    """India VIX (as a fraction) from the local lake if IV_LAKE_DIR is set, else None."""
    lake = os.environ.get("IV_LAKE_DIR")
    if not lake:
        return None
    m = pd.read_parquet(Path(lake) / "market.parquet").set_index("date")
    return (m["vix"] / 100).dropna()


def report_surface(out_dir: Path) -> dict:
    """Building the surface, 2015-2026: forwards, what the cleaned quotes cover, how well SVI,
    a quadratic and straight lines predict held-out quotes, and the shape of the fitted surface
    (skew by maturity, the term structure, India VIX against 30-day at-the-money volatility)."""
    from iv.build import OPTION_COLS
    from iv.data import NIFTY_OPTIONS
    from iv.shape import atm_slope, holdout_day, term_points

    res = {}
    d = svi_table()
    both = d.dropna(subset=["fut_settle"])
    gap = (both["F"] / both["fut_settle"] - 1).abs() * 1e4
    res["forwards"] = (
        f"put-call parity for {(d['source'] == 'parity').mean():.1%} of slices (futures settle "
        f"otherwise); against the same expiry's futures settle the parity forward differed by a "
        f"median {gap.median():.1f} bp (90th percentile {gap.quantile(0.9):.1f} bp, "
        f"{len(both):,} monthly slices)"
    )
    yr = d.groupby(d["date"].dt.year)
    per_day = d.groupby("date").size()
    res["coverage"] = (
        f"{len(d):,} slices on {d['date'].nunique():,} days; expiries a day median "
        f"{per_day.groupby(per_day.index.year).median().iloc[0]:.0f} in 2015, "
        f"{per_day.groupby(per_day.index.year).median().iloc[-1]:.0f} in 2026; quotes a slice "
        f"median {yr['n'].median().iloc[0]:.0f} in 2015, {yr['n'].median().iloc[-1]:.0f} in 2026"
    )
    bins = pd.cut(d["days"], MATURITY_BINS, labels=MATURITY_LABELS)
    res["SVI fit error by maturity (median vol points)"] = ", ".join(
        f"{k} {v:.2f}" for k, v in d.groupby(bins, observed=True)["rmse"].median().items()
    )

    # Hold-out test on every trading day.
    opts = pd.read_parquet(NIFTY_OPTIONS, columns=OPTION_COLS)
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        rows = [r for rr in ex.map(holdout_day, opts.groupby("date"), chunksize=8) for r in rr]
    h = pd.DataFrame(rows)
    methods = ["svi", "quadratic", "linear"]
    names = {"svi": "SVI", "quadratic": "quadratic", "linear": "straight lines"}
    med = {sc: {m: h[f"{sc}_{m}"].median() for m in methods} for sc in ("between", "wings")}
    for sc, label in (("between", "between quotes"), ("wings", "beyond the end quotes")):
        cols = [f"{sc}_{m}" for m in methods]
        ok = h.dropna(subset=cols)
        best = ok[cols].idxmin(axis=1).str.replace(f"{sc}_", "").value_counts(normalize=True)
        res[f"held-out error {label} (median vol points, {len(ok):,} slices)"] = (
            ", ".join(f"{names[m]} {med[sc][m]:.2f}" for m in methods)
            + "; best for "
            + ", ".join(f"{names[m]} {best.get(m, 0):.0%}" for m in methods)
        )
        hb = pd.cut(ok["days"], MATURITY_BINS, labels=MATURITY_LABELS)
        tab = ok.groupby(hb, observed=True)[cols].median()
        res[f"  by maturity, {label}"] = "; ".join(
            f"{k}: " + ", ".join(f"{names[m]} {r[f'{sc}_{m}']:.2f}" for m in methods)
            for k, r in tab.iterrows()
        )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(2)
    colors = {"svi": plots.COLORS["SVI"], "quadratic": plots.SERIES[1], "linear": plots.MUTED}
    for j, m in enumerate(methods):
        v = [med["between"][m], med["wings"][m]]
        ax.bar(xs + (j - 1) * 0.26, v, 0.26, color=colors[m], label=names[m])
        for x, y in zip(xs + (j - 1) * 0.26, v, strict=True):
            ax.text(x, y + 0.03, f"{y:.2f}", ha="center", va="bottom", fontsize=9, color=plots.TEXT)
    ax.set_xticks(xs, ["a strike between quotes", "a strike beyond the quotes"])
    ax.set_ylabel("error on held-out quotes, vol points")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Between quotes, straight lines did best; beyond them, SVI made half the error")
    fig.tight_layout()
    plots.save(fig, out_dir, "holdout")
    plt.close(fig)

    # Skew by maturity.
    p = d[list(s.PARAMS)].to_numpy()
    d["slope"] = [
        atm_slope(dict(zip(s.PARAMS, r, strict=True)), t) for r, t in zip(p, d["T"], strict=True)
    ]
    x = d[(d["days"] >= 7) & (d["slope"] < 0)]
    expo = np.array(
        [
            np.polyfit(np.log(g["T"]), np.log(-g["slope"]), 1)[0]
            for _, g in x.groupby("date")
            if len(g) >= 3
        ]
    )
    sb = pd.cut(x["days"], [7, 14, 35, 70, 120, 200, 400])
    res["at-the-money skew (d vol / d log-moneyness)"] = (
        f"negative (puts dearer) for {(d['slope'] < 0).mean():.1%} of slices; median by maturity "
        + ", ".join(
            f"{iv.left + 1}-{iv.right} days {v:.2f}"
            for iv, v in x.groupby(sb, observed=True)["slope"].median().items()
        )
    )
    res["skew against maturity (log-log slope per day)"] = (
        f"median {np.median(expo):.2f} (middle half {np.percentile(expo, 25):.2f} to "
        f"{np.percentile(expo, 75):.2f}; {len(expo):,} days with 3+ slices of 7+ days)"
    )

    # Term structure.
    tp = {dt: term_points(g) for dt, g in d.groupby("date")}
    tp = pd.DataFrame({k: v for k, v in tp.items() if v}).T
    tp.columns = ["v30", "v90"]
    tp["slope"] = 100 * (tp["v90"] - tp["v30"])
    q = pd.qcut(tp["v30"], 3, labels=["calmest", "middle", "most volatile"])
    g = tp.groupby(q, observed=True)
    res["term structure (90-day minus 30-day at-the-money vol)"] = (
        f"{len(tp):,} days; median {tp['slope'].median():+.2f} points (10th-90th percentile "
        f"{tp['slope'].quantile(0.1):+.2f} to {tp['slope'].quantile(0.9):+.2f}); correlation with "
        f"the 30-day level {tp[['v30', 'slope']].corr().iloc[0, 1]:.2f}"
    )
    res["  by thirds of the 30-day level (cut on all days)"] = "; ".join(
        f"{k} (30-day {g['v30'].median()[k]:.1%}): slope {g['slope'].median()[k]:+.2f}, inverted "
        f"by over 1 point on {g['slope'].apply(lambda v: (v < -1).mean())[k]:.0%} of days"
        for k in g.groups
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.scatter(100 * tp["v30"], tp["slope"], s=9, color=plots.COLORS["SVI"], alpha=0.35, lw=0)
    edges = [8, 10, 12, 14, 16, 18, 20, 23, 27]  # thin above 27%: dots only
    lv = pd.cut(100 * tp["v30"], edges)
    mid = tp.groupby(lv, observed=True)["slope"].median()
    cnt = tp.groupby(lv, observed=True).size()
    mid = mid[cnt >= 30]
    xm = [iv.mid for iv in mid.index]
    ax.plot(xm, mid.to_numpy(), color=plots.SERIES[1], lw=2)
    plots.label_line_end(ax, xm[-1], mid.iloc[-1], "median", plots.SERIES[1])
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xlim(8, 42)
    ax.set_ylim(-8, 4)
    ax.set_xlabel(
        "30-day at-the-money implied vol, % (each dot a day; 2020's highest days off the chart)"
    )
    ax.set_ylabel("90-day minus 30-day vol, points")
    ax.set_title("The calmer the market, the more 3-month volatility sat above 1-month")
    fig.tight_layout()
    plots.save(fig, out_dir, "term_structure")
    plt.close(fig)

    vix = india_vix()
    if vix is not None:
        j = tp.join(vix.rename("vix"), how="inner").dropna()
        gp = 100 * (j["vix"] - j["v30"])
        res["India VIX against 30-day at-the-money vol"] = (
            f"{len(j):,} days; VIX higher on {(gp > 0).mean():.0%}, by a median {gp.median():.2f} "
            f"points (10th-90th percentile {gp.quantile(0.1):.2f} to {gp.quantile(0.9):.2f}); "
            f"correlation {j[['vix', 'v30']].corr().iloc[0, 1]:.3f}"
        )
    return res


def report_arbitrage(out_dir: Path) -> dict:
    """Static arbitrage, 2015-2026: model-free checks on every day's NIFTY option closes (order,
    spread, butterfly across strikes; calendar across expiries) by how thinly the legs traded and
    for all strikes against out-of-the-money only; then the SVI fits' butterfly and calendar flags
    by maturity, and a refit of every butterfly-flagged slice with a penalty on negative g."""
    from iv import arb
    from iv.build import OPTION_COLS
    from iv.data import NIFTY_OPTIONS, SVI_DAILY

    res = {}
    opts = pd.read_parquet(NIFTY_OPTIONS, columns=OPTION_COLS)
    workers = max(1, (os.cpu_count() or 2) - 1)
    rate = {}
    for otm in (False, True):
        for mc in (1, 100, 1000):
            jobs = ((d, g, mc, otm) for d, g in opts.groupby("date"))
            with ProcessPoolExecutor(workers) as ex:
                out = list(ex.map(arb.quote_checks_day, jobs, chunksize=8))
            v = pd.concat([o[0] for o in out])
            tests = pd.concat([o[1] for o in out]).groupby("check")["tests"].sum()
            r1 = (v[v["amount"] > 1].groupby("check").size() / tests).fillna(0)
            r5 = (v[v["amount"] > 5].groupby("check").size() / tests).fillna(0)
            rate[(otm, mc)] = r1
            label = f"{'out-of-the-money' if otm else 'all traded'} strikes, {mc}+ contracts"
            res[f"breaches over Rs 1 (over Rs 5), {label}"] = ", ".join(
                f"{c} {r1.get(c, 0):.2%} ({r5.get(c, 0):.2%})"
                for c in ("order", "spread", "butterfly", "calendar")
            ) + f"; {int(tests.sum()):,} comparisons"
            if not otm and mc == 1000:
                fwd = pd.read_parquet(SVI_DAILY, columns=["date", "expiry", "F"])
                x = v[v["amount"] > 1].merge(fwd, on=["date", "expiry"], how="left")
                itm = np.where(x["type"] == "CE", x["strike"] < x["F"], x["strike"] > x["F"])
                x = x.assign(itm=itm)
                res["  in the money among those (1,000+ contracts, over Rs 1)"] = ", ".join(
                    f"{c} {g['itm'].mean():.1%}" for c, g in x.groupby("check")
                    if c in ("spread", "butterfly")
                )
            if otm and mc == 1000:
                b = v[(v["check"] == "butterfly") & (v["amount"] > 0)]["amount"]
                res["  liquid out-of-the-money butterfly breaches, size"] = (
                    f"median Rs {b.median():.2f}, 90th percentile Rs {b.quantile(0.9):.2f}"
                )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(3)
    for j, (otm, lab, col) in enumerate(((False, "all traded strikes", plots.MUTED),
                                         (True, "out of the money only", plots.SERIES[0]))):
        vals = [100 * rate[(otm, mc)].get("butterfly", 0) for mc in (1, 100, 1000)]
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, color=col, label=lab)
        for x, y in zip(xs + (j - 0.5) * 0.38, vals, strict=True):
            ax.text(x, y + 0.3, f"{y:.1f}%", ha="center", va="bottom", fontsize=9,
                    color=plots.TEXT)
    ax.set_xticks(xs, ["every leg traded", "every leg 100+ contracts", "every leg 1,000+"])
    ax.set_ylabel("butterflies priced below zero\nby over Rs 1, % of checks")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Closing prices broke the butterfly rule mostly where an option barely traded")
    fig.tight_layout()
    plots.save(fig, out_dir, "quote_arbitrage")
    plt.close(fig)

    d = svi_table()
    bins = pd.cut(d["days"], MATURITY_BINS, labels=MATURITY_LABELS)
    g = d.groupby(bins, observed=True)
    res["SVI fits flagged (butterfly, calendar)"] = (
        f"{d['butterfly'].mean():.1%}, {d['calendar'].mean():.1%} of {len(d):,} slices; "
        "by maturity "
        + "; ".join(f"{k} {g['butterfly'].mean()[k]:.1%}, {g['calendar'].mean()[k]:.1%}"
                    for k in g.groups)
    )
    jobs = ((dt, gg, 1, True) for dt, gg in opts.groupby("date"))
    with ProcessPoolExecutor(workers) as ex:
        v_otm = pd.concat([o[0] for o in ex.map(arb.quote_checks_day, jobs, chunksize=8)])
    qb = (v_otm[(v_otm["check"] == "butterfly") & (v_otm["amount"] > 1)]
          .groupby(["date", "expiry"]).size().rename("qb"))
    d = d.join(qb, on=["date", "expiry"])
    d["qb"] = d["qb"].fillna(0) > 0
    res["  fit flagged when its own quotes broke a butterfly by over Rs 1 / when they didn't"] = (
        f"{d.loc[d['qb'], 'butterfly'].mean():.1%} / {d.loc[~d['qb'], 'butterfly'].mean():.1%}"
    )

    flagged = d[d["butterfly"]]
    fl = {dt: {r.expiry: {p: getattr(r, p) for p in s.PARAMS} for r in gg.itertuples()}
          for dt, gg in flagged.groupby("date")}
    sub = opts[opts["date"].isin(fl)]
    with ProcessPoolExecutor(workers) as ex:
        h = pd.DataFrame([r for rr in ex.map(arb.refit_day,
                                             ((dt, gg, fl[dt]) for dt, gg in sub.groupby("date")),
                                             chunksize=4) for r in rr])
    inc = h["rmse_after"] - h["rmse_before"]
    hb = pd.cut(h["days"], MATURITY_BINS, labels=MATURITY_LABELS)
    res["penalty refit of butterfly-flagged slices"] = (
        f"{len(h):,} slices; arbitrage-free after the refit {h['fixed'].mean():.0%}; fit error "
        f"median {h['rmse_before'].median():.2f} -> {h['rmse_after'].median():.2f} vol points "
        f"(median rise {inc.median():.3f}, 90th percentile {inc.quantile(0.9):.2f})"
    )
    res["  fixed by maturity"] = ", ".join(
        f"{k} {v:.0%}" for k, v in h.groupby(hb, observed=True)["fixed"].mean().items()
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    before = g["butterfly"].mean() * 100
    fixed = h.groupby(hb, observed=True)["fixed"].mean().reindex(before.index).fillna(0)
    after = before * (1 - fixed)
    xs = np.arange(len(before))
    for j, (vals, lab, col) in enumerate(((before, "SVI fit", plots.MUTED),
                                          (after, "refit with the no-butterfly penalty",
                                           plots.SERIES[0]))):
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, color=col, label=lab)
        for x, y in zip(xs + (j - 0.5) * 0.38, vals, strict=True):
            ax.text(x, y + 0.4, f"{y:.1f}%", ha="center", va="bottom", fontsize=9,
                    color=plots.TEXT)
    ax.set_xticks(xs, list(before.index))
    ax.set_ylabel("slices with butterfly arbitrage, %")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("A third of weekly-expiry fits broke the butterfly rule; a penalty removed most")
    fig.tight_layout()
    plots.save(fig, out_dir, "svi_arbitrage")
    plt.close(fig)
    return res


def nifty_close() -> pd.Series:
    """NIFTY's daily close: the local lake (IV_LAKE_DIR) or Yahoo's ^NSEI."""
    lake = os.environ.get("IV_LAKE_DIR")
    if lake:
        m = pd.read_parquet(Path(lake) / "market.parquet").set_index("date")
        return m["nifty"].dropna()
    import yfinance as yf

    c = yf.download("^NSEI", start="2015-01-01", progress=False, auto_adjust=False,
                    threads=False)["Close"]
    return (c.iloc[:, 0] if isinstance(c, pd.DataFrame) else c).dropna()


def report_tails(out_dir: Path) -> dict:
    """What the surface priced against what NIFTY did, 2015-2026: one slice about 30 days before
    each monthly expiry (non-overlapping months), its risk-neutral distribution of the move from
    the forward to the expiry close, against the realised move: priced against realised chances of
    moves beyond 3, 5 and 10%, the probability-integral transform, and the log score against a
    flat smile at the at-the-money volatility."""
    from scipy.stats import binomtest, chisquare, kstest

    from iv import rnd
    from iv.build import OPTION_COLS
    from iv.data import NIFTY_OPTIONS, SVI_DAILY

    res = {}
    svi = pd.read_parquet(SVI_DAILY)
    opts = pd.read_parquet(NIFTY_OPTIONS, columns=OPTION_COLS)
    d = rnd.run(svi, opts, nifty_close())
    n = len(d)
    res["months"] = (
        f"{n} monthly expiries, {d['date'].min():%b %Y} to {d['expiry'].max():%b %Y}; "
        f"{d['days'].median():.0f} days ahead (median); {int(d['refit'].sum())} slices refitted "
        f"for butterfly arbitrage; density checks: total probability {d['mass'].min():.4f}-"
        f"{d['mass'].max():.4f}, expected price / forward {d['mean'].min():.4f}-"
        f"{d['mean'].max():.4f}"
    )
    calm = ~d["date"].between("2020-02-01", "2020-04-30")
    rms = lambda v: 100 * float(np.sqrt((v**2).mean()))  # noqa: E731
    res["move from the forward to the expiry close"] = (
        f"mean {100 * d['x'].mean():+.2f}%; root mean square {rms(d['x']):.2f}% against "
        f"{rms(d['sd_atm']):.2f}% priced at the money; leaving out the three months from Feb-Apr "
        f"2020: {rms(d.loc[calm, 'x']):.2f}% against {rms(d.loc[calm, 'sd_atm']):.2f}%; largest: "
        f"{100 * d['x'].min():+.1f}% (from {d.loc[d['x'].idxmin(), 'date']:%d %b %Y})"
    )
    rows = []
    for side in ("down", "up"):
        for t in rnd.MOVES:
            k = int(d[f"{side}{t}"].sum())
            ci = binomtest(k, n).proportion_ci(0.90)
            q, f = d[f"q_{side}{t}"].mean(), d[f"flat_{side}{t}"].mean()
            p_two = binomtest(k, n, q).pvalue
            rows.append((side, t, q, f, k / n))
            res[f"{'fall' if side == 'down' else 'rise'} of over {t}%"] = (
                f"priced {q:.1%} (flat smile {f:.1%}); happened {k} of {n} = {k / n:.1%} "
                f"(90% interval {ci.low:.1%}-{ci.high:.1%}; p {p_two:.2f} against the priced rate)"
            )
    hist = np.histogram(d["pit"], bins=10, range=(0, 1))[0]
    hist_f = np.histogram(d["pit_flat"], bins=10, range=(0, 1))[0]
    res["where the outcome landed in the priced distribution (deciles, lowest first)"] = (
        f"smile {hist.tolist()}, flat {hist_f.tolist()}; {n / 10:.1f} each if the prices were "
        f"right; even spread: chi-square p {chisquare(hist).pvalue:.2f}, Kolmogorov-Smirnov p "
        f"{kstest(d['pit'], 'uniform').pvalue:.2f} "
        f"(flat {kstest(d['pit_flat'], 'uniform').pvalue:.2f})"
    )
    dl = d["ls"] - d["ls_flat"]
    res["log score, smile minus flat"] = (
        f"{dl.mean():+.3f} a month (t {dl.mean() / dl.std() * np.sqrt(n):.2f}); smile better in "
        f"{(dl > 0).mean():.0%} of months"
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    lab = [f"{'fall' if sd == 'down' else 'rise'} >{t}%" for sd, t, *_ in rows]
    xs = np.arange(len(rows))
    for j, (i, name, col) in enumerate(((2, "priced by the surface", plots.SERIES[0]),
                                        (4, "happened", plots.SERIES[1]))):
        vals = [100 * r[i] for r in rows]
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, color=col, label=name)
        for x, y in zip(xs + (j - 0.5) * 0.38, vals, strict=True):
            ax.text(x, y + 0.4, f"{y:.1f}%", ha="center", va="bottom", fontsize=8.5,
                    color=plots.TEXT)
    ax.set_xticks(xs, lab)
    ax.set_ylabel(f"chance over the next month, % ({n} months)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Over 142 months, the surface's odds of big moves were close to what happened")
    fig.tight_layout()
    plots.save(fig, out_dir, "tails")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    centers = np.arange(10)
    ax.bar(centers, hist, 0.8, color=plots.SERIES[0])
    ax.axhline(n / 10, color=plots.TEXT_2, lw=1, ls="--")
    ax.text(7, n / 10 + 0.3, "if the prices were right", ha="center", va="bottom",
            fontsize=9, color=plots.TEXT_2)  # over the 70-80% bar, clear of the value labels
    for x, y in zip(centers, hist, strict=True):
        ax.text(x, y + 0.2, str(y), ha="center", va="bottom", fontsize=9, color=plots.TEXT)
    ax.set_xticks(centers, ["lowest\n10%"] + [f"{10 * i}-{10 * i + 10}%" for i in range(1, 9)]
                  + ["highest\n10%"])
    ax.set_xlabel("where the expiry close landed in the priced distribution")
    ax.set_ylabel("months")
    ax.set_title("Outcomes spread across the priced range about as evenly as the prices implied")
    fig.tight_layout()
    plots.save(fig, out_dir, "pit")
    plt.close(fig)
    return res


def report_dynamics(out_dir: Path) -> dict:
    """How the surface moves, 2015-2026. (1) Sticky rules on the 30-day constant-maturity smile
    from one trading day to the next: the at-the-money multiplier beta in d(vol) = beta S r, by
    year, and the whole-smile error of the rule "tomorrow's vol at k = today's at k + beta r"
    (beta fitted on 2015-2020, scored on 2021-2026). (2) Principal components of daily changes of
    the vol grid at 95-105% of the forward and 30, 60 and 90 days, with the 2015-2020 components
    scored on 2021-2026."""
    from iv import dynamics as dy

    res = {}
    svi = svi_table()
    st = dy.sticky(svi)
    x = st["slope"] * st["r"]
    beta = lambda m: float((x[m] * st["d_atm"][m]).sum() / (x[m] ** 2).sum())  # noqa: E731
    train, test = st.index.year <= 2020, st.index.year >= 2021
    yrs = sorted(set(st.index.year))
    by = {y: beta(st.index.year == y) for y in yrs}
    res["day pairs"] = (
        f"{len(st):,} ({int(train.sum()):,} in 2015-2020, {int(test.sum()):,} in 2021-2026); "
        f"30-day at-the-money slope d(vol)/dk median {st['slope'].median():.2f}"
    )
    res["at-the-money multiplier beta (1 = sticky strike, 0 = sticky moneyness)"] = (
        f"all {beta(st.index == st.index):.2f}; 2015-2020 {beta(train):.2f}, 2021-2026 "
        f"{beta(test):.2f}; rises {beta(st['r'] > 0):.2f}, falls {beta(st['r'] < 0):.2f}; by year "
        f"{min(by.values()):.2f} to {max(by.values()):.2f}, above 1 in "
        f"{sum(v > 1 for v in by.values())} of {len(by)} years"
    )
    res["  so a 1% fall in NIFTY moved 30-day at-the-money vol by"] = (
        f"{-beta(st.index == st.index) * st['slope'].median() * 0.01 * 100:+.2f} points at the "
        f"median slope (sticky strike: {-st['slope'].median() * 0.01 * 100:+.2f})"
    )
    errs = [c for c in st if c.startswith("err_")]
    bvals = [float(c[4:]) for c in errs]
    best = bvals[int(np.argmin(st.loc[train, errs].mean().to_numpy()))]
    res["whole-smile error of the shift rule, 2021-2026 (vol points, mean over days)"] = (
        ", ".join(f"beta {b:g} {st.loc[test, c].mean():.3f}" for b, c in zip(bvals, errs,
                                                                               strict=True))
        + f"; best on 2015-2020: beta {best:g}"
    )
    big = test & (st["r"].abs() > 0.01)
    res["  on 2021-2026 days with a move over 1%"] = (
        f"{int(big.sum())} days: " + ", ".join(
            f"beta {b:g} {st.loc[big, c].mean():.3f}" for b, c in zip(bvals, errs, strict=True)
            if b in (0.0, 0.5, 1.0, 2.0))
    )

    lv = [c for c in st if c.startswith("lvl_")]
    lbest = lv[int(np.argmin(st.loc[train, lv].mean().to_numpy()))]
    win = lambda m: (st.loc[m, lbest] < st.loc[m, f"err_{best}"]).mean()  # noqa: E731
    res["same shape against the forward, level shifted by b S r (vol points, 2021-2026)"] = (
        ", ".join(f"b {c[4:]} {st.loc[test, c].mean():.3f}" for c in lv)
        + f"; best on 2015-2020: b {lbest[4:]}; days with a move over 1%: "
        + ", ".join(f"b {c[4:]} {st.loc[big, c].mean():.3f}" for c in lv)
        + f"; beats the best slide on {win(test):.0%} of days ({win(big):.0%} of big-move days)"
    )
    rules = [("sticky moneyness", "err_0.0"), ("sticky strike", "err_1.0"),
             (f"best slide ({best:g}x)", f"err_{best}"),
             (f"same shape,\nlevel moves ({float(lbest[4:]):g}x)", lbest)]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(rules))
    for j, (mask, lab, col) in enumerate(((test, "every day", plots.MUTED),
                                          (big, "days NIFTY moved over 1%", plots.SERIES[0]))):
        vals = [st.loc[mask, c].mean() for _, c in rules]
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, color=col, label=lab)
        for xx, y in zip(xs + (j - 0.5) * 0.38, vals, strict=True):
            ax.text(xx, y + 0.02, f"{y:.2f}", ha="center", va="bottom", fontsize=9,
                    color=plots.TEXT)
    ax.set_xticks(xs, [r[0] for r in rules])
    ax.set_ylabel("error predicting tomorrow's\n30-day smile, vol points")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("Tomorrow's smile: same shape against the forward, level moved against NIFTY")
    fig.tight_layout()
    plots.save(fig, out_dir, "sticky")
    plt.close(fig)

    g = dy.grid(svi)
    v = g.drop(columns="F", level=0).dropna()
    gap = v.index.to_series().diff().dt.days
    ch = (v.diff() * 100)[gap <= 5].dropna()
    share, load, sc = dy.pca(ch)
    r = np.log(g[("F", 30)]).diff().reindex(sc.index)
    res["grid changes"] = (
        f"{len(ch):,} day-to-day changes (gaps of 5 days or less), {ch.shape[1]} points "
        f"(95-105% of the forward x 30, 60, 90 days)"
    )
    res["principal components, share of variance"] = ", ".join(
        f"PC{i + 1} {share[i]:.1%}" for i in range(4))
    res["  correlation of each component with NIFTY's move"] = ", ".join(
        f"{c} {sc[c].corr(r):+.2f}" for c in sc)
    tr, te = ch[ch.index.year <= 2020], ch[ch.index.year >= 2021]
    s1, l1, _ = dy.pca(tr)
    s2, l2, _ = dy.pca(te)
    x2 = te.to_numpy() - te.to_numpy().mean(axis=0)
    tot = (x2**2).sum()
    oos = [1 - ((x2 - x2 @ l1[:, :n] @ l1[:, :n].T) ** 2).sum() / tot for n in (1, 2, 3)]
    res["  2015-2020 components on 2021-2026 changes (share explained, 1/2/3 components)"] = (
        " / ".join(f"{v:.1%}" for v in oos)
        + f"; 2021-2026's own: {s2[0]:.1%} / {s2[:2].sum():.1%} / {s2[:3].sum():.1%}; "
        f"first component's loadings correlate {abs(np.corrcoef(l1[:, 0], l2[:, 0])[0, 1]):.2f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), sharey=True)
    mny = sorted({c[1] for c in ch.columns})
    names = ("level: everything up or down", "term: short against long",
             "skew: downside against upside")
    cols = {30: plots.SERIES[1], 60: plots.SERIES[0], 90: plots.BLUE_RAMP[6]}
    for j, ax in enumerate(axes):
        for m in (30, 60, 90):
            idx = [list(ch.columns).index((m, k)) for k in mny]
            ax.plot(mny, load[idx, j], color=cols[m], lw=2, marker="o", ms=4,
                    label=f"{m} days")
        ax.axhline(0, color=plots.TEXT_2, lw=0.8)
        ax.set_title(f"PC{j + 1} ({share[j]:.0%}): {names[j]}", fontsize=10.5)
        ax.set_xlabel("strike, % of forward")
        ax.set_xticks(mny, [f"{k:g}" for k in mny])
    axes[0].set_ylabel("weight in the component")
    axes[0].legend(fontsize=8.5, loc="lower left")
    fig.suptitle("Daily surface changes were mostly one move: the whole surface up or down",
                 x=0.01, ha="left", fontsize=13, fontweight="semibold")
    fig.tight_layout()
    plots.save(fig, out_dir, "pca")
    plt.close(fig)
    return res


def report_forecast(out_dir: Path) -> dict:
    """Forecasting NIFTY's realised variance over the next 5, 21 and 63 trading days from each
    day's close, 2015-2026: at-the-money and full-smile implied variance from the surface, India
    VIX (21 days), GJR-GARCH-t refitted each January, EWMA, against the 20-day variance, by QLIKE.
    Scaling factors and the implied+GARCH blend are fitted on 2015-2020; scores are on 2021-2026.
    Needs IV_LAKE_DIR for India VIX."""
    from iv import forecast as fc
    from iv.data import SVI_DAILY

    res = {}
    d = fc.build(pd.read_parquet(SVI_DAILY), nifty_close(), india_vix())
    v21 = d[d["h"] == 21].dropna(subset=["vix", "varswap"])
    res["full-smile implied against India VIX (21 days)"] = (
        f"ratio median {(v21['varswap'] / v21['vix']).median():.2f}, correlation "
        f"{v21[['varswap', 'vix']].corr().iloc[0, 1]:.3f} ({len(v21):,} days)"
    )
    names = {"atm": "at-the-money implied", "varswap": "full-smile implied (VIX method)",
             "vix": "India VIX", "garch": "GJR-GARCH", "ewma": "EWMA"}
    chart = {}
    for h, g in d.groupby("h"):
        cols = ["atm", "varswap", "garch", "hist20", "ewma"] + (["vix"] if h == 21 else [])
        g = g.dropna(subset=cols).set_index("date")
        tr, te = g[g.index.year <= 2020], g[g.index.year >= 2021]
        ratio = {c: g[c].mean() / g["realised"].mean() for c in cols if c != "hist20"}
        res[f"{h} days: windows (2015-2020 / 2021-2026)"] = (
            f"{len(tr):,} / {len(te):,}; average forecast / average realised (all years): "
            + ", ".join(f"{names[c]} {v:.2f}" for c, v in ratio.items())
        )
        base = fc.qlike(te["realised"], te["hist20"])
        loss = {c: fc.qlike(te["realised"], te[c]) for c in cols if c != "hist20"}
        for c in [c for c in cols if c not in ("hist20", "atm", "ewma")]:
            k = tr["realised"].mean() / tr[c].mean()
            loss[f"{c}_scaled"] = fc.qlike(te["realised"], te[c] * k)
        x = np.column_stack([np.ones(len(tr)), np.log(tr["atm"]), np.log(tr["garch"])])
        y = np.log(np.maximum(tr["realised"], 1e-10))
        b = np.linalg.lstsq(x, y, rcond=None)[0]
        adj = float(np.exp(y - x @ b).mean())
        blend = np.exp(b[0] + b[1] * np.log(te["atm"]) + b[2] * np.log(te["garch"])) * adj
        loss["blend"] = fc.qlike(te["realised"], blend)
        rel = {c: (v.mean() / base.mean() - 1, fc.nw_t(v - base, h)) for c, v in loss.items()}
        chart[h] = rel
        blend_name = f"blend {b[1]:.2f} x implied + {b[2]:.2f} x GARCH (logs)"
        labels = {c: blend_name if c == "blend" else names[c.replace("_scaled", "")]
                  + (" scaled" if c.endswith("_scaled") else "") for c in rel}
        res[f"{h} days: QLIKE against the 20-day variance, 2021-2026"] = "; ".join(
            f"{labels[c]} {v:+.1%} (t {t:.1f})" for c, (v, t) in rel.items())
        res[f"  {h} days: implied against GARCH / blend against implied"] = (
            f"t {fc.nw_t(loss['atm'] - loss['garch'], h):.1f} / "
            f"t {fc.nw_t(loss['blend'] - loss['atm'], h):.1f} (negative: the first is better)"
        )

    fig, ax = plt.subplots(figsize=(10, 4.7))
    show = [("atm", "at-the-money implied", plots.SERIES[0]),
            ("varswap", "full-smile implied (VIX method)", plots.BLUE_RAMP[2]),
            ("garch", "GJR-GARCH", plots.SERIES[1]), ("ewma", "EWMA", plots.MUTED)]
    hs = sorted(chart)
    xs = np.arange(len(hs))
    wd = 0.2
    for j, (c, lab, col) in enumerate(show):
        vals = [100 * chart[h][c][0] for h in hs]
        ax.bar(xs + (j - 1.5) * wd, vals, wd, color=col, label=lab)
        for xx, y in zip(xs + (j - 1.5) * wd, vals, strict=True):
            ax.text(xx, y - 1.2, f"{y:.0f}%", ha="center", va="top", fontsize=8.5,
                    color=plots.TEXT)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(xs, [f"next {h} trading days" for h in hs])
    ax.set_ylabel("forecast loss against the\n20-day variance, % (2021-2026)")
    ax.set_ylim(top=5)
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    ax.set_title("At-the-money implied volatility forecast NIFTY's real volatility best")
    fig.tight_layout()
    plots.save(fig, out_dir, "forecast")
    plt.close(fig)

    m = d[d["h"] == 21].dropna(subset=["atm"]).set_index("date")
    me = m.groupby([m.index.year, m.index.month]).tail(1)
    ann = lambda v: 100 * np.sqrt(v * 365 / me["D"])  # noqa: E731
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.plot(me.index, ann(me["realised"]), color=plots.SERIES[1], lw=1.6)
    ax.plot(me.index, ann(me["atm"]), color=plots.SERIES[0], lw=1.6)
    last_r, last_a = float(ann(me["realised"]).iloc[-1]), float(ann(me["atm"]).iloc[-1])
    up, dn = (5, -5) if last_a >= last_r else (-5, 5)  # keep the two end labels apart
    plots.label_line_end(ax, me.index[-1], last_r + dn, "what happened", plots.SERIES[1])
    plots.label_line_end(ax, me.index[-1], last_a + up, "implied a month ahead", plots.SERIES[0])
    ax.set_ylabel("NIFTY volatility over\nthe next 21 trading days, %")
    ax.set_title("Implied volatility tracked the next month's real volatility, but not 2020's jump")
    fig.tight_layout()
    plots.save(fig, out_dir, "forecast_path")
    plt.close(fig)
    res["month-ends in the path chart"] = f"{len(me)}"
    return res


def report_stocks(out_dir: Path) -> dict:
    """Stock option smiles against NIFTY's at each month-end 2015-2026 (SVI on the stock's
    expiries 15-60 days out), and each stock's at-the-money implied variance as a forecast of its
    variance over the next 21 trading days against GJR with common parameters (garch-volatility's
    rule), EWMA and the 20-day variance, by QLIKE. Needs IV_LAKE_DIR (stock prices)."""
    from iv import forecast as fc
    from iv import stocks
    from iv.data import STOCK_OPTIONS

    res = {}
    so = pd.read_parquet(STOCK_OPTIONS)
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        sl = pd.DataFrame([r for rr in ex.map(stocks.fit_date, so.groupby("date"), chunksize=2)
                           for r in rr])
    res["stock slices"] = (
        f"{len(sl):,} on {sl['date'].nunique()} month-ends, {sl['symbol'].nunique()} stocks "
        f"({sl.groupby('date')['symbol'].nunique().median():.0f} a month-end, median); fit error "
        f"median {sl['rmse'].median():.2f} vol points; butterfly arbitrage "
        f"{sl['butterfly'].mean():.1%}; futures forward {(sl['source'] == 'futures').mean():.0%}"
    )

    nifty = svi_table()
    from iv.shape import atm_slope

    nifty["slope"] = [atm_slope(dict(zip(s.PARAMS, r, strict=True)), t)
                      for r, t in zip(nifty[list(s.PARAMS)].to_numpy(), nifty["T"], strict=True)]
    pair = sl.merge(nifty[["date", "expiry", "slope", "atm_vol", "rr25"]],
                    on=["date", "expiry"], suffixes=("", "_nifty"))
    pair = pair[pair["rr25"].abs() < 1]  # a few degenerate fits put rr25 beyond 100 vol points
    pair["skew_per_vol"] = pair["slope"] / pair["atm_vol"]
    pair["nifty_skew_per_vol"] = pair["slope_nifty"] / pair["atm_vol_nifty"]
    res["same date and expiry as NIFTY"] = (
        f"{len(pair):,} stock slices; at-the-money vol median {pair['atm_vol'].median():.1%} "
        f"against NIFTY's {pair['atm_vol_nifty'].median():.1%}"
    )
    res["at-the-money skew slope d(vol)/dk, median"] = (
        f"stocks {pair['slope'].median():.3f}, NIFTY {pair['slope_nifty'].median():.3f}; "
        f"stock steeper than NIFTY in {(pair['slope'] < pair['slope_nifty']).mean():.0%} of "
        f"pairs; stock slope negative (puts dearer) in {(pair['slope'] < 0).mean():.0%}"
    )
    res["skew per unit of vol (slope / at-the-money vol), median"] = (
        f"stocks {pair['skew_per_vol'].median():.2f}, "
        f"NIFTY {pair['nifty_skew_per_vol'].median():.2f}"
    )
    res["25-delta risk reversal, median vol points"] = (
        f"stocks {100 * pair['rr25'].median():+.2f}, NIFTY {100 * pair['rr25_nifty'].median():+.2f}"
    )

    lake = os.environ.get("IV_LAKE_DIR")
    if not lake:
        res["forecast"] = "skipped: IV_LAKE_DIR not set"
        return res
    p = pd.read_parquet(Path(lake) / "panel.parquet", columns=["date", "symbol", "adj_close"])
    p = p[p["symbol"].isin(sl["symbol"].unique())]
    px = p.pivot_table(index="date", columns="symbol", values="adj_close")
    lr = np.log(px).diff()
    g = stocks.gjr_common({c: 100 * lr[c].dropna() for c in lr})
    d = stocks.forecast_table(sl, lr, g)
    q = lambda c, x: pd.Series(fc.qlike(x["realised"], x[c]), index=x.index)  # noqa: E731
    res["forecast cases"] = (
        f"{len(d):,} stock-months, {d['symbol'].nunique()} stocks, {d['date'].nunique()} "
        f"month-ends; average forecast / average realised: "
        + ", ".join(f"{c} {d[c].mean() / d['realised'].mean():.2f}"
                    for c in ("atm", "garch", "ewma", "hist20"))
    )
    rel = {}
    for per, m in (("2016-2026", d["date"].dt.year > 0), ("2021-2026", d["date"].dt.year >= 2021)):
        x = d[m]
        base = q("hist20", x)
        parts = []
        for c in ("atm", "garch", "ewma"):
            loss = q(c, x)
            diff = (loss - base).groupby(x["date"]).mean()
            t = fc.nw_t(diff.to_numpy(), 1)
            parts.append(f"{c} {loss.mean() / base.mean() - 1:+.1%} (t {t:.1f})")
            rel[(per, c)] = loss.mean() / base.mean() - 1
        dd = (q("atm", x) - q("garch", x)).groupby(x["date"]).mean()
        res[f"QLIKE against the 20-day variance, {per}"] = (
            "; ".join(parts) + f"; implied against GJR: t {fc.nw_t(dd.to_numpy(), 1):.1f}, "
            f"implied better in {(dd < 0).mean():.0%} of month-ends"
        )
    liq = d["contracts"].groupby(d["date"]).rank(pct=True)
    third = pd.cut(liq, [0, 1 / 3, 2 / 3, 1], labels=["least", "middle", "most"])
    res["by option trading that month (thirds within each month-end)"] = "; ".join(
        f"{k} traded: implied {q('atm', g_).mean() / q('hist20', g_).mean() - 1:+.1%}, GJR "
        f"{q('garch', g_).mean() / q('hist20', g_).mean() - 1:+.1%}"
        for k, g_ in d.groupby(third, observed=True)
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    mon = pair.groupby("date").agg(stock=("slope", "median"), nifty=("slope_nifty", "median"))
    ax.plot(mon.index, -mon["nifty"], color=plots.COLORS["NIFTY"], lw=1.8)
    ax.plot(mon.index, -mon["stock"], color=plots.COLORS["stocks"], lw=1.8)
    plots.label_line_end(ax, mon.index[-1], -mon["nifty"].iloc[-1], "NIFTY", plots.COLORS["NIFTY"])
    plots.label_line_end(ax, mon.index[-1], -mon["stock"].iloc[-1], "median stock",
                         plots.COLORS["stocks"])
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_ylabel("how much dearer downside strikes are\n(minus the at-the-money slope)")
    ax.set_title("Stock smiles were nearly flat; NIFTY's always tilted towards the downside")
    fig.tight_layout()
    plots.save(fig, out_dir, "stock_skew")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    show = [("atm", "own at-the-money implied", plots.SERIES[0]),
            ("garch", "GJR, common parameters", plots.SERIES[1]), ("ewma", "EWMA", plots.MUTED)]
    pers_ = ["2016-2026", "2021-2026"]
    xs = np.arange(len(pers_))
    for j, (c, lab, col) in enumerate(show):
        vals = [100 * rel[(pp, c)] for pp in pers_]
        ax.bar(xs + (j - 1) * 0.26, vals, 0.26, color=col, label=lab)
        for xx, y in zip(xs + (j - 1) * 0.26, vals, strict=True):
            ax.text(xx, y - 1, f"{y:.0f}%", ha="center", va="top", fontsize=9, color=plots.TEXT)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(xs, [f"stock-months, {pp}" for pp in pers_])
    ax.set_ylabel("loss forecasting the next 21 days'\nvariance, against the 20-day, %")
    ax.set_ylim(top=4)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title("A stock's own option prices forecast its volatility better than GARCH")
    fig.tight_layout()
    plots.save(fig, out_dir, "stock_forecast")
    plt.close(fig)
    return res


def _partial_rank(d: pd.DataFrame, x: str, y: str, given: str) -> float:
    rk = d[[x, y, given]].rank()
    g = np.column_stack([np.ones(len(rk)), rk[given].to_numpy()])

    def resid(v):
        return v - g @ np.linalg.lstsq(g, v, rcond=None)[0]

    return float(np.corrcoef(resid(rk[x].to_numpy()), resid(rk[y].to_numpy()))[0, 1])


def report_predict(out_dir: Path) -> dict:
    """Do the surface's shapes predict? NIFTY at each month-end 2015-2026: rank correlations of
    level, skew, term structure, implied-minus-realised premium and India VIX with the next 21
    trading days' volatility, worst fall, return and falls over 5%, and what each adds beyond VIX.
    Stocks at each month-end: monthly rank correlation (IC) of skew, risk reversal, implied minus
    realised vol and implied vol with the return to the next month-end, top-minus-bottom fifth,
    halves, and Benjamini-Hochberg across the four. Needs IV_LAKE_DIR."""
    from scipy.stats import spearmanr, ttest_1samp

    from iv import predict as pr
    from iv import stocks
    from iv.data import STOCK_OPTIONS

    res = {}
    n = pr.nifty_signals(svi_table(), nifty_close(), india_vix())
    nn = n.dropna()
    sigs = ["level", "vix", "skew", "term", "premium"]
    outs = ["vol", "worst", "ret", "fall5"]
    corr = {(sg, o): spearmanr(nn[sg], nn[o]).statistic for sg in sigs for o in outs}
    res["NIFTY month-ends"] = (
        f"{len(nn)} ({nn.index.min():%b %Y}-{nn.index.max():%b %Y}); a rank correlation beyond "
        f"+-{2 / np.sqrt(len(nn)):.2f} is about two standard errors"
    )
    for sg in sigs:
        key = f"  {sg}: rank correlation with next month's vol / worst fall / return / fall >5%"
        res[key] = (
            " / ".join(f"{corr[(sg, o)]:+.2f}" for o in outs)
            + ("" if sg == "vix" else "; beyond VIX: " + " / ".join(
                f"{_partial_rank(nn, sg, o, 'vix'):+.2f}" for o in outs[:3]))
        )
    inv = nn["term"] < -0.01
    res["  months with 60-day vol over 1 point below 30-day"] = (
        f"{int(inv.sum())}: next month's vol median {nn.loc[inv, 'vol'].median():.1%} against "
        f"{nn.loc[~inv, 'vol'].median():.1%}, worst fall {nn.loc[inv, 'worst'].median():.1%} "
        f"against "
        f"{nn.loc[~inv, 'worst'].median():.1%}, return {nn.loc[inv, 'ret'].mean():+.2%} against "
        f"{nn.loc[~inv, 'ret'].mean():+.2%}"
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    lab = {"level": "30-day at-the-money vol", "vix": "India VIX", "skew": "skew",
           "term": "term (60 - 30 days)", "premium": "implied - recent realised"}
    show = [("vol", "next month's volatility", plots.SERIES[0]),
            ("worst", "next month's worst fall", plots.SERIES[1]),
            ("ret", "next month's return", plots.MUTED)]
    xs = np.arange(len(sigs))
    band = 2 / np.sqrt(len(nn))
    ax.axhspan(-band, band, color=plots.GRID, zorder=0)
    for j, (o, name, col) in enumerate(show):
        vals = [corr[(sg, o)] for sg in sigs]
        ax.bar(xs + (j - 1) * 0.26, vals, 0.26, color=col, label=name)
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.text(len(sigs) - 0.5, band + 0.02, "within +-2 standard errors (shaded)", ha="right",
            fontsize=8.5, color=plots.TEXT_2)
    ax.set_xticks(xs, [lab[sg] for sg in sigs])
    ax.set_ylabel(f"rank correlation\n({len(nn)} month-ends)")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_title("The surface warned of next month's volatility, not of its return")
    fig.tight_layout()
    plots.save(fig, out_dir, "predict_nifty")
    plt.close(fig)

    lake = os.environ.get("IV_LAKE_DIR")
    if not lake:
        res["stocks"] = "skipped: IV_LAKE_DIR not set"
        return res
    so = pd.read_parquet(STOCK_OPTIONS)
    workers = max(1, (os.cpu_count() or 2) - 1)
    with ProcessPoolExecutor(workers) as ex:
        sl = pd.DataFrame([r for rr in ex.map(stocks.fit_date, so.groupby("date"), chunksize=2)
                           for r in rr])
    p = pd.read_parquet(Path(lake) / "panel.parquet", columns=["date", "symbol", "adj_close"])
    p = p[p["symbol"].isin(sl["symbol"].unique())]
    px = p.pivot_table(index="date", columns="symbol", values="adj_close")
    d = pr.stock_signals(sl, np.log(px).diff(), px)
    res["stock-months"] = (
        f"{len(d):,} on {d['date'].nunique()} month-ends ({d.groupby('date').size().median():.0f} "
        f"stocks a month-end)"
    )
    names = {"skew": "skew (dearer puts)", "rr25": "risk reversal (dearer calls)",
             "gap": "implied minus recent realised vol", "level": "implied vol"}
    pv, halves = {}, {}
    for sg in names:
        ic = pr.rank_ic(d, sg)
        qs = pr.quintile_spread(d, sg)
        pv[sg] = ttest_1samp(ic, 0).pvalue
        h1, h2 = ic[ic.index.year <= 2020], ic[ic.index.year >= 2021]
        halves[sg] = (h1.mean(), h2.mean(), ic.mean(), ic.std() / np.sqrt(len(ic)))
        res[f"  {names[sg]}"] = (
            f"IC {ic.mean():+.4f} (t {ic.mean() / ic.std() * np.sqrt(len(ic)):.2f}, p "
            f"{pv[sg]:.3f}; 2015-2020 {h1.mean():+.4f}, 2021-2026 {h2.mean():+.4f}; negative in "
            f"{(ic < 0).mean():.0%} of months); top minus bottom fifth {100 * qs.mean():+.2f}% a "
            f"month (t {qs.mean() / qs.std() * np.sqrt(len(qs)):.2f})"
        )
    order = sorted(pv, key=pv.get)
    m = len(order)
    passed = [sg for i, sg in enumerate(order) if pv[sg] <= 0.05 * (i + 1) / m]
    res["  Benjamini-Hochberg at 5% across the four"] = (
        ", ".join(f"{names[sg]} p {pv[sg]:.3f} (needs {0.05 * (i + 1) / m:.4f})"
                  for i, sg in enumerate(order))
        + f"; passing: {', '.join(names[x] for x in passed) if passed else 'none'}"
    )
    d["rvol"] = d["level"] - d["gap"]
    ic_rv = pr.rank_ic(d, "rvol")
    res["  recent realised vol alone"] = (
        f"IC {ic_rv.mean():+.4f} (t {ic_rv.mean() / ic_rv.std() * np.sqrt(len(ic_rv)):.2f})"
    )

    fig, ax = plt.subplots(figsize=(10, 4.2))
    xs = np.arange(len(names))
    for j, (k, name, col) in enumerate(((0, "2015-2020", plots.MUTED),
                                        (1, "2021-2026", plots.SERIES[0]))):
        vals = [halves[sg][k] for sg in names]
        ax.bar(xs + (j - 0.5) * 0.38, vals, 0.38, color=col, label=name)
    full = [halves[sg][2] for sg in names]
    err = [2 * halves[sg][3] for sg in names]
    ax.errorbar(xs, full, yerr=err, fmt="o", color=plots.TEXT, ms=5, capsize=4,
                label="all months, +-2 standard errors")
    ax.axhline(0, color=plots.TEXT_2, lw=0.8)
    ax.set_xticks(xs, [names[sg] for sg in names])
    ax.set_ylabel("average monthly rank correlation\nwith next month's return")
    ax.legend(fontsize=9, loc="lower left")
    ax.set_title("No stock option signal ranked next month's returns reliably")
    fig.tight_layout()
    plots.save(fig, out_dir, "predict_stocks")
    plt.close(fig)
    return res


REPORTS: dict[str, Callable[[Path], dict]] = {
    "surface": report_surface,
    "arbitrage": report_arbitrage,
    "tails": report_tails,
    "dynamics": report_dynamics,
    "forecast": report_forecast,
    "stocks": report_stocks,
    "predict": report_predict,
}
