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
                    f"{c} {g['itm'].mean():.0%}" for c, g in x.groupby("check")
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


REPORTS: dict[str, Callable[[Path], dict]] = {
    "surface": report_surface,
    "arbitrage": report_arbitrage,
}
