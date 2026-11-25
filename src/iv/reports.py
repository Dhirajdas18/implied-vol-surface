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


REPORTS: dict[str, Callable[[Path], dict]] = {
    "surface": report_surface,
}
