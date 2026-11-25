"""The surface lab: one static Plotly page built from the committed aggregate
docs/data/lab_surfaces.json (see build.py) and iv.js.

No option data is fetched at build time: NSE's archive can't be reached reliably from CI, so the
page shows the fits as of the last local rerun.
"""

import json
import re
from pathlib import Path
from string import Template

from iv import plots

FONT = 'system-ui, -apple-system, "Segoe UI", Roboto, sans-serif'
ICON = (
    "data:image/svg+xml,%3Csvg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 16 16%22%3E"
    "%3Cpath d=%22M1 5c3 6 5 7 7 7s4-2 7-9%22 fill=%22none%22 stroke=%22%232a78d6%22"
    " stroke-width=%222%22 stroke-linecap=%22round%22/%3E"
    "%3C/svg%3E"
)
HERE = Path(__file__).parent
JS = (HERE / "iv.js").read_text(encoding="utf-8")
LAB = (HERE / "lab.html").read_text(encoding="utf-8")
PAGE = Template((HERE / "page.html").read_text(encoding="utf-8"))
REPO = "https://github.com/Dhirajdas18/implied-vol-surface"


def _day(d: str) -> str:
    from datetime import date

    return f"{date.fromisoformat(d):%-d %b %Y}"


def lab_section(d: dict) -> str:
    return (LAB.replace("$js", JS).replace("$first", _day(d["first"]))
            .replace("$data", json.dumps(d, separators=(",", ":"))))


def method_section(d: dict) -> str:
    return f"""<ul class="plain">
<li>Data: NSE’s daily F&amp;O bhavcopy, NIFTY index options and futures, {_day(d["first"])} to
{_day(d["latest"])}. Option prices are closing prices (the day’s last trades), as in
black-scholes-pricer, not NSE’s settlement prices. Only options that traded that day are
used, out of the money against the forward (puts below it, calls above), priced above the
₹0.05 tick, with 2 to 365 days to expiry.</li>
<li>Forward: for each expiry, from put-call parity, F = K + e<sup>rT</sup>(C − P), the median
over the five traded strikes nearest the futures price (for weekly expiries, which have no future,
nearest the strike where call and put prices are closest); where fewer than three strikes traded
on both sides, the futures settlement price. Rate r = {100 * d["rate"]:.1f}% a year, constant, used
only for discounting. Time to expiry is calendar days / 365.</li>
<li>Implied volatility: Black-76 on the forward, solved with a bracketed Newton method. SVI
(Gatheral’s raw form) is fitted per expiry with at least five quotes by bounded least squares,
weighted so that each residual is an error in volatility: b ≥ 0, |ρ| &lt; 1, σ &gt; 0, a minimum
total variance ≥ 0, each wing’s slope at most 2 (Lee’s bound) and the vertex inside the quoted
range. RMSE is in volatility points.</li>
<li>Arbitrage checks: butterfly, Gatheral’s g(k) below zero anywhere in an expiry’s quoted range
(a negative implied density). Calendar, total variance at some k below the previous expiry’s,
over the range both quote. Flags are reported, not repaired.</li>
<li>Skew: the 25-delta risk reversal (the 25-delta call’s volatility minus the 25-delta
put’s, with forward deltas from the fitted slice) and the volatility at k = −0.1 minus at
k = +0.1. For a weekly expiry k = ±0.1 lies far beyond the quotes, so that measure is an
extrapolation there.</li>
<li>The page holds every trading day for about the last year and each month’s last trading day
before that; market quotes for the last 20 trading days only. It is rebuilt when the studies are
rerun locally; there is no live option feed.</li>
</ul>"""


SECTIONS = [
    (
        "Surface lab",
        "Pick a date and an expiry. Research only, not investment advice: NSE end-of-day prices "
        "and SVI fits, updated when the studies are rerun locally.",
        lab_section,
    ),
    (
        "Method and caveats",
        "What the lab computes, and what it leaves out.",
        method_section,
    ),
]


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def build_html(data: dict) -> str:
    parts, nav = [], []
    for title, blurb, make in SECTIONS:
        body = make(data)
        sid = _slug(title)
        nav.append(f'<li><a href="#{sid}">{title}</a></li>')
        parts.append(f'<section id="{sid}">\n<h2>{title}</h2>\n<p>{blurb}</p>\n{body}\n</section>')
    dk = plots.DARK
    return PAGE.substitute(
        icon=ICON,
        font=FONT,
        repo=REPO,
        updated=_day(data["latest"]),
        surface=plots.SURFACE,
        text=plots.TEXT,
        text2=plots.TEXT_2,
        grid=plots.GRID,
        accent=plots.SERIES[0],
        accent2=plots.SERIES[1],
        muted=plots.MUTED,
        d_surface=dk[plots.SURFACE],
        d_text=dk[plots.TEXT],
        d_text2=dk[plots.TEXT_2],
        d_grid=dk[plots.GRID],
        d_accent=dk[plots.SERIES[0]],
        d_accent2=dk[plots.SERIES[1]],
        d_muted="#6b6a64",
        nav="\n".join(nav),
        sections="\n".join(parts),
    )
