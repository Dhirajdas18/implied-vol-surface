import json
import re
import shutil
import subprocess

import numpy as np
import pandas as pd
import pytest

from iv import surface as s
from iv.build import FIELDS, build, fit_day, lab_data
from iv.site import build_html

F0 = 22000.0


def _svi(T):
    # Total variance growing with T at every k: no calendar arbitrage.
    return {"a": 0.012 * T, "b": 0.12 * np.sqrt(T), "rho": -0.6, "m": 0.0, "sigma": 0.08}


def _chain(date, expiries=(9, 30, 58, 120), monthly=(30, 58)):
    date = pd.Timestamp(date)
    rows = []
    for n in expiries:
        T = n / 365
        F = F0 * np.exp(0.06 * T)
        K = np.arange(18000, 26000, 100.0)
        for kind in ("CE", "PE"):
            iv = s.svi_iv(_svi(T), np.log(K / F), T)
            rows.append(pd.DataFrame({
                "date": date, "expiry": date + pd.Timedelta(days=n), "strike": K, "type": kind,
                "price": np.round(s.black76(F, K, T, iv, kind), 2), "contracts": 5, "oi": 50,
                "fut_settle": F if n in monthly else np.nan}))
    out = pd.concat(rows, ignore_index=True)
    return out[out["price"] > 0.05]


@pytest.fixture(scope="module")
def built():
    days = pd.bdate_range("2026-10-26", "2026-11-25")
    opts = pd.concat([_chain(d) for d in days], ignore_index=True)
    return build(opts, workers=2, quote_days=5)


def test_fit_day_recovers_the_surface_and_finds_no_arbitrage():
    rows, q = fit_day("2026-11-25", _chain("2026-11-25"), keep_quotes=True)
    assert [r["days"] for r in rows] == [9, 30, 58, 120] and len(q) > 40
    for r in rows:
        T = r["T"]
        assert r["source"] == "parity" and r["F"] == pytest.approx(F0 * np.exp(0.06 * T), rel=2e-4)
        assert r["atm_vol"] == pytest.approx(s.atm_vol(_svi(T), T), abs=2e-3)
        assert r["rmse"] < 0.5 and r["n"] >= 5
        assert not r["butterfly"] and not r["calendar"]
        assert r["rr25"] < 0 < r["skew10"]
    assert [r["monthly"] for r in rows] == [False, True, True, False]


def test_lab_data_is_compact_with_quotes_only_for_recent_days(built):
    svi, quotes = built
    assert svi["date"].nunique() == 23 and len(quotes) == 5
    d = lab_data(svi, quotes, days=10)
    assert len(d["dates"]) == 11  # ten recent days plus October's last trading day
    assert d["latest"] == "2026-11-25" and d["first"] == "2026-10-30"
    row = d["slices"]["2026-11-25"][1]
    assert len(row) == len(FIELDS) and row[0] == 30 and row[-1] & d["flags"]["monthly"]
    assert set(d["quotes"]) <= set(d["dates"]) and len(d["quotes"]) == 5
    k, iv = d["quotes"]["2026-11-25"]["30"]
    assert len(k) == len(iv) > 5
    assert json.loads(json.dumps(d)) == d


def test_page_builds_without_placeholders_and_states_the_caveats(built):
    html = build_html(lab_data(*built, days=10))
    assert "function sviW" in html and 'id="surface-lab"' in html
    for p in ("$js", "$data", "$first", "$updated"):
        assert p not in html
    for words in ("not investment advice", "no live option feed", "SVI", "end-of-day"):
        assert words in html


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_page_scripts_parse(built):
    html = build_html(lab_data(*built, days=10))
    for i, src in enumerate(re.findall(r"<script>(.*?)</script>", html, re.S)):
        out = subprocess.run(["node", "--check", "-"], input=src, capture_output=True, text=True)
        assert out.returncode == 0, (i, out.stderr[:500])
