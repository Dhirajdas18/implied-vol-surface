import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from iv import surface as s

JS = Path(__file__).resolve().parents[1] / "src" / "iv" / "iv.js"
pytestmark = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")

SLICES = [
    {"a": 0.01, "b": 0.08, "rho": -0.6, "m": 0.02, "sigma": 0.12},
    {"a": -0.0410, "b": 0.1331, "rho": 0.3060, "m": 0.3586, "sigma": 0.4153},  # Vogt: g < 0
    {"a": 0.0004, "b": 0.0123, "rho": -0.95, "m": -0.003, "sigma": 0.0101},  # a short expiry
]


def _node(expr: str):
    # The script goes in on stdin: Linux caps a single command-line argument at 128 KB.
    script = f"const g = require({json.dumps(str(JS))});console.log(JSON.stringify({expr}));"
    out = subprocess.run(["node", "-"], input=script, capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("p", SLICES)
def test_browser_svi_variance_vol_and_g_match_python(p):
    k = np.linspace(-0.6, 0.6, 121)
    T = 0.137
    js = _node(
        f"(() => {{ const p = {json.dumps(p)}, k = {json.dumps(k.tolist())};"
        f"return {{ w: k.map((x) => g.sviW(p, x)), iv: k.map((x) => g.sviIv(p, x, {T})),"
        f"g: k.map((x) => g.sviG(p, x)) }}; }})()"
    )
    assert js["w"] == pytest.approx(s.svi_w(p, k).tolist(), rel=1e-9, abs=1e-15)
    assert js["iv"] == pytest.approx(s.svi_iv(p, k, T).tolist(), rel=1e-9, abs=1e-15)
    assert js["g"] == pytest.approx(s.svi_g(p, k).tolist(), rel=1e-9, abs=1e-12)


def test_browser_iv_from_w_clips_negative_variance():
    assert _node("[g.ivFromW(0.0025, 0.25), g.ivFromW(-1e-6, 0.25)]") == pytest.approx([0.1, 0.0])
