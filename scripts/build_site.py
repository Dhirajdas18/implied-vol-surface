"""Build site/index.html from the committed docs/data/lab_surfaces.json (no download needed)."""

import json
import time
from pathlib import Path

from iv.data import LAB_JSON
from iv.site import build_html


def main() -> None:
    t0 = time.perf_counter()
    data = json.loads(LAB_JSON.read_text(encoding="utf-8"))
    out = Path("site/index.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(build_html(data), encoding="utf-8")
    n = sum(len(v) for v in data["slices"].values())
    print(
        f"{out} ({out.stat().st_size / 1e6:.2f} MB): {len(data['dates'])} dates, {n} slices, "
        f"quotes for {len(data['quotes'])} dates, built in {time.perf_counter() - t0:.1f} s"
    )


if __name__ == "__main__":
    main()
