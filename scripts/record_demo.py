"""Re-record assets/demo.gif from the running stack, so the GIF never drifts from the UI.

Needs `docker compose up`, Playwright (`pip install playwright && playwright install chromium`)
and ffmpeg.

    python scripts/record_demo.py --url http://localhost:8501
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

VIEWPORT = {"width": 1280, "height": 800}
SKIP_S = 1.5  # page load


def wait_for_result(page: Page, pause_ms: int = 1800) -> None:
    page.wait_for_selector("text=round trip", timeout=60_000)
    page.wait_for_load_state("networkidle")
    page.wait_for_timeout(pause_ms)


def pick_sample(page: Page, name: str) -> None:
    page.get_by_test_id("stSelectbox").click()
    page.wait_for_timeout(400)
    page.get_by_role("option", name=name).click()
    wait_for_result(page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", default="assets/demo.gif")
    parser.add_argument("--width", type=int, default=960)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmp, sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(
            viewport=VIEWPORT,
            record_video_dir=tmp,
            record_video_size=VIEWPORT,
            color_scheme="light",
        )
        page = context.new_page()
        started = time.monotonic()
        page.goto(args.url)
        wait_for_result(page, 2500)

        # Lower the threshold step by step: more, less certain boxes appear.
        slider = page.get_by_role("slider")
        slider.focus()
        for _ in range(6):
            page.keyboard.press("ArrowLeft")
            page.wait_for_timeout(700)
        wait_for_result(page, 2200)

        pick_sample(page, "bedroom.jpg")
        length = time.monotonic() - started - SKIP_S  # cut before the page closes (black frame)
        video = page.video.path() if page.video else None
        context.close()
        browser.close()
        if video is None:
            raise SystemExit("No video recorded.")

        palette = Path(tmp) / "palette.png"
        scale = f"fps=8,scale={args.width}:-1:flags=lanczos"
        cut = ["-ss", str(SKIP_S), "-t", f"{length:.2f}", "-i", video]
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *cut,
             "-vf", f"{scale},palettegen=stats_mode=diff", palette],
            check=True,
        )  # fmt: skip
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", *cut, "-i", palette,
             "-lavfi", f"{scale}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5", args.out],
            check=True,
        )  # fmt: skip
    print(f"{args.out}: {Path(args.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
