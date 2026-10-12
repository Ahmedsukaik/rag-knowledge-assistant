"""Record the web demo's tour (GIF) and its screenshots for the README.

    python docs/make_media.py                       starts the demo locally (quoted answers, no key)
    python docs/make_media.py https://<deployed>    or records a running copy, for example the live one

Needs Playwright with an installed Chrome or Edge, and ffmpeg on the PATH for the GIF.
Writes docs/img/demo-tour.gif and docs/img/demo-*.png:

    demo-0-intro.png   the first screen                        1440x900
    demo-2-meaning.png step 2, found by meaning on page 4       1440x900
    demo-3-access.png  step 3, Maria refused, Hana answered     1440x900
    demo-4-held.png    step 4, the forced bad draft held        1440x900
    demo-6-cost.png    step 6, what you'd get and the cost      1440x900
    demo-2-phone.png   step 2 on a phone                        390x664
"""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "img"
DESKTOP = {"width": 1440, "height": 900}
PHONE = {"width": 390, "height": 664}


def local_server() -> tuple[str, object]:
    import uvicorn

    sys.path.insert(0, str(ROOT))
    from kb.demo_web import create_app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(client=None), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server


def tour(browser, url: str, tmp: str) -> Path | None:
    context = browser.new_context(viewport=DESKTOP, device_scale_factor=1, record_video_dir=tmp, record_video_size=DESKTOP)
    page = context.new_page()

    def click(name: str, pause: int = 900) -> None:
        page.get_by_role("button", name=name, exact=True).click()
        page.wait_for_timeout(pause)

    def until(title: str, pause: int) -> None:
        page.locator(".story-title", has_text=title).wait_for(timeout=60000)
        page.wait_for_timeout(pause)

    page.goto(url + "/")
    page.get_by_role("button", name="Start the demo").wait_for()
    page.wait_for_timeout(2600)
    page.screenshot(path=str(OUT / "demo-0-intro.png"))
    click("Start the demo", 1300)
    click("Ask Maria's question", 0)
    until("Answered, with its source.", 3400)
    click("Next: in her own words", 1300)
    click("Ask it", 0)
    until("Found by meaning", 3400)
    page.screenshot(path=str(OUT / "demo-2-meaning.png"))
    click("Next: who can see what", 1300)
    click("Ask as Maria", 0)
    until("Maria gets nothing.", 2600)
    click("Ask as Hana", 0)
    until("Hana gets the answer", 3200)
    page.screenshot(path=str(OUT / "demo-3-access.png"))
    click("Next: catch a bad AI draft", 1300)
    click("Send the forced draft", 0)
    until("Held before anyone saw it.", 3600)
    page.screenshot(path=str(OUT / "demo-4-held.png"))
    click("Next: where it still misses", 1300)
    click("Ask about sick days", 0)
    page.get_by_role("button", name="Next: what you'd get").wait_for(timeout=60000)  # a near miss, or a refusal
    page.wait_for_timeout(3200)
    page.get_by_role("button", name="Next: what you'd get").click()
    until("What you'd get", 3600)
    page.screenshot(path=str(OUT / "demo-6-cost.png"))
    page.get_by_role("button", name="Try it yourself").click()
    page.wait_for_timeout(1200)
    page.locator(".chip", has_text="A document with a hidden instruction").click()
    page.locator("#desk .planted-note").wait_for(timeout=60000)
    page.wait_for_timeout(3600)
    video = page.video.path() if page.video else None
    context.close()
    return Path(video) if video else None


def phone_shot(browser, url: str) -> None:
    context = browser.new_context(viewport=PHONE, device_scale_factor=2, is_mobile=True, has_touch=True)
    page = context.new_page()
    page.goto(url + "/")
    page.get_by_role("button", name="Start the demo").tap()
    page.get_by_role("button", name="Ask Maria's question").tap()
    page.locator(".story-title", has_text="Answered").wait_for(timeout=60000)
    page.get_by_role("button", name="Next: in her own words").tap()
    page.get_by_role("button", name="Ask it").tap()
    page.locator(".story-title", has_text="Found by meaning").wait_for(timeout=60000)
    page.wait_for_timeout(1500)
    page.screenshot(path=str(OUT / "demo-2-phone.png"))
    context.close()


def to_gif(video: Path, tmp: str) -> None:
    webm = Path(tmp) / "tour.webm"
    shutil.copy(video, webm)
    palette = Path(tmp) / "palette.png"
    scale = "fps=6,scale=900:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-vf", f"{scale},palettegen=max_colors=64:stats_mode=diff",
                    str(palette)], check=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(webm), "-i", str(palette), "-lavfi",
                    f"{scale} [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                    str(OUT / "demo-tour.gif")], check=True)


def main(url: str | None) -> None:
    OUT.mkdir(exist_ok=True)
    server = None
    if url is None:
        url, server = local_server()
    tmp = tempfile.mkdtemp()  # left for the system to clean: Windows can keep the video file open for a moment
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(channel="chrome")
            video = tour(browser, url, tmp)
            phone_shot(browser, url)
            browser.close()
        if video:
            to_gif(video, tmp)
    finally:
        if server is not None:
            server.should_exit = True  # type: ignore[attr-defined]
    print("written:", ", ".join(sorted(f.name for f in OUT.glob("demo-*"))))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
