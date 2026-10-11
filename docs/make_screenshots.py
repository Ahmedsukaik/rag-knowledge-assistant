"""Screenshots of the chat page, taken from the real running app with the demo documents:

    pip install playwright        (once; it drives your installed Chrome or Edge, no browser download)
    python docs/make_screenshots.py

Writes docs/img/chat.png (1440 x 960 at 2x) and docs/img/chat-phone.png (390 x 844 at 2x).
The questions below are typed into the page and answered by the server; nothing is mocked.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from demo import STAFF, TOKENS, free_port, serve  # noqa: E402
from kb.api import build_app  # noqa: E402
from kb.config import TokenTable  # noqa: E402
from kb.ingest import ingest_folder  # noqa: E402
from kb.retrieval import build_retriever  # noqa: E402
from kb.service import KnowledgeBase  # noqa: E402
from kb.store import Store  # noqa: E402

OUT = ROOT / "docs" / "img"
QUESTIONS = [
    "What is the hotel limit per night in London?",
    "Can I bring my dog to the office?",
    "Is the pallet scanner still covered if it breaks after a year?",  # other words than the manual's
]


def start_server() -> str:
    work = Path(tempfile.mkdtemp(prefix="kb-shots-"))
    shutil.copytree(ROOT / "samples" / "docs", work / "docs")
    store = Store(str(work / "kb.db"))
    retriever = build_retriever(store, "hybrid")  # the default search: keywords and meaning
    ingest_folder(store, work / "docs", embedder=getattr(retriever, "embedder", None))
    port = free_port()
    serve(build_app(KnowledgeBase(store, retriever), TokenTable(TOKENS)), port)
    return f"http://127.0.0.1:{port}"


def launch(playwright):
    for channel in ("chrome", "msedge"):
        try:
            return playwright.chromium.launch(channel=channel)
        except Exception:
            continue
    raise SystemExit("Chrome or Edge is needed (Playwright uses the installed browser).")


def ask(page, question: str, count: int) -> None:
    page.fill("#question", question)
    page.press("#question", "Enter")
    # (a locator, not wait_for_function: the page's Content-Security-Policy blocks evaluated strings)
    page.locator("article.answer:not(.pending)").nth(count - 1).wait_for()


def main() -> None:
    from playwright.sync_api import sync_playwright

    url = start_server()
    OUT.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = launch(p)
        page = browser.new_page(viewport={"width": 1440, "height": 960}, device_scale_factor=2)
        page.goto(url + "/")
        page.fill("#token", STAFF)
        page.click("#signin-form button[type=submit]")
        page.wait_for_selector("#docs .doc")
        for n, question in enumerate(QUESTIONS, start=1):
            ask(page, question, n)
        page.locator("article.answer").nth(2).locator(".chip").first.click()
        page.wait_for_selector("#panel mark")
        page.screenshot(path=str(OUT / "chat.png"))

        phone = browser.new_page(viewport={"width": 390, "height": 844}, device_scale_factor=2)
        phone.goto(url + "/")
        phone.fill("#token", STAFF)
        phone.click("#signin-form button[type=submit]")
        phone.wait_for_selector("#question")
        ask(phone, QUESTIONS[0], 1)
        ask(phone, QUESTIONS[1], 2)
        phone.screenshot(path=str(OUT / "chat-phone.png"))
        browser.close()
    for name in ("chat.png", "chat-phone.png"):
        print(f"wrote docs/img/{name} ({(OUT / name).stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
