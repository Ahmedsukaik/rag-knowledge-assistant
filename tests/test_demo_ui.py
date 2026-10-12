"""The web demo page in a real browser (an installed Chrome or Edge, through Playwright): the six
step story at four screen sizes with the thing each step points at fully on screen, a phone
tapped with a finger, free use, the AI path through a simulated API, keyword-only search,
motion, focus and dark mode. Skipped when Playwright or a browser is not available."""
from __future__ import annotations

from typing import Any

import pytest
from conftest import free_port, serve
from demo_fakes import FakeAnthropic

from kb.demo_index import DemoIndex
from kb.demo_web import create_app

SIZES = [(1440, 900), (1280, 720), (390, 844), (360, 640)]


@pytest.fixture(scope="module")
def browser():
    sync_api = pytest.importorskip("playwright.sync_api")
    with sync_api.sync_playwright() as p:
        for channel in ("chrome", "msedge"):
            try:
                launched = p.chromium.launch(channel=channel)
                break
            except Exception:
                continue
        else:
            pytest.skip("no installed Chrome or Edge for Playwright")
        yield launched
        launched.close()


@pytest.fixture(scope="module")
def index() -> DemoIndex:
    return DemoIndex()


def start(index: DemoIndex, client: Any = None) -> tuple[str, Any]:
    port = free_port()
    return f"http://127.0.0.1:{port}", serve(create_app(index, client=client), port)


@pytest.fixture(scope="module")
def site(index):
    url, server = start(index)
    yield url
    server.should_exit = True


class Page:
    """A page at one screen size, collecting script errors (a blocked style or script included)."""

    def __init__(self, browser, url: str, width: int = 1440, height: int = 900, **context: Any):
        self.phone = width < 900
        self.size = (width, height)
        self.context = browser.new_context(viewport={"width": width, "height": height}, is_mobile=self.phone,
                                           has_touch=self.phone, **context)
        self.page = self.context.new_page()
        self.errors: list[str] = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.on("console", lambda m: self.errors.append(m.text) if m.type == "error" else None)
        self.page.goto(url + "/")
        self.button("Start the demo").wait_for()

    def button(self, name: str):
        return self.page.get_by_role("button", name=name, exact=True)

    def title(self, text: str) -> None:
        self.page.locator(".story-title", has_text=text).wait_for()
        self.page.wait_for_timeout(150)  # a frame for the step to bring its target into view

    def on_screen(self, locator) -> None:
        """Fully inside the window, and nothing (such as the story sheet) on top of it."""
        locator.wait_for(state="visible")
        box = locator.bounding_box()
        width, height = self.size
        assert box, locator
        assert box["x"] >= -0.5 and box["y"] >= -0.5, (locator, box)
        assert box["x"] + box["width"] <= width + 0.5 and box["y"] + box["height"] <= height + 0.5, (locator, box, self.size)
        assert locator.evaluate("""el => { const r = el.getBoundingClientRect();
            const hit = document.elementFromPoint(r.x + r.width / 2, r.y + r.height / 2);
            return !!hit && (hit === el || el.contains(hit) || hit.contains(el)); }"""), f"{locator} is covered"
        assert self.page.evaluate("document.documentElement.scrollWidth") <= width, "the page scrolls sideways"

    def tap(self, locator) -> None:
        self.on_screen(locator)
        if self.phone:
            locator.tap()
        else:
            locator.click()

    def card(self, persona: str):
        return self.page.locator(".card.answer", has_text=f"{persona} asked").first

    def close(self) -> None:
        self.context.close()
        assert self.errors == []


@pytest.mark.parametrize("width, height", SIZES, ids=[f"{w}x{h}" for w, h in SIZES])
def test_the_six_step_story_with_each_target_on_screen(browser, site, width, height):
    ui = Page(browser, site, width, height)
    page = ui.page
    assert "Answers you can check." in page.locator("#story").inner_text()
    ui.tap(ui.button("Start the demo"))

    ui.title("A policy question.")                                         # 1
    ui.tap(ui.button("Ask Maria's question"))
    ui.title("Answered, with its source.")
    footnote = ui.card("Maria").locator(".fn").first
    ui.on_screen(footnote)
    if ui.phone:
        ui.tap(footnote)
        page.locator("body[data-pane='source'] #desk mark.hl.active").first.wait_for()
    ui.on_screen(page.locator("#desk mark.hl.active").first)
    assert "25 days of paid annual leave" in page.locator("#desk mark.hl.active").first.inner_text()
    ui.tap(page.get_by_role("button", name="Next: in her own words"))

    ui.title("Ask in your own words.")                                     # 2
    ui.tap(ui.button("Ask it"))
    ui.title("Found by meaning, on page 4 of a PDF.")
    ui.on_screen(page.locator(".card .kw-note"))
    assert "A plain word search would have refused this" in page.locator(".card .kw-note").inner_text()
    assert "Keyword search alone would have refused this question." in page.locator("#story").inner_text()
    assert page.locator(".card .meta").is_hidden()  # compact cards during the story
    if not ui.phone:
        ui.on_screen(page.locator("#desk mark.hl.active").first)
        assert "page 4" in page.locator("#desk").inner_text().lower()
    ui.tap(page.get_by_role("button", name="Next: who can see what"))

    ui.title("Who can see what.")                                          # 3
    ui.tap(ui.button("Ask as Maria"))
    ui.title("Maria gets nothing.")
    ui.on_screen(page.locator(".card .stamp"))
    ui.tap(ui.button("Ask as Hana"))
    ui.title("Hana gets the answer, with its source.")
    hana = ui.card("Hana")
    ui.on_screen(hana.locator(".fn").first)
    assert "38,000" in hana.inner_text() and "Hana can open General and HR documents" in hana.inner_text()
    assert "38,000" not in ui.card("Maria").inner_text()
    ui.tap(page.get_by_role("button", name="Next: catch a bad AI draft"))

    ui.title("Catch a bad AI draft.")                                      # 4
    ui.tap(ui.button("Send the forced draft"))
    ui.title("Held before anyone saw it.")
    ui.on_screen(page.locator(".held"))
    assert "30 days" in page.locator(".held .draft").inner_text()
    assert "A cited statement has a number that its source does not." in page.locator(".held").inner_text()
    ui.tap(page.get_by_role("button", name="Next: where it still misses"))

    ui.title("Where it still misses.")                                     # 5
    ui.tap(ui.button("Ask about sick days"))
    ui.title("A near miss.")
    ui.on_screen(ui.card("Maria").locator(".src").first)
    assert "The footnote makes that easy to spot." in page.locator("#story").inner_text()
    assert page.locator(".card .kw-note").count() == 0  # not on the near miss
    ui.tap(page.get_by_role("button", name="Next: what you'd get"))

    ui.title("What you'd get on your documents.")                          # 6
    ui.on_screen(page.locator(".report .takeaways"))
    assert "0 leaks in 606 test questions" in page.locator(".report .takeaways").inner_text()
    assert page.locator(".report .eval").is_hidden()  # the tables wait behind "How it was tested"
    if width < 900:
        assert page.locator("#switch-row").is_hidden()
    report = page.locator(".report").inner_text()
    assert "$0" in report and "14 of 21" in report and "10 of 16" in report and "after deployment" not in report
    page.locator(".tested summary").click()
    assert "New questions A" in page.locator(".report .eval").inner_text() and page.locator(".report .eval").is_visible()
    ui.tap(page.get_by_role("button", name="Try it yourself"))
    page.locator(".story-title", has_text="Ask anything an employee would.").wait_for()
    assert page.locator("#input").is_visible() and page.locator(".chip").count() >= 6
    ui.close()


def test_a_phone_works_with_single_taps(browser, site):
    ui = Page(browser, site, 390, 844)
    page = ui.page
    ui.button("Start the demo").tap()
    ui.button("Ask Maria's question").tap()
    ui.title("Answered, with its source.")
    ui.card("Maria").locator(".fn").first.tap()           # one tap opens the source; nothing needs a hover
    assert page.evaluate("document.body.dataset.pane") == "source"
    page.locator("#desk mark.hl.active").first.wait_for()
    page.locator(".switch-row button", has_text="Answer").tap()
    assert page.evaluate("document.body.dataset.pane") == "answer"
    page.locator(".switch-row button", has_text="Source").tap()
    assert page.locator("#desk mark.hl.active").is_visible()
    small = page.evaluate("""() => [...document.querySelectorAll('button, a, summary')].filter(el => el.getClientRects().length)
        .map(el => { const r = el.getBoundingClientRect(); const fn = el.classList.contains('fn');
          const before = fn ? getComputedStyle(el, '::before') : null;
          const w = fn ? r.width + 24 : r.width, h = fn ? r.height + 24 : r.height;
          return { text: el.textContent.trim().slice(0, 30), w, h }; })
        .filter(x => x.w < 44 || x.h < 40)""")
    assert small == [], small  # touch targets: 44 px (the segmented switch buttons are 40 px tall, 170 px wide)
    ui.close()


def test_free_use_with_chips_personas_and_a_typed_question(browser, site):
    ui = Page(browser, site)
    page = ui.page
    page.get_by_role("button", name="Skip to free use").click()
    page.locator(".story-title", has_text="Ask anything").wait_for()
    assert page.locator(".chip").first.inner_text() == "Hotel limit in London"  # everyday questions first
    assert "try to break it:" in page.locator("#chips").inner_text().lower()
    page.locator(".chip", has_text="A document with a hidden instruction").click()
    card = ui.card("Maria")
    card.locator(".warn").wait_for()
    assert "addressed to AI assistants" in card.locator(".warn").inner_text()
    page.locator("#desk .planted").first.wait_for()
    assert "Flagged when this file was read in" in page.locator("#desk .planted-note").inner_text()
    assert "ignore all previous" not in card.inner_text()
    page.locator(".persona", has_text="Hana").click()
    page.locator("#input").fill("When is the pay review?")
    page.locator("#input").press("Enter")
    ui.card("Hana").locator(".fn").first.wait_for()
    assert "Matched by meaning" in ui.card("Hana").inner_text()
    page.locator(".chip", has_text="Payment runs, as Maria").click()
    page.locator(".card.answer .stamp").first.wait_for()
    refusal = ui.card("Maria").inner_text()
    assert "treated exactly like one that does not exist" in refusal and "Try naming the document or product." in refusal
    assert "Leave Policy" in refusal and "Salary Bands" not in refusal
    page.locator("#desk .stamp").wait_for()  # the source pane shows the refusal, not an empty page
    page.locator("#input").fill("can I bring my cat to work")
    page.locator("#input").press("Enter")
    page.locator(".card.answer .a-q", has_text="can I bring my cat to work").wait_for()
    assert "Nothing in the documents Maria can open answers this." in ui.card("Maria").inner_text()
    assert page.locator(".persona[aria-pressed='true']").inner_text().startswith("M")
    ui.close()


def test_text_is_shown_as_text(browser, site):
    ui = Page(browser, site)
    page = ui.page
    page.get_by_role("button", name="Skip to free use").click()
    hostile = '<img src=x onerror="window.__xss=1"> how much annual leave do I get?'
    page.locator("#input").fill(hostile)
    page.locator("#input").press("Enter")
    ui.card("Maria").wait_for()
    page.wait_for_timeout(300)
    assert ui.card("Maria").locator(".a-q").inner_text() == hostile
    assert page.locator(".card img").count() == 0 and page.evaluate("window.__xss") is None
    ui.close()


def test_with_the_ai_on_answers_are_marked_and_priced(browser, index):
    url, server = start(index, FakeAnthropic())
    try:
        ui = Page(browser, url)
        page = ui.page
        assert "Answers are written by Claude Haiku 5.5. Every sentence is checked" in page.locator("#story").inner_text()
        ui.button("Start the demo").click()
        ui.button("Ask Maria's question").click()
        ui.title("Answered, with its source.")
        card = ui.card("Maria")
        assert "Written by Claude Haiku 5.5, checked" in card.locator(".by").inner_text()
        for name in ("Next: in her own words", "Ask it", "Next: who can see what", "Ask as Maria", "Ask as Hana",
                     "Next: catch a bad AI draft", "Send the forced draft", "Next: where it still misses", "Ask about sick days",
                     "Next: what you'd get"):
            page.get_by_role("button", name=name).click()
            page.wait_for_timeout(100)
            page.locator("#story-actions .btn-primary").wait_for()
        ui.title("What you'd get")
        figure = page.locator(".report .cost-figure").inner_text()
        assert "measured, this run" in figure.lower() and "$0.00060" in figure and "4 answers written by Claude" in figure
        page.get_by_role("button", name="Try it yourself").click()
        page.locator(".chip", has_text="Hotel limit in London").click()
        free = ui.card("Maria")
        free.locator(".meta").wait_for()
        assert "Cost $0.00015 (1,200 tokens in, 60 out)" in free.inner_text()
        assert page.locator("[data-ai='on']").is_visible()  # the choice between AI and quotes, when the AI is on
        ui.close()
    finally:
        server.should_exit = True


def test_keyword_only_search_is_labelled(browser):
    url, server = start(DemoIndex(model_folder=None))
    try:
        ui = Page(browser, url)
        page = ui.page
        page.get_by_role("button", name="Skip to free use").click()
        assert "searched by keywords only" in page.locator(".composer-meta").inner_text()
        page.locator("#input").fill("what is the hotel limit per night in London")
        page.locator("#input").press("Enter")
        card = ui.card("Maria")
        card.locator(".fn").first.wait_for()
        assert "Matched by keywords only" in card.inner_text() and "meaning model is not on this server" in card.inner_text()
        ui.close()
    finally:
        server.should_exit = True


def test_the_page_keeps_its_place_after_a_reload(browser, site):
    ui = Page(browser, site)
    page = ui.page
    ui.button("Start the demo").click()
    ui.button("Ask Maria's question").click()
    ui.title("Answered, with its source.")
    page.reload()
    ui.title("Answered, with its source.")
    assert ui.card("Maria").count() == 1
    page.locator("#desk mark.hl.active").first.wait_for()
    ui.close()


def test_motion_stops_when_the_visitor_asks_for_less(browser, site):
    ui = Page(browser, site, reduced_motion="reduce")
    page = ui.page
    assert page.locator(".headline .hl-word").evaluate("el => getComputedStyle(el).animationName") == "none"
    ui.button("Start the demo").click()
    ui.button("Ask Maria's question").click()
    ui.title("Answered")
    assert page.locator("#desk mark.hl.active").first.evaluate("el => getComputedStyle(el).animationName") == "none"
    assert page.locator("#desk mark.hl.active").first.evaluate("el => getComputedStyle(el).backgroundSize") == "100% 100%"
    ui.close()


def test_keyboard_focus_is_visible(browser, site):
    ui = Page(browser, site)
    page = ui.page
    for _ in range(8):
        page.keyboard.press("Tab")
        if page.evaluate("document.activeElement.textContent.trim()") == "Start the demo":
            break
    assert page.evaluate("document.activeElement.textContent.trim()") == "Start the demo"
    assert page.evaluate("getComputedStyle(document.activeElement).outlineStyle") != "none"
    page.keyboard.press("Enter")
    ui.title("A policy question.")
    assert page.evaluate("document.activeElement.textContent.trim()") == "Ask Maria's question"  # focus moves on
    ui.close()


def test_dark_mode_keeps_the_pages_white_and_readable(browser, site):
    ui = Page(browser, site, color_scheme="dark")
    page = ui.page
    ui.button("Start the demo").click()
    ui.button("Ask Maria's question").click()
    ui.title("Answered")
    assert page.evaluate("getComputedStyle(document.body).backgroundColor") == "rgb(20, 20, 18)"
    assert page.locator(".card.answer").evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(255, 255, 255)"
    assert page.locator("#desk .page").evaluate("el => getComputedStyle(el).backgroundColor") == "rgb(255, 255, 255)"
    assert page.locator(".card.answer .a-q").evaluate("el => getComputedStyle(el).color") == "rgb(28, 27, 25)"
    ui.close()


def test_no_jargon_on_the_way(browser, site):
    ui = Page(browser, site)
    page = ui.page
    ui.button("Start the demo").click()
    ui.button("Ask Maria's question").click()
    ui.title("Answered")
    visible = page.evaluate("document.body.innerText").lower()
    for word in ("passage_id", "extractive", "bm25", "not_found", "token ", "chunk", "embedding", "rrf"):
        assert word not in visible, word
    ui.close()


def test_without_the_ai_quoted_mode_reads_as_a_choice_not_a_fault(browser, site):
    ui = Page(browser, site)
    page = ui.page
    story = page.locator("#story").inner_text()
    assert "This demo quotes the documents word for word, so nothing is made up." in story and "No AI key" not in story
    page.get_by_role("button", name="Skip to free use").click()
    page.locator("#input").fill("how much holiday do I get each year")
    page.locator("#input").press("Enter")
    ui.card("Maria").locator(".fn").first.wait_for()
    assert ui.card("Maria").locator(".why").count() == 0 and "Cost" not in ui.card("Maria").inner_text()
    assert page.locator("[data-ai]").count() == 0  # no "Written by AI" choice when the AI is off
    ui.close()


def test_a_phone_shows_a_preview_first_and_asks_from_the_sheet(browser, site):
    ui = Page(browser, site, 390, 664)
    page = ui.page
    ui.on_screen(page.locator("#desk .spec-front"))
    assert page.locator("#desk .spec-front").evaluate("el => getComputedStyle(el).transform") == "none"
    assert page.locator(".lede").is_visible()
    page.get_by_role("button", name="Skip to free use").tap()
    ui.on_screen(page.locator("#story #input"))  # the question box sits in the bottom sheet
    assert page.locator("#input").get_attribute("placeholder") == "Ask a question..."
    assert page.locator("#repo").inner_text().strip() == "Code"
    ui.close()
