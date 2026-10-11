"""The web chat page: what the server sends, and how the page behaves in a real browser.

The browser tests drive an installed Chrome or Edge with Playwright. They are skipped when
Playwright (pip install playwright) or a browser is not available.
"""
import base64
import hashlib
import re
import time

import pytest
from conftest import ADMIN, HR, SAMPLES, STAFF, TOKENS, free_port, serve
from test_access import FINANCE_ONLY, HR_ONLY

from kb.api import build_app
from kb.config import TokenTable
from kb.text import split_sentences


def page_html(server) -> str:
    with server.http(None) as anonymous:
        return anonymous.get("/").text


def test_the_page_is_public_static_and_locked_down(server):
    with server.http(None) as anonymous, server.http(HR) as hr:
        plain, with_token = anonymous.get("/"), hr.get("/")
    assert plain.status_code == 200 and plain.headers["content-type"].startswith("text/html")
    assert plain.content == with_token.content  # the same bytes for everyone: nothing personal in it
    html = plain.text
    policy = plain.headers["content-security-policy"]
    for tag in ("script", "style"):
        [block] = re.findall(rf"<{tag}>(.*?)</{tag}>", html, re.S)
        digest = base64.b64encode(hashlib.sha256(block.encode()).digest()).decode()
        assert f"{tag}-src 'sha256-{digest}'" in policy  # only this page's own script and style may run
    assert "unsafe-inline" not in policy and "unsafe-eval" not in policy
    assert "default-src 'none'" in policy and "connect-src 'self'" in policy and "frame-ancestors 'none'" in policy
    assert plain.headers["x-frame-options"] == "DENY" and plain.headers["cache-control"] == "no-store"
    assert plain.headers["referrer-policy"] == "no-referrer" and plain.headers["x-content-type-options"] == "nosniff"
    assert "innerHTML" not in html and "localStorage" not in html  # text is set as text; the token stays in the tab


def test_the_page_never_contains_document_text_or_tokens(server):
    html = page_html(server)
    for path in SAMPLES.rglob("*.md"):
        for sentence in split_sentences(path.read_text(encoding="utf-8")):
            if len(sentence) > 25:
                assert sentence not in html, sentence
    assert not [s for s in HR_ONLY + FINANCE_ONLY if s in html]
    assert not [token for token in TOKENS if token in html]


def test_every_api_call_the_page_makes_needs_a_token(server):
    calls = set(re.findall(r'api\("(GET|POST)", "(/[^"]*)"', page_html(server)))
    assert calls == {("GET", "/me"), ("GET", "/documents"), ("POST", "/ask")}
    with server.http(None) as anonymous, server.http(STAFF) as staff:
        for method, path in calls:
            body = {"question": "How long must a password be?"} if method == "POST" else None
            assert anonymous.request(method, path, json=body).status_code == 401, path
            assert staff.request(method, path, json=body).status_code == 200, path


def test_me_tells_the_page_who_is_signed_in(server):
    with server.http(STAFF) as staff, server.http(ADMIN) as admin:
        assert staff.get("/me").json() == {"name": "maria (sales)", "role": "staff", "collections": ["general"],
                                           "admin": False}
        assert admin.get("/me").json()["collections"] == ["finance", "general", "hr"]


# ---------------------------------------------------------------------------- in a real browser
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


def sign_in(page, url: str, token: str) -> None:
    page.goto(url + "/")
    page.fill("#token", token)
    page.click("#signin-form button[type=submit]")


def ask(page, question: str):
    count = page.locator("article.answer:not(.pending)").count()
    page.fill("#question", question)
    page.press("#question", "Enter")
    card = page.locator("article.answer:not(.pending)").nth(count)
    card.wait_for()
    return card


def test_browser_sign_in_ask_and_open_a_cited_source(server, browser):
    page = browser.new_page(viewport={"width": 1440, "height": 900})
    sign_in(page, server.url, "not-a-real-token-123456")
    page.locator("#signin-fail").wait_for()
    assert page.locator("#signin-fail").inner_text() == "Access denied: this token is not valid."

    sign_in(page, server.url, STAFF)
    page.locator("#docs .doc").first.wait_for()
    assert page.locator("#docs .doc").count() == 6 and page.locator("#who-name").inner_text() == "maria (sales)"
    assert STAFF not in page.url

    answer = ask(page, "How many days of annual leave do full-time employees get?")
    assert answer.get_attribute("data-status") == "answered"
    assert "Extractive" in answer.locator(".mode").inner_text()
    answer.locator(".chip").first.click()
    panel = page.locator("#panel")
    panel.locator("mark").wait_for()
    assert panel.locator("h2").inner_text() == "Leave Policy"
    assert panel.locator("mark").inner_text() == (
        "Full-time employees receive 25 days of paid annual leave per calendar year, plus public holidays.")
    assert "Section: Annual leave" in panel.inner_text()

    refused = ask(page, "Can I bring my dog to the office?")
    assert refused.get_attribute("data-status") == "not_found"
    assert refused.locator(".chip").count() == 0 and refused.locator(".mode").count() == 0
    assert refused.evaluate("e => getComputedStyle(e).borderTopStyle") == "dashed"  # looks different
    assert answer.evaluate("e => getComputedStyle(e).borderTopStyle") == "solid"

    hr_only = ask(page, "What is the salary range for Band C?")
    assert hr_only.get_attribute("data-status") == "not_found" and "38,000" not in page.content()
    page.close()


def test_browser_pdf_sources_show_the_page_and_hr_sees_hr(server, browser):
    page = browser.new_page(viewport={"width": 390, "height": 844})  # a phone
    sign_in(page, server.url, HR)
    page.locator("#question").wait_for()
    pdf = ask(page, "What does error code E31 mean on the pallet tracker?")
    pdf.locator(".src").first.click()
    page.locator("#panel mark").wait_for()
    assert "Page 3" in page.locator("#panel").inner_text()
    page.click("#panel .close")
    salary = ask(page, "What is the salary range for Band C?")
    assert salary.get_attribute("data-status") == "answered" and "38,000" in salary.inner_text()
    page.close()


def test_browser_shows_access_denied_when_a_token_is_withdrawn(kb, browser):
    tokens = TokenTable(TOKENS)
    port = free_port()
    running = serve(build_app(kb, tokens), port)
    try:
        page = browser.new_page()
        sign_in(page, f"http://127.0.0.1:{port}", HR)
        page.locator("#question").wait_for()
        ask(page, "What is the salary range for Band C?").locator(".chip").first.click()
        page.locator("#panel mark").wait_for()
        tokens.withdraw_all()  # the administrator withdraws every token
        page.fill("#question", "How long must a password be?")
        page.press("#question", "Enter")
        denied = page.locator('article.answer[data-status="denied"]')
        denied.wait_for()
        assert "Access denied" in denied.inner_text()
        assert denied.evaluate("e => getComputedStyle(e).backgroundColor") != "rgb(255, 255, 255)"
        # Everything shown with the old token is gone: the HR answer, its source panel and the documents list.
        assert "38,000" not in page.inner_text("body") and page.locator("#panel").is_hidden()
        assert page.locator("article.answer").count() == 1 and page.locator("#docs .doc").count() == 0
        page.close()
    finally:
        running.should_exit = True


def test_browser_small_phone_header_fits(server, browser):
    for width in (320, 360):
        page = browser.new_page(viewport={"width": width, "height": 640})
        sign_in(page, server.url, STAFF)
        page.locator("#question").wait_for()
        title = page.locator(".brand h1").bounding_box()
        docs, signout = page.locator("#docs-toggle").bounding_box(), page.locator("#signout").bounding_box()
        assert title["x"] + title["width"] <= docs["x"] and docs["x"] + docs["width"] <= signout["x"], width
        assert signout["x"] + signout["width"] <= width and page.locator("#signout").inner_text() == "Sign out"
        page.close()


def test_browser_sign_out_forgets_the_previous_persons_answers(server, browser):
    page = browser.new_page(viewport={"width": 1280, "height": 800})
    sign_in(page, server.url, HR)
    page.locator("#question").wait_for()
    ask(page, "What is the salary range for Band C?").locator(".chip").first.click()
    page.locator("#panel mark").wait_for()
    page.click("#signout")
    page.locator("#signin").wait_for()
    assert page.locator("#panel").inner_html() == ""  # the hidden panel no longer holds the HR passage
    page.fill("#token", STAFF)  # the next person signs in on the same page, without a reload
    page.click("#signin-form button[type=submit]")
    page.locator("#docs .doc").first.wait_for()
    body = page.inner_text("body")
    assert "38,000" not in body and "Band C" not in body and page.locator("article.answer").count() == 0
    assert page.locator("#panel").is_hidden() and page.locator("#empty").is_visible()
    assert page.locator("#panel").inner_html() == ""  # the hidden panel no longer holds the HR passage
    page.close()


def test_browser_only_real_citations_become_chips(server, browser):
    trap = "Fire drill\nThe fire drill is on Tuesdays [2] and [3] at noon. See [1] for the assembly point."
    with server.http(ADMIN) as admin:
        assert admin.post("/admin/ingest", json={"collection": "general", "filename": "fire-drill.txt", "text": trap}).status_code == 200
    page = browser.new_page()
    sign_in(page, server.url, STAFF)
    page.locator("#question").wait_for()
    answer = ask(page, "When is the fire drill?")
    assert "Tuesdays [2] and [3]" in answer.inner_text()  # shown as plain text
    assert [b.inner_text() for b in answer.locator(".chip").all()] == ["1"]  # one real citation, one chip
    page.close()


def test_browser_shows_document_text_as_text_never_as_code(server, browser):
    trap = "Kitchen rules\nThe kitchen code is <img src=x onerror=\"document.title='pwned'\"> 4321, changed on Fridays."
    with server.http(ADMIN) as admin:
        assert admin.post("/admin/ingest", json={"collection": "general", "filename": "kitchen.txt", "text": trap}).status_code == 200
    page = browser.new_page()
    sign_in(page, server.url, STAFF)
    page.locator("#question").wait_for()
    answer = ask(page, "What is the kitchen code?")
    assert "<img src=x" in answer.inner_text() and answer.locator("img").count() == 0
    answer.locator(".chip").first.click()
    page.locator("#panel mark").wait_for()
    assert page.locator("#panel img").count() == 0 and page.title() == "Ask the documents"
    page.close()


def test_browser_drops_an_answer_still_on_its_way_when_the_person_signs_out(kb, browser):
    from conftest import running_server

    real_ask = kb.ask

    def slow_ask(*args, **kwargs):
        time.sleep(1.5)  # an answer that takes a while (a model call, a slow disk)
        return real_ask(*args, **kwargs)

    kb.ask = slow_ask
    with running_server(kb) as server:
        page = browser.new_page()
        sign_in(page, server.url, HR)
        page.locator("#question").wait_for()
        page.fill("#question", "What is the salary range for Band C?")
        page.press("#question", "Enter")
        page.locator("article.answer.pending").wait_for()
        page.click("#signout")  # while the HR answer is still on its way
        page.fill("#token", STAFF)  # the next person signs in on the same page (no reload)
        page.click("#signin-form button[type=submit]")
        page.locator("#docs .doc").first.wait_for()
        time.sleep(2.5)  # the HR reply has arrived by now
        assert "38,000" not in page.inner_text("body") and page.locator("article.answer").count() == 0
        assert "38,000" not in page.content()
        page.close()
