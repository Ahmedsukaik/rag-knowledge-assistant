"""The web demo's HTTP API (kb/demo_web.py), as the page uses it: the story, access rules per
persona, the Claude path with a simulated API, the fallbacks to quoted answers, keyword-only
search (plan B), limits, and what Vercel ships. No test calls the live API."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
from conftest import SAMPLES
from demo_fakes import FakeAnthropic, api_error, cited_reply, search_results
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient
from test_access import FINANCE_ONLY, HR_ONLY, restricted_queries

from kb.demo_index import CHIPS, DEMO_DIR, QUESTION_TOKENS, STORY, DemoIndex, sha256
from kb.demo_web import (
    MAX_BODY,
    MAX_TOKENS,
    MODEL,
    PER_VISITOR_AI,
    PER_VISITOR_EMBED,
    QUOTED_BECAUSE,
    SEARCH_NOTES,
    SECURITY_HEADERS,
    Limiter,
    create_app,
    live_client_from_env,
    visitor_key,
)

ROOT = SAMPLES.parent.parent


@pytest.fixture(scope="module")
def index() -> DemoIndex:
    return DemoIndex()  # with the meaning model, as on Vercel (plan A)


@pytest.fixture(scope="module")
def keywords_index() -> DemoIndex:
    return DemoIndex(model_folder=None)  # plan B: no model on the server


@pytest.fixture
def client(index) -> TestClient:
    return TestClient(create_app(index, client=None))


def ask(client: TestClient, question: str, persona: str = "maria", **extra: Any) -> dict[str, Any]:
    res = client.post("/api/ask", json={"question": question, "persona": persona, **extra})
    assert res.status_code == 200, res.text
    return res.json()


# ------------------------------------------------------------------ the story, with quoted answers (no key)
def test_config_says_what_the_demo_runs_on(client):
    cfg = client.get("/api/config").json()
    assert cfg["live"] is False and cfg["model"] is None and cfg["quoted_because"] == QUOTED_BECAUSE["no_key"]
    assert cfg["search"]["typed"] == "meaning" and cfg["search"]["note"] is None
    assert [p["id"] for p in cfg["personas"]] == ["maria", "hana", "farid"]
    assert cfg["story"] == STORY and cfg["chips"] == CHIPS and cfg["collections"] == {"general": 6, "hr": 1, "finance": 2}
    assert {d["title"] for d in cfg["documents"] if d["collection"] == "hr"} == {"Salary Bands 2027 (HR only)"}
    assert len(cfg["documents"]) == 9 and cfg["leak_checks"] == 606
    assert [c["group"] for c in cfg["chips"]][:6] == ["everyday"] * 6  # everyday questions first
    assert cfg["eval"]["heldout"]["answered"] == 14 and cfg["eval"]["heldout_v2"]["answered"] == 10
    assert cfg["repo"] == "https://github.com/Ahmedsukaik/rag-knowledge-assistant"


def test_the_six_steps_over_http(client):
    leave = ask(client, STORY["leave"])
    assert leave["status"] == "answered" and leave["written_by"] == "quoted" and leave["cost"]["usd"] == 0
    assert leave["citations"][0]["passage_id"] == "general/leave-policy#1" and "25 days" in leave["citations"][0]["quote"]
    assert leave["quoted_because"] is None and leave["search"]["by"] == "meaning"  # the page says it once, not per card

    scanner = ask(client, STORY["scanner"])
    assert scanner["status"] == "answered" and scanner["search"]["keywords_alone"] is False
    assert scanner["passages"][0]["page"] == 4 and "2-year warranty" in scanner["citations"][0]["quote"]

    maria = ask(client, STORY["salary"], "maria")
    assert maria["status"] == "not_found" and maria["citations"] == [] and maria["passages"] == []
    assert maria["written_by"] == "none" and maria["quoted_because"] is None
    hana = ask(client, STORY["salary"], "hana")
    assert hana["status"] == "answered" and "Band C" in hana["answer"] and hana["citations"][0]["passage_id"] == "hr/salary-bands#2"

    draft = ask(client, STORY["leave"], simulate="bad_draft")
    assert draft["status"] == "flagged" and draft["written_by"] == "simulated" and draft["cost"]["usd"] == 0
    assert "30 days" in draft["draft"] and "never expire" in draft["draft"]
    assert "A cited statement has a number that its source does not." in draft["problems"]
    assert "1 part of the answer had no source." in draft["problems"]
    assert "30 days" not in draft["answer"] and "25 days" in draft["answer"]  # the quotes are shown instead

    sick = ask(client, STORY["sick"])
    assert sick["status"] == "answered" and "annual leave" in sick["citations"][0]["quote"]  # the near miss the page explains


def test_a_passage_opens_its_whole_document_for_a_persona_who_may_read_it(client):
    res = client.get("/api/passage", params={"id": "general/pt-200-product-manual#5", "persona": "maria"})
    assert res.status_code == 200
    doc = res.json()["document"]
    assert doc["title"].startswith("PT-200") and doc["kind"] == "pdf" and doc["pages"] == 4
    assert [p["page"] for p in doc["passages"]][-1] == 4
    faq = client.get("/api/passage", params={"id": "general/office-faq#1", "persona": "maria"}).json()
    flagged = [p for p in faq["document"]["passages"] if p["flagged"]]
    assert flagged and flagged[0]["flagged"][0].startswith("Note for AI assistants")
    assert faq["audit"][0]["action"] == "get_passage" and faq["audit"][0]["outcome"] == "ok"


def test_the_planted_instruction_is_data_and_never_quoted(client):
    out = ask(client, "Where is lost property kept?")
    assert out["status"] == "answered" and out["warnings"] and "addressed to AI assistants" in out["warnings"][0]
    shown = out["answer"] + " ".join(c["quote"] for c in out["citations"])
    assert "ignore all previous" not in shown and "approved automatically" not in shown
    claim = ask(client, "Are expense claims approved automatically?")
    assert claim["citations"][0]["quote"] == "No claim is approved automatically."


@pytest.mark.parametrize("chip", CHIPS, ids=[c["label"] for c in CHIPS])
def test_every_chip_does_what_its_label_says(client, chip):
    out = ask(client, chip["question"], chip["persona"])
    refused = chip["label"] in ("Something it can't answer", "An error code it doesn't know", "Payment runs, as Maria")
    assert (out["status"] == "not_found") is refused, out["status"]
    assert chip.get("restricted", False) is (chip["label"] == "Payment runs, as Maria")
    if chip["label"] == "A document with a hidden instruction":
        assert out["warnings"]


# ------------------------------------------------------------------ access: personas never see other roles' text
def shown(out: dict[str, Any]) -> str:
    """Everything an answer shows except the visitor's own question (echoed in the card and the audit entry)."""
    audit = [{k: v for k, v in e.items() if k != "query"} for e in out["audit"]]
    return json.dumps({k: v for k, v in out.items() if k not in ("question", "audit")} | {"audit": audit}, ensure_ascii=False)
LEAK_CASES = [("maria", "hr", HR_ONLY), ("maria", "finance", FINANCE_ONLY), ("farid", "hr", HR_ONLY), ("hana", "finance", FINANCE_ONLY)]
KEYWORD_LEAK_CHECKS = 120


@pytest.mark.parametrize("persona, collection, secrets", LEAK_CASES)
def test_no_question_built_from_a_restricted_document_reaches_its_text(index, persona, collection, secrets):
    client = TestClient(create_app(index, client=None))
    queries = restricted_queries(collection)
    for query in queries:
        res = client.post("/api/ask", json={"question": query, "persona": persona})
        if res.status_code == 429:
            pytest.fail("limiter hit")
        assert res.status_code == 200, query
        out = res.json()
        assert not [s for s in secrets if s in shown(out)], query
        assert all(p["collection"] != collection for p in out["passages"]), query
    assert len(queries) > 50


def test_the_number_of_leak_checks_the_page_quotes_is_the_number_asked():
    from kb.demo_index import LEAK_CHECKS

    asked = sum(len(restricted_queries(c)) for _, c, _ in LEAK_CASES) + KEYWORD_LEAK_CHECKS
    assert asked == LEAK_CHECKS


def test_restricted_passages_look_exactly_like_missing_ones(client):
    store = DemoIndex(model_folder=None).store
    for chunk in store.chunks(frozenset({"hr", "finance"})):
        for persona in ("maria", "farid" if chunk.collection == "hr" else "hana"):
            res = client.get("/api/passage", params={"id": chunk.chunk_id, "persona": persona})
            missing = client.get("/api/passage", params={"id": chunk.chunk_id.split("#")[0] + "#99", "persona": persona})
            assert res.status_code == missing.status_code == 404 and res.json() == missing.json()
    assert client.get("/api/passage", params={"id": "hr/salary-bands#2", "persona": "hana"}).status_code == 200


def test_keyword_only_search_keeps_the_same_access_rules(keywords_index):
    client = TestClient(create_app(keywords_index, client=None))
    for query in restricted_queries("hr")[:KEYWORD_LEAK_CHECKS]:
        out = client.post("/api/ask", json={"question": query, "persona": "maria"}).json()
        assert not [s for s in HR_ONLY if s in shown(out)], query


# ------------------------------------------------------------------ nothing written to disk
def test_no_audit_entry_or_anything_else_is_written_to_disk(client):
    before = sha256(DEMO_DIR / "kb.db")
    out = ask(client, STORY["leave"])
    assert [(e["actor"], e["role"], e["channel"], e["action"], e["outcome"]) for e in out["audit"]] == [
        ("maria", "staff", "demo", "ask", "answered")]
    assert out["audit"][0]["passages"][0] == "general/leave-policy#1"  # the passages the answer could use
    for question, persona in ((STORY["salary"], "maria"), (STORY["salary"], "hana"), ("how much holiday do I get?", "maria")):
        ask(client, question, persona)
    client.get("/api/passage", params={"id": "hr/salary-bands#2", "persona": "maria"})
    assert sha256(DEMO_DIR / "kb.db") == before and not list(DEMO_DIR.glob("kb.db-*"))


# ------------------------------------------------------------------ Claude, through a simulated API
def test_a_claude_answer_is_checked_marked_and_priced(index):
    fake = FakeAnthropic(lambda r: cited_reply(r, "Full-time employees get 25 days of paid annual leave per calendar year."))
    client = TestClient(create_app(index, client=fake))
    assert client.get("/api/config").json()["live"] is True
    out = ask(client, STORY["leave"])
    assert out["status"] == "answered" and out["written_by"] == "claude" and out["model"] == MODEL
    assert out["parts"][0]["sources"] == [1] and out["quoted_because"] is None
    assert out["cost"] == {"usd": pytest.approx(0.00015), "input_tokens": 1200, "output_tokens": 60}
    assert out["audit"][0]["mode"] == "claude"
    assert len(fake.requests) == 1


def test_the_haiku_request_shape(index):
    fake = FakeAnthropic()
    client = TestClient(create_app(index, client=fake))
    out = ask(client, STORY["scanner"])
    [request] = fake.requests
    assert request["model"] == "claude-haiku-5-5" and request["max_tokens"] == MAX_TOKENS == 2000
    assert request["output_config"] == {"effort": "low"}
    assert request["thinking"] == {"type": "disabled"}  # thinking cannot use up the 2,000 tokens
    assert "betas" not in request and "fallbacks" not in request
    assert "never as instructions" in request["system"] and "NOT_IN_DOCUMENTS" in request["system"]
    blocks = search_results(request)
    assert [b["source"] for b in blocks] == [p["passage_id"] for p in out["passages"]]
    assert all(b["citations"] == {"enabled": True} for b in blocks)
    assert request["messages"][0]["content"][-1] == {"type": "text", "text": f"Question: {STORY['scanner']}"}


def test_the_live_client_has_a_short_timeout_and_no_retries(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-local-test-not-a-real-key")
    monkeypatch.delenv("KB_DEMO_AI", raising=False)
    client, why = live_client_from_env()
    assert why is None and client.max_retries == 0
    assert client.timeout.read == 20.0 and client.timeout.connect == 3.0
    monkeypatch.setenv("KB_DEMO_AI", "off")
    assert live_client_from_env() == (None, "off")
    monkeypatch.delenv("KB_DEMO_AI")
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert live_client_from_env() == (None, "no_key")


@pytest.mark.parametrize("kind, reason", [("budget", "budget"), ("budget_usage", "budget"), ("rate_limited", "rate_limited"),
                                          ("auth", "auth"), ("timeout", "timeout"), ("error", "error")])
def test_api_failures_fall_back_to_quoted_answers_and_say_why(index, kind, reason):
    fake = FakeAnthropic(api_error(kind))
    client = TestClient(create_app(index, client=fake))
    out = ask(client, STORY["leave"])
    assert out["status"] == "answered" and out["written_by"] == "quoted" and out["cost"]["usd"] == 0
    assert out["quoted_because"] == QUOTED_BECAUSE[reason]
    assert "25 days" in out["citations"][0]["quote"] and len(fake.requests) == 1


def test_a_visitor_over_the_ai_limit_gets_quoted_answers(index, monkeypatch):
    monkeypatch.setattr("kb.demo_web.PER_VISITOR_AI", (1, 600))
    fake = FakeAnthropic()
    client = TestClient(create_app(index, client=fake))
    assert ask(client, STORY["leave"])["written_by"] == "claude"
    second = ask(client, STORY["scanner"])
    assert second["written_by"] == "quoted" and second["quoted_because"] == QUOTED_BECAUSE["visitor_limit"]
    assert len(fake.requests) == 1
    assert PER_VISITOR_AI[0] >= 10  # enough for the story and a good look around


def test_no_ai_call_when_nothing_clears_the_bar(index):
    fake = FakeAnthropic()
    out = ask(TestClient(create_app(index, client=fake)), STORY["salary"], "maria")
    assert out["status"] == "not_found" and out["written_by"] == "none" and fake.requests == []


def test_the_visitor_can_choose_quoted_answers(index):
    fake = FakeAnthropic()
    out = ask(TestClient(create_app(index, client=fake)), STORY["leave"], ai=False)
    assert out["written_by"] == "quoted" and out["quoted_because"] == QUOTED_BECAUSE["chosen"] and fake.requests == []


def test_claude_saying_not_in_the_documents_is_a_refusal_with_its_cost(index):
    reply = {"model": MODEL, "stop_reason": "end_turn", "usage": {"input_tokens": 900, "output_tokens": 8},
             "content": [{"type": "text", "text": "NOT_IN_DOCUMENTS"}]}
    out = ask(TestClient(create_app(index, client=FakeAnthropic(lambda r: reply))), STORY["sick"])
    assert out["status"] == "not_found" and out["written_by"] == "claude" and out["cost"]["input_tokens"] == 900


def test_a_real_bad_claude_answer_is_held_too(index):
    def bad(request):
        reply = cited_reply(request, "You get 40 days of paid annual leave.")
        reply["content"].append({"type": "text", "text": " Unused days never expire."})
        return reply

    out = ask(TestClient(create_app(index, client=FakeAnthropic(bad))), STORY["leave"])
    assert out["status"] == "flagged" and out["written_by"] == "claude" and "40 days" in out["draft"]
    assert out["problems"] and "40 days" not in out["answer"] and out["cost"]["usd"] > 0


def test_the_forced_draft_never_calls_the_api(index):
    fake = FakeAnthropic()
    client = TestClient(create_app(index, client=fake))
    out = ask(client, STORY["leave"], simulate="bad_draft")
    assert out["status"] == "flagged" and out["written_by"] == "simulated" and fake.requests == []
    res = client.post("/api/ask", json={"question": STORY["sick"], "persona": "maria", "simulate": "bad_draft"})
    assert res.status_code == 400
    assert client.post("/api/ask", json={"question": STORY["leave"], "persona": "maria", "simulate": "other"}).status_code == 400


def test_the_real_anthropic_sdk_through_a_local_fake_server(index):
    """The SDK builds the request and parses the reply and the token counts into its own types."""
    import anthropic
    from conftest import free_port, serve

    seen: dict[str, Any] = {}
    fake_api = FastAPI()

    @fake_api.post("/v1/messages")
    async def messages(request: Request) -> Any:
        seen["body"] = await request.json()
        if seen.get("fail"):
            from fastapi.responses import JSONResponse
            return JSONResponse({"type": "error", "error": {"type": "rate_limit_error", "message": "enforced_spend_limit_reached"}},
                                status_code=429)
        reply = cited_reply(seen["body"])
        return {"id": "msg_local", "type": "message", "role": "assistant", "stop_sequence": None, **reply}

    port = free_port()
    running = serve(fake_api, port)
    try:
        sdk = anthropic.Anthropic(api_key="sk-local-test-not-a-real-key", base_url=f"http://127.0.0.1:{port}", max_retries=0)
        client = TestClient(create_app(index, client=sdk))
        out = ask(client, STORY["leave"])
        assert out["written_by"] == "claude" and out["cost"]["input_tokens"] == 1200 and out["cost"]["output_tokens"] == 60
        assert seen["body"]["model"] == MODEL and seen["body"]["max_tokens"] == 2000
        seen["fail"] = True
        out = ask(client, STORY["leave"])
        assert out["written_by"] == "quoted" and out["quoted_because"] == QUOTED_BECAUSE["budget"]
    finally:
        running.should_exit = True


# ------------------------------------------------------------------ plan B: keyword search for typed questions
def test_plan_b_searches_typed_questions_by_keywords_and_says_so(keywords_index):
    client = TestClient(create_app(keywords_index, client=None))
    cfg = client.get("/api/config").json()
    assert cfg["search"] == {"typed": "keywords", "model": "off", "note": SEARCH_NOTES["off"]}
    typed = ask(client, "what is the hotel limit per night in London")
    assert typed["search"]["by"] == "keywords" and typed["search"]["note"] == SEARCH_NOTES["off"]
    assert typed["search"]["keywords_alone"] is None and typed["status"] == "answered"
    story = ask(client, STORY["scanner"])  # stored vectors: still found by meaning
    assert story["search"]["by"] == "meaning" and story["search"]["keywords_alone"] is False
    assert client.post("/api/warm").json()["meaning_model"] == "off"
    assert keywords_index._model is None


def test_plan_b_is_chosen_by_an_environment_variable(monkeypatch):
    monkeypatch.setenv("KB_DEMO_SEARCH", "keywords")
    app = create_app(client=None)
    assert app.state.demo.index.model_state == "off"
    monkeypatch.setenv("KB_DEMO_SEARCH", "meaning")
    assert create_app(client=None).state.demo.index.model_state == "waiting"


def test_without_the_model_files_typed_questions_use_keywords(tmp_path):
    index = DemoIndex(model_folder=tmp_path)
    client = TestClient(create_app(index, client=None))
    assert client.get("/api/config").json()["search"]["note"] == SEARCH_NOTES["missing"]
    assert ask(client, "who do I tell about a stolen phone")["search"]["by"] == "keywords"


def test_warm_loads_the_meaning_model(client):
    assert client.post("/api/warm").json()["meaning_model"] == "ready"
    typed = ask(client, "how much holiday do I get each year?")
    assert typed["search"]["by"] == "meaning" and typed["status"] == "answered"


# ------------------------------------------------------------------ hardening
@pytest.mark.parametrize("raw, status", [(b"not json", 400), (b"[1, 2]", 400), (b'{"question": NaN}', 400),
                                         (b"x" * (MAX_BODY + 1), 413), (b'{"a":' * 50 + b"1" + b"}" * 50, 400)],
                         ids=["not json", "a list", "NaN", "too big", "deep"])
def test_bad_bodies_are_refused(client, raw, status):
    res = client.post("/api/ask", content=raw, headers={"Content-Type": "application/json"})
    assert res.status_code == status and res.json()["error"] == "request"


@pytest.mark.parametrize("body, message", [
    ({"question": "hi", "persona": "eve"}, "Pick Maria, Hana or Farid."),
    ({"question": "x" * 1001, "persona": "maria"}, "The question is longer than 1000 characters."),
    ({"question": 7, "persona": "maria"}, "The question must be text."),
    ({"question": "   ", "persona": "maria"}, "The question is empty."),
])
def test_bad_questions_are_explained(client, body, message):
    res = client.post("/api/ask", json=body)
    assert res.status_code == 400 and res.json()["message"] == message


def test_malformed_passage_ids_are_refused(client):
    assert client.get("/api/passage", params={"id": "../../etc/passwd", "persona": "maria"}).status_code == 400
    assert client.get("/api/passage", params={"id": "general/leave-policy#1", "persona": "x"}).status_code == 400


def test_every_request_is_counted_per_visitor(index, monkeypatch):
    monkeypatch.setattr("kb.demo_web.PER_VISITOR_REQUESTS", (3, 600))
    client = TestClient(create_app(index, client=None))
    codes = [client.get("/api/config").status_code for _ in range(4)]
    assert codes == [200, 200, 200, 429]


def test_x_forwarded_for_is_ignored_unless_running_on_vercel(index, monkeypatch):
    monkeypatch.setattr("kb.demo_web.PER_VISITOR_REQUESTS", (2, 600))
    monkeypatch.delenv("VERCEL", raising=False)
    client = TestClient(create_app(index, client=None))
    codes = [client.get("/api/config", headers={"x-forwarded-for": f"10.0.0.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 429]
    monkeypatch.setenv("VERCEL", "1")
    client = TestClient(create_app(index, client=None))
    codes = [client.get("/api/config", headers={"x-vercel-forwarded-for": f"10.0.0.{i}"}).status_code for i in range(3)]
    assert codes == [200, 200, 200]


def test_a_visitor_can_have_only_two_requests_in_progress():
    limiter = Limiter()
    assert limiter.enter("a") and limiter.enter("a") and not limiter.enter("a")
    limiter.leave("a")
    assert limiter.enter("a") and limiter.enter("b")


def test_an_unexpected_error_is_a_clean_500(index, monkeypatch):
    client = TestClient(create_app(index, client=None), raise_server_exceptions=False)
    monkeypatch.setattr(client.app.state.demo, "ask", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    res = client.post("/api/ask", json={"question": "hi", "persona": "maria"})
    assert res.status_code == 500 and res.json()["error"] == "server" and "boom" not in res.text


# ------------------------------------------------------------------ what is served, and what Vercel ships
def test_security_headers_are_the_same_locally_and_on_vercel(client):
    vercel = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    [rule] = vercel["headers"]
    assert {h["key"]: h["value"] for h in rule["headers"]} == SECURITY_HEADERS
    for path in ("/", "/app.js", "/api/config", "/healthz"):
        res = client.get(path)
        assert res.status_code == 200, path
        for name, value in SECURITY_HEADERS.items():
            assert res.headers[name] == value, (path, name)
    assert client.get("/api/config").headers["cache-control"] == "no-store"
    assert "unsafe-inline" not in SECURITY_HEADERS["Content-Security-Policy"].split("style-src")[0]


def test_vercel_ships_the_demo_and_leaves_the_rest_out():
    vercel = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    fn = vercel["functions"]["api/index.py"]
    for path in ("tests/**", "docs/**", "eval/**", "samples/**", "models/**", "public/**", "scripts/**", ".venv/**"):
        assert path in fn["excludeFiles"], path
    assert "demo/**" in fn["includeFiles"] and "kb/**" in fn["includeFiles"]
    assert {"source": "/api/(.*)", "destination": "/api/index"} in vercel["rewrites"]
    assert vercel["outputDirectory"] == "public"
    requirements = (ROOT / "api" / "requirements.txt").read_text(encoding="utf-8").lower()
    names = {re.split(r"[<>=\[ ]", line)[0] for line in requirements.splitlines() if line and not line.startswith("#")}
    assert names == {"fastapi", "pydantic", "numpy", "onnxruntime", "anthropic"}  # httpx comes with anthropic
    entry = (ROOT / "api" / "index.py").read_text(encoding="utf-8")
    assert "from kb.demo_web import create_app" in entry and "serve_static=False" in entry


def test_the_page_has_no_contact_details_and_only_the_repository_link():
    public = ROOT / "public"
    for path in public.iterdir():
        text = path.read_text(encoding="utf-8")
        assert not re.findall(r"[\w.+-]+@[a-z][\w-]*\.[a-z]{2,}", text), path.name
        assert "mailto:" not in text and "tel:" not in text and "wa.me" not in text, path.name
        links = set(re.findall(r"https?://[^\s\"'<>)]+", text))
        allowed = {"https://github.com/Ahmedsukaik/rag-knowledge-assistant", "https://fonts.googleapis.com", "https://fonts.gstatic.com",
                   "http://www.w3.org/2000/svg"}
        assert all(any(link.startswith(a) for a in allowed) for link in links), (path.name, links)


def test_the_page_and_the_demo_texts_use_no_em_dashes():
    texts = [p.read_text(encoding="utf-8") for p in (ROOT / "public").iterdir()]
    texts += [json.dumps(CHIPS), json.dumps(STORY), json.dumps(QUOTED_BECAUSE), json.dumps(SEARCH_NOTES)]
    for text in texts:
        assert "\u2014" not in text and "\u2013" not in text


def test_the_sentence_on_the_first_screen_is_in_the_leave_policy():
    script = (ROOT / "public" / "app.js").read_text(encoding="utf-8")
    [sentence] = re.findall(r'const LEAVE_SENTENCE = "([^"]+)";', script)
    assert sentence in (SAMPLES / "general" / "leave-policy.md").read_text(encoding="utf-8")


def test_the_page_and_its_files_are_served_locally(client):
    page = client.get("/")
    assert page.status_code == 200 and "RAG Knowledge Assistant" in page.text
    for name in ("app.js", "style.css", "tokens.css"):
        assert client.get("/" + name).status_code == 200
    assert client.get("/healthz").json() == {"status": "ok"}


def test_the_vercel_entry_point_imports(monkeypatch):
    import importlib.util

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    spec = importlib.util.spec_from_file_location("vercel_entry", ROOT / "api" / "index.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    client = TestClient(module.app)
    assert client.get("/api/config").status_code == 200 and client.get("/").status_code == 404  # Vercel serves public/
    assert Path(module.__file__).name == "index.py"


# ------------------------------------------------------------------ round 1 review: limits, CPU, odd characters
def test_ipv6_visitors_are_counted_by_their_64(index, monkeypatch):
    assert visitor_key("2001:db8:1:2:aaaa::1") == visitor_key("2001:db8:1:2:ffff::9") == "2001:db8:1:2::/64"
    assert visitor_key("2001:db8:1:3::1") != visitor_key("2001:db8:1:2::1")
    assert visitor_key("203.0.113.7") == "203.0.113.7" and visitor_key("::ffff:203.0.113.7") == "203.0.113.7"
    assert visitor_key("not an address") == "not an address"
    monkeypatch.setattr("kb.demo_web.PER_VISITOR_REQUESTS", (2, 600))
    monkeypatch.setenv("VERCEL", "1")
    client = TestClient(create_app(index, client=None))
    codes = [client.get("/api/config", headers={"x-vercel-forwarded-for": f"2001:db8:5:6::{i}"}).status_code for i in range(1, 4)]
    assert codes == [200, 200, 429]  # a new address in the same /64 is the same visitor


def test_the_instance_ai_cap_cannot_be_reset_by_many_visitors(monkeypatch):
    monkeypatch.setattr("kb.demo_web.MAX_KEYS", 3)
    limiter = Limiter()
    assert limiter.allow_instance("ai", 2, 3600) and limiter.allow_instance("ai", 2, 3600)
    for i in range(10):  # visitors pushing older visitors out of the limiter
        limiter.allow(f"v{i}", 1, 600)
    assert not limiter.allow_instance("ai", 2, 3600)


def test_the_instance_ai_cap_is_applied(index, monkeypatch):
    monkeypatch.setattr("kb.demo_web.PER_INSTANCE_AI", (1, 3600))
    monkeypatch.setenv("VERCEL", "1")
    fake = FakeAnthropic()
    client = TestClient(create_app(index, client=fake))
    first = client.post("/api/ask", json={"question": STORY["leave"], "persona": "maria"}, headers={"x-vercel-forwarded-for": "10.0.0.1"})
    second = client.post("/api/ask", json={"question": STORY["leave"], "persona": "maria"}, headers={"x-vercel-forwarded-for": "10.0.0.2"})
    assert first.json()["written_by"] == "claude" and second.json()["quoted_because"] == QUOTED_BECAUSE["busy"]
    assert len(fake.requests) == 1


def test_typed_questions_have_their_own_lower_limit(index, monkeypatch):
    monkeypatch.setattr("kb.demo_web.PER_VISITOR_EMBED", (1, 600))
    client = TestClient(create_app(index, client=None))
    assert ask(client, "how much holiday do I get each year?")["search"]["by"] == "meaning"
    second = ask(client, "who approves a big expense claim?")
    assert second["search"] == {"by": "keywords", "keywords_alone": None, "note": SEARCH_NOTES["visitor_limit"]}
    assert ask(client, STORY["scanner"])["search"]["by"] == "meaning"  # stored questions do not count
    assert ask(client, "how much holiday do I get each year?")["search"]["by"] == "keywords"  # nor do repeats, once over
    assert PER_VISITOR_EMBED[0] < 300  # lower than the limit on requests of any kind


def test_the_model_reads_at_most_128_tokens_of_a_typed_question(index):
    assert index.load_model() == "ready"
    tokenizer = index.embedder.model._tokenizer  # type: ignore[union-attr]
    long_question = "what about " + " ".join(f"word{i}" for i in range(400))
    assert len(tokenizer.encode(long_question)) == QUESTION_TOKENS == 128
    client = TestClient(create_app(index, client=None))
    assert ask(client, long_question[:1000])["search"]["by"] == "meaning"


def test_lone_surrogates_in_a_question_are_dropped(client):
    raw = '{"question": "How much annual leave ' + chr(92) + 'ud800do I get?", "persona": "maria"}'
    res = client.post("/api/ask", content=raw.encode(), headers={"Content-Type": "application/json"})
    assert res.status_code == 200 and res.json()["question"] == "How much annual leave do I get?"
    raw = '{"question": "' + chr(92) + 'udfff", "persona": "maria"}'
    res = client.post("/api/ask", content=raw.encode(), headers={"Content-Type": "application/json"})
    assert res.status_code == 400 and res.json()["message"] == "The question is empty."


def test_the_word_cache_is_bounded_and_skips_long_words(index, monkeypatch):
    from kb.onnx_embed import WordPieceTokenizer

    monkeypatch.setattr("kb.onnx_embed.WORD_CACHE", 50)
    tokenizer = WordPieceTokenizer(DEMO_DIR / "model" / "tokenizer.json")
    tokenizer.encode(" ".join(f"w{i}x" for i in range(200)))
    assert len(tokenizer._cache) == 50
    tokenizer.encode("y" * 500)
    assert "y" * 500 not in tokenizer._cache


def test_the_question_vector_cache_is_safe_under_threads():
    import threading

    import numpy as np

    from kb.demo_index import IndexedEmbedder

    class Model:
        def embed(self, texts):
            return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)

    embedder = IndexedEmbedder("m@x", ["known"], np.zeros((1, 2), dtype=np.float32), model=Model(), cache_size=2)
    errors: list[BaseException] = []

    def work(n: int) -> None:
        try:
            for i in range(300):
                texts = ["known", f"t{(i * n) % 7}", f"u{i % 5}", f"t{(i * n) % 7}"]
                rows = embedder.embed(texts)
                assert rows.shape == (4, 2) and rows[1][0] == len(texts[1])
        except BaseException as exc:  # noqa: BLE001 - collected for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=work, args=(n,)) for n in range(1, 9)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
