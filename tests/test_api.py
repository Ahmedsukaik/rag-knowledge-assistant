"""The HTTP API, over real HTTP against a running server."""
import asyncio
import base64

import httpx
import pytest
from conftest import ADMIN, HR, SAMPLES, STAFF, TOKENS, audit, free_port, serve

from kb.api import build_app
from kb.config import TokenTable

pytestmark = pytest.mark.anyio


def test_every_route_except_healthz_needs_a_valid_token(server):
    routes = [("POST", "/ask", {"question": "annual leave"}), ("POST", "/search", {"query": "leave"}),
              ("GET", "/documents", None), ("GET", "/passages/general/leave-policy%231", None),
              ("POST", "/admin/ingest", {}), ("DELETE", "/admin/documents/general/leave-policy", None),
              ("GET", "/admin/audit", None), ("POST", "/mcp", {}), ("GET", "/docs", None), ("GET", "/no-such-route", None)]
    with server.http(None) as anonymous:
        for header in (None, "Bearer wrong-token-000000000", f"Basic {STAFF}", STAFF, "Bearer", f"Token {STAFF}"):
            headers = {"Authorization": header} if header else {}
            for method, path, body in routes:
                resp = anonymous.request(method, path, json=body, headers=headers)
                assert resp.status_code == 401, (header, path)
                assert resp.headers["www-authenticate"] == "Bearer"
        assert anonymous.get("/healthz").json() == {"status": "ok"}
    assert audit(server.kb) == []  # anonymous traffic does not fill the audit log
    assert server.kb.store.documents()  # and nothing was changed


def test_ask_returns_the_answer_its_sources_and_the_mode(server):
    result = server.ask(STAFF, "How long is parental leave for partners?")
    assert result["status"] == "answered" and result["mode"] == "extractive"
    assert result["answer"] == '"Partners receive 4 weeks of parental leave at full pay, to be taken within the first year." [1]'
    assert result["citations"][0]["passage_id"] == "general/leave-policy#4"
    assert result["passages"][0]["label"] == "Leave Policy > Parental leave" and "unverified_draft" not in result
    refused = server.ask(STAFF, "Who won the football world cup in 2022?")
    assert refused == {"status": "not_found", "answer": "I could not find this in the documents you have access to.",
                       "mode": "extractive", "citations": [], "passages": [], "warnings": [], "parts": []}
    assert result["parts"] == [{"text": '"Partners receive 4 weeks of parental leave at full pay, to be taken within '
                                        'the first year."', "sources": [1]}]


def test_bad_questions_are_refused_and_audited(server):
    with server.http(STAFF) as http:
        assert http.post("/ask", json={"question": "x" * 1001}).json() == {"error": "the question is longer than 1000 characters"}
        assert http.post("/ask", json={"question": "  \n\t "}).status_code == 400
        assert http.post("/ask", json={"question": ["a", "list"]}).status_code == 422
        assert http.post("/ask", json={}).status_code == 422
        assert http.post("/ask", content=b"{not json", headers={"Content-Type": "application/json"}).status_code == 422
        # the same 1000-character rule as over MCP, checked in one place
        assert http.post("/ask", json={"question": "x" * 20_000}).json() == {"error": "the question is longer than 1000 characters"}
    assert [a[3] for a in audit(server.kb)] == ["rejected"] * 6


def test_oversized_bodies_are_refused_before_they_are_read(server):
    big = b'{"question": "' + b"x" * 200_000 + b'"}'
    with server.http(STAFF) as http, server.http(ADMIN) as admin:
        assert http.post("/ask", content=big, headers={"Content-Type": "application/json"}).status_code == 413

        def chunks():  # no Content-Length: the body is cut off while it streams in
            yield b'{"question": "'
            for _ in range(100):
                yield b"x" * 10_000
            yield b'"}'

        assert http.post("/ask", content=chunks(), headers={"Content-Type": "application/json"}).status_code == 413
        assert http.post("/mcp", content=big, headers={"Content-Type": "application/json"}).status_code == 413
        assert http.post("/ask", json={"question": "How long must a password be?"}).status_code == 200
        upload = {"collection": "general", "filename": "big.txt", "text": "Big file\n" + "word " * 200_000}
        assert admin.post("/admin/ingest", json=upload).status_code == 200  # 1 MB upload: within its own limit
        assert admin.post("/admin/ingest", content=b"x" * (10 * 1024 * 1024)).status_code == 413


def test_questions_with_control_characters_are_cleaned_before_they_are_used_or_logged(server):
    result = server.ask(STAFF, "How long must a\u202e pass\u200bword\x1b be?\n")
    assert result["status"] == "answered" and "14 characters" in result["answer"]
    logged = server.kb.store.audit_entries(1)[0]["query"]
    assert "\u202e" not in logged and "\x1b" not in logged and "\n" not in logged


def test_search_documents_and_passages_follow_the_role(server):
    with server.http(STAFF) as staff, server.http(HR) as hr:
        passages = staff.post("/search", json={"query": "salary band", "k": 20}).json()["passages"]
        assert passages and all(p["collection"] == "general" for p in passages)
        assert {"score", "coverage", "text", "label", "page"} <= set(passages[0])
        assert staff.post("/search", json={"query": "leave", "k": 50}).status_code == 400
        assert {d["collection"] for d in staff.get("/documents").json()["documents"]} == {"general"}
        assert {d["collection"] for d in hr.get("/documents").json()["documents"]} == {"general", "hr"}
        assert staff.get("/passages/hr/salary-bands%232").json() == {"error": "No passage with that id."}
        assert staff.get("/passages/hr/salary-bands%232").status_code == 404
        assert hr.get("/passages/hr/salary-bands%232").json()["section"] == "Bands"
        assert staff.get("/passages/..%2F..%2Fetc%2Fpasswd").status_code == 400
        page = staff.get("/passages/general/pt-200-product-manual%234").json()
        assert page["page"] == 3 and page["label"].endswith("page 3")


def test_only_admin_can_ingest_delete_or_read_the_audit_log(server):
    upload = {"collection": "general", "filename": "volunteering.md", "text": "# Volunteering\n\nTwo paid days per year."}
    with server.http(STAFF) as staff, server.http(HR) as hr:
        for http in (staff, hr):
            assert http.post("/admin/ingest", json=upload).status_code == 403
            assert http.delete("/admin/documents/general/leave-policy").status_code == 403
            assert http.get("/admin/audit").status_code == 403
    denied = [a for a in audit(server.kb) if a[3] == "denied"]
    assert len(denied) == 6 and {a[2] for a in denied} == {"ingest", "delete", "audit.read"}
    assert "general/volunteering" not in {d["doc_id"] for d in server.kb.store.documents()}


def test_admin_ingest_add_unchanged_update_and_delete(server):
    upload = {"collection": "general", "filename": "volunteering.md", "text": "# Volunteering\n\nTwo paid days per year."}
    with server.http(ADMIN) as admin:
        assert admin.post("/admin/ingest", json=upload).json() == {
            "doc_id": "general/volunteering", "status": "added", "passages": 1, "title": "Volunteering"}
        assert admin.post("/admin/ingest", json=upload).json()["status"] == "unchanged"
        changed = {**upload, "text": "# Volunteering\n\nThree paid days per year."}
        assert admin.post("/admin/ingest", json=changed).json()["status"] == "updated"
        assert "Three paid days" in server.ask(STAFF, "How many volunteering days are there?")["answer"]
        assert admin.delete("/admin/documents/general/volunteering").json()["status"] == "removed"
        assert admin.delete("/admin/documents/general/volunteering").status_code == 404
        assert server.ask(STAFF, "How many volunteering days are there?")["status"] == "not_found"
        entries = admin.get("/admin/audit", params={"limit": 20}).json()["entries"]
    assert [e["outcome"] for e in reversed(entries) if e["action"] in ("ingest", "delete")] == [
        "added", "unchanged", "updated", "ok", "not_found"]


def test_admin_can_upload_a_pdf_and_bad_uploads_are_refused(server):
    pdf = (SAMPLES / "finance" / "quarter-end-checklist.pdf").read_bytes()
    with server.http(ADMIN) as admin:
        done = admin.post("/admin/ingest", json={"collection": "archive", "filename": "q3-close.pdf",
                                                 "content_base64": base64.b64encode(pdf).decode()}).json()
        assert done["doc_id"] == "archive/q3-close" and done["status"] == "added" and done["passages"] == 5
        bad = [
            ({"collection": "general", "filename": "x.pdf", "content_base64": "not base64!!"}, 400, "not valid base64"),
            ({"collection": "general", "filename": "x.md", "text": "a", "content_base64": "YQ=="}, 400, "exactly one"),
            ({"collection": "general", "filename": "x.md"}, 400, "exactly one"),
            ({"collection": "general", "filename": "x.exe", "text": "MZ"}, 400, "unsupported file type"),
            ({"collection": "Bad Name", "filename": "x.md", "text": "# X\n\ny"}, 400, "collection names"),
            ({"collection": "general", "filename": "x.pdf", "content_base64": base64.b64encode(b"%PDF-1.4 junk").decode()},
             400, "could not be read"),
        ]
        for body, status, message in bad:
            resp = admin.post("/admin/ingest", json=body)
            assert resp.status_code == status and message in resp.json()["error"], body
        sneaky = admin.post("/admin/ingest", json={"collection": "general", "filename": "../../etc/evil.md",
                                                   "text": "# Evil\n\nText."}).json()
        assert sneaky["doc_id"] == "general/evil"  # only the file name is used, never the path
    assert server.ask(STAFF, "Who signs off the management accounts?")["status"] == "not_found"  # archive: admin only


async def test_parallel_questions_are_each_answered_and_audited(server):
    async with httpx.AsyncClient(base_url=server.url, headers={"Authorization": f"Bearer {STAFF}"}) as http:
        replies = await asyncio.gather(*[http.post("/ask", json={"question": "How long must a password be?"})
                                         for _ in range(20)])
    assert {r.status_code for r in replies} == {200} and {r.json()["status"] for r in replies} == {"answered"}
    assert audit(server.kb).count(("maria (sales)", "http", "ask", "answered")) == 20


def test_audit_entries_record_who_what_which_passages_outcome_and_time(server):
    server.ask(STAFF, "How long is the probation period?")
    server.ask(HR, "When is the annual pay review?")
    with server.http(ADMIN) as admin:
        entries = admin.get("/admin/audit", params={"limit": 5}).json()["entries"]
    hr_entry, staff_entry = entries[0], entries[1]
    assert staff_entry["actor"] == "maria (sales)" and staff_entry["role"] == "staff" and staff_entry["action"] == "ask"
    assert staff_entry["query"] == "How long is the probation period?" and staff_entry["outcome"] == "answered"
    assert staff_entry["passages"][0] == "general/onboarding-guide#3" and staff_entry["mode"] == "extractive"
    assert hr_entry["passages"] == ["hr/salary-bands#3"] and isinstance(hr_entry["duration_ms"], int)
    assert staff_entry["ts"].endswith("+00:00")


async def test_public_hostname_allow_list_covers_every_route(server):
    app = build_app(server.kb, TokenTable(TOKENS), "0.0.0.0", ["kb.example.com"])
    auth = {"Authorization": f"Bearer {STAFF}"}
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://kb.example.com", headers=auth) as http:
        assert (await http.get("/documents")).status_code == 200
        assert (await http.get("/", headers={"Authorization": ""})).status_code == 200  # the page needs no token
    async with httpx.AsyncClient(transport=transport, base_url="http://kb.example.com:8443", headers=auth) as http:
        assert (await http.get("/documents")).status_code == 200
    async with httpx.AsyncClient(transport=transport, base_url="http://evil.example.net", headers=auth) as http:
        for path in ("/documents", "/admin/audit", "/"):
            assert (await http.get(path)).status_code == 421
        assert (await http.post("/ask", json={"question": "leave"})).status_code == 421
        assert (await http.post("/mcp", json={})).status_code == 421
        assert (await http.get("/healthz")).status_code == 200
        for host in ("kb.example.com.evil.net", "evil-kb.example.com", "kb.example.com:evil", "kb.example.com:1:2", ""):
            assert (await http.get("/documents", headers={"Host": host})).status_code == 421, host


def test_mcp_works_behind_a_public_hostname_with_a_port_and_an_origin(kb):
    port = free_port()
    running = serve(build_app(kb, TokenTable(TOKENS), "127.0.0.1", ["kb.example.com"]), port)
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    headers = {"Authorization": f"Bearer {STAFF}", "Accept": "application/json, text/event-stream",
               "Host": "kb.example.com:8443", "Origin": "https://kb.example.com"}
    try:
        with httpx.Client(base_url=f"http://127.0.0.1:{port}") as http:
            assert http.post("/mcp", json=body, headers=headers).status_code == 200
            assert http.post("/mcp", json=body, headers={**headers, "Host": "kb.example.com"}).status_code == 200
            no_origin = {k: v for k, v in headers.items() if k != "Origin"}
            assert http.post("/mcp", json=body, headers=no_origin).status_code == 200
            assert http.post("/mcp", json=body, headers={**headers, "Origin": "https://evil.example.net"}).status_code == 403
            assert http.post("/mcp", json=body, headers={**headers, "Host": "evil.example.net"}).status_code == 421
    finally:
        running.should_exit = True


def test_an_upload_over_a_different_file_needs_replace(server):
    with server.http(ADMIN) as admin:
        refused = admin.post("/admin/ingest", json={"collection": "general", "filename": "leave-policy.txt",
                                                    "text": "Leave\nEveryone gets 40 days."})
        assert refused.status_code == 400 and "already used by 'leave-policy.md'" in refused.json()["error"]
        replaced = admin.post("/admin/ingest", json={"collection": "general", "filename": "leave-policy.txt",
                                                     "text": "Leave\nEveryone gets 40 days.", "replace": True})
        assert replaced.json()["status"] == "updated"
    assert "40 days" in server.ask(STAFF, "How many days of leave does everyone get?")["answer"]


def test_an_answer_service_failure_is_reported_not_guessed(tmp_path, store):
    from types import SimpleNamespace

    from kb.answer import ClaudeAnswerer
    from kb.service import KnowledgeBase

    def broken(**kwargs):
        raise ConnectionError("upstream down")

    answerer = ClaudeAnswerer(client=SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=broken))))
    kb = KnowledgeBase(store, answerer=answerer)
    port = free_port()
    running = serve(build_app(kb, TokenTable(TOKENS)), port)
    try:
        resp = httpx.post(f"http://127.0.0.1:{port}/ask", json={"question": "How long must a password be?"},
                          headers={"Authorization": f"Bearer {STAFF}"})
    finally:
        running.should_exit = True
    body = resp.json()
    assert resp.status_code == 200 and body["status"] == "error" and body["mode"] == "claude"
    assert body["passages"][0]["passage_id"] == "general/it-security-policy#1" and body["citations"] == []
    assert "upstream down" not in resp.text
