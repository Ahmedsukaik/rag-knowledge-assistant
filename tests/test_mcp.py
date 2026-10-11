"""The MCP tools, over a real MCP Streamable HTTP connection (and once in-process over stdio)."""
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing

import pytest
from conftest import AGENT, HR, STAFF, TOKENS, ToolCallError, audit, who

from kb.config import TokenTable
from kb.mcp_server import build_mcp
from kb.store import Store

pytestmark = pytest.mark.anyio


async def test_tools_are_read_only_and_described(server):
    async with server.mcp() as call:
        tools = (await call.client.list_tools()).tools
    assert [t.name for t in tools] == ["search_documents", "get_passage", "ask", "list_documents"]
    for t in tools:
        assert t.annotations.read_only_hint is True and t.annotations.destructive_hint is False, t.name
        assert t.description and len(t.description) > 40
    ask = next(t for t in tools if t.name == "ask")
    assert ask.input_schema["required"] == ["question"] and set(ask.input_schema["properties"]) == {"question"}
    search = next(t for t in tools if t.name == "search_documents")
    assert set(search.input_schema["properties"]) == {"query", "k"}  # no hidden identity or collection parameter


async def test_mcp_tools_answer_with_the_callers_role(server):
    async with server.mcp(AGENT) as call:
        answer = await call("ask", {"question": "How long is the warranty on the PT-200?"})
        assert answer["status"] == "answered" and answer["passages"][0]["page"] == 4
        assert "2-year warranty" in answer["citations"][0]["quote"]
        found = await call("search_documents", {"query": "salary band", "k": 10})
        assert found["passages"] and {p["collection"] for p in found["passages"]} == {"general"}
        assert (await call("get_passage", {"passage_id": "general/leave-policy#2"}))["section"] == "Carry-over"
        with pytest.raises(ToolCallError, match="No passage with that id"):
            await call("get_passage", {"passage_id": "hr/salary-bands#2"})
        docs = (await call("list_documents", {}))["documents"]
        assert {d["collection"] for d in docs} == {"general"}
        refused = await call("ask", {"question": "What is the salary range for Band C?"})
        assert refused["status"] == "not_found" and refused["passages"] == []
    async with server.mcp(HR) as call:
        hr_answer = await call("ask", {"question": "What is the salary range for Band C?"})
        assert hr_answer["status"] == "answered" and "38,000" in hr_answer["answer"]


async def test_mcp_without_a_token_is_refused_at_the_door(server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    with server.http(None) as anonymous:
        resp = anonymous.post("/mcp", json=body, headers={"Accept": "application/json, text/event-stream"})
    assert resp.status_code == 401 and audit(server.kb) == []


async def test_bad_tool_calls_are_refused_and_still_audited(server):
    async with server.mcp(AGENT) as call:
        for name, args, message in [
            ("search_documents", {"query": "leave", "k": "many"}, "validation error"),
            ("search_documents", {"query": "leave", "k": 99}, "k must be"),
            ("ask", {}, "validation error"),
            ("ask", {"question": "x" * 1001}, "longer than 1000"),
            ("get_passage", {"passage_id": "../../etc/passwd"}, "passage ids look like"),
            ("delete_document", {"doc_id": "general/leave-policy"}, "Unknown tool"),
        ]:
            with pytest.raises(ToolCallError, match=message):
                await call(name, args)
    entries = audit(server.kb)
    assert [e[2:] for e in entries] == [
        ("search_documents", "rejected"), ("search", "rejected"), ("ask", "rejected"), ("ask", "rejected"),
        ("get_passage", "rejected"), ("delete_document", "rejected"),
    ]
    assert all(e[0] == "support-agent" and e[1] == "mcp" for e in entries)
    assert len(server.kb.store.documents()) == 9


async def test_every_tool_call_is_audited_exactly_once(server):
    async with server.mcp(AGENT) as call:
        await call("ask", {"question": "How long must a password be?"})
        await call("search_documents", {"query": "parental leave"})
        await call("list_documents", {})
        await call("get_passage", {"passage_id": "general/leave-policy#4"})
    assert audit(server.kb) == [("support-agent", "mcp", action, outcome) for action, outcome in
                                [("ask", "answered"), ("search", "ok"), ("documents", "ok"), ("get_passage", "ok")]]
    assert server.kb.store.audit_entries(4)[3]["passages"] == ["general/it-security-policy#1"]


async def test_mcp_endpoint_is_stateless(server):
    body = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}}
    headers = {"Accept": "application/json, text/event-stream"}
    with server.http(STAFF) as http:
        resp = http.post("/mcp", json=body, headers=headers)
    assert resp.status_code == 200 and "mcp-session-id" not in {k.lower() for k in resp.headers}


async def test_stdio_transport_needs_an_explicit_identity(kb):
    """Without HTTP headers (a local stdio client) there is no token, so the identity is set when
    the server is built. Without one, nothing is readable."""
    from conftest import tool_payload
    from mcp import Client

    tokens = TokenTable(TOKENS)
    async with Client(build_mcp(kb, tokens, stdio_identity=who(HR))) as client:
        payload = tool_payload(await client.call_tool("ask", {"question": "When is the annual pay review?"}))
        assert payload["status"] == "answered" and "April" in payload["answer"]
    async with Client(build_mcp(kb, tokens)) as client:
        with pytest.raises(ToolCallError, match="Not authenticated"):
            tool_payload(await client.call_tool("list_documents", {}))
    assert audit(kb)[-1] == ("unknown", "mcp", "list_documents", "denied")


@pytest.mark.parametrize("retriever", ["bm25", "hybrid"])
async def test_the_stdio_command_serves_one_identity(tmp_path, sample_db, request, retriever):
    """`python -m kb mcp-stdio` started as a real subprocess, as a desktop MCP client would start it."""
    from conftest import ROOT, tool_payload
    from mcp import Client, StdioServerParameters

    if retriever == "hybrid":
        request.getfixturevalue("embedder")  # skipped, with the reason, if the model is not available

    db = tmp_path / "kb.db"
    with closing(sqlite3.connect(sample_db)) as source, closing(sqlite3.connect(db)) as copy:
        source.backup(copy)
    env = {**os.environ, "PYTHONPATH": str(ROOT), "KB_DB": str(db), "KB_TOKENS": json.dumps(TOKENS),
           "KB_MCP_TOKEN": HR, "KB_ANSWER_MODE": "extractive", "KB_ENV_FILE": str(tmp_path / "no.env"),
           "KB_RETRIEVER": retriever}
    params = StdioServerParameters(command=sys.executable, args=["-m", "kb", "mcp-stdio"], env=env, cwd=str(tmp_path))
    async with Client(params) as client:
        payload = tool_payload(await client.call_tool("ask", {"question": "When is the annual pay review?"}))
        assert payload["status"] == "answered" and payload["passages"][0]["passage_id"] == "hr/salary-bands#3"
    entries = Store(str(db)).audit_entries(5)
    assert (entries[0]["actor"], entries[0]["channel"], entries[0]["outcome"]) == ("hana (hr)", "mcp", "answered")

    env["KB_MCP_TOKEN"] = "not-a-configured-token-1234"
    refused = subprocess.run([sys.executable, "-m", "kb", "mcp-stdio"], env=env, cwd=tmp_path, capture_output=True,
                             text=True, timeout=60, stdin=subprocess.DEVNULL)
    assert refused.returncode != 0 and "Set KB_MCP_TOKEN" in refused.stderr


async def test_internal_errors_do_not_leak_to_the_client(server, monkeypatch):
    def explode(*args, **kwargs):
        raise RuntimeError("database password=hunter2 at C:/secret/path")

    monkeypatch.setattr(server.kb, "search", explode)
    async with server.mcp(AGENT) as call:
        with pytest.raises(ToolCallError) as caught:
            await call("search_documents", {"query": "leave"})
    assert "Internal error" in str(caught.value) and "hunter2" not in str(caught.value)


async def test_http_and_mcp_give_the_same_answer(server):
    question = "Who approves an expense claim over 500 pounds?"
    over_http = server.ask(STAFF, question)
    async with server.mcp(STAFF) as call:
        over_mcp = await call("ask", {"question": question})
    assert over_http == over_mcp
