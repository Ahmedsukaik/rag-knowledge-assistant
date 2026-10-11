"""End-to-end demo in one command:  python demo.py   (add --pause to stop after each scene)

Copies the sample documents to a throwaway folder, ingests them into a throwaway database,
starts the HTTP API and the MCP endpoint on a free local port, and walks through the story:
questions answered with sources, questions refused, access rights, a PDF, a planted
instruction inside a document, MCP, the source check on model answers, and the audit log.

No API key is needed: answers are quoted from the documents (extractive mode). Passages are
found by keywords and by meaning (the hybrid retriever); the embedding model runs on this
machine and is downloaded once, about 67 MB, on the first run (python -m kb download-model
does it ahead of time). Set KB_RETRIEVER=bm25 for keywords only, with nothing to download.
Pass --claude to have Claude write the answers instead (needs ANTHROPIC_API_KEY; not run by
the author).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import socket
import sys
import tempfile
import textwrap
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import httpx
import uvicorn

from kb.answer import ClaudeAnswerer
from kb.api import build_app
from kb.cli import utf8_output
from kb.config import TokenTable
from kb.embeddings import EmbeddingUnavailable
from kb.envfile import load_env_file
from kb.ingest import ingest_folder
from kb.retrieval import BM25Retriever, build_retriever
from kb.service import KnowledgeBase
from kb.store import Store

ROOT = Path(__file__).resolve().parent
STAFF, HR, FINANCE, ADMIN, AGENT = (
    "demo-staff-token-00000001", "demo-hr-token-000000000001", "demo-finance-token-0000001",
    "demo-admin-token-000000001", "demo-agent-token-000000001",
)
TOKENS = {
    STAFF: {"name": "maria (sales, staff role)", "role": "staff"},
    HR: {"name": "hana (hr)", "role": "hr"},
    FINANCE: {"name": "farid (finance)", "role": "finance"},
    ADMIN: {"name": "it-admin", "role": "admin"},
    AGENT: {"name": "support-agent", "role": "staff"},
}
PAUSE = "--pause" in sys.argv
BOLD = sys.stdout.isatty() and (os.name != "nt" or "WT_SESSION" in os.environ)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app: Any, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(200):
        if server.started:
            return server
        time.sleep(0.05)
    raise RuntimeError("server did not start")


SCENE = [0]


def say(title: str) -> None:
    """Start the next numbered scene."""
    if PAUSE and SCENE[0]:
        try:
            input("\n(press Enter for the next scene) ")
        except EOFError:  # no keyboard attached: just carry on
            print()
    SCENE[0] += 1
    title = f"{SCENE[0]}. {title}"
    print(f"\n\033[1m=== {title} ===\033[0m" if BOLD else f"\n=== {title} ===")


def out(label: str, text: object = "") -> None:
    """Print 'label text', wrapped so long lines stay readable in a recording."""
    print(textwrap.fill(f"{label} {text}".rstrip(), width=100, subsequent_indent="    "))


def show_answer(who: str, question: str, result: dict) -> None:
    out(f"{who} asks:", f'"{question}"')
    out("  status:", result["status"])
    for line in (result["answer"] or "").splitlines():
        out("  ", line)
    cited = {c["number"] for c in result.get("citations") or []}
    passages = result.get("passages") or []
    for n, p in enumerate(passages, start=1):
        if n in cited:
            out(f"  source [{n}]:", f"{p['label']}  ({p['passage_id']})")
    others = len(passages) - len(cited)
    if others:
        out("  also retrieved:", f"{others} more passage{'s' if others != 1 else ''}, not quoted")
    for w in result.get("warnings") or []:
        out("  WARNING:", w)


def simulated_reply(text_parts: list[tuple[str, dict | None]]) -> SimpleNamespace:
    """A Messages API response in the documented shape, built locally. No API call is made."""
    content = [SimpleNamespace(type="thinking", thinking="")]
    for text, citation in text_parts:
        cites = [SimpleNamespace(type="search_result_location", **citation)] if citation else None
        content.append(SimpleNamespace(type="text", text=text, citations=cites))
    return SimpleNamespace(content=content, stop_reason="end_turn")


async def main(use_claude: bool) -> None:
    from mcp import Client
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    for noisy in ("httpx", "httpx2", "mcp", "kb"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    load_env_file(os.environ.get("KB_ENV_FILE", ".env"))
    if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("--claude needs ANTHROPIC_API_KEY (set it in .env or in your shell).")

    workdir = Path(tempfile.mkdtemp(prefix="kb-demo-"))
    docs = workdir / "docs"
    shutil.copytree(ROOT / "samples" / "docs", docs)
    store = Store(str(workdir / "kb.db"))
    kind = (os.environ.get("KB_RETRIEVER") or "hybrid").lower()
    retriever = build_retriever(store, kind)
    embedder = getattr(retriever, "embedder", None)
    if embedder is not None:
        try:
            embedder.load()  # the first run downloads the model (about 67 MB)
        except EmbeddingUnavailable as exc:
            raise SystemExit(f"The demo needs the embedding model, but {exc}.") from None
    kb = KnowledgeBase(store, retriever, ClaudeAnswerer() if use_claude else None)
    tokens = TokenTable(TOKENS)

    say("Load the documents: Markdown, plain text and PDF, in three collections")
    if embedder is not None:
        out("Search: by keywords and by meaning, with the local embedding model", f"{embedder.name} (no API)")
    else:
        out("Search: by keywords only (KB_RETRIEVER=bm25)")
    report = ingest_folder(store, docs, embedder=embedder)
    for r in report.results:
        out(f"  {r.status:<6} {r.doc_id:<34}", f"{r.kind:<8} {r.chunks} passages")
    again = ingest_folder(store, docs, embedder=embedder)
    reprocessed = again.count("added") + again.count("updated")
    out("Second run, nothing changed:", f"{again.count('unchanged')} unchanged, {reprocessed} re-processed")
    leave = docs / "general" / "leave-policy.md"
    leave.write_text(leave.read_text(encoding="utf-8") + "\n## Volunteering days\n\nEvery employee may take "
                     "2 paid volunteering days per year.\n", encoding="utf-8")
    edited = [r for r in ingest_folder(store, docs, embedder=embedder).results if r.status != "unchanged"]
    out("After editing leave-policy.md:", ", ".join(f"{r.status} {r.doc_id} ({r.chunks} passages, old ones replaced)" for r in edited))

    port = free_port()
    url = f"http://127.0.0.1:{port}"
    serve(build_app(kb, tokens), port)
    client = {t: httpx.Client(base_url=url, headers={"Authorization": f"Bearer {t}"}, timeout=120)
              for t in (STAFF, HR, ADMIN)}
    mode = "Claude API, checked against the sources" if use_claude else "extractive: quoted from the documents, nothing AI-written"

    def ask(token: str, question: str) -> dict:
        return client[token].post("/ask", json={"question": question}).json()

    say("No token, no entry")
    anonymous = httpx.Client(base_url=url, timeout=30)
    out("POST /ask without a token          ->", f"HTTP {anonymous.post('/ask', json={'question': 'x'}).status_code}")
    out("POST /ask with an unknown token    ->", "HTTP " + str(anonymous.post("/ask", json={"question": "x"},
        headers={"Authorization": "Bearer not-a-real-token-123456"}).status_code))
    out("POST /mcp without a token          ->", f"HTTP {anonymous.post('/mcp', json={}).status_code}")
    out("GET /healthz (public)              ->", f"HTTP {anonymous.get('/healthz').status_code}")

    say("A staff member asks a question; the answer comes with its source")
    out("(answers in this run are", f"{mode})")
    show_answer("maria (sales, staff role)", "How many days of annual leave do full-time employees get?",
                ask(STAFF, "How many days of annual leave do full-time employees get?"))

    if embedder is not None:
        say("Different words: found by meaning, not only by matching words")
        q = "Is the pallet scanner still covered if it breaks after a year?"
        show_answer("maria (sales, staff role)", q, ask(STAFF, q))
        keywords_only = KnowledgeBase(store, BM25Retriever(store)).ask(tokens.identify(f"Bearer {STAFF}"), q,
                                                                       channel="demo")
        out("The same question with keyword search only ->", keywords_only.status)

    say("Not in the documents: the assistant says so instead of guessing")
    for q in ("Can I bring my dog to the office?", "What is the capital of France?"):
        r = ask(STAFF, q)
        out(f'maria asks: "{q}" ->', f"{r['status']}: {r['answer']}")

    say("Access rights: HR documents stay with HR")
    q = "What is the salary range for Band C?"
    r = ask(STAFF, q)
    out(f'maria (sales, staff role) asks: "{q}" ->', f"{r['status']}: {r['answer']}")
    found = client[STAFF].post("/search", json={"query": "salary bands Band C", "k": 5}).json()["passages"]
    out('maria searches "salary bands Band C" ->', f"{len(found)} passages, collections: "
        f"{sorted({p['collection'] for p in found}) or 'none'}")
    direct = client[STAFF].get("/passages/hr/salary-bands%232")
    out("maria opens passage hr/salary-bands#2 directly ->", f"HTTP {direct.status_code}, {direct.json()['error']}")
    show_answer("hana (hr)", q, ask(HR, q))

    say("A PDF: the source includes the page number")
    show_answer("maria (sales, staff role)", "What does error code E31 mean on the pallet tracker?",
                ask(STAFF, "What does error code E31 mean on the pallet tracker?"))

    say("A planted instruction inside a document is treated as data")
    planted = next(line for line in (docs / "general" / "office-faq.txt").read_text(encoding="utf-8").splitlines()
                   if "ignore all previous instructions" in line)
    out("Line planted in general/office-faq.txt:", f'"{planted}"')
    show_answer("maria (sales, staff role)", "Are expense claims approved automatically?",
                ask(STAFF, "Are expense claims approved automatically?"))
    show_answer("maria (sales, staff role)", "Where is lost property kept?",  # from the file with the planted line
                ask(STAFF, "Where is lost property kept?"))
    r = ask(STAFF, "What are the salary bands?")
    out('maria asks: "What are the salary bands?" ->', f"{r['status']}: {r['answer']}")

    say("The same rules over MCP, for AI tools and agents")
    async with create_mcp_http_client(headers={"Authorization": f"Bearer {AGENT}"}) as http:
        async with Client(streamable_http_client(f"{url}/mcp", http_client=http)) as mcp:
            tools = (await mcp.list_tools()).tools
            out("Tools:", ", ".join(f"{t.name} ({'read-only' if t.annotations and t.annotations.read_only_hint else 'WRITE'})"
                                    for t in tools))
            res = await mcp.call_tool("ask", {"question": "How long is the warranty on the PT-200?"})
            body = json.loads(res.content[0].text)
            out("support-agent (staff role) calls ask ->", f"{body['status']}: {body['answer']}  "
                f"source: {body['passages'][0]['label']}")
            res = await mcp.call_tool("get_passage", {"passage_id": "hr/salary-bands#2"})
            error = res.content[0].text.split(": ", 1)[-1]  # drop the SDK's "Error executing tool ...:" prefix
            out("support-agent calls get_passage hr/salary-bands#2 ->", f"error: {error}")

    say("Model answers are checked against their sources (simulated replies, no API call)")
    checker = KnowledgeBase(store, retriever)  # same data and search; only the answer step differs
    who = tokens.identify(f"Bearer {STAFF}")
    question = "Are expense claims approved automatically?"
    hits, _ = checker.select_passages(question, checker.visible(who))  # what the model would be sent
    rule = next(i for i, h in enumerate(hits) if h.chunk.chunk_id == "general/expense-policy#3")
    good = simulated_reply([("Every claim needs approval from your line manager, and no claim is "
                             "approved automatically.", {"cited_text": "No claim is approved automatically.",
                             "search_result_index": rule, "source": hits[rule].chunk.chunk_id,
                             "title": hits[rule].chunk.label(), "start_block_index": 2, "end_block_index": 3})])
    bad = simulated_reply([("Claims under 1,000 pounds are approved automatically.",
                            {"cited_text": "Claims under 1,000 pounds are approved automatically.",
                             "search_result_index": rule, "source": hits[rule].chunk.chunk_id,
                             "title": hits[rule].chunk.label(), "start_block_index": 2, "end_block_index": 3})])
    for label, reply in (("A reply whose quote is in the passage", good), ("A reply whose quote is not in the passage", bad)):
        fake = SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(create=lambda reply=reply, **_: reply)))
        checker.answerer = ClaudeAnswerer(client=fake, model="simulated")
        r = checker.ask(who, question, channel="demo")
        out(f"{label} ->", f"{r.status}")
        out("    shown to the user:", r.answer.replace("\n", " "))
        if r.draft:
            out("    withheld model text:", r.draft)

    say("The audit log: who asked, which passages were used, and the outcome (times in UTC)")
    entries = client[ADMIN].get("/admin/audit", params={"limit": 200}).json()["entries"]
    for e in reversed(entries):
        if e["channel"] == "demo":
            continue
        what = (e["query"] or "")[:44] + ("..." if e["query"] and len(e["query"]) > 44 else "")
        n = len(e["passages"])
        used = f"{n} passage{'s' if n != 1 else ''}" if n else ""
        print(f"  {e['ts'][11:19]}  {e['actor']:<25} {e['channel']:<4} {e['action']:<11} {e['outcome']:<9} "
              f"{used:<10} {what}")
    print(f"\nDatabase and document copies for this run are in {workdir}")


if __name__ == "__main__":
    utf8_output()
    try:
        asyncio.run(main(use_claude="--claude" in sys.argv))
    except Exception as exc:
        if "--claude" not in sys.argv:
            raise
        while getattr(exc, "exceptions", None):  # unwrap an exception group to the real cause
            exc = exc.exceptions[0]
        raise SystemExit(f"The run with --claude failed: {type(exc).__name__}: {str(exc)[:300]}") from None
