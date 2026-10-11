from __future__ import annotations

import json
import re
import socket
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager, closing, contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import httpx
import pytest
import uvicorn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.api import build_app  # noqa: E402
from kb.config import Identity, TokenTable  # noqa: E402
from kb.ingest import ingest_folder  # noqa: E402
from kb.service import KnowledgeBase  # noqa: E402
from kb.store import Store  # noqa: E402

SAMPLES = ROOT / "samples" / "docs"

# Test tokens, independent of demo.py.
STAFF, HR, FINANCE, ADMIN, AGENT = (
    "test-staff-token-00000001", "test-hr-token-000000000001", "test-finance-token-0000001",
    "test-admin-token-000000001", "test-agent-token-000000001",
)
TOKENS = {
    STAFF: {"name": "maria (sales)", "role": "staff"},
    HR: {"name": "hana (hr)", "role": "hr"},
    FINANCE: {"name": "farid (finance)", "role": "finance"},
    ADMIN: {"name": "it-admin", "role": "admin"},
    AGENT: {"name": "support-agent", "role": "staff"},
}


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
__all__ = ["ADMIN", "AGENT", "FINANCE", "HR", "STAFF", "SAMPLES", "TOKENS", "free_port", "serve", "who", "ToolCallError"]


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


# The project's settings, removed for every test (and its subprocesses) so that nothing set in
# the developer's shell or .env file changes a result. Tests that need one set it themselves.
# KB_MODEL_DIR stays: it only says where the embedding model is kept.
PROJECT_SETTINGS = ("KB_TOKENS", "KB_ROLES", "KB_DB", "KB_DOCS_DIR", "KB_ANSWER_MODE", "KB_RETRIEVER", "KB_EMBED_MODEL",
                    "KB_CLAUDE_MODEL", "KB_CLAUDE_EFFORT", "KB_CLAUDE_FALLBACKS", "KB_CLAUDE_PER_MINUTE",
                    "KB_CLAUDE_PER_DAY", "KB_CLAUDE_PER_PERSON_DAY", "KB_URL", "KB_TOKEN", "KB_MCP_TOKEN")


@pytest.fixture(autouse=True)
def no_settings_from_the_shell(monkeypatch, tmp_path_factory):
    for name in PROJECT_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("KB_ENV_FILE", str(tmp_path_factory.getbasetemp() / "no-such.env"))  # never read a .env


def who(token: str) -> Identity:
    identity = TokenTable(TOKENS).identify(f"Bearer {token}")
    assert identity is not None
    return identity


@pytest.fixture(scope="session")
def sample_db(tmp_path_factory) -> Path:
    """The sample documents, ingested once per test run."""
    path = tmp_path_factory.mktemp("template") / "kb.db"
    report = ingest_folder(Store(str(path)), SAMPLES)
    assert not report.skipped
    return path


@pytest.fixture
def store(tmp_path, sample_db) -> Store:
    """A fresh copy of the sample database for each test."""
    target = tmp_path / "kb.db"
    with closing(sqlite3.connect(sample_db)) as source, closing(sqlite3.connect(target)) as copy:
        source.backup(copy)
    return Store(str(target))


@pytest.fixture
def kb(store) -> KnowledgeBase:
    return KnowledgeBase(store)


class ToolCallError(Exception):
    """An MCP tool returned an error."""


def tool_payload(result: Any) -> dict[str, Any]:
    text = " ".join(getattr(block, "text", "") for block in result.content or [])
    if result.is_error:
        raise ToolCallError(re.sub(r"^Error executing tool \w+: ", "", text))
    if result.structured_content is not None:
        return result.structured_content
    return json.loads(text)


@dataclass
class Server:
    url: str
    kb: KnowledgeBase
    clients: dict[str | None, httpx.Client] = field(default_factory=dict)

    @contextmanager
    def http(self, token: str | None = STAFF) -> Iterator[httpx.Client]:
        """A client with this token. Clients are reused: creating one costs ~0.3 s on Windows."""
        if token not in self.clients:
            headers = {"Authorization": f"Bearer {token}"} if token else {}
            self.clients[token] = httpx.Client(base_url=self.url, headers=headers, timeout=30)
        yield self.clients[token]

    def ask(self, token: str, question: str) -> dict[str, Any]:
        with self.http(token) as http:
            resp = http.post("/ask", json={"question": question})
        assert resp.status_code == 200, resp.text
        return resp.json()

    @asynccontextmanager
    async def mcp(self, token: str = AGENT):
        """Yield `call(tool, args) -> dict` over a real MCP Streamable HTTP connection."""
        from mcp import Client
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        async with create_mcp_http_client(headers={"Authorization": f"Bearer {token}"}) as http:
            async with Client(streamable_http_client(f"{self.url}/mcp", http_client=http)) as client:
                async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
                    return tool_payload(await client.call_tool(name, arguments))

                call.client = client  # type: ignore[attr-defined]
                yield call


@contextmanager
def running_server(kb: KnowledgeBase) -> Iterator[Server]:
    port = free_port()
    running = serve(build_app(kb, TokenTable(TOKENS)), port)
    s = Server(f"http://127.0.0.1:{port}", kb)
    try:
        yield s
    finally:
        for client in s.clients.values():
            client.close()
        running.should_exit = True


@pytest.fixture
def server(kb) -> Iterator[Server]:
    with running_server(kb) as s:
        yield s


# ---------------------------------------------------------------------------- the hybrid retriever
# Most tests use the keyword retriever, which needs no model. The tests of the default (hybrid)
# retriever need the local embedding model; if it cannot be loaded (not downloaded and no
# network), they are skipped with the reason, never passed.
@pytest.fixture(scope="session")
def embedder():
    from kb.embeddings import EmbeddingUnavailable, shared_embedder

    model = shared_embedder()
    try:
        model.load()
    except EmbeddingUnavailable as exc:
        pytest.skip(f"embedding model not available, so the hybrid retriever cannot be tested: {exc}")
    return model


@pytest.fixture(scope="session")
def hybrid_db(tmp_path_factory, embedder) -> Path:
    """The sample documents with a vector for every passage, ingested once per test run."""
    path = tmp_path_factory.mktemp("hybrid-template") / "kb.db"
    report = ingest_folder(Store(str(path)), SAMPLES, embedder=embedder)
    assert not report.skipped
    return path


@pytest.fixture
def hybrid_store(tmp_path, hybrid_db) -> Store:
    target = tmp_path / "hybrid.db"
    with closing(sqlite3.connect(hybrid_db)) as source, closing(sqlite3.connect(target)) as copy:
        source.backup(copy)
    return Store(str(target))


@pytest.fixture
def hybrid_kb(hybrid_store, embedder) -> KnowledgeBase:
    from kb.retrieval import HybridRetriever

    return KnowledgeBase(hybrid_store, HybridRetriever(hybrid_store, embedder))


@pytest.fixture
def hybrid_server(hybrid_kb) -> Iterator[Server]:
    with running_server(hybrid_kb) as s:
        yield s


def audit(kb: KnowledgeBase) -> list[tuple[str, str, str, str]]:
    """(actor, channel, action, outcome) for every audit entry, oldest first."""
    return [(e["actor"], e["channel"], e["action"], e["outcome"]) for e in reversed(kb.store.audit_entries(1000))]
