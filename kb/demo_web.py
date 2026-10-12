"""The web demo: ask the sample documents as Maria (sales), Hana (HR) or Farid (finance).

    python -m kb.demo_web --port 8210        (then open http://127.0.0.1:8210)

    GET  /api/config     personas, story and chip questions, whether the AI and the meaning model are on
    POST /api/ask        {"question": "...", "persona": "maria", "ai": true}: an answer with its sources
    GET  /api/passage    ?id=general/leave-policy%231&persona=maria: the document that passage belongs to
    POST /api/warm       loads the meaning model while the visitor reads the first screen
    GET  /healthz

It is this product's own KnowledgeBase (access rules, thresholds, quoting, the source check)
over a prebuilt, read-only copy of the sample documents (demo_index.py). The server keeps
nothing between requests and writes nothing to disk; each answer carries its own audit entries.

Answers are written by Claude Haiku 5.5 when ANTHROPIC_API_KEY is set and KB_DEMO_AI is not
"off": effort low, at most 2,000 output tokens, a 20 second timeout, no retries, and every
sentence through the source check. Without a key, over a limit, or when the API fails (spend
limit, rate limit, key, timeout), the answer is quoted word for word instead, and the page says
which and why.

Questions the demo shows have stored vectors. Questions people type are embedded by the ONNX
model in demo/model; with KB_DEMO_SEARCH=keywords (or without the model files) they are searched
by keywords only, and the page says so.
"""
from __future__ import annotations

import argparse
import ipaddress
import json
import logging
import os
import threading
import time
from collections import OrderedDict, deque
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .answer import Answer, AnswerUnavailable, ClaudeAnswerer, ExtractiveAnswerer
from .demo_index import CHIPS, LEAK_CHECKS, PERSONAS, STORY, DemoIndex, collect_audit
from .embeddings import Embedder
from .retrieval import Hit
from .service import BadRequest, KnowledgeBase, ModelBudget, NotFound, clean_query
from .store import MAX_QUERY_CHARS

log = logging.getLogger("kb.demo_web")
PUBLIC = Path(__file__).resolve().parent.parent / "public"
REPO = "https://github.com/Ahmedsukaik/rag-knowledge-assistant"
MODEL = "claude-haiku-5-5"
MAX_TOKENS = 2000
TIMEOUT_S = 20.0
# USD per million tokens (Anthropic's pricing page, October 2026), for the cost shown on the page.
PRICES = {MODEL: {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01}}
AI_SETTING, SEARCH_SETTING = "KB_DEMO_AI", "KB_DEMO_SEARCH"
MAX_BODY = 8_000                    # a question is at most 1,000 characters
MAX_DEPTH = 4
PER_VISITOR_REQUESTS = (300, 600)   # requests of any kind per visitor (IP) per 10 minutes
PER_VISITOR_AI = (20, 600)          # AI answers per visitor per 10 minutes
PER_INSTANCE_AI = (600, 3600)       # AI answers per server instance per hour
PER_VISITOR_EMBED = (40, 600)       # typed questions for the meaning model per visitor per 10 minutes
PER_INSTANCE_EMBED = (3000, 3600)   # and per server instance per hour (the story and chips do not count)
MAX_ACTIVE = 2                      # requests in progress at once per visitor
MAX_KEYS = 5000                     # visitors the limiter remembers per instance
UNLIMITED = ModelBudget(10**9, 10**9, 10**9)  # the Limiter below decides instead (per visitor, not per persona)

SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self'; style-src 'self' https://fonts.googleapis.com; "
                                "font-src https://fonts.gstatic.com; img-src 'self' data:; connect-src 'self'; "
                                "frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "X-Frame-Options": "DENY",
}

# Why an answer is quoted instead of written by the AI, as the page says it.
QUOTED_BECAUSE = {
    "no_key": "Quoted mode: every answer is the documents' own sentence, word for word.",
    "off": "Quoted mode: every answer is the documents' own sentence, word for word.",
    "chosen": "You chose quoted answers, so this one is quoted word for word from the documents (no AI call).",
    "visitor_limit": "You have asked many questions in a few minutes, so this answer is quoted word for word instead.",
    "busy": "The demo's AI is busy right now, so this answer is quoted word for word instead.",
    "budget": "The demo's AI budget is used up, so this answer is quoted word for word instead.",
    "rate_limited": "The AI service is busy right now, so this answer is quoted word for word instead.",
    "auth": "The AI service is not available to this demo right now, so this answer is quoted word for word instead.",
    "timeout": "The AI took too long to answer, so this answer is quoted word for word instead.",
    "error": "The AI service is not available right now, so this answer is quoted word for word instead.",
    "invalid": "The AI could not take this question, so this answer is quoted word for word instead.",
}
SEARCH_NOTES = {
    "off": "Questions you type are searched by keywords only in this copy of the demo: the meaning model is not on "
           "this server. The story and the suggested questions use stored meaning vectors.",
    "missing": "The meaning model is not on this server, so questions you type are searched by keywords only.",
    "broken": "The meaning model could not be loaded, so questions you type are searched by keywords only.",
    "mismatch": "The meaning model on this server does not match the stored vectors, so questions you type are "
                "searched by keywords only.",
    "visitor_limit": "You have asked many new questions in a few minutes, so this one was searched by keywords only.",
    "busy": "The demo is busy right now, so this question was searched by keywords only.",
}


class ActionError(Exception):
    """The request cannot be served as sent. The message is safe to show."""

    def __init__(self, message: str, status: int = 400):
        super().__init__(message)
        self.status = status


class Limiter:
    """Sliding-window counters, per server instance. The hard limit is the spend cap on the API
    key itself; this only stops one visitor from using up the demo's AI budget alone."""

    def __init__(self) -> None:
        self._hits: OrderedDict[str, deque[float]] = OrderedDict()
        self._instance: dict[str, deque[float]] = {}  # whole-instance caps: never forgotten to make room
        self._active: dict[str, int] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _take(hits: deque[float], limit: int, window: float, now: float) -> bool:
        while hits and now - hits[0] > window:
            hits.popleft()
        if len(hits) >= limit:
            return False
        hits.append(now)
        return True

    def allow(self, key: str, limit: int, window: float) -> bool:
        """One more for this visitor, if under `limit` in the last `window` seconds."""
        now = time.monotonic()
        with self._lock:
            hits = self._hits.pop(key, None) or deque()
            self._hits[key] = hits  # most recently seen last
            allowed = self._take(hits, limit, window, now)
            while len(self._hits) > MAX_KEYS:  # forget the visitor seen longest ago
                self._hits.popitem(last=False)
            return allowed

    def allow_instance(self, name: str, limit: int, window: float) -> bool:
        """One more for the whole server instance. Kept apart from the visitors, so many new
        visitors cannot push this counter out and reset it."""
        with self._lock:
            return self._take(self._instance.setdefault(name, deque()), limit, window, time.monotonic())

    def enter(self, key: str) -> bool:
        """One more request in progress for this visitor, if under MAX_ACTIVE."""
        with self._lock:
            if self._active.get(key, 0) >= MAX_ACTIVE:
                return False
            self._active[key] = self._active.get(key, 0) + 1
            return True

    def leave(self, key: str) -> None:
        with self._lock:
            left = self._active.get(key, 1) - 1
            if left > 0:
                self._active[key] = left
            else:
                self._active.pop(key, None)


# ---------------------------------------------------------------------------- Claude
def _get(obj: Any, name: str) -> Any:
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def cost_usd(model: str, usage: dict[str, int]) -> float:
    price = PRICES.get(model)
    if not price:
        return 0.0
    return (usage.get("input_tokens", 0) * price["input"] + usage.get("output_tokens", 0) * price["output"]
            + usage.get("cache_creation_input_tokens", 0) * price["cache_write"]
            + usage.get("cache_read_input_tokens", 0) * price["cache_read"]) / 1_000_000


class RecordingClient:
    """Wraps an Anthropic client and keeps the token counts of its last reply, for the cost card."""

    def __init__(self, client: Any):
        self._client = client
        self.usage: dict[str, int] = {}
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **request: Any) -> Any:
        response = self._client.beta.messages.create(**request)
        raw = _get(response, "usage")
        self.usage = {k: int(_get(raw, k) or 0) for k in ("input_tokens", "output_tokens", "cache_creation_input_tokens",
                                                            "cache_read_input_tokens")}
        return response


def failure_kind(exc: BaseException | None) -> str:
    """budget, rate_limited, auth, timeout, invalid or error, from an Anthropic SDK exception."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - the demo installs it
        return "error"
    body = str(getattr(exc, "body", None) or getattr(exc, "message", "") or "")
    if isinstance(exc, anthropic.RateLimitError):
        return "budget" if "spend_limit" in body or "spend limit" in body.lower() else "rate_limited"
    if isinstance(exc, anthropic.BadRequestError):
        return "budget" if "usage limits" in body.lower() or "credit balance" in body.lower() else "invalid"
    if isinstance(exc, (anthropic.AuthenticationError, anthropic.PermissionDeniedError)):
        return "auth"
    if isinstance(exc, anthropic.APITimeoutError):
        return "timeout"
    return "error"


class ClaudeOrQuotes:
    """The answer step of one request: Claude when this visitor may have an AI answer and the
    API answers, the same passages quoted word for word otherwise. Records which one answered,
    why, and the token counts."""

    mode = "claude"

    def __init__(self, client: Any, embedder: Embedder | None, gate: Callable[[], str | None]):
        self.recorder = RecordingClient(client)
        # Thinking off, so it cannot use up the 2,000 tokens (Haiku 5.5 accepts that at effort low).
        self.claude = ClaudeAnswerer(client=self.recorder, model=MODEL, effort="low", max_tokens=MAX_TOKENS, thinking="disabled",
                                     embedder=embedder)
        self.quoted = ExtractiveAnswerer(embedder=embedder)
        self.gate = gate
        self.used = "none"  # none: no passage cleared the bar, nothing was asked
        self.reason: str | None = None
        self.usage: dict[str, int] = {}

    def answer(self, question: str, hits: list[Hit]) -> Answer:
        over = self.gate()
        if over:
            self.used, self.reason = "quoted", over
            return self.quoted.answer(question, hits)
        try:
            answer = self.claude.answer(question, hits)
        except AnswerUnavailable as exc:
            self.used, self.reason = "quoted", failure_kind(exc.__cause__)
            return self.quoted.answer(question, hits)
        self.used, self.usage = "claude", dict(self.recorder.usage)
        return answer


class SimulatedBadDraft:
    """Stands in for the Anthropic API in step 4 of the story, and makes no call: its draft says
    30 days of annual leave where the cited sentence says 25, and adds a claim with no source."""

    def __init__(self) -> None:
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self.requests: list[dict[str, Any]] = []

    def _create(self, **request: Any) -> dict[str, Any]:
        self.requests.append(request)
        results = [b for b in request["messages"][0]["content"] if b.get("type") == "search_result"]
        first = results[0]
        return {
            "model": request["model"], "stop_reason": "end_turn", "usage": {"input_tokens": 0, "output_tokens": 0},
            "content": [
                {"type": "text", "text": "Full-time employees get 30 days of paid annual leave a year.",
                 "citations": [{"type": "search_result_location", "cited_text": first["content"][0]["text"],
                                "search_result_index": 0, "source": first["source"], "title": first["title"],
                                "start_block_index": 0, "end_block_index": 1}]},
                {"type": "text", "text": " Unused days never expire, so you can save them up."},
            ],
        }


def live_client_from_env() -> tuple[Any, str | None]:
    """(client, None) when the AI is on, else (None, why): no_key or off."""
    if os.environ.get(AI_SETTING, "on").strip().lower() in ("off", "0", "false", "no"):
        return None, "off"
    if not (os.environ.get("ANTHROPIC_API_KEY") or "").strip():
        return None, "no_key"
    import anthropic

    # No retries: a rate limit can ask to wait longer than a serverless request may run. A failed
    # call falls back to quoted answers at once.
    return anthropic.Anthropic(timeout=anthropic.Timeout(TIMEOUT_S, connect=3.0), max_retries=0), None


def search_from_env() -> str:
    value = (os.environ.get(SEARCH_SETTING) or "meaning").strip().lower()
    return "keywords" if value in ("keywords", "keyword", "bm25", "off", "b") else "meaning"


# ---------------------------------------------------------------------------- the demo itself
class Demo:
    def __init__(self, index: DemoIndex, client: Any = None, why_no_ai: str | None = None):
        self.index, self.client = index, client
        self.why_no_ai = why_no_ai if client is None else None

    def config(self) -> dict[str, Any]:
        docs = self.index.store.documents()
        collections: dict[str, int] = {}
        for d in docs:
            collections[d["collection"]] = collections.get(d["collection"], 0) + 1
        state = self.index.model_state
        return {
            "live": self.client is not None, "model": MODEL if self.client is not None else None,
            "quoted_because": QUOTED_BECAUSE.get(self.why_no_ai or "") if self.client is None else None,
            "search": {"typed": "meaning" if state in ("waiting", "ready") else "keywords", "model": state,
                       "note": SEARCH_NOTES.get(state)},
            "personas": [p.public() for p in PERSONAS.values()],
            "collections": collections,
            "documents": [{"title": d["title"], "collection": d["collection"]} for d in docs],
            "leak_checks": LEAK_CHECKS,
            "story": STORY, "chips": CHIPS,
            "eval": self.index.manifest.get("eval", {}),
            "price": PRICES[MODEL], "max_question": MAX_QUERY_CHARS, "repo": REPO,
        }

    def warm(self) -> dict[str, Any]:
        started = time.monotonic()
        state = self.index.load_model()
        return {"meaning_model": state, "ms": int((time.monotonic() - started) * 1000)}

    def ask(self, question: Any, persona_id: Any, want_ai: bool = True, simulate: Any = None,
            gate: Callable[[], str | None] = lambda: None,
            embed_gate: Callable[[], str | None] = lambda: None) -> dict[str, Any]:
        """gate: asked right before an AI answer; embed_gate: asked before a typed question goes to the
        meaning model. Each returns None to go ahead, or the reason it may not."""
        persona = PERSONAS.get(persona_id) if isinstance(persona_id, str) else None
        if persona is None:
            raise ActionError("Pick Maria, Hana or Farid.")
        text = clean_query(question)
        if simulate not in (None, "bad_draft"):
            raise ActionError("Unknown test.")
        if simulate and text != STORY["leave"]:
            raise ActionError("The forced draft is only for the annual leave question in the story.")
        why_keywords = None  # a key of SEARCH_NOTES when the question is searched by keywords only
        if not self.index.embedder.knows(text):  # stored questions (story, chips, eval) cost nothing
            state = self.index.load_model()
            why_keywords = None if state == "ready" else state
            if why_keywords is None:
                why_keywords = embed_gate()
        meaning = why_keywords is None
        retriever = self.index.hybrid if meaning else self.index.keywords
        embedder: Embedder | None = self.index.embedder if meaning else None
        answerer: Any
        written, reason, step = "quoted", None, None
        if simulate:
            answerer, written = ClaudeAnswerer(client=SimulatedBadDraft(), model=MODEL, effort="low", max_tokens=MAX_TOKENS,
                                               thinking="disabled", embedder=embedder), "simulated"
        elif self.client is None:
            answerer, reason = ExtractiveAnswerer(embedder=embedder), self.why_no_ai or "no_key"
        elif not want_ai:
            answerer, reason = ExtractiveAnswerer(embedder=embedder), "chosen"
        else:
            step = ClaudeOrQuotes(self.client, embedder, gate)
            answerer = step
        kb = KnowledgeBase(self.index.store, retriever, answerer=answerer, budget=UNLIMITED)
        started = time.monotonic()
        with collect_audit() as audit:
            result = kb.ask(persona.identity, text, channel="demo")
        elapsed = int((time.monotonic() - started) * 1000)
        if step is not None:
            written, reason = ("claude", None) if step.used == "claude" else ("quoted", step.reason or "none")
        if result.status == "not_found" and written != "simulated" and (step is None or step.used == "none"):
            written, reason = "none", None  # nothing was asked or quoted: no passage this persona may read cleared the bar
        keywords_alone = None
        if meaning and result.status != "not_found":
            found, _ = KnowledgeBase(self.index.store, self.index.keywords).select_passages(text, kb.visible(persona.identity))
            keywords_alone = bool(found)
        usage = step.usage if step is not None and written == "claude" else {}
        warnings, problems = split_warnings(result.warnings)
        return {
            "question": text, "persona": persona.public(), "status": result.status, "answer": result.answer,
            "parts": result.parts, "citations": result.citations, "passages": result.passages,
            "warnings": warnings, "problems": problems,
            "draft": result.draft if result.status == "flagged" else None,
            "written_by": written, "model": MODEL if written in ("claude", "simulated") else None,
            # The demo-wide reasons (no key, AI off) are said once on the page, not on every card.
            "quoted_because": None if reason in ("no_key", "off") else QUOTED_BECAUSE.get(reason or ""),
            "search": {"by": "meaning" if meaning else "keywords", "keywords_alone": keywords_alone,
                       "note": None if meaning else SEARCH_NOTES.get(why_keywords or "missing", SEARCH_NOTES["missing"])},
            "cost": {"usd": round(cost_usd(MODEL, usage), 7), "input_tokens": usage.get("input_tokens", 0),
                     "output_tokens": usage.get("output_tokens", 0)},
            "audit": audit, "ms": elapsed,
        }

    def passage(self, passage_id: Any, persona_id: Any) -> dict[str, Any]:
        persona = PERSONAS.get(persona_id) if isinstance(persona_id, str) else None
        if persona is None:
            raise ActionError("Pick Maria, Hana or Farid.")
        kb = KnowledgeBase(self.index.store, self.index.keywords)
        with collect_audit() as audit:
            found = kb.get_passage(persona.identity, passage_id, channel="demo")  # NotFound for a restricted one
        doc = self.index.store.document(found["doc_id"]) or {}
        chunks = [c for c in self.index.store.chunks(frozenset({found["collection"]})) if c.doc_id == found["doc_id"]]
        return {
            "passage_id": found["passage_id"],
            "document": {"doc_id": found["doc_id"], "title": doc.get("title", found["title"]), "collection": found["collection"],
                         "kind": doc.get("kind"), "pages": doc.get("pages"),
                         "passages": [{"passage_id": c.chunk_id, "section": c.section, "page": c.page, "text": c.text,
                                       "flagged": list(c.flagged)} for c in chunks]},
            "audit": audit,
        }


def split_warnings(warnings: list[str]) -> tuple[list[str], list[str]]:
    """(notes for the reader, the source check's reasons). A flagged answer's warnings are the
    service's own notes, then "The AI answer did not pass the source check...", then the reasons."""
    for i, w in enumerate(warnings):
        if w.startswith("The AI answer did not pass"):
            return warnings[:i], warnings[i + 1:]
    return list(warnings), []


def visitor_key(address: str) -> str:
    """The key the limits count by: an IPv4 address as it is, an IPv6 address by its /64 network
    (one home or phone connection usually holds a whole /64, so each address in it is the same visitor)."""
    try:
        ip = ipaddress.ip_address(address.strip("[]"))
    except ValueError:
        return address[:64]
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is None:
        return str(ipaddress.ip_network(f"{ip}/64", strict=False))
    return str(ip.ipv4_mapped or ip) if isinstance(ip, ipaddress.IPv6Address) else str(ip)


def _no_constants(name: str) -> Any:
    raise ValueError(f"{name} is not allowed")


def _depth(value: Any, level: int = 0) -> int:
    if level > MAX_DEPTH:
        return level
    if isinstance(value, dict):
        return max([_depth(v, level + 1) for v in value.values()] or [level + 1])
    if isinstance(value, list):
        return max([_depth(v, level + 1) for v in value] or [level + 1])
    return level


def create_app(index: DemoIndex | None = None, client: Any = "env", serve_static: bool = True) -> FastAPI:
    """client: an Anthropic client (or a stand-in in tests), None for quoted answers only, or
    "env" to decide from ANTHROPIC_API_KEY and KB_DEMO_AI. Without `index`, the demo/ folder is
    loaded, with or without the meaning model as KB_DEMO_SEARCH says."""
    if index is None:
        index = DemoIndex(model_folder=None) if search_from_env() == "keywords" else DemoIndex()
    why = None
    if client == "env":
        client, why = live_client_from_env()
    app = FastAPI(title="RAG Knowledge Assistant (demo)", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.demo = Demo(index, client, why or ("no_key" if client is None else None))
    app.state.limiter = Limiter()

    def visitor(request: Request) -> str:
        """The visitor's address. Behind Vercel, its own headers carry the real client; anywhere
        else, a client could write X-Forwarded-For itself, so the connection's address is used."""
        if os.environ.get("VERCEL"):
            forwarded = request.headers.get("x-vercel-forwarded-for") or request.headers.get("x-forwarded-for", "")
            if forwarded.strip():
                return visitor_key(forwarded.split(",")[0].strip()[:64])
        return visitor_key(request.client.host if request.client else "unknown")

    def count(request: Request) -> str:
        who = visitor(request)
        if not app.state.limiter.allow("r:" + who, *PER_VISITOR_REQUESTS):
            raise ActionError("Too many requests from this connection. Wait a few minutes and try again.", 429)
        return who

    async def body(request: Request) -> dict[str, Any]:
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > MAX_BODY:
            raise ActionError("That question is too long for the demo.", 413)
        raw = bytearray()
        async for chunk in request.stream():  # stop reading as soon as the body is too big
            raw += chunk
            if len(raw) > MAX_BODY:
                raise ActionError("That question is too long for the demo.", 413)
        try:
            data = json.loads(bytes(raw) or b"{}", parse_constant=_no_constants)  # NaN and Infinity are refused
        except (ValueError, RecursionError) as e:
            raise ActionError("Bad request.") from e
        if not isinstance(data, dict) or _depth(data) > MAX_DEPTH:
            raise ActionError("Bad request.")
        return data

    @app.middleware("http")
    async def headers(request: Request, call_next: Any) -> Any:
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.exception_handler(ActionError)
    async def _action_error(request: Request, exc: ActionError) -> JSONResponse:
        return JSONResponse({"error": "request", "message": str(exc)}, status_code=exc.status)

    @app.exception_handler(BadRequest)
    async def _bad_request(request: Request, exc: BadRequest) -> JSONResponse:
        message = str(exc)
        return JSONResponse({"error": "request", "message": message[:1].upper() + message[1:] + "."}, status_code=400)

    @app.exception_handler(NotFound)
    async def _not_found(request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"error": "not_found", "message": "No passage with that id in your documents."}, status_code=404)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unexpected error")
        return JSONResponse({"error": "server", "message": "Something went wrong on our side. Try again."}, status_code=500)

    @app.get("/api/config")
    def config(request: Request) -> dict[str, Any]:
        count(request)
        result: dict[str, Any] = app.state.demo.config()
        return result

    @app.post("/api/warm")
    async def warm(request: Request) -> dict[str, Any]:
        count(request)
        result: dict[str, Any] = await run_in_threadpool(app.state.demo.warm)
        return result

    @app.post("/api/ask")
    async def ask(request: Request) -> dict[str, Any]:
        who = count(request)
        data = await body(request)

        def gate() -> str | None:
            """Asked right before an AI answer: may this visitor have one now?"""
            if not app.state.limiter.allow("ai:" + who, *PER_VISITOR_AI):
                return "visitor_limit"
            if not app.state.limiter.allow_instance("ai", *PER_INSTANCE_AI):
                return "busy"
            return None

        def embed_gate() -> str | None:
            """Asked before a typed question goes to the meaning model (it costs CPU time)."""
            if not app.state.limiter.allow("embed:" + who, *PER_VISITOR_EMBED):
                return "visitor_limit"
            if not app.state.limiter.allow_instance("embed", *PER_INSTANCE_EMBED):
                return "busy"
            return None

        if not app.state.limiter.enter(who):
            raise ActionError("One moment: the previous answer is still on its way.", 409)
        try:
            # The search and an AI answer block: run them in a worker thread so other visitors are not kept waiting.
            result: dict[str, Any] = await run_in_threadpool(
                app.state.demo.ask, data.get("question"), data.get("persona"), data.get("ai", True) is not False,
                data.get("simulate"), gate, embed_gate)
        finally:
            app.state.limiter.leave(who)
        return result

    @app.get("/api/passage")
    def passage(request: Request, id: str = "", persona: str = "") -> dict[str, Any]:  # noqa: A002 - the query name
        count(request)
        result: dict[str, Any] = app.state.demo.passage(id, persona)
        return result

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    if serve_static and PUBLIC.is_dir():
        @app.get("/")
        def index_page() -> FileResponse:
            return FileResponse(PUBLIC / "index.html")

        app.mount("/", StaticFiles(directory=PUBLIC), name="static")
    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description="RAG Knowledge Assistant: the web demo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8210)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    demo: Demo = app.state.demo
    answers = f"written by Claude ({MODEL})" if demo.client is not None else \
        "quoted word for word (no ANTHROPIC_API_KEY, or KB_DEMO_AI=off)"
    typed = "by meaning and keywords" if demo.index.model_state == "waiting" else "by keywords only"
    print(f"Demo: http://{args.host}:{args.port}/   (answers {answers}; typed questions searched {typed})", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
