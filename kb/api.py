"""The HTTP API, the web chat page and the MCP endpoint, served together on one port.

    GET    /                          public    the chat page (static: no document text, no data)
    GET    /healthz                   public    liveness check
    GET    /me                        any role  who the token belongs to and what it may read
    POST   /ask                       any role  {"question": "..."}
    POST   /search                    any role  {"query": "...", "k": 5}
    GET    /documents                 any role  the documents your role may read
    GET    /passages/{passage_id}     any role  one passage, e.g. /passages/general/leave-policy%231
    POST   /admin/ingest              admin     add or replace one document
    DELETE /admin/documents/{doc_id}  admin     remove a document
    GET    /admin/audit?limit=50      admin     latest audit entries
    POST   /mcp                       any role  MCP (Streamable HTTP, stateless), read-only tools

Every request except / and /healthz needs `Authorization: Bearer <token>`. A request without a
known token is refused with 401 before it reaches any route (and written to the application
log, not the audit table, so anonymous traffic cannot fill it). The chat page itself is public
because it holds nothing: the person pastes their token into it, and every call it makes to
the API carries that token.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import logging
import re
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import BaseModel, Field

from .config import Identity, TokenTable
from .ingest import MAX_FILENAME
from .mcp_server import build_mcp
from .service import BadRequest, Forbidden, KnowledgeBase, NotFound
from .text import one_line

log = logging.getLogger("kb.api")

MAX_UPLOAD_CHARS = 8 * 1024 * 1024    # base64 of a 5 MB file, with room to spare
BODY_LIMIT = 64 * 1024                # bytes, for every request body except an upload
INGEST_BODY_LIMIT = 9 * 1024 * 1024   # bytes, for POST /admin/ingest
PUBLIC_PATHS = frozenset({"/", "/healthz"})


class Unauthenticated(Exception):
    """No valid bearer token (normally stopped by the Gate already)."""


def _page_and_headers() -> tuple[str, dict[str, str]]:
    """The chat page, and headers that only allow its own inline script and style to run."""
    page = (Path(__file__).resolve().parent / "web" / "index.html").read_text(encoding="utf-8")

    def digest(tag: str) -> str:
        blocks = re.findall(rf"<{tag}>(.*?)</{tag}>", page, re.S)
        if len(blocks) != 1:
            raise RuntimeError(f"kb/web/index.html must have exactly one <{tag}> block")
        return "'sha256-" + base64.b64encode(hashlib.sha256(blocks[0].encode("utf-8")).digest()).decode() + "'"

    policy = (f"default-src 'none'; script-src {digest('script')}; style-src {digest('style')}; "
              "connect-src 'self'; img-src data:; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
    return page, {"Content-Security-Policy": policy, "X-Content-Type-Options": "nosniff",
                  "Referrer-Policy": "no-referrer", "Cache-Control": "no-store", "X-Frame-Options": "DENY"}


PAGE, PAGE_HEADERS = _page_and_headers()


# Length limits on questions are checked by KnowledgeBase (the same rule for HTTP and MCP);
# the Gate caps the size of every request body.
class AskBody(BaseModel):
    question: str = Field(description="Your question, up to 1000 characters.")


class SearchBody(BaseModel):
    query: str = Field(description="Search words, up to 1000 characters.")
    k: int = Field(default=5, description="Number of passages, 1 to 20.")


class IngestBody(BaseModel):
    collection: str = Field(max_length=31, description="Collection name, for example general, hr or finance.")
    filename: str = Field(max_length=MAX_FILENAME, description="File name with extension: .md, .txt or .pdf.")
    text: str | None = Field(default=None, max_length=MAX_UPLOAD_CHARS, description="Content of a .md or .txt file.")
    content_base64: str | None = Field(default=None, max_length=MAX_UPLOAD_CHARS, description="Any file, base64-encoded.")
    replace: bool = Field(default=False, description="Replace a document with the same id from a different file.")


def build_api(kb: KnowledgeBase, tokens: TokenTable, mcp_app: Any | None = None) -> FastAPI:
    lifespan = (lambda app: mcp_app.router.lifespan_context(mcp_app)) if mcp_app is not None else None
    app = FastAPI(title="RAG knowledge assistant", version="1.0.0", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)

    def caller(request: Request) -> Identity:
        who = tokens.identify(request.headers.get("authorization"))
        if who is None:  # normally stopped by Gate already
            raise Unauthenticated()
        return who

    @app.exception_handler(BadRequest)
    async def bad_request(request: Request, exc: BadRequest) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=400)

    @app.exception_handler(NotFound)
    async def not_found(request: Request, exc: NotFound) -> JSONResponse:
        return JSONResponse({"error": str(exc)}, status_code=404)

    @app.exception_handler(Forbidden)
    async def forbidden(request: Request, exc: Forbidden) -> JSONResponse:
        return JSONResponse({"error": "admin role required"}, status_code=403)

    @app.exception_handler(Unauthenticated)
    async def unauthenticated(request: Request, exc: Unauthenticated) -> JSONResponse:
        return JSONResponse({"error": "missing or invalid bearer token"}, status_code=401,
                            headers={"WWW-Authenticate": "Bearer"})

    @app.exception_handler(RequestValidationError)
    async def invalid(request: Request, exc: RequestValidationError) -> JSONResponse:
        """Malformed requests are refused and still written to the audit log."""
        who = tokens.identify(request.headers.get("authorization"))
        fields = sorted({".".join(str(p) for p in e.get("loc", [])[1:]) or "body" for e in exc.errors()})
        try:
            kb.store.audit(who.name if who else "unknown", who.role if who else "none", "http",
                           one_line(request.url.path, 40), None, "rejected", detail=f"invalid fields: {', '.join(fields)}")
        except Exception:
            log.exception("could not write audit entry for a rejected request")
        return JSONResponse({"error": "invalid request", "fields": fields}, status_code=422)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/", response_class=HTMLResponse)
    def chat_page() -> HTMLResponse:
        return HTMLResponse(PAGE, headers=PAGE_HEADERS)

    @app.get("/me")
    def me(request: Request) -> dict[str, Any]:
        who = caller(request)
        return {"name": who.name, "role": who.role, "collections": sorted(kb.visible(who)), "admin": who.is_admin}

    @app.post("/ask")
    def ask(body: AskBody, request: Request) -> dict[str, Any]:
        return kb.ask(caller(request), body.question, channel="http").public()

    @app.post("/search")
    def search(body: SearchBody, request: Request) -> dict[str, Any]:
        return {"passages": kb.search(caller(request), body.query, body.k, channel="http")}

    @app.get("/documents")
    def documents(request: Request) -> dict[str, Any]:
        return {"documents": kb.documents(caller(request), channel="http")}

    @app.get("/passages/{passage_id:path}")
    def passage(passage_id: str, request: Request) -> dict[str, Any]:
        return kb.get_passage(caller(request), passage_id, channel="http")

    @app.post("/admin/ingest")
    def ingest(body: IngestBody, request: Request) -> dict[str, Any]:
        who = caller(request)
        if (body.text is None) == (body.content_base64 is None):
            raise BadRequest("send exactly one of 'text' or 'content_base64'")
        if body.text is not None:
            data = body.text.encode("utf-8")
        else:
            try:
                data = base64.b64decode(body.content_base64 or "", validate=True)
            except (binascii.Error, ValueError):
                raise BadRequest("content_base64 is not valid base64") from None
        result = kb.ingest(who, data, body.filename, body.collection, channel="http", replace=body.replace)
        return {"doc_id": result.doc_id, "status": result.status, "passages": result.chunks, "title": result.title}

    @app.delete("/admin/documents/{doc_id:path}")
    def delete(doc_id: str, request: Request) -> dict[str, Any]:
        kb.delete_document(caller(request), doc_id, channel="http")
        return {"doc_id": doc_id, "status": "removed"}

    @app.get("/admin/audit")
    def audit(request: Request, limit: int = 50) -> dict[str, Any]:
        return {"entries": kb.audit_log(caller(request), max(1, min(limit, 1000)), channel="http")}

    if mcp_app is not None:
        app.router.routes.extend(mcp_app.routes)  # POST /mcp
    return app


def _too_large() -> JSONResponse:
    return JSONResponse({"error": "request body too large"}, status_code=413)


def _replay(body: bytes, receive: Any) -> Any:
    """An ASGI receive() that first returns the already-read body, then the real messages."""
    delivered = False

    async def replayed() -> dict:
        nonlocal delivered
        if not delivered:
            delivered = True
            return {"type": "http.request", "body": body, "more_body": False}
        return await receive()

    return replayed


class Gate:
    """ASGI middleware, the outer door:

    * if public hostnames are configured, requests for any other Host are refused (421),
      on every path except /healthz;
    * a request body larger than the limit is refused (413), whether or not it declares its
      length: 64 KB in general, 9 MB for POST /admin/ingest;
    * requests without a known bearer token are refused (401) before they reach a route,
      on every path except the public ones (the static chat page and /healthz).

    Roles are still checked per request by KnowledgeBase.
    """

    def __init__(self, app: Any, tokens: TokenTable, allowed_hosts: list[str] | None = None,
                 public_paths: frozenset[str] = PUBLIC_PATHS):
        self.app, self.tokens, self.public_paths = app, tokens, public_paths
        self.allowed_hosts = {h.lower() for h in allowed_hosts or []}

    async def __call__(self, scope: dict, receive: Any, send: Any) -> None:
        if scope["type"] != "http" or scope["path"] == "/healthz":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in scope.get("headers", [])}
        if self.allowed_hosts:
            host = headers.get("host", "").lower()
            name, _, port = host.rpartition(":") if host.count(":") == 1 else ("", "", "")
            if host not in self.allowed_hosts and not (port.isdigit() and name in self.allowed_hosts):
                log.warning("refused request for host %r on %s", one_line(host, 80), one_line(scope["path"], 80))
                await JSONResponse({"error": "invalid host"}, status_code=421)(scope, receive, send)
                return
        limit = INGEST_BODY_LIMIT if scope["path"] == "/admin/ingest" else BODY_LIMIT
        declared = headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > limit):
            await _too_large()(scope, receive, send)
            return
        if scope["path"] in self.public_paths:
            await self.app(scope, receive, send)
            return
        if self.tokens.identify(headers.get("authorization")) is None:
            log.warning("refused request without a valid token on %s", one_line(scope["path"], 80))
            response = JSONResponse({"error": "missing or invalid bearer token"}, status_code=401,
                                    headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return
        if scope.get("method") in ("POST", "PUT", "PATCH", "DELETE"):
            # Read the body here, up to the limit, so a body without a declared length (chunked)
            # cannot grow without bound; then hand the same bytes to the app.
            body = bytearray()
            while True:
                message = await receive()
                if message["type"] == "http.disconnect":
                    return
                body += message.get("body", b"")
                if len(body) > limit:
                    await _too_large()(scope, receive, send)
                    return
                if not message.get("more_body", False):
                    break
            receive = _replay(bytes(body), receive)
        started = time.monotonic()
        await self.app(scope, receive, send)
        log.debug("%s %s in %d ms", scope.get("method"), one_line(scope["path"], 80), (time.monotonic() - started) * 1000)


def build_app(kb: KnowledgeBase, tokens: TokenTable, host: str = "127.0.0.1",
              allowed_hosts: list[str] | None = None) -> Any:
    """The ASGI app to serve: HTTP API + MCP at /mcp, behind the Gate.

    The MCP endpoint is stateless: every request carries its own token, and there is no
    server-side session another caller could reuse.
    """
    security = None
    if allowed_hosts:
        names = [h.lower() for h in allowed_hosts]
        security = TransportSecuritySettings(
            enable_dns_rebinding_protection=True,
            allowed_hosts=[p for h in names for p in (h, f"{h}:*")],
            allowed_origins=[f"{scheme}://{h}{port}" for h in names for scheme in ("https", "http") for port in ("", ":*")],
        )
    mcp_app = build_mcp(kb, tokens).streamable_http_app(host=host, transport_security=security, stateless_http=True)
    return Gate(build_api(kb, tokens, mcp_app), tokens, allowed_hosts)
