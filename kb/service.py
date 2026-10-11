"""KnowledgeBase: the one place where access rules, thresholds and the audit log are applied.

The HTTP API, the MCP tools, the demo and the eval all go through this class, so the rules
are the same everywhere:

1. The caller's collections are worked out from their token's role, and retrieval only looks
   at those collections. Results are checked again afterwards, so even a faulty retriever
   cannot pass a restricted passage on.
2. If no passage clears the thresholds, the answer is "not in the documents" and no model is
   asked to guess. With the hybrid retriever the test is mainly how close in meaning the
   passage is to the question; with keywords only, how many of the question's words it has.
3. When answers are written by a paid model (Claude mode), each person may get a limited
   number per minute and everyone together a limited number per day. Above the limit the
   answer is quoted from the documents instead (no model call), with a warning.
4. Every question, search, passage lookup and admin action is written to the audit log: who,
   what, which passages were used, the outcome and the time taken.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .answer import Answer, Answerer, AnswerUnavailable, ExtractiveAnswerer
from .config import Identity
from .embeddings import Embedder
from .ingest import MAX_DOC_ID, MAX_FILENAME, IngestError, IngestResult, ingest_bytes
from .retrieval import BM25Retriever, Hit, Retriever
from .store import MAX_QUERY_CHARS, Store
from .text import identifiers, one_line, words_with_digits

log = logging.getLogger("kb.service")

NOT_FOUND_TEXT = "I could not find this in the documents you have access to."
UNAVAILABLE_TEXT = "The answer service is not available right now. The matching passages are listed below."
INJECTION_WARNING = ("A document used here contains text addressed to AI assistants. "
                     "It was treated as document content, not as an instruction.")
PASSAGE_ID = re.compile(rf"[a-z0-9][a-z0-9/_-]{{0,{MAX_DOC_ID - 1}}}#\d{{1,5}}")

# When a passage may be used for an answer. See README, "Retrieval". Chosen on the 64 demo
# questions only (eval/questions.jsonl), never on the held-out sets.
# Hybrid retriever: the passage is close enough in meaning to the question, or a little less
# close but it contains nearly all of the question's words.
DEFAULT_MIN_SIMILARITY = 0.72            # cosine similarity of passage and question
DEFAULT_MIN_SIMILARITY_WITH_WORDS = 0.65
DEFAULT_MIN_COVERAGE_WITH_SIMILARITY = 0.9
# Keywords only (KB_RETRIEVER=bm25): both of these.
DEFAULT_MIN_COVERAGE = 0.6   # share of the question's words found in the passage, rare words counting more
DEFAULT_MIN_SCORE = 1.0      # BM25 score


class BadRequest(ValueError):
    """The caller sent something we will not process. The message is safe to show."""


class NotFound(LookupError):
    pass


class Forbidden(Exception):
    """The caller is known but their role may not do this."""


@dataclass
class AskResult:
    status: str   # answered | not_found | flagged | error
    answer: str
    mode: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    passages: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    draft: str | None = None  # a model answer that failed the source check: never sent to the caller
    parts: list[dict[str, Any]] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {"status": self.status, "answer": self.answer, "mode": self.mode, "citations": self.citations,
                "passages": self.passages, "warnings": self.warnings, "parts": self.parts}


def clean_query(value: Any, what: str = "question") -> str:
    if not isinstance(value, str):
        raise BadRequest(f"the {what} must be text")
    if len(value) > MAX_QUERY_CHARS:
        raise BadRequest(f"the {what} is longer than {MAX_QUERY_CHARS} characters")
    text = one_line(value, MAX_QUERY_CHARS)
    if not text:
        raise BadRequest(f"the {what} is empty")
    return text


class ModelBudget:
    """How many model-written answers may be produced: per person per minute, and per day for
    everyone together (the day runs in UTC). In memory: it resets when the server restarts."""

    def __init__(self, per_minute: int = 10, per_day: int = 500, per_person_day: int = 100):
        self.per_minute, self.per_day, self.per_person_day = per_minute, per_day, per_person_day
        self._recent: dict[str, deque[float]] = {}
        self._day, self._used_today = "", 0
        self._person_today: dict[str, int] = {}
        self._lock = threading.Lock()

    def take(self, who: str) -> str | None:
        """Use one answer from the budget for `who` (one key per token). Returns None if allowed,
        else the reason it is not."""
        now = time.monotonic()
        today = datetime.now(timezone.utc).date().isoformat()
        with self._lock:
            if today != self._day:
                self._day, self._used_today, self._person_today = today, 0, {}
            recent = self._recent.setdefault(who, deque())
            while recent and now - recent[0] > 60:
                recent.popleft()
            if self._used_today >= self.per_day:
                return f"the daily limit of {self.per_day} AI-written answers has been reached"
            if self._person_today.get(who, 0) >= self.per_person_day:
                return f"the limit of {self.per_person_day} AI-written answers per person per day has been reached"
            if len(recent) >= self.per_minute:
                return f"the limit of {self.per_minute} AI-written answers per person per minute has been reached"
            recent.append(now)
            self._used_today += 1
            self._person_today[who] = self._person_today.get(who, 0) + 1
            return None


class KnowledgeBase:
    def __init__(self, store: Store, retriever: Retriever | None = None, answerer: Answerer | None = None,
                 k: int = 5, max_passages: int = 3, min_coverage: float = DEFAULT_MIN_COVERAGE,
                 min_score: float = DEFAULT_MIN_SCORE, budget: ModelBudget | None = None,
                 min_similarity: float = DEFAULT_MIN_SIMILARITY,
                 min_similarity_with_words: float = DEFAULT_MIN_SIMILARITY_WITH_WORDS,
                 min_coverage_with_similarity: float = DEFAULT_MIN_COVERAGE_WITH_SIMILARITY):
        """Without a retriever this uses keywords only (no model); the server and the command
        line build the one named by KB_RETRIEVER (retrieval.build_retriever)."""
        self.store = store
        self.retriever = retriever or BM25Retriever(store)
        self.embedder: Embedder | None = getattr(self.retriever, "embedder", None)
        self.answerer = answerer or ExtractiveAnswerer(embedder=self.embedder)
        if self.embedder is not None and getattr(self.answerer, "embedder", False) is None:
            # Quotes are picked the same way everywhere (served app, eval, demo): by meaning when
            # the retriever has an embedding model, also inside Claude mode's fallback.
            self.answerer.embedder = self.embedder  # type: ignore[attr-defined]  # both answerers have it
        self.k, self.max_passages = k, max_passages
        self.min_coverage, self.min_score = min_coverage, min_score
        self.min_similarity, self.min_similarity_with_words = min_similarity, min_similarity_with_words
        self.min_coverage_with_similarity = min_coverage_with_similarity
        self.budget = budget or ModelBudget()

    # ---------------------------------------------------------------- helpers
    def visible(self, who: Identity) -> frozenset[str]:
        return who.visible(self.store.collections())

    def _audit(self, who: Identity | None, channel: str, action: str, query: str | None, outcome: str,
               started: float, passages: list[str] | None = None, detail: str | None = None, mode: str | None = None) -> None:
        try:
            self.store.audit(who.name if who else "unknown", who.role if who else "none", channel, action, query, outcome,
                             mode, passages, detail, int((time.monotonic() - started) * 1000))
        except Exception:  # the audit log must never break a request, but a failure must be visible
            log.exception("could not write audit entry for %s", action)

    def _retrieve(self, query: str, visible: frozenset[str], k: int) -> tuple[list[Hit], str | None]:
        hits = self.retriever.search(query, visible, k)
        allowed = [h for h in hits if h.chunk.collection in visible]
        if len(allowed) != len(hits):  # a retriever bug; never pass restricted text on
            log.error("retriever %s returned %d passages outside the caller's collections; dropped",
                      getattr(self.retriever, "name", "?"), len(hits) - len(allowed))
            return allowed, "retriever returned restricted passages; dropped"
        return allowed, None

    def select_passages(self, question: str, visible: frozenset[str]) -> tuple[list[Hit], str | None]:
        """The passages an answer may use: retrieved from `visible` only, and clearing both thresholds."""
        hits, detail = self._retrieve(question, visible, self.k)
        good = [h for h in hits if self.usable(h)]
        wanted = identifiers(question)
        if wanted and good:
            # An error code or model number the question names must be in the passage (its text,
            # title or section): "What does E22 mean?" must not get the passage about E12.
            found = {h.chunk.chunk_id: words_with_digits(f"{h.chunk.title} {h.chunk.section or ''} {h.chunk.text}")
                     for h in good}
            missing = wanted - set().union(*found.values())
            good = [h for h in good if wanted <= found[h.chunk.chunk_id]]
            if not good:
                note = (f"{', '.join(sorted(missing))} is in none of the matching passages" if missing
                        else f"no matching passage has all of {', '.join(sorted(wanted))}")
                detail = "; ".join(filter(None, [detail, note]))
        return good[: self.max_passages], detail

    def usable(self, hit: Hit) -> bool:
        """May this passage be used for an answer?"""
        if hit.similarity is None:  # keywords only
            return hit.coverage >= self.min_coverage and hit.score >= self.min_score
        return hit.similarity >= self.min_similarity or (
            hit.similarity >= self.min_similarity_with_words and hit.coverage >= self.min_coverage_with_similarity)

    def _require_admin(self, who: Identity, channel: str, action: str, query: str | None, started: float) -> None:
        if not who.is_admin:
            self._audit(who, channel, action, query, "denied", started, detail=f"role {who.role} is not admin")
            raise Forbidden("admin role required")

    # ---------------------------------------------------------------- questions
    def ask(self, who: Identity, question: Any, channel: str = "http") -> AskResult:
        started = time.monotonic()
        outcome, notes, used, mode, text = "error", [], [], self.answerer.mode, None
        try:
            text = clean_query(question)
            good, retrieval_note = self.select_passages(text, self.visible(who))
            notes += [retrieval_note] if retrieval_note else []
            used = [h.chunk.chunk_id for h in good]
            if not good:
                result = AskResult("not_found", NOT_FOUND_TEXT, mode)
            else:
                answerer, warnings = self.answerer, []
                if any(h.chunk.flagged for h in good):
                    warnings.append(INJECTION_WARNING)
                if answerer.mode == "claude":
                    over = self.budget.take(who.key or f"{who.role}/{who.name}")  # one budget per token
                    if over:  # no model call: quote the documents instead
                        answerer, mode = ExtractiveAnswerer(embedder=self.embedder), "extractive"
                        warnings.append(f"This answer is quoted from the documents because {over}.")
                        notes.append(f"model budget: {over}")
                passages = [h.chunk.public() for h in good]
                try:
                    answer: Answer = answerer.answer(text, good)
                except AnswerUnavailable as exc:
                    notes.append(f"answer service failed: {exc}")
                    result = AskResult("error", UNAVAILABLE_TEXT, mode, passages=passages, warnings=warnings)
                else:
                    if answer.served_by:
                        notes.append(f"answered by {answer.served_by} (fallback)")
                    if answer.status == "not_found":
                        result = AskResult("not_found", NOT_FOUND_TEXT, mode)
                    else:
                        result = AskResult(answer.status, answer.text, mode, [c.public() for c in answer.citations],
                                           passages, warnings + answer.warnings, answer.draft, answer.parts)
                        if answer.status == "flagged":  # the withheld text is kept for the admin, in the audit log
                            problems = "; ".join(answer.warnings[1:]) or "flagged"
                            notes.append(f"{problems[:280]} | withheld: {one_line(answer.draft or '', 200)}")
            outcome = result.status
            return result
        except BadRequest as exc:
            outcome = "rejected"
            notes.append(str(exc))
            raise
        finally:
            self._audit(who, channel, "ask", text if text is not None else one_line(str(question)[:200]), outcome,
                        started, used, "; ".join(notes) or None, mode)

    def search(self, who: Identity, query: Any, k: int = 5, channel: str = "http") -> list[dict[str, Any]]:
        started = time.monotonic()
        outcome, detail, used, text = "error", None, [], None
        try:
            text = clean_query(query, "query")
            if not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= 20:
                raise BadRequest("k must be a whole number from 1 to 20")
            hits, detail = self._retrieve(text, self.visible(who), k)
            used = [h.chunk.chunk_id for h in hits]
            outcome = "ok" if hits else "not_found"
            return [{**h.chunk.public(), "score": h.score, "coverage": h.coverage}
                    | ({"similarity": h.similarity} if h.similarity is not None else {}) for h in hits]
        except BadRequest as exc:
            outcome, detail = "rejected", str(exc)
            raise
        finally:
            self._audit(who, channel, "search", text if text is not None else one_line(str(query)[:200]), outcome,
                        started, used, detail)

    def get_passage(self, who: Identity, passage_id: Any, channel: str = "http") -> dict[str, Any]:
        """A passage by id. A passage the caller may not read is reported exactly like a
        missing one, so its existence is not revealed (the audit log records the difference)."""
        started = time.monotonic()
        pid = passage_id if isinstance(passage_id, str) else ""
        if not PASSAGE_ID.fullmatch(pid):
            self._audit(who, channel, "get_passage", one_line(str(passage_id)[:200]), "rejected", started,
                        detail="malformed passage id")
            raise BadRequest("passage ids look like 'general/leave-policy#2'")
        chunk = self.store.chunk(pid)
        if chunk is None or chunk.collection not in self.visible(who):
            outcome = "not_found" if chunk is None else "denied"
            self._audit(who, channel, "get_passage", pid, outcome, started,
                        detail=None if chunk is None else f"collection {chunk.collection} not visible to role {who.role}")
            raise NotFound("No passage with that id.")
        self._audit(who, channel, "get_passage", pid, "ok", started, [pid])
        return chunk.public()

    def documents(self, who: Identity, channel: str = "http") -> list[dict[str, Any]]:
        started = time.monotonic()
        docs = self.store.documents(self.visible(who))
        keys = ["doc_id", "collection", "title", "kind", "pages", "chunk_count", "ingested_at"]
        if who.is_admin:
            keys += ["source_name", "origin", "flagged_sentences"]
        out = [{k: d[k] for k in keys} for d in docs]
        self._audit(who, channel, "documents", None, "ok", started, detail=f"{len(out)} documents")
        return out

    # ---------------------------------------------------------------- admin (the admin role reads everything)
    def ingest(self, who: Identity, data: bytes, filename: str, collection: str, channel: str = "http",
               replace: bool = False) -> IngestResult:
        started = time.monotonic()
        name = one_line(str(filename).replace("\\", "/").rsplit("/", 1)[-1], MAX_FILENAME + 10)
        self._require_admin(who, channel, "ingest", f"{collection}/{name}"[:300], started)
        try:
            if len(name) > MAX_FILENAME:
                raise IngestError(f"the file name is longer than {MAX_FILENAME} characters")
            result = ingest_bytes(self.store, data, name, collection, origin="upload", replace=replace,
                                  embedder=self.embedder)
        except IngestError as exc:
            self._audit(who, channel, "ingest", f"{collection}/{name}"[:300], "rejected", started, detail=str(exc))
            raise BadRequest(str(exc)) from None
        self._audit(who, channel, "ingest", f"{collection}/{name}", result.status, started,
                    detail=f"{result.doc_id}, {result.chunks} passages" + (", replace requested" if replace else ""))
        return result

    def delete_document(self, who: Identity, doc_id: str, channel: str = "http") -> None:
        started = time.monotonic()
        self._require_admin(who, channel, "delete", doc_id[:200], started)
        if not self.store.delete_document(doc_id):
            self._audit(who, channel, "delete", doc_id[:200], "not_found", started)
            raise NotFound("No document with that id.")
        self._audit(who, channel, "delete", doc_id, "ok", started)

    def audit_log(self, who: Identity, limit: int = 50, channel: str = "http") -> list[dict[str, Any]]:
        started = time.monotonic()
        self._require_admin(who, channel, "audit.read", None, started)
        entries = self.store.audit_entries(limit)
        self._audit(who, channel, "audit.read", None, "ok", started, detail=f"limit {limit}")
        return entries
