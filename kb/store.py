"""SQLite storage: documents, their chunks (passages), passage vectors, and the audit log."""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    collection TEXT NOT NULL,
    title TEXT NOT NULL,
    source_name TEXT NOT NULL,
    kind TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    pages INTEGER,
    chunk_count INTEGER NOT NULL,
    flagged_sentences INTEGER NOT NULL DEFAULT 0,
    ingested_at TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'folder'
);
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
    collection TEXT NOT NULL,
    position INTEGER NOT NULL,
    title TEXT NOT NULL,
    section TEXT,
    page INTEGER,
    text TEXT NOT NULL,
    flagged TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_chunks_collection ON chunks(collection);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
-- One embedding per passage and model (see embeddings.py). Removed with its passage.
CREATE TABLE IF NOT EXISTS vectors (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    vector BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    actor TEXT NOT NULL,
    role TEXT NOT NULL,
    channel TEXT NOT NULL,
    action TEXT NOT NULL,
    query TEXT,
    outcome TEXT NOT NULL,
    mode TEXT,
    passages TEXT NOT NULL DEFAULT '[]',
    detail TEXT,
    duration_ms INTEGER
);
"""

MAX_QUERY_CHARS = 1000


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True)
class Chunk:
    """One passage of one document. The unit that is retrieved, quoted and cited."""

    chunk_id: str
    doc_id: str
    collection: str
    position: int
    title: str
    section: str | None
    page: int | None
    text: str
    flagged: tuple[str, ...] = ()  # sentences that look like instructions to an AI

    def label(self) -> str:
        """Human-readable location, for example 'Leave Policy > Carry-over, page 2'."""
        out = self.title
        if self.section and self.section != self.title:
            out += f" > {self.section}"
        if self.page:
            out += f", page {self.page}"
        return out

    def public(self, text: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "passage_id": self.chunk_id, "doc_id": self.doc_id, "collection": self.collection,
            "title": self.title, "section": self.section, "page": self.page, "label": self.label(),
        }
        if text:
            out["text"] = self.text
        return out


@dataclass
class DocumentRecord:
    doc_id: str
    collection: str
    title: str
    source_name: str
    kind: str
    content_hash: str
    pages: int | None
    chunks: list[Chunk] = field(default_factory=list)
    origin: str = "folder"  # folder: from the documents folder; upload: one file sent by an admin


class Store:
    def __init__(self, db_path: str):
        self._path = str(db_path)
        with self._conn() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(SCHEMA)
            columns = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
            if "origin" not in columns:  # a database made by an earlier version
                conn.execute("ALTER TABLE documents ADD COLUMN origin TEXT NOT NULL DEFAULT 'folder'")
            conn.execute("INSERT OR IGNORE INTO meta (key, value) VALUES ('index_version', '0')")

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self._path, timeout=10, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
        finally:
            conn.close()

    # ---------------------------------------------------------------- documents
    def index_version(self) -> int:
        """Changes whenever any document is added, replaced or removed (cache key for retrievers)."""
        with self._conn() as conn:
            return int(conn.execute("SELECT value FROM meta WHERE key = 'index_version'").fetchone()[0])

    def document_hash(self, doc_id: str) -> tuple[str, str] | None:
        """(collection, content hash) of a stored document, or None."""
        with self._conn() as conn:
            row = conn.execute("SELECT collection, content_hash FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return (row["collection"], row["content_hash"]) if row else None

    def document(self, doc_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM documents WHERE doc_id = ?", (doc_id,)).fetchone()
        return dict(row) if row else None

    def replace_document(self, doc: DocumentRecord, vectors: list[bytes] | None = None, model: str | None = None) -> None:
        """Store a document and its chunks (and, if given, one vector per chunk), replacing any
        older version in one transaction."""
        if vectors is not None and (model is None or len(vectors) != len(doc.chunks)):
            raise ValueError("one vector per chunk, and the model name, are needed")
        flagged = sum(len(c.flagged) for c in doc.chunks)
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc.doc_id,))
                conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc.doc_id,))
                conn.execute(
                    "INSERT INTO documents (doc_id, collection, title, source_name, kind, content_hash, pages, chunk_count, "
                    "flagged_sentences, ingested_at, origin) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (doc.doc_id, doc.collection, doc.title, doc.source_name, doc.kind, doc.content_hash, doc.pages,
                     len(doc.chunks), flagged, now(), doc.origin),
                )
                conn.executemany(
                    "INSERT INTO chunks (chunk_id, doc_id, collection, position, title, section, page, text, flagged) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    [(c.chunk_id, c.doc_id, c.collection, c.position, c.title, c.section, c.page, c.text,
                      json.dumps(list(c.flagged))) for c in doc.chunks],
                )
                if vectors is not None:
                    conn.executemany("INSERT INTO vectors (chunk_id, model, vector) VALUES (?,?,?)",
                                     [(c.chunk_id, model, v) for c, v in zip(doc.chunks, vectors, strict=True)])
                self._bump(conn)
                conn.execute("COMMIT")
            except BaseException:
                conn.execute("ROLLBACK")
                raise

    def delete_document(self, doc_id: str) -> bool:
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            deleted = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,)).rowcount
            if deleted:
                self._bump(conn)
            conn.execute("COMMIT")
        return bool(deleted)

    @staticmethod
    def _bump(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'index_version'")

    def collections(self) -> set[str]:
        with self._conn() as conn:
            return {r[0] for r in conn.execute("SELECT DISTINCT collection FROM documents")}

    def documents(self, collections: frozenset[str] | None = None) -> list[dict]:
        """Document metadata. `collections=None` means all of them (internal use only)."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM documents ORDER BY collection, doc_id").fetchall()
        return [dict(r) for r in rows if collections is None or r["collection"] in collections]

    @staticmethod
    def _chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(row["chunk_id"], row["doc_id"], row["collection"], row["position"], row["title"],
                     row["section"], row["page"], row["text"], tuple(json.loads(row["flagged"])))

    def chunks(self, collections: frozenset[str]) -> list[Chunk]:
        """Every chunk in the given collections, and nothing else. The filter is in the SQL."""
        if not collections:
            return []
        marks = ",".join("?" * len(collections))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT * FROM chunks WHERE collection IN ({marks}) ORDER BY doc_id, position", sorted(collections)
            ).fetchall()
        return [self._chunk(r) for r in rows]

    def chunk(self, chunk_id: str) -> Chunk | None:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return self._chunk(row) if row else None

    # ---------------------------------------------------------------- vectors
    def vector_ids(self, model: str) -> set[str]:
        with self._conn() as conn:
            return {r[0] for r in conn.execute("SELECT chunk_id FROM vectors WHERE model = ?", (model,))}

    def vectors(self, collections: frozenset[str], model: str) -> dict[str, bytes]:
        """Vectors of the chunks in the given collections only. The filter is in the SQL."""
        if not collections:
            return {}
        marks = ",".join("?" * len(collections))
        with self._conn() as conn:
            rows = conn.execute(
                f"SELECT v.chunk_id, v.vector FROM vectors v JOIN chunks c ON c.chunk_id = v.chunk_id "
                f"WHERE c.collection IN ({marks}) AND v.model = ?", (*sorted(collections), model)
            ).fetchall()
        return {r[0]: bytes(r[1]) for r in rows}

    def save_vectors(self, model: str, items: list[tuple[str, bytes]]) -> None:
        """Store vectors for existing chunks (a chunk removed in the meantime is skipped)."""
        with self._conn() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.executemany(
                "INSERT OR REPLACE INTO vectors (chunk_id, model, vector) "
                "SELECT ?, ?, ? WHERE EXISTS (SELECT 1 FROM chunks WHERE chunk_id = ?)",
                [(cid, model, blob, cid) for cid, blob in items],
            )
            conn.execute("COMMIT")

    # ---------------------------------------------------------------- audit log
    def audit(self, actor: str, role: str, channel: str, action: str, query: str | None, outcome: str,
              mode: str | None = None, passages: list[str] | None = None, detail: str | None = None,
              duration_ms: int | None = None) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO audit_log (ts, actor, role, channel, action, query, outcome, mode, passages, detail, duration_ms) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (now(), actor[:80], role[:40], channel[:10], action[:40], (query or "")[:MAX_QUERY_CHARS] or None,
                 outcome[:20], mode, json.dumps(passages or []), (detail or "")[:500] or None, duration_ms),
            )

    def audit_entries(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM audit_log ORDER BY id DESC LIMIT ?", (max(1, min(limit, 1000)),)).fetchall()
        out = []
        for r in rows:
            entry = dict(r)
            entry["passages"] = json.loads(entry["passages"])
            out.append(entry)
        return out
