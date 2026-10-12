"""The web demo's prebuilt index: the sample documents, their vectors, and the questions it shows.

Built once by scripts/build_demo_index.py into demo/:

    demo/kb.db           the sample documents, ingested by this product (passages and a vector each)
    demo/vectors.npz     vectors for every question the demo shows (story, chips, the three eval
                         sets) and for every sentence the quoting step can compare, so those
                         questions need no model at run time
    demo/manifest.json   what was built: the SHA-256 of every sample document, the content hash
                         this product stored for it, the SHA-256 of kb.db and vectors.npz, the
                         model, and the eval numbers the demo reproduces
    demo/model/          the ONNX embedding model (onnx_embed.py), for questions people type

At run time the database is opened read-only (SQLite's immutable mode): nothing is written to
disk, not even the audit log. Each request collects its own audit entries instead
(collect_audit) and the page shows them.

A question that is not in vectors.npz is embedded by the ONNX model. Without the model (the
KB_DEMO_SEARCH=keywords setting, or a server without the model files) such a question is
searched by keywords only, and the page says so.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import sqlite3
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
from urllib.parse import quote

import numpy as np

from .answer import sentence_text
from .config import DEFAULT_ROLES, Identity
from .embeddings import DEFAULT_EMBED_MODEL, ROOT, SHIPPED_MODEL_FILES, SHIPPED_MODEL_SHA256, EmbeddingUnavailable
from .evaluate import evaluate_items, load_questions
from .onnx_embed import DEMO_MODEL_DIR, OnnxEmbedder
from .retrieval import BM25Retriever, HybridRetriever
from .service import KnowledgeBase, clean_query
from .store import Store, now
from .text import split_sentences

DEMO_DIR = ROOT / "demo"
SAMPLES_DIR = ROOT / "samples" / "docs"
EVAL_SETS = {"main": ROOT / "eval" / "questions.jsonl", "heldout": ROOT / "eval" / "heldout.jsonl",
             "heldout_v2": ROOT / "eval" / "heldout_v2.jsonl"}
MODEL_SOURCE = {"repository": "Qdrant/bge-small-en-v1.5-onnx-Q", "revision": "aa8f8b060edb00e03bfdd08813a2949946c8ba55",
                "onnx_sha256": SHIPPED_MODEL_SHA256, "licence": "MIT"}
MODEL_FILES = SHIPPED_MODEL_FILES
DIMENSION = 384
QUESTION_TOKENS = 128   # the longest typed question the model reads, in tokens (about 90 words)


@dataclass(frozen=True)
class Persona:
    id: str
    name: str
    team: str
    role: str

    @property
    def identity(self) -> Identity:
        return Identity(self.id, self.role, frozenset(DEFAULT_ROLES[self.role]), key=f"demo-{self.id}")

    def public(self) -> dict[str, Any]:
        return {"id": self.id, "name": self.name, "team": self.team, "role": self.role,
                "collections": sorted(DEFAULT_ROLES[self.role])}


PERSONAS = {p.id: p for p in (Persona("maria", "Maria", "Sales", "staff"), Persona("hana", "Hana", "HR", "hr"),
                              Persona("farid", "Farid", "Finance", "finance"))}

# The questions of the six step story (the page sends them by key).
STORY = {
    "leave": "How many days of annual leave do full-time employees get?",
    "scanner": "Is the pallet scanner still covered if it breaks after a year?",
    "salary": "What is the salary range for Band C?",
    "sick": "How many paid sick days am I entitled to per year?",
}
# Ready-made questions for free use: everyday questions first, then the ones that probe a rule
# ("restricted": the answer is in a document this persona may not open).
CHIPS: list[dict[str, Any]] = [
    {"label": "Hotel limit in London", "question": "What is the hotel limit per night in London?", "persona": "maria",
     "group": "everyday"},
    {"label": "Report a lost laptop", "question": "How quickly must I report a lost laptop?", "persona": "maria",
     "group": "everyday"},
    {"label": "Parental leave for partners", "question": "How long is parental leave for partners?", "persona": "maria",
     "group": "everyday"},
    {"label": "Scanner battery life", "question": "How long does the PT-200 battery last on a full charge?", "persona": "maria",
     "group": "everyday"},
    {"label": "Salary bands, as Hana", "question": "What is the salary range for Band C?", "persona": "hana", "group": "everyday"},
    {"label": "Payment runs, as Farid", "question": "When do supplier payment runs take place?", "persona": "farid",
     "group": "everyday"},
    {"label": "A document with a hidden instruction", "question": "Where is lost property kept?", "persona": "maria",
     "group": "break"},
    {"label": "Does it obey the hidden instruction?", "question": "Are expense claims approved automatically?", "persona": "maria",
     "group": "break"},
    {"label": "Something it can't answer", "question": "Can I bring my dog to the office?", "persona": "maria", "group": "break"},
    {"label": "An error code it doesn't know", "question": "What does error code E22 mean on the PT-200 scanner?",
     "persona": "maria", "group": "break"},
    {"label": "Payment runs, as Maria", "question": "When do supplier payment runs take place?", "persona": "maria",
     "group": "break", "restricted": True},
]
# How many questions the web demo's leak tests ask (tests/test_demo_web.py: every heading, sentence
# and word of the HR and finance documents, asked by each persona that may not read them, plus
# the first 120 again with keyword search only). The page quotes it; a test keeps it true.
LEAK_CHECKS = 606


# ---------------------------------------------------------------------------- read-only storage
_AUDIT: contextvars.ContextVar[list[dict[str, Any]] | None] = contextvars.ContextVar("demo_audit", default=None)


@contextmanager
def collect_audit() -> Iterator[list[dict[str, Any]]]:
    """Audit entries written while this block runs, in order. Nothing goes to disk."""
    entries: list[dict[str, Any]] = []
    token = _AUDIT.set(entries)
    try:
        yield entries
    finally:
        _AUDIT.reset(token)


class ReadOnlyStore(Store):
    """The shipped database, opened read-only and immutable: any write fails, and SQLite creates
    no journal or WAL file next to it (the folder may itself be read-only, as on a serverless
    host). The audit log goes to the current request instead (collect_audit)."""

    def __init__(self, db_path: str | Path):  # no schema set-up: that would write
        self._path = str(Path(db_path).resolve())

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        uri = f"file:{quote(Path(self._path).as_posix(), safe='/:')}?mode=ro&immutable=1"
        conn = sqlite3.connect(uri, uri=True, timeout=10, isolation_level=None, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def audit(self, actor: str, role: str, channel: str, action: str, query: str | None, outcome: str,
              mode: str | None = None, passages: list[str] | None = None, detail: str | None = None,
              duration_ms: int | None = None) -> None:
        entries = _AUDIT.get()
        if entries is not None:
            entries.append({"ts": now(), "actor": actor, "role": role, "channel": channel, "action": action,
                            "query": query, "outcome": outcome, "mode": mode, "passages": list(passages or []),
                            "detail": detail, "duration_ms": duration_ms})


# ---------------------------------------------------------------------------- vectors
class IndexedEmbedder:
    """Vectors computed when the demo was built, looked up by their exact text. A text that is
    not in the table goes to `model` (the ONNX model); without a model it raises
    EmbeddingUnavailable, so a missing vector can never pass unnoticed."""

    name = DEFAULT_EMBED_MODEL

    def __init__(self, key: str, texts: Sequence[str], vectors: np.ndarray, model: OnnxEmbedder | None = None,
                 cache_size: int = 2048):
        if len(texts) != len(vectors):
            raise ValueError("one vector per text is needed")
        self._key = key
        self._table = {t: np.asarray(v, dtype=np.float32) for t, v in zip(texts, vectors, strict=True)}
        self.model = model
        self._fresh: OrderedDict[str, np.ndarray] = OrderedDict()  # vectors the model made, most recent last
        self._cache_size = cache_size
        self._lock = threading.Lock()

    @property
    def key(self) -> str:
        return self._key

    def knows(self, text: str) -> bool:
        return text in self._table

    def __len__(self) -> int:
        return len(self._table)

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        have: dict[str, np.ndarray] = {}
        with self._lock:  # one look at the cache: another thread may push entries out right after
            for t in dict.fromkeys(texts):
                if t in self._table:
                    have[t] = self._table[t]
                elif t in self._fresh:
                    self._fresh.move_to_end(t)
                    have[t] = self._fresh[t]
        missing = [t for t in dict.fromkeys(texts) if t not in have]
        if missing:
            model = self.model
            if model is None:
                raise EmbeddingUnavailable(f"no stored vector for {len(missing)} text(s) and no model to make one")
            made = model.embed(missing)
            have.update(zip(missing, made, strict=True))
            with self._lock:
                for text in missing:
                    self._fresh[text] = have[text]
                while len(self._fresh) > self._cache_size:
                    self._fresh.popitem(last=False)
        return np.stack([have[t] for t in texts]).astype(np.float32)


# ---------------------------------------------------------------------------- what is precomputed
def shown_questions() -> list[str]:
    """Every question the demo itself sends or measures, as the service sees it (cleaned)."""
    questions = list(STORY.values()) + [c["question"] for c in CHIPS]
    for path in EVAL_SETS.values():
        questions += [item["question"] for item in load_questions(path)]
    return list(dict.fromkeys(clean_query(q) for q in questions))


def sentence_texts(store: Store) -> list[str]:
    """Every text the quoting step may embed for a candidate sentence (answer.sentence_text)."""
    texts = []
    for chunk in store.chunks(frozenset(store.collections())):
        texts += [sentence_text(chunk, s) for s in split_sentences(chunk.text)]
    return list(dict.fromkeys(texts))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_vectors(path: Path) -> tuple[list[str], np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        texts = [str(t) for t in data["texts"]]
        vectors = np.asarray(data["vectors"], dtype=np.float32)
    if vectors.shape != (len(texts), DIMENSION):
        raise ValueError(f"{path.name}: expected {len(texts)} vectors of {DIMENSION} numbers, found {vectors.shape}")
    return texts, vectors


# ---------------------------------------------------------------------------- the index
class DemoIndex:
    """The demo's documents and vectors, loaded once per server process."""

    def __init__(self, folder: Path = DEMO_DIR, model_folder: Path | None = DEMO_MODEL_DIR, verify: bool = True):
        self.folder = Path(folder)
        self.manifest: dict[str, Any] = json.loads((self.folder / "manifest.json").read_text(encoding="utf-8"))
        if verify:
            for name, expected in self.manifest["files"].items():
                if sha256(self.folder / name) != expected:
                    raise RuntimeError(f"demo/{name} does not match demo/manifest.json; run scripts/build_demo_index.py")
        self.store = ReadOnlyStore(self.folder / "kb.db")
        texts, vectors = load_vectors(self.folder / "vectors.npz")
        self.key: str = self.manifest["model"]["key"]
        model = OnnxEmbedder(model_folder, max_tokens=QUESTION_TOKENS) if model_folder is not None else None
        self.model_state = "off" if model is None else ("waiting" if model.available() else "missing")
        self.embedder = IndexedEmbedder(self.key, texts, vectors, model if self.model_state == "waiting" else None)
        self._model = model
        self._model_lock = threading.Lock()
        self.hybrid = HybridRetriever(self.store, self.embedder)
        self.hybrid._dim = DIMENSION  # known from the manifest: no probe text to embed
        self.keywords = BM25Retriever(self.store)

    def load_model(self) -> str:
        """Load the ONNX model if this index has one: ready, off (not wanted), missing (no files),
        mismatch (other model files than the stored vectors), or broken (it would not load)."""
        with self._model_lock:
            if self.model_state != "waiting" or self._model is None:
                return self.model_state
            try:
                self._model.load()
            except EmbeddingUnavailable:
                self.model_state, self.embedder.model = "broken", None
                return self.model_state
            if self._model.key != self.key:
                self.model_state, self.embedder.model = "mismatch", None
            else:
                self.model_state = "ready"
            return self.model_state

    def can_embed(self, text: str) -> bool:
        """Can this text be searched by meaning? (Stored, or the model is there to embed it.)"""
        return self.embedder.knows(text) or self.load_model() == "ready"

    def eval_numbers(self) -> dict[str, dict[str, Any]]:
        """The three question sets, run through the demo's own index with quoted answers."""
        kb = KnowledgeBase(self.store, self.hybrid)
        return {name: evaluate_items(kb, load_questions(path)).summary() for name, path in EVAL_SETS.items()}
