"""Finding the passages that match a question.

The contract every retriever must keep: it is given the set of collections the caller may
read, and it only ever *looks at* chunks from those collections. Filtering happens before
ranking, so text the caller may not read never takes part in scoring, is never returned,
and never reaches the answer step. (The service checks the results again anyway.)

HybridRetriever is the default (KB_RETRIEVER=hybrid). It ranks every passage the caller may
read twice, by keywords (BM25) and by meaning (cosine similarity of local embeddings, see
embeddings.py), and merges the two rankings with reciprocal rank fusion. Each hit carries
both signals, so the service can decide to answer mainly on the similarity of the best
passage, with keyword coverage as a second signal.

BM25Retriever (KB_RETRIEVER=bm25) is keyword ranking only: no model, nothing to download.

Neither has a side channel. BM25 statistics (document frequencies) are computed over the
caller's own collections only, a passage's similarity depends only on the passage and the
question, and the fused ranks are taken among the caller's passages. The scores a user sees
are the same whether or not restricted documents exist (tested).
"""
from __future__ import annotations

import math
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .embeddings import DEFAULT_EMBED_MODEL, Embedder, from_blob, passage_text, shared_embedder, to_blob
from .store import Chunk, Store
from .text import tokenize

MAX_QUERY_TERMS = 64
UNSEEN_CAP: float | None = None  # see _Index.weights; set after measuring (README, "Retrieval")
RETRIEVERS = ("hybrid", "bm25")
RRF_K = 60               # reciprocal rank fusion constant (the usual value)
SEARCH_FLOOR = 0.5       # hybrid search lists a passage with no question word only above this similarity


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float     # retriever score (BM25, or the fused rank score); only comparable within one search
    coverage: float  # 0..1: how much of the question's words (IDF-weighted) this passage contains
    # How much each question word matters (IDF over the caller's collections). Optional: an
    # answerer falls back to its own weights when a retriever does not provide them.
    term_weights: dict[str, float] = field(default_factory=dict, compare=False, hash=False)
    similarity: float | None = None  # cosine similarity of passage and question (hybrid only)


class Retriever(Protocol):
    name: str

    def search(self, query: str, collections: frozenset[str], k: int) -> list[Hit]:
        """Best `k` chunks for `query`, considering ONLY chunks in `collections`."""
        ...


def query_terms(query: str) -> list[str]:
    terms: list[str] = []
    for t in tokenize(query):
        if t not in terms:
            terms.append(t)
    return terms[:MAX_QUERY_TERMS]


class _Index:
    """BM25 statistics over one fixed set of chunks."""

    def __init__(self, chunks: list[Chunk], k1: float = 1.2, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # The title and section heading are indexed with the text: "Carry-over" matters.
        self.tf = [Counter(tokenize(f"{c.title}\n{c.section or ''}\n{c.text}")) for c in chunks]
        self.lengths = [sum(tf.values()) for tf in self.tf]
        self.avg_length = (sum(self.lengths) / len(self.lengths)) if self.lengths else 1.0
        self.df: Counter[str] = Counter()
        for tf in self.tf:
            self.df.update(tf.keys())
        self.n = len(chunks)

    def idf(self, term: str) -> float:
        """Inverse document frequency. A term found nowhere gets the highest weight, so a
        question about something the documents never mention scores a low coverage."""
        df = self.df.get(term, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def weights(self, terms: list[str], unseen_cap: float | None) -> dict[str, float]:
        """How much each question word counts. A word that appears in none of the caller's
        passages would get the highest weight of all; with unseen_cap it counts at most
        `unseen_cap` times the weight of the most specific word that does appear."""
        weights = {t: self.idf(t) for t in terms}
        seen = [w for t, w in weights.items() if self.df.get(t)]
        if unseen_cap is not None and seen:
            limit = unseen_cap * max(seen)
            weights = {t: (w if self.df.get(t) else min(w, limit)) for t, w in weights.items()}
        return weights

    def search(self, terms: list[str], k: int, unseen_cap: float | None = None) -> list[Hit]:
        if not terms or not self.n:
            return []
        weights = self.weights(terms, unseen_cap)
        total_weight = sum(weights.values()) or 1.0
        hits = []
        for chunk, tf, length in zip(self.chunks, self.tf, self.lengths, strict=True):
            score, matched = 0.0, 0.0
            for t in terms:
                f = tf.get(t, 0)
                if not f:
                    continue
                matched += weights[t]
                score += weights[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_length))
            if score > 0:
                hits.append(Hit(chunk, round(score, 4), round(matched / total_weight, 4), weights))
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[:k]


class BM25Retriever:
    name = "bm25"

    def __init__(self, store: Store, unseen_cap: float | None = UNSEEN_CAP):
        self._store = store
        self.unseen_cap = unseen_cap
        self._cache: dict[frozenset[str], tuple[int, _Index]] = {}
        self._lock = threading.Lock()

    def _index(self, collections: frozenset[str]) -> _Index:
        version = self._store.index_version()
        with self._lock:
            cached = self._cache.get(collections)
            if cached and cached[0] == version:
                return cached[1]
        index = _Index(self._store.chunks(collections))  # the SQL filter: only these collections
        with self._lock:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[collections] = (version, index)
        return index

    def search(self, query: str, collections: frozenset[str], k: int) -> list[Hit]:
        if not collections:
            return []
        return self._index(collections).search(query_terms(query), k, self.unseen_cap)


class HybridRetriever:
    """BM25 and embedding similarity, merged by reciprocal rank fusion (RRF).

    Both rankings are made over the same list of passages: the caller's collections only.
    """

    name = "hybrid"

    def __init__(self, store: Store, embedder: Embedder, bm25: BM25Retriever | None = None, rrf_k: int = RRF_K,
                 search_floor: float = SEARCH_FLOOR):
        self._store = store
        self.embedder = embedder
        self.bm25 = bm25 or BM25Retriever(store)
        self.rrf_k, self.search_floor = rrf_k, search_floor
        # collections -> (the passage ids in matrix order, the matrix). Checked against the ids of
        # the passages being ranked, so a matrix can never be used for a different list of passages.
        self._cache: dict[frozenset[str], tuple[tuple[str, ...], np.ndarray]] = {}
        self._dim = 0
        self._lock = threading.Lock()

    def _vector_size(self) -> int:
        if not self._dim:
            self._dim = int(self.embedder.embed(["size check"]).shape[1])
        return self._dim

    def _matrix(self, collections: frozenset[str], chunks: list[Chunk]) -> np.ndarray:
        """One vector per chunk, in the order of `chunks`. Missing or damaged vectors are computed
        again and stored."""
        ids = tuple(c.chunk_id for c in chunks)
        with self._lock:
            cached = self._cache.get(collections)
            if cached and cached[0] == ids:
                return cached[1]
        key, size = self.embedder.key, self._vector_size() * 4
        stored = {cid: blob for cid, blob in self._store.vectors(collections, key).items()  # only these collections
                  if len(blob) == size}  # a blob of the wrong length is treated as missing
        missing = [c for c in chunks if c.chunk_id not in stored]
        if missing:
            fresh = self.embedder.embed([passage_text(c) for c in missing])
            blobs = [(c.chunk_id, to_blob(v)) for c, v in zip(missing, fresh, strict=True)]
            self._store.save_vectors(key, blobs)
            stored.update(blobs)
        matrix = np.stack([from_blob(stored[c.chunk_id]) for c in chunks]) if chunks else np.zeros((0, 0), np.float32)
        with self._lock:
            if len(self._cache) > 32:
                self._cache.clear()
            self._cache[collections] = (ids, matrix)
        return matrix

    def warm(self, collections: frozenset[str]) -> None:
        """Load the model and make sure every passage has a vector (at server start)."""
        load = getattr(self.embedder, "load", None)
        if load is not None:
            load()
        self._matrix(collections, self.bm25._index(collections).chunks)

    def search(self, query: str, collections: frozenset[str], k: int) -> list[Hit]:
        if not collections:
            return []
        index = self.bm25._index(collections)
        if not index.n:
            return []
        chunks = index.chunks
        terms = query_terms(query)
        similarity = self._matrix(collections, chunks) @ self.embedder.embed([query])[0]
        keyword = {h.chunk.chunk_id: h for h in index.search(terms, index.n, self.bm25.unseen_cap)}
        weights = index.weights(terms, self.bm25.unseen_cap)
        by_meaning = sorted(range(len(chunks)), key=lambda i: (-float(similarity[i]), chunks[i].chunk_id))
        fused: dict[int, float] = {}
        for rank, i in enumerate(by_meaning, start=1):
            fused[i] = 1 / (self.rrf_k + rank)
        position = {c.chunk_id: i for i, c in enumerate(chunks)}
        for rank, hit in enumerate(keyword.values(), start=1):  # keyword hits are already in BM25 order
            fused[position[hit.chunk.chunk_id]] += 1 / (self.rrf_k + rank)
        hits = []
        for i, score in fused.items():
            chunk, sim = chunks[i], float(similarity[i])
            matched = keyword.get(chunk.chunk_id)
            if matched is None and sim < self.search_floor:
                continue  # neither a question word nor close in meaning
            hits.append(Hit(chunk, round(score, 6), matched.coverage if matched else 0.0, weights, round(sim, 4)))
        hits.sort(key=lambda h: (-h.score, h.chunk.chunk_id))
        return hits[:k]


def build_retriever(store: Store, kind: str = "hybrid", embed_model: str = DEFAULT_EMBED_MODEL,
                    model_dir: str | None = None) -> BM25Retriever | HybridRetriever:
    """The retriever named by KB_RETRIEVER. The hybrid one shares one loaded model per process."""
    if kind == "bm25":
        return BM25Retriever(store)
    if kind == "hybrid":
        return HybridRetriever(store, shared_embedder(embed_model, model_dir))
    raise ValueError(f"unknown retriever {kind!r}; use one of: {', '.join(RETRIEVERS)}")
