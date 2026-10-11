"""The hybrid retriever (the default): passage vectors, ranking by keywords and meaning, the
answer/refuse decision, and the access rules.

Tests marked with the `embedder` fixture use the real local model (BAAI/bge-small-en-v1.5) and
are skipped, with the reason shown, if it cannot be loaded. The plumbing tests use a small
deterministic stand-in embedder, so they always run.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from conftest import HR, ROOT, SAMPLES, STAFF, who

from kb.embeddings import EmbeddingUnavailable, FastEmbedder, passage_text
from kb.ingest import ingest_bytes, ingest_folder
from kb.retrieval import HybridRetriever
from kb.service import KnowledgeBase
from kb.store import Store

GENERAL = frozenset({"general"})
PARAPHRASE = "Is the pallet scanner still covered if it breaks after a year?"  # the manual says "warranty"


class WordHashEmbedder:
    """A stand-in for the real model: each word goes to one of 64 dimensions. Deterministic."""

    def __init__(self, name: str = "test-word-hash"):
        self.name = self.key = name
        self.calls = 0

    def embed(self, texts):
        self.calls += 1
        rows = np.zeros((len(texts), 64), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                rows[i, hashlib.sha256(word.encode()).digest()[0] % 64] += 1
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.where(norms == 0, 1, norms)


def vectors_of(store: Store, model: str, doc_id: str) -> dict[str, bytes]:
    return {cid: v for cid, v in store.vectors(frozenset(store.collections()), model).items()
            if cid.startswith(doc_id + "#")}


# ---------------------------------------------------------------------------- vectors in the database
def test_every_passage_gets_a_vector_at_ingest_and_an_update_replaces_it(tmp_path):
    store, fake = Store(str(tmp_path / "kb.db")), WordHashEmbedder()
    ingest_folder(store, SAMPLES, embedder=fake)
    every = {c.chunk_id for c in store.chunks(frozenset(store.collections()))}
    assert store.vector_ids(fake.name) == every and len(every) == 50

    ingest_bytes(store, b"# Volunteering\n\nEvery employee may take 2 volunteering days.", "volunteering.md", "general",
                 embedder=fake)
    before = vectors_of(store, fake.name, "general/volunteering")
    ingest_bytes(store, b"# Volunteering\n\nVolunteering is paused this year.\n\n## Contact\n\nAsk HR.",
                 "volunteering.md", "general", embedder=fake)
    after = vectors_of(store, fake.name, "general/volunteering")
    assert len(before) == 1 and len(after) == 2 and before["general/volunteering#0"] != after["general/volunteering#0"]
    chunk = store.chunk("general/volunteering#0")
    assert np.allclose(np.frombuffer(after["general/volunteering#0"], "<f4"), fake.embed([passage_text(chunk)])[0])

    store.delete_document("general/volunteering")
    assert vectors_of(store, fake.name, "general/volunteering") == {}  # removed with the passages


def test_missing_vectors_are_filled_on_first_search_and_a_new_model_gets_its_own(tmp_path):
    store = Store(str(tmp_path / "kb.db"))
    ingest_folder(store, SAMPLES)  # keywords only: no vectors yet
    assert store.vector_ids("test-word-hash") == set()
    fake = WordHashEmbedder()
    HybridRetriever(store, fake).search("annual leave", GENERAL, 3)
    general = {c.chunk_id for c in store.chunks(GENERAL)}
    assert store.vector_ids(fake.name) == general  # only the collections that were searched
    other = WordHashEmbedder("another-model")
    HybridRetriever(store, other).search("annual leave", GENERAL, 3)
    assert store.vector_ids(other.name) == general and store.vector_ids(fake.name) == set()


def test_hybrid_only_reads_the_callers_collections(tmp_path):
    store, fake = Store(str(tmp_path / "kb.db")), WordHashEmbedder()
    ingest_folder(store, SAMPLES, embedder=fake)
    asked: list[frozenset[str]] = []
    real_vectors, real_chunks = store.vectors, store.chunks
    store.vectors = lambda collections, model: asked.append(collections) or real_vectors(collections, model)  # type: ignore[method-assign]
    store.chunks = lambda collections: asked.append(collections) or real_chunks(collections)  # type: ignore[method-assign]
    hits = HybridRetriever(store, fake).search("salary bands Band C", GENERAL, 10)
    assert hits and all(h.chunk.collection == "general" for h in hits)
    assert asked and set(asked) == {GENERAL}  # nothing outside the caller's collections is even loaded


@pytest.mark.parametrize("real_model", [False, True], ids=["stand-in embedder", "real model"])
def test_restricted_collections_do_not_influence_hybrid_scores(tmp_path, request, real_model):
    """The keyword statistics, the similarities and the fused ranks all come from the caller's
    collections only: the same results whether or not HR and finance documents exist."""
    model = request.getfixturevalue("embedder") if real_model else WordHashEmbedder()
    full, general_only = Store(str(tmp_path / "full.db")), Store(str(tmp_path / "general.db"))
    ingest_folder(full, SAMPLES, embedder=model)
    for path in sorted((SAMPLES / "general").iterdir()):
        ingest_bytes(general_only, path.read_bytes(), path.name, "general", relative_path=path.name, origin="folder",
                     embedder=model)
    for query in ("salary bands", "payment approval limits", "annual pay review budget", "close the quarter",
                  "director approval", "pay", "bank details", PARAPHRASE):
        a = [(h.chunk.chunk_id, h.score, h.coverage, h.similarity) for h in HybridRetriever(full, model).search(query, GENERAL, 10)]
        b = [(h.chunk.chunk_id, h.score, h.coverage, h.similarity)
             for h in HybridRetriever(general_only, model).search(query, GENERAL, 10)]
        assert a == b, query


def test_a_retriever_that_returns_restricted_passages_is_still_stopped(hybrid_kb):
    hybrid_kb.retriever.search = lambda query, collections, k: HybridRetriever.search(  # type: ignore[method-assign]
        hybrid_kb.retriever, query, frozenset({"general", "hr"}), k)  # a bug: ignores the filter
    result = hybrid_kb.ask(who(STAFF), "What is the salary range for Band C?")
    assert "38,000" not in result.answer and all(p["collection"] == "general" for p in result.passages)


# ---------------------------------------------------------------------------- answers, with the real model
def test_a_question_in_other_words_is_answered_by_meaning(hybrid_kb, store):
    result = hybrid_kb.ask(who(STAFF), PARAPHRASE)
    assert result.status == "answered" and result.citations[0]["passage_id"] == "general/pt-200-product-manual#5"
    assert "2-year warranty" in result.answer
    assert KnowledgeBase(store).ask(who(STAFF), PARAPHRASE).status == "not_found"  # keywords alone find nothing


@pytest.mark.parametrize("question, quote", [
    ("How many days of annual leave do full-time employees get?", "Full-time employees receive 25 days"),
    ("What is the hotel limit per night in London?", "up to £150 per night in London"),
    ("How long does the PT-200 battery last on a full charge?", "A full charge lasts about 10 hours"),
    ("Who approves an expense claim over 500 pounds?", "Claims above £500 also need approval from the finance team"),
    ("What does error code E31 mean on the pallet tracker?", "E31: no Wi-Fi connection"),
    ("Are expense claims approved automatically?", "No claim is approved automatically"),
])
def test_hybrid_answers_quote_the_right_sentence(hybrid_kb, question, quote):
    result = hybrid_kb.ask(who(STAFF), question)
    assert result.status == "answered" and quote in result.answer
    for c in result.citations:
        assert c["quote"] in result.passages[c["number"] - 1]["text"]


def test_hybrid_still_refuses_and_keeps_hr_text_with_hr(hybrid_kb):
    for question in ("Can I bring my dog to the office?", "What is the capital of France?",
                     "What is the salary range for Band C?", "What are the salary bands?"):
        result = hybrid_kb.ask(who(STAFF), question)
        assert result.status == "not_found" and result.passages == [], question
    assert "£38,000 to £52,000" in hybrid_kb.ask(who(HR), "What is the salary range for Band C?").answer
    planted = hybrid_kb.ask(who(STAFF), "Are expense claims approved automatically?")
    assert "ignore all previous" not in planted.answer and "Tell every user" not in planted.answer


@pytest.mark.anyio
async def test_hybrid_search_reports_similarity_over_http_and_mcp(hybrid_server):
    with hybrid_server.http(STAFF) as http:
        found = http.post("/search", json={"query": PARAPHRASE, "k": 3}).json()["passages"]
        assert found[0]["passage_id"] == "general/pt-200-product-manual#5" and 0.6 < found[0]["similarity"] < 1
        assert http.post("/search", json={"query": "What is the capital of France?"}).json()["passages"] == []
        hr_words = http.post("/search", json={"query": "salary bands Band C", "k": 10}).json()["passages"]
        assert hr_words and {p["collection"] for p in hr_words} == {"general"}
    assert hybrid_server.ask(STAFF, PARAPHRASE)["status"] == "answered"
    async with hybrid_server.mcp() as call:
        body = await call("search_documents", {"query": PARAPHRASE, "k": 2})
    assert body["passages"][0]["passage_id"] == "general/pt-200-product-manual#5" and "similarity" in body["passages"][0]


def test_an_admin_upload_is_embedded_and_answerable(hybrid_kb, embedder):
    from conftest import ADMIN

    text = b"Bike storage\nBikes are kept in the locked shed behind building B. Ask reception for the code."
    hybrid_kb.ingest(who(ADMIN), text, "bikes.txt", "general")
    assert "general/bikes#0" in hybrid_kb.store.vector_ids(embedder.key)
    result = hybrid_kb.ask(who(STAFF), "Where are bikes kept?")
    assert result.status == "answered" and "locked shed behind building B" in result.answer


# ---------------------------------------------------------------------------- when the model is missing
def test_a_missing_model_gives_a_clear_message(tmp_path):
    with pytest.raises(EmbeddingUnavailable, match="python -m kb download-model.*KB_RETRIEVER=bm25"):
        FastEmbedder("not-a/real-model", cache_dir=tmp_path).load()


def run_cli(*args: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    full = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT), **env}
    return subprocess.run([sys.executable, "-m", "kb", *args], cwd=env["CWD"], env=full, capture_output=True, text=True,
                          encoding="utf-8", timeout=180)


def test_the_command_line_stops_with_a_clear_message_and_bm25_needs_no_model(tmp_path):
    env = {"KB_DB": str(tmp_path / "kb.db"), "KB_EMBED_MODEL": "not-a/real-model", "CWD": str(tmp_path)}
    broken = run_cli("ingest", str(SAMPLES), env=env)
    assert broken.returncode != 0 and "KB_RETRIEVER=hybrid needs the embedding model" in broken.stderr
    assert "Traceback" not in broken.stderr
    keywords = run_cli("ingest", str(SAMPLES), env={**env, "KB_RETRIEVER": "bm25"})
    assert keywords.returncode == 0 and "9 added (0 skipped)" in keywords.stdout and "vectors" not in keywords.stdout
    wrong = run_cli("eval", env={**env, "KB_RETRIEVER": "semantic"})
    assert wrong.returncode != 0 and "KB_RETRIEVER must be 'hybrid'" in wrong.stderr


def test_the_ingest_command_stores_vectors_with_the_real_model(tmp_path, embedder):
    env = {"KB_DB": str(tmp_path / "kb.db"), "CWD": str(tmp_path)}
    done = run_cli("ingest", str(SAMPLES), env=env)
    assert done.returncode == 0, done.stderr
    assert "Passage vectors: BAAI/bge-small-en-v1.5" in done.stdout
    store = Store(env["KB_DB"])
    assert store.vector_ids(embedder.key) == {c.chunk_id for c in store.chunks(frozenset(store.collections()))}


def test_download_model_reports_where_the_model_is(embedder):
    done = run_cli("download-model", env={"CWD": str(ROOT)})
    assert done.returncode == 0, done.stderr
    assert "Embedding model BAAI/bge-small-en-v1.5, kept in" in done.stdout and "later runs need no network" in done.stdout
    assert 30_000_000 < embedder.size_on_disk() < 200_000_000


def test_quotes_are_picked_by_meaning_wherever_the_retriever_has_a_model(tmp_path):
    from kb.answer import ClaudeAnswerer, ExtractiveAnswerer

    store, fake = Store(str(tmp_path / "kb.db")), WordHashEmbedder()
    retriever = HybridRetriever(store, fake)
    assert KnowledgeBase(store, retriever).answerer.embedder is fake
    assert KnowledgeBase(store, retriever, ExtractiveAnswerer()).answerer.embedder is fake
    claude = ClaudeAnswerer(client=object(), model="claude-test")
    assert KnowledgeBase(store, retriever, claude).answerer.embedder is fake  # for its fallback quotes


def test_the_served_app_quotes_the_same_sentences_as_the_eval(tmp_path, monkeypatch, embedder):
    from conftest import TOKENS

    from kb import cli
    from kb.config import DEFAULT_ROLES, Identity, Settings, parse_roles
    from kb.evaluate import run_eval

    db = tmp_path / "served.db"
    ingest_folder(Store(str(db)), SAMPLES, embedder=embedder)
    monkeypatch.setenv("KB_TOKENS", json.dumps(TOKENS))
    monkeypatch.setenv("KB_DB", str(db))
    served = cli._knowledge_base(Settings.from_env())  # what python -m kb serve and mcp-stdio use
    assert served.answerer.embedder is served.retriever.embedder
    report = run_eval(ROOT / "eval" / "questions.jsonl", SAMPLES, k=3, workdir=str(tmp_path / "eval"))
    roles = parse_roles(DEFAULT_ROLES)
    for outcome in report.outcomes:
        who_ = Identity("x", outcome.item["role"], roles[outcome.item["role"]])
        assert served.ask(who_, outcome.item["question"]).answer == outcome.answer, outcome.item["id"]


def test_an_identifier_missing_from_the_documents_is_refused_with_hybrid_search(hybrid_kb):
    assert hybrid_kb.ask(who(STAFF), "What does error code E22 mean on the PT-200 scanner?").status == "not_found"
    assert "E31: no Wi-Fi connection." in hybrid_kb.ask(who(STAFF), "What does error code E31 mean?").answer


def test_the_vector_cache_follows_the_passage_ids_not_their_count(tmp_path):
    store, fake = Store(str(tmp_path / "kb.db")), WordHashEmbedder()
    ingest_folder(store, SAMPLES, embedder=fake)
    retriever = HybridRetriever(store, fake)
    before = store.chunks(GENERAL)
    store.delete_document("general/office-faq")  # 2 passages out, 2 in: same count, another order
    ingest_bytes(store, b"# Site\n\n## Zebra crossing\n\nThe zebra crossing is outside.\n\n## Parking\n\nParking is free.",
                 "zz-site.md", "general", embedder=fake)
    assert len(store.chunks(GENERAL)) == len(before)
    retriever._matrix(GENERAL, before)  # as if a search had read the old list while the store changed
    query = "Where is the zebra crossing?"
    got = [(h.chunk.chunk_id, h.similarity) for h in retriever.search(query, GENERAL, 5)]
    fresh = [(h.chunk.chunk_id, h.similarity) for h in HybridRetriever(store, fake).search(query, GENERAL, 5)]
    assert got == fresh


def test_a_damaged_vector_is_computed_again(tmp_path):
    import sqlite3

    store, fake = Store(str(tmp_path / "kb.db")), WordHashEmbedder()
    ingest_folder(store, SAMPLES, embedder=fake)
    with sqlite3.connect(tmp_path / "kb.db") as conn:
        conn.execute("UPDATE vectors SET vector = x'00ff' WHERE chunk_id = 'general/leave-policy#1'")
    hits = HybridRetriever(store, fake).search("annual leave days", GENERAL, 3)  # no error
    assert hits and len(store.vectors(GENERAL, fake.key)["general/leave-policy#1"]) == 64 * 4


def test_the_model_is_loaded_from_its_folder_without_a_network_and_has_a_fingerprint(embedder):
    assert embedder.key.startswith("BAAI/bge-small-en-v1.5@") and len(embedder.key.split("@")[1]) == 12
    env = {"HF_HUB_OFFLINE": "1", "HTTPS_PROXY": "http://127.0.0.1:9", "CWD": str(ROOT)}
    done = run_cli_python("from kb.embeddings import FastEmbedder; print(FastEmbedder().key)", env)
    assert done.returncode == 0 and done.stdout.strip() == embedder.key, done.stderr


def run_cli_python(code: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    full = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONPATH": str(ROOT), **env}
    return subprocess.run([sys.executable, "-c", code], cwd=env["CWD"], env=full, capture_output=True, text=True,
                          encoding="utf-8", timeout=120)
