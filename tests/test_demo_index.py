"""The web demo's prebuilt index (demo/): built from the sample documents as they are now, read
without writing anything, and giving the same eval numbers as the product."""
from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest
from conftest import SAMPLES

from kb.config import DEFAULT_ROLES, Identity
from kb.demo_index import (
    CHIPS,
    DEMO_DIR,
    PERSONAS,
    STORY,
    DemoIndex,
    IndexedEmbedder,
    ReadOnlyStore,
    collect_audit,
    sha256,
    shown_questions,
)
from kb.embeddings import EmbeddingUnavailable
from kb.ingest import content_hash, make_doc_id
from kb.service import KnowledgeBase, NotFound, clean_query

ROOT = SAMPLES.parent.parent
MANIFEST = json.loads((DEMO_DIR / "manifest.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def index() -> DemoIndex:
    return DemoIndex(model_folder=None)  # the stored vectors only: what the tests below need


def test_the_index_was_built_from_the_sample_documents_as_they_are_now():
    files = {p.relative_to(SAMPLES).as_posix(): p for p in SAMPLES.rglob("*") if p.is_file()}
    listed = {d["file"]: d for d in MANIFEST["documents"]}
    assert set(listed) == set(files), "a sample document was added or removed: run python scripts/build_demo_index.py"
    for name, path in files.items():
        assert listed[name]["sha256"] == sha256(path), f"{name} changed: run python scripts/build_demo_index.py"
        assert listed[name]["content_hash"] == content_hash(path.read_bytes())
        collection, inner = name.split("/", 1)
        assert listed[name]["doc_id"] == make_doc_id(collection, inner)


def test_the_shipped_files_match_the_manifest():
    for name, digest in MANIFEST["files"].items():
        assert sha256(DEMO_DIR / name) == digest, name
    store = ReadOnlyStore(DEMO_DIR / "kb.db")
    stored = {d["doc_id"]: d for d in store.documents()}
    assert {d["doc_id"]: d["content_hash"] for d in MANIFEST["documents"]} == {k: v["content_hash"] for k, v in stored.items()}
    assert sum(d["chunk_count"] for d in stored.values()) == MANIFEST["passages"]
    key = MANIFEST["model"]["key"]
    assert key.endswith("@" + MANIFEST["model"]["onnx_sha256"][:12])
    assert len(store.vector_ids(key)) == MANIFEST["passages"]  # every passage has its vector, for this model


def test_a_changed_file_is_refused_at_start(tmp_path):
    import shutil

    for name in ("kb.db", "vectors.npz", "manifest.json"):
        shutil.copyfile(DEMO_DIR / name, tmp_path / name)
    with (tmp_path / "vectors.npz").open("ab") as f:
        f.write(b"x")
    with pytest.raises(RuntimeError, match="does not match"):
        DemoIndex(tmp_path, model_folder=None)


def test_the_demo_reproduces_the_eval_baseline(index):
    baseline = json.loads((ROOT / "eval" / "baseline.json").read_text(encoding="utf-8"))
    numbers = index.eval_numbers()  # no model: every eval question must have its stored vector
    for name in ("main", "heldout", "heldout_v2"):
        for key, expected in baseline[name].items():
            assert numbers[name][key] == expected, (name, key)
        assert numbers[name] == MANIFEST["eval"][name]


def test_every_question_the_demo_shows_has_a_stored_vector(index):
    for question in list(STORY.values()) + [c["question"] for c in CHIPS]:
        assert index.embedder.knows(clean_query(question)), question
    assert all(index.embedder.knows(q) for q in shown_questions())


def test_without_the_model_an_unknown_text_is_refused_never_guessed(index):
    with pytest.raises(EmbeddingUnavailable):
        index.embedder.embed(["a question nobody stored"])
    assert index.model_state == "off" and index.load_model() == "off"
    assert not index.can_embed("a question nobody stored") and index.can_embed(clean_query(STORY["leave"]))


def test_the_database_is_read_only_and_audit_entries_stay_in_memory(index):
    before = sha256(DEMO_DIR / "kb.db")
    with pytest.raises(sqlite3.OperationalError, match="readonly|read-only"):
        with index.store._conn() as conn:
            conn.execute("DELETE FROM chunks")
    maria = PERSONAS["maria"].identity
    kb = KnowledgeBase(index.store, index.hybrid)
    with collect_audit() as audit:
        kb.ask(maria, STORY["leave"], channel="demo")
        kb.ask(maria, STORY["salary"], channel="demo")
        with pytest.raises(NotFound):
            kb.get_passage(maria, "hr/salary-bands#2", channel="demo")
    assert [(e["actor"], e["action"], e["outcome"]) for e in audit] == [
        ("maria", "ask", "answered"), ("maria", "ask", "not_found"), ("maria", "get_passage", "denied")]
    kb.ask(maria, STORY["leave"])  # outside a request: nothing is kept anywhere
    assert sha256(DEMO_DIR / "kb.db") == before
    assert not list(DEMO_DIR.glob("kb.db-*"))  # no journal, no write-ahead log
    with sqlite3.connect(DEMO_DIR / "kb.db") as conn:
        assert conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0] == 0
        assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"


def test_the_personas_are_the_products_roles():
    assert {p.id: (p.role, sorted(p.identity.collections)) for p in PERSONAS.values()} == {
        "maria": ("staff", ["general"]), "hana": ("hr", ["general", "hr"]), "farid": ("finance", ["finance", "general"])}
    assert all(isinstance(p.identity, Identity) and p.identity.collections == frozenset(DEFAULT_ROLES[p.role])
               for p in PERSONAS.values())


def test_vectors_the_model_makes_are_kept_for_a_while():
    calls = []

    class Model:
        def embed(self, texts):
            calls.append(list(texts))
            return np.ones((len(texts), 2), dtype=np.float32)

    embedder = IndexedEmbedder("m@x", ["known"], np.zeros((1, 2), dtype=np.float32), model=Model(), cache_size=2)
    embedder.embed(["known", "new", "new"])
    embedder.embed(["new"])
    assert calls == [["new"]]
    embedder.embed(["b", "c"])
    embedder.embed(["new"])  # pushed out by b and c
    assert calls[-1] == ["new"]
