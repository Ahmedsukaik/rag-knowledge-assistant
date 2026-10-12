"""Retrieval: ranking, thresholds, the collection filter, and plugging in another retriever."""
import logging

import pytest
from conftest import SAMPLES, STAFF, audit, who

from kb.ingest import ingest_bytes, ingest_folder
from kb.retrieval import BM25Retriever, Hit
from kb.service import BadRequest, KnowledgeBase
from kb.store import Store

GENERAL = frozenset({"general"})


@pytest.mark.parametrize("query, passage", [
    ("How many days of annual leave do full-time employees get?", "general/leave-policy#1"),
    ("hotel limit per night in London", "general/expense-policy#2"),
    ("error code E31", "general/pt-200-product-manual#4"),
    ("multi-factor authentication text message codes", "general/it-security-policy#2"),
])
def test_bm25_puts_the_right_passage_first(store, query, passage):
    hits = BM25Retriever(store).search(query, GENERAL, 3)
    assert hits[0].chunk.chunk_id == passage and hits[0].coverage >= 0.6
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)


def test_questions_about_things_the_documents_never_mention_have_low_coverage(store):
    retriever = BM25Retriever(store)
    assert retriever.search("What is the capital of France?", GENERAL, 3) == []
    dog = retriever.search("Can I bring my dog to the office?", GENERAL, 3)
    assert dog and all(h.coverage < 0.6 for h in dog)  # "office" matches; "dog" is nowhere


def test_only_stopwords_or_no_collections_find_nothing(store):
    retriever = BM25Retriever(store)
    assert retriever.search("what is it and how do I do that?", GENERAL, 5) == []
    assert retriever.search("annual leave", frozenset(), 5) == []
    assert retriever.search("annual leave", frozenset({"no-such-collection"}), 5) == []


def test_restricted_collections_do_not_even_influence_the_scores(tmp_path, store):
    """Filtering happens before ranking: the statistics come from the caller's collections only,
    so a staff member gets exactly the same scores whether or not HR and finance documents exist."""
    general_only = Store(str(tmp_path / "general-only.db"))
    for path in sorted((SAMPLES / "general").iterdir()):
        ingest_bytes(general_only, path.read_bytes(), path.name, "general")
    for query in ("salary bands", "payment approval limits", "annual pay review budget", "close the quarter",
                  "director approval", "pay", "bank details"):
        full = [(h.chunk.chunk_id, h.score, h.coverage) for h in BM25Retriever(store).search(query, GENERAL, 10)]
        alone = [(h.chunk.chunk_id, h.score, h.coverage) for h in BM25Retriever(general_only).search(query, GENERAL, 10)]
        assert full == alone, query


def test_the_index_follows_new_changed_and_removed_documents(store):
    retriever = BM25Retriever(store)
    assert retriever.search("volunteering", GENERAL, 3) == []
    ingest_bytes(store, b"# Volunteering\n\nEvery employee may take 2 volunteering days.", "volunteering.md", "general")
    assert retriever.search("volunteering days", GENERAL, 3)[0].chunk.doc_id == "general/volunteering"
    ingest_bytes(store, b"# Volunteering\n\nVolunteering is paused this year.", "volunteering.md", "general")
    assert "paused" in retriever.search("volunteering", GENERAL, 3)[0].chunk.text
    store.delete_document("general/volunteering")
    assert retriever.search("volunteering", GENERAL, 3) == []


class LeakyRetriever:
    """A broken retriever that ignores the collection filter."""

    name = "leaky"

    def __init__(self, store: Store):
        self.store = store

    def search(self, query, collections, k):
        everything = self.store.chunks(frozenset(self.store.collections()))
        return [Hit(c, 10.0, 1.0) for c in everything if c.collection == "hr"][:k]


def test_results_are_checked_again_so_a_faulty_retriever_cannot_leak(store, caplog):
    kb = KnowledgeBase(store, retriever=LeakyRetriever(store))
    with caplog.at_level(logging.ERROR, logger="kb.service"):
        result = kb.ask(who(STAFF), "What is the salary range for Band C?")
        found = kb.search(who(STAFF), "salary bands")
    assert result.status == "not_found" and result.passages == [] and found == []
    assert "outside the caller's collections" in caplog.text
    entry = store.audit_entries(2)[1]
    assert entry["action"] == "ask" and "restricted passages; dropped" in entry["detail"]


class FixedRetriever:
    """Stands in for an embeddings retriever: any object with the same search() contract works."""

    name = "fixed"

    def __init__(self, store: Store):
        self.store = store
        self.calls: list[tuple[str, frozenset[str], int]] = []

    def search(self, query, collections, k):
        self.calls.append((query, collections, k))
        chunk = self.store.chunk("general/leave-policy#2")
        return [Hit(chunk, 0.9, 1.0)] if chunk.collection in collections else []


def test_another_retriever_can_be_plugged_in(store):
    retriever = FixedRetriever(store)
    kb = KnowledgeBase(store, retriever=retriever, min_score=0.5)
    result = kb.ask(who(STAFF), "Can I carry over unused days?")
    assert result.status == "answered" and result.passages[0]["passage_id"] == "general/leave-policy#2"
    assert retriever.calls == [("Can I carry over unused days?", frozenset({"general"}), 5)]


def test_bad_search_requests_are_refused_and_audited(kb):
    bad = [("annual leave", 0), ("annual leave", 21), ("annual leave", True), ("annual leave", "5"), ("", 5),
           ("   \n ", 5), (None, 5), ("x" * 1001, 5)]
    for query, k in bad:
        with pytest.raises(BadRequest):
            kb.search(who(STAFF), query, k)
    assert audit(kb) == [("maria (sales)", "http", "search", "rejected")] * len(bad)


def test_long_questions_with_many_words_still_work(kb):
    question = "annual leave " + " ".join(f"word{i}" for i in range(100))
    assert len(question) <= 1000
    assert kb.search(who(STAFF), question, 3)  # extra words cost coverage, but search still runs
    assert kb.ask(who(STAFF), question).status == "not_found"  # mostly unknown words: refused, not guessed


def test_known_limit_shared_ordinary_words_can_match_the_wrong_passage(kb):
    """Documented in the README ("Retrieval, and its limits"). Keyword search cannot tell that
    volunteering days are not annual leave days. The source shown makes the mismatch visible.
    If retrieval improves and this fails, update the README."""
    result = kb.ask(who(STAFF), "How many paid volunteering days per year?")
    assert result.status == "answered" and result.passages[0]["passage_id"] == "general/leave-policy#1"
    assert "annual leave" in result.answer


def test_ingest_folder_is_also_safe_to_run_twice_in_a_row(tmp_path):
    store = Store(str(tmp_path / "kb.db"))
    ingest_folder(store, SAMPLES)
    before = [(c.chunk_id, c.text) for c in store.chunks(frozenset(store.collections()))]
    assert ingest_folder(store, SAMPLES).count("unchanged") == 9
    assert [(c.chunk_id, c.text) for c in store.chunks(frozenset(store.collections()))] == before
