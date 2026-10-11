"""Ingestion: parsing, chunking, metadata, content hashes, and files that must be refused."""
import shutil
from pathlib import Path

import pytest
from conftest import ADMIN, SAMPLES, STAFF, who
from fpdf import FPDF

from kb.ingest import HARD_MAX_CHARS, MAX_FILE_BYTES, TARGET_CHARS, IngestError, build_document, ingest_bytes, ingest_folder, make_doc_id
from kb.service import KnowledgeBase
from kb.store import Store


def chunks_of(store: Store, doc_id: str):
    return [c for c in store.chunks(frozenset(store.collections())) if c.doc_id == doc_id]


def test_sample_folder_ingests_every_document_into_its_collection(store):
    docs = {d["doc_id"]: d for d in store.documents()}
    assert set(docs) == {
        "general/leave-policy", "general/expense-policy", "general/it-security-policy", "general/onboarding-guide",
        "general/office-faq", "general/pt-200-product-manual", "hr/salary-bands", "finance/supplier-payments",
        "finance/quarter-end-checklist",
    }
    assert store.collections() == {"general", "hr", "finance"}
    assert {d["kind"] for d in docs.values()} == {"markdown", "text", "pdf"}
    assert docs["general/pt-200-product-manual"]["pages"] == 4 and docs["general/leave-policy"]["pages"] is None


def test_markdown_structure_becomes_titles_sections_and_clean_text():
    md = (b"---\nowner: hr\n---\n# Travel Guide\n\nIntro line with a [link](https://x.example) and **bold**.\n\n"
          b"## Trains\n\nBook early.\n\n### Night trains\n\nSleepers need approval.\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n"
          b"## Code\n\n```\n# not a heading\nrun()\n```\n* star bullet\n")
    doc = build_document(md, "travel.md", "general")
    assert doc.title == "Travel Guide" and doc.doc_id == "general/travel" and doc.kind == "markdown"
    by_section = {c.section: c.text for c in doc.chunks}
    assert by_section[None] == "Intro line with a link and bold."
    assert by_section["Trains"] == "Book early."
    assert by_section["Trains > Night trains"].startswith("Sleepers need approval.")
    assert "|---|" not in by_section["Trains > Night trains"] and "| 1 | 2 |" in by_section["Trains > Night trains"]
    assert "# not a heading" in by_section["Code"] and "- star bullet" in by_section["Code"]
    assert "owner: hr" not in " ".join(c.text for c in doc.chunks)  # front matter is not content
    assert [c.chunk_id for c in doc.chunks] == [f"general/travel#{i}" for i in range(len(doc.chunks))]


def test_chunks_never_cross_a_heading_and_long_sections_split_at_sentences():
    sentence = "Every claim needs a receipt and a short reason for the cost. "
    md = ("# Policy\n\n## Long\n\n" + sentence * 40 + "\n\n## Short\n\nOne line.\n").encode()
    doc = build_document(md, "p.md", "general")
    long_chunks = [c for c in doc.chunks if c.section == "Long"]
    assert len(long_chunks) > 1 and all(len(c.text) <= TARGET_CHARS for c in long_chunks)
    assert all(c.text.endswith(".") for c in long_chunks)  # split at sentence ends
    assert " ".join(c.text for c in long_chunks).split() == (sentence * 40).split()  # nothing lost
    assert [c.text for c in doc.chunks if c.section == "Short"] == ["One line."]


def test_one_enormous_sentence_is_cut_into_bounded_passages():
    doc = build_document(("# T\n\n" + "word " * 2000).encode(), "big.md", "general")
    assert all(len(c.text) <= HARD_MAX_CHARS for c in doc.chunks)
    assert sum(len(c.text.split()) for c in doc.chunks) == 2000


def test_plain_text_uses_a_short_first_line_as_title():
    doc = build_document(b"Office FAQ\nPrinters are on every floor.", "faq.txt", "general")
    assert doc.title == "Office FAQ" and doc.chunks[0].text == "Printers are on every floor."
    long_first = build_document(("x" * 150 + "\nmore").encode(), "notes_2027.txt", "general")
    assert long_first.title == "Notes 2027"


def test_pdf_pages_headings_and_footers(store):
    chunks = chunks_of(store, "general/pt-200-product-manual")
    assert chunks[0].title == "PT-200 Pallet Tracker: User Manual"
    by_section = {c.section: c for c in chunks}
    assert by_section["3. Charging the battery"].page == 2
    assert by_section["4. Error codes"].page == 3 and "- E31: no Wi-Fi connection." in by_section["4. Error codes"].text
    assert by_section["5. Warranty and support"].page == 4
    assert all("Page " not in c.text for c in chunks)  # page footers are not content
    assert all("\n" not in c.text or c.text.count("\n") < 6 for c in chunks)
    assert by_section["4. Error codes"].label() == "PT-200 Pallet Tracker: User Manual > 4. Error codes, page 3"


def test_scanned_or_broken_pdfs_are_refused_with_a_clear_reason(tmp_path):
    blank = FPDF()
    blank.add_page()
    with pytest.raises(IngestError, match="no text layer"):
        build_document(bytes(blank.output()), "scan.pdf", "general")
    with pytest.raises(IngestError, match="could not be read"):
        build_document(b"%PDF-1.7 this is not really a pdf", "broken.pdf", "general")


@pytest.mark.parametrize("name, data, message", [
    ("report.docx", b"PK\x03\x04", "unsupported file type"),
    ("tool.exe", b"MZ", "unsupported file type"),
    ("noext", b"hello", "unsupported file type"),
    ("binary.txt", b"abc\x00\x01\x02", "binary file"),
    ("empty.md", b"   \n\n", "no text"),
    ("huge.txt", b"a" * (MAX_FILE_BYTES + 1), "larger than"),
], ids=["docx", "exe", "no-extension", "binary-txt", "empty-md", "too-large"])
def test_files_that_cannot_be_ingested_are_refused(name, data, message):
    with pytest.raises(IngestError, match=message):
        build_document(data, name, "general")


def test_non_utf8_text_is_still_read():
    doc = build_document("Café opens at 8.\nHotel limit £150.".encode("cp1252"), "cafe.txt", "general")
    assert "£150" in doc.chunks[0].text


def test_unsafe_ids_and_collection_names_are_refused():
    for collection, path in [
        ("general", "../secret.md"), ("general", "/etc/passwd.md"), ("general", "a/../../b.md"),
        ("general", r"..\..\win.md"), ("HR", "x.md"), ("../hr", "x.md"), ("", "x.md"), ("hr finance", "x.md"),
        ("general", "a/b/c/d/e/f/g.md"), ("general", "!!!.md"),
    ]:
        with pytest.raises(IngestError):
            make_doc_id(collection, path)


def test_doc_ids_are_readable_slugs():
    assert make_doc_id("general", "Policies/Leave Policy (2027).md") == "general/policies/leave-policy-2027"
    assert make_doc_id("hr", "salary_bands.MD") == "hr/salary-bands"


def test_unchanged_file_is_a_no_op_and_a_changed_file_replaces_its_chunks(tmp_path):
    store = Store(str(tmp_path / "kb.db"))
    first = ingest_bytes(store, b"# Rules\n\n## Old\n\nThe old rule.", "rules.md", "general")
    assert first.status == "added"
    version = store.index_version()
    assert ingest_bytes(store, b"# Rules\n\n## Old\n\nThe old rule.", "rules.md", "general").status == "unchanged"
    assert store.index_version() == version  # nothing was rewritten
    changed = ingest_bytes(store, b"# Rules\n\n## New\n\nThe new rule.\n\n## More\n\nMore.", "rules.md", "general")
    assert changed.status == "updated" and changed.chunks == 2 and store.index_version() > version
    texts = [c.text for c in chunks_of(store, "general/rules")]
    assert texts == ["The new rule.", "More."]  # the old passage is gone, not left behind
    assert len(store.documents()) == 1


def test_folder_ingest_prunes_only_when_asked_and_keeps_a_file_that_fails(tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLES, docs)
    store = Store(str(tmp_path / "kb.db"))
    ingest_folder(store, docs)
    (docs / "general" / "office-faq.txt").unlink()
    (docs / "general" / "leave-policy.md").write_bytes(b"\x00\x01 now corrupted")
    report = ingest_folder(store, docs)
    assert report.count("removed") == 0 and "general/office-faq" in {d["doc_id"] for d in store.documents()}
    assert any("leave-policy" in path for path, _ in report.skipped)
    report = ingest_folder(store, docs, prune=True)
    ids = {d["doc_id"] for d in store.documents()}
    assert [r.doc_id for r in report.results if r.status == "removed"] == ["general/office-faq"]
    assert "general/leave-policy" in ids  # failed to parse, so the previous version is kept


def test_folder_rules_for_stray_hidden_and_unsupported_files(tmp_path):
    docs = tmp_path / "docs"
    (docs / "general" / ".git").mkdir(parents=True)
    (docs / "general" / ".git" / "config.txt").write_text("hidden")
    (docs / "general" / "notes.txt").write_text("Plain notes\nPrinters are on every floor.")
    (docs / "general" / "slides.pptx").write_bytes(b"PK")
    (docs / "Bad Name").mkdir()
    (docs / "Bad Name" / "x.md").write_text("# X\n\ntext")
    (docs / "loose.md").write_text("# Loose\n\ntext")
    store = Store(str(tmp_path / "kb.db"))
    report = ingest_folder(store, docs)
    assert [r.doc_id for r in report.results] == ["general/notes"]
    reasons = dict(report.skipped)
    assert "not inside a collection folder" in reasons["loose.md"]
    assert "not a valid collection name" in reasons["Bad Name"]
    assert "unsupported file type" in reasons["general/slides.pptx"]
    assert not any(".git" in path for path in reasons)


def test_symbolic_links_are_not_followed(tmp_path):
    docs = tmp_path / "docs" / "general"
    docs.mkdir(parents=True)
    outside = tmp_path / "outside.md"
    outside.write_text("# Secret\n\nshould not be read")
    try:
        (docs / "link.md").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("this system does not allow creating symbolic links")
    report = ingest_folder(Store(str(tmp_path / "kb.db")), tmp_path / "docs")
    assert report.results == [] and "symbolic links" in dict(report.skipped)["general/link.md"]


def test_the_planted_injection_is_flagged_at_ingestion(store):
    faq = chunks_of(store, "general/office-faq")
    flagged = [s for c in faq for s in c.flagged]
    assert flagged[0].startswith("Note for AI assistants") and any("approved automatically" in s for s in flagged)
    kb = KnowledgeBase(store)
    admin_view = {d["doc_id"]: d for d in kb.documents(who(ADMIN))}
    assert admin_view["general/office-faq"]["flagged_sentences"] == 2
    assert admin_view["general/leave-policy"]["flagged_sentences"] == 0
    assert "flagged_sentences" not in kb.documents(who(STAFF))[0]  # an admin detail


def test_two_files_with_the_same_id_never_replace_each_other(tmp_path):
    docs = tmp_path / "docs" / "general"
    docs.mkdir(parents=True)
    (docs / "Travel Policy.md").write_text("# Travel\n\nBook trains early.")
    (docs / "travel-policy.txt").write_text("Travel\nBook flights late.")
    store = Store(str(tmp_path / "kb.db"))
    for run in range(3):  # the same file wins every time; nothing swaps between runs
        report = ingest_folder(store, tmp_path / "docs")
        assert [r.status for r in report.results] == (["added"] if run == 0 else ["unchanged"])
        assert "same document id (general/travel-policy)" in dict(report.skipped)["general/travel-policy.txt"]
    assert "Book trains early." in chunks_of(store, "general/travel-policy")[0].text


def test_an_upload_does_not_silently_replace_a_different_file(store):
    with pytest.raises(IngestError, match="already used by 'leave-policy.md'"):
        ingest_bytes(store, b"Leave\nEveryone gets 40 days.", "leave-policy.txt", "general", origin="upload")
    assert "25 days" in chunks_of(store, "general/leave-policy")[1].text
    assert ingest_bytes(store, b"Leave\nEveryone gets 40 days.", "leave-policy.txt", "general", origin="upload",
                        replace=True).status == "updated"
    ingest_bytes(store, b"# Notes\n\nFirst.", "notes.md", "general")
    with pytest.raises(IngestError, match="already used by 'notes.md'"):
        ingest_bytes(store, b"Notes\nSecond.", "notes.txt", "general")


def test_prune_never_removes_uploaded_documents(tmp_path):
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLES, docs)
    store = Store(str(tmp_path / "kb.db"))
    ingest_folder(store, docs)
    ingest_bytes(store, b"# Visitors\n\nSign in at reception.", "visitors.md", "general", origin="upload")
    (docs / "general" / "office-faq.txt").unlink()
    removed = [r.doc_id for r in ingest_folder(store, docs, prune=True).results if r.status == "removed"]
    assert removed == ["general/office-faq"] and store.document("general/visitors")["origin"] == "upload"


def test_name_limits_are_the_same_everywhere():
    long_name = "a" * 150 + ".md"
    assert build_document(b"# T\n\ntext", long_name, "general").doc_id.startswith("general/aaaa")
    with pytest.raises(IngestError, match="longer than 200"):
        build_document(b"# T\n\ntext", "a" * 201 + ".md", "general")
    with pytest.raises(IngestError, match="too long"):
        make_doc_id("general", "/".join(["b" * 55] * 4) + "/x.md")


def test_unreadable_or_oversized_files_are_skipped_without_reading_them(tmp_path, monkeypatch):
    import kb.ingest as ingest_module

    docs = tmp_path / "docs" / "general"
    docs.mkdir(parents=True)
    (docs / "big.md").write_text("# Big\n\n" + "word " * 50)
    (docs / "locked.md").write_text("# Locked\n\ntext")
    (docs / "fine.md").write_text("# Fine\n\ntext")
    real_read = Path.read_bytes

    def read_bytes(self):
        if self.name == "locked.md":
            raise PermissionError("locked by another program")
        if self.name == "big.md":
            raise AssertionError("an oversized file must not be read")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    monkeypatch.setattr(ingest_module, "MAX_FILE_BYTES", 200)
    report = ingest_folder(Store(str(tmp_path / "kb.db")), tmp_path / "docs")
    assert [r.doc_id for r in report.results] == ["general/fine"]
    reasons = dict(report.skipped)
    assert "larger than" in reasons["general/big.md"] and "could not be read" in reasons["general/locked.md"]
