"""Documents in, chunks out.

Supported: Markdown (.md, .markdown), plain text (.txt) and PDFs with a text layer (.pdf).
Scanned PDFs (images only) are refused with a clear message; OCR is not included.

Each chunk keeps its document id, title, section heading, page number (PDFs) and position.
Chunks follow the document's own structure: a chunk never crosses a heading or a PDF page,
and long sections are split at paragraph, line or sentence boundaries (about 700 characters).

Ingesting is idempotent. The content hash of each file is stored; an unchanged file is a
no-op, and a changed file replaces all of its old chunks in one transaction. With an
embedder (the hybrid retriever, the default), each chunk's vector is computed here and
stored in the same transaction, so an updated document never keeps old vectors.

Folder layout:  <root>/<collection>/<any sub folders>/<file>
The first folder level is the collection (for example general, hr, finance); it decides who
may read the document.
"""
from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

from .config import NAME
from .embeddings import Embedder, fill_missing_vectors, passage_text, to_blob
from .store import Chunk, DocumentRecord, Store
from .text import clean_document_text, instruction_sentences, split_sentences

PARSER_VERSION = "2"  # part of the content hash: a change in parsing re-ingests every file
KINDS = {".md": "markdown", ".markdown": "markdown", ".txt": "text", ".pdf": "pdf"}
MAX_FILE_BYTES = 5 * 1024 * 1024
TARGET_CHARS = 700
HARD_MAX_CHARS = 1500
MAX_CHUNKS = 2000
MAX_DOC_ID = 200      # characters; passage ids add '#' and a number
MAX_FILENAME = 200


class IngestError(ValueError):
    """The file cannot be ingested. The message says why, in plain words."""


@dataclass
class Section:
    heading: str | None
    page: int | None
    text: str


@dataclass
class IngestResult:
    doc_id: str
    status: str  # added | updated | unchanged | removed
    chunks: int = 0
    title: str = ""
    kind: str = ""


@dataclass
class IngestReport:
    results: list[IngestResult] = field(default_factory=list)
    skipped: list[tuple[str, str]] = field(default_factory=list)  # (path, reason)
    vectors_filled: int = 0  # vectors computed for passages that had none (unchanged documents)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


# ---------------------------------------------------------------------------- ids
def _slug(part: str) -> str:
    out = re.sub(r"[^a-z0-9]+", "-", part.lower()).strip("-")[:60].strip("-")
    if not out:
        raise IngestError(f"cannot make a document id from {part!r}")
    return out


def make_doc_id(collection: str, relative_path: str) -> str:
    """'general' + 'policies/Leave Policy.md' -> 'general/policies/leave-policy'."""
    if not NAME.fullmatch(collection):
        raise IngestError("collection names use lower-case letters, digits, '-' and '_' (max 31 characters)")
    path = PurePosixPath(relative_path.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise IngestError("the file path must be relative and must not contain '..'")
    parts = list(path.parts[:-1]) + [path.stem]
    if len(parts) > 6:
        raise IngestError("the file is nested too deeply (max 5 sub folders)")
    doc_id = collection + "/" + "/".join(_slug(p) for p in parts)
    if len(doc_id) > MAX_DOC_ID:
        raise IngestError(f"the folder and file names are too long (the document id would be over {MAX_DOC_ID} characters)")
    return doc_id


def content_hash(data: bytes) -> str:
    return hashlib.sha256(PARSER_VERSION.encode() + b"\0" + data).hexdigest()


# ---------------------------------------------------------------------------- parsing
def _decode(data: bytes) -> str:
    if b"\x00" in data[:4096]:
        raise IngestError("this looks like a binary file, not text")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


_MD_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")


def _clean_markdown_line(line: str) -> str:
    line = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", line)  # images -> alt text
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)  # links -> link text
    line = re.sub(r"<[^>\n]{1,200}>", "", line)  # inline HTML tags
    line = line.replace("**", "").replace("__", "").replace("`", "")
    line = re.sub(r"^\s*>\s?", "", line)  # block quotes
    line = re.sub(r"^(\s*)[*+]\s+", r"\1- ", line)  # bullets -> "- "
    if re.fullmatch(r"\s*(?:[-*_]\s*){3,}", line) or re.fullmatch(r"\s*\|?[\s:|-]*-{3,}[\s:|-]*\|?\s*", line):
        return ""  # horizontal rules and table separator rows
    return line.rstrip()


def parse_markdown(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    text = clean_document_text(text)
    if text.startswith("---\n"):  # front matter
        end = text.find("\n---", 4)
        if end != -1:
            text = text[end + 4 :]
    title: str | None = None
    sections: list[Section] = []
    h2: str | None = None
    current: str | None = None
    buf: list[str] = []
    in_code = False

    def flush() -> None:
        body = "\n".join(buf).strip()
        if body:
            sections.append(Section(current, None, body))
        buf.clear()

    for line in text.split("\n"):
        if line.strip().startswith("```"):
            in_code = not in_code
            continue
        m = None if in_code else _MD_HEADING.match(line)
        if m:
            level, heading = len(m.group(1)), _clean_markdown_line(m.group(2)).strip()
            flush()
            if level == 1 and title is None:
                title, current, h2 = heading, None, None
            elif level <= 2:
                h2 = current = heading
            else:
                current = f"{h2} > {heading}" if h2 else heading
            continue
        buf.append(line.rstrip() if in_code else _clean_markdown_line(line))
    flush()
    return (title or fallback_title), sections


def parse_text(text: str, fallback_title: str) -> tuple[str, list[Section]]:
    """Plain text has no headings. A short first line is used as the title."""
    text = clean_document_text(text).strip()
    first, _, rest = text.partition("\n")
    if first.strip() and len(first.strip()) <= 100 and rest.strip():
        return first.strip(), [Section(None, None, rest.strip())]
    return fallback_title, ([Section(None, None, text)] if text else [])


_PDF_HEADING = re.compile(r"\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z][^.!?:;]{1,70}")
_PDF_NEW_LINE = re.compile(r"^(?:[-*•]\s|[A-Z]\d{1,3}:|\d{1,2}\)\s)")
_PDF_PAGE_NUMBER = re.compile(r"(?:page\s+)?\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?", re.IGNORECASE)  # footers


def parse_pdf(data: bytes, fallback_title: str) -> tuple[str, list[Section], int]:
    """Text of each page; numbered lines such as '3. Charging' are read as headings."""
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError

    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted and not reader.decrypt(""):
            raise IngestError("the PDF is password protected")
        pages = [clean_document_text(p.extract_text() or "") for p in reader.pages]
        meta_title = (reader.metadata.title if reader.metadata else None) or ""
    except IngestError:
        raise
    except (PdfReadError, ValueError, KeyError, TypeError, OSError) as exc:
        raise IngestError(f"the PDF could not be read ({type(exc).__name__})") from None
    if not any(p.strip() for p in pages):
        raise IngestError("the PDF has no text layer (a scanned document?). OCR is not included")

    title = clean_document_text(str(meta_title)).strip() or ""
    sections: list[Section] = []
    current: str | None = None

    def flush(lines: list[str], heading: str | None, page: int) -> None:
        body = "".join(lines).strip()
        if body:
            sections.append(Section(heading, page, body))
        lines.clear()

    for number, page_text in enumerate(pages, start=1):
        lines: list[str] = []
        for raw in page_text.split("\n"):
            line = re.sub(r"\s+", " ", raw).strip()
            if not line or _PDF_PAGE_NUMBER.fullmatch(line):
                continue
            if not title:
                title = line[:100]
                continue
            if number == 1 and not sections and not lines and line == title:
                continue  # the title printed on page 1
            if _PDF_HEADING.fullmatch(line):
                flush(lines, current, number)
                current = line
                continue
            if lines and (_PDF_NEW_LINE.match(line) or re.search(r"[.!?:]$", lines[-1])):
                lines.append("\n" + line)
            elif lines:
                lines.append(" " + line)
            else:
                lines.append(line)
        flush(lines, current, number)
    return (title or fallback_title), sections, len(pages)


# ---------------------------------------------------------------------------- chunking
def _units(paragraph: str) -> list[str]:
    """Lines of a paragraph; a line that is too long is split into sentences."""
    out: list[str] = []
    for line in paragraph.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= TARGET_CHARS:
            out.append(line)
            continue
        for sentence in split_sentences(line):
            while len(sentence) > HARD_MAX_CHARS:  # one enormous "sentence": cut at a space
                cut = sentence.rfind(" ", 0, HARD_MAX_CHARS)
                cut = cut if cut > 0 else HARD_MAX_CHARS
                out.append(sentence[:cut].strip())
                sentence = sentence[cut:].strip()
            if sentence:
                out.append(sentence)
    return out


def chunk_sections(sections: list[Section], doc_id: str, collection: str, title: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section in sections:
        # Text addressed to an AI is flagged per paragraph, before the section is cut into passages.
        flagged_here = set(instruction_sentences(section.text))
        pieces: list[str] = []
        current = ""
        for paragraph in re.split(r"\n\s*\n", section.text):
            for unit in _units(paragraph):
                if current and len(current) + 1 + len(unit) > TARGET_CHARS:
                    pieces.append(current)
                    current = unit
                else:
                    current = f"{current}\n{unit}" if current else unit
        if current:
            pieces.append(current)
        for text in pieces:
            flagged = tuple(s for s in split_sentences(text) if s in flagged_here)
            position = len(chunks)
            chunks.append(Chunk(f"{doc_id}#{position}", doc_id, collection, position, title,
                                section.heading, section.page, text, flagged))
            if len(chunks) > MAX_CHUNKS:
                raise IngestError(f"the document is too long (more than {MAX_CHUNKS} passages)")
    return chunks


# ---------------------------------------------------------------------------- ingest
def build_document(data: bytes, filename: str, collection: str, relative_path: str | None = None,
                   origin: str = "upload") -> DocumentRecord:
    name = Path(filename).name
    if len(name) > MAX_FILENAME:
        raise IngestError(f"the file name is longer than {MAX_FILENAME} characters")
    suffix = Path(name).suffix.lower()
    kind = KINDS.get(suffix)
    if kind is None:
        raise IngestError(f"unsupported file type '{suffix or name}' (supported: .md, .txt, .pdf)")
    if len(data) > MAX_FILE_BYTES:
        raise IngestError(f"the file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
    doc_id = make_doc_id(collection, relative_path or name)
    fallback_title = Path(name).stem.replace("-", " ").replace("_", " ").strip().capitalize() or doc_id
    pages = None
    if kind == "pdf":
        title, sections, pages = parse_pdf(data, fallback_title)
    elif kind == "markdown":
        title, sections = parse_markdown(_decode(data), fallback_title)
    else:
        title, sections = parse_text(_decode(data), fallback_title)
    title = re.sub(r"\s+", " ", title).strip()[:200]
    chunks = chunk_sections(sections, doc_id, collection, title)
    if not chunks:
        raise IngestError("the document has no text")
    return DocumentRecord(doc_id, collection, title, relative_path or name, kind, content_hash(data), pages, chunks,
                          origin)


def ingest_bytes(store: Store, data: bytes, filename: str, collection: str, relative_path: str | None = None,
                 origin: str = "upload", replace: bool = False, embedder: Embedder | None = None) -> IngestResult:
    """Add or update one document. An unchanged file is a no-op.

    A document is only updated from the same source: the same file name and the same origin
    (the documents folder, or an upload). If a different file maps to the same document id
    (manual.pdf and manual.md, or an upload over a folder document), it is refused unless
    replace=True, so one file can never silently overwrite another.
    """
    name = Path(filename).name
    if len(name) > MAX_FILENAME:
        raise IngestError(f"the file name is longer than {MAX_FILENAME} characters")
    doc_id = make_doc_id(collection, relative_path or name)
    source = relative_path or name
    existing = store.document(doc_id)
    same_source = existing is not None and existing["source_name"] == source and existing["origin"] == origin
    if existing is not None and not same_source and not replace:
        where = "the documents folder" if existing["origin"] == "folder" else "an upload"
        raise IngestError(f"the document id {doc_id} is already used by '{existing['source_name']}' (from {where}); "
                          "rename this file, or replace that document explicitly")
    if same_source and existing is not None and existing["content_hash"] == content_hash(data):
        return IngestResult(doc_id, "unchanged")
    doc = build_document(data, name, collection, relative_path, origin)
    if embedder is not None:
        vectors = embedder.embed([passage_text(c) for c in doc.chunks])
        store.replace_document(doc, [to_blob(v) for v in vectors], embedder.key)
    else:
        store.replace_document(doc)
    return IngestResult(doc.doc_id, "updated" if existing else "added", len(doc.chunks), doc.title, doc.kind)


def ingest_folder(store: Store, root: str | Path, prune: bool = False, embedder: Embedder | None = None) -> IngestReport:
    """Ingest <root>/<collection>/... . With prune=True, documents that came from this folder
    and whose file is gone are removed (uploaded documents are never pruned). A file that
    fails to read or parse keeps its previous version. With an embedder, missing vectors of
    unchanged documents are filled in too."""
    root = Path(root)
    if not root.is_dir():
        raise IngestError(f"folder not found: {root}")
    report = IngestReport()
    seen: dict[str, str] = {}  # doc id -> the file that claimed it in this run
    scanned: set[str] = set()
    for folder in sorted(root.iterdir()):
        if folder.name.startswith("."):
            continue
        if not folder.is_dir() or folder.is_symlink():
            if folder.is_file():
                report.skipped.append((folder.name, "not inside a collection folder"))
            continue
        collection = folder.name
        if not NAME.fullmatch(collection):
            report.skipped.append((folder.name, "folder name is not a valid collection name"))
            continue
        scanned.add(collection)
        for path in sorted(folder.rglob("*")):
            relative = path.relative_to(folder).as_posix()
            if any(part.startswith(".") for part in path.relative_to(folder).parts):
                continue
            if path.is_symlink():
                report.skipped.append((f"{collection}/{relative}", "symbolic links are not followed"))
                continue
            if not path.is_file():
                continue
            if path.suffix.lower() not in KINDS:
                report.skipped.append((f"{collection}/{relative}", f"unsupported file type '{path.suffix}'"))
                continue
            label = f"{collection}/{relative}"
            try:
                doc_id = make_doc_id(collection, relative)
                if doc_id in seen:
                    report.skipped.append((label, f"same document id ({doc_id}) as {collection}/{seen[doc_id]}; "
                                                  "rename one of the two files"))
                    continue
                seen[doc_id] = relative
                if path.stat().st_size > MAX_FILE_BYTES:
                    raise IngestError(f"the file is larger than {MAX_FILE_BYTES // (1024 * 1024)} MB")
                existing = store.document(doc_id)
                # The folder is the source of truth for documents that came from it (a renamed or
                # converted file replaces its old version); never for uploaded documents.
                replace = existing is not None and existing["origin"] == "folder"
                report.results.append(ingest_bytes(store, path.read_bytes(), path.name, collection, relative,
                                                   origin="folder", replace=replace, embedder=embedder))
            except IngestError as exc:
                report.skipped.append((label, str(exc)))
            except OSError as exc:
                report.skipped.append((label, f"the file could not be read ({type(exc).__name__})"))
    if prune:
        for doc in store.documents():
            if doc["origin"] == "folder" and doc["collection"] in scanned and doc["doc_id"] not in seen:
                store.delete_document(doc["doc_id"])
                report.results.append(IngestResult(doc["doc_id"], "removed", 0, doc["title"], doc["kind"]))
    if embedder is not None:
        report.vectors_filled = fill_missing_vectors(store, embedder)
    return report
