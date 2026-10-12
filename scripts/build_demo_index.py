"""Build the web demo's index: demo/kb.db, demo/vectors.npz, demo/manifest.json and demo/model/.

    python scripts/build_demo_index.py

Needs the full development install (requirements-dev.txt) and the embedding model in the
models folder (python -m kb download-model). Run it again after changing a sample document,
the ingestion or search code, or the demo's questions (kb/demo_index.py): the tests compare
demo/manifest.json with samples/docs and fail until it is rebuilt.

What it does:
1. Ingests samples/docs with this product's own ingestion and the fastembed model, exactly as
   `python -m kb ingest` does, into a fresh database. The ingestion time is set to a fixed value
   and the database is stored without a write-ahead log, so the server can open it read-only.
2. Computes, with the same model, a vector for every question the demo shows (the story, the
   chips and the three eval sets) and for every sentence the quoting step may compare.
3. Copies the model files the server needs (ONNX file and tokenizer, 67 MB) into demo/model,
   after checking the ONNX file's SHA-256 against the published one.
4. Runs the three eval sets through the finished index, without the model, and stops if the
   numbers differ from eval/baseline.json.
5. Writes demo/manifest.json with the SHA-256 of every input and output.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import tempfile
from contextlib import closing
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.demo_index import (  # noqa: E402
    DEMO_DIR,
    MODEL_FILES,
    MODEL_SOURCE,
    SAMPLES_DIR,
    DemoIndex,
    sentence_texts,
    sha256,
    shown_questions,
)
from kb.embeddings import EmbeddingUnavailable, shared_embedder  # noqa: E402
from kb.ingest import content_hash, ingest_folder, make_doc_id  # noqa: E402
from kb.store import Store  # noqa: E402

FIXED_TIME = "2026-10-12T00:00:00+00:00"


def build(out: Path = DEMO_DIR, copy_model: bool = True) -> dict:
    embedder = shared_embedder()
    try:
        embedder.load()
    except EmbeddingUnavailable as exc:
        raise SystemExit(f"The embedding model is needed to build the demo index: {exc}") from None
    model_dir = Path(embedder._model.model._model_dir)
    out.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "kb.db"
        store = Store(str(db))
        report = ingest_folder(store, SAMPLES_DIR, embedder=embedder)
        if report.skipped:
            raise SystemExit(f"some sample documents could not be ingested: {report.skipped}")
        with closing(sqlite3.connect(db)) as conn:
            conn.execute("UPDATE documents SET ingested_at = ?", (FIXED_TIME,))
            conn.execute("DELETE FROM audit_log")
            conn.commit()
            conn.execute("PRAGMA journal_mode=DELETE")
            conn.execute("VACUUM")
        texts = list(dict.fromkeys(shown_questions() + sentence_texts(store)))
        vectors = embedder.embed(texts)
        documents = store.documents()
        shutil.copyfile(db, out / "kb.db")
    np.savez_compressed(out / "vectors.npz", texts=np.array(texts), vectors=vectors.astype(np.float32))

    if copy_model:
        target = out / "model"
        target.mkdir(exist_ok=True)
        onnx = model_dir / MODEL_FILES[0]
        if sha256(onnx) != MODEL_SOURCE["onnx_sha256"]:
            raise SystemExit(f"{onnx} is not the published model file (SHA-256 differs)")
        for name in MODEL_FILES:
            if not (target / name).is_file() or sha256(target / name) != sha256(model_dir / name):
                shutil.copyfile(model_dir / name, target / name)

    files = sorted(p for p in SAMPLES_DIR.rglob("*") if p.is_file())
    by_id = {d["doc_id"]: d for d in documents}
    manifest = {
        "built": date.today().isoformat(),
        "built_by": "scripts/build_demo_index.py",
        "model": {"name": embedder.name, "key": embedder.key, **MODEL_SOURCE,
                  "files": {name: sha256(out / "model" / name) for name in MODEL_FILES if (out / "model" / name).is_file()}},
        "documents": [],
        "passages": sum(d["chunk_count"] for d in documents),
        "texts": len(texts),
        "files": {"kb.db": sha256(out / "kb.db"), "vectors.npz": sha256(out / "vectors.npz")},
    }
    for path in files:
        relative = path.relative_to(SAMPLES_DIR).as_posix()
        collection, inner = relative.split("/", 1)
        stored = by_id.get(make_doc_id(collection, inner))
        if stored is None:
            raise SystemExit(f"{relative} is not in the index")
        data = path.read_bytes()
        if stored["content_hash"] != content_hash(data):
            raise SystemExit(f"{relative}: the stored content hash does not match the file")
        manifest["documents"].append({"file": relative, "sha256": sha256(path), "doc_id": stored["doc_id"],
                                      "collection": stored["collection"], "title": stored["title"],
                                      "content_hash": stored["content_hash"], "passages": stored["chunk_count"]})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")

    # The finished index, without the model: every shown question must have its vector, and the
    # numbers must be the ones the README reports.
    numbers = DemoIndex(out, model_folder=None).eval_numbers()
    baseline = json.loads((ROOT / "eval" / "baseline.json").read_text(encoding="utf-8"))
    for name, summary in numbers.items():
        differ = {k: (baseline[name][k], summary[k]) for k in baseline[name] if summary.get(k) != baseline[name][k]}
        if differ:
            raise SystemExit(f"eval set {name} differs from eval/baseline.json (expected, got): {differ}")
    manifest["eval"] = numbers
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEMO_DIR)
    parser.add_argument("--no-model", action="store_true", help="do not copy the model files into demo/model")
    args = parser.parse_args(argv)
    manifest = build(args.out, copy_model=not args.no_model)
    print(f"{len(manifest['documents'])} documents, {manifest['passages']} passages, {manifest['texts']} stored vectors; "
          f"eval numbers match eval/baseline.json. Written to {args.out}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
