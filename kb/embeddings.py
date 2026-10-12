"""Sentence embeddings, computed on this machine: no API key and no cost per question.

The default model is BAAI/bge-small-en-v1.5 (MIT licence, 384 numbers per text), run on the
CPU with ONNX Runtime through the fastembed package. The first time it is needed, fastembed
downloads a quantised ONNX copy of it (Qdrant/bge-small-en-v1.5-onnx-Q on Hugging Face,
about 67 MB) into KB_MODEL_DIR, by default the `models` folder of this project. After that
it loads from that folder only, without a network connection and without checking Hugging Face
for a newer revision, so the model cannot change under the stored vectors.

    python -m kb download-model      fetch it ahead of time (for example before going offline)

The repository also holds a copy of that file for the web demo (demo/model, see the README's
"Live demo"). When the models folder does not have the model yet, that copy is used instead of
a second download, after its SHA-256 is checked against the published file.

A vector is stored for every passage when it is ingested (table `vectors`), under the
embedder's `key`: the model name plus a fingerprint of the model file. A passage whose vector
is missing, has the wrong size, or was made by other model files (a database made with
KB_RETRIEVER=bm25, another model, a new download) gets a new one the first time it is searched.
"""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, Sequence

import numpy as np

if TYPE_CHECKING:
    from .store import Chunk, Store

DEFAULT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
ROOT = Path(__file__).resolve().parent.parent
# The copy of the default model kept for the web demo, and the SHA-256 of its ONNX file as
# published (Qdrant/bge-small-en-v1.5-onnx-Q, revision aa8f8b060edb00e03bfdd08813a2949946c8ba55).
SHIPPED_MODEL_DIR = ROOT / "demo" / "model"
SHIPPED_MODEL_SHA256 = "51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431"
SHIPPED_MODEL_FILES = ("model_optimized.onnx", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "config.json")


class EmbeddingUnavailable(RuntimeError):
    """The embedding model could not be loaded or downloaded. The message says what to do."""


class Embedder(Protocol):
    name: str

    @property
    def key(self) -> str:
        """Identifies the exact model files; vectors are stored under it."""
        ...

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """One unit-length float32 row per text."""
        ...


def model_dir() -> Path:
    return Path(os.environ.get("KB_MODEL_DIR") or ROOT / "models")


def passage_text(chunk: Chunk) -> str:
    """What is embedded for a passage: the same text the keyword index sees."""
    return f"{chunk.title}\n{chunk.section or ''}\n{chunk.text}"


def to_blob(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def from_blob(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype="<f4")


def shipped_model(model: str) -> Path | None:
    """The repository's own copy of `model` (demo/model), if it is that model and its ONNX file is
    the published one; else None."""
    folder = SHIPPED_MODEL_DIR
    if model != DEFAULT_EMBED_MODEL or not all((folder / name).is_file() for name in SHIPPED_MODEL_FILES):
        return None
    digest = hashlib.sha256()
    with (folder / SHIPPED_MODEL_FILES[0]).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return folder if digest.hexdigest() == SHIPPED_MODEL_SHA256 else None


def _unit(rows: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(rows, axis=1, keepdims=True)
    return (rows / np.where(norms == 0, 1, norms)).astype(np.float32)


class FastEmbedder:
    """fastembed's TextEmbedding, loaded once and shared by every thread."""

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, cache_dir: str | Path | None = None):
        self.name = model
        self.cache_dir = Path(cache_dir or model_dir())
        self._model: Any = None
        self._key = ""
        self.source = ""  # where it was loaded from: "folder", "shipped copy" or "download"
        self._lock = threading.Lock()

    @property
    def key(self) -> str:
        """The model name and a fingerprint of its ONNX file, for example BAAI/bge-small-en-v1.5@3f2a9c...."""
        self.load()
        return self._key

    def _fingerprint(self) -> str:
        folder = Path(getattr(getattr(self._model, "model", None), "_model_dir", "") or "")
        files = sorted(folder.glob("*.onnx")) if folder.is_dir() else []
        if not files:
            return folder.name or "unknown"
        digest = hashlib.sha256()
        for path in files:
            with path.open("rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    digest.update(block)
        return digest.hexdigest()[:12]

    def load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                from fastembed import TextEmbedding
            except ImportError:
                raise EmbeddingUnavailable("the fastembed package is not installed. Run pip install -r requirements.txt, "
                                           "or set KB_RETRIEVER=bm25 to search by keywords only") from None
            _quiet_fastembed()
            try:  # already downloaded: load it from the folder, without asking for a newer revision
                self._model = TextEmbedding(self.name, cache_dir=str(self.cache_dir), local_files_only=True)
                self.source = "folder"
            except Exception:
                shipped = shipped_model(self.name)
                try:
                    if shipped is not None:  # the repository's own copy: no second download
                        self._model = TextEmbedding(self.name, cache_dir=str(self.cache_dir), specific_model_path=str(shipped))
                        self.source = "shipped copy"
                    else:  # first use: download it
                        self._model = TextEmbedding(self.name, cache_dir=str(self.cache_dir))
                        self.source = "download"
                except Exception as exc:
                    raise EmbeddingUnavailable(
                        f"the embedding model {self.name} is not in {self.cache_dir} and could not be downloaded "
                        f"({type(exc).__name__}: {str(exc)[:200]}). Run python -m kb download-model on a machine "
                        "with internet access and copy that folder, or set KB_RETRIEVER=bm25 to search by keywords only"
                    ) from None
            self._key = f"{self.name}@{self._fingerprint()}"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        with self._lock:  # one ONNX session; keep calls simple and ordered
            rows = np.array(list(self._model.embed(list(texts), batch_size=64)), dtype=np.float32)
        return _unit(rows)

    def size_on_disk(self) -> int:
        return sum(p.stat().st_size for p in self.cache_dir.rglob("*") if p.is_file() and not p.is_symlink())


def _quiet_fastembed() -> None:
    """fastembed logs an ERROR line for every place it looks for a missing model; the message
    raised here says what to do instead."""
    try:
        from loguru import logger

        logger.disable("fastembed")
    except ImportError:
        pass


_shared: dict[tuple[str, str], FastEmbedder] = {}
_shared_lock = threading.Lock()


def shared_embedder(model: str = DEFAULT_EMBED_MODEL, cache_dir: str | Path | None = None) -> FastEmbedder:
    """One embedder per model and folder for the whole process (the model takes about a second to load)."""
    key = (model, str(Path(cache_dir or model_dir()).resolve()))
    with _shared_lock:
        if key not in _shared:
            _shared[key] = FastEmbedder(model, cache_dir)
        return _shared[key]


def fill_missing_vectors(store: Store, embedder: Embedder, collections: frozenset[str] | None = None) -> int:
    """Compute and store the vectors that are missing for this model. Returns how many."""
    chunks = store.chunks(collections if collections is not None else frozenset(store.collections()))
    have = store.vector_ids(embedder.key)
    missing = [c for c in chunks if c.chunk_id not in have]
    if missing:
        vectors = embedder.embed([passage_text(c) for c in missing])
        store.save_vectors(embedder.key, [(c.chunk_id, to_blob(v)) for c, v in zip(missing, vectors, strict=True)])
    return len(missing)
