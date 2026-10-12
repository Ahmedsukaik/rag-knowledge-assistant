"""The same embedding model as embeddings.py, run with ONNX Runtime directly: no fastembed.

The web demo runs on a serverless host with a size limit on what it may ship. fastembed pulls in
the Hugging Face hub client, the `tokenizers` package and image libraries that the demo does not
need, so this module does the two things fastembed does for BAAI/bge-small-en-v1.5 itself:

1. Tokenizing, in plain Python, by the rules in the model's own tokenizer.json (BERT WordPiece:
   clean the text, split Chinese characters, strip accents, lower-case, split on spaces and
   punctuation, longest-match word pieces, [CLS] ... [SEP], at most 512 tokens). A test checks
   that the token ids are identical to those of the `tokenizers` package on every passage,
   sentence and question of the demo, and on awkward text (accents, control characters,
   Chinese, long words, special tokens).
2. Running the ONNX file and keeping the first token's output, normalised to unit length.

A test checks that its vectors match fastembed's (cosine similarity at least 0.9999) on every
passage and question. The key is the same as FastEmbedder's: the model name and the first 12
hex digits of the ONNX file's SHA-256, so it reads the vectors fastembed stored.

    OnnxEmbedder(Path("demo/model")).embed(["How much annual leave do I get?"])
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .embeddings import DEFAULT_EMBED_MODEL, ROOT, EmbeddingUnavailable

MODEL_FILE = "model_optimized.onnx"
TOKENIZER_FILE = "tokenizer.json"
DEMO_MODEL_DIR = ROOT / "demo" / "model"
BATCH = 16

# Unicode White_Space (what Rust's char::is_whitespace accepts), which the tokenizer uses.
_WHITE_SPACE = frozenset("\t\n\x0b\x0c\r \x85\xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009"
                         "\u200a\u2028\u2029\u202f\u205f\u3000")
_CJK = ((0x4E00, 0x9FFF), (0x3400, 0x4DBF), (0x20000, 0x2A6DF), (0x2A700, 0x2B73F), (0x2B740, 0x2B81F),
        (0x2B920, 0x2CEAF), (0xF900, 0xFAFF), (0x2F800, 0x2FA1F))
_ASCII_PUNCT = frozenset("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")


def _is_control(c: str) -> bool:
    return c not in "\t\n\r" and unicodedata.category(c) in ("Cc", "Cf", "Cs", "Co", "Cn")


def _is_chinese(c: str) -> bool:
    code = ord(c)
    return any(low <= code <= high for low, high in _CJK)


def _is_punctuation(c: str) -> bool:
    return c in _ASCII_PUNCT or unicodedata.category(c).startswith("P")


class WordPieceTokenizer:
    """BERT WordPiece tokenization as described by a Hugging Face tokenizer.json. Only the
    combination this model uses is supported (BertNormalizer, BertPreTokenizer, WordPiece,
    [CLS] $A [SEP]); anything else is refused, so a different model cannot be tokenized wrongly
    without notice."""

    def __init__(self, path: str | Path, max_length: int = 512):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        normalizer, pre, model, post = (spec.get("normalizer") or {}, spec.get("pre_tokenizer") or {},
                                        spec.get("model") or {}, spec.get("post_processor") or {})
        if (normalizer.get("type"), pre.get("type"), model.get("type"), post.get("type")) != (
                "BertNormalizer", "BertPreTokenizer", "WordPiece", "TemplateProcessing"):
            raise EmbeddingUnavailable(f"{path} is not a BERT WordPiece tokenizer; this module only runs that kind")
        self.clean_text = bool(normalizer.get("clean_text", True))
        self.chinese = bool(normalizer.get("handle_chinese_chars", True))
        self.lowercase = bool(normalizer.get("lowercase", True))
        strip = normalizer.get("strip_accents")
        self.strip_accents = self.lowercase if strip is None else bool(strip)
        self.vocab: dict[str, int] = model["vocab"]
        self.unk = self.vocab[model.get("unk_token", "[UNK]")]
        self.prefix = model.get("continuing_subword_prefix", "##")
        self.max_chars = int(model.get("max_input_chars_per_word", 100))
        single = [next(iter(p.items())) for p in post.get("single", [])]
        if [kind for kind, _ in single] != ["SpecialToken", "Sequence", "SpecialToken"]:
            raise EmbeddingUnavailable(f"{path}: unexpected post-processing template")
        specials = post.get("special_tokens", {})
        self.cls = specials[single[0][1]["id"]]["ids"][0]
        self.sep = specials[single[2][1]["id"]]["ids"][0]
        self.max_length = max_length
        # Special tokens written in the text itself ("[MASK]") become that token, as in the
        # `tokenizers` package. Longest first, so a longer token wins over its prefix.
        self.added = {t["content"]: t["id"] for t in spec.get("added_tokens", []) if not t.get("normalized", True)}
        names = sorted(self.added, key=len, reverse=True)
        self._added = re.compile("(" + "|".join(map(re.escape, names)) + ")") if names else None
        self._cache: dict[str, list[int]] = {}

    # ------------------------------------------------------------------ the steps
    def normalize(self, text: str) -> str:
        if self.clean_text:
            text = "".join(" " if c in _WHITE_SPACE else c for c in text
                           if not (c == "\x00" or c == "\ufffd" or _is_control(c)))
        if self.chinese:
            text = "".join(f" {c} " if _is_chinese(c) else c for c in text)
        if self.strip_accents:
            text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
        if self.lowercase:  # one character at a time, as the tokenizers package does (no final-sigma rule)
            text = "".join(c.lower() for c in text)
        return text

    @staticmethod
    def pre_tokenize(text: str) -> list[str]:
        words: list[str] = []
        current: list[str] = []
        for c in text:
            if c in _WHITE_SPACE:
                if current:
                    words.append("".join(current))
                    current = []
            elif _is_punctuation(c):
                if current:
                    words.append("".join(current))
                    current = []
                words.append(c)
            else:
                current.append(c)
        if current:
            words.append("".join(current))
        return words

    def word_pieces(self, word: str) -> list[int]:
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        if len(word) > self.max_chars:
            pieces = [self.unk]
        else:
            pieces, start = [], 0
            while start < len(word):
                end, found = len(word), None
                while start < end:
                    piece = word[start:end] if start == 0 else self.prefix + word[start:end]
                    if piece in self.vocab:
                        found = self.vocab[piece]
                        break
                    end -= 1
                if found is None:
                    pieces = [self.unk]
                    break
                pieces.append(found)
                start = end
        if len(self._cache) < 50_000:
            self._cache[word] = pieces
        return pieces

    def encode(self, text: str) -> list[int]:
        """Token ids with [CLS] and [SEP], cut to max_length."""
        ids: list[int] = []
        segments = self._added.split(text) if self._added else [text]
        for i, segment in enumerate(segments):
            if i % 2 == 1:  # a special token written in the text
                ids.append(self.added[segment])
                continue
            for word in self.pre_tokenize(self.normalize(segment)):
                ids.extend(self.word_pieces(word))
        return [self.cls, *ids[: self.max_length - 2], self.sep]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class OnnxEmbedder:
    """BAAI/bge-small-en-v1.5 from a folder with model_optimized.onnx and tokenizer.json.
    Loaded once, on first use, and shared by every thread."""

    def __init__(self, folder: str | Path = DEMO_MODEL_DIR, name: str = DEFAULT_EMBED_MODEL, threads: int | None = None):
        self.name = name
        self.folder = Path(folder)
        self.threads = threads
        self._session: Any = None
        self._tokenizer: WordPieceTokenizer | None = None
        self._key = ""
        self._lock = threading.Lock()

    def available(self) -> bool:
        return (self.folder / MODEL_FILE).is_file() and (self.folder / TOKENIZER_FILE).is_file()

    @property
    def key(self) -> str:
        """The model name and a fingerprint of its ONNX file, as FastEmbedder writes it."""
        self.load()
        return self._key

    @property
    def loaded(self) -> bool:
        return self._session is not None

    def load(self) -> None:
        with self._lock:
            if self._session is not None:
                return
            if not self.available():
                raise EmbeddingUnavailable(f"the embedding model is not in {self.folder} ({MODEL_FILE} and {TOKENIZER_FILE})")
            try:
                import onnxruntime as ort
            except ImportError:
                raise EmbeddingUnavailable("the onnxruntime package is not installed") from None
            tokenizer = WordPieceTokenizer(self.folder / TOKENIZER_FILE)
            options = ort.SessionOptions()
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            if self.threads:
                options.intra_op_num_threads = options.inter_op_num_threads = self.threads
            try:
                session = ort.InferenceSession(str(self.folder / MODEL_FILE), sess_options=options,
                                               providers=["CPUExecutionProvider"])
            except Exception as exc:
                raise EmbeddingUnavailable(f"ONNX Runtime could not load {MODEL_FILE}: {type(exc).__name__}") from None
            self._key = f"{self.name}@{file_sha256(self.folder / MODEL_FILE)[:12]}"
            self._tokenizer, self._session = tokenizer, session

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.load()
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        assert self._tokenizer is not None
        encoded = [self._tokenizer.encode(t) for t in texts]
        order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]))  # similar lengths together: less padding
        rows: dict[int, np.ndarray] = {}
        for start in range(0, len(order), BATCH):
            batch = order[start:start + BATCH]
            width = max(len(encoded[i]) for i in batch)
            ids = np.zeros((len(batch), width), dtype=np.int64)
            mask = np.zeros((len(batch), width), dtype=np.int64)
            for row, i in enumerate(batch):
                ids[row, : len(encoded[i])] = encoded[i]
                mask[row, : len(encoded[i])] = 1
            with self._lock:  # one session; keep calls simple and ordered
                hidden = self._session.run(None, {"input_ids": ids, "attention_mask": mask,
                                                  "token_type_ids": np.zeros_like(ids)})[0]
            first = np.asarray(hidden[:, 0], dtype=np.float32)
            norms = np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
            for row, i in enumerate(batch):
                rows[i] = first[row] / norms[row]
        out = np.stack([rows[i] for i in range(len(encoded))]).astype(np.float32)
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.where(norms == 0, 1, norms)).astype(np.float32)
