"""The ONNX embedder the web demo ships (no fastembed): the same tokens and the same vectors as
the product's own embedder, on every passage and every question the demo can show."""
from __future__ import annotations

import json

import numpy as np
import pytest
from conftest import SAMPLES

from kb.answer import sentence_text
from kb.demo_index import CHIPS, MODEL_SOURCE, STORY, sentence_texts
from kb.embeddings import EmbeddingUnavailable, passage_text
from kb.onnx_embed import DEMO_MODEL_DIR, OnnxEmbedder, WordPieceTokenizer, file_sha256
from kb.store import Store

AWKWARD = [
    "Caf\xe9 na\xefve r\xe9sum\xe9 \xc5ngstr\xf6m", "\u4e2d\u6587\u6d4b\u8bd5 mixed with English", "x" * 150,
    "a[MASK]b [CLS] and [SEP] written in the text", "tab\there\x00\x07 control characters", "\ufffd replacement",
    "\xa3500, \u20ac50 and $30", "emoji \U0001F600 here", "\u0130stanbul \u03a3\u039f\u03a3", "e\u0301 combining accent",
    "\u200bzero\u200dwidth", "line\u2028separator\x1c\x85next", "supercalifragilisticexpialidocious" * 3,
    " ".join(["word"] * 700), "\u3000\uff01\uff1f full width", "\u0e01\u0e02 \u05d0 \u0627\u0644", "", "   ",
    "E31: no Wi-Fi connection.", "3.5% of the total salary bill; \xa324,000 to \xa329,000",
]


def demo_questions() -> list[str]:
    questions = list(STORY.values()) + [c["question"] for c in CHIPS]
    for name in ("questions.jsonl", "heldout.jsonl", "heldout_v2.jsonl"):
        path = SAMPLES.parent.parent / "eval" / name
        questions += [json.loads(line)["question"] for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return questions


@pytest.fixture(scope="module")
def onnx() -> OnnxEmbedder:
    model = OnnxEmbedder()
    if not model.available():
        pytest.skip(f"the demo model is not in {DEMO_MODEL_DIR} (python scripts/build_demo_index.py puts it there)")
    model.load()
    return model


@pytest.fixture(scope="module")
def texts(hybrid_db) -> dict[str, list[str]]:
    store = Store(str(hybrid_db))
    chunks = store.chunks(frozenset(store.collections()))
    return {"passages": [passage_text(c) for c in chunks], "sentences": sentence_texts(store), "questions": demo_questions()}


def test_the_shipped_model_is_the_published_file(onnx):
    assert file_sha256(onnx.folder / "model_optimized.onnx") == MODEL_SOURCE["onnx_sha256"]


def test_tokens_are_identical_to_the_tokenizers_package(onnx, texts, embedder):
    from fastembed.common.preprocessor_utils import load_tokenizer  # the tokenizer fastembed runs

    reference, _ = load_tokenizer(onnx.folder)
    ours = WordPieceTokenizer(onnx.folder / "tokenizer.json")
    everything = [t for group in texts.values() for t in group] + AWKWARD
    for text in everything:
        assert ours.encode(text) == reference.encode(text).ids, repr(text[:80])
    assert len(everything) > 300


def test_vectors_match_the_products_embedder_on_every_passage_and_question(onnx, embedder, texts):
    assert onnx.key == embedder.key  # same model file, so the stored vectors are read
    for group, items in texts.items():
        ours, theirs = onnx.embed(items), embedder.embed(items)
        similarity = (ours * theirs).sum(axis=1)
        assert similarity.min() >= 0.9999, (group, items[int(similarity.argmin())])
        assert np.allclose(np.linalg.norm(ours, axis=1), 1.0, atol=1e-5)


def test_order_and_batching_do_not_change_a_vector(onnx):
    texts = ["How much leave do I get?", "a much longer question about the expense policy and the hotel limit in London",
             "E31", "Where can I print?"] * 9
    together = onnx.embed(texts)
    alone = np.stack([onnx.embed([t])[0] for t in texts])
    assert ((together * alone).sum(axis=1) >= 0.99999).all()


def test_a_sentence_embeds_with_its_context(onnx):
    from kb.store import Chunk

    chunk = Chunk("general/m#4", "general/m", "general", 4, "Manual", "Errors", 3, "E31: no Wi-Fi connection.")
    assert sentence_text(chunk, "E31: no Wi-Fi connection.") == "Manual, Errors: E31: no Wi-Fi connection."


def test_a_missing_model_is_reported_not_guessed(tmp_path):
    model = OnnxEmbedder(tmp_path)
    assert not model.available()
    with pytest.raises(EmbeddingUnavailable, match="not in"):
        model.embed(["anything"])


def test_another_kind_of_tokenizer_is_refused(tmp_path):
    (tmp_path / "tokenizer.json").write_text(json.dumps({"normalizer": {"type": "NFC"}, "pre_tokenizer": {"type": "Metaspace"},
                                                         "model": {"type": "BPE", "vocab": {}}, "post_processor": None}))
    with pytest.raises(EmbeddingUnavailable, match="not a BERT WordPiece tokenizer"):
        WordPieceTokenizer(tmp_path / "tokenizer.json")


def test_the_web_demo_never_imports_fastembed():
    import subprocess
    import sys

    code = ("import sys, kb.demo_web\nfrom kb.demo_index import DemoIndex\nDemoIndex().load_model()\n"
            "bad = sorted(m for m in sys.modules if m.split('.')[0] in ('fastembed', 'tokenizers', 'huggingface_hub', 'mcp', 'pypdf'))\n"
            "print(bad)\nassert not bad, bad")
    result = subprocess.run([sys.executable, "-c", code], cwd=SAMPLES.parent.parent, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
