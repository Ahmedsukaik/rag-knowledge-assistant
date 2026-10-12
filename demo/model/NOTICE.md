# The embedding model shipped with the web demo

These files are the quantised ONNX copy of BAAI/bge-small-en-v1.5 that `fastembed` downloads
for this product (`python -m kb download-model`), copied here unchanged by
`scripts/build_demo_index.py` so the Vercel function can run it without fastembed
(`kb/onnx_embed.py`).

- Source: https://huggingface.co/Qdrant/bge-small-en-v1.5-onnx-Q, revision aa8f8b060edb00e03bfdd08813a2949946c8ba55
- Original model: BAAI/bge-small-en-v1.5
- Licence: MIT (as stated on both model pages)
- `model_optimized.onnx` SHA-256: 51f1bd0addd6e859e42c2c8021a5e5461385bb676a649f4b269aa445449f2431
  (checked by the build script and by `tests/test_onnx_embed.py`)

The demo runs without these files too: questions people type are then searched by keywords
only, and the page says so (see "Live demo" in the README).
