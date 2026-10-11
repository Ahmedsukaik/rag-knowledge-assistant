"""Document Q&A that answers only from a company's own documents, with sources.

Modules:
    text       tokenizer, sentence splitter, cleaning, the "instructions to an AI" detector
    ingest     Markdown / text / PDF -> chunks with metadata
    store      SQLite: documents, chunks, passage vectors, audit log
    embeddings the local embedding model (fastembed, BAAI/bge-small-en-v1.5), no API
    retrieval  the Retriever interface: hybrid (keywords and meaning, the default) and BM25
    answer     extractive answers (no model) and Claude answers with citations, plus the source check
    service    KnowledgeBase: access rules, thresholds, audit around every operation
    api        FastAPI HTTP API and the MCP endpoint on one port
    mcp_server MCP tools (read-only)
    evaluate   retrieval and refusal metrics on the question sets in eval/
    cli        python -m kb ...
"""
