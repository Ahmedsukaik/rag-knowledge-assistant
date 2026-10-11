"""Command line.

On the server machine (work on the local database, KB_DB):
    python -m kb ingest [FOLDER] [--prune]           ingest <FOLDER>/<collection>/... (default KB_DOCS_DIR)
    python -m kb ingest FILE --collection hr         ingest one file
    python -m kb serve [--port 8200]                 HTTP API + MCP endpoint at /mcp
    python -m kb mcp-stdio                           MCP over stdio for a local client (identity: KB_MCP_TOKEN)
    python -m kb eval                                retrieval and refusal metrics on the question sets in eval/
    python -m kb download-model                      fetch the embedding model now (about 67 MB, once)

Anywhere (talk to a running server, KB_URL + KB_TOKEN):
    python -m kb ask "How many days of annual leave do I get?"
    python -m kb search "parental leave" [--k 5]
    python -m kb docs
    python -m kb audit [--limit 20]                  needs an admin token

Settings come from the environment or from .env in the current folder (or the file named by KB_ENV_FILE).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

from .envfile import load_env_file
from .text import one_line

ROOT = Path(__file__).resolve().parent.parent
MODE_LABEL = {
    "extractive": "extractive: sentences quoted from the documents, nothing AI-written",
    "claude": "written by Claude from the passages below; every citation is checked to quote the passage it names",
}


def utf8_output() -> None:
    """Windows consoles and pipes may not be UTF-8; never crash on a pound sign."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except ValueError:
                pass


def print_answer(result: dict[str, Any]) -> None:
    status = result.get("status")
    if status == "not_found":  # nothing was quoted or written, so no mode label
        print(f"Status: {one_line(status, 20)}")
    else:
        print(f"Status: {one_line(status, 20)}  ({MODE_LABEL.get(result.get('mode', ''), one_line(result.get('mode'), 20))})")
    print()
    for line in str(result.get("answer") or "").splitlines():
        print(f"  {one_line(line, 1000)}")
    passages = result.get("passages") or []
    if passages:
        print("\nSources:")
        for number, p in enumerate(passages, start=1):
            print(f"  [{number}] {one_line(p.get('label'), 150)}  ({one_line(p.get('passage_id'), 120)})")
    for w in result.get("warnings") or []:
        print(f"WARNING: {one_line(w, 300)}")


def _http(args: argparse.Namespace) -> Any:
    import httpx

    if not args.token:
        raise SystemExit("Set KB_TOKEN in .env or pass --token.")
    return httpx.Client(base_url=args.url.rstrip("/"), headers={"Authorization": f"Bearer {args.token}"}, timeout=120)


def _remote(args: argparse.Namespace) -> int:
    import httpx

    try:
        with _http(args) as http:
            if args.command == "ask":
                resp = http.post("/ask", json={"question": args.question})
            elif args.command == "search":
                resp = http.post("/search", json={"query": args.query, "k": args.k})
            elif args.command == "docs":
                resp = http.get("/documents")
            else:
                resp = http.get("/admin/audit", params={"limit": args.limit})
    except httpx.HTTPError as exc:
        print(f"Could not reach the server at {args.url} ({type(exc).__name__}).", file=sys.stderr)
        return 1
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if resp.status_code >= 400:
        print(f"Error {resp.status_code}: {one_line(body.get('error', resp.text), 300)}", file=sys.stderr)
        return 1
    if args.command == "ask":
        print_answer(body)
    elif args.command == "search":
        passages = body["passages"]
        if not passages:
            print("No matching passages in the documents you have access to.")
        for p in passages:  # similarity in meaning when the server uses the hybrid retriever, else the BM25 score
            measure = p["similarity"] if isinstance(p.get("similarity"), (int, float)) else p["score"]
            print(f"{measure:7.2f}  {one_line(p['passage_id'], 80)}  {one_line(p['label'], 100)}")
            print(f"         {one_line(p['text'], 200)}")
    elif args.command == "docs":
        for d in body["documents"]:
            pages = f", {d['pages']} pages" if d.get("pages") else ""
            print(f"{one_line(d['doc_id'], 60):<40} {one_line(d['title'], 60)}  ({d['kind']}{pages}, {d['chunk_count']} passages)")
    else:
        print("Times are UTC.")
        for e in reversed(body["entries"]):
            print(f"{e['ts'][:19].replace('T', ' ')}Z  {one_line(e['actor'], 18):<18} {one_line(e['role'], 8):<8} "
                  f"{e['channel']:<5} {one_line(e['action'], 16):<16} {one_line(e['outcome'], 10):<10} "
                  f"{one_line(e.get('query') or '', 60)}")
    return 0


def _embedder_or_exit(model: str | None = None) -> Any:
    """The shared embedding model, loaded now so a problem shows up at start, with a clear message."""
    from .embeddings import DEFAULT_EMBED_MODEL, EmbeddingUnavailable, shared_embedder

    embedder = shared_embedder(model or os.environ.get("KB_EMBED_MODEL") or DEFAULT_EMBED_MODEL)
    try:
        embedder.load()
    except EmbeddingUnavailable as exc:
        raise SystemExit(f"KB_RETRIEVER=hybrid needs the embedding model, but {exc}.") from None
    return embedder


def _ingest(args: argparse.Namespace) -> int:
    from .config import retriever_setting
    from .ingest import IngestError, ingest_bytes, ingest_folder
    from .store import Store

    store = Store(os.environ.get("KB_DB", "kb.db"))
    embedder = _embedder_or_exit() if retriever_setting() == "hybrid" else None
    path = Path(args.path or os.environ.get("KB_DOCS_DIR", "samples/docs"))
    started = time.monotonic()
    try:
        if path.is_file():
            if not args.collection:
                raise SystemExit("Ingesting one file needs --collection (for example --collection hr).")
            results = [ingest_bytes(store, path.read_bytes(), path.name, args.collection, origin="upload",
                                    replace=args.replace, embedder=embedder)]
            skipped: list[tuple[str, str]] = []
        else:
            report = ingest_folder(store, path, prune=args.prune, embedder=embedder)
            results, skipped = report.results, report.skipped
    except IngestError as exc:
        print(f"Not ingested: {exc}", file=sys.stderr)
        return 1
    for r in results:
        extra = f"  {r.chunks} passages  {one_line(r.title, 60)}" if r.status in ("added", "updated") else ""
        print(f"{r.status:<9} {r.doc_id:<34}{extra}")
    for name, reason in skipped:
        print(f"skipped   {one_line(name, 80)}: {one_line(reason, 150)}")
    counts = {s: sum(r.status == s for r in results) for s in ("added", "updated", "unchanged", "removed")}
    print(", ".join(f"{n} {s}" for s, n in counts.items() if n) or "nothing to do", f"({len(skipped)} skipped)")
    if embedder is not None:
        print(f"Passage vectors: {embedder.name}, stored with each added or updated document.")
    store.audit("local-cli", "admin", "cli", "ingest", str(path)[:300], "ok", passages=None,
                detail=", ".join(f"{n} {s}" for s, n in counts.items()), duration_ms=int((time.monotonic() - started) * 1000))
    return 0


def _knowledge_base(settings: Any) -> Any:
    """The KnowledgeBase for these settings. In Claude mode the settings and the presence of
    credentials are checked here, at start, instead of on the first question."""
    from .answer import ClaudeAnswerer, ExtractiveAnswerer, claude_settings_problems, ignored_settings
    from .service import KnowledgeBase, ModelBudget
    from .store import Store

    answerer: Any = ExtractiveAnswerer()
    if settings.answer_mode == "claude":
        problems = claude_settings_problems(settings.model, settings.claude_effort, settings.claude_fallbacks)
        if problems:
            raise SystemExit("Claude settings: " + "; ".join(problems) + ".")
        if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
            raise SystemExit("KB_ANSWER_MODE=claude needs ANTHROPIC_API_KEY (or ANTHROPIC_AUTH_TOKEN) in .env "
                             "or the environment. Only these two are supported here; other credential sources of "
                             "the anthropic SDK (ant profiles, workload identity) are not checked or used.")
        for name, ours in ignored_settings():
            print(f"Note: {name} is set in this environment but not used; this project reads {ours}.", file=sys.stderr)
        answerer = ClaudeAnswerer(model=settings.model, effort=settings.claude_effort,
                                  fallbacks=settings.claude_fallbacks)  # the embedder is added below
        print(f"Claude model: {answerer.model}, effort {answerer.effort or 'not sent'}, "
              f"fallback {answerer.fallbacks}.", file=sys.stderr)
        if os.environ.get("ANTHROPIC_BASE_URL"):  # read by the anthropic SDK itself, not by this project
            print(f"Claude API address from ANTHROPIC_BASE_URL: {one_line(os.environ['ANTHROPIC_BASE_URL'], 200)}",
                  file=sys.stderr)
    budget = ModelBudget(settings.claude_per_minute, settings.claude_per_day, settings.claude_per_person_day)
    store = Store(settings.db_path)
    if settings.retriever == "hybrid":
        from .retrieval import HybridRetriever

        embedder = _embedder_or_exit(settings.embed_model)
        retriever: Any = HybridRetriever(store, embedder)
        retriever.warm(frozenset(store.collections()))  # every passage has a vector before the first question
        answerer.embedder = embedder  # quotes are picked by meaning, as in the eval and the demo
    else:
        from .retrieval import BM25Retriever

        retriever = BM25Retriever(store)
    return KnowledgeBase(store, retriever, answerer=answerer, budget=budget)


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    from .api import build_app
    from .config import Settings

    settings = Settings.from_env()
    if settings.uses_placeholders:
        if args.host not in ("127.0.0.1", "localhost", "::1") or args.allowed_host:
            raise SystemExit("Refusing to serve on a public address or hostname with tokens published in this "
                             "repository (the change-me tokens from .env.example, or the demo or test tokens). "
                             "Set real tokens in .env first.")
        print("WARNING: using tokens published in this repository (change-me, demo or test tokens). Fine for a local "
              "try-out only.", file=sys.stderr)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    kb = _knowledge_base(settings)
    count = len(kb.store.documents())
    how = f"hybrid, {settings.embed_model}" if settings.retriever == "hybrid" else "bm25, keywords only"
    print(f"Answer mode: {settings.answer_mode}. Retriever: {how}. Documents in the database: {count}.", file=sys.stderr)
    if count == 0:
        print(f"WARNING: the database {settings.db_path} has no documents, so every question will be answered "
              "'not in the documents'. Run python -m kb ingest first.", file=sys.stderr)
    uvicorn.run(build_app(kb, settings.tokens, args.host, args.allowed_host or None), host=args.host, port=args.port)
    return 0


def _mcp_stdio(args: argparse.Namespace) -> int:
    from .config import Settings
    from .mcp_server import build_mcp

    settings = Settings.from_env()
    who = settings.tokens.identify(f"Bearer {os.environ.get('KB_MCP_TOKEN', '')}")
    if who is None:
        raise SystemExit("Set KB_MCP_TOKEN to one of the tokens in KB_TOKENS; its role decides what this client may read.")
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    build_mcp(_knowledge_base(settings), settings.tokens, stdio_identity=who).run("stdio")
    return 0


def _label(path: str) -> str:
    p = Path(path)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return p.as_posix()


SET_TITLES = {
    "main": "Evaluation on the {n} demo questions written for this project ({path}). The thresholds were chosen "
            "on this set, so it is optimistic",
    "heldout": "Held-out set 1: {n} questions ({path}), written by a separate assistant that saw only the documents "
               "(source fresh) and by the reviewers (source reviewer). Not used for tuning",
    "heldout_v2": "Held-out set 2: {n} questions ({path}), written by a separate assistant that saw only the documents, "
                  "after the thresholds were fixed. Not used for tuning",
}


def _eval(args: argparse.Namespace) -> int:
    import json

    from .embeddings import EmbeddingUnavailable
    from .evaluate import format_report, load_questions, regressions, run_eval

    answerer = None
    if args.mode == "claude":
        from .answer import ClaudeAnswerer

        try:
            answerer = ClaudeAnswerer()  # reads KB_CLAUDE_MODEL, KB_CLAUDE_EFFORT, KB_CLAUDE_FALLBACKS
        except ValueError as exc:
            raise SystemExit(f"Claude settings: {exc}.") from None
    sets = [("main", args.questions)]
    sets += [(name, path) for name, path in (("heldout", args.heldout), ("heldout_v2", args.heldout2))
             if path and Path(path).exists()]
    summaries: dict[str, Any] = {}
    leaks = 0
    for name, path in sets:
        try:
            report = run_eval(path, args.docs, k=args.k, answerer=answerer, retriever=args.retriever)
        except EmbeddingUnavailable as exc:
            raise SystemExit(f"KB_RETRIEVER=hybrid needs the embedding model, but {exc}.") from None
        print(format_report(report, SET_TITLES[name].format(n=len(load_questions(path)), path=_label(path))))
        print()
        summaries[name] = report.summary()
        leaks += report.leaks
    baseline_path = Path(args.baseline)
    if args.write_baseline:
        baseline_path.write_text(json.dumps({"retriever": args.retriever, **summaries}, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {_label(str(baseline_path))}.")
        return 1 if leaks else 0
    problems, compared = [], False
    if baseline_path.exists() and args.mode == "extractive":
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("retriever", "bm25") == args.retriever:
            compared = True
            for name, summary in summaries.items():
                problems += regressions(name, summary, baseline)
    if leaks:
        problems.append(f"restricted text was found in {leaks} results")
    if problems:
        print("REGRESSIONS against " + _label(str(baseline_path)) + ":")
        print("\n".join(f"  {p}" for p in problems))
        return 1
    print(f"No regressions against {_label(str(baseline_path))}." if compared
          else "Not compared with a baseline (none for this retriever and answer mode).")
    return 0


def _download_model(args: argparse.Namespace) -> int:
    from .embeddings import DEFAULT_EMBED_MODEL, EmbeddingUnavailable, shared_embedder

    embedder = shared_embedder(os.environ.get("KB_EMBED_MODEL") or DEFAULT_EMBED_MODEL)
    print(f"Embedding model {embedder.name}, kept in {embedder.cache_dir}")
    print("Downloading if needed (about 67 MB for the default model)...")
    try:
        embedder.embed(["a test sentence"])
    except EmbeddingUnavailable as exc:
        print(f"Failed: {exc}.", file=sys.stderr)
        return 1
    print(f"Ready. The folder holds {embedder.size_on_disk() / 1_000_000:.0f} MB; later runs need no network.")
    return 0


def main(argv: list[str] | None = None) -> int:
    utf8_output()
    load_env_file(os.environ.get("KB_ENV_FILE", ".env"))  # KB_ENV_FILE: for clients that start us elsewhere
    parser = argparse.ArgumentParser(prog="python -m kb", description="Document Q&A with sources and access rights")
    sub = parser.add_subparsers(dest="command", required=True)
    remote = argparse.ArgumentParser(add_help=False)  # options of the commands that talk to a server
    remote.add_argument("--url", default=os.environ.get("KB_URL", "http://127.0.0.1:8200"), help="default: KB_URL")
    remote.add_argument("--token", default=os.environ.get("KB_TOKEN"), help="default: KB_TOKEN")

    p = sub.add_parser("ingest", help="add or update documents in the local database")
    p.add_argument("path", nargs="?", help="a folder of collection folders, or one file (default KB_DOCS_DIR)")
    p.add_argument("--collection", help="collection for a single file")
    p.add_argument("--replace", action="store_true",
                   help="for a single file: replace a document with the same id that came from a different file")
    p.add_argument("--prune", action="store_true", help="remove documents whose file is gone from the folder")

    p = sub.add_parser("serve", help="start the HTTP API and the MCP endpoint")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8200)
    p.add_argument("--allowed-host", action="append", default=[], help="public hostname(s) to answer on (repeatable)")

    sub.add_parser("mcp-stdio", help="MCP over stdio for a local client (identity from KB_MCP_TOKEN)")

    def default(relative: str) -> str:
        return relative if Path(relative).exists() else str(ROOT / relative)

    p = sub.add_parser("eval", help="retrieval and refusal metrics on the demo and held-out question sets")
    p.add_argument("--questions", default=default("eval/questions.jsonl"))
    p.add_argument("--heldout", default=default("eval/heldout.jsonl"), help="held-out set 1, reported separately")
    p.add_argument("--heldout2", default=default("eval/heldout_v2.jsonl"), help="held-out set 2, reported separately")
    p.add_argument("--retriever", choices=["hybrid", "bm25"], default=None, help="default: KB_RETRIEVER, else hybrid")
    p.add_argument("--docs", default=default("samples/docs"))
    p.add_argument("--baseline", default=default("eval/baseline.json"),
                   help="numbers to compare with; the command fails if any got worse")
    p.add_argument("--write-baseline", action="store_true", help="save this run's numbers as the new baseline")
    p.add_argument("--k", type=int, default=3)
    p.add_argument("--mode", choices=["extractive", "claude"], default="extractive",
                   help="claude calls the live API once per question (needs ANTHROPIC_API_KEY)")

    sub.add_parser("download-model", help="download the embedding model now (once; later runs work offline)")

    sub.add_parser("ask", help="ask a question", parents=[remote]).add_argument("question")
    p = sub.add_parser("search", help="search passages", parents=[remote])
    p.add_argument("query")
    p.add_argument("--k", type=int, default=5)
    sub.add_parser("docs", help="list the documents you may read", parents=[remote])
    p = sub.add_parser("audit", help="show the audit log (admin token)", parents=[remote])
    p.add_argument("--limit", type=int, default=20)

    args = parser.parse_args(argv)
    if args.command == "eval" and args.retriever is None:
        from .config import retriever_setting

        args.retriever = retriever_setting()
    handlers = {"ingest": _ingest, "serve": _serve, "mcp-stdio": _mcp_stdio, "eval": _eval,
                "download-model": _download_model}
    return handlers.get(args.command, _remote)(args)
