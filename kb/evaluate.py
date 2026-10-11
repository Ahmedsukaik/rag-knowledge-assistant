"""Retrieval and refusal metrics on hand-written question sets.

    python -m kb eval            (eval/questions.jsonl, eval/heldout.jsonl and eval/heldout_v2.jsonl over samples/docs)

Each line of a questions file is one JSON object:

    {"id": "a01", "type": "answerable",   "role": "staff", "question": "...", "expected_docs": ["general/leave-policy"],
     "key_phrases": ["25 days of paid annual leave"]}
    {"id": "o01", "type": "out_of_scope", "role": "staff", "question": "..."}
    {"id": "r01", "type": "restricted",   "role": "staff", "question": "...", "restricted_doc": "hr/salary-bands"}

"key_phrases" (optional) are words from the sentence that answers the question; an answer
quotes the right sentence if one of its quotes contains one of them. An optional "source"
field groups questions in the report (the first held-out file uses
"fresh" and "reviewer").

The documents are ingested into a throwaway database, and every question goes through the
same KnowledgeBase the server uses, with the role named in the question.

Metrics:
    retrieval hit@1 / hit@k   answerable questions whose expected document is the first / in the top k results
    answered                  answerable questions that got an answer (not "not in the documents")
    cites the right document  answered questions whose citations include an expected document
    quotes the right sentence answered questions with key phrases whose quotes contain one of them
    refused                   out-of-scope and restricted questions answered with "not in the documents"
    restricted text found     results (search hits, passages, answers) containing a sentence that exists
                              only in documents the role may not read. Checked on the text itself, read
                              straight from the database, so it does not rely on the access filter. Must be 0.

The command compares the numbers with eval/baseline.json and exits with status 1 if any of
them got worse or if restricted text was found.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .answer import Answerer
from .config import DEFAULT_ROLES, Identity, parse_roles
from .ingest import ingest_folder
from .retrieval import build_retriever
from .service import KnowledgeBase
from .store import Store
from .text import split_sentences, squash

TYPES = ("answerable", "out_of_scope", "restricted")
GOOD = ("hit_at_1", "hit_at_k", "answered", "cites_expected", "quotes_expected", "out_of_scope_refused",
        "restricted_refused")


@dataclass
class Outcome:
    item: dict[str, Any]
    top_docs: list[str]
    status: str
    cited_docs: list[str]
    leaked: int  # results that contained restricted text
    answer: str = ""
    quotes: list[str] = field(default_factory=list)  # the text each citation quotes

    @property
    def hit1(self) -> bool:
        return bool(self.top_docs) and self.top_docs[0] in self.item.get("expected_docs", [])

    @property
    def hitk(self) -> bool:
        return any(d in self.item.get("expected_docs", []) for d in self.top_docs)

    @property
    def cites_expected(self) -> bool:
        return any(d in self.item.get("expected_docs", []) for d in self.cited_docs)

    @property
    def quotes_expected(self) -> bool:
        quoted = " ".join(squash(q).lower() for q in self.quotes)
        return any(squash(phrase).lower() in quoted for phrase in self.item.get("key_phrases", []))


@dataclass
class Report:
    k: int
    mode: str
    retriever: str = "hybrid"
    outcomes: list[Outcome] = field(default_factory=list)

    def summary(self, source: str | None = None) -> dict[str, Any]:
        rows = [o for o in self.outcomes if source is None or o.item.get("source") == source]
        a = [o for o in rows if o.item["type"] == "answerable"]
        oos = [o for o in rows if o.item["type"] == "out_of_scope"]
        r = [o for o in rows if o.item["type"] == "restricted"]
        answered = [x for x in a if x.status == "answered"]
        return {
            "questions": len(rows),
            "answerable": len(a),
            "hit_at_1": sum(x.hit1 for x in a),
            "hit_at_k": sum(x.hitk for x in a),
            "answered": len(answered),
            "cites_expected": sum(x.cites_expected for x in answered),
            "quote_checked": sum(bool(x.item.get("key_phrases")) for x in answered),
            "quotes_expected": sum(x.quotes_expected for x in answered),
            "out_of_scope": len(oos),
            "out_of_scope_refused": sum(x.status == "not_found" for x in oos),
            "restricted": len(r),
            "restricted_refused": sum(x.status == "not_found" for x in r),
            "restricted_text_found": sum(o.leaked for o in rows),
        }

    @property
    def leaks(self) -> int:
        return sum(o.leaked for o in self.outcomes)

    def sources(self) -> list[str]:
        return sorted({o.item["source"] for o in self.outcomes if o.item.get("source")})


def load_questions(path: str | Path) -> list[dict[str, Any]]:
    items = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        item = json.loads(line)
        if item.get("type") not in TYPES or not item.get("question") or not item.get("role") or not item.get("id"):
            raise ValueError(f"line {number}: needs id, type ({', '.join(TYPES)}), role and question")
        if item["type"] == "answerable" and not item.get("expected_docs"):
            raise ValueError(f"line {number}: an answerable question needs expected_docs")
        phrases = item.get("key_phrases", [])
        if not isinstance(phrases, list) or not all(isinstance(x, str) and x.strip() for x in phrases):
            raise ValueError(f"line {number}: key_phrases must be a list of words from the answering sentence")
        if item["type"] == "restricted" and not item.get("restricted_doc"):
            raise ValueError(f"line {number}: a restricted question needs restricted_doc")
        items.append(item)
    ids = [i["id"] for i in items]
    if len(ids) != len(set(ids)):
        raise ValueError("question ids must be unique")
    return items


def _sentences(store: Store, collections: frozenset[str]) -> set[str]:
    return {s for c in store.chunks(collections) for s in split_sentences(c.text) if len(s) >= 20}


def run_eval(questions: str | Path, docs: str | Path, k: int = 3, answerer: Answerer | None = None,
             workdir: str | None = None, retriever: str = "hybrid", options: dict[str, Any] | None = None) -> Report:
    """Run one question set. `options` are passed to KnowledgeBase (thresholds), for tuning."""
    items = load_questions(questions)
    workdir = workdir or tempfile.mkdtemp(prefix="kb-eval-")
    Path(workdir).mkdir(parents=True, exist_ok=True)
    store = Store(str(Path(workdir) / "eval.db"))
    chosen = build_retriever(store, retriever)
    report_ingest = ingest_folder(store, docs, embedder=getattr(chosen, "embedder", None))
    if report_ingest.skipped:
        raise ValueError(f"some documents could not be ingested: {report_ingest.skipped}")
    kb = KnowledgeBase(store, chosen, answerer=answerer, **(options or {}))
    roles = parse_roles(DEFAULT_ROLES)
    everything = frozenset(store.collections())
    report = Report(k, kb.answerer.mode, retriever)
    secret_cache: dict[frozenset[str], set[str]] = {}
    for item in items:
        if item["role"] not in roles:
            raise ValueError(f"{item['id']}: unknown role {item['role']!r}")
        who = Identity(f"eval-{item['role']}", item["role"], roles[item["role"]])
        visible = kb.visible(who)
        if visible not in secret_cache:  # sentences that exist only in documents this role may not read
            secret_cache[visible] = _sentences(store, everything - visible) - _sentences(store, visible)
        secrets = secret_cache[visible]
        hits = kb.search(who, item["question"], k=k, channel="eval")
        result = kb.ask(who, item["question"], channel="eval")
        shown = [h["text"] for h in hits] + [p["text"] for p in result.passages] + [result.answer]
        leaked = sum(any(s in text for s in secrets) for text in shown)
        leaked += sum(h["collection"] not in visible for h in hits) + sum(p["collection"] not in visible for p in result.passages)
        report.outcomes.append(Outcome(
            item, [h["doc_id"] for h in hits], result.status,
            [c["passage_id"].split("#")[0] for c in result.citations], leaked,
            result.answer, [c["quote"] for c in result.citations],
        ))
    return report


def pct(n: int, d: int) -> str:
    return f"{n}/{d}".ljust(7) + (f"{100 * n / d:5.1f}%" if d else "    -")


def _block(s: dict[str, Any], k: int) -> list[str]:
    return [
        f"Answerable questions ({s['answerable']})",
        f"  retrieval hit@1                  {pct(s['hit_at_1'], s['answerable'])}",
        f"  retrieval hit@{k}                  {pct(s['hit_at_k'], s['answerable'])}",
        f"  answered                         {pct(s['answered'], s['answerable'])}",
        f"  answer cites the right document  {pct(s['cites_expected'], s['answered'])}",
        f"  answer quotes the right sentence {pct(s['quotes_expected'], s['quote_checked'])}",
        f"Out-of-scope questions ({s['out_of_scope']})",
        f"  refused (not in the documents)   {pct(s['out_of_scope_refused'], s['out_of_scope'])}",
        f"Access-restricted questions ({s['restricted']})",
        f"  refused (not in the documents)   {pct(s['restricted_refused'], s['restricted'])}",
        f"  restricted text found            {s['restricted_text_found']}",
    ]


def _quoted(o: Outcome) -> str:
    """The first quote of an answer that counts as a miss, so the report shows what was said."""
    if o.status != "answered" or not o.quotes:
        return ""
    first = " ".join(o.quotes[0].split())
    return f'\n      quoted: "{first[:90]}{"..." if len(first) > 90 else ""}"'


def format_report(report: Report, title: str) -> str:
    lines = [title, f"Answer mode: {report.mode}. Retriever: {report.retriever}. Passages per search: {report.k}.", ""]
    lines += _block(report.summary(), report.k)
    for source in report.sources() if len(report.sources()) > 1 else []:
        s = report.summary(source)
        lines.append(f"  by source: {source:<9} answered {s['answered']}/{s['answerable']}, "
                     f"out of scope refused {s['out_of_scope_refused']}/{s['out_of_scope']}, "
                     f"restricted refused {s['restricted_refused']}/{s['restricted']}")
    misses = []
    for o in report.outcomes:
        kind, q = o.item["type"], f"{o.item['id']} [{o.item['role']}] {o.item['question']}"
        labelled = bool(o.item.get("key_phrases"))
        if kind == "answerable" and not (o.hit1 and o.status == "answered" and o.cites_expected
                                         and (o.quotes_expected or not labelled)):
            misses.append(f"  {q}\n      expected {', '.join(o.item['expected_docs'])}; top results "
                          f"{', '.join(o.top_docs) or 'none'}; status {o.status}" + _quoted(o))
        elif kind != "answerable" and (o.status != "not_found" or o.leaked):
            misses.append(f"  {q}\n      status {o.status}; top results {', '.join(o.top_docs) or 'none'}"
                          + (f"; RESTRICTED TEXT IN {o.leaked} RESULTS" if o.leaked else "") + _quoted(o))
    lines += ["", "Misses:" if misses else "Misses: none"] + misses
    return "\n".join(lines)


def regressions(name: str, summary: dict[str, Any], baseline: dict[str, Any]) -> list[str]:
    """Numbers that got worse than the baseline for this question set."""
    old = baseline.get(name)
    if not old:
        return []
    out = [f"{name}: {key} went from {old[key]} to {summary[key]}" for key in GOOD
           if key in old and summary[key] < old[key]]
    if summary["restricted_text_found"] > old.get("restricted_text_found", 0):
        out.append(f"{name}: restricted text found in {summary['restricted_text_found']} results")
    return out
