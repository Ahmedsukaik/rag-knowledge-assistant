"""The command line, settings, the eval command, and the demo itself."""
import json
import os
import subprocess
import sys

import pytest
from conftest import ADMIN, ROOT, SAMPLES, STAFF

from kb import cli, evaluate
from kb.config import Settings
from kb.evaluate import format_report, load_questions, regressions, run_eval
from kb.service import KnowledgeBase

QUESTIONS = ROOT / "eval" / "questions.jsonl"
HELDOUT = ROOT / "eval" / "heldout.jsonl"
HELDOUT2 = ROOT / "eval" / "heldout_v2.jsonl"
SETS = (("main", QUESTIONS), ("heldout", HELDOUT), ("heldout_v2", HELDOUT2))
# The numbers quoted in README.md (hybrid retriever, extractive answers), copied here by hand.
# If retrieval changes, this test fails until the README (and this copy) are updated.
README_NUMBERS = {
    "main": {"questions": 64, "answerable": 37, "hit_at_1": 37, "hit_at_k": 37, "answered": 36, "cites_expected": 36,
             "quote_checked": 36, "quotes_expected": 36, "out_of_scope": 17, "out_of_scope_refused": 16,
             "restricted": 10, "restricted_refused": 10, "restricted_text_found": 0},
    "heldout": {"questions": 31, "answerable": 21, "hit_at_1": 20, "hit_at_k": 21, "answered": 14, "cites_expected": 14,
                "quote_checked": 14, "quotes_expected": 14, "out_of_scope": 6, "out_of_scope_refused": 2,
                "restricted": 4, "restricted_refused": 4, "restricted_text_found": 0},
    "heldout_v2": {"questions": 30, "answerable": 16, "hit_at_1": 15, "hit_at_k": 16, "answered": 10, "cites_expected": 9,
                   "quote_checked": 10, "quotes_expected": 8, "out_of_scope": 8, "out_of_scope_refused": 7,
                   "restricted": 6, "restricted_refused": 5, "restricted_text_found": 0},
}
# The same sets with keyword search only (KB_RETRIEVER=bm25), also quoted in the README.
README_BM25 = {
    "main": {**README_NUMBERS["main"], "out_of_scope_refused": 15},
    "heldout": {**README_NUMBERS["heldout"], "hit_at_k": 20, "answered": 6, "cites_expected": 6, "quote_checked": 6,
                "quotes_expected": 5, "out_of_scope_refused": 4},
    "heldout_v2": {**README_NUMBERS["heldout_v2"], "hit_at_1": 16, "answered": 3, "cites_expected": 3, "quote_checked": 3,
                   "quotes_expected": 3, "out_of_scope_refused": 8, "restricted_refused": 6},
}


def run(*args: str, cwd=ROOT, env: dict | None = None, timeout: int = 120) -> subprocess.CompletedProcess:
    full_env = {**os.environ, "PYTHONIOENCODING": "utf-8", **(env or {})}
    return subprocess.run([sys.executable, *args], cwd=cwd, env=full_env, capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)


def test_eval_numbers_match_the_readme_and_the_baseline(tmp_path, embedder):
    baseline = json.loads((ROOT / "eval" / "baseline.json").read_text(encoding="utf-8"))
    assert baseline["retriever"] == "hybrid"
    reports = {}
    for name, path in SETS:
        reports[name] = run_eval(path, SAMPLES, k=3, workdir=str(tmp_path / name))
        assert reports[name].summary() == README_NUMBERS[name] == baseline[name], name
    text = format_report(reports["main"], "title")
    assert "a31 [staff] What does the amber battery light mean?" in text  # misses are listed, not hidden
    held = reports["heldout"]
    assert {s: held.summary(s)["questions"] for s in held.sources()} == {"fresh": 24, "reviewer": 7}
    for name, path in SETS:
        keywords = run_eval(path, SAMPLES, k=3, workdir=str(tmp_path / f"bm25-{name}"), retriever="bm25")
        assert keywords.summary() == README_BM25[name], name


def test_the_question_sets_are_well_formed(store):
    docs = {f"{p.parent.name}/{p.stem}" for p in SAMPLES.rglob("*.*")}
    text_of: dict[str, str] = {}
    for chunk in store.chunks(frozenset(store.collections())):
        text_of[chunk.doc_id] = text_of.get(chunk.doc_id, "") + "".join(chunk.text.split()).lower()
    for _, path in SETS:
        items = load_questions(path)
        assert {i["type"] for i in items} == {"answerable", "out_of_scope", "restricted"}
        for item in items:
            for doc in item.get("expected_docs", []) + [item.get("restricted_doc")] * ("restricted_doc" in item):
                assert doc in docs, item["id"]
            if item["type"] == "answerable":  # every key phrase is in the document that answers the question
                assert item["key_phrases"], item["id"]
                for phrase in item["key_phrases"]:
                    assert any("".join(phrase.split()).lower() in text_of[d] for d in item["expected_docs"]), (item["id"], phrase)
            if item["type"] == "restricted":
                assert not item["restricted_doc"].startswith("general/") and not item["restricted_doc"].startswith(item["role"])
    tuned_on = {i["question"].lower() for i in load_questions(QUESTIONS)}
    for path in (HELDOUT, HELDOUT2):  # no held-out question is in the set the thresholds were chosen on
        assert not {i["question"].lower() for i in load_questions(path)} & tuned_on, path
    # The two held-out sets were written separately and happen to share one question word for word.
    shared = {i["question"].lower() for i in load_questions(HELDOUT)} & {i["question"].lower() for i in load_questions(HELDOUT2)}
    assert shared == {"what's the pay range for someone at senior specialist level?"}


@pytest.mark.parametrize("retriever", ["bm25", "hybrid"])
def test_the_restricted_text_metric_can_actually_fail(tmp_path, monkeypatch, request, retriever):
    """The leak check reads the restricted text from the database itself. With the access filter
    switched off, it must find leaks."""
    if retriever == "hybrid":
        request.getfixturevalue("embedder")  # skipped, with the reason, if the model is not available

    def no_filter(self, query, visible, k):
        return self.retriever.search(query, frozenset(self.store.collections()), k), None

    monkeypatch.setattr(KnowledgeBase, "_retrieve", no_filter)
    report = run_eval(QUESTIONS, SAMPLES, k=3, workdir=str(tmp_path), retriever=retriever)
    assert report.summary()["restricted_text_found"] > 0
    assert regressions("main", report.summary(), README_NUMBERS)  # and the eval command would fail
    assert evaluate.regressions("main", README_NUMBERS["main"], README_NUMBERS) == []


def test_malformed_question_files_are_refused(tmp_path):
    bad = [
        {"id": "x", "type": "answerable", "role": "staff", "question": "q"},  # no expected_docs
        {"id": "x", "type": "restricted", "role": "staff", "question": "q"},  # no restricted_doc
        {"id": "x", "type": "trivia", "role": "staff", "question": "q"},
        {"id": "x", "type": "out_of_scope", "question": "q"},
    ]
    for item in bad:
        path = tmp_path / "q.jsonl"
        path.write_text(json.dumps(item) + "\n", encoding="utf-8")
        with pytest.raises(ValueError):
            load_questions(path)
    path.write_text("\n".join(json.dumps({"id": "same", "type": "out_of_scope", "role": "staff", "question": "q"})
                              for _ in range(2)), encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_questions(path)


def test_eval_command_prints_all_sets_and_fails_on_a_regression(tmp_path, embedder):
    done = run("-m", "kb", "eval", timeout=300)
    assert done.returncode == 0, done.stderr
    assert "Evaluation on the 64 demo questions written for this project (eval/questions.jsonl)" in done.stdout
    assert "Held-out set 1: 31 questions (eval/heldout.jsonl)" in done.stdout
    assert "Held-out set 2: 30 questions (eval/heldout_v2.jsonl)" in done.stdout
    assert done.stdout.count("Retriever: hybrid.") == 3 and done.stdout.count("restricted text found            0") == 3
    assert "No regressions against eval/baseline.json" in done.stdout
    better = json.loads((ROOT / "eval" / "baseline.json").read_text(encoding="utf-8"))
    better["heldout"]["answered"] += 1  # pretend it used to be better
    (tmp_path / "baseline.json").write_text(json.dumps(better), encoding="utf-8")
    worse = run("-m", "kb", "eval", "--baseline", str(tmp_path / "baseline.json"), timeout=300)
    assert worse.returncode == 1 and "heldout: answered went from 15 to 14" in worse.stdout
    assert "answer quotes the right sentence 8/10" in done.stdout


def test_ingest_command_adds_then_skips_unchanged_files(tmp_path):
    env = {"KB_DB": str(tmp_path / "kb.db"), "PYTHONPATH": str(ROOT), "KB_RETRIEVER": "bm25"}  # from another folder
    first = run("-m", "kb", "ingest", str(SAMPLES), cwd=tmp_path, env=env)
    assert first.returncode == 0 and "9 added (0 skipped)" in first.stdout
    assert "added     general/pt-200-product-manual       6 passages  PT-200 Pallet Tracker: User Manual" in first.stdout
    again = run("-m", "kb", "ingest", str(SAMPLES), cwd=tmp_path, env=env)
    assert "9 unchanged (0 skipped)" in again.stdout
    one = SAMPLES / "general" / "leave-policy.md"
    assert run("-m", "kb", "ingest", str(one), cwd=tmp_path, env=env).returncode != 0  # needs --collection
    single = run("-m", "kb", "ingest", str(one), "--collection", "archive", cwd=tmp_path, env=env)
    assert single.returncode == 0 and "added     archive/leave-policy" in single.stdout


def test_remote_commands_print_answers_sources_and_errors(server, capsys):
    base = ["--url", server.url, "--token", STAFF]
    assert cli.main(["ask", "How long is the PT-200 warranty?"] + base) == 0
    out = capsys.readouterr().out
    assert "Status: answered  (extractive: sentences quoted from the documents, nothing AI-written)" in out
    assert "[1] PT-200 Pallet Tracker: User Manual > 5. Warranty and support, page 4" in out
    assert cli.main(["search", "parental leave", "--k", "2"] + base) == 0
    assert "general/leave-policy#4" in capsys.readouterr().out
    assert cli.main(["docs"] + base) == 0
    listing = capsys.readouterr().out
    assert "general/office-faq" in listing and "hr/" not in listing
    assert cli.main(["audit"] + base) == 1
    assert "Error 403: admin role required" in capsys.readouterr().err
    assert cli.main(["audit", "--limit", "3", "--url", server.url, "--token", ADMIN]) == 0
    assert "maria (sales)" in capsys.readouterr().out
    assert cli.main(["ask", "x", "--url", "http://127.0.0.1:9", "--token", STAFF]) == 1
    assert "Could not reach the server" in capsys.readouterr().err


def test_printed_answers_cannot_fake_extra_lines(capsys):
    cli.print_answer({"status": "answered\n[2] fake source", "mode": "extractive", "answer": "Real\x1b[2J \u202eanswer",
                      "passages": [{"label": "Leave\nPolicy", "passage_id": "general/x#0\u202e"}],
                      "warnings": ["careful\nStatus: answered"]})
    out = capsys.readouterr().out
    assert "\x1b" not in out and "\u202e" not in out
    assert out.splitlines() == ["Status: answered [2] fake...  (extractive: sentences quoted from the documents, "
                                "nothing AI-written)", "", "  Real[2J answer", "", "Sources:",
                                "  [1] Leave Policy  (general/x#0)", "WARNING: careful Status: answered"]


def test_settings_are_checked(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no stray .env
    for name in ("KB_TOKENS", "KB_ROLES", "KB_ANSWER_MODE", "KB_DB", "KB_RETRIEVER", "KB_EMBED_MODEL"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="KB_TOKENS"):
        Settings.from_env()
    monkeypatch.setenv("KB_TOKENS", "{not json")
    with pytest.raises(SystemExit, match="not valid JSON"):
        Settings.from_env()
    monkeypatch.setenv("KB_TOKENS", json.dumps({"a-real-looking-token-0001": {"name": "x", "role": "staff"}}))
    monkeypatch.setenv("KB_ANSWER_MODE", "magic")
    with pytest.raises(SystemExit, match="KB_ANSWER_MODE"):
        Settings.from_env()
    monkeypatch.setenv("KB_ANSWER_MODE", "extractive")
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-6")  # a global variable for other tools: not read
    monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
    settings = Settings.from_env()
    assert settings.claude_effort is None and settings.claude_fallbacks is None
    assert settings.uses_placeholders is False and settings.db_path == "kb.db" and settings.model == "claude-sonnet-5-5"
    assert settings.retriever == "hybrid" and settings.embed_model == "BAAI/bge-small-en-v1.5"
    monkeypatch.setenv("KB_RETRIEVER", "semantic")
    with pytest.raises(SystemExit, match="KB_RETRIEVER must be 'hybrid'"):
        Settings.from_env()
    monkeypatch.setenv("KB_RETRIEVER", "BM25")
    assert Settings.from_env().retriever == "bm25"
    for tokens, roles, message in [
        ({"short": {"name": "x", "role": "staff"}}, None, "shorter than 16"),
        ({"a-real-looking-token-0001": {"name": "x", "role": "boss"}}, None, "unknown role 'boss'"),
        (["a-real-looking-token-0001"], None, "must be a JSON object"),
        ({"a-real-looking-token-0001": {"name": "x", "role": "staff"}}, {"staff": ["general", "Not Valid"]},
         "invalid collection name"),
        ({"a-real-looking-token-0001": {"name": "x", "role": "admin"}}, {"admin": ["general"]}, "admin role always"),
    ]:  # friendly messages, no tracebacks
        monkeypatch.setenv("KB_TOKENS", json.dumps(tokens))
        if roles:
            monkeypatch.setenv("KB_ROLES", json.dumps(roles))
        else:
            monkeypatch.delenv("KB_ROLES", raising=False)
        with pytest.raises(SystemExit, match=message):
            Settings.from_env()


@pytest.fixture
def no_server(monkeypatch, tmp_path):
    """Settings for `serve`, with uvicorn replaced so nothing really starts."""
    import uvicorn

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("KB_DB", str(tmp_path / "kb.db"))
    monkeypatch.setenv("KB_ANSWER_MODE", "extractive")
    monkeypatch.setenv("KB_RETRIEVER", "bm25")  # the hybrid start is tested on its own, with the model
    started = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: started.append(k))
    return started


def test_placeholder_tokens_are_never_served_publicly(monkeypatch, no_server):
    monkeypatch.setenv("KB_TOKENS", json.dumps({"change-me-staff-token-0000": {"name": "x", "role": "staff"}}))
    for args in (["serve", "--host", "0.0.0.0"], ["serve", "--allowed-host", "kb.example.com"]):
        with pytest.raises(SystemExit, match="Refusing to serve on a public address or hostname"):
            cli.main(args)
    assert no_server == []


def test_serve_checks_claude_settings_and_warns_about_an_empty_database(monkeypatch, no_server, capsys):
    monkeypatch.setenv("KB_TOKENS", json.dumps({"a-real-looking-token-0001": {"name": "x", "role": "staff"}}))
    monkeypatch.setenv("KB_ANSWER_MODE", "claude")
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(SystemExit, match="needs ANTHROPIC_API_KEY"):
        cli.main(["serve"])
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-local-test-not-a-real-key")
    monkeypatch.setenv("KB_CLAUDE_EFFORT", "turbo")
    with pytest.raises(SystemExit, match="KB_CLAUDE_EFFORT must be one of"):
        cli.main(["serve"])
    monkeypatch.setenv("KB_CLAUDE_MODEL", "claude-sonnet-4-6")
    monkeypatch.setenv("KB_CLAUDE_EFFORT", "xhigh")
    with pytest.raises(SystemExit, match="claude-sonnet-4-6 accepts KB_CLAUDE_EFFORT low, medium, high, max, not xhigh"):
        cli.main(["serve"])
    monkeypatch.setenv("KB_CLAUDE_EFFORT", "high")
    monkeypatch.setenv("KB_CLAUDE_FALLBACKS", "default")
    with pytest.raises(SystemExit, match="claude-sonnet-4-6 has no server-side fallback"):
        cli.main(["serve"])
    for name in ("KB_CLAUDE_MODEL", "KB_CLAUDE_EFFORT", "KB_CLAUDE_FALLBACKS"):
        monkeypatch.delenv(name)
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-opus-4-6")  # set globally for other tools
    monkeypatch.setenv("CLAUDE_EFFORT", "xhigh")
    assert cli.main(["serve"]) == 0
    err = capsys.readouterr().err
    assert "Claude model: claude-sonnet-5-5, effort low, fallback default." in err  # the generic names changed nothing
    assert "Note: ANTHROPIC_MODEL is set in this environment but not used; this project reads KB_CLAUDE_MODEL." in err
    assert "Note: CLAUDE_EFFORT is set in this environment but not used; this project reads KB_CLAUDE_EFFORT." in err
    monkeypatch.setenv("KB_ANSWER_MODE", "extractive")
    assert cli.main(["serve"]) == 0 and len(no_server) == 2
    assert "has no documents" in capsys.readouterr().err


def test_settings_can_come_from_an_env_file_and_the_real_environment_wins(monkeypatch, tmp_path):
    env_file = tmp_path / "settings.env"
    env_file.write_text('KB_TOKENS=\'{"a-real-looking-token-0001": {"name": "x", "role": "staff"}}\'\n'
                        "KB_CLAUDE_MODEL=claude-opus-5-5\nKB_CLAUDE_EFFORT=medium\nKB_RETRIEVER=bm25\n", encoding="utf-8")
    monkeypatch.setenv("KB_ENV_FILE", str(env_file))
    monkeypatch.setenv("KB_CLAUDE_EFFORT", "high")
    settings = Settings.from_env()
    assert (settings.model, settings.claude_effort, settings.retriever) == ("claude-opus-5-5", "high", "bm25")


def test_serve_with_the_hybrid_retriever_loads_the_model_first(monkeypatch, no_server, capsys, embedder, store):
    monkeypatch.setenv("KB_TOKENS", json.dumps({"a-real-looking-token-0001": {"name": "x", "role": "staff"}}))
    monkeypatch.setenv("KB_RETRIEVER", "hybrid")
    monkeypatch.setenv("KB_DB", store._path)  # a copy ingested without vectors: they are filled at start
    assert cli.main(["serve"]) == 0 and len(no_server) == 1
    assert "Retriever: hybrid, BAAI/bge-small-en-v1.5. Documents in the database: 9." in capsys.readouterr().err


def test_the_demo_runs_end_to_end_without_an_api_key(embedder):
    env = {k: v for k, v in os.environ.items() if not k.startswith("ANTHROPIC_")}
    done = subprocess.run([sys.executable, "demo.py"], cwd=ROOT, env={**env, "PYTHONIOENCODING": "utf-8"},
                          capture_output=True, text=True, encoding="utf-8", timeout=300)
    assert done.returncode == 0, done.stderr
    titles = [line for line in done.stdout.splitlines() if line.startswith("=== ")]
    assert len(titles) == 11 and titles[-1].startswith("=== 11. The audit log")
    assert "=== 4. Different words: found by meaning" in done.stdout
    assert "The same question with keyword search only -> not_found" in done.stdout
    assert "Traceback" not in done.stdout + done.stderr
    assert "A reply whose quote is not in the passage -> flagged" in done.stdout


def test_published_demo_and_test_tokens_are_never_served_publicly(monkeypatch, no_server):
    for token in ("demo-staff-token-00000001", "test-hr-token-000000000001"):
        monkeypatch.setenv("KB_TOKENS", json.dumps({token: {"name": "x", "role": "staff"}}))
        with pytest.raises(SystemExit, match="tokens published in this repository"):
            cli.main(["serve", "--host", "0.0.0.0"])
    assert no_server == []
