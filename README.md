# RAG Knowledge Assistant

A reference build: document Q and A for a company. It answers only from the company's own documents, shows the sentence and page behind every answer, and gives each person only the documents their role allows.

**Live demo: https://rag-knowledge-assistant-demo.vercel.app** (six steps, about a minute)

**Video walkthrough (1 minute): https://youtu.be/xf6qv67UPb0**

- Finds answers by meaning as well as keywords, with a small model on the server: no API cost for search.
- Says "not in your documents" instead of guessing; HR text never reaches sales, however they ask.
- Optional Claude mode: every sentence must cite a passage, checked in code, or it is withheld.
- Measured: on unseen questions the documents do answer, it answered 14 of 21 and 10 of 16. Every miss is listed.
- 391 tests, three question sets, a chat page, an HTTP API and an MCP server for AI tools.

Fictional company and documents. Not a client system.

![The chat page answering from the demo documents: an answer with its source, a question the documents do not answer, and a question in other words answered from a PDF with the source panel open on page 4](docs/img/chat.png)

*The chat page on the demo documents, signed in as Maria (sales, staff role): an answer with
its source, a question the documents do not cover, and a question in other words ("still
covered if it breaks") answered from the warranty section of a PDF manual, with the source
panel open on page 4 and the quoted sentences highlighted. Taken from the running app by
`docs/make_screenshots.py`.*

## What it does in 30 seconds

Your staff ask questions in plain English, such as "How many days of annual leave do I get?"
The assistant looks only in your company's own documents (policies, guides, manuals) and
answers with the sentences it found, plus the document, section and page they come from.

By default no AI model writes the answers: they are quoted word for word. An optional mode
has Claude write them, with every citation checked in code. That mode is tested with
simulated replies only; it has not been run on the live Claude API.

It finds passages by meaning as well as by keywords, so "Is the pallet scanner still covered if it
breaks after a year?" finds the warranty section. The model that compares meanings is small
and runs on the server itself: no API key and no cost per question.

If nothing in the documents matches the question well enough, it says so instead of guessing.
It makes mistakes both ways. On new test questions the documents do answer, it answered 14 of
21 and 10 of 16 (two sets, not used for tuning). On questions they do not answer, it sometimes
quotes the nearest passage instead of refusing: asked "How many paid sick days am I entitled to
per year?", it quotes the 25 days of annual leave from the leave policy. Such answers are still
marked "Answered"; the source shown under every quote is what lets a reader spot them (see
"Evaluation").

Each person sees only the documents their role allows. Someone in sales gets nothing from the
HR salary bands, however they phrase the question.

Every question is logged: who asked, what, and which documents were used.

AI tools such as Claude Desktop and Claude Code can also connect to it (through MCP, a standard
way for AI apps to use outside tools), under the same rules. That connection is tested with
the official MCP client, not yet inside those apps.

It runs here on nine short demo documents for a fictional company, Northwind Demo Ltd. It is
not a client system.

![A 40-second tour of the chat page: an answer with its source, the source panel, a question in other words answered from a PDF, a refusal, an HR question refused for staff and answered for HR](docs/img/tour.gif)

*A silent 40-second recording of the real chat page on the demo documents. The captions at
the top were added for the recording.*

## Live demo

A web demo of the same search, access rules and source check, for a quick look in a browser:
six short steps on the demo documents, then free use. You pick who you are: Maria (sales,
staff role), Hana (HR) or Farid (finance). It runs on Vercel (`api/index.py`, `vercel.json`):
https://rag-knowledge-assistant-demo.vercel.app

![The web demo: an answer with its footnote and the quoted sentence marked on the Leave Policy; a question in other words found on page 4 of a PDF; the salary bands refused for Maria and answered for Hana; a bad AI draft held; what you'd get on your own documents; a document with a hidden instruction.](docs/img/demo-tour.gif)

1. **A policy question.** Maria asks how much annual leave full-time staff get. The answer
   has footnote 1, which opens the Leave Policy with the quoted sentence marked.
2. **The same question in other words.** "Is the pallet scanner still covered if it breaks
   after a year?" finds the warranty section on page 4 of a PDF manual. Keyword search alone
   would have refused this question: too few of its words are in the manual.
3. **Who can see what.** Maria asks for the Band C salary range and gets "Not in your
   documents". Hana from HR asks the same and gets the answer with its source.
4. **A bad AI draft is held.** For this step the page forces a draft that says 30 days of
   leave and that unused days never expire. The policy says 25; the source check withholds
   the draft, and the real sentence is shown instead. This step makes no AI call.
5. **Where it still misses.** "How many paid sick days am I entitled to per year?" gets the
   annual leave sentence: a near miss, which the footnote makes easy to spot.
6. **What you'd get on your documents.** Answers that quote your own documents with the page;
   "Not in your documents" instead of a guess; 0 leaks of HR or finance text in 606 test
   questions. Then the held-out numbers (14 of 21 and 10 of 16), the measured cost, and how it
   was tested.

After that, free use: type any question, switch between Maria, Hana and Farid, choose answers
written by the AI or quoted word for word, and try the suggestions: the document with the
planted instruction, a question the documents do not answer, an error code the manual does
not list, and the finance payment runs as Maria and as Farid.

What the demo says about itself (the first line shows once its AI key is set; until then every
answer is quoted, and the card says so):

- Answers are written by Claude Haiku 5.5. Every sentence is checked against the passage it
  cites. When the demo's AI budget is used up, answers are quoted word for word, and the page
  says which.
- It refuses about a third of unseen everyday questions and sometimes quotes a near miss.
  Check the source.
- In this demo you pick who you are. In your version the role comes from your company sign-in.
- The company and its documents are fictional. There are no uploads.

| The first screen | Step 2: found by meaning | Step 3: who can see what |
|---|---|---|
| ![The first screen: Answers you can check, with an answer and its footnote, and a refusal](docs/img/demo-0-intro.png) | ![The scanner question answered from page 4 of the PDF manual; a plain word search would have refused it](docs/img/demo-2-meaning.png) | ![Hana's answer from the salary bands, Maria's refusal below it](docs/img/demo-3-access.png) |
| **Step 4: a bad draft held** | **Step 6: what you'd get** | **Step 2 on a phone** |
| ![The forced draft struck through with the two reasons, and the quoted sentence shown instead](docs/img/demo-4-held.png) | ![What you'd get: three takeaways, the held-out numbers and the cost of the run](docs/img/demo-6-cost.png) | ![Step 2 on a phone](docs/img/demo-2-phone.png) |

How it works:

- **The product itself, read-only.** Every answer goes through this product's own
  `KnowledgeBase` (access rules, thresholds, quoting, the source check) over a prebuilt copy
  of the sample documents in `demo/`, made by `scripts/build_demo_index.py`. The database is
  opened read-only, so nothing is written to disk, not even the audit log: each answer carries
  its own audit entries, and the card shows them. A footnote opens the whole document, and
  the server checks the persona may read it.
- **The meaning model without fastembed.** The server runs the same BAAI/bge-small-en-v1.5
  ONNX file (a copy in `demo/model`, 67 MB, MIT licence) with onnxruntime and a plain Python
  tokenizer (`kb/onnx_embed.py`). Tests check that its token ids are identical to the
  `tokenizers` package's, and its vectors match fastembed's (cosine similarity at least
  0.9999), on every passage, sentence and question. The questions the demo shows (the story,
  the suggestions and the three eval sets) have stored vectors (`demo/vectors.npz`); only
  typed questions use the model, which loads while the first screen is read.
- **The AI.** Claude Haiku 5.5 at effort low, at most 2,000 output tokens, a 20 second
  timeout and no retries: one call per question that clears the bar, through the same source
  check as Claude mode. Each AI answer shows its cost, from the API's own token counts.
  Limits: 20 AI answers per visitor per 10 minutes and 600 per server instance per hour; the
  hard limit is the spend cap on the key. Over a limit, without a key, or when the API reports
  a spend limit, a rate limit, a key problem, a timeout or an outage, the answer is quoted
  word for word and the card says why.
- **Keywords only.** With `KB_DEMO_SEARCH=keywords` the server never loads the model: typed questions
  are searched by keywords only (as with `KB_RETRIEVER=bm25`), and every such answer says so.
  The story and the suggestions still use their stored vectors. The same happens on its own
  if the model files are missing. Typed questions are read up to 128 tokens and have their own
  limit (40 per visitor per 10 minutes); over it they are searched by keywords, and the card says so.
- Code: `kb/demo_web.py` (the HTTP API and its limits), `kb/demo_index.py` (the prebuilt
  index, read-only storage, the personas and the questions), `kb/onnx_embed.py`, `public/`
  (the page, plain HTML, CSS and JavaScript).

Run it locally:

```bash
pip install -r requirements-dev.txt
python -m kb.demo_web            # then open http://127.0.0.1:8210
```

Without `ANTHROPIC_API_KEY` the answers are quoted word for word. With a key in the
environment it calls the live API; `KB_DEMO_AI=off` switches that off. After changing a sample
document, the search code or the demo's questions, run `python scripts/build_demo_index.py`
(the tests fail until it is rebuilt). On Vercel, set `ANTHROPIC_API_KEY` (a key with a spend
cap) for Production. `python docs/make_media.py` records the tour and the screenshots above.

**Bundle size.** Deployed with the model: 238 MB of Vercel's 250 MB limit (210 MB without .pyc
files). If a change goes over, set `KB_DEMO_SEARCH=keywords` and add `demo/model/**` to
`excludeFiles`; typed questions then use keywords only, and the page says so.
`api/requirements.txt` holds only what the demo needs (no fastembed, MCP SDK or pypdf), and
`vercel.json` leaves the tests, docs, samples, eval sets and the page itself out of the function. `python scripts/bundle_size.py`
installs the Linux wheels for Python 3.12 and measures what would ship (Vercel allows 250 MB per
function, unzipped):

```
                                                 model  no model
dependencies (api/requirements.txt)            170.4 MB   170.4 MB
  without the .pyc files pip compiles          141.5 MB
  of which onnxruntime (with protobuf)          71.3 MB
  after excludeFiles, if Vercel applies it     131.0 MB
project files (after excludeFiles)              68.0 MB     0.8 MB
  of which the model (demo/model)               67.2 MB         -
total                                          238.4 MB   171.2 MB
  without .pyc files                           209.6 MB
  if excludeFiles applies to packages too      199.0 MB   131.8 MB
no model, and onnxruntime removed as well                 99.9 MB
limit                                            250 MB     250 MB
```

Adding `demo/model/**` to `excludeFiles` is a change by hand, because `vercel.json` cannot read
environment variables.

| What was run | How | Result |
|---|---|---|
| The demo's API: the story, personas, fallbacks, keyword-only search, limits, headers | `pytest tests/test_demo_web.py tests/test_demo_index.py` | 82 tests pass; no question built from an HR or finance document shows its text to another role (606 asked) |
| Invented claims next to real citations | `pytest tests/test_source_check.py tests/test_answer.py` | the planted claim withheld in 7 of 7 forms; 28 of 30 invented answers made of a passage's own words and numbers withheld; 18 of 19 legitimate rewordings pass, and every eval answer restated as Claude cites it |
| The demo reproduces the eval | `tests/test_demo_index.py`, on `demo/` without the model | the three sets give exactly the numbers in `eval/baseline.json` |
| The model without fastembed | `pytest tests/test_onnx_embed.py` | token ids identical on 352 texts; cosine similarity at least 0.9999 on every passage, sentence and question |
| The story in a browser | `pytest tests/test_demo_ui.py`, Chrome at 1440x900, 1280x720, 390x844 and 360x640 | 17 tests pass; at every step the footnote, stamp, note or card it points at is fully on screen |
| Claude Haiku 5.5 answers | a simulated API, and the real SDK against a local fake server | marked as written by Claude, priced from the token counts; spend limit, rate limit, key, timeout and outage fall back to quoted answers |
| **Claude Haiku 5.5 live, on the story and the three sets** | the deployed demo with its key | Spot check only, 12 Oct 2026: questions answered by Claude on the live demo, citations checked, about $0.0002 each. The full run on the story and the three sets is not done yet. |

---

## The parts

A reference build, made to be read and run.

| Part | Folder | What it is |
|---|---|---|
| Knowledge base | `kb/` | Ingestion, search, answers, access rules and the audit log (SQLite). |
| Server | `kb/api.py`, `kb/mcp_server.py`, `kb/web/` | One server with the chat page, an HTTP API for apps, and an MCP endpoint for AI tools. |
| Web demo | `kb/demo_web.py`, `kb/demo_index.py`, `kb/onnx_embed.py`, `public/`, `demo/`, `api/` | The six step browser demo for Vercel, over a prebuilt read-only index (see "Live demo"). |
| Demo documents | `samples/docs/` | Leave, expenses, IT security, onboarding, an office FAQ and a product manual (PDF) for everyone; salary bands for HR; two finance documents (one PDF). |
| Question sets | `eval/` | 64 questions written with the system, and two held-out sets (31 and 30 questions), written separately and not used for tuning, for `python -m kb eval`. |

## The idea in one picture

```
 documents                          question + bearer token
 general/  hr/  finance/            (chat page, HTTP API, MCP or CLI)
      |                                          |
      v                                          v
 +-------------+                     +-----------------------+
 |   ingest    |  passages with      |  token -> role ->     |
 | .md .txt    |  title, section,    |  collections it may   |
 | .pdf        |  page, collection,  |  read                 |
 |             |  and a vector       |                       |
 +------+------+                     +-----------+-----------+
        |          +-----------+                 |
        +--------->|  SQLite   |---------------->|
                   +-----------+                 v
                             +------------------------------------+
                             | search by keywords (BM25) and by   |
                             | meaning (local embeddings), over   |
                             | the allowed collections only       |
                             +-----------------+------------------+
                                               |
             nothing relevant enough?  ------> "not in the documents" (no model is asked)
                                               |
                                               v
                             +------------------------------------+
                             | answer: quote the best sentences   |
                             | (default, no model) or one Claude  |
                             | call with citations                |
                             +-----------------+------------------+
                                               |
                                               v
                             +------------------------------------+
                             | source check (Claude mode), in     |  fails: the model's text
                             | code: every statement needs a      |  is withheld, the passages
                             | citation of a retrieved passage    |  are shown, "flagged"
                             +-----------------+------------------+
                                               |
                                               v
                                answer + sources  ----->  audit log
```

What the code enforces:

1. **Access rights come before search.** Each token has a role, and each role may read
   certain collections (the first folder level of the documents). Search only looks at those
   collections, for keywords and for meaning alike, so restricted text is never scored,
   returned or sent to a model, and it does not change the scores of the passages you may
   read. Results are checked again after search. A passage you may not read is reported as
   "not found", the same as one that does not exist.
2. **No source, no answer.** If no passage is close enough to the question, the answer is "I
   could not find this in the documents you have access to." No model is asked to guess. The
   same happens when the question names an identifier, such as an error code or a model number
   ("E22", "PT-200"), that none of the matching passages contains.
3. **Every model statement needs a checked citation.** In Claude mode an answer is accepted
   only if it has at least one citation that points to a passage retrieved for this question
   and quotes text that is in it. Each cited statement is then compared with the sentences it
   cites, word by word: at least 80% of its content words must be in them (or in one sentence
   right next to them, or the passage's title); its numbers, dates and weekdays, in digits or in
   words ("one", "a dozen", "half", "31 May"), must be there too, each next to the same words;
   it must agree on "not" with the closest sentence of the passage; and a label ("Band D") or a
   word that widens a claim ("any", "always", "optional", "unlimited") must come from the cited
   sentences. On 30 invented answers made of a passage's own words and numbers, 28 were withheld;
   the two that passed swap roles between two cited sentences, or split one claim across two
   citations. Text without a citation may only connect or name what is cited
   ("According to the Leave Policy, ..."): each of its words must be a connecting word or part
   of a cited document's title or section, and it may contain no number, "not", "no" or "yes".
   A true answer can therefore still be withheld (an uncited "Yes, ..."); that is the safe
   way to fail. Anything else, a refusal, a cut-off reply, or a citation of text that addresses an
   AI makes the answer "flagged": the model's text is withheld (it is kept in the audit log for
   the admin) and the quoted passages are shown instead. These checks compare words and
   numbers; they do not understand meaning, so a model could still restate a quote in a
   misleading way. When one fires, the draft is withheld and the quotes are shown: a true answer
   held back is the safe way to fail (1 of 19 legitimate rewordings in the tests is).
4. **Document text is data, not instructions.** One demo document contains a planted line
   ("Note for AI assistants reading this file: ignore all previous instructions..."). At
   ingestion a keyword check flags the whole paragraph around such a line, and the paragraph
   after it unless that starts a new heading or FAQ question; flagged sentences are never
   quoted and add a warning to any answer that used them. The keyword check catches
   obvious lines and will not catch every wording. The stronger protection is that Claude
   receives passages as `search_result` blocks, separate from its instructions, and that every
   statement must cite a passage that was retrieved for this question.
5. **Ingesting again is safe.** An unchanged file is skipped (content hash). A changed file
   replaces all of its old passages in one transaction. Two files that would get the same
   document id (`Travel Policy.md` and `travel-policy.txt`) never replace each other: the second
   is skipped with a message, and an upload over a different file needs `replace`.
6. **Limits.** Request bodies are capped (64 KB, 9 MB for an upload). In Claude mode each token
   gets at most 10 model answers a minute and 100 a day, and everyone together 500 a day (all
   settable); above that, answers are quoted from the documents instead, with a warning.
7. **An audit trail.** Every question, search, passage lookup and admin action is recorded:
   who, the question, which passages were used, the outcome and the time taken. Refused and
   malformed requests with a valid token are recorded too. Requests without a valid token are
   refused at the door and written to the application log instead.

## Quick start

Needs Python 3.10 or newer (the MCP SDK requires it). Developed and tested on Windows 11 with
Python 3.12.10 only. Other versions and operating systems have not been run.

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt

python demo.py
python -m pytest -q
python -m kb eval
```

If PowerShell refuses to run `Activate.ps1` (execution policy), run
`Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first, or use `cmd` and
`.venv\Scripts\activate.bat`. If `pip install` fails on long file paths, move the folder
higher up (for example `C:\work\rag-knowledge-assistant`) or enable long paths in Windows.

macOS / Linux (bash):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt

python demo.py
python -m pytest -q
python -m kb eval
```

`requirements.txt` is enough to run the assistant; `requirements-dev.txt` adds the test tools.

**The embedding model comes with the repository.** The clone is about 75 MB: `demo/model` holds
the 67 MB model the web demo runs, and the product loads the same file from there (after
checking its SHA-256) instead of downloading it again. Without that folder, the first run
downloads it into the `models` folder (see "Search, and its limits"; `python -m kb
download-model` does it ahead of time), and the Hugging Face library may warn that no
`HF_TOKEN` is set: that can be ignored, the model is public. With `KB_RETRIEVER=bm25` (keywords
only) no model is needed.

`python demo.py` needs no API key. It copies the sample documents to a throwaway folder,
starts the server on a free local port, and prints eleven short scenes: loading the documents
(a second run skips unchanged files, an edited file is replaced), the token check, a question
answered with its source, a question in other words found by meaning (keyword search alone
finds nothing for it), two questions refused, access rights (HR text stays with HR), an answer
from a PDF with its page number, the planted instruction, the same rules over MCP, the source
check on two simulated model replies, and the audit log. Add `--pause` to stop after each
scene (useful when recording).

The test suite has 391 tests. 25 drive a real browser and are skipped if Playwright or an
installed Chrome or Edge is missing. 26 use the embedding model through fastembed (the copy in
`demo/model`, or the `models` folder) and are skipped, with the reason shown, only if neither
can be loaded; the web demo's tests run the copy in `demo/model` directly, and the other tests
use keyword search or stored vectors.

## Running the pieces yourself

Copy `.env.example` to `.env` (`copy` on Windows, `cp` elsewhere). The `change-me` tokens work
for a local try-out; the server prints a warning while they, or the demo and test tokens
published in this repository, are in use, and refuses to serve on a public address or
hostname with them.

```bash
python -m kb ingest          # reads KB_DOCS_DIR (samples/docs) into KB_DB (kb.db), with a vector per passage
python -m kb serve           # chat page, HTTP API and MCP endpoint on http://127.0.0.1:8200
```

If you start the server with `--port`, change `KB_URL` in `.env` to match, so the command line
finds it. The server warns at start if the database has no documents.

### The chat page

Open `http://127.0.0.1:8200` in a browser and paste a token from `.env`, for example
`change-me-staff-token-00000` (staff) or `change-me-hr-token-000000000` (HR). Ask a question;
click a number to see the passage it comes from, with the quoted text highlighted.

<img src="docs/img/chat-phone.png" alt="The chat page on a phone" width="260" align="right">

- The page is one static file (`kb/web/index.html`, plain HTML and JavaScript, no build step).
  It holds no document text and no data; everything comes from the API, with the token.
- The token is kept in the browser tab only (`sessionStorage`) and is forgotten when the tab
  closes. It never goes into a URL. Signing out, or a token that stops working, clears every
  answer and source shown so far.
- Answers, "not in the documents", a withheld answer, an unavailable service and "access
  denied" each look different. Every answer shows whether it was quoted (extractive) or
  written by Claude.
- Document text is always shown as text, never run as code. Source numbers come only from
  checked citations, never from text such as "[2]" inside a document. The page's security
  policy only lets its own script run.
- It was tested in Chrome at 1440, 1280 and 390 pixels wide, and its header at 360 and 320.

### The command line

In a second terminal (with the virtual environment active):

```bash
python -m kb ask "How many days of annual leave do full-time employees get?"
python -m kb search "parental leave"
python -m kb docs
python -m kb ask "What is the salary range for Band C?" --token change-me-hr-token-000000000
python -m kb audit --token change-me-admin-token-000000
```

`search` lists passages in the order the search ranks them (keywords and meaning merged); the
number in front is the similarity in meaning only, so a lower number can sit above a higher one.
`ask`, `search`, `docs` and `audit` use `KB_URL` and `KB_TOKEN` from `.env` unless you pass
`--url` or `--token`. All commands read `.env` from the current folder; real environment
variables win over the file. Audit times are UTC. Stop the server with Ctrl+C.

Documents live in `<KB_DOCS_DIR>/<collection>/<files>`. Run `python -m kb ingest` again after
adding or editing files. Add `--prune` to also remove documents whose file was deleted
(uploaded documents are never pruned). One file can be added with
`python -m kb ingest path/to/file.md --collection hr`.

### Answers: quotes or Claude

By default, answers are **extractive**: the sentences from the retrieved passages that best
match the question, quoted word for word, each with its source. No language model writes
anything and no API key is needed; the sentences are chosen by meaning with the same local
embedding model as the search, with a bonus for the question's words. It cannot summarise or
combine, and it can quote a loosely related sentence next to the right one, which is why the
sources are always shown.

With `KB_ANSWER_MODE=claude` and `ANTHROPIC_API_KEY` set, each question that clears the
relevance bar becomes one Claude API call. Up to three passages are sent as `search_result`
blocks with citations enabled, so every cited statement is tied to a passage, and the reply
goes through the source check above. Settings are checked when the server starts.

- Model `claude-sonnet-5-5` at effort `low` by default (`KB_CLAUDE_MODEL`, `KB_CLAUDE_EFFORT`):
  quoting and citing short passages does not need the largest model.
- Effort and Anthropic's server-side fallback (a retry on another model when a request is
  declined; documented for Claude Fable 5.1, Opus 5.5, Opus 5 and Sonnet 5.5) are only sent to
  models that accept them; `KB_CLAUDE_FALLBACKS=off` turns the fallback off. When a fallback
  model answers, the audit log says which one.
- Credentials: `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` only. The SDK's other sources
  (`ant` profiles, workload identity) are not checked or used; the server says so at start.
- The settings are checked when the server starts: an effort the model does not accept (for
  example `xhigh` on `claude-sonnet-4-6`) or a fallback for a model without one stops it with a
  message. The server prints the model, effort and fallback it will use.
- Only the project's own `KB_CLAUDE_*` names are read. `ANTHROPIC_MODEL` or `CLAUDE_EFFORT` are
  often set globally for other tools (a Claude Code shell sets `CLAUDE_EFFORT`, for example);
  they are ignored, with a note at start if they are set. `ANTHROPIC_API_KEY` keeps its usual
  name. The `anthropic` SDK itself also reads `ANTHROPIC_BASE_URL`; when it is set, the server
  prints the address it will call.
- 60-second timeout and one retry. A failed call is logged with its status code and request
  id, and the user gets "service unavailable" with the matching passages, never a guess.
- Questions that do not clear the bar never reach the API.
- If an answer fails the source check, the quotes shown instead are picked the same way as in
  extractive mode (by meaning, with the hybrid search).

Search and extractive answers cost nothing per question: the embedding model runs on the
server's CPU. Only Claude mode has a per-question cost.

Estimated cost per answered question in Claude mode (an estimate from published prices, not
measured):

| Model | Price per million tokens (input / output) | 1,000 in + 300 out | per 1,000 questions |
|---|---|---|---|
| `claude-haiku-5-5` | $0.10 / $0.50 | $0.00025 | about $0.25 |
| `claude-sonnet-5-5` (default) | $2 / $10 | $0.005 | about $5 |
| `claude-opus-5-5` | $4 / $20 | $0.010 | about $10 |

Assumptions: prices as listed by Anthropic in October 2026 (check current pricing); 1,000 input
tokens per question (the demo's requests measure about 1,400 to 2,200 characters, roughly 350
to 550 tokens; real documents with full-size passages are longer); 300 output tokens for the
answer and the model's thinking. Thinking can use more at higher effort.

**The Claude call has not been run against the live API.** It is tested with simulated replies,
and one test runs the real `anthropic` SDK against a local fake server to check the request it
sends and the parsing of the reply. `python demo.py --claude` and `python -m kb eval --mode
claude` use the live API; neither has been run by the author.

### Connecting an MCP client

Any MCP client that supports Streamable HTTP and a custom header can connect:

- URL: `http://127.0.0.1:8200/mcp`
- Header: `Authorization: Bearer <a token from KB_TOKENS>`

For example, with Claude Code (not yet tried inside Claude Code):

```bash
claude mcp add --transport http company-docs http://127.0.0.1:8200/mcp --header "Authorization: Bearer <token>"
```

For a local client that starts servers itself, such as Claude Desktop, use stdio. The client
then reads with the role of `KB_MCP_TOKEN`. On Windows the Claude Desktop settings file is
`%APPDATA%\Claude\claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "company-docs": {
      "command": "C:\\path\\to\\rag-knowledge-assistant\\.venv\\Scripts\\python.exe",
      "args": ["-m", "kb", "mcp-stdio"],
      "env": {
        "PYTHONPATH": "C:\\path\\to\\rag-knowledge-assistant",
        "KB_ENV_FILE": "C:\\path\\to\\rag-knowledge-assistant\\.env",
        "KB_DB": "C:\\path\\to\\rag-knowledge-assistant\\kb.db"
      }
    }
  }
}
```

The tools are `search_documents`, `get_passage`, `ask` and `list_documents`, all read-only.
See [docs/TOOLS.md](docs/TOOLS.md). The stdio command is tested by starting it as a
subprocess with the MCP Python SDK client; it has not been tried inside Claude Desktop itself.
An AI app connected this way only gets passages the token may read, but it writes its own
final wording, so the source check above does not apply to that wording.

## Roles

| Role | Collections it may read | Add and remove documents, read the audit log |
|---|---|---|
| `staff` | general | no |
| `hr` | general, hr | no |
| `finance` | general, finance | no |
| `admin` | all, always | yes |

Tokens are set in `KB_TOKENS`, roles in `KB_ROLES` (both JSON). A request without a known token
gets `401` before it reaches any route. The role is applied again on every request. A new
collection is invisible to every role except `admin` until you add it to a role.

`admin` means full read access: the admin role reads every collection and the audit log, which
holds everyone's questions and the answers that were withheld. It cannot be limited to some
collections (the server refuses such a setting). Give it only to the people who run the system.

## Answer statuses

| Status | Meaning |
|---|---|
| `answered` | An answer with numbered sources. Each source has the exact quoted text. |
| `not_found` | No passage the caller may read is relevant enough, or the model said the answer is not in them. |
| `flagged` | Claude mode: the reply failed the source check. The quoted passages are shown instead; the withheld text is kept in the audit log. |
| `error` | Claude mode: the API call failed. No answer; the matching passages are listed. |

In the audit log, refused requests (bad input) are `rejected`, admin actions by other roles
and lookups of restricted passages are `denied`.

## HTTP endpoints

| Endpoint | Who | Purpose |
|---|---|---|
| `GET /` | public | the chat page (static: no data in it; it calls the API with your token) |
| `GET /healthz` | public | liveness check |
| `GET /me` | any role | your name, role and the collections you may read |
| `POST /ask` | any role | `{"question": "..."}` (up to 1000 characters) |
| `POST /search` | any role | `{"query": "...", "k": 5}` passages with their similarity to the query and keyword coverage |
| `GET /documents` | any role | the documents your role may read |
| `GET /passages/{id}` | any role | one passage, for example `/passages/general/leave-policy%231` |
| `POST /admin/ingest` | admin | add or replace one document: `collection`, `filename`, `text` or `content_base64`, and `replace` (default false) |
| `DELETE /admin/documents/{doc_id}` | admin | remove a document |
| `GET /admin/audit?limit=50` | admin | latest audit entries |
| `POST /mcp` | any role | MCP (Streamable HTTP, stateless) |

## Search, and its limits

Documents are split into passages of about 700 characters that follow their own headings and
PDF pages. A question is compared with every passage the caller may read, in two ways:

- **by keywords**: BM25 ranking, with the synonym list in `kb/synonyms.txt` (a short starter
  list such as "holiday, leave, vacation"; edit it, or switch it off with `KB_SYNONYMS=off`);
- **by meaning**: the cosine similarity between the question and the passage, computed by a
  small embedding model that runs on the server.

The two rankings are merged (reciprocal rank fusion). A passage may be used for an answer if
its similarity to the question is at least 0.72, or at least 0.65 when it also contains
nearly all (90%) of the question's words, rare words counting more. Below that, the answer is
"not in the documents". These numbers were chosen on the 64 main questions only (see
"Evaluation"); they depend on the model and should be checked on your own documents.

**The embedding model.**

| | |
|---|---|
| Model | `BAAI/bge-small-en-v1.5` (MIT licence), 384 numbers per text, English |
| Runs on | the CPU, with ONNX Runtime, through the `fastembed` package. No API, no GPU |
| Download | none in a full clone: `demo/model` holds the quantised copy (`Qdrant/bge-small-en-v1.5-onnx-Q` on Hugging Face, 67 MB), used after a SHA-256 check. Otherwise once, on first use |
| Kept in | `demo/model` (in git, for the web demo), else the `models` folder of the project (not in git) or the folder in `KB_MODEL_DIR`; the `models` folder comes first when it has the model |
| Download ahead of time | `python -m kb download-model` (for example before going offline, or to copy the folder to a server without internet) |
| Offline | once downloaded, it loads from that folder without a network (tried with the network blocked) |
| Without the model and without a network | `ingest`, `serve`, `eval` and the demo stop at start with a message saying what to do, after the download attempts time out (about 45 seconds here; at once with `HF_HUB_OFFLINE=1`) |
| Keywords only | `KB_RETRIEVER=bm25`: nothing to download; the earlier rule applies (the passage must contain 60% of the question's words, and a minimum score) |

A vector is stored for each passage in SQLite (table `vectors`). It is computed at ingest and
written in the same transaction as the passage, replaced when the document changes and removed
with it. Passages without a vector for the current model (a database built with `bm25`, or a
new model) get one when the server starts.

Speed on the development PC, with the 50 demo passages: loading the model about 1.4 s,
ingesting the nine documents about 1.6 s, a question about 0.16 s (search and choosing the
sentences). Similarities are computed in memory for each request over the caller's passages:
expect this to suit thousands of passages, not millions. It has not been measured at volume.

**The model cannot change under the stored vectors.** After the first download the model is
loaded from its folder only, without asking Hugging Face for a newer revision, and every vector
is stored with a fingerprint of the model file (`BAAI/bge-small-en-v1.5@<12 hex digits>`).
Different model files, a new model, or a damaged vector mean the vectors are computed again.

**Identifiers.** If the question names an identifier (a word that starts with a letter and has
a digit: an error code, a model number, a SKU, a policy number), only passages that contain it,
in their text, title or section, may be used. "What does error code E22 mean on the PT-200
scanner?" is refused, because the manual lists E12, E31 and E47 only; without this rule it
quoted "E12: the scanner window is dirty." Times, dates and quarters (6:30am, 24th, Q3) are not
treated as identifiers.

Known limits, all visible in the held-out evaluation:

- **It still refuses some natural questions.** A question in everyday words often lands just
  under the 0.72 bar even when the right passage is the top result. On the two held-out sets
  the right document was among the top three for every answerable question, but 7 of 21 and 6
  of 16 were still refused. A lower bar was not chosen: on the main set, 0.70 answered no more
  questions and refused one fewer out-of-scope question and one fewer restricted one.
- **Near misses are answered more often than with keywords alone.** A question on a
  document's topic that the document does not answer is close in meaning, so it can get the
  nearest passage quoted instead of "not in the documents", and the card still says
  "Answered". "How many paid sick days am I entitled to per year?" gets the 25 days of annual
  leave. On held-out set 1, 4 of its 6 out-of-scope questions were answered this way (keyword
  search alone: 2); on set 2, 1 of 8 (keyword search alone: 0).
- **It can quote the wrong sentence or document.** On held-out set 2, one answer cited a
  different document from the one that answers the question, and one quoted the right
  document but the wrong sentence (the close deadline instead of the bank reconciliation).
- **With `KB_RETRIEVER=bm25`** the old limit stays: "How many paid volunteering days per year?"
  gets the annual leave sentence, because "paid", "days" and "year" match. The hybrid search
  refuses it.

The usual next steps, not built here: a reranker (a second, slower model that reads the
question and each candidate passage together) for the answer/refuse decision, or Claude mode,
which is told to reply `NOT_IN_DOCUMENTS` when the passages do not answer (spot checked live on the demo; the full sets not run live yet).

One keyword-only change was measured and left off: limiting the weight of question words that
appear in no document (`unseen_cap` in `kb/retrieval.py`).

## Evaluation

```bash
python -m kb eval                    # hybrid search, the default
python -m kb eval --retriever bm25   # keyword search only
```

It runs three question sets and compares the numbers with `eval/baseline.json`; it exits with
an error if any number got worse or if restricted text appears in any result. "Restricted
text found" is checked on the text itself, read straight from the database, so it does not
depend on the access filter it is testing (a test switches the filter off and checks that the
metric catches it).

Each answerable question lists key phrases from the sentence that answers it, taken from the
documents. "Answer quotes the right sentence" counts answers with a quote that contains one of
them; "cites the right document" only checks the document.

How the sets were used:

1. **Main set, 64 questions** (`eval/questions.jsonl`), written with the system. The
   thresholds of the hybrid search and the sentence choice were chosen on this set only, so it
   is optimistic.
2. **Held-out set 1, 31 questions** (`eval/heldout.jsonl`): 24 written by a separate assistant
   given only the documents, and 7 from the reviewers of this project. Run first with keyword
   search, then with hybrid search once its thresholds were fixed. Not used for tuning.
3. **Held-out set 2, 30 questions** (`eval/heldout_v2.jsonl`): written by another separate
   assistant given only the documents (not the thresholds, not the other questions), after the
   thresholds were fixed. Run with hybrid search first; run once with keyword search
   afterwards, with no tuning. One of its questions happens to be word for word the same as one
   in set 1.

The key phrases were added to all three sets after those runs; they measure, and nothing was
tuned on them. Two changes made after the held-out runs moved their numbers, and both are
rules, not thresholds: the identifier rule above, added after a set 2 question about error code
E22 was answered with the E12 sentence (so that question is no longer an unseen test; set 2's
out-of-scope refusals went from 6 to 7 of 8), and the served app now picks quotes by meaning
like the eval (the numbers below did not change from it). The tests run all three sets again
to check that the numbers stay as reported.

Real output, main set (hybrid search):

```
Answerable questions (37)
  retrieval hit@1                  37/37  100.0%
  retrieval hit@3                  37/37  100.0%
  answered                         36/37   97.3%
  answer cites the right document  36/36  100.0%
  answer quotes the right sentence 36/36  100.0%
Out-of-scope questions (17)
  refused (not in the documents)   16/17   94.1%
Access-restricted questions (10)
  refused (not in the documents)   10/10  100.0%
  restricted text found            0
```

Held-out set 1:

```
Answerable questions (21)
  retrieval hit@1                  20/21   95.2%
  retrieval hit@3                  21/21  100.0%
  answered                         14/21   66.7%
  answer cites the right document  14/14  100.0%
  answer quotes the right sentence 14/14  100.0%
Out-of-scope questions (6)
  refused (not in the documents)   2/6     33.3%
Access-restricted questions (4)
  refused (not in the documents)   4/4    100.0%
  restricted text found            0
  by source: fresh     answered 9/16, out of scope refused 2/4, restricted refused 4/4
  by source: reviewer  answered 5/5, out of scope refused 0/2, restricted refused 0/0
```

Held-out set 2:

```
Answerable questions (16)
  retrieval hit@1                  15/16   93.8%
  retrieval hit@3                  16/16  100.0%
  answered                         10/16   62.5%
  answer cites the right document  9/10    90.0%
  answer quotes the right sentence 8/10    80.0%
Out-of-scope questions (8)
  refused (not in the documents)   7/8     87.5%
Access-restricted questions (6)
  refused (not in the documents)   5/6     83.3%
  restricted text found            0
```

The same sets with keyword search only (`--retriever bm25`):

| Set | Answered | Right sentence | Out of scope refused | Restricted refused | Restricted text |
|---|---|---|---|---|---|
| Main | 36 of 37 | 36 of 36 | 15 of 17 | 10 of 10 | 0 |
| Held-out 1 | 6 of 21 | 5 of 6 | 4 of 6 | 4 of 4 | 0 |
| Held-out 2 | 3 of 16 | 3 of 3 | 8 of 8 | 6 of 6 | 0 |

How to read this:

- **More unseen questions are answered**: 14 of 21 instead of 6 on set 1 (the fresh questions
  alone: 9 of 16 instead of 2), and 10 of 16 instead of 3 on set 2. About a third of answerable
  questions in everyday words are still refused.
- **Near misses got worse.** On set 1, out-of-scope questions refused fell from 4 of 6 to 2 of
  6. On set 2, one of its 8 out-of-scope questions (7 of them near misses) was answered: how
  often passwords must be changed (the policy gives a minimum length, not a schedule). Keyword
  search refused all 8. The main set shows the opposite (15 to 16 of 17), but the thresholds
  were chosen on it.
- **Wrong document or sentence**: on set 2, one answer cited a different document, and one
  quoted the right document but not the sentence that answers the question.
- **Access held on every set**: no restricted text appeared in any result. One restricted
  question on set 2 (a staff member asking about a finance procedure) was answered from a
  general document instead of refused; no finance text was shown.
- **Every miss is listed** by the command, with the question, the top results, the status and
  the first quote.

The expected numbers for all three sets, with both kinds of search, are copied into a test, so
a change that moves them fails until the numbers in the test and in this README are updated.

## What is in the box

```
kb/ingest.py          Markdown, text and PDF -> passages with title, section, page; content hashes
kb/store.py           SQLite: documents, passages, passage vectors, audit log
kb/embeddings.py      the local embedding model (fastembed, BAAI/bge-small-en-v1.5): loading, download, vectors
kb/retrieval.py       the Retriever interface; hybrid (BM25 + embeddings, fused) and BM25 alone
kb/synonyms.txt       a starter list of workplace synonyms
kb/answer.py          extractive answers; Claude answers with citations; the source check
kb/service.py         access rules, thresholds, the model budget and the audit log around every operation
kb/api.py             FastAPI HTTP API, the token and size gate, the chat page and the MCP endpoint on one port
kb/web/index.html     the chat page: one file, plain HTML and JavaScript
kb/mcp_server.py      MCP tools: search_documents, get_passage, ask, list_documents
kb/evaluate.py        hit@k, refusal and restricted-text metrics
kb/cli.py             python -m kb ingest | serve | mcp-stdio | eval | download-model | ask | search | docs | audit
kb/onnx_embed.py      the same embedding model on onnxruntime alone, with a plain Python tokenizer (web demo)
kb/demo_index.py      the web demo's prebuilt index: read-only storage, stored vectors, personas, questions
kb/demo_web.py        the web demo's HTTP API: ask, passage, config, warm; Claude Haiku 5.5 or quoted answers
public/               the web demo's page (plain HTML, CSS and JavaScript)
demo/                 the prebuilt index (kb.db, vectors.npz, manifest.json) and the model (model/)
api/                  the Vercel entry point and its slim requirements file
scripts/              build_demo_index.py (rebuilds demo/), bundle_size.py (what Vercel would ship)
samples/docs/         9 demo documents in 3 collections (2 PDFs, made by samples/make_pdfs.py)
eval/questions.jsonl  64 questions: 37 answerable, 17 out of scope (4 near misses), 10 access-restricted
eval/heldout.jsonl    held-out set 1: 31 questions, 24 fresh, 7 from the reviewers
eval/heldout_v2.jsonl held-out set 2: 30 questions (16 answerable, 8 out of scope, 6 restricted)
                      (every answerable question has key phrases from the sentence that answers it)
eval/baseline.json    the numbers `kb eval` compares with
models/               the downloaded embedding model (created on first use, not in git)
docs/USER_GUIDE.md    for the people who ask questions, and for whoever adds documents
docs/TOOLS.md         every MCP tool with its arguments (generated by docs/gen_tools.py)
docs/img/             screenshots of the chat page (docs/make_screenshots.py) and the web demo (docs/make_media.py)
tests/                391 tests; 63 start a server or a command and use real HTTP, MCP, stdio or a browser
```

## What the tests cover, and what they do not

Covered (`python -m pytest -q`): parsing Markdown, text and PDF (headings, pages, page footers,
front matter, links); passage sizes; files that are refused (scanned PDF, broken PDF, binary,
too large, unsupported type, unreadable, unsafe paths, long names, invalid collection names,
symbolic links); skipping unchanged files, replacing changed ones, pruning deleted ones (but not
uploads); two files with the same document id; the tokenizer, synonyms and sentence rules;
BM25 ranking and the refusal threshold; scores that are identical whether or not restricted
documents exist, for keyword and hybrid search (with the real model and with a stand-in);
the hybrid search loading nothing outside the caller's collections; a faulty retriever that
ignores the filter (its results are dropped); a plugged-in retriever. Passage vectors: written
at ingest, replaced on update, removed with the document, filled in when missing, kept per
model. With the real model: a question in other words answered by meaning (and refused by
keywords alone), the right sentence quoted for six demo questions, refusals and HR text still
kept with HR, similarity over HTTP and MCP, an uploaded document embedded and answerable, the
ingest and download commands; the served app picking the same quotes as the eval on all 64
main questions; an identifier the documents do not contain refused; the model loaded from its
folder without a network, with its fingerprint; the vector cache following the passage ids; a
damaged vector computed again; and a clear message when the model is missing. For every heading, sentence and word of the HR and finance documents, a
search and a question by each role that may not read them: no restricted passage and none of
the restricted figures come back. Per-role documents and passage lookups, a new collection,
custom roles, token rules, friendly setting errors. Extractive answers (exact quotes, no FAQ
questions, no injected text including a two-line one, the same answers with and without the
injected document). The Claude request (`search_result` blocks, citations on, model, effort and
fallback only where supported, the injected line only inside a search result) and the source
check (ten kinds of bad citation; a reply with no citation; an uncited "yes", even before a
quote that agrees; claims cut into short uncited pieces; uncited numbers; an uncited claim made
of the cited passage's own words that reverses it; the planted line's payload, uncited, next to
a harmless citation from the same file; a cited sentence that adds a number or a "not" to its
quote; an invented claim next to a real citation, in the same block or its own (7 planted
forms, 30 invented answers, of which 2 still pass and are marked so); and, accepted, 19
rewordings from a review (1 still withheld, marked) and 18 more, every eval answer restated as
Claude cites it, an attribution such as "According to the Leave Policy" and connecting
words between cited claims; refusal, cut-off and empty replies; "not in the
documents"; a citation of the injected sentence; an API failure, logged with status and
request id; a fallback model recorded; the model budget per token, per token per day and in
total). The HTTP API
(tokens on every route, bad input, oversized and streamed bodies, admin rules, uploads
including a PDF, path tricks and replace, 20 questions at once, audit entries, the hostname
allow-list, MCP behind a hostname with a port and an Origin header). The MCP tools over
Streamable HTTP and over stdio. The chat page: served without a token but with a strict
security policy, no document text in it, every API call it makes refused without a token; and,
in Chrome, sign-in, answers and the source panel, "not in the documents" and "access denied"
shown differently, sign-out and a withdrawn token clearing earlier answers, an answer still
on its way when the person signs out never shown to the next one, the hidden source panel
emptied on sign-out, only real
citations becoming numbers, document HTML shown as text, and the header on small phones. The
command line, the settings, the eval numbers in this README, and the demo itself.

Not covered:

- **The live Claude API call.** `ClaudeAnswerer` is tested with simulated replies and with the
  real SDK against a local fake server. It has not been run against the real API, so the source
  check has not seen a real reply.
- **Real documents.** Nine short, clean files. Real PDFs have columns, tables, headers and
  scanned pages. The PDF heading rule (a numbered line such as "3. Charging the battery") will
  not fit every document.
- **Retrieval quality on your documents.** See "Evaluation" and "Search, and its limits".
- **Load.** No performance tests beyond 20 questions at once, and none beyond the 50 demo
  passages for the embedding search.
- **Claude Desktop and Claude Code themselves.** The MCP server was tested with the MCP Python
  SDK client, not inside those apps.
- **Other browsers.** The chat page was tested in Chrome only. Other browsers were not tried.
- **Other systems.** Only Windows 11 with Python 3.12.10 was run.

## What this is not

It is a starting point: ingestion, search, answers with sources, access rules, the audit trail
and a simple chat page. Not included: follow-up questions (each question stands alone), user
accounts and sign-in pages (people paste a token), connectors to SharePoint, Google Drive or
Confluence (documents are files in a folder), Word, Excel or HTML files, OCR for scanned PDFs,
tables inside PDFs, a reranker, languages other than English (the embedding model, stemmer,
stopwords and synonyms are English), per-person permissions on single documents (access is per
collection), single sign-on, hosting and TLS.

## Before using this for real

- Put the server behind HTTPS (a reverse proxy or a platform that terminates TLS). Start it
  with `python -m kb serve --host 0.0.0.0 --allowed-host your.domain`; requests for any other
  hostname are refused on every route except `/healthz`.
- Generate long random tokens, one per person and per agent, and keep `.env` out of git.
- For staff using the chat page, a pasted token is a stopgap. Connect it to your company sign-in
  (single sign-on) so people do not handle tokens at all.
- Map your permission groups to collections. If your permissions are set per document rather
  than per folder, sync documents into collections that match them, or extend the passage
  record with its own access list.
- The audit log stores every question as it was asked, and the answers that were withheld.
  Questions can contain personal data. Decide who may read the log and how long to keep it.
- Passage ids change when a document changes, and the audit log keeps ids, not the quoted text.
  If you must later show exactly what an answer quoted, keep old document versions.
- Measure on your own questions before you trust it: write 50 to 100 real questions with the
  document that answers each, in the format of `eval/questions.jsonl`, run
  `python -m kb eval --questions yours.jsonl --docs your-folder`, and set the similarity bar from
  the results (`KnowledgeBase(min_similarity=...)`). Keep some questions back, unseen, to check
  the choice. For near misses, consider a reranker or Claude mode.
- Claude mode: run the eval with `--mode claude` and read real replies before relying on it,
  and set `KB_CLAUDE_PER_MINUTE` and `KB_CLAUDE_PER_DAY` to match your budget. The limits are
  kept in memory and reset when the server restarts.
- SQLite is fine for one server. For several, move `kb/store.py` to Postgres (and the model
  budget to a shared store).

## Adapting it to another system

1. **Documents:** sync your document system into one folder per collection (for example with a
   scheduled export) and run `python -m kb ingest --prune` after each sync, or call
   `POST /admin/ingest` from your own sync job.
2. **Access:** map groups to collections in `KB_ROLES` and issue one token per person or agent.
   For single sign-on, replace `TokenTable` in `kb/config.py` with a check of your identity
   provider's tokens.
3. **Search:** another embedding model supported by fastembed can be set with `KB_EMBED_MODEL`
   (passages get new vectors at the next start); choose the thresholds again with the eval, as
   they depend on the model. A different search (a vector database, a reranker) implements
   `Retriever.search` in `kb/retrieval.py` with the same contract (look only at the collections
   you are given) and is passed to `KnowledgeBase(retriever=...)`. The tests in
   `tests/test_access.py` and `tests/test_hybrid.py` show what must stay true. Edit
   `kb/synonyms.txt` for your own vocabulary.
4. **File types:** add a parser in `kb/ingest.py` that returns sections with a heading and a page.
5. **MCP tools:** after changing `kb/mcp_server.py`, run `python docs/gen_tools.py`.
