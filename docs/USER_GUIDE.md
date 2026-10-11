# User guide

This guide is for the people who ask the assistant questions, and for the person who adds
documents to it. You do not need to know how the system is built.

Words used here: a **document** is one file, such as the leave policy. A **passage** is a short
part of a document, usually one section. A **collection** is a group of documents with the same
readers, such as `general` (everyone), `hr` (the HR team) or `finance` (the finance team). Your
**token** is a long password that tells the system who you are and which collections you may read.

## What the assistant does

When you ask a question, the assistant:

1. looks only in the collections your token may read,
2. finds the passages that match your question, by meaning as well as by words (you do not
   need the document's exact wording),
3. answers with the matching sentences and tells you where each one comes from.

The assistant **only answers from the documents**. If they do not contain the answer, it says:

```
I could not find this in the documents you have access to.
```

If nothing in the documents matches your question well enough, it says so instead of
guessing, and it does not use general knowledge. It can still quote a passage on the same
topic that does not answer your exact question: asked "How many paid sick days am I entitled
to per year?", it quotes the 25 days of annual leave from the leave policy, and the answer is
still marked **Answered**. So always glance at the source. It also sometimes refuses a question
the documents do answer: ask again more directly (see "Common questions" below).

## Using the web page (the easiest way)

![The chat page](img/chat.png)

1. Open the address your administrator gave you in your browser (on a local try-out it is
   `http://127.0.0.1:8200`).
2. Paste your token and press **Connect**. The page shows your name, what you can read, and on
   the left the list of documents you may read.
3. Type your question at the bottom and press Enter.
4. The answer appears with small numbered boxes, such as **1**. Click a number (or the source
   under the answer) to open the passage on the right. The words used in the answer are
   highlighted, and you see the document, the section and, for PDFs, the page.

The page keeps your token only in this browser tab. When you close the tab, it is forgotten,
and you paste it again next time. To switch to another token, click **Sign out**: the page
also clears every answer shown so far, so the next person does not see them.

How the answers look:

| What you see | Meaning |
|---|---|
| A white card with a green edge, marked **Answered** | An answer with its sources. The label says whether it was quoted (Extractive) or written by Claude |
| A grey dashed card, **Not in the documents you can read** | No passage matched well enough. Nothing was guessed. The documents may still cover it: ask more directly |
| A yellow note inside an answer | A warning, for example that a document contains text addressed to AI assistants (it was ignored) |
| A card with an amber edge, **Withheld: failed the source check** | Claude's answer did not match its sources, so the matching sentences are shown instead |
| An orange card, **Answer service unavailable** | No answer right now; the matching passages are still listed |
| A red card, **Access denied** | Your token is not valid (any more). Ask your administrator for a new one |

On a phone, the list of documents is behind the **Documents** button at the top (**Docs** on
small screens). A source opens over the whole screen; close it with the **x** button.

If your company has connected the assistant to Claude Desktop or another AI app, you can ask
there too. That app can only read the documents your token allows, but it writes the final
wording itself, so check the sources it names.

## Using the command line

Your administrator sets these up once:

- the assistant is running, and its address is in a file named `.env` in your project folder,
- the same file holds your personal token.

Each time you use the command line, open a terminal (a command window) in the project folder
and switch on the Python environment:

```
.venv\Scripts\activate
```

On macOS or Linux: `source .venv/bin/activate`

### Asking a question

```
python -m kb ask "How many days of annual leave do full-time employees get?"
```

Put the question in double quotes. One question at a time; the assistant does not remember
earlier questions, so ask in full ("How long is parental leave for partners?", not "And for
partners?"). Questions can be up to 1000 characters.

### Reading an answer

```
Status: answered  (extractive: sentences quoted from the documents, nothing AI-written)

  "Full-time employees receive 25 days of paid annual leave per calendar year, plus public holidays." [1]

Sources:
  [1] Leave Policy > Annual leave  (general/leave-policy#1)
```

- The sentences in quotes are copied word for word from the document.
- The number in square brackets points to a source below. The source names the document, the
  section and, for PDFs, the page. The part in brackets is the passage id.
- Read the source if the answer matters. A sentence can be correct but about something
  slightly different from what you asked (for example, the error codes the manual lists, when
  you asked about a code it does not list).

If your company uses the Claude mode, the answer is written in plain sentences instead of
quotes, and the label says so. Every fact still has a numbered source, and every citation is
checked to quote the passage it names before you see it.

### What the statuses mean

| Status | Meaning | What you do |
|---|---|---|
| `answered` | An answer with its sources | Read it, and the source if it matters |
| `not_found` | Nothing you may read matched well enough | Ask more directly, or with the words your documents use, or ask the team that owns the topic |
| `flagged` | The written answer did not match its sources, so the matching passages are shown instead | Read the passages; the system has already withheld the doubtful text |
| `error` | The answer service is not available right now. The matching passages are still listed | Read the passages, or try again later |

### Lines to look for

| Line | Meaning |
|---|---|
| `WARNING: A document used here contains text addressed to AI assistants...` | Someone put an instruction for AI systems inside a document. The assistant ignored it. Tell the document's owner so they can remove it |
| `WARNING: The AI answer did not pass the source check...` | See `flagged` above |
| `WARNING: This answer is quoted from the documents because the limit...` | The limit of AI-written answers (per person per minute or per day, or for everyone per day) was reached, so this answer was quoted instead. It is still from the documents |

### Searching instead of asking

To see the passages that match some words, with the best match first:

```
python -m kb search "parental leave"
```

To see which documents you may read:

```
python -m kb docs
```

## Common questions

**The assistant says "not found", but I know the policy covers this.**
Ask more directly, one thing at a time: "How many days of annual leave do I get?" works better
than a long question with several parts. Using the words the document uses also helps. If it
still fails, the document may be in a collection you cannot read, or it may not have been added
yet. Ask your administrator.

**My colleague gets an answer and I do not.**
You probably have different roles. HR and finance documents are only visible to those teams.
The assistant does not tell you that a restricted document exists; it answers as if it did not.

**Can I get around this by asking in a clever way?**
No. Access is decided by your token before the search starts, not by the question. Asking "as an
admin" or naming a restricted document changes nothing.

**Is my question stored?**
Yes. Every question is recorded with your name, the time and which passages were used, so that
answers can be checked later. Do not put personal details in a question unless you need to.

**The command says "Could not reach the server".**
The assistant is not running, or the address in `.env` is wrong. Ask your administrator.

**The command says "missing or invalid bearer token".**
The token in `.env` is mistyped or no longer valid. Ask your administrator.

**The command says "No module named ...".**
The Python environment is not switched on. See "Using the command line".

## For the person who adds documents

You need the admin token and access to the documents folder on the server.

**Where documents go.** Inside the documents folder there is one folder per collection:

```
samples/docs/
    general/     everyone may read these
    hr/          only the hr and admin roles
    finance/     only the finance and admin roles
```

Use Markdown (`.md`), plain text (`.txt`) or PDF files with real text in them. Scanned PDFs
(pictures of pages) are refused, because they contain no text to search. Save Word files as PDF
first. Excel sheets and tables are not read well. Each file can be up to 5 MB.

Two files that differ only in punctuation or file type (`Travel Policy.md` and
`travel-policy.txt`) count as the same document: the second is skipped with a message. Rename
one of them.

**Adding or updating documents.** Copy the files into the right folder, then run:

```
python -m kb ingest
```

Each file is listed as `added`, `updated` or `unchanged`. Unchanged files are skipped, and an
updated file replaces its old version completely. Files that cannot be read are listed as
`skipped`, with the reason; their previous version, if any, stays in place.

The search by meaning uses a small model that runs on the server. The first time, it is
downloaded (about 67 MB, once); after that it works without internet. If the server has no
internet, your administrator can copy the model folder to it (see the README, "Search, and its
limits").

**Removing documents.** Delete the file, then run:

```
python -m kb ingest --prune
```

Documents whose file is gone are listed as `removed`.

**Writing documents that answer well.**

- Use clear headings. Each heading starts a new passage, and its name appears in the source.
- Write the words people will use in their questions. If people say "holiday", mention it.
- Keep one topic per section. A section about two things gives answers about both.
- In PDFs, number your headings ("3. Charging the battery") so they are recognised as headings.

**Who asked what.**

```
python -m kb audit --limit 50 --token <admin token>
```

Each line shows the time (UTC), who asked, their role, the channel (`http`, `mcp` or `cli`), the
action, the outcome and the question. `denied` lines show someone trying to read a document their
role may not read, or a non-admin trying an admin action.

The admin token reads every document and this log, which holds everyone's questions. Keep it to
the people who run the system.
