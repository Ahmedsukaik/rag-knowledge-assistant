/* RAG Knowledge Assistant demo page.
   A six step story (one action at a time), then free use. The server keeps nothing between
   requests and writes nothing: every answer comes back with its sources and its own audit
   entries, and this page keeps them in the tab (sessionStorage) so a reload keeps your place.
   Document text is always escaped before it is shown. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const STEPS = 6;
  const STORE_KEY = "rag-demo-v2";  // a new version of the page starts fresh instead of reading an old shape
  const COLL = { general: "General", hr: "HR", finance: "Finance" };
  const S = {
    config: null, step: 0, persona: "maria", answers: [], next: 1, story: {}, open: null, docs: {},
    busy: false, wait: "", pending: null, pane: "answer", ai: true, keyboard: false, deskNote: null, fresh: 0,
  };

  const icon = (name, cls = "i") => `<svg class="${cls}" aria-hidden="true" focusable="false"><use href="#${name}"/></svg>`;
  const esc = (t) => String(t ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const isPhone = () => window.matchMedia("(max-width: 899px)").matches;
  const money = (n) => "$" + (n < 0.01 ? Number(n || 0).toFixed(5) : Number(n).toFixed(4));
  const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  const listing = (items) => items.length < 2 ? (items[0] || "") : items.slice(0, -1).join(", ") + " and " + items[items.length - 1];
  const persona = (id) => (S.config?.personas || []).find((p) => p.id === id) || { id, name: id, team: "", role: "", collections: [] };
  const byId = (id) => S.answers.find((a) => a.id === id);
  const Q = () => S.config?.story || {};
  const live = () => !!S.config?.live;
  const store = {
    get(k) { try { return JSON.parse(sessionStorage.getItem(k)); } catch { return null; } },
    set(k, v) { try { sessionStorage.setItem(k, JSON.stringify(v)); } catch { /* private mode: fine */ } },
  };
  function save() {
    store.set(STORE_KEY, { step: S.step, persona: S.persona, answers: S.answers.slice(0, 12), next: S.next, story: S.story,
      open: S.open, ai: S.ai, deskNote: S.deskNote });
  }

  let toastTimer = null;
  function toast(msg, ms = 4500) {
    const el = $("#toast"); el.textContent = msg; el.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => { el.hidden = true; }, ms);
  }
  let announceTimer = null;
  function announce(msg) {
    const el = $("#announce"); el.textContent = ""; clearTimeout(announceTimer);
    announceTimer = setTimeout(() => { el.textContent = msg; }, 60);
  }
  async function api(method, path, body) {
    const res = await fetch(path, { method, headers: body ? { "Content-Type": "application/json" } : {}, body: body ? JSON.stringify(body) : undefined });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) { const err = new Error(data.message || "Something went wrong. Try again."); err.status = res.status; throw err; }
    return data;
  }

  // ------------------------------------------------------------------ asking
  function waitText(p) {
    const who = persona(p).name;
    return live() && S.ai ? `Searching ${who}'s documents, then Claude Haiku 5.5 writes the answer (usually 2 to 5 seconds).`
      : `Searching ${who}'s documents.`;
  }
  // meta.restricted: the answer is in a document this persona may not open (the story's salary
  // question for Maria, the "as Maria" chip). Typed questions never say so: that would reveal it.
  async function ask(question, p, extra = {}, meta = {}) {
    question = String(question || "").trim();
    if (!question || S.busy) return null;
    S.persona = p;
    setBusy(true);
    S.pending = { question, persona: p };
    S.wait = extra.simulate ? "Sending the question with the forced draft." : waitText(p);
    if (isPhone()) setPane("answer");
    render();
    reveal(".pending-card");
    try {
      const data = await api("POST", "/api/ask", { question, persona: p, ai: S.ai, ...extra });
      const a = { ...data, id: S.next++, step: S.step, restricted: !!meta.restricted };
      S.answers.unshift(a);
      S.answers = S.answers.slice(0, 12);
      S.fresh = a.id;
      announce(`${a.persona.name}: ` + (a.status === "not_found" ? "Not in your documents." : a.status === "flagged"
        ? "The AI draft was held; the matching sentences are quoted instead." : "Answered, with " + count(new Set(a.citations.map((c) => c.number)).size, "source", "sources") + "."));
      return a;
    } catch (err) {
      toast(err.message);
      return null;
    } finally {
      S.pending = null; S.wait = ""; setBusy(false); save();
    }
  }
  function setBusy(b) {
    S.busy = b;
    document.body.classList.toggle("busy", b);
    document.querySelectorAll("#story-actions button, .chip, #send, .persona").forEach((x) => { if (b) x.disabled = true; });
  }
  async function landed(a) {
    if (!a) { render(); return; }
    S.open = null; S.deskNote = null;
    if (a.status === "not_found") S.deskNote = { id: a.id };
    render();
    if (!isPhone() && a.citations.length) await openSource(a.id, a.citations[0].number, { auto: true });
    requestAnimationFrame(revealTarget);
  }

  // ------------------------------------------------------------------ the source document
  async function openSource(id, n, opts = {}) {
    const a = byId(id);
    const passage = a && a.passages[n - 1];
    if (!passage) return;
    const key = `${a.persona.id}|${passage.passage_id}`;
    if (isPhone() && !opts.auto) setPane("source");  // at once: the page follows as soon as it arrives
    if (!S.docs[key]) {
      try {
        S.docs[key] = await api("GET", `/api/passage?id=${encodeURIComponent(passage.passage_id)}&persona=${encodeURIComponent(a.persona.id)}`);
      } catch (err) { toast(err.message); return; }
    }
    S.open = { id, n, key }; S.deskNote = null;
    save();
    renderAnswers(); renderDoc();
    if (!opts.quiet) {
      const mark = $("#desk mark.hl.active");
      const desk = $("#desk");
      if (mark && isPhone()) reveal(mark, "center");
      else if (mark) {  // the top of the page when the marked sentence fits there, else the sentence in the middle
        desk.scrollTop = 0;
        if (mark.getBoundingClientRect().bottom > desk.getBoundingClientRect().bottom - 48) mark.scrollIntoView({ block: "center" });
      } else if (isPhone()) window.scrollTo(0, 0);
    }
  }
  function squashed(text) {
    let s = ""; const at = [];
    for (let i = 0; i < text.length; i++) if (!/\s/.test(text[i])) { s += text[i]; at.push(i); }
    return { s, at };
  }
  function findRange(text, quote) {
    if (!quote) return null;
    const exact = text.indexOf(quote);
    if (exact >= 0) return [exact, exact + quote.length];
    const t = squashed(text), q = quote.replace(/\s+/g, "");
    const i = q ? t.s.indexOf(q) : -1;
    return i < 0 ? null : [t.at[i], t.at[i + q.length - 1] + 1];
  }
  function markup(segment, base, ranges) {
    const cuts = ranges.map((r) => ({ s: Math.max(r.start, base) - base, e: Math.min(r.end, base + segment.length) - base, r }))
      .filter((c) => c.e > c.s).sort((x, y) => x.s - y.s);
    let out = "", at = 0;
    for (const c of cuts) {
      if (c.s < at) continue;
      out += esc(segment.slice(at, c.s));
      const inner = esc(segment.slice(c.s, c.e));
      if (c.r.kind === "planted") out += `<span class="planted">${inner}</span>`;
      else out += `<mark class="hl${c.r.active ? " active" : " soft"}">${inner}</mark>` + (c.r.active && c.r.end <= base + segment.length ? `<span class="mark-fn" aria-hidden="true">${c.r.n}</span>` : "");
      at = c.e;
    }
    return out + esc(segment.slice(at));
  }
  function passageHtml(text, ranges) {
    let html = "", inList = false, pos = 0;
    for (const line of text.split("\n")) {
      const start = pos; pos += line.length + 1;
      if (!line.trim()) continue;
      const bullet = /^\s*[-*]\s+/.exec(line);
      const offset = bullet ? bullet[0].length : 0;
      const inner = markup(line.slice(offset), start + offset, ranges);
      if (bullet) { if (!inList) { html += "<ul>"; inList = true; } html += `<li>${inner}</li>`; continue; }
      if (inList) { html += "</ul>"; inList = false; }
      html += `<p>${inner}</p>`;
    }
    return html + (inList ? "</ul>" : "");
  }
  function emptyDesk(text) {
    return `<div class="desk-empty"><div><div class="ghost-page" aria-hidden="true"><i class="t"></i><i></i><i></i><i class="hl"></i><i></i><i></i><i></i><b>1</b></div>
      <p>${esc(text)}</p></div></div>`;
  }
  function refusalDesk(a) {
    const p = a.persona;
    return `<div class="desk-empty"><div class="page refusal-page"><span class="stamp">Not in your documents</span>
      <p>Nothing in the documents ${esc(p.name)} can open answers this question, so nothing is quoted${live() ? " and no AI is asked" : ""}.</p>
      ${a.restricted ? `<p>A document ${esc(p.name)} may not open is treated exactly like one that does not exist.</p>` : ""}</div></div>`;
  }
  function renderDoc() {
    const box = $("#desk");
    if (S.step === 0 && isPhone()) { box.innerHTML = `<div class="desk-preview">${specimenFront()}</div>`; return; }
    const a = S.open && byId(S.open.id);
    const data = S.open && S.docs[S.open.key];
    if (!a || !data) {
      const refused = S.deskNote && byId(S.deskNote.id);
      box.innerHTML = refused ? refusalDesk(refused) : emptyDesk(S.step === 6
        ? "Every number on this page comes from the eval command in the repository, run on the same index as this demo."
        : "Sources open here. Each footnote shows the exact sentence an answer quotes, marked on the page it comes from.");
      return;
    }
    const doc = data.document;
    const cited = new Set(a.citations.filter((c) => c.number === S.open.n).map((c) => c.passage_id));
    const page = (a.passages[S.open.n - 1] || {}).page;
    let html = `<article class="page" aria-label="Source document: ${esc(doc.title)}">
      <div class="page-meta">${esc(COLL[doc.collection] || doc.collection)}${page ? ` &middot; page ${page}` : ""}</div>
      <h2 class="page-title">${esc(doc.title)}</h2>
      <p class="page-from">Opened from footnote ${S.open.n} of ${esc(a.persona.name)}'s answer. Only documents ${esc(a.persona.name)} may open are shown here.</p>`;
    let pageNo = null, section = null;
    for (const p of doc.passages) {
      if (p.page && p.page !== pageNo) { html += `<div class="page-break">Page ${p.page}</div>`; pageNo = p.page; }
      if (p.section && p.section !== section && p.section !== doc.title) { html += `<h3>${esc(p.section)}</h3>`; }
      section = p.section;
      const ranges = [];
      a.citations.forEach((c) => {
        if (c.passage_id !== p.passage_id) return;
        const r = findRange(p.text, c.quote);
        if (r) ranges.push({ start: r[0], end: r[1], active: c.number === S.open.n, n: c.number });
      });
      (p.flagged || []).forEach((f) => { const r = findRange(p.text, f); if (r) ranges.push({ start: r[0], end: r[1], kind: "planted" }); });
      html += `<div class="passage${cited.has(p.passage_id) ? " cited" : ""}" data-passage="${esc(p.passage_id)}">${passageHtml(p.text, ranges)}</div>`;
      if ((p.flagged || []).length) {
        html += `<div class="planted-note">${icon("alert")}<span>Flagged when this file was read in: the underlined text is addressed to an AI. It is treated as document text, never quoted, and any answer that uses this passage carries a warning.</span></div>`;
      }
    }
    box.innerHTML = html + "</article>";
  }

  // ------------------------------------------------------------------ answer cards
  function footnotes(a, numbers) {
    return numbers.map((n) => {
      const p = a.passages[n - 1];
      const current = S.open && S.open.id === a.id && S.open.n === n;
      return `<button type="button" class="fn" data-open="${a.id}:${n}" aria-label="Source ${n}: ${esc(p ? p.label.replace(/ > /g, ", ") : "")}"${current ? ' aria-current="true"' : ""}>${n}</button>`;
    }).join("");
  }
  function answerBody(a) {
    const prose = a.written_by === "claude" && a.status === "answered";
    return (a.parts || []).map((part) => {
      const text = String(part.text || "").replace(/^\n+/, "");
      if (prose) return `<span>${esc(text).replace(/\n/g, "<br>")}</span>${footnotes(a, part.sources || [])}`;
      const inner = text.replace(/^"/, "").replace(/"$/, "");
      return `<span class="quote">&ldquo;<span class="qt">${esc(inner)}</span>&rdquo;${footnotes(a, part.sources || [])}</span>`;
    }).join("");
  }
  function sourcesList(a) {
    const numbers = [...new Set(a.citations.map((c) => c.number))].sort((x, y) => x - y);
    if (!numbers.length) return "";
    return `<ol class="sources" aria-label="Sources">${numbers.map((n) => {
      const p = a.passages[n - 1] || {};
      const [title, ...rest] = String(p.label || "").split(" > ");
      return `<li><button type="button" class="src" data-open="${a.id}:${n}"><span class="n" aria-hidden="true">${n}</span><span class="where"><span class="coll">${esc(COLL[p.collection] || p.collection || "")}</span><b>${esc(title)}</b>${rest.length ? " &rsaquo; " + esc(rest.join(" > ")) : ""}</span></button></li>`;
    }).join("")}</ol>`;
  }
  function whoNote(a) {
    const p = a.persona;
    const all = Object.keys(S.config?.collections || COLL);
    const can = p.collections.map((c) => COLL[c] || c);
    const cannot = all.filter((c) => !p.collections.includes(c)).map((c) => COLL[c] || c);
    const refused = a.status !== "not_found" ? "" : a.restricted
      ? " A document this person may not open is treated exactly like one that does not exist."
      : ` Nothing in the documents ${esc(p.name)} can open answers this.`;
    return `<aside class="margin-note"><span class="eyebrow">${icon("lock")}Who can see this</span>Asked as <b>${esc(p.name)}</b> (${esc(p.team)}).
      ${esc(p.name)} can open ${esc(listing(can))} documents${cannot.length ? `, not ${esc(listing(cannot))}` : ""}.${refused}</aside>`;
  }
  function refusalHint(a) {
    if (a.status !== "not_found" || S.step !== "free") return "";
    const titles = (S.config?.documents || []).filter((d) => a.persona.collections.includes(d.collection)).map((d) => d.title);
    return `<p class="hint">${icon("page", "i i-sm")}<span>${esc(a.persona.name)} can open: ${esc(titles.join("; "))}. Try naming the document or product.</span></p>`;
  }
  function byLabel(a) {
    if (a.written_by === "claude") return a.status === "flagged" ? "Written by Claude, held" : "Written by Claude Haiku 5.5, checked";
    if (a.written_by === "quoted") return "Quoted word for word";
    if (a.written_by === "simulated") return "Forced test draft";
    return "";
  }
  function kwNote(a) {
    // Not on the story's near miss (step 5): there the sentence it found is the wrong one.
    if (a.search.keywords_alone !== false || a.status === "not_found" || a.step === 5) return "";
    return `<p class="kw-note">${icon("search", "i i-sm")}<span>A plain word search would have refused this. Matched by meaning.</span></p>`;
  }
  function tags(a) {
    const out = [];
    if (a.written_by === "claude") {
      out.push(`<span class="tag ai">${icon("spark")}Written by Claude Haiku 5.5</span>`);
      if (a.status === "answered") out.push(`<span class="tag ok">${icon("check")}Every sentence checked against its source</span>`);
    } else if (a.written_by === "quoted") out.push(`<span class="tag">${icon("quote")}Quoted word for word, no AI</span>`);
    else if (a.written_by === "simulated") out.push(`<span class="tag test">${icon("flask")}Forced test draft: no AI call</span>`);
    if (a.status !== "not_found") out.push(`<span class="tag">${icon("search")}${a.search.by === "meaning" ? "Matched by meaning" : "Matched by keywords only"}</span>`);
    if (a.written_by === "claude" && a.cost) {
      out.push(`<span class="tag">${icon("coin")}Cost ${money(a.cost.usd)} (${a.cost.input_tokens.toLocaleString("en-US")} tokens in, ${a.cost.output_tokens.toLocaleString("en-US")} out)</span>`);
    }
    return `<div class="meta">${out.join("")}</div>`;
  }
  function auditList(a) {
    const rows = (a.audit || []).map((e) => {
      const time = String(e.ts || "").slice(11, 19);
      const used = (e.passages || []).length ? " &middot; " + esc(e.passages.join(", ")) : "";
      return `<li>${esc(time)} UTC &middot; ${esc(e.actor)} (${esc(e.role)}) &middot; ${esc(e.action)}: ${esc(e.outcome)}${used}${e.detail ? " &middot; " + esc(e.detail) : ""} &middot; ${esc(e.duration_ms)} ms</li>`;
    }).join("");
    return rows ? `<details class="audit"><summary>${icon("list", "i i-sm")}Audit log for this question: ${count((a.audit || []).length, "entry", "entries")}${icon("chev-down", "i i-sm chev")}</summary><ul>${rows}</ul></details>` : "";
  }
  function card(a) {
    const p = a.persona;
    const fresh = a.id === S.fresh ? " fresh" : "";
    const past = S.step === "free" && a.id !== S.answers[0]?.id ? " past" : "";
    let body;
    if (a.status === "not_found") {
      body = `<div class="refusal"><span class="stamp">Not in your documents</span><p>${esc(a.answer)}</p></div>`;
    } else if (a.status === "flagged") {
      const draft = a.draft ? `<blockquote class="draft"><del>${esc(a.draft.replace(/ ?\[\d+\]/g, ""))}</del></blockquote>` : "";
      body = `<div class="held" role="note"><div class="held-head">${icon("shield")}Held: the AI's draft failed the source check</div>
        <p class="held-sub">Nobody saw this draft. It is shown here only because this is a demo; in your version it goes to the audit log for the admin.</p>${draft}
        <ul class="problems">${(a.problems || []).map((x) => `<li>${icon("alert")}<span>${esc(x)}</span></li>`).join("")}</ul></div>
        <p class="eyebrow shown-instead">Shown instead, quoted word for word</p><div class="a-body">${answerBody(a)}</div>`;
    } else {
      body = `<div class="a-body">${answerBody(a)}</div>`;
    }
    const warn = (a.warnings || []).map((w) => `<p class="warn">${icon("alert")}<span>${esc(w)}</span></p>`).join("");
    const why = a.quoted_because ? `<p class="why">${esc(a.quoted_because)}</p>` : "";
    const note = a.search.note ? `<p class="why">${esc(a.search.note)}</p>` : "";
    const by = byLabel(a);
    return `<article class="card answer${fresh}${past}" data-card="${a.id}" data-status="${esc(a.status)}" aria-label="${esc(p.name)}'s question">
      <header class="a-head"><span class="avatar ${esc(p.id)}" aria-hidden="true">${esc(p.name[0])}</span><div><span class="eyebrow">${esc(p.name)} asked${by ? `<span class="by"> &middot; ${esc(by)}</span>` : ""}</span><p class="a-q">${esc(a.question)}</p></div></header>
      ${body}${kwNote(a)}${warn}${sourcesList(a)}${tags(a)}${why}${note}${whoNote(a)}${refusalHint(a)}${auditList(a)}</article>`;
  }
  function pendingCard() {
    const p = persona(S.pending.persona);
    return `<article class="card pending-card" aria-busy="true"><header class="a-head"><span class="avatar ${esc(p.id)}" aria-hidden="true">${esc(p.name[0])}</span><div><span class="eyebrow">${esc(p.name)} asked</span><p class="a-q">${esc(S.pending.question)}</p></div></header>
      <div class="pending"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>${esc(S.wait)}</div></article>`;
  }
  function storyAnswerIds() {
    const st = S.story;
    return ({ 1: [st.leave], 2: [st.scanner], 3: [st.salaryMaria, st.salaryHana], 4: [st.draft], 5: [st.sick] })[S.step] || [];
  }
  function reportCard() {
    const cfg = S.config || {};
    const claude = S.answers.filter((a) => a.written_by === "claude");
    const total = claude.reduce((sum, a) => sum + (a.cost?.usd || 0), 0);
    const price = cfg.price || { input: 0.1, output: 0.5 };
    const estimate = (1000 * price.input + 300 * price.output) / 1e6;
    const figures = claude.length
      ? `<div class="figure cost-figure"><span>Measured, this run</span><strong>${money(total)}</strong><small>${count(claude.length, "answer", "answers")} written by Claude Haiku 5.5, priced from the API's own token counts</small></div>
         <div class="figure"><span>1,000 answers like these</span><strong>about $${(total / claude.length * 1000).toFixed(2)}</strong><small>Search and quoting run on the server: no cost per question</small></div>`
      : `<div class="figure cost-figure"><span>This run</span><strong>$0</strong><small>Every answer was quoted word for word: no AI call was made.</small></div>
         <div class="figure"><span>With Claude on</span><strong>about $${(estimate * 1000).toFixed(2)}</strong><small>per 1,000 answers. An estimate (1,000 tokens in, 300 out at Haiku 5.5 prices), not measured.</small></div>`;
    const e = cfg.eval || {};
    const sets = [["main", "Tuning set", "64 questions, used to set the thresholds"], ["heldout", "New questions A", "31 questions, never used for tuning"],
      ["heldout_v2", "New questions B", "30 questions, written separately"]];
    const rows = sets.filter(([k]) => e[k]).map(([k, name, about]) => {
      const x = e[k];
      return `<tr><th scope="row">${name}<small>${about}</small></th><td class="num" data-label="Answered">${x.answered} of ${x.answerable}</td>
        <td class="num" data-label="Refused, not in the documents">${x.out_of_scope_refused} of ${x.out_of_scope}<sup>${k === "heldout" ? "*" : ""}</sup></td>
        <td class="num" data-label="Refused, another role's documents">${x.restricted_refused} of ${x.restricted}</td>
        <td class="num" data-label="Restricted text shown">${x.restricted_text_found}</td></tr>`;
    }).join("");
    const one = e.heldout, two = e.heldout_v2;
    const honest = one && two ? `On new questions the documents do answer, it answered ${one.answered} of ${one.answerable} and ${two.answered} of ${two.answerable}. It refuses about a third of unseen everyday questions and sometimes quotes a near miss. Check the source.` : "";
    return `<article class="card report" aria-labelledby="report-title"><h3 id="report-title" class="sr-only">What you would get</h3>
      <ul class="takeaways">
        <li>${icon("check-circle")}<span>Answers quote your own documents, with the page.</span></li>
        <li>${icon("check-circle")}<span>Says "Not in your documents" instead of guessing.</span></li>
        <li>${icon("check-circle")}<span>HR and finance text never reached the wrong person: 0 leaks in ${Number(cfg.leak_checks || 0).toLocaleString("en-US")} test questions.</span></li>
      </ul>
      ${honest ? `<p class="honest">${esc(honest)}</p>` : ""}
      <h3 class="report-sub">What this run cost</h3><div class="figures">${figures}</div>
      <details class="tested"><summary>How it was tested${icon("chev-down", "i i-sm chev")}</summary>
      <table class="eval"><thead><tr><th scope="col">Question set</th><th scope="col">Answered</th><th scope="col">Refused, not in the documents</th><th scope="col">Refused, another role's documents</th><th scope="col">Restricted text shown</th></tr></thead>
      <tbody>${rows}</tbody></table>
      <p class="fine"><sup>*</sup> "2 of 6": of 6 questions the documents do not answer, it said so for 2 and quoted the nearest passage for the other 4. "Answered" counts questions the documents do answer. Quoted answers, on the same index this demo uses; the test suite runs all three sets again and fails if a number moves.</p>
      </details></article>`;
  }
  function renderAnswers() {
    const box = $("#answers");
    let html = S.pending ? pendingCard() : "";
    if (S.step === 0) {
      html = `<article class="card empty-card"><div class="a-body">Answers appear here, each with numbered footnotes that open the sentence they quote.</div>
        <div class="ghost" aria-hidden="true"><i></i><i class="hl"></i><i></i></div></article>`;
    } else if (S.step === 6) {
      html = reportCard();
    } else {
      const shown = S.step === "free" ? S.answers.filter((a) => a.step === "free") : storyAnswerIds().map(byId).filter(Boolean);
      html += shown.map(card).join("");
      if (!html && S.step === "free") html = `<article class="card empty-card"><div class="a-body">Ask a question, or pick a suggestion. Switch between Maria, Hana and Farid to see what each one may read.</div></article>`;
    }
    box.innerHTML = html;
  }

  // ------------------------------------------------------------------ who is asking, the question box
  function placeComposer() {
    const composer = $("#composer");
    const inSheet = isPhone() && S.step === "free";
    const home = inSheet ? $("#story") : $("#ask-pane");
    if (composer.parentElement !== home) {
      if (inSheet) $("#story").appendChild(composer);
      else $("#ask-pane").insertBefore(composer, $("#chips"));
    }
    $("#input").placeholder = isPhone() ? "Ask a question..." : "Ask about leave, expenses, IT, the office...";
  }
  function renderAsker() {
    const free = S.step === "free";
    const people = (S.config?.personas || []).map((p) => `<button type="button" class="persona" data-persona="${esc(p.id)}" aria-pressed="${p.id === S.persona}"${free && !S.busy ? "" : " disabled"}>
      <span class="avatar ${esc(p.id)}" aria-hidden="true">${esc(p.name[0])}</span><span><span class="persona-name">${esc(p.name)}</span> <span class="persona-team">${esc(p.team)}</span></span></button>`).join("");
    $("#asker").innerHTML = `<span class="asker-label" id="asker-label">Asking as</span><div class="personas" role="group" aria-labelledby="asker-label">${people}</div>`;
    const composer = $("#composer"), chips = $("#chips");
    composer.hidden = !free; chips.hidden = !free;
    placeComposer();
    if (!free) return;
    const cfg = S.config || {};
    const typed = cfg.search?.typed === "keywords"
      ? `<span class="search-note keywords-only">${icon("alert", "i i-sm")}${esc(cfg.search.note || "Questions you type are searched by keywords only.")}</span>`
      : `<span class="search-note">${icon("search", "i i-sm")}Questions you type are matched by meaning and by keywords.</span>`;
    $("#composer-meta").innerHTML = (live() ? `<div class="seg" role="group" aria-label="How answers are written">
        <button type="button" data-ai="on" aria-pressed="${!!S.ai}">Written by AI</button>
        <button type="button" data-ai="off" aria-pressed="${!S.ai}">Quoted word for word</button></div>` : "") + typed;
    const chip = (c) => `<button type="button" class="chip${c.group === "break" ? " test" : ""}" data-chip="${(cfg.chips || []).indexOf(c)}"${S.busy ? " disabled" : ""}>
      ${c.group === "break" ? icon("flask", "i i-sm") : ""}${esc(c.label)}</button>`;
    const everyday = (cfg.chips || []).filter((c) => c.group !== "break");
    const breaking = (cfg.chips || []).filter((c) => c.group === "break");
    chips.innerHTML = `<div class="chip-row">${everyday.map(chip).join("")}</div>
      <p class="chip-label">Try to break it:</p><div class="chip-row">${breaking.map(chip).join("")}</div>`;
    $("#send").disabled = S.busy;
  }

  // ------------------------------------------------------------------ the story
  const ticks = (n) => `<span class="ticks" aria-hidden="true">${Array.from({ length: STEPS }, (_, i) => `<i class="${i < n ? "on" : ""}"></i>`).join("")}</span>`;
  const head = (n, title) => `<div class="kicker">${ticks(n)}Step ${n} of ${STEPS}</div><h2 class="story-title" id="story-title">${esc(title)}</h2>`;
  const primary = (go, label) => `<button type="button" class="btn btn-primary" data-go="${go}">${esc(label)}${icon("arrow-right")}</button>`;
  const waitLine = () => S.busy && S.wait ? `<p class="story-wait"><span class="dots" aria-hidden="true"><i></i><i></i><i></i></span>Working on it</p>` : "";
  function aiLine() {
    return live() ? "Answers are written by Claude Haiku 5.5. Every sentence is checked against the passage it cites. When the demo's AI budget is used up, answers are quoted word for word, and the page says which."
      : "This demo quotes the documents word for word, so nothing is made up. With Claude on, answers are written in plain words and checked the same way (step 4).";
  }
  // The first screen shows what the story is about to do: an answer with its footnote, and a refusal.
  // The quoted sentence is the Leave Policy's own (a test checks it is still in the document).
  const LEAVE_SENTENCE = "Full-time employees receive 25 days of paid annual leave per calendar year, plus public holidays.";
  function specimenFront() {
    return `<div class="spec-card spec-front"><span class="eyebrow">Maria asked</span><p class="a-q">${esc(Q().leave)}</p>
        <p class="a-body">&ldquo;<span class="qt">${esc(LEAVE_SENTENCE)}</span>&rdquo;<span class="fn">1</span></p>
        <div class="spec-src"><span class="n">1</span><span><b>Leave Policy</b> &rsaquo; Annual leave</span></div></div>`;
  }
  function specimen() {
    return `<div class="specimen" aria-hidden="true">${specimenFront()}
      <div class="spec-card spec-back"><span class="eyebrow">Maria asked</span><p class="a-q">${esc(Q().salary)}</p><span class="stamp">Not in your documents</span></div></div>`;
  }
  function docTitle(a) {
    const n = a.citations[0]?.number || 1;
    return String(a.passages[n - 1]?.label || "documents").split(" > ")[0];
  }
  function storyContent() {
    const st = S.story, q = Q();
    const busy = S.busy;
    switch (S.step) {
      case 0: return {
        main: `<span class="pill">Live demo &middot; fictional company documents</span>
          <h2 class="headline" id="story-title">Answers you can <span class="hl-word">check</span>.</h2>
          <p class="lede">Staff ask the company's documents a question. Every answer has footnotes that open the exact sentence it quotes, and each person sees only what their role allows.</p>
          <ul class="hero-points"><li>${icon("check")}Footnotes open the source</li><li>${icon("check")}Says so when the documents don't</li><li>${icon("check")}HR text stays with HR</li><li>${icon("check")}Six steps, about a minute</li></ul>
          <p class="ai-line">${icon(live() ? "spark" : "quote")}<span>${esc(aiLine())}</span></p>`,
        actions: `<button type="button" class="btn btn-primary" data-go="start">Start the demo${icon("arrow-right")}</button><button type="button" class="btn-link" data-go="free">Skip to free use</button>`,
        art: specimen(), target: "#story-actions .btn-primary" };
      case 1: {
        const a = byId(st.leave);
        if (!a) return { main: head(1, "A policy question.") + `<p class="story-text">Maria works in sales. She asks: <q>${esc(q.leave)}</q></p>` + waitLine(),
          actions: busy ? "" : primary("ask-leave", "Ask Maria's question"), target: "#story-actions .btn-primary" };
        const claude = a.written_by === "claude";
        return { main: head(1, "Answered, with its source.") + `<p class="story-text">${isPhone()
          ? "Tap footnote 1 to open the page it comes from." : "Footnote 1 opens the page it comes from."} ${claude
          ? `Claude wrote the answer in plain words; the marked sentence in the ${esc(docTitle(a))} is the one it rests on.`
          : `The marked sentence is word for word from the ${esc(docTitle(a))}.`}</p>`,
          actions: primary("to-2", "Next: in her own words"), target: `[data-card="${a.id}"] .fn`, ring: true };
      }
      case 2: {
        const a = byId(st.scanner);
        if (!a) return { main: head(2, "Ask in your own words.") + `<p class="story-text">Maria asks: <q>${esc(q.scanner)}</q> Too few of its words are in the manual for keyword search to answer it.</p>` + waitLine(),
          actions: busy ? "" : primary("ask-scanner", "Ask it"), target: "#story-actions .btn-primary" };
        if (a.status === "not_found") return { main: head(2, "Refused this time.") + `<p class="story-text">Nothing cleared the bar, so it said so instead of guessing.</p>`,
          actions: primary("to-3", "Next: who can see what"), target: `[data-card="${a.id}"] .stamp` };
        const page = a.passages[0]?.page;
        const alone = a.search.keywords_alone === false;
        return { main: head(2, `Found by meaning${page ? `, on page ${page} of a PDF` : ""}.`) + `<p class="story-text">${alone ? "Keyword search alone would have refused this question. " : ""}Comparing meanings found the warranty section of the ${esc(docTitle(a))}.</p>`,
          actions: primary("to-3", "Next: who can see what"), target: alone ? `[data-card="${a.id}"] .kw-note` : `[data-card="${a.id}"] .fn` };
      }
      case 3: {
        const m = byId(st.salaryMaria), h = byId(st.salaryHana);
        if (!m) return { main: head(3, "Who can see what.") + `<p class="story-text">Salary bands are for HR only. Maria asks: <q>${esc(q.salary)}</q></p>` + waitLine(),
          actions: busy ? "" : primary("ask-salary-maria", "Ask as Maria"), target: "#story-actions .btn-primary" };
        if (!h) return { main: head(3, m.status === "not_found" ? "Maria gets nothing." : "Maria's answer.") + `<p class="story-text">To her, an HR document looks exactly like one that does not exist. Now the same question as Hana from HR.</p>` + waitLine(),
          actions: busy ? "" : primary("ask-salary-hana", "Ask as Hana"), target: m.status === "not_found" ? `[data-card="${m.id}"] .stamp` : `[data-card="${m.id}"]` };
        return { main: head(3, "Hana gets the answer, with its source.") + `<p class="story-text">Access is checked before the search, so HR text is never searched, quoted or sent to the AI for Maria. In this demo you pick who you are. In your version the role comes from your company sign-in.</p>`,
          actions: primary("to-4", "Next: catch a bad AI draft"), target: `[data-card="${h.id}"] .fn`, ring: true };
      }
      case 4: {
        const a = byId(st.draft);
        if (!a) return { main: head(4, "Catch a bad AI draft.") + `<p class="story-text">For this test we force the AI's draft to say 30 days of leave, and that unused days never expire. The policy says 25, and only 5 carry over.</p>` + waitLine(),
          actions: busy ? "" : primary("ask-draft", "Send the forced draft"), target: "#story-actions .btn-primary" };
        return { main: head(4, "Held before anyone saw it.") + `<p class="story-text">The AI's draft said 30 days; the policy says 25. It was blocked and the real sentence shown instead.</p>`,
          actions: primary("to-5", "Next: where it still misses"), target: `[data-card="${a.id}"] .held` };
      }
      case 5: {
        const a = byId(st.sick);
        if (!a) return { main: head(5, "Where it still misses.") + `<p class="story-text">No document gives a number of sick days. Maria asks: <q>${esc(q.sick)}</q></p>` + waitLine(),
          actions: busy ? "" : primary("ask-sick", "Ask about sick days"), target: "#story-actions .btn-primary" };
        const miss = a.status !== "not_found";
        return { main: head(5, miss ? "A near miss." : "Refused this time.") + `<p class="story-text">${miss
          ? "The documents give no sick-day number, so it quoted the closest sentence: annual leave. The footnote makes that easy to spot."
          : "Claude saw that the passage does not answer it. Quoted mode gives the annual leave sentence here: a near miss the footnote makes easy to spot."}</p>`,
          actions: primary("to-6", "Next: what you'd get"), target: miss ? `[data-card="${a.id}"] .src` : `[data-card="${a.id}"] .stamp` };
      }
      case 6: return {
        main: head(6, "What you'd get on your documents.") + `<p class="story-text">Your version runs on your own policies and manuals.</p>`,
        actions: primary("free", "Try it yourself") + `<button type="button" class="btn-link" data-go="replay">Replay the demo</button>`, target: ".report .takeaways" };
      default: return {
        main: `<div class="kicker">Free use</div><h2 class="story-title" id="story-title">Ask anything an employee would.</h2>
          <p class="story-text">Pick who you are, then ask, or try a suggestion. In this demo you pick who you are. In your version the role comes from your company sign-in.</p>`,
        actions: `<button type="button" class="btn-link" data-go="replay">Replay the demo</button>`, target: null };
    }
  }
  function renderStory() {
    const c = storyContent();
    $("#story-main").innerHTML = c.main;
    $("#story-actions").innerHTML = c.actions;
    $("#story-art").innerHTML = c.art || "";
    document.body.dataset.step = String(S.step);
    document.body.classList.toggle("in-story", S.step !== "free");
    document.body.classList.toggle("free", S.step === "free");
    $("#replay").hidden = S.step === 0 || S.step === 1 && !S.story.leave;
    $("#workspace").inert = S.step === 0;
    document.querySelectorAll("#story-actions button").forEach((x) => { x.disabled = S.busy; });
    measure();
    return c;
  }
  function measure() {
    const story = $("#story");
    const h = getComputedStyle(story).position === "fixed" ? story.offsetHeight : 0;
    document.documentElement.style.setProperty("--story-h", h + "px");
  }
  function reveal(sel, block = "nearest") {
    const el = typeof sel === "string" ? document.querySelector(sel) : sel;
    if (!el || !el.getClientRects().length) return;
    const sticky = !isPhone() && S.step === "free" && !$("#composer").hidden ? $("#composer").offsetHeight : 0;  // the sticky question box
    const top = isPhone() ? ($("#switch-row")?.offsetHeight || 0) + 8 : 12 + sticky;
    const bottom = (isPhone() ? $("#story").offsetHeight : 0) + 12;
    el.style.setProperty("scroll-margin", `${top}px 0 ${bottom}px 0`);
    el.scrollIntoView({ block });
  }
  function revealTarget() {
    const c = storyContent();
    if (!c.target) return;
    document.querySelectorAll(".ring").forEach((x) => x.classList.remove("ring"));
    const el = document.querySelector(c.target);
    if (!el) return;
    if (c.ring && !el.matches(".btn")) el.classList.add("ring");
    reveal(el);
  }
  function setStep(n) {
    const before = S.step;
    S.step = n;
    if (before !== n) { S.open = null; S.deskNote = null; S.fresh = 0; }
    save();
    render();
    if (isPhone()) setPane("answer");
    if (before !== n) {
      announce($("#story-title")?.textContent || "");
      const btn = $("#story-actions .btn-primary");
      if (btn && S.keyboard) btn.focus({ preventScroll: true });
      if (!isPhone()) { $("#ask-pane").scrollTop = 0; $("#desk").scrollTop = 0; }
      requestAnimationFrame(revealTarget);
    }
  }
  function setPersona(id) { S.persona = id; renderAsker(); }

  async function go(what) {
    if (S.busy) return;
    const q = Q();
    const run = async (key, question, p, extra, meta) => {
      setPersona(p);
      const a = await ask(question, p, extra, meta);
      if (a) S.story[key] = a.id;
      save();
      await landed(a);
      if (S.keyboard) $("#story-actions .btn-primary")?.focus({ preventScroll: true });
    };
    switch (what) {
      case "start": case "replay":
        S.answers = S.answers.filter((a) => a.step === "free"); S.story = {}; S.persona = "maria"; setStep(1); break;
      case "ask-leave": await run("leave", q.leave, "maria"); break;
      case "ask-scanner": await run("scanner", q.scanner, "maria"); break;
      case "ask-salary-maria": await run("salaryMaria", q.salary, "maria", {}, { restricted: true }); break;
      case "ask-salary-hana": await run("salaryHana", q.salary, "hana"); break;
      case "ask-draft": await run("draft", q.leave, "maria", { simulate: "bad_draft" }); break;
      case "ask-sick": await run("sick", q.sick, "maria"); break;
      case "to-2": setPersona("maria"); setStep(2); break;
      case "to-3": setStep(3); break;
      case "to-4": setPersona("maria"); setStep(4); break;
      case "to-5": setStep(5); break;
      case "to-6": setStep(6); break;
      case "free":
        S.persona = "maria"; setStep("free");
        if (!isPhone()) $("#input").focus();
        break;
      default: break;
    }
  }

  // ------------------------------------------------------------------ panes and rendering
  function setPane(pane) {
    S.pane = pane;
    document.body.dataset.pane = pane;
    document.querySelectorAll(".switch-row button").forEach((b) => b.setAttribute("aria-pressed", String(b.dataset.pane === pane)));
    if (isPhone()) window.scrollTo(0, 0);
  }
  function render() { renderStory(); renderAsker(); renderAnswers(); renderDoc(); measure(); }

  function autosize() { const t = $("#input"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 140) + "px"; }
  async function freeAsk(question, p, meta = {}) {
    const a = await ask(question, p, {}, meta);
    if (a) { $("#input").value = ""; autosize(); }
    await landed(a);
    if (a) reveal(`[data-card="${a.id}"]`, "start");
  }
  function wire() {
    document.addEventListener("keydown", (e) => { if (e.key === "Tab" || e.key === "Enter" || e.key === " ") S.keyboard = true; }, true);
    document.addEventListener("pointerdown", () => { S.keyboard = false; }, true);
    $("#composer").addEventListener("submit", (e) => { e.preventDefault(); freeAsk($("#input").value, S.persona); });
    $("#input").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); freeAsk($("#input").value, S.persona); } });
    $("#input").addEventListener("input", () => { autosize(); measure(); });
    document.addEventListener("click", (e) => {
      const t = e.target.closest("button");
      if (!t || t.disabled) return;
      if (t.dataset.go) go(t.dataset.go);
      else if (t.dataset.open) {
        const [id, n] = t.dataset.open.split(":").map(Number);
        openSource(id, n);
      } else if (t.dataset.chip !== undefined) {
        const c = S.config.chips[+t.dataset.chip];
        setPersona(c.persona);
        freeAsk(c.question, c.persona, { restricted: !!c.restricted });
      } else if (t.dataset.persona) {
        setPersona(t.dataset.persona); save();
      } else if (t.dataset.ai) {
        S.ai = t.dataset.ai === "on"; save(); renderAsker();
      } else if (t.dataset.pane) {
        setPane(t.dataset.pane);
        if (t.dataset.pane === "source") { const mark = $("#desk mark.hl.active"); if (mark) reveal(mark, "center"); }
      }
    });
    $("#replay").addEventListener("click", () => go("replay"));
    let resizeTimer = null;
    window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(() => { placeComposer(); renderDoc(); measure(); }, 80); });
  }

  async function init() {
    wire();
    try {
      S.config = await api("GET", "/api/config");
    } catch {
      $("#story-main").innerHTML = `<h2 class="story-title" id="story-title">The demo server is not reachable.</h2><p class="story-text">Reload the page in a moment.</p>`;
      return;
    }
    if (!S.config.live) S.ai = false;
    const saved = store.get(STORE_KEY);
    if (saved && Array.isArray(saved.answers)) {
      Object.assign(S, { step: saved.step ?? 0, persona: saved.persona || "maria", answers: saved.answers, next: saved.next || 1,
        story: saved.story || {}, open: saved.open || null, ai: S.config.live && saved.ai !== false, deskNote: saved.deskNote || null });
    }
    render();
    if (S.open) {
      const { id, n } = S.open; S.open = null;
      await openSource(id, n, { auto: true, quiet: true });
    }
    if (S.step === 0) api("POST", "/api/warm").catch(() => {});  // loads the meaning model while the visitor reads
    else requestAnimationFrame(revealTarget);
  }
  init();
})();
