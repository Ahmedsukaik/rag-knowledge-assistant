"""Small text helpers shared by ingestion, retrieval and answering. No dependencies."""
from __future__ import annotations

import re

# Control characters, zero-width characters, bidi overrides and lone surrogates (half of a
# character pair, which cannot even be written as UTF-8). They can hide text from a
# human reader (and make printed output lie about what it shows), so they are removed from
# document text at ingestion and from anything printed on one line.
_INVISIBLE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\u200b-\u200f\u202a-\u202e\u2060-\u2064\u2066-\u2069\ufeff\ud800-\udfff]")


def clean_document_text(text: str) -> str:
    """Normalise newlines and drop invisible characters. Line breaks and tabs are kept."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return _INVISIBLE.sub("", text)


def one_line(value: object, limit: int = 300) -> str:
    """Text that came from documents or callers, made safe to print on one line."""
    text = _INVISIBLE.sub("", str(value)).replace("\n", " ").replace("\t", " ")
    text = re.sub(r" {2,}", " ", text).strip()
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def squash(text: str) -> str:
    """Text with all whitespace removed. Used to compare quotes regardless of line breaks."""
    return re.sub(r"\s+", "", text)


# ---------------------------------------------------------------------------- tokens
STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because been before being below
    between both but by can could did do does doing down during each few for from further had has have
    having he her here hers herself him himself his how i if in into is it its itself just me more most
    my myself no nor not now of off on once only or other our ours ourselves out over own same she should
    so some such than that the their theirs them themselves then there these they this those through to
    too under until up very was we were what when where which while who whom why will with would you your
    yours yourself yourselves
    also am anyone anything get gets got may might must need needs please shall tell us what's whats
    i'm im much many does doesn't dont don't can't cant know want wants like
    s t d ll re ve m
    happen happens happened mean means
    """.split()
)


def _stem(token: str) -> str:
    """A deliberately light English stemmer: plurals, -ing, -ed, and a final -e.

    "approve", "approved", "approves" and "approving" all become "approv". It does not try
    to be clever; a wrong merge would make unrelated words match.
    """
    if len(token) <= 3 or not token.isalpha():
        return token
    if token.endswith("sses"):
        token = token[:-2]
    elif token.endswith(("ies", "ied")) and len(token) > 4:  # policies, carried
        return token[:-3] + "y"
    elif token.endswith("s") and not token.endswith(("ss", "us", "is")):
        token = token[:-1]
    for suffix in ("ing", "ed"):
        stem = token[: -len(suffix)]
        if token.endswith(suffix) and len(stem) >= 4 and re.search(r"[aeiouy]", stem):
            token = stem
            break
    if token.endswith("e") and len(token) > 4:
        token = token[:-1]
    return token


_TOKEN = re.compile(r"\d+(?:\.\d+)?|[^\W_]+")


def parse_synonyms(text: str) -> dict[str, str]:
    """'leave, holiday, vacation' lines -> {'holiday': 'leave', 'vacation': 'leave'} (stemmed)."""
    mapping: dict[str, str] = {}
    for line in text.splitlines():
        words = [_stem(w.strip().lower()) for w in line.split(",") if w.strip() and not line.lstrip().startswith("#")]
        words = [w for w in words if w and w not in STOPWORDS]
        for word in words[1:]:
            if word != words[0]:
                mapping[word] = words[0]
    return mapping


def load_synonyms(setting: str | None = None) -> dict[str, str]:
    """The synonym list named by KB_SYNONYMS (default: kb/synonyms.txt; 'off' for none)."""
    import os
    from pathlib import Path

    setting = os.environ.get("KB_SYNONYMS", "") if setting is None else setting
    if setting.strip().lower() in ("off", "none", "no", "0"):
        return {}
    path = Path(setting) if setting.strip() else Path(__file__).with_name("synonyms.txt")
    try:
        return parse_synonyms(path.read_text(encoding="utf-8"))
    except OSError:
        raise SystemExit(f"KB_SYNONYMS: cannot read {path}") from None


SYNONYMS: dict[str, str] = load_synonyms()


def tokenize(text: str) -> list[str]:
    """Lower-case words and numbers, stopwords removed, lightly stemmed, synonyms merged.

    "£24,000" becomes "24000" (thousand separators removed), "3.5%" becomes "3.5",
    "E31" stays "e31". Single letters are kept ("Band C"), except stopwords such as "a".
    Words on one line of the synonym list become the same word ("holidays" -> "leave").
    """
    text = re.sub(r"(?<=\d)[,'](?=\d{3}(?!\d))", "", text.lower())
    out = []
    for token in _TOKEN.findall(text):
        if token in STOPWORDS:
            continue
        stem = _stem(token)
        out.append(SYNONYMS.get(stem, stem))
    return out


# ---------------------------------------------------------------------------- sentences
_SENTENCE_END = re.compile(r"(?<=[.!?])[\"')\]]?\s+(?=[\"'(\[]?[A-Z0-9£$€])")


def split_sentences(text: str) -> list[str]:
    """Sentences and list items, each an exact substring of `text` (stripped)."""
    out: list[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        start = 0
        for m in _SENTENCE_END.finditer(line):
            piece = line[start : m.end()].strip()
            if piece:
                out.append(piece)
            start = m.end()
        tail = line[start:].strip()
        if tail:
            out.append(tail)
    return out


# ---------------------------------------------------------------------------- instructions aimed at an AI
_INSTRUCTION_PATTERNS = [
    r"\bignore\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier|other)\s+(?:instructions|prompts|rules|messages)",
    r"\bdisregard\s+(?:all\s+|any\s+)?(?:the\s+|your\s+)?(?:previous|prior|above|earlier)?\s*(?:instructions|prompts|rules)",
    r"\b(?:note|message|instructions?)\s+(?:for|to)\s+(?:any\s+|all\s+|the\s+)?(?:ai|a\.i\.|assistant|chatbot|language\s+model|llm|bot)s?\b",
    r"\b(?:ai|assistant|chatbot|llm)s?\s+reading\s+this\b",
    r"\byou\s+are\s+now\s+(?:a|an|the|in)\b",
    r"\b(?:reveal|print|show|repeat)\s+(?:your|the)\s+(?:system\s+)?prompt\b",
    r"\bnew\s+instructions\s*:",
    r"\btell\s+(?:every|all|any|the)\s+users?\s+that\b",
    r"<\s*/?\s*(?:system|instructions?)\s*>",
]
_INSTRUCTION = re.compile("|".join(_INSTRUCTION_PATTERNS), re.IGNORECASE)


def looks_like_instructions(sentence: str) -> bool:
    """True for text that addresses an AI system ("ignore all previous instructions ...").

    A heuristic for obvious cases, not a guarantee. The real protection is elsewhere: the
    answer step treats documents as data, access rules are enforced in code, and every
    quoted source is checked. This only adds a warning and keeps such sentences out of
    extractive answers.
    """
    return bool(_INSTRUCTION.search(sentence))


_ADDRESSES_AN_AI = re.compile(r"\b(?:ai|a\.i\.|assistants?|chatbots?|llms?|language\s+models?|instructions?|prompts?)\b",
                              re.IGNORECASE)


def instruction_sentences(text: str) -> tuple[str, ...]:
    """Every sentence of a paragraph (up to the next blank line) that contains text addressed
    to an AI. The whole paragraph is taken, because an injected instruction usually continues
    past its first sentence or line. If that paragraph mentions an AI, an assistant or
    instructions, the next paragraph is taken too, however the first one ends ("Note for AI
    assistants." followed by the payload), unless the next one starts a new topic: a heading
    or a question line, as in an FAQ.

    This is a keyword check. It catches obvious attempts, not every possible wording."""
    out: list[str] = []
    carry = False
    for paragraph in re.split(r"\n\s*\n", text):
        sentences = split_sentences(paragraph)
        first_line = paragraph.strip().splitlines()[0].strip() if paragraph.strip() else ""
        new_topic = first_line.startswith("#") or first_line.endswith("?")
        hit = any(looks_like_instructions(s) for s in sentences)
        if hit or (carry and not new_topic):
            out.extend(sentences)
        carry = hit and bool(_ADDRESSES_AN_AI.search(paragraph))
    return tuple(out)


# ---------------------------------------------------------------------------- identifiers
_ID_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
_NOT_IDENTIFIERS = re.compile(r"(?:q[1-4]|h[12]|fy\d{2,4})", re.IGNORECASE)  # quarters, halves, years


def identifiers(text: str) -> set[str]:
    """Identifier-like words: they start with a letter and contain a digit, such as an error code
    (E31), a model number (PT-200), a SKU or a policy number. Lower case, without - and _."""
    out = set()
    for token in _ID_TOKEN.findall(text):
        token = token.strip("-_")
        if token[:1].isalpha() and any(ch.isdigit() for ch in token) and not _NOT_IDENTIFIERS.fullmatch(token):
            out.add(re.sub(r"[-_]", "", token.lower()))
    return out


def words_with_digits(text: str) -> set[str]:
    """Every word of `text` in the same form as identifiers(), to look them up."""
    return {re.sub(r"[-_]", "", t.strip("-_").lower()) for t in _ID_TOKEN.findall(text)}
