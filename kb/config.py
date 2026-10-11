"""Settings, roles and the token -> identity table.

Every document belongs to one collection (the folder it was ingested from, for example
`general`, `hr` or `finance`). Every API token has one role. A role lists the collections
it may read.

The role named `admin` always reads every collection: it may add and remove documents and
read the audit log, which holds everyone's questions. There is no "admin for HR only"; give
the admin role only to the people who run the system.

Default roles (override with KB_ROLES):

    staff    general
    hr       general, hr
    finance  general, finance
    admin    every collection ("*"), always
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .envfile import load_env_file

ALL = "*"
ADMIN_ROLE = "admin"
DEFAULT_ROLES: dict[str, list[str]] = {
    "staff": ["general"],
    "hr": ["general", "hr"],
    "finance": ["general", "finance"],
    ADMIN_ROLE: [ALL],
}
NAME = re.compile(r"[a-z][a-z0-9_-]{0,30}")
DEFAULT_MODEL = "claude-sonnet-5-5"
# Tokens that are published in this repository: the change-me ones in .env.example and the
# demo and test tokens. Fine on 127.0.0.1, refused on a public address.
PUBLISHED_TOKEN = re.compile(r"change-me|^(?:demo|test)-[a-z]+-token-\d+$")
DEFAULT_EFFORT = "low"


class ConfigError(ValueError):
    """A setting is wrong. The message says which one and how to fix it."""


@dataclass(frozen=True)
class Identity:
    name: str
    role: str
    collections: frozenset[str]  # may contain "*" (every collection)
    key: str = field(default="", compare=False)  # a short fingerprint of the token (never the token)

    @property
    def is_admin(self) -> bool:
        return self.role == ADMIN_ROLE

    def visible(self, existing: set[str] | frozenset[str]) -> frozenset[str]:
        """The collections this identity may read, out of those that exist."""
        if ALL in self.collections:
            return frozenset(existing)
        return frozenset(c for c in self.collections if c in existing)


def parse_roles(roles: Any) -> dict[str, frozenset[str]]:
    if not isinstance(roles, dict):
        raise ConfigError('KB_ROLES must be a JSON object, for example {"staff": ["general"]}')
    out: dict[str, frozenset[str]] = {}
    for role, collections in roles.items():
        if not isinstance(role, str) or not NAME.fullmatch(role):
            raise ConfigError(f"invalid role name {role!r}: use lower-case letters, digits, '-' and '_'")
        if not isinstance(collections, list) or not all(isinstance(c, str) for c in collections):
            raise ConfigError(f"role {role!r} needs a list of collection names")
        for c in collections:
            if c != ALL and not NAME.fullmatch(c):
                raise ConfigError(f"invalid collection name {c!r} in role {role!r}")
        if role == ADMIN_ROLE and collections != [ALL]:
            raise ConfigError('the admin role always reads every collection: set it to ["*"] (or leave it out)')
        out[role] = frozenset(collections)
    out.setdefault(ADMIN_ROLE, frozenset({ALL}))
    return out


def _digest(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


class TokenTable:
    """Maps bearer tokens to identities. Only SHA-256 digests of the tokens are kept."""

    def __init__(self, tokens: Any, roles: Any = None):
        role_map = parse_roles(roles if roles is not None else DEFAULT_ROLES)
        if not isinstance(tokens, dict):
            raise ConfigError('KB_TOKENS must be a JSON object: {"<token>": {"name": "...", "role": "..."}}')
        self._by_digest: dict[str, Identity] = {}
        for token, info in tokens.items():
            if not isinstance(info, dict):
                raise ConfigError("each token in KB_TOKENS needs an object with a name and a role")
            if not isinstance(token, str) or len(token) < 16:
                raise ConfigError(f"the token of {info.get('name')!r} is shorter than 16 characters")
            role = info.get("role")
            if role not in role_map:
                raise ConfigError(f"unknown role {role!r} for the token of {info.get('name')!r}; "
                                  f"known roles: {', '.join(sorted(role_map))}")
            digest = _digest(token)
            self._by_digest[digest] = Identity(str(info.get("name") or role), role, role_map[role], digest[:16])

    def identify(self, authorization_header: str | None) -> Identity | None:
        if not authorization_header:
            return None
        scheme, _, token = authorization_header.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        return self._by_digest.get(_digest(token.strip()))

    def withdraw_all(self) -> None:
        """Forget every token (used by tests to simulate an administrator withdrawing access)."""
        self._by_digest.clear()


def _int_env(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    if not raw.isdigit():
        raise SystemExit(f"{name} must be a whole number.")
    return int(raw)


@dataclass(frozen=True)
class Settings:
    db_path: str
    docs_dir: str
    tokens: TokenTable
    answer_mode: str = "extractive"  # or "claude"
    model: str = DEFAULT_MODEL
    claude_per_minute: int = 10      # Claude answers per person per minute; more are quoted instead
    claude_per_day: int = 500        # Claude answers per day for everyone together
    claude_per_person_day: int = 100  # Claude answers per day for one person (one token)
    uses_placeholders: bool = False  # a token published in this repository (.env.example, demo.py, tests)
    retriever: str = "hybrid"        # hybrid (keywords and a local embedding model) or bm25 (keywords only)
    embed_model: str = "BAAI/bge-small-en-v1.5"
    claude_effort: str | None = None     # KB_CLAUDE_EFFORT; None: the model's default (low where accepted)
    claude_fallbacks: str | None = None  # KB_CLAUDE_FALLBACKS; None: "default" where the model has one

    @classmethod
    def from_env(cls) -> "Settings":
        load_env_file(os.environ.get("KB_ENV_FILE", ".env"))
        if not os.environ.get("KB_TOKENS"):
            raise SystemExit("Missing setting KB_TOKENS. Copy .env.example to .env and edit it.")
        try:
            tokens_raw = json.loads(os.environ["KB_TOKENS"])
            roles_raw = json.loads(os.environ["KB_ROLES"]) if os.environ.get("KB_ROLES") else None
        except json.JSONDecodeError as exc:
            raise SystemExit(f"KB_TOKENS or KB_ROLES is not valid JSON: {exc}") from None
        try:
            tokens = TokenTable(tokens_raw, roles_raw)
        except ConfigError as exc:
            raise SystemExit(f"Setting error: {exc}.") from None
        mode = os.environ.get("KB_ANSWER_MODE", "extractive").strip().lower()
        if mode not in ("extractive", "claude"):
            raise SystemExit("KB_ANSWER_MODE must be 'extractive' or 'claude'.")
        retriever = retriever_setting()
        return cls(
            db_path=os.environ.get("KB_DB", "kb.db"),
            docs_dir=os.environ.get("KB_DOCS_DIR", "samples/docs"),
            tokens=tokens,
            answer_mode=mode,
            # Only the project's own KB_CLAUDE_* names are read: ANTHROPIC_MODEL or CLAUDE_EFFORT may be
            # set globally for other tools and must not change this server silently.
            model=os.environ.get("KB_CLAUDE_MODEL") or DEFAULT_MODEL,
            claude_effort=(os.environ.get("KB_CLAUDE_EFFORT") or "").strip().lower() or None,
            claude_fallbacks=(os.environ.get("KB_CLAUDE_FALLBACKS") or "").strip().lower() or None,
            claude_per_minute=_int_env("KB_CLAUDE_PER_MINUTE", 10),
            claude_per_day=_int_env("KB_CLAUDE_PER_DAY", 500),
            claude_per_person_day=_int_env("KB_CLAUDE_PER_PERSON_DAY", 100),
            uses_placeholders=any(PUBLISHED_TOKEN.search(str(t)) for t in tokens_raw),
            retriever=retriever,
            embed_model=os.environ.get("KB_EMBED_MODEL") or "BAAI/bge-small-en-v1.5",
        )


def retriever_setting() -> str:
    value = (os.environ.get("KB_RETRIEVER") or "hybrid").strip().lower()
    if value not in ("hybrid", "bm25"):
        raise SystemExit("KB_RETRIEVER must be 'hybrid' (keywords and meaning, the default) or 'bm25' (keywords only).")
    return value
