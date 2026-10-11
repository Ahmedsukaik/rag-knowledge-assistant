"""Regenerate docs/TOOLS.md from the MCP server's own tool list:  python docs/gen_tools.py"""
import asyncio
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from kb.config import TokenTable  # noqa: E402
from kb.mcp_server import INSTRUCTIONS, build_mcp  # noqa: E402
from kb.service import KnowledgeBase  # noqa: E402
from kb.store import Store  # noqa: E402


def arg_type(spec: dict) -> str:
    if "anyOf" in spec:
        return " or ".join(o.get("type", "") for o in spec["anyOf"] if o.get("type") != "null")
    return spec.get("type", "")


def main() -> None:
    kb = KnowledgeBase(Store(str(Path(tempfile.mkdtemp()) / "kb.db")))
    tools = asyncio.run(build_mcp(kb, TokenTable({})).list_tools())
    out = [
        "# Tool reference", "",
        "Generated from the MCP server's own tool list (`python docs/gen_tools.py`).", "",
        "Endpoint: `POST /mcp` on the server started by `python -m kb serve` (Streamable HTTP, stateless), "
        "with the header `Authorization: Bearer <token>`. Or over stdio with `python -m kb mcp-stdio`.", "",
        "Every tool is read-only and answers only from the collections the token's role may read. "
        "Tools return JSON objects. A passage id looks like `general/leave-policy#2` "
        "(document id, `#`, position in the document).", "",
        "Instructions the server sends to the client:", "",
        f"> {INSTRUCTIONS}", "",
        "| Tool | Kind | Roles allowed |", "|---|---|---|",
    ]
    for t in tools:
        kind = "read-only" if t.annotations and t.annotations.read_only_hint else "WRITES"
        out.append(f"| `{t.name}` | {kind} | every role (results limited to the role's collections) |")
    out.append("")
    for t in tools:
        out += [f"## {t.name}", "", " ".join((t.description or "").split()), ""]
        props = t.input_schema.get("properties", {})
        required = set(t.input_schema.get("required", []))
        if props:
            out += ["| Argument | Type | Required | Default |", "|---|---|---|---|"]
            out += [f"| `{n}` | {arg_type(s)} | {'yes' if n in required else 'no'} | "
                    f"{s['default'] if 'default' in s else ''} |" for n, s in props.items()]
        else:
            out.append("No arguments.")
        out.append("")
    (ROOT / "docs" / "TOOLS.md").write_text("\n".join(out), encoding="utf-8", newline="\n")
    print(f"wrote docs/TOOLS.md ({len(tools)} tools)")


if __name__ == "__main__":
    main()
