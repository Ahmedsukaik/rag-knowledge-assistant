"""Report how big the Vercel function would be, unzipped, with the meaning model (as deployed) and
without it (KB_DEMO_SEARCH=keywords). Vercel's limit for a function is 250 MB unzipped.

    python scripts/bundle_size.py            needs a network connection (it downloads Linux wheels)
    python scripts/bundle_size.py --json

What it measures:
- Dependencies: api/requirements.txt installed for Linux x86-64 and Python 3.12 (the Vercel
  runtime) into a temporary folder, from manylinux wheels, the way pip installs them, with and
  without the .pyc files pip compiles. Vercel's own installer may or may not compile them.
- Project files: every file git would give Vercel (tracked, plus new files not ignored), minus
  vercel.json's excludeFiles for api/index.py.
- Whether Vercel applies excludeFiles to the installed packages as well is not documented
  clearly; the report gives both figures. vercel.json also lists **/__pycache__/**,
  **/tests/** and onnxruntime's model-conversion tools, which the demo never imports.
- Without the model: the same, without demo/model. vercel.json cannot read environment variables, so
  dropping the model from the bundle as well means adding demo/model/** to excludeFiles; with
  KB_DEMO_SEARCH=keywords alone the server never loads it, but it still ships.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LIMIT_MB = 250
PLATFORMS = ("manylinux2014_x86_64", "manylinux_2_17_x86_64", "manylinux_2_28_x86_64", "manylinux_2_34_x86_64")
MB = 1_000_000


def expand_braces(pattern: str) -> list[str]:
    """{a/**,b/**} -> [a/**, b/**] (one level, as vercel.json uses it)."""
    match = re.fullmatch(r"\{(.*)\}", pattern.strip())
    return [p.strip() for p in match.group(1).split(",")] if match else [pattern.strip()]


def glob_regex(pattern: str) -> re.Pattern[str]:
    out, i = "", 0
    while i < len(pattern):
        if pattern.startswith("**/", i):
            out, i = out + "(?:.*/)?", i + 3
        elif pattern.startswith("**", i):
            out, i = out + ".*", i + 2
        elif pattern[i] == "*":
            out, i = out + "[^/]*", i + 1
        elif pattern[i] == "?":
            out, i = out + "[^/]", i + 1
        else:
            out, i = out + re.escape(pattern[i]), i + 1
    return re.compile(out)


def project_files() -> list[str]:
    listed = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"], cwd=ROOT, capture_output=True,
                            text=True, check=True).stdout.splitlines()
    return sorted({f for f in listed if (ROOT / f).is_file()})


def shipped(files: list[str], exclude: list[str]) -> list[str]:
    patterns = [glob_regex(p) for p in exclude]
    return [f for f in files if not any(p.fullmatch(f) for p in patterns)]


def folder_size(folder: Path, pyc: bool = True) -> int:
    return sum(p.stat().st_size for p in folder.rglob("*") if p.is_file() and (pyc or p.suffix != ".pyc"))


def install_dependencies(target: Path) -> None:
    command = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--target", str(target),
               "--python-version", "3.12", "--implementation", "cp", "--only-binary=:all:",
               *[arg for platform in PLATFORMS for arg in ("--platform", platform)], "-r", str(ROOT / "api" / "requirements.txt")]
    subprocess.run(command, check=True)


def measure() -> dict:
    vercel = json.loads((ROOT / "vercel.json").read_text(encoding="utf-8"))
    exclude = expand_braces(vercel["functions"]["api/index.py"].get("excludeFiles", ""))
    files = shipped(project_files(), exclude)
    model = [f for f in files if f.startswith("demo/model/")]
    sizes = {f: (ROOT / f).stat().st_size for f in files}
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "site"
        install_dependencies(target)
        deps, deps_no_pyc = folder_size(target), folder_size(target, pyc=False)
        patterns = [glob_regex(p) for p in exclude]
        deps_excluded = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()
                            and not any(x.fullmatch(p.relative_to(target).as_posix()) for x in patterns))
        packages = {}
        for entry in target.iterdir():
            if entry.is_dir() and not entry.name.endswith((".dist-info", ".data")) and entry.name != "bin":
                packages[entry.name] = folder_size(entry)
        onnx = sum(size for name, size in packages.items() if name in ("onnxruntime", "flatbuffers", "google", "coloredlogs",
                                                                         "humanfriendly", "sympy", "mpmath"))
    project = sum(sizes.values())
    model_size = sum(sizes[f] for f in model)
    plan_a = deps + project
    plan_b = deps + project - model_size
    return {
        "limit_mb": LIMIT_MB,
        "dependencies_mb": round(deps / MB, 1), "dependencies_without_pyc_mb": round(deps_no_pyc / MB, 1),
        "onnxruntime_and_its_dependencies_mb": round(onnx / MB, 1),
        "project_files_mb": round(project / MB, 1), "model_mb": round(model_size / MB, 1),
        "plan_a_mb": round(plan_a / MB, 1), "plan_a_without_pyc_mb": round((plan_a - deps + deps_no_pyc) / MB, 1),
        "plan_b_mb": round(plan_b / MB, 1), "plan_b_without_onnxruntime_mb": round((plan_b - onnx) / MB, 1),
        "dependencies_after_exclude_mb": round(deps_excluded / MB, 1),
        "plan_a_if_exclude_applies_to_packages_mb": round((deps_excluded + project) / MB, 1),
        "plan_b_if_exclude_applies_to_packages_mb": round((deps_excluded + project - model_size) / MB, 1),
        "largest_packages_mb": {k: round(v / MB, 1) for k, v in sorted(packages.items(), key=lambda kv: -kv[1])[:8]},
        "project_files": len(files),
        "largest_project_files_mb": {f: round(s / MB, 2) for f, s in sorted(sizes.items(), key=lambda kv: -kv[1])[:5]},
    }


def report(r: dict) -> str:
    lines = [
        "What the Vercel function would ship, unzipped (Linux x86-64 wheels, Python 3.12)",
        "",
        f"{'':44}{'model':>10}{'no model':>10}",
        f"{'dependencies (api/requirements.txt)':44}{r['dependencies_mb']:>8} MB{r['dependencies_mb']:>8} MB",
        f"{'  without the .pyc files pip compiles':44}{r['dependencies_without_pyc_mb']:>8} MB",
        f"{'  of which onnxruntime (with protobuf)':44}{r['onnxruntime_and_its_dependencies_mb']:>8} MB",
        f"{'  after excludeFiles, if Vercel applies it':44}{r['dependencies_after_exclude_mb']:>8} MB",
        f"{'project files (after excludeFiles)':44}{r['project_files_mb']:>8} MB{round(r['project_files_mb'] - r['model_mb'], 1):>8} MB",
        f"{'  of which the model (demo/model)':44}{r['model_mb']:>8} MB{'-':>10}",
        f"{'total':44}{r['plan_a_mb']:>8} MB{r['plan_b_mb']:>8} MB",
        f"{'  without .pyc files':44}{r['plan_a_without_pyc_mb']:>8} MB",
        f"{'  if excludeFiles applies to packages too':44}{r['plan_a_if_exclude_applies_to_packages_mb']:>8} MB"
        f"{r['plan_b_if_exclude_applies_to_packages_mb']:>8} MB",
        f"{'no model, and onnxruntime removed as well':44}{'':>10}{r['plan_b_without_onnxruntime_mb']:>8} MB",
        f"{'limit':44}{r['limit_mb']:>8} MB{r['limit_mb']:>8} MB",
        "",
        "Largest packages: " + ", ".join(f"{k} {v} MB" for k, v in r["largest_packages_mb"].items()),
        "Largest project files: " + ", ".join(f"{k} {v} MB" for k, v in r["largest_project_files_mb"].items()),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    r = measure()
    print(json.dumps(r, indent=2) if args.json else report(r))
    return 0 if r["plan_a_mb"] <= LIMIT_MB else 1


if __name__ == "__main__":
    raise SystemExit(main())
