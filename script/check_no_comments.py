from __future__ import annotations

import ast
import io
import pathlib
import sys
import tokenize

ROOT = pathlib.Path(__file__).resolve().parent.parent
SKIP_DIRS = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".ruff_cache",
    "data",
    "build",
    "dist",
    "node_modules",
    ".ipynb_checkpoints",
}
CONFIG_SUFFIXES = {".service", ".toml", ".sh", ".cfg", ".ini", ".conf"}
DOC_WHITELIST = {"AGENT.md", "README.md", "history.md"}
NODES = (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)


def ignored(path: pathlib.Path) -> bool:
    parts = path.relative_to(ROOT).parts
    return any(part in SKIP_DIRS or part.endswith(".egg-info") for part in parts)


def python_violations(path: pathlib.Path) -> list[tuple[int, str]]:
    src = path.read_text(encoding="utf-8")
    found = [
        (token.start[0], "comment")
        for token in tokenize.generate_tokens(io.StringIO(src).readline)
        if token.type == tokenize.COMMENT
    ]
    for node in ast.walk(ast.parse(src)):
        body = getattr(node, "body", [])
        if not isinstance(node, NODES) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            found.append((first.lineno, "docstring"))
    return found


def config_violations(path: pathlib.Path) -> list[tuple[int, str]]:
    return [
        (number, "comment")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.lstrip().startswith("#")
    ]


def doc_violations() -> list[tuple[str, str]]:
    found = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or ignored(path):
            continue
        if path.suffix.lower() == ".md":
            rel = path.relative_to(ROOT).as_posix()
            if rel not in DOC_WHITELIST:
                found.append((rel, "unexpected markdown file"))
    return found


def repo_files() -> list[pathlib.Path]:
    return [
        path
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not ignored(path)
        and (path.suffix == ".py" or path.suffix in CONFIG_SUFFIXES or path.name == ".gitignore")
    ]


def main() -> int:
    total = 0
    for path in repo_files():
        found = python_violations(path) if path.suffix == ".py" else config_violations(path)
        for number, kind in sorted(found):
            print(f"{path.relative_to(ROOT)}:{number}: {kind}")
            total += 1
    for rel, kind in doc_violations():
        print(f"{rel}: {kind}")
        total += 1
    if total:
        print(f"FAIL: {total} violation(s)")
        return 1
    print("OK: no comments or docstrings; markdown whitelist ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
