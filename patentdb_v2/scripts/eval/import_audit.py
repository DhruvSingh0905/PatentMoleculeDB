"""Dead-code / reachability audit for the `patentdb` package.

Builds the internal import graph with the AST (function-level imports
included) and reports which modules are reachable from each real entry
point. Anything unreachable is a dead-code candidate.

This exists because the package accumulated 19 unreachable modules
(4,487 LOC) and 18 unused config constants before anyone noticed —
including a whole step-cache DAG that the docs told you to maintain.
Run it after any significant restructuring.

Usage:
    python3 -m patentdb.scripts.eval.import_audit
    python3 -m patentdb.scripts.eval.import_audit --json
    python3 -m patentdb.scripts.eval.import_audit --config    # unused constants too
"""
from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict, deque
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]   # patentdb/
REPO_ROOT = PACKAGE_ROOT.parent
PKG_NAME = PACKAGE_ROOT.name

# The only real ways into this code. Everything else is a library.
ENTRY_POINTS = [f"{PKG_NAME}.routes.process_patent"]


def _mod_name(p: Path) -> str:
    parts = list(p.relative_to(REPO_ROOT).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _resolve_relative(current: str, level: int, module: str | None) -> str:
    parts = current.split(".")
    base = parts[: len(parts) - level] if level <= len(parts) else []
    if module:
        base = base + module.split(".")
    return ".".join(base)


def build_graph() -> tuple[dict[str, Path], dict[str, set[str]]]:
    files = sorted(p for p in PACKAGE_ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    mods = {_mod_name(p): p for p in files}
    edges: dict[str, set[str]] = defaultdict(set)

    for m, p in mods.items():
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            target = None
            if isinstance(node, ast.ImportFrom):
                if node.level:
                    target = _resolve_relative(m, node.level, node.module)
                elif node.module and node.module.startswith(PKG_NAME):
                    target = node.module
                if target:
                    # `from pkg.x import y` may be importing submodule pkg.x.y
                    for a in node.names:
                        if (cand := f"{target}.{a.name}") in mods:
                            edges[m].add(cand)
            elif isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(PKG_NAME):
                        target = a.name
            if target and target in mods:
                edges[m].add(target)
    return mods, edges


def _reach(seeds, mods, edges) -> set[str]:
    seen: set[str] = set()
    q = deque(s for s in seeds if s in mods)
    while q:
        cur = q.popleft()
        if cur in seen:
            continue
        seen.add(cur)
        q.extend(n for n in edges.get(cur, ()) if n not in seen)
    return seen


def unused_config_constants() -> list[str]:
    """Uppercase module-level names in config.py that nothing else reads."""
    cfg = PACKAGE_ROOT / "core" / "config.py"
    names = [
        t.id
        for n in ast.parse(cfg.read_text()).body
        if isinstance(n, ast.Assign)
        for t in n.targets
        if isinstance(t, ast.Name) and t.id.isupper()
    ]
    blob = "\n".join(
        p.read_text(errors="ignore")
        for p in PACKAGE_ROOT.rglob("*.py")
        if "__pycache__" not in p.parts and p != cfg
    )
    return [n for n in names if not re.search(rf"\b{n}\b", blob)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="emit machine-readable output")
    ap.add_argument("--config", action="store_true", help="also audit config constants")
    args = ap.parse_args()

    mods, edges = build_graph()
    live = _reach(ENTRY_POINTS, mods, edges)
    tests = _reach([m for m in mods if ".tests" in m], mods, edges) - live
    evals = _reach([m for m in mods if ".scripts.eval." in m], mods, edges) - live - tests

    # A package __init__ runs whenever a submodule is imported, so static
    # analysis never sees an edge into it. Not dead.
    orphans = sorted(
        m for m in set(mods) - (live | tests | evals)
        if mods[m].name != "__init__.py"
    )

    result = {
        "modules": len(mods),
        "live": sorted(live),
        "test_only": sorted(tests),
        "eval_only": sorted(evals),
        "orphans": orphans,
    }
    if args.config:
        result["unused_config_constants"] = unused_config_constants()

    if args.json:
        print(json.dumps(result, indent=2))
        return 1 if orphans else 0

    print(f"{len(mods)} modules")
    print(f"  live (from process_patent) : {len(live)}")
    print(f"  test-only                  : {len(tests)}")
    print(f"  eval-script-only           : {len(evals)}")
    print(f"  ORPHANS                    : {len(orphans)}")
    for o in orphans:
        loc = len(mods[o].read_text(errors="ignore").splitlines())
        print(f"      {loc:5d} LOC  {o}")
    if args.config:
        unused = result["unused_config_constants"]
        print(f"  unused config constants    : {len(unused)}")
        for c in unused:
            print(f"      {c}")
    return 1 if orphans else 0


if __name__ == "__main__":
    raise SystemExit(main())
