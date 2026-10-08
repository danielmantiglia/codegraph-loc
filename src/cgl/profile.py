"""What do bug fixes touch in a repository, and what must its code graph therefore contain?

Mines the repository's own history for small code+test commits (the same rules as the paper's
benchmark, :mod:`cgl.mining`), labels the definitions each fix edits, and reports which node types
those edit locations need:

* sync functions and methods, classes   -> present in the competition's released graph schema
* async functions and methods           -> missing from the released schema
* module-level code, including new      -> missing from the released schema (no module nodes)
  top-level functions and classes

It also reports the share of fixes whose edit locations a released-style graph (sync definitions only,
no module nodes) can fully represent. Shares of fixes come with bootstrap 95% CIs.

Usage:
    cgl-profile path/to/repo [--package PKG] [--since 2019-01-01] [--until YYYY-MM-DD]
                [--no-pr-merges] [--json profile.json]
    python -m cgl.profile path/to/repo ...

The repository must be a git clone with history. ``--package`` is the import package to analyse
(detected automatically when omitted). GitHub merge commits ("Merge pull request #N") are counted as
fixes too, for projects that do not squash-merge; ``--no-pr-merges`` turns this off.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from .labels import is_test_path
from .mining import mine

SKIP_DIRS = {"tests", "test", "testing", "docs", "doc", "examples", "example", "benchmarks", "scripts",
             "build", "dist", "site-packages"}


def detect_package(root: Path) -> str:
    """The top-level import package with the most non-test .py files (at the root or under src/)."""
    best, best_n = None, -1
    for base in (root, root / "src", root / "lib"):
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            if not d.is_dir() or d.name.startswith(".") or d.name in SKIP_DIRS:
                continue
            if not (d / "__init__.py").exists():
                continue
            n = sum(1 for p in d.rglob("*.py") if not is_test_path(p.relative_to(root).as_posix()))
            if n > best_n:
                best, best_n = d.name, n
    if best is None:  # namespace package without __init__.py, named like the repository
        for base in (root, root / "src", root / "lib"):
            if (base / root.name).is_dir():
                return root.name
        raise SystemExit(f"No Python package found in {root} (or its src/ or lib/ folder); pass --package.")
    return best


def _ci(flags, seed: int = 0, b: int = 2000):
    x = np.asarray(flags, float)
    if len(x) == 0:
        return None
    bs = x[np.random.default_rng(seed).integers(0, len(x), (b, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def summarize(rows: list[dict]) -> dict:
    """Location- and fix-level shares of the node types that fixes edit.

    A "fix" is a small code+test commit (see :mod:`cgl.mining`). Commits whose edits are all incidental
    (e.g. import-only) have no edit location and are left out, as in the benchmark.
    """
    rows = [r for r in rows if sum(r["gold_kinds"].values()) > 0]
    kinds, n_async = Counter(), 0
    any_async, any_module, covered, none_covered = [], [], [], []
    for r in rows:
        kinds.update(r["gold_kinds"])
        n_async += len(r["gold_async"])
        n_gold = sum(r["gold_kinds"].values())
        n_mod = r["gold_kinds"].get("module", 0)
        n_hidden = n_mod + len(r["gold_async"])
        any_async.append(bool(r["gold_async"]))
        any_module.append(n_mod > 0)
        covered.append(n_hidden == 0)
        none_covered.append(n_gold > 0 and n_hidden >= n_gold)
    n_loc = sum(kinds.values())
    n_mod = kinds.get("module", 0)
    return {
        "n_fixes": len(rows),
        "n_edit_locations": n_loc,
        "edit_locations": {
            "sync functions/methods": n_loc - n_mod - kinds.get("class", 0) - n_async,
            "async functions/methods": n_async,
            "classes (class body)": kinds.get("class", 0),
            "module-level code (incl. new top-level definitions)": n_mod,
        },
        "share_of_edit_locations": {
            "async": round(n_async / n_loc, 4) if n_loc else None,
            "module-level": round(n_mod / n_loc, 4) if n_loc else None,
        },
        "share_of_fixes": {
            "touching async code": _ci(any_async),
            "touching module-level code": _ci(any_module),
            "fully representable by a released-style graph": _ci(covered),
            "not representable at all by a released-style graph": _ci(none_covered),
        },
    }


def _pct(v):
    return "n/a" if v is None else f"{100 * v:.1f}%"


def _pct_ci(v):
    return "n/a" if v is None else f"{100 * v[0]:.1f}% (95% CI {100 * v[1]:.1f}-{100 * v[2]:.1f})"


def report(name: str, pkg: str, since: str, until: str | None, s: dict) -> str:
    window = f"{since} to {until or 'today'}"
    if s["n_fixes"] == 0:
        return (f"{name} (package '{pkg}', {window}): no qualifying code+test commits found. "
                f"Try an earlier --since, --pr-merges, or check --package.")
    L, F = s["share_of_edit_locations"], s["share_of_fixes"]
    lines = [f"{name}: package '{pkg}', commits {window}",
             f"  {s['n_fixes']} fixes (small code+test commits), {s['n_edit_locations']} edited definitions",
             "",
             "  Edited definitions by node type:"]
    for k, v in s["edit_locations"].items():
        lines.append(f"    {k:52s} {v:6d}")
    lines += ["",
              f"  Async definitions:  {_pct(L['async'])} of edit locations; "
              f"{_pct_ci(F['touching async code'])} of fixes touch one.",
              f"  Module-level code:  {_pct(L['module-level'])} of edit locations; "
              f"{_pct_ci(F['touching module-level code'])} of fixes touch it.",
              "  A released-style graph (sync definitions only, no module nodes) contains every edit location",
              f"  for {_pct_ci(F['fully representable by a released-style graph'])} of fixes, "
              f"and none for {_pct_ci(F['not representable at all by a released-style graph'])}.",
              "",
              "  Reading: a node type that holds a non-negligible share of edit locations must be in the graph,",
              "  or an agent that searches the graph cannot reach those fixes. Module nodes also compete with",
              "  functions at the top of a ranking, so weigh their share before adding them."]
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="cgl-profile", description=__doc__.split("\n\n")[0])
    ap.add_argument("repo", help="path to a git clone of the repository")
    ap.add_argument("--package", default=None, help="import package to analyse (default: detected)")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--until", default=None)
    ap.add_argument("--pr-merges", action=argparse.BooleanOptionalAction, default=True,
                    help="also count GitHub 'Merge pull request' commits (default: on)")
    ap.add_argument("--json", default=None, help="also write the profile (and mined fixes) to this JSON file")
    a = ap.parse_args(argv)
    root = Path(a.repo).expanduser().resolve()
    if not (root / ".git").exists():
        raise SystemExit(f"{root} is not a git repository (a clone with history is needed).")
    pkg = a.package or detect_package(root)
    rows = mine(root, pkg, a.since, a.until, a.pr_merges, prefixes=("src", "lib"))
    s = summarize(rows)
    print(report(root.name, pkg, a.since, a.until, s))
    if a.json:
        out = {"repository": root.name, "package": pkg, "since": a.since, "until": a.until,
               "pr_merges": a.pr_merges, **s,
               "fixes": [{k: r[k] for k in ("sha", "date", "subject", "gold_functions", "gold_kinds", "gold_async")}
                         for r in rows]}
        Path(a.json).write_text(json.dumps(out, indent=1))
        print(f"\n  Written: {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
