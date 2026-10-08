"""Mining of small code+test commits from a repository's git history, with gold labels.

A commit qualifies if it modifies at least one library file and at least one test file, touches at most
5 library files and 200 library lines, and its title passes a skip-list (releases, docs, formatting...).
Gold locations are the innermost definitions enclosing each changed line in the pre-fix snapshot
(:func:`cgl.labels.locate_changes`). Used by ``scripts/02_mine_commits.py`` and ``cgl.profile``.
"""
from __future__ import annotations

import re
import subprocess
from collections import Counter
from pathlib import Path

from .graph import module_name_for
from .labels import def_table, is_test_path, locate_changes, parse_unified_diff

MAX_FILES, MAX_LINES = 5, 200
SKIP_SUBJECT = re.compile(
    r"(^\s*(⬆|⬇|📝|🔧|🎨|👷|🌐|🔖|💚|🍱|🙈|📌|👥|🔨)|\bbump\b|\brelease\b|\bversion\b|\btypo|\blint|"
    r"\bformat|\bpre-commit\b|\bdocs?\b|\btranslat|\bchangelog\b|\breadme\b|\bmypy\b|\bruff\b|\bblack\b)",
    re.I)
FIX_LIKE = re.compile(r"(🐛|\bfix|\bbug|\berror|\bcrash|\bregression|\bhandle|\bcorrect|\bissue|\bbroken|\braise)", re.I)
PR_RE = re.compile(r"\(#(\d+)\)")
MERGE_PR = re.compile(r"^Merge pull request #(\d+)")


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, errors="replace")


def lib_file(path: str, pkg: str, prefixes=("src",)) -> bool:
    parts = path.split("/")
    if parts[0] in prefixes:
        parts = parts[1:]
    return path.endswith(".py") and parts[0] == pkg and not is_test_path(path)


def mine(root: Path, pkg: str, since: str, until: str | None = None, pr_merges: bool = False,
         prefixes=("src",)):
    args = ["log", "--first-parent", f"--since={since}"] + ([f"--until={until}"] if until else []) \
        + (["--diff-merges=first-parent"] if pr_merges else ["--no-merges"]) + ["--format=@@@%H|%P|%cs|%s", "--numstat"]
    log = git(root, *args)
    out = []
    for block in log.split("@@@")[1:]:
        head, *stat = block.strip().split("\n")
        sha, parents, date, subject = head.split("|", 3)
        parent = parents.split()[0] if parents else None
        merge_pr = None
        if len(parents.split()) > 1:  # only with --pr-merges: keep GitHub PR merges, title from the body
            m = MERGE_PR.match(subject)
            if not m:
                continue
            merge_pr = int(m.group(1))
            body = [l.strip() for l in git(root, "log", "-1", "--format=%b", sha).splitlines() if l.strip()]
            subject = f"{body[0] if body else ''} (#{merge_pr})"
        if not parent or SKIP_SUBJECT.search(subject):
            continue
        lib, tests, lib_lines = [], [], 0
        for row in stat:
            if not row.strip():
                continue
            add, dele, path = row.split("\t", 2)
            if "=>" in path:  # renames: skip commit (labels ambiguous)
                lib = None
                break
            if path.endswith(".py") and is_test_path(path):
                tests.append(path)
            elif lib_file(path, pkg, prefixes):
                lib.append(path)
                lib_lines += (int(add) if add.isdigit() else 0) + (int(dele) if dele.isdigit() else 0)
        if not lib or not tests or len(lib) > MAX_FILES or lib_lines > MAX_LINES:
            continue
        diff = git(root, "diff", "--unified=0", parent, sha, "--", *lib)
        gold_files, gold_funcs, gold_defs, gold_async, kinds = [], set(), set(), set(), Counter()
        mod_pts = Counter()
        for fc in parse_unified_diff(diff):
            if fc.is_new_file or not fc.path.endswith(".py"):
                continue
            try:
                old = git(root, "show", f"{parent}:{fc.path}")
            except subprocess.CalledProcessError:
                continue
            mid, _ = module_name_for(Path(fc.path), Path("."))
            try:
                loc = locate_changes(fc, old, mid)
                table = def_table(old, mid)
            except SyntaxError:
                continue
            gold_files.append(fc.path)
            gold_funcs |= loc["functions"]
            mod_pts.update(loc["module_points"])
            gold_defs |= loc["defs"]
            for g in loc["functions"]:
                k = table[g]["kind"] if g in table else "module"
                kinds[k] += 1
                if g in table and table[g]["is_async"]:
                    gold_async.add(g)
        if not gold_files:
            continue
        refs = PR_RE.findall(subject)  # GitHub appends the PR number last: "Revert "X (#1)" (#2)" -> #2
        pr = refs[-1] if refs else None
        out.append(dict(repo=pkg, sha=sha, parent=parent, date=date, subject=subject,
                        pr=int(pr) if pr else None, merge=merge_pr is not None,
                        fix_like=bool(FIX_LIKE.search(subject)),
                        n_lib_files=len(lib), n_lib_lines=lib_lines, test_files=tests,
                        gold_files=gold_files, gold_functions=sorted(gold_funcs), gold_defs=sorted(gold_defs),
                        gold_kinds=dict(kinds), gold_async=sorted(gold_async),
                        module_points=dict(mod_pts)))
    return out
