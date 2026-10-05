"""Mine bug-fix-like commits from public git history and derive function-level gold labels.

A candidate is a non-merge commit on the default branch that
  * modifies (not only adds) >= 1 library .py file and
  * touches >= 1 test .py file (the same "code + tests" signal used to build the Kaggle tasks),
  * changes <= MAX_FILES library files and <= MAX_LINES library lines (no mass refactors),
  * is not a release / dependency bump / docs / formatting / translation commit.

Only public GitHub history is used, so the output can be redistributed.

Usage: python scripts/02_mine_commits.py --repos-dir ../repos --since 2019-01-01
Held-out repositories (Experiment 4), e.g. pymatgen, which merges PRs with both squash and merge commits:
       python scripts/02_mine_commits.py --repos-dir ../repos --only pymatgen=pymatgen \
              --since 2019-01-01 --until 2026-03-02 --pr-merges --out-dir results/heldout/pymatgen
With --pr-merges, a first-parent merge commit "Merge pull request #N ..." is one candidate, diffed against its
first parent; its title is the first line of the merge message body (the PR title).
"""
import argparse
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from cgl.graph import module_name_for
from cgl.labels import def_table, is_test_path, locate_changes, parse_unified_diff

REPOS = {"fastapi": "fastapi", "rich": "rich", "requests": "requests", "httpx": "httpx"}
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


def lib_file(path: str, pkg: str) -> bool:
    parts = path.split("/")
    if parts[0] == "src":
        parts = parts[1:]
    return path.endswith(".py") and parts[0] == pkg and not is_test_path(path)


def mine(root: Path, pkg: str, since: str, until: str | None = None, pr_merges: bool = False):
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
            elif lib_file(path, pkg):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-dir", default="../repos")
    ap.add_argument("--since", default="2019-01-01")
    ap.add_argument("--until", default=None)
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--only", action="append", default=[], metavar="NAME=PKG",
                    help="mine these repositories instead of the four development ones")
    ap.add_argument("--pr-merges", action="store_true", help="also keep first-parent 'Merge pull request' commits")
    a = ap.parse_args()
    repos = dict(x.split("=", 1) for x in a.only) if a.only else REPOS
    summary = {}
    for name, pkg in repos.items():
        rows = mine(Path(a.repos_dir) / name, pkg, a.since, a.until, a.pr_merges)
        Path(a.out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(a.out_dir) / f"mined_{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        n = len(rows)
        fix = [r for r in rows if r["fix_like"]]
        def share(rs, pred):
            return f"{sum(map(pred, rs))}/{len(rs)}" if rs else "0/0"
        summary[name] = dict(
            candidates=n, fix_like=len(fix),
            any_async=share(rows, lambda r: bool(r["gold_async"])),
            any_module_level=share(rows, lambda r: r["gold_kinds"].get("module", 0) > 0),
            only_module_level=share(rows, lambda r: set(r["gold_kinds"]) == {"module"}),
        )
        print(name, summary[name])
    Path(a.out_dir, "mined_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
