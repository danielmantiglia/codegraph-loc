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
from pathlib import Path

from cgl.mining import mine

REPOS = {"fastapi": "fastapi", "rich": "rich", "requests": "requests", "httpx": "httpx"}


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
