"""Fix profiles of the 16 repositories used in the paper (4 development + 12 held-out).

Runs ``cgl.profile`` on each clone with a fixed window (2019-01-01 to 2026-09-30, GitHub merge commits
included) and writes results/profiles/<repo>.json plus results/profiles/summary.json.

Usage: python scripts/16_profile_repos.py --repos ../repos [--jobs 2]
"""
import argparse
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from cgl.mining import mine
from cgl.profile import detect_package, summarize

REPOS = ["fastapi", "httpx", "requests", "rich",                       # development
         "astropy", "django", "flask", "matplotlib", "pylint", "pytest",  # held-out: SWE-bench Lite
         "scikit-learn", "seaborn", "sphinx", "sympy", "xarray",
         "pymatgen"]                                                     # held-out: materials science
SINCE, UNTIL = "2019-01-01", "2026-09-30"
# pymatgen: same window as the pre-registered held-out set (pymatgen-core moved out of the repository on
# 2026-03-03, see the pre-registration); it is a namespace package, so the package is given explicitly.
OVERRIDES = {"pymatgen": {"package": "pymatgen", "until": "2026-03-02"}}


def one(args):
    root, out = args
    t0 = time.time()
    o = OVERRIDES.get(root.name, {})
    pkg = o.get("package") or detect_package(root)
    until = o.get("until", UNTIL)
    s = summarize(mine(root, pkg, SINCE, until, pr_merges=True, prefixes=("src", "lib")))
    rec = {"repository": root.name, "package": pkg, "since": SINCE, "until": until,
           "head": subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%H %cs"], capture_output=True,
                                  text=True).stdout.strip(),
           "elapsed_s": None, **s}
    rec["elapsed_s"] = round(time.time() - t0, 1)
    (out / f"{root.name}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="../repos")
    ap.add_argument("--out", default="results/profiles")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    todo = [(Path(a.repos) / r, out) for r in (a.only or REPOS)
            if (Path(a.repos) / r / ".git").exists() and not (out / f"{r}.json").exists()]
    with ProcessPoolExecutor(a.jobs) as ex:
        for rec in ex.map(one, todo):
            F = rec["share_of_fixes"]
            print(f"{rec['repository']:13s} pkg={rec['package']:12s} fixes={rec['n_fixes']:5d} "
                  f"async={F['touching async code']} module={F['touching module-level code']} "
                  f"covered={F['fully representable by a released-style graph']} ({rec['elapsed_s']} s)", flush=True)
    summary = {p.stem: json.load(open(p)) for p in sorted(out.glob("*.json")) if p.stem != "summary"}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
