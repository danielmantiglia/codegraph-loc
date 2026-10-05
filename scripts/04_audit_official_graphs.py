"""Audit of the released competition graphs against gold edit locations.

For every base commit used by the 129 public tasks:
  1. extract the repository at that commit from a public git clone;
  2. rebuild the graph with cgl (package-style module ids, like the released graphs);
  3. compare node sets and `calls` edges with the released graph;
  4. derive gold edit locations from each task's reference patch and check
     whether the released graph (and ours) contains them.

Must run where the competition data lives: it reads the data but never copies it.
Resumable: one JSON per commit in <out>/commits/; call again until it prints "ALL DONE",
then run with --summarize.

Usage:
  python scripts/04_audit_official_graphs.py --data ~/Desktop/Kaggle/dati/competition \
         --repos ~/repos --out results/audit [--max-seconds 160] [--summarize]
"""
import argparse
import io
import json
import subprocess
import tarfile
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from cgl.graph import CodeGraphBuilder, flat_id, module_name_for, official_view

# Id convention of the released graphs (reverse-engineered on all 127 commits): module = import
# path inside packages, else repo path without a leading "src/" or "docs_src/"; see cgl.graph.flat_id.
MODULE_KW = dict(scheme="official", strip_prefixes=("src", "docs_src"))
from cgl.labels import def_table, is_test_path, locate_changes, parse_unified_diff

REPO_DIR = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich",
            "psf/requests": "requests", "encode/httpx": "httpx"}


def official_kind(text: str) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if not s or s.startswith(("@", "#")):
            continue
        if s.startswith("class "):
            return "class"
        if s.startswith("async def "):
            return "async"
        if s.startswith("def "):
            return "def"
        return "other"
    return "empty"


def extract(repo_path: Path, commit: str, dest: str):
    data = subprocess.run(["git", "-C", str(repo_path), "archive", commit],
                          check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        members = [m for m in tf.getmembers() if m.isfile() or m.isdir()]  # skip symlinks
        try:
            tf.extractall(dest, members=members, filter="data")
        except TypeError:  # very old Python
            tf.extractall(dest, members=members)


def gold_for(patch: str, root: Path) -> dict:
    out = {}
    for fc in parse_unified_diff(patch):
        if fc.is_new_file or not fc.path.endswith(".py") or is_test_path(fc.path):
            continue
        f = root / fc.path
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        mid, _ = module_name_for(f, root, **MODULE_KW)
        try:
            loc = locate_changes(fc, src, mid)
            table = def_table(src, mid)
        except SyntaxError:
            continue
        kind_of = lambda i: "module" if i == mid else table.get(i, {}).get("kind")
        for g in loc["functions"]:
            t = table.get(g)
            gid = flat_id(g, kind_of, mid)
            out[gid] = {"full_id": g, "kind": t["kind"] if t else "module",
                        "is_async": bool(t and t["is_async"]),
                        "nested": gid != g, "file": fc.path}
    return out


def audit_commit(short: str, commit: str, tasks: list, data: Path, repos: Path) -> dict:
    og = json.load(open(data / "graphs" / f"{short}_{commit}.json"))
    O = {n["id"]: n for n in og["nodes"]}
    okinds = Counter(official_kind(n.get("text", "")) for n in og["nodes"])
    oedges = {(e["source"], e["target"]) for e in og.get("edges", og.get("links", []))
              if e.get("type") == "calls"}
    otypes = Counter(e.get("type") for e in og.get("edges", og.get("links", [])))
    rec = {"repo": short, "commit": commit, "official": {
        "nodes": len(O), "kinds": dict(okinds), "edge_types": dict(otypes)}}
    with tempfile.TemporaryDirectory() as tmp:
        extract(repos / short, commit, tmp)
        root = Path(tmp)
        t0 = time.time()
        b = CodeGraphBuilder(root, module_scheme="official", strip_prefixes=MODULE_KW["strip_prefixes"])
        G = official_view(b.build())
        rec["build_seconds"] = round(time.time() - t0, 2)
        kind = {n: a["kind"] for n, a in G.nodes(data=True)}
        is_async = {n: a["is_async"] for n, a in G.nodes(data=True)}
        defs = {n for n, k in kind.items() if k in ("class", "function", "method")}
        sync_defs = {n for n in defs if not is_async[n]}
        inter = set(O) & defs
        official_only = sorted(set(O) - defs)
        ours_only_sync = sorted(sync_defs - set(O))
        our_calls = {(u, v) for u, v, a in G.edges(data=True) if a["type"] == "calls"}
        our_calls_in_O = {(u, v) for u, v in our_calls if u in O and v in O}
        rec["ours"] = {
            "defs": len(defs), "sync_defs": len(sync_defs),
            "async_defs": sum(is_async[n] for n in defs), "modules": sum(k == "module" for k in kind.values()),
            "edge_types": dict(Counter(a["type"] for *_, a in G.edges(data=True))),
        }
        rec["match"] = {
            "official_in_ours": len(inter), "official_only": len(official_only),
            "official_only_examples": official_only[:15],
            "ours_sync_not_in_official": len(ours_only_sync),
            "ours_sync_not_in_official_examples": ours_only_sync[:15],
            "calls_official": len(oedges), "calls_ours_between_official_nodes": len(our_calls_in_O),
            "calls_overlap": len(oedges & our_calls_in_O),
        }
        rec["tasks"] = []
        for t in tasks:
            gold = gold_for(t["patch"], root)
            items = []
            for g, info in sorted(gold.items()):
                items.append({"id": g, **info, "in_official": g in O, "in_ours": g in G})
            n = len(items)
            rec["tasks"].append({
                "instance_id": t["instance_id"], "n_gold": n,
                "n_in_official": sum(i["in_official"] for i in items),
                "n_in_ours": sum(i["in_ours"] for i in items),
                "gold": items,
            })
    return rec


def summarize(out: Path):
    rng = np.random.default_rng(0)

    def ci(x):
        x = np.asarray(x, float)
        if len(x) == 0:
            return [None, None, None]
        bs = rng.choice(x, (5000, len(x))).mean(1)
        return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4)]

    recs = [json.load(open(p)) for p in sorted((out / "commits").glob("*.json"))]
    tasks = [t | {"repo": r["repo"]} for r in recs for t in r["tasks"]]
    with_gold = [t for t in tasks if t["n_gold"] > 0]
    reasons = Counter()
    for t in with_gold:
        for g in t["gold"]:
            if not g["in_official"]:
                reasons["async" if g["is_async"] else "module-level" if g["kind"] == "module"
                        else "nested" if g["nested"] else "other"] += 1
    summ = {
        "n_commits": len(recs), "n_tasks": len(tasks), "n_tasks_with_py_gold": len(with_gold),
        "tasks_by_repo": dict(Counter(t["repo"] for t in tasks)),
        "all_gold_in_official": ci([t["n_in_official"] == t["n_gold"] for t in with_gold]),
        "no_gold_in_official": ci([t["n_in_official"] == 0 for t in with_gold]),
        "all_gold_in_ours": ci([t["n_in_ours"] == t["n_gold"] for t in with_gold]),
        "no_gold_in_ours": ci([t["n_in_ours"] == 0 for t in with_gold]),
        "gold_nodes_total": sum(t["n_gold"] for t in with_gold),
        "gold_missing_from_official_by_reason": dict(reasons),
        "official_nodes_reproduced_by_cgl": ci([r["match"]["official_in_ours"] / max(1, r["official"]["nodes"]) for r in recs]),
        "official_kinds_total": dict(sum((Counter(r["official"]["kinds"]) for r in recs), Counter())),
        "official_edge_types_total": dict(sum((Counter(r["official"]["edge_types"]) for r in recs), Counter())),
        "cgl_async_defs_total": sum(r["ours"]["async_defs"] for r in recs),
        "calls_official_total": sum(r["match"]["calls_official"] for r in recs),
        "calls_overlap_total": sum(r["match"]["calls_overlap"] for r in recs),
        "calls_ours_between_official_nodes_total": sum(r["match"]["calls_ours_between_official_nodes"] for r in recs),
    }
    by_repo = defaultdict(list)
    for t in with_gold:
        by_repo[t["repo"]].append(t)
    summ["by_repo"] = {r: {"n": len(ts),
                           "all_gold_in_official": ci([t["n_in_official"] == t["n_gold"] for t in ts]),
                           "all_gold_in_ours": ci([t["n_in_ours"] == t["n_gold"] for t in ts])}
                       for r, ts in sorted(by_repo.items())}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--max-seconds", type=float, default=160)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    data, repos, out = Path(a.data).expanduser(), Path(a.repos).expanduser(), Path(a.out)
    (out / "commits").mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize(out)
        return
    tasks = [json.loads(l) for l in open(data / "tasks.jsonl")]
    by_commit = defaultdict(list)
    for t in tasks:
        by_commit[(REPO_DIR[t["repo"]], t["base_commit"])].append(t)
    start = time.time()
    todo = [(s, c) for (s, c) in sorted(by_commit) if not (out / "commits" / f"{s}_{c}.json").exists()]
    print(f"{len(by_commit)} commits, {len(todo)} to do", flush=True)
    for s, c in todo:
        if time.time() - start > a.max_seconds:
            print("time budget reached; run again to continue", flush=True)
            return
        rec = audit_commit(s, c, by_commit[(s, c)], data, repos)
        (out / "commits" / f"{s}_{c}.json").write_text(json.dumps(rec))
        m = rec["match"]
        print(f"{s} {c[:10]} official={rec['official']['nodes']} in_ours={m['official_in_ours']} "
              f"tasks={len(rec['tasks'])} ({rec['build_seconds']}s)", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
