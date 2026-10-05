"""Localization benchmark on the public-history split (commits mined from GitHub, outside
the 129 competition tasks). Needs only public data: git clones + fetched PR texts.

Query = PR title + description (same construction as the competition's problem statements).
Gold = edited functions/classes/modules of the commit's non-test diff, competition id convention.
Methods = the cgl-only methods of Experiment 1 (cgl.pipeline.cgl_rankings), including
``*_official_schema`` variants that emulate the released graph's coverage
(sync defs only, no module nodes, `calls` edges only). Every method is also scored with
test-code nodes removed from the ranking ("|notest").

Resumable (one JSON per task in <out>/tasks/).
Usage:
  python scripts/09_public_history_benchmark.py --mined results --prs results/public/pr_texts.jsonl \
         --repos ../repos --out results/public_loc [--max-seconds 150]
  python scripts/09_public_history_benchmark.py --out results/public_loc --summarize
"""
import argparse
import json
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cgl.bench import gold_ranks, metrics_from_ranks
from cgl.pipeline import build_cgl, cgl_rankings, extract, gold_for, is_test_node

MIN_QUERY_CHARS = 20


def run_task(row, query, repos: Path):
    repo_path = repos / row["repo"]
    diff = subprocess.run(["git", "-C", str(repo_path), "diff", row["parent"], row["sha"]],
                          check=True, capture_output=True, text=True, errors="replace").stdout
    with tempfile.TemporaryDirectory() as tmp:
        extract(repo_path, row["parent"], tmp)
        root = Path(tmp)
        gold = gold_for(diff, root)
        if not gold:
            return None
        G = build_cgl(root)
    files = {n: G.nodes[n].get("file") for n in G.nodes}
    rankings = cgl_rankings(G, query)
    rankings.update({f"{m}|notest": [n for n in rk if not is_test_node(n, files.get(n))]
                     for m, rk in list(rankings.items())})
    gold_ids = sorted(gold)
    gold_files = sorted(set(gold.values()))
    rec = {"task_id": f"{row['repo']}_{row['pr']}", "repo": row["repo"], "pr": row["pr"], "sha": row["sha"],
           "parent": row["parent"], "date": row["date"], "n_gold": len(gold_ids), "gold": gold_ids,
           "gold_files": gold_files, "gold_in_graph": [g in G for g in gold_ids], "methods": {}}
    for m, rk in rankings.items():
        flist = list(dict.fromkeys(files.get(n) for n in rk if files.get(n)))
        rec["methods"][m] = {"ranks": gold_ranks(rk, gold_ids), "file_ranks": gold_ranks(flist, gold_files),
                             "n_ranked": len(rk), "top5": rk[:5]}
    return rec


def summarize(out: Path):
    rng = np.random.default_rng(0)
    recs = [json.load(open(p)) for p in sorted((out / "tasks").glob("*.json"))]
    methods = list(recs[0]["methods"])
    boot = rng.integers(0, len(recs), (5000, len(recs)))

    def ci(x):
        bs = x[boot].mean(1)
        return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    keys = ["hit@1", "hit@5", "hit@10", "recall@5", "recall@10", "mrr"]
    per = {k: {m: np.array([metrics_from_ranks(r["methods"][m]["ranks"])[k] for r in recs]) for m in methods} for k in keys}
    filek = {m: np.array([metrics_from_ranks(r["methods"][m]["file_ranks"])["hit@5"] for r in recs]) for m in methods}
    summ = {"n_tasks": len(recs), "tasks_by_repo": {}, "by_method": {}, "paired_differences": {}, "by_repo_hit@5": {}}
    for r in recs:
        summ["tasks_by_repo"][r["repo"]] = summ["tasks_by_repo"].get(r["repo"], 0) + 1
    for m in methods:
        summ["by_method"][m] = {k: ci(per[k][m]) for k in keys} | {"file_hit@5": ci(filek[m])}
    contrasts = []
    for suf in ("", "|notest"):
        contrasts += [(f"bm25_cgl{suf}", f"bm25_official_schema{suf}"),
                      (f"ident+ppr_cgl{suf}", f"ident+ppr_official_schema{suf}"),
                      (f"rrf_cgl{suf}", f"rrf_cgl-official_schema{suf}"),
                      (f"rrf_cgl{suf}", f"bm25_cgl{suf}")]
        contrasts += [(f"rrf_cgl{suf}", f"rrf_cgl-{a}{suf}") for a in
                      ("no_async", "no_module", "no_contains", "no_imports", "no_inherits", "calls_only")]
    for a, b in contrasts:
        for k in ("hit@5", "recall@10", "mrr"):
            summ["paired_differences"][f"{a} - {b} [{k}]"] = ci(per[k][a] - per[k][b])
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[r["repo"]].append(i)
    for repo, ix in by.items():
        summ["by_repo_hit@5"][repo] = {"n": len(ix), **{m: round(float(per["hit@5"][m][ix].mean()), 4) for m in methods}}
    summ["share_gold_all_in_cgl"] = round(float(np.mean([all(r["gold_in_graph"]) for r in recs])), 4)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default="results")
    ap.add_argument("--prs", default="results/public/pr_texts.jsonl")
    ap.add_argument("--repos", default="../repos")
    ap.add_argument("--out", default="results/public_loc")
    ap.add_argument("--max-seconds", type=float, default=150)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    if a.summarize:
        s = summarize(out)
        print("n =", s["n_tasks"], s["tasks_by_repo"])
        for m, v in s["by_method"].items():
            print(f"{m:36s} hit@1={v['hit@1'][0]:.3f} hit@5={v['hit@5'][0]:.3f} [{v['hit@5'][1]:.3f},{v['hit@5'][2]:.3f}] "
                  f"rec@10={v['recall@10'][0]:.3f} mrr={v['mrr'][0]:.3f} file@5={v['file_hit@5'][0]:.3f}")
        return
    prs = {}
    for l in open(a.prs):
        r = json.loads(l)
        if r.get("status") == 200 and r.get("title") is not None:
            prs[(r["repo"], r["pr"])] = (r["title"] + "\n\n" + (r["body"] or "")).strip()
    rows = []
    for f in sorted(Path(a.mined).glob("mined_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            q = prs.get((r["repo"], r.get("pr")))
            if q and len(q) >= MIN_QUERY_CHARS:
                rows.append((r, q))
    done = {p.stem for p in (out / "tasks").glob("*.json")}
    skipped_file = out / "skipped.txt"
    skipped = set(skipped_file.read_text().split()) if skipped_file.exists() else set()
    todo = [(r, q) for r, q in rows if f"{r['repo']}_{r['pr']}" not in done | skipped]
    print(f"{len(rows)} tasks with PR text, {len(todo)} to do", flush=True)
    start = time.time()
    for r, q in todo:
        if time.time() - start > a.max_seconds:
            print("time budget reached; run again to continue", flush=True)
            return
        rec = run_task(r, q, Path(a.repos).expanduser())
        tid = f"{r['repo']}_{r['pr']}"
        if rec is None:
            with open(skipped_file, "a") as fs:
                fs.write(tid + "\n")
            continue
        (out / "tasks" / f"{tid}.json").write_text(json.dumps(rec))
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
