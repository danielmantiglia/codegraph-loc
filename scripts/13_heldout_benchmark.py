"""Experiment 4 — held-out confirmation (pre-registered: docs/preregistration_heldout.md, 5 Oct 2026).

Two held-out splits, built only from public data:
  * swebl    — SWE-bench Lite test split (princeton-nlp/SWE-bench_Lite), minus the 6 psf/requests
               instances; query = problem_statement; gold = non-test edits of the reference patch;
  * pymatgen — commits mined from materialsproject/pymatgen with 02_mine_commits.py (2019-01-01 →
               2026-03-02, PR merges included); query = PR title + description (08_fetch_pr_texts.py).
Every task is scored with exactly the retrieval methods of Experiment 2 (cgl.pipeline.cgl_rankings),
with and without test code in the ranking ("|notest"). The Gemma 4 prompts of each task (Experiment 3
design, built by 10_build_llm_prompts.prompt_records on the same graph, so each graph is built once) are
written to <out>/prompts/<task_id>.jsonl; `--merge-prompts` concatenates them into <out>/prompts.jsonl.

Usage:
  python scripts/13_heldout_benchmark.py --swebench ../data/swebench_lite_test.jsonl \
      --pymatgen results/heldout/pymatgen --repos ../repos --out results/heldout/loc [--max-seconds 3000]
  python scripts/13_heldout_benchmark.py --out results/heldout/loc --summarize
"""
import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from cgl.bench import gold_ranks, metrics_from_ranks
from cgl.pipeline import build_cgl, cgl_rankings, extract, gold_for, is_test_node

_spec = importlib.util.spec_from_file_location("build_prompts", Path(__file__).with_name("10_build_llm_prompts.py"))
_b = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_b)
prompt_records = _b.prompt_records

# Speed-up only (outputs unchanged): the same node texts are tokenized three times per task
# (retrieval + two candidate sets), so tokenization is memoized within a task.
import functools  # noqa: E402

from cgl import bench as _bench, pipeline as _pipeline  # noqa: E402

_tok = functools.lru_cache(maxsize=None)(lambda s: tuple(_bench.tokenize(s)))


def _cached_tokenize(s):
    return list(_tok(s))


_pipeline.tokenize = _cached_tokenize
_b.tokenize = _cached_tokenize

MIN_QUERY_CHARS = 20
sys.setrecursionlimit(20000)  # some sympy snapshots nest deeply enough to exceed the default AST-visitor depth
EXCLUDED_REPOS = {"psf/requests"}  # development repository (pre-registration §2)


def load_inputs(swebench: str | None, pymatgen: str | None):
    rows = []
    if swebench:
        for r in map(json.loads, open(swebench)):
            if r["repo"] in EXCLUDED_REPOS:
                continue
            rows.append({"split": "swebl", "task_id": r["instance_id"], "repo": r["repo"].split("/")[1],
                         "parent": r["base_commit"], "patch": r["patch"], "query": r["problem_statement"] or ""})
    if pymatgen:
        d = Path(pymatgen)
        prs = {}
        for r in map(json.loads, open(d / "pr_texts.jsonl")):
            if r.get("status") == 200 and r.get("title") is not None:
                prs[r["pr"]] = (r["title"] + "\n\n" + (r["body"] or "")).strip()
        for r in map(json.loads, open(d / "mined_pymatgen.jsonl")):
            q = prs.get(r.get("pr"))
            if q and len(q) >= MIN_QUERY_CHARS:
                rows.append({"split": "pymatgen", "task_id": f"pymatgen_{r['pr']}", "repo": "pymatgen",
                             "parent": r["parent"], "sha": r["sha"], "query": q})
    return rows


def run_task(t, repos: Path, prompts_only: bool = False):
    """Score one task; returns (record, prompt records). With prompts_only the rankings are skipped."""
    _tok.cache_clear()
    repo_path = repos / t["repo"]
    diff = t.get("patch") or subprocess.run(
        ["git", "-C", str(repo_path), "diff", t["parent"], t["sha"]],
        check=True, capture_output=True, text=True, errors="replace").stdout
    with tempfile.TemporaryDirectory() as tmp:
        extract(repo_path, t["parent"], tmp)
        root = Path(tmp)
        gold = gold_for(diff, root)
        if not gold:
            return None, None
        G = build_cgl(root)
    prompts = prompt_records(G, t["split"], t["task_id"], t["query"], sorted(gold))
    if prompts_only:
        return None, prompts
    files = {n: G.nodes[n].get("file") for n in G.nodes}
    rankings = cgl_rankings(G, t["query"])
    rankings.update({f"{m}|notest": [n for n in rk if not is_test_node(n, files.get(n))]
                     for m, rk in list(rankings.items())})
    gold_ids = sorted(gold)
    gold_files = sorted(set(gold.values()))
    kind = {g: ("missing" if g not in G else "module" if G.nodes[g].get("kind") == "module"
                else "async" if G.nodes[g].get("is_async") else "sync") for g in gold_ids}
    rec = {k: t[k] for k in ("split", "task_id", "repo", "parent", "query")}
    rec.update({"n_nodes": G.number_of_nodes(), "n_gold": len(gold_ids), "gold": gold_ids,
                "gold_files": gold_files, "gold_kind": kind, "methods": {}})
    for m, rk in rankings.items():
        flist = list(dict.fromkeys(files.get(n) for n in rk if files.get(n)))
        rec["methods"][m] = {"ranks": gold_ranks(rk, gold_ids), "file_ranks": gold_ranks(flist, gold_files),
                             "n_ranked": len(rk), "top5": rk[:5]}
    return rec, prompts


# Pre-registered confirmatory contrasts (§5) and the Experiment 2 contrasts (secondary, §6).
H1 = ("rrf_cgl-calls_only|notest", "rrf_cgl-official_schema|notest", "recall@10")
H2 = ("rrf_cgl-calls_only|notest", "rrf_cgl|notest", "mrr")


def summarize(out: Path):
    recs = [json.load(open(p)) for p in sorted((out / "tasks").glob("*.json"))]
    methods = list(recs[0]["methods"])
    keys = ["hit@1", "hit@5", "hit@10", "recall@5", "recall@10", "mrr"]
    summ = {"preregistered": {}, "splits": {}}
    groups = {"pooled": list(range(len(recs)))}
    for i, r in enumerate(recs):
        groups.setdefault(r["split"], []).append(i)
    for gname, ix in groups.items():
        rng = np.random.default_rng(0)
        sub = [recs[i] for i in ix]
        n = len(sub)
        boot = rng.integers(0, n, (5000, n))

        def ci(x):
            bs = x[boot].mean(1)
            return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
                    round(float(np.percentile(bs, 97.5)), 4)]

        per = {k: {m: np.array([metrics_from_ranks(r["methods"][m]["ranks"])[k] for r in sub]) for m in methods}
               for k in keys}
        filek = {m: np.array([metrics_from_ranks(r["methods"][m]["file_ranks"])["hit@5"] for r in sub])
                 for m in methods}
        kinds = [k for r in sub for k in r["gold_kind"].values()]
        S = {"n_tasks": n,
             "tasks_by_repo": dict(sorted({r["repo"]: sum(x["repo"] == r["repo"] for x in sub) for r in sub}.items())),
             "gold_locations": len(kinds),
             "share_gold_async": round(kinds.count("async") / len(kinds), 4),
             "share_gold_module": round(kinds.count("module") / len(kinds), 4),
             "share_tasks_any_async_or_module_gold": round(float(np.mean(
                 [any(k in ("async", "module") for k in r["gold_kind"].values()) for r in sub])), 4),
             "by_method": {m: {k: ci(per[k][m]) for k in keys} | {"file_hit@5": ci(filek[m])} for m in methods},
             "paired_differences": {}}
        for a, b, k in (H1, H2):
            S["paired_differences"][f"{a} - {b} [{k}]"] = ci(per[k][a] - per[k][b])
        for suf in ("", "|notest"):
            contrasts = [(f"bm25_cgl{suf}", f"bm25_official_schema{suf}"),
                         (f"ident+ppr_cgl{suf}", f"ident+ppr_official_schema{suf}"),
                         (f"rrf_cgl{suf}", f"rrf_cgl-official_schema{suf}"),
                         (f"rrf_cgl-calls_only{suf}", f"rrf_cgl-official_schema{suf}"),
                         (f"rrf_cgl{suf}", f"bm25_cgl{suf}")]
            contrasts += [(f"rrf_cgl{suf}", f"rrf_cgl-{x}{suf}") for x in
                          ("no_async", "no_module", "no_contains", "no_imports", "no_inherits", "calls_only")]
            for a, b in contrasts:
                for k in ("hit@5", "recall@10", "mrr"):
                    S["paired_differences"][f"{a} - {b} [{k}]"] = ci(per[k][a] - per[k][b])
        summ["splits"][gname] = S
    P = summ["splits"]["pooled"]["paired_differences"]
    for name, (a, b, k), rule in (("H1", H1, "> 0"), ("H2", H2, "> 0")):
        v = P[f"{a} - {b} [{k}]"]
        summ["preregistered"][name] = {"contrast": f"{a} - {b}", "metric": k, "predicted": rule,
                                       "estimate_ci": v, "confirmed": v[1] > 0}
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--swebench", default=None)
    ap.add_argument("--pymatgen", default=None, help="directory with mined_pymatgen.jsonl and pr_texts.jsonl")
    ap.add_argument("--repos", default="../repos")
    ap.add_argument("--out", default="results/heldout/loc")
    ap.add_argument("--max-seconds", type=float, default=3000)
    ap.add_argument("--only-split", default=None)
    ap.add_argument("--shard", default="0/1", help="K/N: process every N-th task starting at K (parallel runs)")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--merge-prompts", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    if a.merge_prompts:
        files = sorted((out / "prompts").glob("*.jsonl"))
        with open(out / "prompts.jsonl", "w") as fo:
            for f in files:
                fo.write(f.read_text())
        print(f"{len(files)} tasks -> {out / 'prompts.jsonl'}")
        return
    if a.summarize:
        s = summarize(out)
        for g, S in s["splits"].items():
            print(f"[{g}] n={S['n_tasks']} async gold={S['share_gold_async']} module gold={S['share_gold_module']}")
        for h, v in s["preregistered"].items():
            print(h, v["contrast"], v["metric"], v["estimate_ci"], "CONFIRMED" if v["confirmed"] else "not confirmed")
        return
    rows = load_inputs(a.swebench, a.pymatgen)
    if a.only_split:
        rows = [r for r in rows if r["split"] == a.only_split]
    done = {p.stem for p in (out / "tasks").glob("*.json")}
    prompted = {p.stem for p in (out / "prompts").glob("*.jsonl")}
    skipped_file = out / "skipped.txt"
    skipped = set(skipped_file.read_text().split()) if skipped_file.exists() else set()
    k, n = map(int, a.shard.split("/"))  # shards split the tasks still to do, so a rerun is balanced too
    todo = [r for r in rows if r["task_id"] not in (done & prompted) | skipped]
    todo = [r for i, r in enumerate(todo) if i % n == k]
    print(f"{len(rows)} tasks, {len(todo)} to do", flush=True)
    start = time.time()
    for t in todo:
        if time.time() - start > a.max_seconds:
            print("time budget reached; run again to continue", flush=True)
            return
        t0 = time.time()
        only = t["task_id"] in done  # scored before prompts were added: build the prompts only
        try:
            rec, prompts = run_task(t, Path(a.repos).expanduser(), prompts_only=only)
        except subprocess.CalledProcessError as e:  # e.g. clone missing or incomplete: retried on the next run
            print(f"  ! {t['task_id']}: {e}", flush=True)
            continue
        if prompts is None:  # no non-test, pre-existing .py edit in the patch
            with open(skipped_file, "a") as fs:
                fs.write(t["task_id"] + "\n")
            continue
        if rec is not None:
            (out / "tasks" / f"{t['task_id']}.json").write_text(json.dumps(rec))
        (out / "prompts" / f"{t['task_id']}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in prompts))
        print(f"  {t['task_id']}: {'prompts only' if only else str(rec['n_nodes']) + ' nodes'}, {time.time() - t0:.1f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
