"""Exploratory sensitivity analysis (not pre-registered): are the coverage effects driven by
module-level gold labels?

Module nodes exist only in the cgl graph, and the label audit found that module-level gold can
come from trivial edits (e.g. added ``__author__`` metadata). Here every module-level gold
location is removed; tasks left without gold are dropped. The main contrasts are then recomputed:
  * retrieval (non-test rankings): Recall@10 of BM25 and of BM25+PPR fusion with `calls`-only
    propagation, cgl minus the released schema;
  * Gemma 4 re-ranking: gold among candidates and Gemma Hit@5, cgl minus official.
Splits: public history (Experiments 2-3) and the held-out splits (Experiment 4).
Paired bootstrap, 5,000 resamples, seed 0.

Module-level ids: held-out records carry each gold location's kind; for the public split the
module id of every gold file is recomputed from the snapshot's file list with the competition
convention (cgl.graph.module_name_for logic: package path inside packages, else repo path).

Usage: python scripts/15_sensitivity_module_gold.py --repos ../repos [--out results/sensitivity_module_gold.json]
"""
import argparse
import json
import subprocess
from pathlib import Path, PurePosixPath

import numpy as np

from cgl.bench import metrics_from_ranks

STRIP = ("src", "docs_src")


def ci(x, seed=0):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    bs = x[np.random.default_rng(seed).integers(0, len(x), (5000, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def module_id(path: str, files: set[str]) -> str:
    """Competition module id of a .py path, given the snapshot's file set."""
    p = PurePosixPath(path)
    in_pkg = p.name == "__init__.py" or str(p.parent / "__init__.py") in files
    if in_pkg:
        parts = [] if p.name == "__init__.py" else [p.stem]
        d = p.parent
        while str(d) not in (".", "") and str(d / "__init__.py") in files:
            parts.insert(0, d.name)
            d = d.parent
        return ".".join(parts or [p.parent.name])
    parts = list(p.with_suffix("").parts)
    if parts and parts[0] in STRIP and len(parts) > 1:
        parts = parts[1:]
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def public_module_gold(rec, repos: Path) -> set[str]:
    out = subprocess.run(["git", "-C", str(repos / rec["repo"]), "ls-tree", "-r", "--name-only", rec["parent"]],
                         capture_output=True, text=True).stdout
    files = set(out.split())
    mods = {module_id(f, files) for f in rec["gold_files"]}
    return {g for g in rec["gold"] if g in mods}


def retrieval(recs, module_gold, suf, off):
    rows = {"bm25": [], "fusion_calls": []}
    pairs = {"bm25": (f"bm25_cgl{suf}", f"bm25_{off}{suf}"),
             "fusion_calls": (f"rrf_cgl-calls_only{suf}", f"rrf_cgl-{off}{suf}")}
    n = 0
    for r in recs:
        keep = [i for i, g in enumerate(r["gold"]) if g not in module_gold[r["task_id"]]]
        if not keep:
            continue
        n += 1
        for name, (a, b) in pairs.items():
            ra = [r["methods"][a]["ranks"][i] for i in keep]
            rb = [r["methods"][b]["ranks"][i] for i in keep]
            rows[name].append(metrics_from_ranks(ra)["recall@10"] - metrics_from_ranks(rb)["recall@10"])
    return {"n_tasks": n, **{f"{k} recall@10 (cgl - official)": ci(v) for k, v in rows.items()}}


def gemma(prompts_file, responses_file, module_gold, splits):
    sys_path = Path(__file__).with_name("12_eval_llm.py")
    import importlib.util
    spec = importlib.util.spec_from_file_location("ev", sys_path)
    ev = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ev)
    P = {p["custom_id"]: p for p in map(json.loads, open(prompts_file)) if p["split"] in splits}
    R = {}
    for r in map(json.loads, open(responses_file)):
        if r.get("ok") and r["custom_id"] in P:
            R[r["custom_id"]] = r
    per = {}
    for cid, r in R.items():
        p = P[cid]
        gold = [g for g in p["gold"] if g not in module_gold.get(p["task_id"], set())]
        if not gold:
            continue
        picks, _ = ev.parse_answer(r["content"], len(p["candidates"]))
        top = [p["candidates"][i - 1] for i in picks]
        per.setdefault(p["task_id"], {})[p["condition"]] = (
            float(any(g in p["candidates"] for g in gold)), float(any(g in top for g in gold)))
    both = [v for v in per.values() if {"official", "cgl"} <= set(v)]
    return {"n_tasks": len(both),
            "gold among candidates (cgl - official)": ci([v["cgl"][0] - v["official"][0] for v in both]),
            "Gemma Hit@5 (cgl - official)": ci([v["cgl"][1] - v["official"][1] for v in both])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos", default="../repos")
    ap.add_argument("--out", default="results/sensitivity_module_gold.json")
    a = ap.parse_args()
    repos = Path(a.repos).expanduser()
    res = {}
    # public split (Experiments 2-3)
    pub = [json.load(open(p)) for p in sorted(Path("results/public_loc/tasks").glob("*.json"))]
    mg_pub = {r["task_id"]: public_module_gold(r, repos) for r in pub}
    res["public"] = {"share_gold_module": round(sum(len(v) for v in mg_pub.values()) / sum(len(r["gold"]) for r in pub), 4),
                     "retrieval": retrieval(pub, mg_pub, "|notest", "official_schema"),
                     "gemma": gemma("results/llm/prompts.jsonl", "results/llm/responses.jsonl", mg_pub, {"public"})}
    # held-out splits (Experiment 4)
    held = [json.load(open(p)) for p in sorted(Path("results/heldout/loc/tasks").glob("*.json"))]
    mg_held = {r["task_id"]: {g for g, k in r["gold_kind"].items() if k == "module"} for r in held}
    for split in ("swebl", "pymatgen", None):
        recs = [r for r in held if split is None or r["split"] == split]
        name = split or "heldout_pooled"
        res[name] = {"share_gold_module": round(sum(len(mg_held[r["task_id"]]) for r in recs) / sum(len(r["gold"]) for r in recs), 4),
                     "retrieval": retrieval(recs, mg_held, "|notest", "official_schema"),
                     "gemma": gemma("results/heldout/llm/prompts.jsonl", "results/heldout/llm/responses.jsonl", mg_held,
                                    {split} if split else {"swebl", "pymatgen"})}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for k, v in res.items():
        print(k, "| module share of gold:", v["share_gold_module"])
        print("   retrieval:", v["retrieval"])
        print("   gemma:    ", v["gemma"])


if __name__ == "__main__":
    main()
