"""Issue -> function-level localization on the 129 public competition tasks.

For every task, each method ranks the candidate nodes of a graph given the issue text
(`problem_statement` only; hints are not used). Gold = functions/classes/modules edited by the
reference patch, in the competition id convention (see cgl.graph.flat_id). Gold nodes absent
from a graph can never be retrieved from it, so coverage gaps count against that graph.

Methods (pre-registered; RRF with k=60 and PPR alpha=0.85 fixed before looking at results):
  ident                identifiers mentioned in the issue, resolved to nodes with the harness's resolver
  ident+emb_raw        ... then nodes ranked by max cosine to those seeds (released embeddings,
                       i.e. what search_similar_code returns)
  ident+emb_centered   same with mean-centered embeddings
  ident+ppr_official   ... then Personalized PageRank from the seeds on the released `calls` graph
  ident+ppr_cgl        same on the cgl graph (calls + contains + imports + inherits)
  bm25_official        BM25 over node id + source, candidates = released graph nodes
  bm25_cgl             same, candidates = cgl nodes (incl. async defs and module nodes)
  rrf_official         RRF(bm25_official, PPR on released graph from identifiers + BM25 top-10)
  rrf_cgl              same on the cgl graph
  rrf_cgl-<ablation>   rrf_cgl with parts of the cgl graph removed:
                       no_async, no_module, no_contains, no_imports, no_inherits,
                       calls_only, official_schema (calls only, no async, no module nodes)

Reads the competition data in place; resumable (one JSON per task in <out>/tasks/).
Usage:
  python scripts/07_localization_benchmark.py --data ../dati/competition --repos ~/repos --out results/loc
  python scripts/07_localization_benchmark.py --data x --repos x --out results/loc --summarize
"""
import argparse
import io
import json
import subprocess
import tarfile
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from cgl.bench import (BM25, Resolver, gold_ranks, issue_identifiers, metrics_from_ranks,
                       personalized_pagerank, rank_from_scores, rrf, tokenize)
from cgl.graph import CodeGraphBuilder, flat_id, module_name_for, official_view
from cgl.labels import def_table, is_test_path, locate_changes, parse_unified_diff

REPO_DIR = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich",
            "psf/requests": "requests", "encode/httpx": "httpx"}
MODULE_KW = dict(scheme="official", strip_prefixes=("src", "docs_src"))
TOP = 1000
ABLATIONS = {
    "no_async": dict(drop_async=True),
    "no_module": dict(drop_module=True),
    "no_contains": dict(drop_types={"contains"}),
    "no_imports": dict(drop_types={"imports"}),
    "no_inherits": dict(drop_types={"inherits"}),
    "calls_only": dict(drop_types={"contains", "imports", "inherits"}),
    "official_schema": dict(drop_types={"contains", "imports", "inherits"}, drop_async=True, drop_module=True),
}


def extract(repo_path: Path, commit: str, dest: str):
    data = subprocess.run(["git", "-C", str(repo_path), "archive", commit], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        members = [m for m in tf.getmembers() if m.isfile() or m.isdir()]
        try:
            tf.extractall(dest, members=members, filter="data")
        except TypeError:
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
            out[flat_id(g, kind_of, mid)] = fc.path
    return out


def cgl_view(G, drop_types=(), drop_async=False, drop_module=False):
    nodes = [n for n, a in G.nodes(data=True)
             if not (drop_async and a.get("is_async")) and not (drop_module and a.get("kind") == "module")]
    keep = set(nodes)
    edges = [(u, v) for u, v, a in G.edges(data=True)
             if a.get("type") not in drop_types and u in keep and v in keep]
    return nodes, edges


def ppr_rank(nodes, edges, seeds):
    if not seeds:
        return []
    return rank_from_scores(nodes, personalized_pagerank(nodes, edges, seeds), TOP)


def seeds_then(seeds: list[str], ranking: list[str]) -> list[str]:
    seen, out = set(), []
    for n in list(seeds) + list(ranking):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:TOP]


def is_test_node(node_id: str, file: str | None) -> bool:
    """Test code? Uses the source file when known, else the id segments."""
    if file:
        return is_test_path(file)
    parts = node_id.split(".")
    return any(p in ("tests", "test", "testing", "conftest") or p.startswith("test_") for p in parts[:-1]) \
        or parts[0] in ("tests", "test")


def run_commit(short, commit, tasks, data: Path, repos: Path, exclude_tests: bool = False):
    og = json.load(open(data / "graphs" / f"{short}_{commit}.json"))
    o_nodes = [n["id"] for n in og["nodes"]]
    o_text = {n["id"]: n.get("text") or "" for n in og["nodes"]}
    o_edges = [(e["source"], e["target"]) for e in og.get("edges", og.get("links", []))]
    z = np.load(data / "embeddings" / f"{short}_{commit}.npz")
    emb_ids = [k for k in o_nodes if k in z.files]
    E = np.stack([z[k] for k in emb_ids]).astype(np.float64)
    En = E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)
    C = E - E.mean(0, keepdims=True)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
    emb_idx = {k: i for i, k in enumerate(emb_ids)}

    with tempfile.TemporaryDirectory() as tmp:
        extract(repos / short, commit, tmp)
        root = Path(tmp)
        G = official_view(CodeGraphBuilder(root, module_scheme="official",
                                           strip_prefixes=MODULE_KW["strip_prefixes"]).build())
        golds = {t["instance_id"]: gold_for(t["patch"], root) for t in tasks}

    c_nodes = list(G.nodes)
    c_text = {n: G.nodes[n].get("text") or "" for n in c_nodes}
    c_file = {n: G.nodes[n].get("file") for n in c_nodes}
    bm_o = BM25([tokenize(n + " " + o_text[n]) for n in o_nodes])
    bm_c = BM25([tokenize(n + " " + c_text[n]) for n in c_nodes])
    res_o, res_c = Resolver(o_nodes), Resolver(c_nodes)
    views = {"full": cgl_view(G), **{k: cgl_view(G, **v) for k, v in ABLATIONS.items()}}

    out = []
    for t in tasks:
        gold = golds[t["instance_id"]]
        if not gold:
            continue
        q = t["problem_statement"] or ""
        qtok = tokenize(q)
        idents = issue_identifiers(q)
        seeds_o = list(dict.fromkeys(r for r in (res_o.resolve(i) for i in idents) if r))
        seeds_c = list(dict.fromkeys(r for r in (res_c.resolve(i) for i in idents) if r))

        s_o, s_c = bm_o.scores(qtok), bm_c.scores(qtok)
        r_bm_o, r_bm_c = rank_from_scores(o_nodes, s_o, TOP), rank_from_scores(c_nodes, s_c, TOP)

        def emb_rank(Mn):
            idx = [emb_idx[s] for s in seeds_o if s in emb_idx]
            if not idx:
                return list(seeds_o)
            sim = (Mn @ Mn[idx].T).max(1)
            return seeds_then(seeds_o, rank_from_scores(emb_ids, sim + 2.0, TOP))  # +2: keep negatives

        def hybrid_seeds(seeds, ranking, scores_by_id):
            w = {s: 1.0 for s in seeds}
            top = ranking[:10]
            tot = sum(scores_by_id[n] for n in top) or 1.0
            for n in top:
                w[n] = w.get(n, 0.0) + scores_by_id[n] / tot * max(1, len(seeds))
            return w

        sc_o = dict(zip(o_nodes, s_o))
        sc_c = dict(zip(c_nodes, s_c))
        rankings = {
            "ident": list(seeds_o),
            "ident+emb_raw": emb_rank(En),
            "ident+emb_centered": emb_rank(Cn),
            "ident+ppr_official": seeds_then(seeds_o, ppr_rank(o_nodes, o_edges, {s: 1.0 for s in seeds_o})),
            "ident+ppr_cgl": seeds_then(seeds_c, ppr_rank(*views["full"], {s: 1.0 for s in seeds_c})),
            "bm25_official": r_bm_o,
            "bm25_cgl": r_bm_c,
            "rrf_official": rrf([r_bm_o, ppr_rank(o_nodes, o_edges, hybrid_seeds(seeds_o, r_bm_o, sc_o))])[:TOP],
        }
        for name, (vn, ve) in views.items():
            keep = set(vn)
            bm_rank = [n for n in r_bm_c if n in keep]
            seeds = [s for s in seeds_c if s in keep]
            key = "rrf_cgl" if name == "full" else f"rrf_cgl-{name}"
            rankings[key] = rrf([bm_rank, ppr_rank(vn, ve, hybrid_seeds(seeds, bm_rank, sc_c))])[:TOP]

        if exclude_tests:  # secondary analysis: test code is never a gold location
            rankings = {m: [n for n in rk if not is_test_node(n, c_file.get(n))] for m, rk in rankings.items()}

        gold_ids = sorted(gold)
        gold_files = sorted(set(gold.values()))
        rec = {"instance_id": t["instance_id"], "repo": short, "commit": commit,
               "n_gold": len(gold_ids), "gold": gold_ids, "gold_files": gold_files,
               "n_identifiers": len(idents), "n_seeds_official": len(seeds_o), "n_seeds_cgl": len(seeds_c),
               "gold_in_official": [g in o_text for g in gold_ids], "methods": {}}
        for m, rk in rankings.items():
            files = list(dict.fromkeys(c_file.get(n) for n in rk if c_file.get(n)))
            rec["methods"][m] = {"ranks": gold_ranks(rk, gold_ids), "file_ranks": gold_ranks(files, gold_files),
                                 "n_ranked": len(rk), "top5": rk[:5]}
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
def summarize(out: Path):
    rng = np.random.default_rng(0)
    recs = [json.load(open(p)) for p in sorted((out / "tasks").glob("*.json"))]
    methods = list(recs[0]["methods"])
    B = 5000
    boot = rng.integers(0, len(recs), (B, len(recs)))

    def table(metric_fn):
        vals = {m: np.array([metric_fn(r["methods"][m]) for r in recs]) for m in methods}
        return vals

    def ci(x):
        bs = x[boot].mean(1)
        return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    fn_metrics = ["hit@1", "hit@5", "hit@10", "recall@5", "recall@10", "mrr"]
    summ = {"n_tasks": len(recs), "by_method": {}, "file_level": {}, "paired_differences": {}, "by_repo": {}}
    per = {k: table(lambda mm, k=k: metrics_from_ranks(mm["ranks"])[k]) for k in fn_metrics}
    per_file = {k: table(lambda mm, k=k: metrics_from_ranks(mm["file_ranks"])[k]) for k in ("hit@1", "hit@5")}
    for m in methods:
        summ["by_method"][m] = {k: ci(per[k][m]) for k in fn_metrics}
        summ["by_method"][m]["returns_anything"] = round(float(np.mean([r["methods"][m]["n_ranked"] > 0 for r in recs])), 4)
        summ["file_level"][m] = {k: ci(per_file[k][m]) for k in per_file}
    contrasts = [("bm25_cgl", "bm25_official"), ("rrf_cgl", "rrf_official"), ("ident+ppr_cgl", "ident+ppr_official"),
                 ("ident+emb_raw", "ident"), ("ident+emb_centered", "ident+emb_raw"), ("ident+ppr_official", "ident"),
                 ("rrf_official", "bm25_official"), ("rrf_cgl", "bm25_cgl")] + \
                [("rrf_cgl", f"rrf_cgl-{a}") for a in ABLATIONS]
    for a, b in contrasts:
        for k in ("hit@5", "recall@10", "mrr"):
            d = per[k][a] - per[k][b]
            summ["paired_differences"][f"{a} - {b} [{k}]"] = ci(d)
    by = defaultdict(list)
    for i, r in enumerate(recs):
        by[r["repo"]].append(i)
    for repo, ix in by.items():
        summ["by_repo"][repo] = {"n": len(ix), **{m: round(float(per["hit@5"][m][ix].mean()), 4) for m in methods}}
    summ["share_tasks_with_identifier_seed"] = round(float(np.mean([r["n_seeds_official"] > 0 for r in recs])), 4)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    return summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--repos", required=True)
    ap.add_argument("--out", default="results/loc")
    ap.add_argument("--max-seconds", type=float, default=150)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--exclude-tests", action="store_true",
                    help="drop test-code nodes from every ranking (secondary analysis)")
    a = ap.parse_args()
    data, repos, out = Path(a.data).expanduser(), Path(a.repos).expanduser(), Path(a.out)
    (out / "tasks").mkdir(parents=True, exist_ok=True)
    if a.summarize:
        s = summarize(out)
        for m, v in s["by_method"].items():
            print(f"{m:28s} hit@1={v['hit@1'][0]:.3f} hit@5={v['hit@5'][0]:.3f} [{v['hit@5'][1]:.3f},{v['hit@5'][2]:.3f}] "
                  f"hit@10={v['hit@10'][0]:.3f} rec@10={v['recall@10'][0]:.3f} mrr={v['mrr'][0]:.3f} "
                  f"any={v['returns_anything']:.2f} | file hit@1={s['file_level'][m]['hit@1'][0]:.3f} "
                  f"hit@5={s['file_level'][m]['hit@5'][0]:.3f}")
        return
    tasks = [json.loads(l) for l in open(data / "tasks.jsonl")]
    by_commit = defaultdict(list)
    for t in tasks:
        by_commit[(REPO_DIR[t["repo"]], t["base_commit"])].append(t)
    marker = out / "commits_done.txt"
    done_commits = set(marker.read_text().split()) if marker.exists() else set()
    todo = [(s, c) for (s, c) in sorted(by_commit) if f"{s}_{c}" not in done_commits]
    print(f"{len(by_commit)} commits, {len(todo)} to do", flush=True)
    start = time.time()
    for s, c in todo:
        if time.time() - start > a.max_seconds:
            print("time budget reached; run again to continue", flush=True)
            return
        t0 = time.time()
        for rec in run_commit(s, c, by_commit[(s, c)], data, repos, a.exclude_tests):
            (out / "tasks" / f"{rec['instance_id']}.json").write_text(json.dumps(rec))
        with open(marker, "a") as f:
            f.write(f"{s}_{c}\n")
        print(f"{s} {c[:10]} {len(by_commit[(s, c)])} task(s) {time.time() - t0:.1f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
