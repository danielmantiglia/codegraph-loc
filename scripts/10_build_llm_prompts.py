"""Build the prompts for the Gemma 4 re-ranking baseline (two graph conditions per task).

For every task, the same candidate-generation procedure is applied to two graphs:
  * ``official`` — the released graph (competition tasks) or its schema emulated on cgl
    (public-history tasks: sync definitions only, no module nodes, `calls` edges only);
  * ``cgl``      — the full codegraph-loc graph (async definitions and module nodes included).
Candidates = top-25 BM25 nodes for the issue (test code excluded) + up to 15 `calls`
neighbours of the BM25 top-5, shuffled with a fixed seed. Gemma sees the issue and, for each
candidate, its id, signature and first docstring line, and must name the 5 to edit.

Output: <out>/prompts.jsonl (one chat request per task x condition). The file contains
competition text and stays on the machine that holds the competition data.

Usage (on the machine with the data):
  python scripts/10_build_llm_prompts.py --data ../dati/competition --repos ~/repos \
      --comp-tasks results/loc_notest/tasks --public-tasks results/public_loc/tasks \
      --prs results/public/pr_texts.jsonl --out results/llm [--max-seconds 150]
Held-out splits (Experiment 4; task records of 13_heldout_benchmark.py carry their own query and split name;
the released schema is emulated exactly as for the public split):
  python scripts/10_build_llm_prompts.py --comp-tasks "" --public-tasks "" \
      --heldout-tasks results/heldout/loc/tasks --repos ../repos --out results/heldout/llm
"""
import argparse
import json
import random
import re
import tempfile
import time
from collections import defaultdict
from pathlib import Path

from cgl.bench import BM25, rank_from_scores, tokenize
from cgl.pipeline import build_cgl, cgl_view, extract, is_test_node

REPO_DIR = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
N_BM25, N_NEIGH, ISSUE_CHARS = 25, 15, 3000
SYSTEM = ("You are an expert Python developer. Given a bug report or change request and a list of "
          "candidate code locations from the repository, identify where the code must be edited.")
INSTR = ("Choose the 5 candidates most likely to need editing to resolve the issue, most likely first. "
         'Answer with JSON only, e.g. {"answer": [12, 3, 27, 8, 19]}.')


def signature(node_id: str, text: str, kind: str | None) -> str:
    lines = (text or "").splitlines()
    sig = ""
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(("def ", "async def ", "class ")):
            parts = [s]
            for nxt in lines[i + 1:i + 8]:  # multi-line signatures: read up to the closing ':'
                if parts[-1].endswith(":"):
                    break
                parts.append(nxt.strip())
            sig = re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", " ".join(parts)))[:200]
            break
    doc = ""
    m = re.search(r'("""|\'\'\')\s*(.+?)\s*(\n|\1)', "\n".join(lines[:12]))
    if m:
        doc = m.group(2).strip()[:150]
    if not sig:  # module node: its docstring, else its first meaningful non-import line
        if doc:
            return f"(module)  # {doc}"
        for l in lines:
            s = l.strip()
            if s and not s.startswith(("import ", "from ", "#", '"""', "'''", "@")):
                return "(module) " + s[:160]
        return "(module)"
    return sig + (f"  # {doc}" if doc else "")


def candidates(nodes, text, files, edges, query):
    keep = [n for n in nodes if not is_test_node(n, files.get(n))]
    bm = BM25([tokenize(n + " " + text.get(n, "")) for n in keep])
    top = rank_from_scores(keep, bm.scores(tokenize(query)), N_BM25)
    adj = defaultdict(list)
    keepset = set(keep)
    for u, v in edges:
        if u in keepset and v in keepset:
            adj[u].append(v)
            adj[v].append(u)
    neigh = []
    for n in top[:5]:
        for m in adj.get(n, []):
            if m not in top and m not in neigh:
                neigh.append(m)
    return top + neigh[:N_NEIGH]


def make_prompt(query, cands, text, kind, seed_key):
    order = list(cands)
    random.Random(seed_key).shuffle(order)
    lines = [f"[{i + 1}] {n}\n    {signature(n, text.get(n, ''), kind.get(n))}" for i, n in enumerate(order)]
    user = (f"Issue:\n{query[:ISSUE_CHARS]}\n\nCandidates:\n" + "\n".join(lines) + "\n\n" + INSTR)
    return order, [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def record(task_id, split, cond, order, ranked, messages, gold):
    """``candidates`` = order shown to the model; ``ranked`` = BM25 order (then neighbours), for baselines."""
    return {"custom_id": f"{split}|{task_id}|{cond}", "split": split, "task_id": task_id,
            "condition": cond, "candidates": order, "ranked": ranked, "gold": gold, "messages": messages}


def prompt_records(G, split, tid, query, gold, official=None):
    """The two prompt records of one task. ``official`` = (nodes, edges, text) of the released graph;
    None emulates the released schema on cgl (public and held-out splits)."""
    text = {n: G.nodes[n].get("text") or "" for n in G.nodes}
    kind = {n: G.nodes[n].get("kind") for n in G.nodes}
    files = {n: G.nodes[n].get("file") for n in G.nodes}
    full_nodes, full_edges = cgl_view(G, drop_types={"contains", "imports", "inherits"})
    if official is not None:
        o_nodes, o_edges, o_text = official
    else:
        o_nodes, o_edges = cgl_view(G, drop_types={"contains", "imports", "inherits"},
                                    drop_async=True, drop_module=True)
        o_text = text
    out = []
    for cond, (nodes, edges, tx) in {"official": (o_nodes, o_edges, o_text),
                                     "cgl": (full_nodes, full_edges, text)}.items():
        cands = candidates(nodes, tx, files, edges, query)
        order, msgs = make_prompt(query, cands, tx, kind, f"{split}|{tid}")
        out.append(record(tid, split, cond, order, cands, msgs, gold))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help="competition data (only needed for --comp-tasks)")
    ap.add_argument("--repos", required=True)
    ap.add_argument("--comp-tasks", default="results/loc_notest/tasks")
    ap.add_argument("--public-tasks", default="results/public_loc/tasks")
    ap.add_argument("--prs", default="results/public/pr_texts.jsonl")
    ap.add_argument("--out", default="results/llm")
    ap.add_argument("--heldout-tasks", default="")
    ap.add_argument("--max-seconds", type=float, default=150)
    a = ap.parse_args()
    data, repos, out = Path(a.data or ".").expanduser(), Path(a.repos).expanduser(), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pfile = out / "prompts.jsonl"
    done = set()
    if pfile.exists():  # resume; drop a trailing line cut off by an interrupted run
        raw = pfile.read_text().split("\n")
        good = []
        for l in raw:
            try:
                good.append(json.loads(l)) if l else None
            except json.JSONDecodeError:
                pass
        per_task = defaultdict(set)
        for g in good:
            per_task[g["custom_id"].rsplit("|", 1)[0]].add(g["condition"])
        done = {t for t, c in per_task.items() if c == {"official", "cgl"}}
        good = [g for g in good if g["custom_id"].rsplit("|", 1)[0] in done]  # keep complete tasks only
        if len(good) != sum(1 for l in raw if l):
            pfile.write_text("".join(json.dumps(g) + "\n" for g in good))
    tasks = {json.loads(l)["instance_id"]: json.loads(l) for l in open(data / "tasks.jsonl")} if a.comp_tasks else {}
    prs = {(r["repo"], r["pr"]): (r["title"] + "\n\n" + (r["body"] or "")).strip()
           for r in map(json.loads, open(a.prs)) if r.get("status") == 200} if a.public_tasks else {}
    jobs = []
    for p in sorted(Path(a.comp_tasks).glob("*.json")) if a.comp_tasks else []:
        r = json.load(open(p))
        jobs.append(("comp", r["instance_id"], r))
    for p in sorted(Path(a.public_tasks).glob("*.json")) if a.public_tasks else []:
        r = json.load(open(p))
        jobs.append(("public", r["task_id"], r))
    for p in sorted(Path(a.heldout_tasks).glob("*.json")) if a.heldout_tasks else []:
        r = json.load(open(p))
        jobs.append((r["split"], r["task_id"], r))
    todo = [j for j in jobs if f"{j[0]}|{j[1]}" not in done]
    print(f"{len(jobs)} tasks, {len(todo)} to do", flush=True)
    start = time.time()
    with open(pfile, "a") as fo:
        for split, tid, r in todo:
            if time.time() - start > a.max_seconds:
                print("time budget reached; run again to continue", flush=True)
                return
            if split == "comp":
                t = tasks[tid]
                short, commit, query = REPO_DIR[t["repo"]], t["base_commit"], t["problem_statement"] or ""
            elif split == "public":
                short, commit, query = r["repo"], r["parent"], prs[(r["repo"], r["pr"])]
            else:  # held-out split: the record carries its query
                short, commit, query = r["repo"], r["parent"], r["query"]
            with tempfile.TemporaryDirectory() as tmp:
                extract(repos / short, commit, tmp)
                G = build_cgl(Path(tmp))
            official = None
            if split == "comp":
                og = json.load(open(data / "graphs" / f"{short}_{commit}.json"))
                official = ([n["id"] for n in og["nodes"]], [(e["source"], e["target"]) for e in og.get("edges", og.get("links", []))],
                            {n["id"]: n.get("text") or "" for n in og["nodes"]})
            for rec in prompt_records(G, split, tid, query, r["gold"], official):
                fo.write(json.dumps(rec) + "\n")
            fo.flush()
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
