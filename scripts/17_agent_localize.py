"""Experiment 5: Gemma 4 as a graph-navigating localization agent, released graph vs codegraph-loc.

For every task the agent of ``cgl.agent`` runs twice with identical prompt, tools, budget, model and
decoding; only the graph changes:
  * ``official`` — the released graph (competition tasks) or its schema emulated on cgl (public and
    held-out tasks: sync definitions only, no module nodes, `calls` edges only);
  * ``cgl``      — the full codegraph-loc graph with `calls` edges.
This mirrors the two candidate graphs of Experiment 3, with the model now exploring instead of choosing.

Model access: OpenRouter chat completions with native tool calling, temperature 0, seed 0, reasoning off,
one pinned provider (``--provider``; recorded per request). Standard library + numpy/networkx only.
The API key is read from a file (or asked once and stored with mode 600), never printed or saved elsewhere.

Outputs (resumable; finished task x condition pairs are skipped):
  results/agent/<model>/answers.jsonl      one line per episode: answer, tools used, tokens, cost (no texts)
  results/agent/<model>/transcripts.jsonl  full conversations (contain issue texts: local only, git-ignored)

Usage (on the machine with the repositories and the task records):
  python3 scripts/17_agent_localize.py --splits public --sample-tasks 20 --out results/agent/pilot   # pilot
  python3 scripts/17_agent_localize.py --splits swebl,pymatgen,public,comp                          # full run
Options: --model google/gemma-4-31b-it --provider CoreWeave --workers 4 --budget-usd 20 (cumulative per folder)
         --repos ../repos --data ../dati/competition --key-file ~/Desktop/Kaggle/openrouter_key.txt
"""
import argparse
import getpass
import importlib.util
import json
import multiprocessing
import os
import random
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

from cgl.agent import MAX_TURNS, PROTOCOL, TOOL_BUDGET, GraphEnv, run_episode
from cgl.pipeline import build_cgl, cgl_view, extract

HERE = Path(__file__).resolve().parent
REPO_DIR = {"fastapi/fastapi": "fastapi", "Textualize/rich": "rich", "psf/requests": "requests", "encode/httpx": "httpx"}
CALLS_ONLY = {"contains", "imports", "inherits"}


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", "").lstrip("0123456789_"), HERE / name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LLM = _load("11_run_llm.py")  # call(), ssl_context(), URL


def load_tasks(a) -> list[dict]:
    """Task dicts: split, task_id, repo (clone dir), commit, query, gold, official (released graph file or None)."""
    want = set(a.splits.split(","))
    out = []
    if "comp" in want:
        meta = {json.loads(l)["instance_id"]: json.loads(l) for l in open(Path(a.data) / "tasks.jsonl")}
        for p in sorted(Path(a.comp_tasks).glob("*.json")):
            r = json.load(open(p))
            t = meta[r["instance_id"]]
            short, commit = REPO_DIR[t["repo"]], t["base_commit"]
            out.append(dict(split="comp", task_id=r["instance_id"], repo=short, commit=commit,
                            query=t["problem_statement"] or "", gold=r["gold"],
                            official=str(Path(a.data) / "graphs" / f"{short}_{commit}.json")))
    if "public" in want:
        prs = {(r["repo"], r["pr"]): (r["title"] + "\n\n" + (r["body"] or "")).strip()
               for r in map(json.loads, open(a.prs)) if r.get("status") == 200}
        for p in sorted(Path(a.public_tasks).glob("*.json")):
            r = json.load(open(p))
            out.append(dict(split="public", task_id=r["task_id"], repo=r["repo"], commit=r["parent"],
                            query=prs[(r["repo"], r["pr"])], gold=r["gold"], official=None))
    for p in sorted(Path(a.heldout_tasks).glob("*.json")) if want & {"swebl", "pymatgen"} else []:
        r = json.load(open(p))
        if r["split"] in want:
            out.append(dict(split=r["split"], task_id=r["task_id"], repo=r["repo"], commit=r["parent"],
                            query=r["query"], gold=r["gold"], official=None))
    return out


def envs_for(task: dict, repos: Path) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        extract(repos / task["repo"], task["commit"], tmp)
        G = build_cgl(Path(tmp))
    text = {n: G.nodes[n].get("text") or "" for n in G.nodes}
    files = {n: G.nodes[n].get("file") for n in G.nodes}
    envs = {"cgl": GraphEnv(*cgl_view(G, drop_types=CALLS_ONLY), text, files)}
    if task["official"]:
        og = json.load(open(task["official"]))
        o_text = {n["id"]: n.get("text") or "" for n in og["nodes"]}
        o_edges = [(e["source"], e["target"]) for e in og.get("edges", og.get("links", []))]
        envs["official"] = GraphEnv([n["id"] for n in og["nodes"]], o_edges, o_text, files)
    else:
        envs["official"] = GraphEnv(*cgl_view(G, drop_types=CALLS_ONLY, drop_async=True, drop_module=True),
                                    text, files)
    return envs


def work(task: dict, conds: list, cfg: dict) -> list:
    """Worker process: build the graphs of one task and run the requested conditions (in parallel threads)."""
    sys.setrecursionlimit(20000)
    if os.environ.get("CGL_TEST_CRASH_TASK") == task["task_id"]:  # test hook: simulate a worker killed by the OS
        os._exit(1)
    ctx = LLM.ssl_context()
    try:
        envs = envs_for(task, Path(cfg["repos"]))
    except Exception as e:  # noqa: BLE001 — recorded, the task is retried on the next run
        return [dict(task_key(task, c, cfg), ok=False, error=f"graph: {type(e).__name__}: {e}"[:300]) for c in conds]

    def chat(messages, tools):
        body = {"model": cfg["model"], "messages": messages, "tools": tools, "tool_choice": "auto",
                "temperature": 0, "seed": 0, "max_tokens": cfg["max_tokens"], "usage": {"include": True},
                "reasoning": {"enabled": False, "exclude": True}}
        if cfg["provider"]:
            body["provider"] = {"order": [cfg["provider"]], "allow_fallbacks": cfg["allow_fallbacks"]}
        return LLM.call(body, cfg["key"], cfg["url"], ctx, timeout=180)

    def one(cond):
        rec = task_key(task, cond, cfg)
        t0 = time.time()
        try:
            ep = run_episode(envs[cond], task["query"], chat, budget=cfg["budget_calls"])
        except Exception as e:  # noqa: BLE001
            return dict(rec, ok=False, error=f"{type(e).__name__}: {e}"[:300]), None
        gold = set(task["gold"])
        rec.update(ok=True, answer=ep.answer, answer_raw=ep.answer_raw, gold=task["gold"],
                   gold_in_graph=sorted(g for g in gold if g in envs[cond].ids),
                   gold_seen=[g for g in ep.seen if g in gold], n_seen=len(ep.seen),
                   tool_counts=ep.tool_counts, n_tool_calls=ep.n_tool_calls, n_turns=ep.n_turns,
                   n_nudges=ep.n_nudges, n_outside_graph=ep.n_outside_graph, finish=ep.finish,
                   budget_calls=cfg["budget_calls"], max_turns=MAX_TURNS, max_tokens=cfg["max_tokens"],
                   protocol=PROTOCOL, prompt_tokens=ep.prompt_tokens,
                   completion_tokens=ep.completion_tokens, cost_usd=round(ep.cost_usd, 6),
                   providers=sorted(set(ep.providers)), seconds=round(time.time() - t0, 1),
                   time=time.strftime("%Y-%m-%dT%H:%M:%S"))
        return rec, {"key": rec["key"], "messages": ep.messages}

    with ThreadPoolExecutor(len(conds)) as ex:
        return list(ex.map(one, conds))


def task_key(task, cond, cfg):
    return {"key": f"{task['split']}|{task['task_id']}|{cond}", "split": task["split"],
            "task_id": task["task_id"], "repo": task["repo"], "condition": cond, "model": cfg["model"],
            "protocol": PROTOCOL}


def read_key(path: Path) -> str:
    key = path.read_text().strip() if path.exists() else ""
    if not key.startswith("sk-or-"):
        print(f"No valid OpenRouter key in {path} (keys start with 'sk-or-').")
        key = getpass.getpass("Paste your OpenRouter API key and press Enter (input is hidden): ").strip()
        if not key.startswith("sk-or-"):
            sys.exit("That does not look like an OpenRouter key. Nothing saved.")
        path.write_text(key + "\n")
        path.chmod(0o600)
        print(f"Key saved to {path} (readable only by you).")
    return key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="swebl,pymatgen,public,comp")
    ap.add_argument("--limit-tasks", type=int, default=0, help="only the first N tasks (debugging)")
    ap.add_argument("--sample-tasks", type=int, default=0, help="a random sample of N tasks (seed 0; pilot)")
    ap.add_argument("--exclude-tasks-from", default=None, help="answers file whose tasks are skipped (pilot tasks)")
    ap.add_argument("--repos", default="../repos")
    ap.add_argument("--data", default="../dati/competition")
    ap.add_argument("--comp-tasks", default="results/loc_notest/tasks")
    ap.add_argument("--public-tasks", default="results/public_loc/tasks")
    ap.add_argument("--prs", default="results/public/pr_texts.jsonl")
    ap.add_argument("--heldout-tasks", default="results/heldout/loc/tasks")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--provider", default="CoreWeave", help="pinned OpenRouter provider ('' = let OpenRouter route)")
    ap.add_argument("--allow-fallbacks", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=800)
    ap.add_argument("--budget-calls", type=int, default=TOOL_BUDGET)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--budget-usd", type=float, default=20.0)
    ap.add_argument("--key-file", default="~/Desktop/Kaggle/openrouter_key.txt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--url", default=LLM.URL, help="chat-completions endpoint (tests use a local mock)")
    a = ap.parse_args()

    out = Path(a.out or f"results/agent/{a.model.split('/')[-1]}")
    out.mkdir(parents=True, exist_ok=True)
    answers, transcripts = out / "answers.jsonl", out / "transcripts.jsonl"
    done, spent_before, protocols = set(), 0.0, set()
    if answers.exists():
        for l in open(answers):
            try:
                r = json.loads(l)
            except json.JSONDecodeError:
                continue
            spent_before += r.get("cost_usd") or 0.0
            if r.get("ok"):  # failure records carry no episode, so they say nothing about the protocol
                protocols.add(r.get("protocol", "v1"))
                done.add(r["key"])
    if protocols - {PROTOCOL}:
        sys.exit(f"{answers} holds episodes of protocol {sorted(protocols)}, this code is {PROTOCOL}: "
                 f"use a new --out folder so that protocols are never mixed.")
    tasks = load_tasks(a)
    if a.exclude_tasks_from:
        skip = {(r["split"], r["task_id"]) for r in map(json.loads, open(a.exclude_tasks_from))}
        tasks = [t for t in tasks if (t["split"], t["task_id"]) not in skip]
    if a.sample_tasks:
        tasks = sorted(random.Random(0).sample(tasks, min(a.sample_tasks, len(tasks))),
                       key=lambda t: (t["split"], t["task_id"]))
    if a.limit_tasks:
        tasks = tasks[:a.limit_tasks]
    todo = []
    for t in tasks:
        conds = [c for c in ("official", "cgl") if f"{t['split']}|{t['task_id']}|{c}" not in done]
        if conds:
            todo.append((t, conds))
    print(f"{len(tasks)} tasks, {len(todo)} with episodes to run | model {a.model} via "
          f"{a.provider or 'any provider'} | {a.workers} workers | budget ${a.budget_usd} for this folder "
          f"(already spent: ${spent_before:.2f})", flush=True)
    if spent_before >= a.budget_usd:
        sys.exit("Budget already used up for this folder; raise --budget-usd to continue.")
    if not todo:
        print("Nothing to do.")
        return
    cfg = dict(model=a.model, provider=a.provider, allow_fallbacks=a.allow_fallbacks, max_tokens=a.max_tokens,
               budget_calls=a.budget_calls, repos=str(Path(a.repos).expanduser()), url=a.url,
               key=read_key(Path(a.key_file).expanduser()))
    spent, n_ok, n_fail, stop = 0.0, 0, 0, None
    queue, crashes, workers = list(todo), Counter(), a.workers
    spawn = multiprocessing.get_context("spawn")  # the macOS default, used everywhere for identical behaviour
    ex = ProcessPoolExecutor(workers, mp_context=spawn)

    def failed(t, conds, why):
        return [dict(task_key(t, c, cfg), ok=False, error=why[:300]) for c in conds]

    with open(answers, "a") as fa, open(transcripts, "a") as ft:
        running = {}
        while queue or running:
            while queue and len(running) < workers and stop is None:
                t, conds = queue.pop(0)
                running[ex.submit(work, t, conds, cfg)] = (t, conds)
            if not running:
                break
            fin, _ = wait(running, return_when=FIRST_COMPLETED)
            broken = any(isinstance(f.exception(), BrokenProcessPool) for f in fin)
            if broken:  # a worker died (usually memory): every task of this pool is lost; collect them all
                wait(running, timeout=60)
                fin = set(running)
            results = []
            for f in fin:
                t, conds = running.pop(f)
                if f.done() and f.exception() is None:
                    results += f.result()
                    continue
                e = f.exception() if f.done() else None
                if e is None or isinstance(e, BrokenProcessPool):
                    crashes[t["task_id"]] += 1
                    if crashes[t["task_id"]] <= 2:
                        queue.insert(0, (t, conds))  # retried in this run, with fewer workers
                        continue
                results += failed(t, conds, f"worker: {type(e).__name__}: {e}")
            if broken:
                ex.shutdown(wait=False, cancel_futures=True)
                workers = max(1, workers - 1)
                ex = ProcessPoolExecutor(workers, mp_context=spawn)
                print(f"  ! a worker process died (often: not enough memory); continuing with {workers} workers",
                      flush=True)
            for item in results:
                rec, tr = item if isinstance(item, tuple) else (item, None)
                fa.write(json.dumps(rec) + "\n")
                if tr:
                    ft.write(json.dumps(tr) + "\n")
                spent += rec.get("cost_usd") or 0.0
                if rec.get("ok"):
                    n_ok += 1
                else:
                    n_fail += 1
                    print(f"  ! {rec['key']}: {rec.get('error', '')[:160]}", flush=True)
                    if any(s in rec.get("error", "") for s in ("HTTP 401", "HTTP 402", "HTTP 403", "SSL error")):
                        stop = "key, credit or SSL problem"
            fa.flush()
            ft.flush()
            if spent_before + spent > a.budget_usd and stop is None:
                stop = f"budget ${a.budget_usd} reached"
            n = n_ok + n_fail
            if results and (n % 20 < 2 or not queue):
                print(f"  {n} episodes done ({n_fail} failed), cost ${spent:.3f}, "
                      f"{len(queue)} tasks queued", flush=True)
    ex.shutdown()
    msg = f"STOPPED EARLY ({stop})" if stop else "ALL DONE"
    print(f"{msg}: {n_ok} episodes ok, {n_fail} failed, run cost ${spent:.3f}")
    if n_fail:
        print("Run the same command again to retry the failed episodes.")


if __name__ == "__main__":
    main()
