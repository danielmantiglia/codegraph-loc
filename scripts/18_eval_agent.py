"""Score Experiment 5 (graph-navigating agent, released graph vs codegraph-loc).

Per split and condition: Hit@1, Hit@5, Recall@5 and MRR@5 of the submitted answer (at most 5 ids), the share
of episodes in which the agent *saw* a gold location in any tool output ("reached"), Hit@5 given reached
(selection), tool use, turns, finish reasons and cost. Paired differences cgl − official with bootstrap
95% CIs (5,000 resamples, seed 0) over tasks answered in both conditions. "pooled_heldout" pools swebl and
pymatgen.

Usage: python scripts/18_eval_agent.py [--answers results/agent/gemma-4-31b-it/answers.jsonl]
           [--exclude-tasks-from results/agent/pilot/gemma-4-31b-it/answers.jsonl]   # drop the pilot tasks
       python scripts/18_eval_agent.py --answers <pilot answers.jsonl> --mechanics-only
           # pilot check: how the loop behaves (finish reasons, tool errors, tokens, cost), pooled over both
           # conditions and with no accuracy, so protocol fixes cannot be steered by the graph comparison
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

K = 5


def ci(x, seed=0, b=5000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return None
    bs = x[np.random.default_rng(seed).integers(0, len(x), (b, len(x)))].mean(1)
    return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def ci_strata_gap(a, b, seed=0, B=5000):
    """mean(a) - mean(b) with a stratified paired bootstrap (each stratum resampled separately)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not len(a) or not len(b):
        return None
    rng = np.random.default_rng(seed)
    bs = a[rng.integers(0, len(a), (B, len(a)))].mean(1) - b[rng.integers(0, len(b), (B, len(b)))].mean(1)
    return [round(float(a.mean() - b.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
            round(float(np.percentile(bs, 97.5)), 4)]


def complete(conds) -> bool:
    """Every edited location is a node of the `official` graph (depends only on the task and the graphs)."""
    r = conds["official"]
    return set(r["gold"]) <= set(r["gold_in_graph"])


def hit5(r):
    return float(any(a in set(r["gold"]) for a in r["answer"][:K]))


def hypotheses(groups) -> dict:
    """The four pre-registered contrasts of docs/preregistration_agent.md (Hit@5, cgl - official)."""
    d = {split: [hit5(c["cgl"]) - hit5(c["official"]) for c in rows] for split, rows in groups.items()}
    allrows = [c for split, rows in groups.items() if split not in ("pooled_heldout",) for c in rows]
    miss = [hit5(c["cgl"]) - hit5(c["official"]) for c in allrows if not complete(c)]
    comp = [hit5(c["cgl"]) - hit5(c["official"]) for c in allrows if complete(c)]
    held_complete = [hit5(c["cgl"]) - hit5(c["official"]) for c in groups.get("pooled_heldout", []) if complete(c)]
    h8 = ci(held_complete)
    out = {
        "H5 public history: hit@5 cgl - official > 0": dict(n=len(d.get("public", [])), est=ci(d.get("public", []))),
        "H6 held-out pooled: hit@5 cgl - official > 0": dict(n=len(d.get("pooled_heldout", [])),
                                                             est=ci(d.get("pooled_heldout", []))),
        "H7 all splits: gain(edited location missing from official) - gain(all present) > 0":
            dict(n_missing=len(miss), n_present=len(comp), gain_missing=ci(miss), gain_present=ci(comp),
                 est=ci_strata_gap(miss, comp)),
        "H8 held-out pooled, all edited locations in official: hit@5 cgl - official within +-0.05":
            dict(n=len(held_complete), est=h8),
    }
    for k, v in out.items():
        e = v["est"]
        if e is None:
            v["outcome"] = "no data"
        elif k.startswith("H8"):
            v["outcome"] = "confirmed" if -0.05 < e[1] and e[2] < 0.05 else "not confirmed"
        else:
            v["outcome"] = "confirmed" if e[1] > 0 else "not confirmed"
    return out


def scores(answer, gold):
    gold = set(gold)
    ranks = [i + 1 for i, a in enumerate(answer[:K]) if a in gold]
    return {"hit@1": float(bool(ranks) and ranks[0] == 1), "hit@5": float(bool(ranks)),
            "recall@5": len({a for a in answer[:K] if a in gold}) / len(gold) if gold else 0.0,
            "mrr@5": 1.0 / ranks[0] if ranks else 0.0}


def mechanics(answers: Path):
    recs = [json.loads(l) for l in open(answers)]
    ok = [r for r in recs if r.get("ok")]
    fails = Counter(r.get("error", "")[:80] for r in recs if not r.get("ok"))
    n = max(len(ok), 1)
    out = {"episodes_ok": len(ok), "episodes_failed": len(recs) - len(ok), "failure_messages": dict(fails),
           "finish": dict(Counter(r["finish"] for r in ok)),
           "mean_tool_calls": round(sum(r["n_tool_calls"] for r in ok) / n, 2),
           "share_using_whole_budget": round(sum(r["n_tool_calls"] >= r.get("budget_calls", 12) for r in ok) / n, 3),
           "mean_turns": round(sum(r["n_turns"] for r in ok) / n, 2),
           "mean_nudges": round(sum(r["n_nudges"] for r in ok) / n, 2),
           "share_empty_answer": round(sum(not r["answer"] for r in ok) / n, 3),
           "mean_answer_len": round(sum(len(r["answer"]) for r in ok) / n, 2),
           "answer_len_counts": dict(sorted(Counter(len(r["answer"]) for r in ok).items())),
           "protocols": dict(Counter(r.get("protocol", "v1") for r in ok)),
           "tool_calls_total": dict(sum((Counter(r["tool_counts"]) for r in ok), Counter())),
           "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in ok) / n),
           "mean_completion_tokens": round(sum(r["completion_tokens"] for r in ok) / n),
           "max_prompt_tokens": max((r["prompt_tokens"] for r in ok), default=0),
           "cost_usd_total": round(sum(r.get("cost_usd") or 0 for r in recs), 4),
           "cost_usd_per_episode": round(sum(r.get("cost_usd") or 0 for r in ok) / n, 5),
           "providers": dict(Counter(p for r in ok for p in r["providers"])),
           "mean_seconds": round(sum(r["seconds"] for r in ok) / n, 1)}
    tr = answers.with_name("transcripts.jsonl")
    if tr.exists():  # tool-result errors, from the local transcripts
        kinds = Counter()
        for l in open(tr):
            for m in json.loads(l)["messages"]:
                if m["role"] == "tool":
                    c = m["content"]
                    kinds["tool results"] += 1
                    for k, pre in (("no such node", "No node named"), ("ambiguous name", "'"),
                                   ("unknown tool", "Unknown tool"), ("budget exhausted", "Tool budget exhausted"),
                                   ("search: no match", "No matching")):
                        if c.startswith(pre) and (k != "ambiguous name" or "is ambiguous" in c[:200]):
                            kinds[k] += 1
        out["tool_results"] = dict(kinds)
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", default="results/agent/gemma-4-31b-it/answers.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--exclude-tasks-from", default=None, help="answers file whose tasks are excluded (pilot)")
    ap.add_argument("--mechanics-only", action="store_true", help="loop behaviour only, no accuracy (pilot)")
    a = ap.parse_args()
    if a.mechanics_only:
        return mechanics(Path(a.answers))
    recs = {}
    for l in open(a.answers):
        r = json.loads(l)
        if r.get("ok"):
            recs[r["key"]] = r  # last successful episode wins
    pilot = set()
    if a.exclude_tasks_from:
        pilot = {(r["split"], r["task_id"]) for r in map(json.loads, open(a.exclude_tasks_from))}
    by = defaultdict(dict)
    for r in recs.values():
        if (r["split"], r["task_id"]) in pilot:
            continue
        by[(r["split"], r["task_id"])][r["condition"]] = r
    groups = defaultdict(list)
    for (split, tid), conds in by.items():
        if {"official", "cgl"} <= set(conds):
            groups[split].append(conds)
            if split in ("swebl", "pymatgen"):
                groups["pooled_heldout"].append(conds)
    summary = {"answers": a.answers, "excluded_tasks": len(pilot), "hypotheses": hypotheses(groups),
               "by_repo": {}, "splits": {}}
    by_repo = defaultdict(list)
    for split, rows in groups.items():
        if split != "pooled_heldout":
            for c in rows:
                by_repo[c["cgl"].get("repo") or c["cgl"]["task_id"].split("_")[0]].append(c)
    for repo, rows in sorted(by_repo.items()):
        summary["by_repo"][repo] = {"n": len(rows), "share_complete": round(float(np.mean([complete(c) for c in rows])), 3),
                                    "hit@5 cgl - official": ci([hit5(c["cgl"]) - hit5(c["official"]) for c in rows])}
    for split, rows in sorted(groups.items()):
        S = {"n_tasks_paired": len(rows), "by_condition": {}, "paired_differences": {}}
        per = {c: defaultdict(list) for c in ("official", "cgl")}
        for conds in rows:
            for c in ("official", "cgl"):
                r = conds[c]
                sc = scores(r["answer"], r["gold"])
                for k, v in sc.items():
                    per[c][k].append(v)
                per[c]["reached"].append(float(bool(r["gold_seen"])))
                per[c]["gold_in_graph"].append(float(bool(r["gold_in_graph"])))
                per[c]["tool_calls"].append(r["n_tool_calls"])
                per[c]["turns"].append(r["n_turns"])
                per[c]["cost"].append(r.get("cost_usd") or 0.0)
                per[c]["answered"].append(float(bool(r["answer"])))
                per[c]["outside_graph_answers"].append(r.get("n_outside_graph", 0))
                # gold hits that only count because the model named an id absent from its graph
                per[c]["hit@5_outside_graph"].append(float(any(x in set(r["gold"]) and x not in
                                                                set(r["gold_in_graph"]) for x in r["answer"][:K])))
        for c in ("official", "cgl"):
            P = per[c]
            assert len(P["hit@5"]) == len(rows)
            reached = np.array(P["reached"]) == 1
            tools = Counter()
            fin = Counter()
            for conds in rows:
                tools.update(conds[c]["tool_counts"])
                fin[conds[c]["finish"]] += 1
            S["by_condition"][c] = {
                **{k: ci(P[k]) for k in ("hit@1", "hit@5", "recall@5", "mrr@5", "reached", "gold_in_graph", "answered",
                                         "hit@5_outside_graph")},
                "mean_outside_graph_answers": round(float(np.mean(P["outside_graph_answers"])), 3),
                "hit@5_given_reached": ci(np.array(P["hit@5"])[reached]),
                "mean_tool_calls": round(float(np.mean(P["tool_calls"])), 2),
                "mean_turns": round(float(np.mean(P["turns"])), 2),
                "tool_calls_total": dict(tools), "finish": dict(fin),
                "cost_usd": round(float(np.sum(P["cost"])), 4)}
        for k in ("hit@1", "hit@5", "recall@5", "mrr@5", "reached"):
            S["paired_differences"][f"cgl - official [{k}]"] = ci(np.array(per["cgl"][k]) - np.array(per["official"][k]))
        both = (np.array(per["cgl"]["reached"]) == 1) & (np.array(per["official"]["reached"]) == 1)
        S["paired_differences"][f"cgl - official [hit@5 | reached in both, n={int(both.sum())}]"] = \
            ci(np.array(per["cgl"]["hit@5"])[both] - np.array(per["official"]["hit@5"])[both])
        summary["splits"][split] = S
    out = Path(a.out or Path(a.answers).with_name("summary.json"))
    out.write_text(json.dumps(summary, indent=1))
    print("Pre-registered hypotheses (docs/preregistration_agent.md):")
    for k, v in summary["hypotheses"].items():
        e = v["est"]
        print(f"  {k}\n     {v['outcome']}: " + (f"{e[0]:+.3f} [{e[1]:+.3f}, {e[2]:+.3f}]" if e else "-") +
              "  " + ", ".join(f"{x}={y}" for x, y in v.items() if x.startswith("n")))
    for split, S in summary["splits"].items():
        print(f"\n[{split}] paired tasks: {S['n_tasks_paired']}")
        for c, v in S["by_condition"].items():
            print(f"  {c:9s} hit@1={v['hit@1'][0]:.3f} hit@5={v['hit@5'][0]:.3f} rec@5={v['recall@5'][0]:.3f} "
                  f"reached={v['reached'][0]:.3f} hit@5|reached={(v['hit@5_given_reached'] or [float('nan')])[0]:.3f} "
                  f"calls={v['mean_tool_calls']} cost=${v['cost_usd']}")
        for k, v in S["paired_differences"].items():
            if v:
                print(f"  {k:45s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")


if __name__ == "__main__":
    main()
