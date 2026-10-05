"""Score the Gemma 4 re-ranking baseline (responses of 11_run_llm.py).

For each task x condition (``official`` vs ``cgl`` candidates):
  * cand_hit / cand_recall — is any / which share of the gold locations among the candidates
    (upper bound for any re-ranker);
  * bm25@5  — the first 5 candidates in BM25 order (no LLM);
  * llm@5   — the 5 candidates chosen by Gemma (invalid or missing picks are not replaced);
  * rand@5  — expected score of 5 candidates drawn uniformly at random (chance level).
Also reported: Gemma's Hit@5 when the gold is among the candidates (selection accuracy), the
cgl - official difference restricted to tasks whose gold is in both candidate sets, and a
sensitivity check restricted to tasks whose two requests were served by the same provider.
Metrics: hit@1, hit@5, recall@5, mrr@5. Paired bootstrap (5,000 resamples, seed 0) for
cgl - official and llm - bm25, per split (comp, public) and pooled. Only tasks answered
in *both* conditions enter the paired comparisons.

Usage: python scripts/12_eval_llm.py [--prompts results/llm/prompts.jsonl]
           [--responses results/llm/responses.jsonl] [--out results/llm/summary.json]
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from math import comb
from pathlib import Path

import numpy as np

K = 5


def parse_answer(text: str, n: int):
    """Candidate numbers (1-based) in the model's order; JSON first, then any integers."""
    text = text or ""
    picks, ok_json = [], False
    for m in re.finditer(r"\{[^{}]*\}", text, re.S):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        ans = obj.get("answer") if isinstance(obj, dict) else None
        if isinstance(ans, list):
            picks, ok_json = ans, True
            break
    if not ok_json:
        picks = re.findall(r"\d+", text)
    out = []
    for p in picks:
        try:
            i = int(p)
        except (TypeError, ValueError):
            continue
        if 1 <= i <= n and i not in out:
            out.append(i)
    return out[:K], ok_json


def scores(top, gold):
    g = set(gold)
    ranks = [i + 1 for i, x in enumerate(top[:K]) if x in g]
    return {"hit@1": float(bool(top) and top[0] in g), "hit@5": float(bool(ranks)),
            "recall@5": len(set(top[:K]) & g) / (len(g) or 1), "mrr@5": 1.0 / ranks[0] if ranks else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="results/llm/prompts.jsonl")
    ap.add_argument("--responses", default="results/llm/responses.jsonl")
    ap.add_argument("--out", default="results/llm/summary.json")
    ap.add_argument("--heldout", action="store_true",
                    help="also evaluate pre-registered H3/H4 on the pooled set (Experiment 4)")
    a = ap.parse_args()
    prompts = {p["custom_id"]: p for p in map(json.loads, open(a.prompts))}
    resp = {}
    for r in map(json.loads, open(a.responses)):
        if r.get("ok"):
            resp[r["custom_id"]] = r  # last successful answer wins

    rows, meta = {}, {}  # (split, task, cond) -> metrics / serving provider
    n_json, n_short, providers, cost, ptok, ctok, finish = 0, 0, Counter(), 0.0, [], [], Counter()
    for cid, r in resp.items():
        p = prompts.get(cid)
        if p is None:
            continue
        picks, ok_json = parse_answer(r["content"], len(p["candidates"]))
        n_json += ok_json
        n_short += len(picks) < K
        llm_top = [p["candidates"][i - 1] for i in picks]
        gold = p["gold"]
        cs = set(p["candidates"])
        m = {f"llm_{k}": v for k, v in scores(llm_top, gold).items()}
        m |= {f"bm25_{k}": v for k, v in scores(p["ranked"], gold).items()}
        n_c, g_in = len(p["candidates"]), sum(g in cs for g in gold)
        m["cand_hit"] = float(g_in > 0)
        m["cand_recall"] = g_in / (len(gold) or 1)
        m["n_candidates"] = n_c
        # expected score of 5 candidates drawn uniformly at random (chance baseline)
        m["rand_hit@5"] = 1.0 - comb(n_c - g_in, K) / comb(n_c, K) if n_c >= K else float(g_in > 0)
        m["rand_recall@5"] = m["cand_recall"] * min(1.0, K / n_c) if n_c else 0.0
        m["n_picks"] = len(picks)
        rows[(p["split"], p["task_id"], p["condition"])] = m
        meta[(p["split"], p["task_id"], p["condition"])] = str(r.get("provider"))
        providers[str(r.get("provider"))] += 1
        finish[str(r.get("finish_reason"))] += 1
        cost += r.get("cost_usd") or 0.0
        ptok.append(r.get("prompt_tokens") or 0)
        ctok.append(r.get("completion_tokens") or 0)

    rng = np.random.default_rng(0)
    keys = ["cand_hit", "cand_recall", "rand_hit@5", "rand_recall@5", "bm25_hit@5", "bm25_recall@5", "bm25_mrr@5",
            "llm_hit@1", "llm_hit@5", "llm_recall@5", "llm_mrr@5", "n_picks"]
    main_provider = providers.most_common(1)[0][0] if providers else None

    def ci(x):
        if len(x) == 0:
            return None
        bs = x[rng.integers(0, len(x), (5000, len(x)))].mean(1)
        return [round(float(x.mean()), 4), round(float(np.percentile(bs, 2.5)), 4),
                round(float(np.percentile(bs, 97.5)), 4)]

    summ = {"n_responses": len(rows), "json_answers": n_json, "answers_with_fewer_than_5_picks": n_short,
            "providers": dict(providers), "finish_reasons": dict(finish), "cost_usd": round(cost, 4),
            "mean_prompt_tokens": round(float(np.mean(ptok)), 1) if ptok else None,
            "mean_completion_tokens": round(float(np.mean(ctok)), 1) if ctok else None, "splits": {}}
    tasks = defaultdict(set)
    for (s, t, c) in rows:
        tasks[(s, t)].add(c)
    for split in sorted({s for s, _ in tasks}) + ["all"]:
        paired = sorted(k for k, cs in tasks.items() if {"official", "cgl"} <= cs and (split == "all" or k[0] == split))
        if not paired:
            continue
        S = {"n_tasks_paired": len(paired), "by_condition": {}, "paired_differences": {}}
        arr = {c: {k: np.array([rows[(s, t, c)][k] for s, t in paired]) for k in keys + ["n_candidates"]}
               for c in ("official", "cgl")}
        for c in ("official", "cgl"):
            S["by_condition"][c] = {k: ci(arr[c][k]) for k in keys}
            S["by_condition"][c]["mean_candidates"] = round(float(arr[c]["n_candidates"].mean()), 1)
        for k in ("cand_hit", "cand_recall", "bm25_hit@5", "llm_hit@1", "llm_hit@5", "llm_recall@5", "llm_mrr@5"):
            S["paired_differences"][f"cgl - official [{k}]"] = ci(arr["cgl"][k] - arr["official"][k])
        for c in ("official", "cgl"):
            for k in ("hit@5", "recall@5", "mrr@5"):
                S["paired_differences"][f"llm - bm25 [{c}, {k}]"] = ci(arr[c][f"llm_{k}"] - arr[c][f"bm25_{k}"])
            for k in ("hit@5", "recall@5"):
                S["paired_differences"][f"llm - random [{c}, {k}]"] = ci(arr[c][f"llm_{k}"] - arr[c][f"rand_{k}"])
        # Where does the cgl gain come from? Selection accuracy when the gold is among the candidates.
        S["llm_hit@5_given_gold_in_candidates"] = {
            c: ci(arr[c]["llm_hit@5"][arr[c]["cand_hit"] == 1]) for c in ("official", "cgl")}
        both = (arr["official"]["cand_hit"] == 1) & (arr["cgl"]["cand_hit"] == 1)
        S["paired_differences"][f"cgl - official [llm_hit@5 | gold in both candidate sets, n={int(both.sum())}]"] = \
            ci(arr["cgl"]["llm_hit@5"][both] - arr["official"]["llm_hit@5"][both])
        # Sensitivity: only tasks whose two requests were served by the same (most common) provider.
        same = np.array([meta[(s, t, "official")] == meta[(s, t, "cgl")] == main_provider for s, t in paired])
        for k in ("llm_hit@5", "llm_recall@5"):
            S["paired_differences"][f"cgl - official [{k} | both served by {main_provider}, n={int(same.sum())}]"] = \
                ci(arr["cgl"][k][same] - arr["official"][k][same])
        S["answers_with_fewer_than_5_picks"] = {c: int((arr[c]["n_picks"] < K).sum()) for c in ("official", "cgl")}
        summ["splits"][split] = S

    if a.heldout and "all" in summ["splits"]:
        d = summ["splits"]["all"]["paired_differences"]
        h3 = d["cgl - official [llm_hit@5]"]
        k4 = next(k for k in d if k.startswith("cgl - official [llm_hit@5 | gold in both"))
        h4 = d[k4]
        summ["preregistered"] = {
            "H3": {"contrast": "cgl - official [llm_hit@5]", "predicted": "> 0", "estimate_ci": h3,
                   "confirmed": bool(h3 and h3[1] > 0)},
            "H4": {"contrast": k4, "predicted": "95% CI inside [-0.05, +0.05]", "estimate_ci": h4,
                   "confirmed": bool(h4 and h4[1] >= -0.05 and h4[2] <= 0.05)}}
        for h, v in summ["preregistered"].items():
            print(h, v["contrast"], v["estimate_ci"], "CONFIRMED" if v["confirmed"] else "not confirmed")
    Path(a.out).write_text(json.dumps(summ, indent=1))
    print(f"responses: {len(rows)} | JSON answers: {n_json} | <5 picks: {n_short} | cost ${cost:.4f} | "
          f"providers: {dict(providers)} | finish: {dict(finish)}")
    for split, S in summ["splits"].items():
        print(f"\n[{split}] paired tasks: {S['n_tasks_paired']}")
        for c, v in S["by_condition"].items():
            print(f"  {c:9s} cand_hit={v['cand_hit'][0]:.3f} rand_hit@5={v['rand_hit@5'][0]:.3f} "
                  f"bm25_hit@5={v['bm25_hit@5'][0]:.3f} "
                  f"llm_hit@1={v['llm_hit@1'][0]:.3f} llm_hit@5={v['llm_hit@5'][0]:.3f} "
                  f"llm_rec@5={v['llm_recall@5'][0]:.3f} llm_mrr@5={v['llm_mrr@5'][0]:.3f} (cands {v['mean_candidates']})")
        for c, v in S["llm_hit@5_given_gold_in_candidates"].items():
            if v:
                print(f"  P(llm hit@5 | gold in candidates) {c:9s} {v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]")
        print(f"  answers with < 5 picks: {S['answers_with_fewer_than_5_picks']}")
        for k, v in S["paired_differences"].items():
            if v:
                print(f"  {k:40s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]")


if __name__ == "__main__":
    main()
