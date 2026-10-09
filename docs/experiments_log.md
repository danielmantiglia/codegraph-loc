# Localization Experiments — Results

## Experiment 1 — Issue → Function Localization on the 129 Public Tasks

*30 September 2026 · `scripts/07_localization_benchmark.py` · n = 128 tasks with Python gold edits (fastapi 66, rich 48, requests 13, httpx 1) · 95% bootstrap CIs (5,000 resamples, paired where noted)*

## Setup

- **Query:** the task's `problem_statement` only (no hints).
- **Gold:** functions/classes/modules edited by the reference patch, in the competition id convention. A gold node missing from a graph can never be retrieved from it, so coverage gaps count against that graph.
- **Candidates:** every node of the graph (tests included; see caveats).
- **Metrics:**
  - Hit@k: at least one gold node in the top k;
  - Recall@k: share of gold nodes in the top k;
  - MRR: mean reciprocal rank of the first gold node;
  - file-level Hit@k.
- **Methods (pre-registered, no tuning):** RRF k = 60, PPR α = 0.85, BM25 k1 = 1.2 and b = 0.75.
  - `ident`: identifiers found in the issue, resolved to nodes with the harness's own resolver.
  - `+emb`: seeds followed by neighbours in the released embeddings, i.e. what `search_similar_code` returns.
  - `+ppr`: seeds followed by Personalized PageRank on a graph.
  - `bm25`: BM25 over node id + source code.
  - `rrf`: reciprocal-rank fusion of BM25 and PPR, with PPR seeded by the identifiers plus the BM25 top 10.

## Main results

| Method | Hit@1 | Hit@5 (95% CI) | Hit@10 | Recall@10 | MRR | File Hit@5 |
|---|---|---|---|---|---|---|
| `ident` (identifiers only) | 0.094 | 0.180 (0.117–0.250) | 0.211 | 0.146 | 0.131 | 0.312 |
| `ident+emb_raw` (released embeddings) | 0.094 | 0.180 (0.117–0.250) | 0.234 | 0.156 | 0.137 | 0.367 |
| `ident+emb_centered` | 0.094 | 0.188 (0.117–0.258) | 0.234 | 0.156 | 0.138 | 0.352 |
| `ident+ppr_official` | 0.094 | 0.203 (0.133–0.273) | 0.242 | 0.169 | 0.142 | 0.391 |
| `ident+ppr_cgl` | 0.102 | 0.234 (0.164–0.312) | 0.305 | 0.203 | 0.162 | 0.398 |
| `bm25_official` | 0.117 | 0.266 (0.195–0.344) | 0.320 | 0.215 | 0.195 | 0.438 |
| `bm25_cgl` | **0.164** | 0.320 (0.242–0.398) | 0.375 | 0.262 | 0.236 | 0.445 |
| `rrf_official` | 0.109 | 0.297 (0.219–0.375) | 0.375 | 0.259 | 0.205 | 0.484 |
| `rrf_cgl` | 0.148 | **0.336** (0.258–0.422) | **0.422** | **0.307** | **0.245** | 0.484 |

Only 57.8% of issues mention an identifier that resolves to a node. For the other 42.2%, a `search_similar_code` call based on the issue returns nothing.

## Paired differences (95% CI; an interval excluding 0 means the difference is significant)

| Contrast | Hit@5 | Recall@10 | MRR |
|---|---|---|---|
| Released embeddings vs identifiers alone (`ident+emb_raw − ident`) | **+0.000** (0.000, 0.000) | +0.010 (0.000, 0.028) | +0.006 (0.002, 0.010) |
| Centering the embeddings (`emb_centered − emb_raw`) | +0.008 (0.000, 0.023) | 0.000 | +0.001 (0.000, 0.003) |
| Released call graph vs identifiers alone (`ident+ppr_official − ident`) | +0.023 (0.000, 0.055) | +0.023 (0.007, 0.047) | +0.011 (0.004, 0.020) |
| **Same BM25, cgl nodes vs released nodes** | **+0.055 (0.008, 0.109)** | **+0.047 (0.008, 0.089)** | **+0.041 (0.006, 0.079)** |
| Same PPR, cgl graph vs released graph (`ident+ppr`) | +0.031 (−0.008, 0.078) | **+0.034 (0.004, 0.070)** | **+0.020 (0.003, 0.043)** |
| Same fusion, cgl vs released (`rrf`) | +0.039 (−0.016, 0.094) | +0.048 (−0.002, 0.098) | **+0.041 (0.001, 0.082)** |

## Ablation — what the cgl graph needs (`rrf_cgl` minus the ablated variant)

| Removed from the cgl graph | Hit@5 | Recall@10 | MRR |
|---|---|---|---|
| async definitions | +0.023 (0.000, 0.055) | **+0.023 (0.005, 0.046)** | −0.003 (−0.028, 0.020) |
| module nodes | +0.008 (−0.039, 0.055) | +0.012 (−0.031, 0.055) | +0.008 (−0.028, 0.044) |
| `contains` edges | 0.000 (−0.031, 0.031) | +0.003 (−0.027, 0.031) | **−0.025 (−0.046, −0.007)** |
| `imports` edges | 0.000 (−0.023, 0.023) | +0.001 (−0.005, 0.008) | **−0.020 (−0.038, −0.006)** |
| `inherits` edges | 0.000 | −0.003 (−0.008, 0.000) | −0.004 (−0.012, 0.001) |
| everything except `calls` | +0.016 (−0.016, 0.047) | −0.005 (−0.036, 0.024) | **−0.030 (−0.061, −0.001)** |
| official schema (calls only, no async, no modules) | +0.016 (−0.039, 0.070) | +0.025 (−0.026, 0.075) | +0.011 (−0.030, 0.052) |

A positive value means the full graph is better; a negative value means removing that part *helps*.

## Findings

1. **The released embeddings add no measurable localization signal.** Expanding the identifier seeds with `search_similar_code` neighbours leaves Hit@5 exactly unchanged on all 128 tasks and raises MRR by 0.006. Centering does not rescue them. This agrees with the audit: the vectors are effectively ~4-dimensional.
2. **Node coverage is what matters.** The same BM25 retriever gains +5.5 points of Hit@5 simply from searching cgl's node set, which includes async definitions and module-level code. Removing async nodes from cgl costs 2.3 points of Recall@10.
3. **More edge types are not automatically better.** Untyped PPR over `contains` and `imports` edges spreads probability onto module and class hubs. Removing those edges *improves* MRR by 2–3 points; `calls` edges alone give the best MRR (0.275). The released graph's call edges do help a little over identifiers alone (+2.3 points Recall@10).
4. **The task is hard.** The best method still places a gold function in the top 5 for only a third of issues. Many problem statements are PR templates with little signal, for example a one-line title followed by an unfilled checklist.

## Caveats and follow-ups

- n = 128 with fastapi at 52%; httpx has a single task. Per-repo numbers are in `results/loc/summary.json`. All claims need confirmation on the larger benchmark mined from public history.
- Candidates include test code, while gold never does. A secondary analysis restricted to non-test candidates is a fair and common choice and should be added, applied to every method.
- PPR is untyped and unweighted, and no parameter was tuned. Typed propagation (for example, down-weighting `contains` and `imports`) is a natural next step and must be reported as exploratory.
- The Gemma 4 baseline is reported in Experiment 3 (re-ranking of graph candidates).


## Experiment 1b — Secondary analysis: test code removed from every ranking (5 Oct)

Gold locations are never in test code, so removing test nodes from every ranking is a fair secondary analysis (`--exclude-tests`, applied identically to all methods; `results/loc_notest`).

| Method | Hit@1 | Hit@5 (95% CI) | Recall@10 | MRR | File Hit@5 |
|---|---|---|---|---|---|
| `ident` | 0.102 | 0.195 (0.125–0.266) | 0.153 | 0.142 | 0.328 |
| `ident+emb_raw` | 0.102 | 0.203 (0.133–0.273) | 0.170 | 0.150 | 0.391 |
| `ident+ppr_official` | 0.102 | 0.227 (0.156–0.305) | 0.178 | 0.158 | 0.422 |
| `ident+ppr_cgl` | 0.102 | 0.273 (0.195–0.352) | 0.226 | 0.175 | 0.430 |
| `bm25_official` | 0.148 | 0.383 (0.297–0.469) | 0.318 | 0.270 | 0.633 |
| `bm25_cgl` | 0.180 | 0.414 (0.328–0.500) | 0.364 | 0.286 | 0.578 |
| `rrf_official` | 0.188 | 0.391 (0.305–0.477) | 0.328 | 0.281 | 0.656 |
| `rrf_cgl` | 0.203 | 0.422 (0.336–0.508) | 0.367 | 0.311 | 0.648 |

Paired differences (non-test):

- Released embeddings vs identifiers alone: Hit@5 +0.008 (0.000, 0.023), Recall@10 +0.016 (0.002, 0.037), MRR +0.008 (0.004, 0.014). The signal is small but not zero once tests are removed, so "negligible" is the accurate word, not "none".
- Same BM25, cgl vs released nodes: Hit@5 +0.031 (−0.023, 0.086), Recall@10 +0.046 (−0.006, 0.098), MRR +0.017 (−0.021, 0.058). The direction matches the primary analysis, but with n = 128 the result is not significant.
- Same identifier-seeded PPR, cgl vs released graph: Recall@10 **+0.048 (0.017, 0.086)**; Hit@5 +0.047 (0.000, 0.094).
- Ablations: removing `imports` edges improves MRR by 2.8 points (CI 1.2–4.8); removing async nodes costs 2.3 points of Hit@5 (0.0–5.5).

Conclusion: the 128 competition tasks are under-powered for the coverage question. This motivated Experiment 2.

## Experiment 2 — Public-history split (n = 507, 5 Oct; corrected the same day)

- **Data.** Code+test commits mined from public git history (2019 → Sep 2026), excluding the 129 competition PRs. The query is the PR title + description, fetched from GitHub with `scripts/08_fetch_pr_texts.py`; this is the same construction as the competition's problem statements. Gold labels use the competition id convention.
- **Composition.** n = 507: httpx 297, fastapi 191, requests 13, rich 6. It complements Experiment 1, which is dominated by fastapi and rich.
- **No released graphs exist for these commits.** The released graph's coverage is therefore *emulated* by restricting cgl to the released schema (`official_schema`: sync definitions only, no module nodes, `calls` edges only). On the 127 competition commits this schema reproduces 98.4% of the released nodes.
- Gold coverage of cgl: 100% of tasks.
- Script: `scripts/09_public_history_benchmark.py`; results in `results/public_loc`.

> **Correction (5 Oct).** The first version had n = 505. Six commit titles cite two PR numbers (e.g. `Revert "X (#2523)" (#2539)`), and the miner took the first number instead of the last, which GitHub appends. As a result:
> - 4 tasks used the text of a linked issue or of the reverted PR;
> - 2 pairs of commits (an original and its revert) collapsed onto one task id.
>
> `02_mine_commits.py` now takes the last number. The 6 affected tasks were rebuilt with the right PR text, and the 2 collapsed tasks were recovered: n = 507. Point estimates moved by at most 0.007 (paired differences by at most 0.002); no confidence interval changed side of zero and no conclusion changed. The superseded files are kept in `results/public_loc/superseded/`.

| Method | Hit@1 | Hit@5 (95% CI) | Recall@10 | MRR | Hit@5 non-test | Recall@10 non-test |
|---|---|---|---|---|---|---|
| `ident+ppr_official_schema` | 0.164 | 0.274 (0.237–0.314) | 0.169 | 0.212 | 0.288 | 0.181 |
| `ident+ppr_cgl` | 0.168 | 0.304 (0.264–0.345) | 0.222 | 0.231 | 0.329 | 0.242 |
| `bm25_official_schema` | 0.248 | 0.442 (0.398–0.485) | 0.326 | 0.347 | 0.546 | 0.373 |
| `bm25_cgl` | 0.258 | 0.460 (0.416–0.501) | 0.354 | 0.360 | 0.576 | 0.444 |
| `rrf_cgl-official_schema` | 0.258 | 0.509 (0.466–0.550) | 0.347 | 0.373 | 0.560 | 0.392 |
| `rrf_cgl` | 0.227 | 0.489 (0.448–0.531) | 0.378 | 0.352 | 0.584 | 0.469 |
| `rrf_cgl-calls_only` | 0.258 | 0.536 (0.493–0.580) | 0.375 | 0.382 | 0.596 | 0.450 |

### Paired differences

The intervals are 95% CIs. "All" means every node is a candidate; "non-test" means test code is removed from the ranking.

**1. Coverage: full cgl node set vs the released schema**

| Contrast | Hit@5 (all) | Recall@10 (all) | Recall@10 (non-test) | MRR (non-test) |
|---|---|---|---|---|
| BM25 | +0.018 (−0.008, 0.043) | **+0.028 (0.006, 0.049)** | **+0.070 (0.048, 0.093)** | **+0.028 (0.010, 0.046)** |
| Identifier-seeded PPR | **+0.030 (0.004, 0.055)** | **+0.053 (0.034, 0.073)** | **+0.061 (0.041, 0.083)** | **+0.026 (0.013, 0.040)** |
| BM25+PPR fusion | −0.020 (−0.053, 0.014) | **+0.031 (0.007, 0.056)** | **+0.077 (0.052, 0.103)** | −0.002 (−0.025, 0.022) |

**2. Async definitions (`rrf_cgl` minus the variant without them)**

| Analysis | Hit@5 | Recall@10 | MRR |
|---|---|---|---|
| all | +0.018 (−0.006, 0.041) | **+0.031 (0.014, 0.049)** | +0.013 (−0.003, 0.029) |
| non-test | **+0.039 (0.014, 0.065)** | **+0.060 (0.041, 0.080)** | **+0.028 (0.011, 0.047)** |

**3. Module nodes (`rrf_cgl` minus the variant without them)**

| Analysis | Hit@5 | Recall@10 | MRR |
|---|---|---|---|
| all | **−0.028 (−0.049, −0.004)** | −0.010 (−0.028, 0.007) | **−0.019 (−0.033, −0.006)** |
| non-test | −0.004 (−0.028, 0.020) | +0.017 (−0.001, 0.035) | −0.011 (−0.024, 0.002) |

A negative value means the module nodes hurt.

**4. Structural edges (`rrf_cgl` minus the `calls`-only variant)**

| Analysis | Hit@5 | Recall@10 | MRR |
|---|---|---|---|
| all | **−0.047 (−0.071, −0.024)** | +0.003 (−0.009, 0.014) | **−0.030 (−0.046, −0.014)** |
| non-test | −0.012 (−0.035, 0.012) | **+0.019 (0.003, 0.036)** | **−0.025 (−0.040, −0.010)** |

**5. Graph propagation over BM25 alone (`rrf_cgl − bm25_cgl`, all candidates)**

| Hit@5 | Recall@10 | MRR |
|---|---|---|
| **+0.030 (0.002, 0.055)** | **+0.024 (0.012, 0.037)** | −0.009 (−0.029, 0.012) |

### What Experiment 2 establishes (with power)

1. **Covering every definition raises recall significantly** across all three retrievers: +2.8 to +5.3 points of Recall@10 with all candidates, +6.1 to +7.7 in the non-test analysis (the one reported in the paper). Async definitions are the main reason: removing them from cgl costs 6.0 points of Recall@10 and 3.9 points of Hit@5 in the non-test analysis.
2. **The coverage gain is not free at the top of the ranking.** Extra nodes, especially module nodes, compete for the first positions. For fusion, Hit@5 and MRR do not improve, and module nodes lower Hit@5 by 2.8 points.
3. **Untyped structural edges trade precision for recall.** `calls`-only propagation gives the best Hit@5 (0.536 all, 0.596 non-test) and MRR. Adding `contains`/`imports`/`inherits` gains about 2 points of Recall@10 (non-test) but costs 2.5–3.0 points of MRR. This replicates the Experiment 1 ablation on a different repository mix.
4. **The recommended configuration is cgl nodes with async definitions and `calls` edges, used for propagation.** It was selected on this split, so it is tested on held-out repositories in Experiment 4 (pre-registered in `docs/preregistration_heldout.md`).

### Caveats

- Per repository, fastapi alone shows the fusion method doing worse on the full cgl graph than on the official schema (Hit@5 0.346 vs 0.414). The likely cause is the >1,000 `docs_src` example modules full of async endpoints. This deserves a per-repo breakdown in the paper.
- rich (6) and requests (13) are thin in this split.
- Queries are PR descriptions, which sometimes name the changed function; this applies equally to every method.

## Experiment 3 — Gemma 4 re-ranking of graph candidates (n = 635, 5 Oct; corrected the same day)

*`scripts/10_build_llm_prompts.py`, `11_run_llm.py`, `12_eval_llm.py` · results in `results/llm/summary.json` (aggregates only; prompts and responses contain competition text and stay local) · 95% paired bootstrap CIs, 5,000 resamples, seed 0*

**Question.** Does the coverage gap of the released graph still matter when a capable model, Gemma 4, chooses the locations, or can the model compensate?

**Design.** For each task the same candidate generator runs on two graphs:

- `official`: the released graph for the 128 competition tasks. For the 507 public-history tasks, the released schema is emulated on cgl (sync definitions only, no module nodes, `calls` edges).
- `cgl`: the full cgl graph with `calls` edges.

The candidates are:

- the BM25 top 25 for the issue (test code excluded);
- plus up to 15 `calls` neighbours of the BM25 top 5;
- shuffled with a fixed seed. The mean is 31–32 candidates.

Gemma 4 31B-it sees the issue (truncated to 3,000 characters) and each candidate's id, signature and first docstring line, and names the 5 candidates to edit.

Inference settings:

- served through OpenRouter, restricted to bf16 providers (Novita 1,227 requests, Crusoe 43);
- temperature 0, seed 0, reasoning off.

The run:

- 1,270 requests, 0 failures, all answers valid JSON;
- 243 answers named fewer than 5 candidates. Missing picks are not filled in.
- **Total cost: about $0.30.**
- After the Experiment 2 correction, the 12 requests of the 6 superseded tasks were dropped (kept in `results/llm/superseded/`) and 16 requests for the 8 rebuilt tasks were added.

Baselines on the same candidates:

- BM25 order (first 5 candidates);
- the exact expected score of 5 random candidates.

### Results

| Split | Condition | Gold in candidates | Random@5 | BM25 Hit@5 | **Gemma Hit@1** | **Gemma Hit@5** | Gemma Recall@5 | Gemma MRR@5 |
|---|---|---|---|---|---|---|---|---|
| public (507) | official schema | 0.757 | 0.255 | 0.564 | 0.489 | 0.700 | 0.415 | 0.575 |
| public (507) | cgl | 0.826 | 0.293 | 0.602 | **0.564** | **0.763** | **0.498** | **0.647** |
| competition (128) | released graph | 0.594 | 0.144 | 0.383 | 0.430 | 0.555 | 0.408 | 0.481 |
| competition (128) | cgl | 0.633 | 0.163 | 0.383 | 0.445 | 0.570 | 0.408 | 0.493 |

**Paired differences, cgl − official**

| Metric | public (n = 507) | competition (n = 128) | pooled (n = 635) |
|---|---|---|---|
| Gold among candidates (Hit) | **+0.069 (0.039, 0.099)** | +0.039 (−0.039, 0.117) | **+0.063 (0.035, 0.093)** |
| Gold among candidates (Recall) | **+0.117 (0.087, 0.146)** | +0.029 (−0.038, 0.097) | **+0.099 (0.071, 0.126)** |
| BM25 Hit@5 | **+0.037 (0.008, 0.069)** | 0.000 (−0.062, 0.070) | **+0.030 (0.003, 0.057)** |
| Gemma Hit@1 | **+0.075 (0.037, 0.112)** | +0.016 (−0.055, 0.086) | **+0.063 (0.030, 0.098)** |
| Gemma Hit@5 | **+0.063 (0.030, 0.097)** | +0.016 (−0.055, 0.094) | **+0.053 (0.022, 0.085)** |
| Gemma Recall@5 | **+0.083 (0.054, 0.112)** | 0.000 (−0.057, 0.055) | **+0.066 (0.041, 0.092)** |
| Gemma MRR@5 | **+0.072 (0.040, 0.105)** | +0.012 (−0.053, 0.078) | **+0.060 (0.032, 0.089)** |

**Gemma vs the baselines on the same candidates (Hit@5)**

| Comparison | public, official | public, cgl | competition, official | competition, cgl |
|---|---|---|---|---|
| Gemma − BM25 | **+0.136 (0.104, 0.170)** | **+0.162 (0.124, 0.201)** | **+0.172 (0.102, 0.250)** | **+0.188 (0.109, 0.273)** |
| Gemma − random | **+0.446 (0.412, 0.479)** | **+0.470 (0.440, 0.502)** | **+0.410 (0.341, 0.479)** | **+0.407 (0.337, 0.478)** |

### Where the gain comes from

Gemma's Hit@5 *when a gold location is among its candidates* is the same in both conditions:

| Split | official | cgl |
|---|---|---|
| public | 0.924 (0.896, 0.951) | 0.924 (0.897, 0.948) |
| competition | 0.934 (0.868, 0.987) | 0.901 (0.827, 0.963) |

On the tasks whose gold is in *both* candidate sets, the difference is −0.008 (−0.033, 0.016) on public (n = 367) and −0.030 (−0.091, 0.030) on competition (n = 66).

The model does not choose better from cgl candidates. It succeeds more often because the gold is more often there to be chosen. The whole cgl gain is a coverage effect.

**Robustness.** Restricting the analysis to tasks whose two requests were both served by Novita gives the same public result: Hit@5 +0.065 (0.029, 0.100) and Recall@5 +0.081 (0.052, 0.111), n = 478.

### What Experiment 3 establishes

1. **Gemma 4 is a strong selector but cannot recover what the graph lacks.** When the right location is among about 30 candidates, Gemma 4 puts it in its top 5 90–93% of the time. That is 14–19 points of Hit@5 above BM25 and 41–47 above chance on the same candidates. Its success therefore depends on candidate coverage, which the graph determines.
2. **On the public-history split, building candidates from the full cgl graph instead of the released schema raises Gemma 4's Hit@1 by 7.5 points, Hit@5 by 6.3 and Recall@5 by 8.3, with all CIs above zero.** With BM25 alone, the same switch adds 3.7 points of Hit@5. The coverage effect is larger with the model than without it, because the model converts covered candidates into hits more often. This amplification is observed; it is not formally tested.
3. **On the 128 competition tasks, the direction is the same but the result is not significant** (Hit@5 +1.6, CI −5.5 to +9.4). This matches Experiment 1b: n = 128 is under-powered for this question, and fastapi dominates the split.

### Caveats

- The API-served model (bf16, two providers) is not the competition's QAT W4A16 build, and the run uses one temperature-0 sample per request.
- The candidate generator, BM25 plus `calls` neighbours, is part of the system: a different generator would change absolute numbers. Both conditions use the same generator.
- In the public split, the `official` condition is an *emulation* of the released schema. On the competition split it is the real released graph.
- Gold locations come from one reference patch; Gemma may sometimes name a valid alternative.
- Module-level gold can come from trivial edits (see the pymatgen label audit), and module nodes exist only in cgl. The sensitivity analysis in Experiment 4 shows that on this split the Gemma gain persists without module-level gold (+3.3 Hit@5, CI −0.2 to 6.5), and the coverage gain in candidates is +4.3 (1.4–7.4).

## Experiment 4 — Pre-registered held-out confirmation (n = 744, 5 Oct)

*Pre-registration: `docs/preregistration_heldout.md`, registered before any held-out task was built. Scripts: `13_heldout_benchmark.py` (retrieval + prompts), `11_run_llm.py`, `12_eval_llm.py --heldout`, `run_heldout_mac.sh`. Results: `results/heldout/loc/summary.json`, `results/heldout/llm/summary.json`.*

**Data.**

- **H-SWE: SWE-bench Lite, 294 tasks from 11 repositories** (requests excluded): django 114, sympy 77, matplotlib 23, scikit-learn 23, pytest 17, sphinx 16, astropy 6, pylint 6, xarray 5, seaborn 4, flask 3.
- **H-PMG: pymatgen, 450 PR tasks.** 451 were mined; one has no Python gold edit.
- **Composition of the gold locations:**
  - none is async, in either split;
  - 7.3% are module-level;
  - 17.1% of tasks have at least one module-level gold location (8.8% in SWE-bench Lite, 22.4% in pymatgen).

The released schema's main gap on the development repositories, async definitions, is therefore absent here. The only node type it misses is module-level code.

**Runs.**

- 169 tasks were built in the cloud (Python 3.11) and the rest on a Mac (Python 3.14). The same task gives identical records and prompts in both environments (checked on two tasks).
- One sympy snapshot exceeded Python's default recursion depth during AST traversal. The limit was raised to 20,000; outputs are otherwise unchanged.
- Gemma: 1,488 requests, 0 failures, $0.41 (Novita 1,418, Crusoe 70).

### Pre-registered hypotheses (pooled, n = 744)

| # | Contrast | Metric | Estimate (95% CI) | Outcome |
|---|---|---|---|---|
| H1 | Fusion, `calls`-only: all definitions − released schema (non-test) | Recall@10 | +0.007 (−0.003, 0.017) | **not confirmed** |
| H2 | Fusion: `calls`-only − all edge types (non-test) | MRR | +0.017 (0.003, 0.032) | **confirmed** |
| H3 | Gemma: cgl candidates − official-schema candidates | Hit@5 | +0.021 (0.007, 0.036) | **confirmed** |
| H4 | Gemma, gold in both candidate sets (n = 552) | Hit@5 | +0.007 (−0.002, 0.018) | **confirmed** (inside ±0.05) |

**Per split (secondary).**

| Split | H1 | H2 | H3 |
|---|---|---|---|
| SWE-bench Lite | −0.004 (−0.019, 0.010) | +0.010 (−0.010, 0.029) | +0.010 (−0.014, 0.034) |
| pymatgen | **+0.014 (0.001, 0.027)** | **+0.022 (0.003, 0.042)** | **+0.029 (0.011, 0.049)** |

### Other contrasts (secondary)

**Retrieval, pooled, non-test.**

- BM25, all definitions vs released schema: Recall@10 +0.005 (−0.005, 0.015).
- Identifier-seeded PPR: Recall@10 **+0.077 (0.057, 0.096)**. Issue identifiers often resolve to module nodes, which then seed the propagation.
- Fusion with all edge types vs the released schema: **+0.033 (0.020, 0.048)**.
- Module nodes alone: **+0.014 (0.003, 0.026)**.
- With all candidates, untyped structural edges cost **2.9 points of MRR (1.4–4.4)**, replicating Experiments 1–2.

**Gemma, pooled.**

| Measure | official | cgl | Difference (95% CI) |
|---|---|---|---|
| Gold among candidates | 0.753 | 0.771 | +0.019 (0.005, 0.034) |
| Hit@1 | | | +0.007 (−0.013, 0.027) |
| Recall@5 | | | +0.017 (0.005, 0.031) |
| Hit@5 when the gold is among the candidates | 0.952 | 0.956 | |

- Gemma beats BM25 on the same candidates by **16–18 points of Hit@5** and chance by **52–54**.
- Single-provider sensitivity (Novita, n = 681): Hit@5 +0.022 (0.007, 0.038).

### Exploratory: without module-level gold (`15_sensitivity_module_gold.py`, not pre-registered)

The label audit found that module-level gold can come from trivial edits such as added `__author__` metadata, and module nodes exist only in cgl. We therefore removed every module-level gold location and dropped tasks left without gold.

| Split | Module share of gold | BM25 Recall@10 | Fusion (`calls`) Recall@10 | Gold among Gemma candidates | Gemma Hit@5 |
|---|---|---|---|---|---|
| Public history (n = 490) | 9.4% | **+0.047 (0.023, 0.071)** | **+0.037 (0.013, 0.061)** | **+0.043 (0.014, 0.074)** | +0.033 (−0.002, 0.065) |
| Held-out pooled (n = 718) | 7.3% | **−0.012 (−0.019, −0.006)** | −0.007 (−0.015, 0.000) | **−0.011 (−0.021, −0.003)** | −0.004 (−0.015, 0.007) |

The two splits behave differently:

- **Development repositories.** The coverage gain survives without module-level gold; it comes from async definitions.
- **Held-out repositories.** The small confirmed gain (H3) comes entirely from module-level targets. On function-level targets the extra module nodes slightly *crowd out* the right functions: −1.2 points of BM25 Recall@10.

### What Experiment 4 establishes

1. **Whether coverage helps depends on what the fixes touch.** On repositories whose fixes never touch async code, adding the missing definitions barely changes retrieval (H1 not confirmed). On the development repositories, where fixes often touch async definitions, it clearly does. The released graph's gap matters in proportion to the share of fixes it cannot represent, which can be measured per repository before a graph is built.
2. **The edge-type finding generalizes (H2).** `calls`-only propagation ranks the right location higher than propagation over all edge types, on both held-out splits pooled and in the all-candidate analysis.
3. **Gemma's behaviour generalizes (H3, H4).** Selection accuracy given coverage is the same with either graph: 95% held-out, 92–93% development. Gemma's Hit@5 gain therefore tracks the change in candidate coverage:

   | Split | Coverage change | Gemma Hit@5 change |
   |---|---|---|
   | Public | +6.9 | +6.3 |
   | Held-out | +1.9 | +2.1 |
   | Competition | +3.9 | +1.6 |
4. **Module nodes are a trade-off.** They make module-level fixes (7–9% of gold) reachable, but add candidates that compete with functions. Whether to include them should depend on how often a repository's fixes touch module-level code.

## Experiment 5 — Gemma 4 as a graph-navigating agent (pre-registered; 2,718 episodes, 8–9 Oct)

**Design** (`docs/preregistration_agent.md`; `src/cgl/agent.py`, `scripts/17_agent_localize.py`, `scripts/18_eval_agent.py`, `scripts/run_agent_mac.sh`). Gemma 4 31B-it receives the issue and four tools over one graph: `search_code` (BM25 over non-test definitions, top 10), `get_neighbors` (callers and callees along `calls` edges, 10 each), `read_code` (first 60 lines) and `submit_answer` (5 ranked ids). Budget 20 tool calls, 30 turns (protocol v2). Every task runs twice: with the released graph (competition tasks) or its emulated schema (public and held-out tasks), and with our full graph (`calls` edges). OpenRouter, CoreWeave fp4 pinned, temperature 0, seed 0, reasoning off. A correct full id outside the agent's graph is kept (conservative for our graph).

**Pilots** (20 public tasks, mechanics only, no accuracy inspected): v1 (budget 12, "up to 5" ids) → 40% of episodes exhausted the budget and answers had 1.9 ids on average; v2 (budget 20, "5 ranked ids") → frozen. Pilot tasks are excluded from all analyses.

**Run.** 1,359 tasks × 2 graphs; all episodes ended with an answer; 4 episodes failed on the OpenRouter key limit and were re-run after a resume-guard fix (deviation logged). Cost $18.38 ($0.0068 per episode). Scoring identical on macOS (Python 3.14) and Linux (Python 3.11).

### Pre-registered hypotheses (Hit@5, cgl − official, points)

| # | Population | Estimate (95% CI) | Outcome |
|---|---|---|---|
| H5 | public history, n = 487 | **+6.4 (3.1, 9.9)** | confirmed |
| H6 | held-out pooled, n = 744 | +1.6 (−0.4, 3.6) | not confirmed |
| H7 | all splits: gain with an edited location missing from `official` (n = 449) +11.8 (8.0, 15.6) vs all present (n = 910) −1.0 (−2.9, 0.8) | **+12.8 (8.6, 17.0)** | confirmed |
| H8 | held-out, every edited location in `official`, n = 617 | +0.3 (−1.8, 2.4), inside ±5 | confirmed |

### Secondary and exploratory (`14_figures.py` → `results/figures/figure_data.json`, key `agent`)

| Split | Hit@5 diff | Missing stratum (n) | Present stratum (n) |
|---|---|---|---|
| Competition (real released graph) | +0.8 (−7.0, 8.6) | +16.7 (2.1, 31.2) (48) | −8.8 (−16.2, −1.2) (80) |
| Public history | +6.4 (3.1, 9.9) | +12.8 (7.7, 17.9) (274) | −1.9 (−5.6, 1.4) (213) |
| SWE-bench Lite | +2.7 (−0.7, 6.1) | +7.7 (0.0, 19.2) (26) | +2.2 (−1.1, 6.0) (268) |
| pymatgen | +0.9 (−1.6, 3.3) | +7.9 (2.0, 13.9) (101) | −1.1 (−3.7, 1.4) (349) |
| Held-out pooled | +1.6 (−0.4, 3.6) | +7.9 (3.1, 13.4) (127) | +0.3 (−1.8, 2.4) (617) |

- *Reached* (a tool output showed an edited location), public: 80.9% → 91.6%; Hit@5 given reached in both (n = 388, descriptive): +0.5 (−2.3, 3.4).
- Competition tasks fully covered by the real released graph: reached 88.8% (released) vs 80.0% (ours). The emulated schema shows no such loss on the public split (−1.9, −5.6 to 1.4); unexplained, and one of eight split × stratum cells.
- Per repository: gain vs the `cgl-profile` share of fixes not fully representable by a released-style graph, Spearman ρ = 0.485 (p = 0.057), 16 repositories (5 with fewer than 10 tasks).
- Mechanics: 14.2 tool calls per episode (40% used all 20); 2.55 ids per answer (equal across graphs); `get_neighbors` 2% of 38,555 calls.

### What Experiment 5 establishes

1. **The graph is the ceiling for an agent too.** With our graph the agent reaches the edited code more often; once both reach it, they select equally well.
2. **The gain sits where the released graph lacks the edited code (H7), and only there (H8).** Overall gains therefore depend on how often a repository's fixes touch omitted code: large on the async-heavy development repositories (H5), small and not significant held out (H6).
3. **An agent barely uses edges.** Node coverage matters more than edge structure for this kind of agent.
