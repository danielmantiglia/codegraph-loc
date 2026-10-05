# Pre-registration — Held-out Confirmation (Experiment 4)

*Registered: 2026-10-05 11:15 UTC, before any held-out task was built or scored. Saved to the Kaggle project at that time; any later change is logged in §7 with its date.*

## 1. Why

The analysis choices in Experiments 1–3 were made on fastapi, rich, requests and httpx. They include:

- the recommended configuration: all definitions including async, and `calls`-only propagation;
- the Gemma 4 re-ranking design.

Experiment 4 tests the same claims on repositories that played no part in any decision.

## 2. Data

**H-SWE: SWE-bench Lite, test split**

- Source: `princeton-nlp/SWE-bench_Lite` (300 instances). We record the sha256 of the parquet file on receipt.
- Excluded: the 6 `psf/requests` instances, because requests is a development repository. That leaves 294 instances from 10 repositories.
- Query: `problem_statement` only. `hints_text` is not used.
- Snapshot: `base_commit`.
- Gold: the non-test, pre-existing `.py` edits of `patch`, labelled with `cgl.pipeline.gold_for` exactly as in Experiments 1–2. Instances with no such edit are dropped and counted.

**H-PMG: pymatgen (materialsproject/pymatgen), first-parent history**

- Window: 2019-01-01 → 2026-03-02. The end date is the day before the pymatgen-core code moved to a separate repository (#4595). After that, snapshots no longer contain `pymatgen.core`.
- Selection: the mining rules of Experiment 2 (`02_mine_commits.py`), unchanged. A commit qualifies if it:
  - modifies at least one library file and at least one test file;
  - touches at most 5 library files and at most 200 library lines;
  - has a title that passes the same skip-list.
- Extension: commits merged with a GitHub merge commit ("Merge pull request #N") are included and diffed against their first parent. pymatgen uses both squash and merge commits.
- Query: PR title + description fetched from GitHub, at least 20 characters. This is the construction of Experiment 2 and of the competition tasks.
- All qualifying tasks are used. There is no subsampling.

## 3. Methods (frozen)

- **Retrieval** (`cgl.pipeline.cgl_rankings`, unchanged): BM25 (k1 = 1.2, b = 0.75), identifier-seeded PPR (α = 0.85), BM25+PPR fusion (RRF, k = 60), and the graph views of Experiment 2. The released graph is emulated as `official_schema`: sync definitions only, no module nodes, `calls` edges only.
- **Gemma 4 re-ranking**: identical to Experiment 3.
  - Same candidate generator: BM25 top 25 (test code excluded) + up to 15 `calls` neighbours of the BM25 top 5.
  - Same prompt, model (`google/gemma-4-31b-it`) and settings: bf16 providers, temperature 0, seed 0, reasoning off.
  - Shuffle seed `"{split}|{task_id}"`.
  - `10_build_llm_prompts.py` is extended only so that it can read held-out task files; `signature`, `candidates`, `make_prompt`, `SYSTEM` and `INSTR` are not changed.
- **Code fingerprints at registration** (sha256, first 16 hex): `pipeline.py` c46fb56cae0b6687 · `bench.py` b0460d3d1229e221 · `graph.py` 9e217f214b8280fa · `labels.py` 7351a13dc6a60624 · `11_run_llm.py` b37720a5ee0b3a71.
- **Statistics**: paired bootstrap, 5,000 resamples, seed 0, 95% percentile intervals. A hypothesis is **confirmed** when the interval excludes 0 in the predicted direction.

## 4. Primary population

**The primary population is the pooled held-out set, H-SWE ∪ H-PMG.** Each split is also reported separately as a secondary result.

**Retrieval hypotheses use the non-test analysis**: test code is removed from every ranking. Gold locations are never test code, and Experiment 3's candidates also exclude it.

## 5. Confirmatory hypotheses

| # | Claim | Contrast | Metric | Predicted |
|---|---|---|---|---|
| H1 | Node coverage helps retrieval | `rrf_cgl-calls_only` − `rrf_cgl-official_schema`: same `calls`-only propagation; they differ only in the async definitions and module nodes | Recall@10 | > 0 |
| H2 | Untyped structural edges hurt the top of the ranking | `rrf_cgl-calls_only` − `rrf_cgl` | MRR | > 0 |
| H3 | The coverage gain survives an LLM re-ranker | Gemma, cgl candidates − official-schema candidates | Hit@5 | > 0 |
| H4 | The gain is coverage, not better selection | Gemma, cgl − official, on tasks with a gold location in *both* candidate sets | Hit@5 | 95% CI inside [−0.05, +0.05] (equivalence margin of 5 points) |

## 6. Expectations and secondary reporting (not confirmatory)

- **H1 may be smaller on H-SWE.** The coverage effect in Experiment 2 came mostly from async definitions. Several SWE-bench repositories (sympy, astropy, scikit-learn) contain little or no async code. For each split we report the share of gold locations that are async or module-level.
- **Reported for every split**, whatever the outcome:
  - all Experiment 2 contrasts (BM25, identifier PPR, fusion; each ablation);
  - the "all candidates" analysis;
  - Gemma's Hit@1 / Recall@5 / MRR@5;
  - BM25 and random baselines;
  - selection accuracy given coverage;
  - a single-provider sensitivity analysis.
- **Comparison with published SWE-bench Lite localization numbers** is contextual only: definitions of gold and metrics differ.
- **Every hypothesis is reported as confirmed or not confirmed**, including negative outcomes. No parameter is changed after held-out results are seen. Any analysis not listed here is labelled exploratory.

## 7. Deviations log

- **2026-10-05 ~11:45 UTC: PR-number fix in the miner (not a change to the held-out design).** `02_mine_commits.py` now takes the *last* PR number in a commit title instead of the first, which fixes 6 development-split tasks (Experiment 2). No pymatgen commit title in the H-PMG window cites two PR numbers, so H-PMG is unaffected (re-mining with the fixed script gives identical records for all 680 commits); this was checked before any held-out result was looked at.
- **2026-10-05: `10_build_llm_prompts.py` extended** to read held-out task files (`--heldout-tasks`), as announced in §3. Its output for the existing splits was checked to be byte-identical before and after the change.

- **2026-10-05: counting error in §2 (no change to the design).** The 294 SWE-bench Lite instances come from **11** repositories, not 10 (django, sympy, matplotlib, scikit-learn, pytest, sphinx, astropy, pylint, xarray, seaborn, flask). The instance list and every analysis are unchanged; the registered text is left as written.

## 8. Outcome (added 5 Oct 2026, after the held-out run)

Pooled held-out set: n = 744 (SWE-bench Lite 294, pymatgen 450). None of the gold locations is async; 7.3% are module-level.

| # | Estimate (95% CI) | Outcome |
|---|---|---|
| H1 | Recall@10 +0.007 (−0.003, 0.017) | not confirmed |
| H2 | MRR +0.017 (0.003, 0.032) | confirmed |
| H3 | Gemma Hit@5 +0.021 (0.007, 0.036) | confirmed |
| H4 | +0.007 (−0.002, 0.018), inside ±0.05 | confirmed |

**Exploratory, not pre-registered.** Without module-level gold, the H3 gain disappears (−0.004, −0.015 to 0.007). The held-out gain therefore comes from module-level targets only. Details are in `docs/experiments_log.md`, Experiment 4.
