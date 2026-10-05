# codegraph-loc

**What should a code graph contain?** An open code-graph generator and a function-level
bug-localization benchmark for local coding agents such as Gemma 4.

Work in progress for the Kaggle *Google – The Gemma 4 Developer Agent* Paper Track (2026).
License: Apache-2.0.

## What is here (v0.4, 5 Oct 2026)

| Component | File | Status |
|---|---|---|
| Graph generator: module/class/function/method nodes **incl. async**; `contains`, `imports`, `inherits`, `calls` edges with a `resolution` tag per edge; export to the competition's node-link JSON schema; optional competition id convention (`official_view`) | `src/cgl/graph.py` | v0, unit-tested |
| Gold localization labels from a unified diff (innermost function/method; indentation-aware insertions; import-only edits marked incidental) | `src/cgl/labels.py` | v0, unit-tested |
| Download of the competition files needed for the audit (graphs, embeddings, tasks, harness wheels) | `scripts/00_download_competition_data.py` | done |
| Graph statistics for fastapi / rich / requests / httpx | `scripts/01_repo_stats.py` | done |
| Mining of code+test commits from public git history, with gold labels | `scripts/02_mine_commits.py` | done |
| How many gold targets a sync-only, module-less graph can represent (public history) | `scripts/03_visibility.py` | done |
| Audit of the released graphs: node reproduction, `calls` overlap, gold coverage on the 129 public tasks | `scripts/04_audit_official_graphs.py` | done |
| Audit of the released embeddings: anisotropy, nearest-neighbour cosine, duplicates, `search_similar_code` output size | `scripts/05_audit_embeddings.py` | done |
| What the released embeddings encode: effective dimensionality, correlation with name/length/degree | `scripts/06_embedding_geometry.py` | done |
| Issue → function localization benchmark on the 129 public tasks (BM25, identifier seeds, released embeddings, PPR on released vs cgl graph, RRF, ablations; `--exclude-tests` secondary analysis) | `scripts/07_localization_benchmark.py`, `src/cgl/bench.py` | done |
| PR title + description for mined commits (GitHub API with a token, or polite web fetch) | `scripts/08_fetch_pr_texts.py` | done |
| Public-history localization benchmark (507 tasks; released schema emulated on any commit) | `scripts/09_public_history_benchmark.py`, `src/cgl/pipeline.py` | done |
| Gemma 4 re-ranking baseline: candidates from the released vs the cgl graph, prompts → OpenRouter (standard library only, key read from a local file) → paired evaluation | `scripts/10_build_llm_prompts.py`, `scripts/11_run_llm.py`, `scripts/12_eval_llm.py` | done |
| Pre-registered held-out confirmation: SWE-bench Lite (294) + pymatgen (450); retrieval, Gemma prompts, one-command Mac runner | `scripts/13_heldout_benchmark.py`, `scripts/run_heldout_mac.sh` | done |
| Paper figures; sensitivity analysis without module-level gold | `scripts/14_figures.py`, `scripts/15_sensitivity_module_gold.py` | done |

## Reproduce

```bash
mkdir -p ../repos && for r in fastapi/fastapi Textualize/rich psf/requests encode/httpx; do
  git clone https://github.com/$r ../repos/$(basename $r); done
pip install -e ".[dev,baselines]"
pytest

# public-history analyses (no competition data needed)
python scripts/01_repo_stats.py --repos-dir ../repos
python scripts/02_mine_commits.py --repos-dir ../repos --since 2019-01-01
python scripts/03_visibility.py

# audits of the released competition data (requires accepting the competition rules;
# the data is read in place and never redistributed)
python scripts/00_download_competition_data.py ../dati
python scripts/04_audit_official_graphs.py --data ../dati/competition --repos ../repos --out results/audit
python scripts/04_audit_official_graphs.py --data x --repos x --out results/audit --summarize
python scripts/05_audit_embeddings.py --data ../dati/competition --out results/emb
python scripts/05_audit_embeddings.py --data x --out results/emb --summarize
python scripts/06_embedding_geometry.py --data ../dati/competition --out results/emb_geometry.json
python scripts/07_localization_benchmark.py --data ../dati/competition --repos ../repos --out results/loc
python scripts/07_localization_benchmark.py --data x --repos x --out results/loc --summarize

# public-history split (public data only)
python scripts/08_fetch_pr_texts.py --mined results --out results/public/pr_texts.jsonl --exclude-tasks ../dati/competition/tasks.jsonl
python scripts/09_public_history_benchmark.py --mined results --prs results/public/pr_texts.jsonl --repos ../repos --out results/public_loc
python scripts/09_public_history_benchmark.py --out results/public_loc --summarize

# Gemma 4 re-ranking baseline (needs the outputs of 07 --exclude-tests and 09, and an OpenRouter key)
python scripts/10_build_llm_prompts.py --data ../dati/competition --repos ../repos --out results/llm
python3 scripts/11_run_llm.py --limit 10      # pilot; asks for the key once and stores it in a local file
python3 scripts/11_run_llm.py --workers 8     # full run (1,266 requests, about $0.40), resumable
python scripts/12_eval_llm.py

# pre-registered held-out test (Experiment 4; pre-registration: docs/preregistration_heldout.md)
# needs the SWE-bench Lite test split (princeton-nlp/SWE-bench_Lite, Hugging Face) as ../dati/swebench_lite_test.jsonl
git clone https://github.com/materialsproject/pymatgen ../repos/pymatgen
python scripts/02_mine_commits.py --repos-dir ../repos --only pymatgen=pymatgen \
       --since 2019-01-01 --until 2026-03-02 --pr-merges --out-dir results/heldout/pymatgen
python scripts/08_fetch_pr_texts.py --mined results/heldout/pymatgen --out results/heldout/pymatgen/pr_texts.jsonl
bash scripts/run_heldout_mac.sh     # clones the other repositories, builds 744 tasks, runs Gemma (1,488 requests, about $0.41)
python scripts/12_eval_llm.py --heldout --prompts results/heldout/llm/prompts.jsonl \
       --responses results/heldout/llm/responses.jsonl --out results/heldout/llm/summary.json
python scripts/14_figures.py                         # Figures 1-2 and results/figures/figure_data.json
python scripts/15_sensitivity_module_gold.py --repos ../repos
```

Full experiment log: `docs/experiments_log.md`.

**What is published.** Code, gold labels (`results/mined_*.jsonl`, `results/heldout/pymatgen/mined_pymatgen.jsonl`),
per-task rankings of the public split, Gemma's answers on the held-out split, every summary and the figures.
Not published: anything derived from the competition data beyond aggregate statistics, and the texts of issues and
pull requests (the scripts fetch them again from GitHub and Hugging Face).

## Results so far (preliminary)

**Public history.** On 653 code+test commits (2019 → Sep 2026) from the four repositories, a graph
without async definitions and without module nodes can represent **all** gold edit locations for only
**50.5%** of commits (95% bootstrap CI 46.9–54.2) and **none** of them for **11.6%** (9.3–14.2).

**Released competition data** (129 public tasks, 127 graphs, 127 embedding files; harness `swegemma` 0.2.7):

- All gold nodes present in the released graph for **62.5%** of tasks (53.9–71.1), none for 8.6% (3.9–14.1);
  the cgl graph contains all of them for 100% of tasks. The missing 19.6% of gold nodes are module-level
  code (61) and async functions (51); the released graphs contain no async definition at all.
- With the reverse-engineered id convention, cgl reproduces **98.4%** of the released nodes (97.5–99.2).
- The released 256-d embeddings are strongly anisotropic (mean random-pair cosine 0.79 raw, 0.09 centered)
  and effectively ~4-dimensional (4 principal components explain 90% of the variance; random control: 187).
  97% of nodes have a nearest neighbour with cosine ≥ 0.99; similarity barely correlates with code, name
  or length similarity (Spearman ≤ 0.02) and only weakly with call-graph degree (0.17).
- `search_similar_code` resolves its query to a node *name* (free text is never embedded) and returns full
  source: for k = 10 the output exceeds 5,000 characters for 28% of nodes and 100,000 characters for 0.9%.

**Localization on the 128 public tasks with Python gold edits** (query = problem statement; 95% paired bootstrap CIs):

- Expanding issue identifiers with `search_similar_code` neighbours leaves Hit@5 unchanged (Δ = 0.000) and adds 0.006 MRR:
  the released embeddings carry negligible localization signal (non-test analysis: Recall@10 +1.6, CI 0.2–3.7).
- The same BM25 retriever over cgl nodes instead of the released nodes: Hit@5 0.320 vs 0.266 (Δ +0.055, CI 0.008–0.109).
  Best overall: BM25 + PPR fusion on the cgl graph, Hit@5 0.336, Recall@10 0.307.
- Ablation: dropping async nodes costs 2.3 points Recall@10 (CI 0.5–4.6); untyped `contains`/`imports` edges *hurt*
  PPR (MRR −2 to −3 points), `calls`-only propagation gives the best MRR (0.275).

**Public-history split** (507 tasks mined from GitHub, query = PR title + description, released graph emulated by
restricting cgl to its schema): covering every definition raises Recall@10 by 6.1–7.7 points (non-test rankings) for BM25, identifier-seeded
PPR and their fusion (all paired 95% CIs above zero); async definitions alone account for 6.0 points (non-test analysis).
Module nodes and untyped structural edges lower Hit@5/MRR by 2–5 points; `calls`-only propagation is best at the top of
the ranking (Hit@5 0.536, 0.596 without test code).

**Gemma 4 re-ranking** (635 tasks; Gemma 4 31B-it chooses 5 of ~31 graph candidates; 1,270 requests, about $0.30): on the
public-history split, candidates from the full cgl graph instead of the released schema raise Gemma's Hit@1 by 7.5 points
and Hit@5 by 6.3 (0.763 vs 0.700; paired 95% CIs above zero). When the gold location is among the candidates, Gemma puts
it in its top 5 about 93% of the time with either graph, so the gain is entirely candidate coverage. Gemma beats BM25 by
14–19 points of Hit@5 on the same candidates. On the 128 competition tasks the effect has the same sign but is not significant.

**Pre-registered held-out test** (744 tasks: SWE-bench Lite and pymatgen; hypotheses registered before any held-out task was built).

- **Confirmed.** `calls`-only propagation ranks better than untyped structural edges (MRR +1.7). Gemma gains +2.1 Hit@5 with cgl candidates. Its selection accuracy given coverage is unchanged (95%).
- **Not confirmed.** The retrieval coverage gain (Recall@10 +0.7, CI −0.3 to 1.7). None of the held-out fixes touches async code, the released schema's main gap.
- **Exploratory: without module-level gold.** The held-out gain disappears, while the public-history gain survives through async definitions.
