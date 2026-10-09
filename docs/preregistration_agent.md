# Pre-registration — Graph-navigating agent (Experiment 5)

*Registered: 2026-10-08 19:50 UTC, after the two pilots (§3, §8) and before any full-run episode was run. Saved to the Kaggle project and committed to the public repository at that time; any later change is logged in §8 with its date.*

## 1. Question

Experiments 3–4 showed that Gemma 4 picks the edited location from a fixed list of about 30 candidates equally well with either graph, so the graph sets the ceiling. The competition's agent does not receive a list: it explores the graph with tools. Experiment 5 asks whether the ceiling still holds when Gemma 4 searches the graph itself, and whether the gain follows the code that fixes touch (the `cgl-profile` claim of §3 of the paper).

## 2. Design

- **Agent** (`src/cgl/agent.py`): Gemma 4 receives the issue and four tools over one graph:
  - `search_code`: BM25 over non-test definitions, top 10, as `id — signature`;
  - `get_neighbors`: up to 10 callers and 10 callees along `calls` edges;
  - `read_code`: the first 60 lines of a definition;
  - `submit_answer`: 5 ids ranked from most to least likely (at most 5 are scored); ends the episode.
- **Limits:** 20 tool calls (submission is free), 30 model turns, 2 reminders if the model stops calling tools; after that, dotted ids are parsed from its last message. (Protocol v2; the pilot changes from v1 are in §8.)
- **Conditions** (same prompt, tools, limits, model and decoding; only the graph differs):
  - `official`: the released graph for competition tasks; elsewhere its emulated schema on our graph (sync definitions and classes only, no module nodes, `calls` edges only), as in Experiments 2–4;
  - `cgl`: our full graph, `calls` edges only.
- **Answers:** a submitted name is mapped to a node id (exact id, or a unique short name). A full dotted id that is not in the agent's graph is **kept**. This is conservative for our hypotheses: a missing node blocks discovery through the tools, not the answer.
- **Model:** `google/gemma-4-31b-it` on OpenRouter with native tool calling, one pinned provider (CoreWeave, fp4: the closest available build to the competition's 4-bit model), no fallbacks, temperature 0, seed 0, reasoning off, at most 800 output tokens per turn. Provider, tokens and cost are recorded per episode.
- **Tasks:** every task of the four splits: held-out SWE-bench Lite (294) and pymatgen (450), public history (507, of which the 20 pilot tasks are skipped), competition (128): 1,359 tasks, 2,718 episodes. Gold labels are those of Experiments 2–4, unchanged.
- **Scripts:** `scripts/17_agent_localize.py` (episodes), `scripts/18_eval_agent.py` (scoring), `scripts/run_agent_mac.sh` (both).

## 3. Pilot and freeze

- 20 public-history tasks sampled with seed 0 (`--sample-tasks 20`), both conditions, written to `results/agent/pilot/`.
- Only the mechanics report is inspected (`18_eval_agent.py --mechanics-only`): finish reasons, tool-call errors, budget use, tokens, cost and failures, pooled over both conditions. **No accuracy is computed or looked at.**
- Allowed changes, decided from that report only: tool budget, turn limit, output-token limit, output caps of the tools, the wording of the system prompt and reminders, the provider. Any change is listed in §8.
- **The 20 pilot tasks are excluded from every analysis below** (`--exclude-tasks-from`).

## 4. Frozen at registration

- **Protocol v2** (`cgl.agent.PROTOCOL`). Code fingerprints (sha256, first 16 hex): `agent.py` a429c25540cfb9c2 · `17_agent_localize.py` fa84e01b2e10b1e8 · `18_eval_agent.py` a800dc285d83c7d6 · `run_agent_mac.sh` 1c3f800dc07dc50d · `pipeline.py` c46fb56cae0b6687 · `bench.py` b0460d3d1229e221 · `graph.py` 9e217f214b8280fa · `labels.py` 7351a13dc6a60624 · `11_run_llm.py` b37720a5ee0b3a71. The last five are unchanged since the Experiment 4 registration.
- **Statistics:** paired bootstrap over tasks, 5,000 resamples, seed 0, 95% percentile intervals, as in Experiment 4. A hypothesis is **confirmed** when the interval excludes 0 in the predicted direction (H8: the interval lies inside the margin).
- **Missing data:** an episode that fails (API or graph error) is re-run by running the script again, at most three times. A task still missing one condition is dropped from the paired analyses and counted. Tasks are never dropped for their outcome.
- **Budget:** pilot 2 cost $0.007 per episode, so the 2,718 episodes of the full run should cost about $19. The run is capped at $25 for the 31B model. If the cap stops it early, it is raised (up to the project's $40 total) and the run is resumed; if it cannot be completed, the completed tasks are reported with the count of missing ones.

## 5. Hypotheses

Metric: **Hit@5**, the share of tasks with an edited location among the agent's (at most) five answers. Contrast: `cgl` − `official`, paired by task.

| # | Claim | Population | Predicted |
|---|---|---|---|
| H5 | With the full graph the agent localizes better where fixes often touch code the released graph omits | Public history, pilot excluded (n = 487) | > 0 |
| H6 | The gain persists on repositories that played no part in any decision | Held-out pooled, SWE-bench Lite ∪ pymatgen (n = 744) | > 0 |
| H7 | The gain is concentrated where the released graph lacks an edited location: the gain on tasks with an edited location absent from the `official` graph exceeds the gain on tasks with all of them present | All non-pilot tasks of the four splits, pooled | difference of the two gains > 0 |
| H8 | The fuller graph does not distract the agent: when every edited location is in both graphs, the two graphs give the same Hit@5 | Held-out pooled, tasks with every edited location in the `official` graph | 95% CI inside [−0.05, +0.05] |

The strata of H7 and H8 depend only on the task and the graphs, not on what the agent does.

**Expectations, not tests.** The profiles predict a large H5 effect (30–35% of fastapi and httpx fixes touch async code) and a small H6 effect: no held-out fix touches async code, and 7.3% of held-out edited locations are module-level. In Experiment 4 the held-out re-ranking gain was +2.1 points and came from module-level code only.

## 6. Secondary reporting (not confirmatory)

Reported whatever the outcome:

- each split separately, including the competition split with the **real released graph** (n = 128; under-powered for differences of a few points);
- Hit@1, Recall@5, MRR@5;
- *reached*: the share of episodes in which a tool output showed an edited location; Hit@5 given reached (selection; this conditions on the agent's behaviour, so it is descriptive);
- hits obtained by naming an id absent from the agent's graph;
- tool use, turns, finish reasons, tokens and cost per condition;
- per repository: the agent's gain against the `cgl-profile` share of fixes a released-style graph does not fully represent (Spearman correlation over the 16 repositories);
- if the budget allows after the main run, the same analysis with `google/gemma-4-26b-a4b-it` (NextBit, bf16; same protocol, cap $15) as a second model.

Any other analysis is labelled exploratory.

## 7. Limits stated in advance

- The agent localizes; it does not edit or test code. End-to-end resolution is out of scope, as in Experiments 1–4.
- Tool outputs are ours, not the harness's (the harness tools are audited in §2 of the paper); both conditions share them.
- The public and held-out splits emulate the released schema.

## 8. Changes after the pilot and deviations

- **2026-10-08, pilot 1 (protocol v1), mechanics only.** 40 episodes (20 tasks × 2 graphs), 0 failures, all ended with `submit_answer`, $0.11 in total ($0.0028 per episode), CoreWeave for every request. Two problems with the loop, both seen without any accuracy figure:
  1. **The tool budget bound too often:** 40% of episodes used all 12 calls, and 15 further calls were refused. → Budget raised to **20** calls and the turn limit to **30**.
  2. **Answers were short:** 1.9 ids on average although 5 are scored, so "Hit@5" would mostly have measured the top two. → The prompt, the `submit_answer` description and the reminders now ask for **5 ids ranked from most to least likely**.
  Everything else is unchanged. This is protocol **v2** (`cgl.agent.PROTOCOL`), recorded in every episode; the scripts refuse to mix protocols in one output folder. Pilot 2 repeats the same 20 tasks with v2 (`results/agent/pilot_v2/`); those tasks stay excluded from every analysis.
- **2026-10-08, pilot 2 (protocol v2, same 20 tasks), mechanics only.** 40 episodes, 0 failures, all ended with `submit_answer`, $0.28 in total ($0.0070 per episode), CoreWeave for every request. Mean answer length 2.98 ids (5 ids in 10 of 40 episodes; 1 id in 7). 47.5% of episodes used all 20 tool calls, about the same share as with 12 calls, so the model tends to spend whatever budget it has; raising it further would mainly add cost. **No further change: protocol v2 is frozen.** The full run skips the 20 pilot tasks (`--exclude-tasks-from`) and is capped at $25 (§4).
- Observation kept for the paper, not acted on: the agent rarely follows edges (12 `get_neighbors` calls against 217 `read_code` and 147 `search_code`), so this experiment mainly tests which nodes the graph contains. Pilot 2 is similar (19 `get_neighbors` calls against 366 `read_code` and 215 `search_code`).

- **Registration commit.** The header gives 19:50 UTC, written just before committing; the commit that registered this document is `99d5c17` in the public repository, 2026-10-08 19:48:27 UTC.
- **2026-10-09, bug fix in the resume guard of `17_agent_localize.py`, during the full run (no effect on episodes, prompts or scoring).** After 1,390 of 2,718 episodes the run stopped because the OpenRouter key's own spending limit ($10) was reached (HTTP 403 on 4 episodes; the account still had credit). On restart, the guard that refuses to mix protocols read the 4 failure records, which carried no `protocol` field, as protocol v1 and refused to resume. Fix: failure records now carry the protocol, and only successful episodes are checked. Nothing else changed; `17_agent_localize.py` fingerprint is now 0965244cbf9632d5 (was fa84e01b2e10b1e8). The key limit was raised by Daniel and the run resumed in the same folder.

## 9. Outcome (added 9 Oct 2026, after the full run)

Full run: 2,718 episodes (1,359 tasks × 2 graphs, the 20 pilot tasks excluded), all ended with an answer (2,717 by `submit_answer`, 1 by the text fallback); 4 episodes failed on the key limit and were re-run (§8). Provider: CoreWeave for every request. Cost $18.38 ($0.0068 per episode); with the pilots, Experiment 5 cost $18.77. Scoring (`18_eval_agent.py`) gives identical summaries on macOS (Python 3.14) and Linux (Python 3.11).

| # | Estimate, Hit@5 cgl − official (95% CI) | Outcome |
|---|---|---|
| H5 | public history, n = 487: **+0.064 (0.031, 0.099)** | confirmed |
| H6 | held-out pooled, n = 744: +0.016 (−0.004, 0.036) | not confirmed |
| H7 | gain with an edited location missing from `official` (n = 449): +0.118 (0.080, 0.156); with all present (n = 910): −0.010 (−0.029, 0.008); difference **+0.128 (0.086, 0.170)** | confirmed |
| H8 | held-out tasks with every edited location in `official`, n = 617: +0.003 (−0.018, 0.024), inside ±0.05 | confirmed |

**Secondary (§6), as registered.**
- Per split, Hit@5 cgl − official: competition (real released graph, n = 128) +0.008 (−0.070, 0.086); SWE-bench Lite (n = 294) +0.027 (−0.007, 0.061); pymatgen (n = 450) +0.009 (−0.016, 0.033).
- *Reached* (a tool output showed an edited location), public split: 0.809 → 0.916, +0.107 (0.076, 0.138). Hit@5 given reached in both conditions: +0.005 (−0.023, 0.034). As in Experiments 3–4, the gain is in what the agent can find, not in how it chooses.
- With the `official` graph, 6.6% of public tasks were hit by naming an id absent from that graph (kept by the conservative answer rule, §2).
- Per repository, the agent's gain against the `cgl-profile` share of fixes a released-style graph does not fully represent: Spearman ρ = 0.48 (p = 0.057) over the 16 repositories, several of which have fewer than 10 tasks.
- Mechanics: 14.2 tool calls per episode (40% used all 20); 2.55 ids per answer (5 ids in 25% of episodes, 1 id in 38%); `get_neighbors` was 2% of tool calls.
- The second model (Gemma 4 26B A4B) was not run at the time of writing.

Any other analysis of these data is exploratory.
