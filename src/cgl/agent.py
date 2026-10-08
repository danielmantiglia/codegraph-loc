"""A graph-navigating localization agent (Experiment 5).

The agent receives an issue and four tools over one code graph:

* ``search_code(query)``      — BM25 over the graph's non-test nodes (id + source text); top 10 as
                                 ``id — signature`` lines;
* ``get_neighbors(node)``     — callers and callees of a node along ``calls`` edges (up to 10 each);
* ``read_code(node)``         — the node's source, first 60 lines;
* ``submit_answer(locations)``— 5 node ids ranked from most to least likely (at most 5 are scored);
                                 ends the episode.

Only the graph changes between conditions; prompt, tools, budget, model and decoding are identical.
Every tool output is bounded, so the context never overflows. Test code is excluded everywhere,
as in the other experiments. The model is reached through an injected ``chat(messages, tools)``
function, so the loop is testable without network access.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field

from .bench import BM25, rank_from_scores, tokenize
from .pipeline import is_test_node

PROTOCOL = "v2"           # v1 (pilot 1): budget 12, 20 turns, "up to 5" ids; v2: changes logged in
                          # docs/preregistration_agent.md §8 after the pilot's mechanics report
TOOL_BUDGET = 20          # search/neighbors/read calls per episode (submit_answer is free)
MAX_TURNS = 30            # model turns per episode, including reminders
SEARCH_K, NEIGH_K, READ_LINES = 10, 10, 60
ISSUE_CHARS = 3000

SYSTEM = (
    "You are an expert Python developer. You must find where the code of a repository has to be edited "
    "to resolve an issue. You cannot run the code. You can explore the repository's code graph with tools: "
    "search_code finds definitions by keywords, get_neighbors lists the callers and callees of a definition, "
    "and read_code shows a definition's source. Node ids are dotted paths such as package.module.Class.method "
    "(a module id stands for the module's top-level code). You have a budget of {budget} tool calls. "
    "When you are confident, call submit_answer with 5 node ids ranked from most to least likely.")

TOOLS = [
    {"type": "function", "function": {
        "name": "search_code",
        "description": "Search the repository's definitions (functions, methods, classes, modules) by keywords. "
                       "Returns the 10 best matches as 'node_id — signature'.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "keywords, identifiers or a short description"}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "get_neighbors",
        "description": "List the definitions that call the given node and the definitions it calls.",
        "parameters": {"type": "object", "properties": {
            "node": {"type": "string", "description": "a node id (or a unique name)"}},
            "required": ["node"]}}},
    {"type": "function", "function": {
        "name": "read_code",
        "description": "Show the source code of a node (first 60 lines).",
        "parameters": {"type": "object", "properties": {
            "node": {"type": "string", "description": "a node id (or a unique name)"}},
            "required": ["node"]}}},
    {"type": "function", "function": {
        "name": "submit_answer",
        "description": "Submit the 5 node ids most likely to need editing, ranked from most to least likely. "
                       "Ends the task.",
        "parameters": {"type": "object", "properties": {
            "locations": {"type": "array", "items": {"type": "string"}, "description": "node ids"}},
            "required": ["locations"]}}},
]

NUDGE = "Continue with the tools, and finish by calling submit_answer with 5 ranked node ids."
BUDGET_DONE = "Tool budget exhausted. Call submit_answer now with 5 ranked node ids."


def signature(text: str) -> str:
    """First line of a def/class header (multi-line headers joined), or '(module)'."""
    lines = (text or "").splitlines()
    for i, l in enumerate(lines):
        s = l.strip()
        if s.startswith(("def ", "async def ", "class ")):
            parts = [s]
            for nxt in lines[i + 1:i + 8]:
                if parts[-1].endswith(":"):
                    break
                parts.append(nxt.strip())
            return re.sub(r"\(\s+", "(", re.sub(r"\s+\)", ")", " ".join(parts)))[:160]
    return "(module)"


class GraphEnv:
    """The tools over one graph view (nodes, directed `calls` edges, node source text)."""

    def __init__(self, nodes, edges, text: dict, files: dict):
        self.nodes = [n for n in nodes if not is_test_node(n, files.get(n))]
        self.ids = set(self.nodes)
        self.text = text
        self.bm = BM25([tokenize(n + " " + (text.get(n) or "")) for n in self.nodes])
        self.callees, self.callers = defaultdict(list), defaultdict(list)
        for u, v in edges:
            if u in self.ids and v in self.ids and u != v:
                if v not in self.callees[u]:
                    self.callees[u].append(v)
                if u not in self.callers[v]:
                    self.callers[v].append(u)
        self.by_last = defaultdict(list)
        for n in self.nodes:
            self.by_last[n.rsplit(".", 1)[-1]].append(n)

    def resolve(self, name: str):
        """(node id or None, candidates when ambiguous)."""
        s = (name or "").strip().strip("`'\"").removesuffix("()")
        if s in self.ids:
            return s, []
        cands = [n for n in self.by_last.get(s.rsplit(".", 1)[-1], []) if n == s or n.endswith("." + s)]
        if len(cands) == 1:
            return cands[0], []
        return None, cands[:8]

    def line(self, n: str) -> str:
        return f"{n} — {signature(self.text.get(n, ''))}"

    def search_code(self, query: str) -> str:
        top = rank_from_scores(self.nodes, self.bm.scores(tokenize(query or "")), SEARCH_K)
        return "\n".join(self.line(n) for n in top) if top else "No matching definitions."

    def _missing(self, name, cands):
        if cands:
            return f"'{name}' is ambiguous; did you mean one of:\n" + "\n".join(self.line(c) for c in cands)
        return f"No node named '{name}'. Use search_code to find node ids."

    def get_neighbors(self, node: str) -> str:
        n, cands = self.resolve(node)
        if n is None:
            return self._missing(node, cands)
        out = [f"Node: {n}", "Called by:"]
        out += [f"  {self.line(c)}" for c in self.callers[n][:NEIGH_K]] or ["  (none)"]
        out.append("Calls:")
        out += [f"  {self.line(c)}" for c in self.callees[n][:NEIGH_K]] or ["  (none)"]
        return "\n".join(out)

    def read_code(self, node: str) -> str:
        n, cands = self.resolve(node)
        if n is None:
            return self._missing(node, cands)
        lines = (self.text.get(n) or "").splitlines()
        body = "\n".join(lines[:READ_LINES])
        more = f"\n... ({len(lines) - READ_LINES} more lines)" if len(lines) > READ_LINES else ""
        return f"Node: {n}\n{body}{more}"

    def run(self, name: str, args: dict) -> str:
        if name == "search_code":
            return self.search_code(str(args.get("query", "")))
        if name == "get_neighbors":
            return self.get_neighbors(str(args.get("node", "")))
        if name == "read_code":
            return self.read_code(str(args.get("node", "")))
        return f"Unknown tool '{name}'."


ID_IN_LINE = re.compile(r"^\s*(?:Node: )?([A-Za-z_][\w.]*)(?: — |$)", re.M)
DOTTED = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+")


@dataclass
class Episode:
    answer_raw: list = field(default_factory=list)
    answer: list = field(default_factory=list)
    seen: list = field(default_factory=list)
    tool_counts: dict = field(default_factory=dict)
    n_tool_calls: int = 0
    n_turns: int = 0
    n_nudges: int = 0
    finish: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    providers: list = field(default_factory=list)
    n_outside_graph: int = 0   # submitted dotted ids kept although they are not nodes of this graph
    messages: list = field(default_factory=list)


def _args(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        v = json.loads(raw or "{}")
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def run_episode(env: GraphEnv, issue: str, chat, budget: int = TOOL_BUDGET, max_turns: int = MAX_TURNS) -> Episode:
    """Run one episode. ``chat(messages, tools)`` returns the API response dict (OpenAI format)."""
    ep = Episode()
    msgs = [{"role": "system", "content": SYSTEM.format(budget=budget)},
            {"role": "user", "content": f"Issue:\n{(issue or '')[:ISSUE_CHARS]}"}]
    seen, counts = [], defaultdict(int)
    for _ in range(max_turns):
        r = chat(msgs, TOOLS)
        ep.n_turns += 1
        u = r.get("usage") or {}
        ep.prompt_tokens += u.get("prompt_tokens") or 0
        ep.completion_tokens += u.get("completion_tokens") or 0
        ep.cost_usd += u.get("cost") or 0.0
        if r.get("provider"):
            ep.providers.append(r["provider"])
        m = ((r.get("choices") or [{}])[0].get("message")) or {}
        tcs = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content") or "",
                     **({"tool_calls": tcs} if tcs else {})})
        if not tcs:
            if ep.n_nudges < 2:
                ep.n_nudges += 1
                msgs.append({"role": "user", "content": NUDGE})
                continue
            ep.answer_raw = DOTTED.findall(m.get("content") or "")[:5]
            ep.finish = "text_fallback"
            break
        submitted = None
        for tc in tcs:
            fn = tc.get("function") or {}
            name, args = fn.get("name", ""), _args(fn.get("arguments"))
            if name == "submit_answer":
                locs = args.get("locations") or []
                submitted = [str(x) for x in (locs if isinstance(locs, list) else [locs])][:5]
                result = "Answer received."
            elif ep.n_tool_calls >= budget:
                result = BUDGET_DONE
            else:
                ep.n_tool_calls += 1
                counts[name] += 1
                result = env.run(name, args)
                for nid in ID_IN_LINE.findall(result):
                    if nid in env.ids and nid not in seen:
                        seen.append(nid)
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", ""), "name": name, "content": result})
        if submitted is not None:
            ep.answer_raw, ep.finish = submitted, "submit"
            break
    else:
        ep.finish = "max_turns"
    ep.answer = resolve_answer(env, ep.answer_raw)
    ep.n_outside_graph = sum(a not in env.ids for a in ep.answer)
    ep.seen, ep.tool_counts, ep.messages = seen, dict(counts), msgs
    return ep


def resolve_answer(env: GraphEnv, raw: list) -> list:
    """Map submitted names to node ids (exact id or unique short name). A full dotted id that is not a node
    of this graph is kept verbatim, so a model may still name code the graph lacks (e.g. a module or an
    async function it inferred from what it read). This is conservative for the graph comparison: missing
    nodes block discovery through the tools, not the answer itself. Ambiguous short names are dropped."""
    ans = []
    for a in raw:
        n, _ = env.resolve(a)
        if n is None:
            s = str(a).strip().strip("`'\"").removesuffix("()")
            n = s if DOTTED.fullmatch(s) else None
        if n and n not in ans:
            ans.append(n)
    return ans[:5]
