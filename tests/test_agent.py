import json
from pathlib import Path

from cgl.agent import BUDGET_DONE, GraphEnv, run_episode
from cgl.pipeline import build_cgl, cgl_view

FIXTURE = Path(__file__).parent / "fixture_repo"
CALLS_ONLY = {"contains", "imports", "inherits"}


def _envs():
    G = build_cgl(FIXTURE)
    text = {n: G.nodes[n].get("text") or "" for n in G.nodes}
    files = {n: G.nodes[n].get("file") for n in G.nodes}
    full = GraphEnv(*cgl_view(G, drop_types=CALLS_ONLY), text, files)
    released = GraphEnv(*cgl_view(G, drop_types=CALLS_ONLY, drop_async=True, drop_module=True), text, files)
    return full, released


def test_tools_respect_the_graph():
    full, released = _envs()
    assert "pkg.core.Client.send" in full.ids and "pkg.core.Client.send" not in released.ids  # async
    assert "pkg.core" in full.ids and "pkg.core" not in released.ids                          # module node
    assert not any(n.startswith("tests.") for n in full.ids)                                   # test code hidden
    hits = [l.split(" — ")[0] for l in full.search_code("async helper request").splitlines()]
    assert "pkg.sub.util.async_helper" in hits and "pkg.sub.util.async_helper" not in released.ids
    assert full.search_code("async_helper").splitlines()[0].startswith("pkg.sub.util.async_helper — async def")
    nb = full.get_neighbors("send")                       # unique short name resolves
    assert "pkg.core.Base.ping" in nb and "pkg.sub.util.async_helper" in nb
    assert "No node named" in released.get_neighbors("send")
    assert full.read_code("pkg.core.make").startswith("Node: pkg.core.make\ndef make()")
    assert "ambiguous" in full.read_code("value") or full.resolve("value")[0] == "pkg.core.Base.value"


def _scripted(replies):
    """A fake model that returns the scripted assistant messages in order."""
    it = iter(replies)

    def chat(messages, tools):
        assert messages[0]["role"] == "system" and tools
        m = next(it)
        return {"choices": [{"message": m}], "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001},
                "provider": "Mock"}
    return chat


def _call(name, i, **args):
    return {"id": f"c{i}", "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


def test_episode_submit_and_accounting():
    full, _ = _envs()
    chat = _scripted([
        {"content": "", "tool_calls": [_call("search_code", 1, query="async helper")]},
        {"content": "", "tool_calls": [_call("read_code", 2, node="pkg.sub.util.async_helper"),
                                        _call("get_neighbors", 3, node="async_helper")]},
        {"content": "", "tool_calls": [_call("submit_answer", 4, locations=["async_helper", "pkg.core.Client.send",
                                                                            "not.a.node"])]},
    ])
    ep = run_episode(full, "The async helper drops the request", chat)
    assert ep.finish == "submit" and ep.n_turns == 3 and ep.n_tool_calls == 3
    assert ep.answer == ["pkg.sub.util.async_helper", "pkg.core.Client.send", "not.a.node"]
    assert ep.n_outside_graph == 1
    assert ep.tool_counts == {"search_code": 1, "read_code": 1, "get_neighbors": 1}
    assert "pkg.sub.util.async_helper" in ep.seen and abs(ep.cost_usd - 0.003) < 1e-9
    assert [m["role"] for m in ep.messages].count("tool") == 4


def test_budget_and_text_fallback():
    full, _ = _envs()
    chat = _scripted([{"content": "", "tool_calls": [_call("search_code", i, query="ping") for i in range(3)]},
                      {"content": "I think it is pkg.core.Base.ping"},
                      {"content": "still thinking"},
                      {"content": "Final: pkg.core.Base.ping and pkg.core.make"}])
    ep = run_episode(full, "ping is wrong", chat, budget=2)
    assert ep.n_tool_calls == 2 and any(m.get("content") == BUDGET_DONE for m in ep.messages)
    assert ep.finish == "text_fallback" and ep.n_nudges == 2
    assert ep.answer == ["pkg.core.Base.ping", "pkg.core.make"]


def test_answers_outside_the_graph_are_kept():
    full, released = _envs()
    from cgl.agent import resolve_answer
    # the released-style graph has no async or module nodes, but a full id the model names still counts
    assert resolve_answer(released, ["pkg.core.Client.send", "pkg.core", "make", "`pkg.core.make()`"]) == \
        ["pkg.core.Client.send", "pkg.core", "pkg.core.make"]
    assert resolve_answer(full, ["send", "nonexistent_short_name"]) == ["pkg.core.Client.send"]
