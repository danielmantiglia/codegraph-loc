from pathlib import Path

from cgl.graph import build_graph, restrict, to_node_link

FIX = Path(__file__).parent / "fixture_repo"


def _edges(G, t):
    return {(u, v) for u, v, a in G.edges(data=True) if a["type"] == t}


def test_nodes_include_async_and_modules():
    G = build_graph(FIX)
    assert G.nodes["pkg.core"]["kind"] == "module"
    assert G.nodes["pkg.core.Client.send"]["is_async"] is True
    assert G.nodes["pkg.sub.util.async_helper"]["is_async"] is True
    assert G.nodes["pkg.core.Client.run.inner"]["kind"] == "function"
    assert G.nodes["tests.test_core.test_run"]["kind"] == "function"
    # property getter + setter merged into one node with both spans in its text
    assert "setter" in G.nodes["pkg.core.Base.value"]["text"]


def test_contains_imports_inherits():
    G = build_graph(FIX)
    assert ("pkg.core", "pkg.core.Client") in _edges(G, "contains")
    assert ("pkg.core.Client", "pkg.core.Client.send") in _edges(G, "contains")
    assert ("pkg.core.Client", "pkg.core.Base") in _edges(G, "inherits")
    imp = _edges(G, "imports")
    assert ("pkg.core", "pkg.sub.util") in imp
    assert ("pkg.core", "pkg.sub.util.async_helper") in imp
    assert ("pkg", "pkg.core.Client") in imp


def test_calls_resolution_rules():
    G = build_graph(FIX)
    calls = {(u, v): a["resolution"] for u, v, a in G.edges(data=True) if a["type"] == "calls"}
    assert calls[("pkg.core.Client.__init__", "pkg.sub.util.helper")] == "import"      # module alias
    assert calls[("pkg.core.Client.send", "pkg.sub.util.async_helper")] == "import"    # async callee
    assert calls[("pkg.core.Client.send", "pkg.core.Base.ping")] == "self"             # via MRO
    assert calls[("pkg.core.Client.run.inner", "pkg.core.Base.ping")] == "self"        # closure keeps self
    assert calls[("pkg.core.Client.run", "pkg.core.Base.ping")] == "super"
    assert calls[("pkg.core.Client.run", "pkg.core.Client.run.inner")] == "local"
    assert calls[("pkg.core.make", "pkg.core.Client")] == "module"
    assert calls[("pkg.core.make", "pkg.core.Client.run")] == "typed"                  # c = Client()
    assert calls[("pkg.sub.util.async_helper", "pkg.sub.util.helper")] == "module"
    assert calls[("tests.test_core.test_run", "pkg.core.Client")] == "import"          # re-export


def test_restrict_and_export():
    G = build_graph(FIX)
    H = restrict(G, drop_async=True, edge_types={"calls"})
    assert "pkg.core.Client.send" not in H
    assert all(a["type"] == "calls" for *_, a in H.edges(data=True))
    js = to_node_link(G)
    assert js["directed"] and js["multigraph"]
    assert {"id", "name", "text"} <= set(js["nodes"][0])
    assert {"source", "target", "type", "key"} <= set(js["edges"][0])


def test_package_module_scheme():
    from cgl.graph import module_name_for
    # src/ has no __init__.py -> dropped; tests/ has no __init__.py in the fixture -> bare stem
    assert module_name_for(FIX / "src/pkg/core.py", FIX, scheme="package") == ("pkg.core", False)
    assert module_name_for(FIX / "src/pkg/__init__.py", FIX, scheme="package") == ("pkg", True)
    assert module_name_for(FIX / "tests/test_core.py", FIX, scheme="package") == ("test_core", False)
    G = build_graph(FIX, module_scheme="package")
    assert "pkg.core.Client.send" in G and "test_core.test_run" in G


def test_flat_id_and_official_view():
    from cgl.graph import flat_id, official_view
    kinds = {"m": "module", "m.C": "class", "m.C.run": "method", "m.C.run.inner": "function",
             "m.f": "function", "m.f.K": "class", "m.f.K.meth": "method", "m.C.D": "class",
             "m.C.D.x": "method"}
    assert flat_id("m.C.run.inner", kinds.get, "m") == "m.C.inner"
    assert flat_id("m.f.K", kinds.get, "m") == "m.K"
    assert flat_id("m.f.K.meth", kinds.get, "m") == "m.K.meth"
    assert flat_id("m.C.run", kinds.get, "m") == "m.C.run"
    assert flat_id("m.C.D", kinds.get, "m") == "m.D"          # nested class -> module level
    assert flat_id("m.C.D.x", kinds.get, "m") == "m.D.x"
    H = official_view(build_graph(FIX))
    assert "pkg.core.Client.inner" in H and "pkg.core.Client.run.inner" not in H
