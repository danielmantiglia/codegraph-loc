"""Shared pipeline pieces for the localization experiments: repository snapshots,
gold labels in the competition id convention, graph views for ablations, rankers."""

from __future__ import annotations

import io
import subprocess
import tarfile
from pathlib import Path

from .bench import BM25, Resolver, issue_identifiers, personalized_pagerank, rank_from_scores, rrf, tokenize
from .graph import CodeGraphBuilder, flat_id, module_name_for, official_view
from .labels import def_table, is_test_path, locate_changes, parse_unified_diff

MODULE_KW = dict(scheme="official", strip_prefixes=("src", "docs_src"))
TOP = 1000
ABLATIONS = {
    "no_async": dict(drop_async=True),
    "no_module": dict(drop_module=True),
    "no_contains": dict(drop_types={"contains"}),
    "no_imports": dict(drop_types={"imports"}),
    "no_inherits": dict(drop_types={"inherits"}),
    "calls_only": dict(drop_types={"contains", "imports", "inherits"}),
    "official_schema": dict(drop_types={"contains", "imports", "inherits"}, drop_async=True, drop_module=True),
}


def extract(repo_path: Path, commit: str, dest: str) -> None:
    """Write the tree of ``commit`` into ``dest`` (regular files only; symlinks skipped)."""
    data = subprocess.run(["git", "-C", str(repo_path), "archive", commit], check=True, capture_output=True).stdout
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        members = [m for m in tf.getmembers() if m.isfile() or m.isdir()]
        try:
            tf.extractall(dest, members=members, filter="data")
        except TypeError:
            tf.extractall(dest, members=members)


def gold_for(patch: str, root: Path) -> dict[str, str]:
    """Gold node id (competition convention) -> file, for the non-test, pre-existing .py files of a patch."""
    out = {}
    for fc in parse_unified_diff(patch):
        if fc.is_new_file or not fc.path.endswith(".py") or is_test_path(fc.path):
            continue
        f = root / fc.path
        if not f.exists():
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        mid, _ = module_name_for(f, root, **MODULE_KW)
        try:
            loc = locate_changes(fc, src, mid)
            table = def_table(src, mid)
        except SyntaxError:
            continue
        kind_of = lambda i, mid=mid, table=table: "module" if i == mid else table.get(i, {}).get("kind")
        for g in loc["functions"]:
            out[flat_id(g, kind_of, mid)] = fc.path
    return out


def build_cgl(root: Path):
    return official_view(CodeGraphBuilder(root, module_scheme="official",
                                          strip_prefixes=MODULE_KW["strip_prefixes"]).build())


def cgl_view(G, drop_types=(), drop_async=False, drop_module=False):
    nodes = [n for n, a in G.nodes(data=True)
             if not (drop_async and a.get("is_async")) and not (drop_module and a.get("kind") == "module")]
    keep = set(nodes)
    edges = [(u, v) for u, v, a in G.edges(data=True)
             if a.get("type") not in drop_types and u in keep and v in keep]
    return nodes, edges


def ppr_rank(nodes, edges, seeds):
    if not seeds:
        return []
    return rank_from_scores(nodes, personalized_pagerank(nodes, edges, seeds), TOP)


def seeds_then(seeds, ranking):
    seen, out = set(), []
    for n in list(seeds) + list(ranking):
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out[:TOP]


def hybrid_seeds(seeds, ranking, scores_by_id):
    w = {s: 1.0 for s in seeds}
    top = ranking[:10]
    tot = sum(scores_by_id[n] for n in top) or 1.0
    for n in top:
        w[n] = w.get(n, 0.0) + scores_by_id[n] / tot * max(1, len(seeds))
    return w


def is_test_node(node_id: str, file: str | None) -> bool:
    if file:
        return is_test_path(file)
    parts = node_id.split(".")
    return any(p in ("tests", "test", "testing", "conftest") or p.startswith("test_") for p in parts[:-1]) \
        or parts[0] in ("tests", "test")


def cgl_rankings(G, query: str) -> dict[str, list[str]]:
    """All cgl-only methods for one query (no released data needed).

    ``*_official_schema`` variants restrict the cgl graph to what the released graphs
    contain (sync functions/methods/classes, ``calls`` edges only), so the released
    graph's coverage can be emulated on any repository and commit.
    """
    nodes_all = list(G.nodes)
    text = {n: G.nodes[n].get("text") or "" for n in nodes_all}
    qtok = tokenize(query)
    idents = issue_identifiers(query)
    out = {}
    views = {"full": cgl_view(G), **{k: cgl_view(G, **v) for k, v in ABLATIONS.items()}}
    bm = BM25([tokenize(n + " " + text[n]) for n in nodes_all])
    s = bm.scores(qtok)
    sc = dict(zip(nodes_all, s))
    r_bm = rank_from_scores(nodes_all, s, TOP)
    res_full = Resolver(nodes_all)
    seeds_full = list(dict.fromkeys(r for r in (res_full.resolve(i) for i in idents) if r))
    out["bm25_cgl"] = r_bm
    out["ident+ppr_cgl"] = seeds_then(seeds_full, ppr_rank(*views["full"], {x: 1.0 for x in seeds_full}))
    for name, (vn, ve) in views.items():
        keep = set(vn)
        bm_rank = [n for n in r_bm if n in keep]
        res = Resolver(vn) if name != "full" else res_full
        seeds = list(dict.fromkeys(r for r in (res.resolve(i) for i in idents) if r))
        key = "rrf_cgl" if name == "full" else f"rrf_cgl-{name}"
        out[key] = rrf([bm_rank, ppr_rank(vn, ve, hybrid_seeds(seeds, bm_rank, sc))])[:TOP]
        if name == "official_schema":
            out["bm25_official_schema"] = bm_rank
            out["ident+ppr_official_schema"] = seeds_then(seeds, ppr_rank(vn, ve, {x: 1.0 for x in seeds}))
    return out
