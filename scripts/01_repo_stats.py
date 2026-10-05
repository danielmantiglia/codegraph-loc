"""Graph statistics for the four competition repositories (public GitHub HEAD or a given commit).

Usage: python scripts/01_repo_stats.py --repos-dir ../repos [--out results/repo_stats.json]
"""
import argparse
import collections
import json
import subprocess
import time
from pathlib import Path

from cgl.graph import build_graph

REPOS = ["fastapi", "rich", "requests", "httpx"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repos-dir", default="../repos")
    ap.add_argument("--out", default="results/repo_stats.json")
    args = ap.parse_args()
    rows = []
    for r in REPOS:
        root = Path(args.repos_dir) / r
        sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        t0 = time.time()
        G = build_graph(root)
        dt = time.time() - t0
        kinds = collections.Counter(a["kind"] for _, a in G.nodes(data=True))
        asyncs = collections.Counter(a["kind"] for _, a in G.nodes(data=True) if a.get("is_async"))
        etypes = collections.Counter(a["type"] for *_, a in G.edges(data=True))
        res = collections.Counter(a.get("resolution") for *_, a in G.edges(data=True) if a["type"] == "calls")
        # restrict to library code (exclude tests/docs/scripts) for the async share
        lib = [a for n, a in G.nodes(data=True) if n.split(".")[0] == r and a["kind"] in ("function", "method")]
        lib_async = sum(a["is_async"] for a in lib)
        rows.append(dict(repo=r, commit=sha, seconds=round(dt, 2), nodes=G.number_of_nodes(),
                         edges=G.number_of_edges(), kinds=dict(kinds), async_by_kind=dict(asyncs),
                         edge_types=dict(etypes), call_resolution=dict(res),
                         lib_functions=len(lib), lib_async=lib_async))
        print(f"{r:9s} {sha[:10]} nodes={G.number_of_nodes():6d} edges={G.number_of_edges():6d} "
              f"kinds={dict(kinds)} async={dict(asyncs)} lib_async={lib_async}/{len(lib)} "
              f"edges={dict(etypes)} calls_by_rule={dict(res)} ({dt:.1f}s)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
