"""How many gold localization targets can a graph *represent*?

Compares a graph schema like the one released for the competition (sync functions,
methods and classes only; no module nodes; no async defs) with the full schema
produced by cgl (adds async defs and module nodes).
Input: results/mined_*.jsonl from 02_mine_commits.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

rng = np.random.default_rng(0)


def boot_ci(x, n=2000):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"),) * 3
    bs = rng.choice(x, (n, len(x))).mean(1)
    return x.mean(), *np.percentile(bs, [2.5, 97.5])


rows_all = []
out = {}
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "results").glob("mined_*.jsonl")):
    rows = [json.loads(l) for l in open(f)]
    for r in rows:
        mods = {Path(p).with_suffix("").as_posix().replace("/", ".").removeprefix("src.").removesuffix(".__init__")
                for p in r["gold_files"]}
        r["_module_nodes"] = [g for g in r["gold_functions"] if g in mods]
        vis = [g not in r["gold_async"] and g not in r["_module_nodes"] for g in r["gold_functions"]]
        r["all_visible_sync"] = all(vis)
        r["none_visible_sync"] = not any(vis)
        r["frac_visible_sync"] = float(np.mean(vis)) if vis else 1.0
    rows_all += rows
    name = f.stem.removeprefix("mined_")
    out[name] = {k: [round(v, 3) for v in boot_ci([r[k] for r in rows])]
                 for k in ("all_visible_sync", "none_visible_sync", "frac_visible_sync")}
    out[name]["n"] = len(rows)
out["ALL"] = {k: [round(v, 3) for v in boot_ci([r[k] for r in rows_all])]
              for k in ("all_visible_sync", "none_visible_sync", "frac_visible_sync")}
out["ALL"]["n"] = len(rows_all)
for k, v in out.items():
    print(f"{k:9s} n={v['n']:4d}  all gold visible (sync, no-module schema) = {v['all_visible_sync'][0]:.1%} "
          f"[{v['all_visible_sync'][1]:.1%}, {v['all_visible_sync'][2]:.1%}]   "
          f"no gold visible = {v['none_visible_sync'][0]:.1%} [{v['none_visible_sync'][1]:.1%}, {v['none_visible_sync'][2]:.1%}]")
Path("results/visibility_summary.json").write_text(json.dumps(out, indent=2))
