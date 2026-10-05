"""What do the released 256-d node embeddings encode?

Per commit: effective dimensionality (PCA on centered vectors), share of isolated nodes in the
released call graph, and Spearman correlations between pairwise cosine similarity and
(a) node-name similarity (character 3-gram Jaccard), (b) source-length similarity,
(c) degree similarity in the released call graph.

numpy only; reads the data in place.
Usage: python scripts/06_embedding_geometry.py --data ~/Desktop/Kaggle/dati/competition --out results/emb_geometry.json
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def grams(s, n=3):
    s = s.lower()
    return {s[i:i + n] for i in range(max(1, len(s) - n + 1))}


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def one(npz: Path, graph: Path, rng) -> dict:
    z = np.load(npz)
    keys = z.files
    X = np.stack([z[k] for k in keys]).astype(np.float64)
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    G = json.load(open(graph))
    deg = Counter()
    for e in G.get("edges", G.get("links", [])):
        deg[e["source"]] += 1
        deg[e["target"]] += 1
    d = np.array([deg[k] for k in keys])
    txt = {n["id"]: n.get("text") or "" for n in G["nodes"]}
    L = np.array([len(txt.get(k, "")) for k in keys])
    names = [k.split(".")[-1] for k in keys]
    i = rng.integers(0, len(keys), 6000)
    j = rng.integers(0, len(keys), 6000)
    m = i != j
    i, j = i[m], j[m]
    cos = np.einsum("ij,ij->i", Xn[i], Xn[j])
    namesim = np.array([len(grams(names[a]) & grams(names[b])) / max(1, len(grams(names[a]) | grams(names[b])))
                        for a, b in zip(i, j)])
    C = X - X.mean(0)
    ev = np.linalg.svd(C, compute_uv=False) ** 2
    ev = ev / ev.sum()
    return {
        "npz": npz.name, "nodes": len(keys), "isolated_share": float((d == 0).mean()),
        "pc1_var_centered": float(ev[0]), "pc1_5_var_centered": float(ev[:5].sum()),
        "dims_for_90pct_var": int(np.searchsorted(np.cumsum(ev), 0.9)) + 1,
        "spearman_cos_name": spearman(cos, namesim),
        "spearman_cos_length": spearman(cos, -np.abs(np.log1p(L[i]) - np.log1p(L[j]))),
        "spearman_cos_degree": spearman(cos, -np.abs(np.log1p(d[i]) - np.log1p(d[j]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/emb_geometry.json")
    a = ap.parse_args()
    data = Path(a.data).expanduser()
    rng = np.random.default_rng(0)
    recs = [one(f, data / "graphs" / (f.stem + ".json"), rng) for f in sorted((data / "embeddings").glob("*.npz"))]
    by = defaultdict(list)
    for r in recs:
        by[r["npz"].split("_")[0]].append(r)
    by["ALL"] = recs
    summ = {k: {"commits": len(v), **{m: round(float(np.median([r[m] for r in v])), 4)
                                      for m in recs[0] if m != "npz"}} for k, v in by.items()}
    Path(a.out).write_text(json.dumps({"summary_median_per_repo": summ, "per_commit": recs}, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
