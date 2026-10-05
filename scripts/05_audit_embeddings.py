"""Audit of the released node embeddings (256-d, one .npz per commit) and of the
`search_similar_code` tool output they drive.

Facts established from the harness source (swegemma 0.2.7, graph/retrieval_utils.py,
graph/embedding_utils.py): the query of `search_similar_code` is resolved to a *node name*
(exact, suffix, case-insensitive or substring match) and the tool returns the k nodes whose
stored vectors are most cosine-similar to that node's vector, each with its full source.
Free text or code snippets are never embedded.

Per commit this script measures:
  * anisotropy: mean cosine of random node pairs, raw and after mean-centering;
  * top-1 neighbour cosine; share of nodes whose nearest neighbour has cosine >= 0.99;
  * exact duplicate vectors and whether duplicates share identical source text;
  * whether embedding similarity tracks lexical similarity of the source (Spearman);
  * tool-output size: characters returned by search_similar_code(k=10) for every node.

numpy only; reads the data in place.
Usage: python scripts/05_audit_embeddings.py --data ~/Desktop/Kaggle/dati/competition --out results/emb
       [--max-seconds 160] [--summarize]
"""
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

TOK = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def rankdata(x):
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x), float)
    r[order] = np.arange(len(x))
    # average ties
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = rankdata(np.asarray(a)), rankdata(np.asarray(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def audit(npz: Path, graph: Path, rng) -> dict:
    z = np.load(npz)
    keys = list(z.files)
    X = np.stack([z[k] for k in keys]).astype(np.float32)
    texts = {n["id"]: n.get("text") or "" for n in json.load(open(graph))["nodes"]}
    T = [texts.get(k, "") for k in keys]
    N, D = X.shape
    norms = np.linalg.norm(X, axis=1)
    zero = norms == 0
    Xn = X / np.where(zero, 1, norms)[:, None]
    C = X - X.mean(0, keepdims=True)
    Cn = C / np.maximum(np.linalg.norm(C, axis=1), 1e-12)[:, None]

    i = rng.integers(0, N, 20000)
    j = rng.integers(0, N, 20000)
    keep = i != j
    i, j = i[keep], j[keep]
    cos_raw = np.einsum("ij,ij->i", Xn[i], Xn[j])
    cos_cen = np.einsum("ij,ij->i", Cn[i], Cn[j])

    # lexical similarity of the same pairs (token-set Jaccard)
    toks = [set(TOK.findall(t)) for t in T]
    jac = np.array([len(toks[a] & toks[b]) / max(1, len(toks[a] | toks[b])) for a, b in zip(i[:5000], j[:5000])])

    # nearest neighbours for every node (exclude self), raw vectors as used by the tool
    lens = np.array([len(t) for t in T])
    top1 = np.empty(N, np.float32)
    top1_cen = np.empty(N, np.float32)
    out_chars = np.empty(N, np.int64)
    for s in range(0, N, 512):
        e = min(N, s + 512)
        S = Xn[s:e] @ Xn.T
        S[np.arange(e - s), np.arange(s, e)] = -np.inf
        idx = np.argpartition(-S, 10, axis=1)[:, :10]
        top1[s:e] = S.max(1)
        out_chars[s:e] = lens[idx].sum(1)
        Sc = Cn[s:e] @ Cn.T
        Sc[np.arange(e - s), np.arange(s, e)] = -np.inf
        top1_cen[s:e] = Sc.max(1)

    # exact duplicate vectors
    _, inv, counts = np.unique(X.round(6), axis=0, return_inverse=True, return_counts=True)
    inv = inv.ravel()
    dup = counts[inv] > 1
    groups = defaultdict(list)
    for k, g in enumerate(inv):
        if counts[g] > 1:
            groups[g].append(k)
    same_text_groups = sum(len({T[k] for k in ks}) == 1 for ks in groups.values())

    q = lambda a, p: float(np.percentile(a, p))
    return {
        "npz": npz.name, "nodes": N, "dim": D, "zero_vectors": int(zero.sum()),
        "norm_mean": float(norms.mean()), "norm_std": float(norms.std()),
        "random_pair_cos_raw": [float(cos_raw.mean()), float(cos_raw.std())],
        "random_pair_cos_centered": [float(cos_cen.mean()), float(cos_cen.std())],
        "top1_cos_raw": {"median": q(top1, 50), "p10": q(top1, 10), "share_ge_0.99": float((top1 >= 0.99).mean())},
        "top1_cos_centered": {"median": q(top1_cen, 50), "p10": q(top1_cen, 10),
                              "share_ge_0.99": float((top1_cen >= 0.99).mean())},
        "dup_vector_share": float(dup.mean()), "dup_groups": len(groups),
        "dup_groups_identical_text": same_text_groups,
        "spearman_cos_vs_token_jaccard_raw": spearman(cos_raw[:5000], jac),
        "spearman_cos_vs_token_jaccard_centered": spearman(cos_cen[:5000], jac),
        "search_k10_chars": {"median": q(out_chars, 50), "p90": q(out_chars, 90), "max": int(out_chars.max()),
                             "share_gt_5k": float((out_chars > 5_000).mean()),
                             "share_gt_20k": float((out_chars > 20_000).mean()),
                             "share_gt_100k": float((out_chars > 100_000).mean())},
        "node_text_chars": {"median": q(lens, 50), "p99": q(lens, 99), "max": int(lens.max())},
    }


def summarize(out: Path):
    recs = [json.load(open(p)) for p in sorted((out / "commits").glob("*.json"))]
    by_repo = defaultdict(list)
    for r in recs:
        by_repo[r["npz"].split("_")[0]].append(r)

    def med(rs, f):
        return round(float(np.median([f(r) for r in rs])), 4)

    summ = {}
    for repo, rs in sorted(by_repo.items()) + [("ALL", recs)]:
        summ[repo] = {
            "commits": len(rs), "nodes_median": med(rs, lambda r: r["nodes"]),
            "random_pair_cos_raw": med(rs, lambda r: r["random_pair_cos_raw"][0]),
            "random_pair_cos_centered": med(rs, lambda r: r["random_pair_cos_centered"][0]),
            "top1_raw_median": med(rs, lambda r: r["top1_cos_raw"]["median"]),
            "top1_raw_share_ge_0.99": med(rs, lambda r: r["top1_cos_raw"]["share_ge_0.99"]),
            "top1_centered_median": med(rs, lambda r: r["top1_cos_centered"]["median"]),
            "dup_vector_share": med(rs, lambda r: r["dup_vector_share"]),
            "dup_groups_identical_text_share": med(rs, lambda r: r["dup_groups_identical_text"] / max(1, r["dup_groups"])),
            "spearman_vs_lexical_raw": med(rs, lambda r: r["spearman_cos_vs_token_jaccard_raw"]),
            "spearman_vs_lexical_centered": med(rs, lambda r: r["spearman_cos_vs_token_jaccard_centered"]),
            "search_k10_chars_median": med(rs, lambda r: r["search_k10_chars"]["median"]),
            "search_k10_share_gt_5k": med(rs, lambda r: r["search_k10_chars"]["share_gt_5k"]),
            "search_k10_share_gt_100k": med(rs, lambda r: r["search_k10_chars"]["share_gt_100k"]),
            "search_k10_max": max(r["search_k10_chars"]["max"] for r in rs),
        }
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="results/emb")
    ap.add_argument("--max-seconds", type=float, default=160)
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    data, out = Path(a.data).expanduser(), Path(a.out)
    (out / "commits").mkdir(parents=True, exist_ok=True)
    if a.summarize:
        summarize(out)
        return
    rng = np.random.default_rng(0)
    start = time.time()
    files = sorted((data / "embeddings").glob("*.npz"))
    todo = [f for f in files if not (out / "commits" / (f.stem + ".json")).exists()]
    print(f"{len(files)} files, {len(todo)} to do", flush=True)
    for f in todo:
        if time.time() - start > a.max_seconds:
            print("time budget reached; run again to continue", flush=True)
            return
        rec = audit(f, data / "graphs" / (f.stem + ".json"), rng)
        (out / "commits" / (f.stem + ".json")).write_text(json.dumps(rec))
        print(f"{f.stem[:30]} N={rec['nodes']} rand_cos={rec['random_pair_cos_raw'][0]:.3f} "
              f"top1_med={rec['top1_cos_raw']['median']:.4f} dup={rec['dup_vector_share']:.2f}", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
