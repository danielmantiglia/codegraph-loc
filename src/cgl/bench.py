"""Building blocks for the localization benchmark: tokenization, BM25, identifier
extraction from issue text, the harness's node-name resolver, Personalized PageRank,
rank fusion and metrics.  Dependency-light (numpy; scipy only for PageRank).
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Iterable, Sequence

import numpy as np

# --------------------------------------------------------------------------- #
# Tokenization
# --------------------------------------------------------------------------- #
_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")
STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may might more most must my no not of on or our should so some such than that the their them then there
these they this those to was we were what when where which while who why will with would you your
def self cls return none true false import class pass else elif try except finally raise lambda yield
async await global nonlocal assert del break continue print args kwargs
""".split())


def split_identifier(tok: str) -> list[str]:
    out = []
    for part in tok.split("_"):
        out += _CAMEL.findall(part)
    return out


def tokenize(text: str) -> list[str]:
    """Identifier-aware tokens: the full identifier plus its snake/camel parts, lowercased."""
    toks = []
    for t in _IDENT.findall(text or ""):
        low = t.lower()
        parts = [p.lower() for p in split_identifier(t)]
        cand = [low] + (parts if len(parts) > 1 else [])
        toks += [c for c in cand if len(c) > 1 and c not in STOPWORDS]
    return toks


# --------------------------------------------------------------------------- #
# BM25 (Okapi)
# --------------------------------------------------------------------------- #
class BM25:
    def __init__(self, docs: Sequence[list[str]], k1: float = 1.2, b: float = 0.75):
        self.n = len(docs)
        self.k1, self.b = k1, b
        self.dl = np.array([len(d) for d in docs], float)
        self.avgdl = float(self.dl.mean()) if self.n else 0.0
        self.post: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, d in enumerate(docs):
            for t, c in Counter(d).items():
                self.post[t].append((i, c))
        self.idf = {t: math.log((self.n - len(p) + 0.5) / (len(p) + 0.5) + 1.0) for t, p in self.post.items()}

    def scores(self, query: Iterable[str]) -> np.ndarray:
        s = np.zeros(self.n)
        for t in set(query):  # unique query terms: boilerplate repetition does not dominate
            if t not in self.post:
                continue
            idf = self.idf[t]
            for i, tf in self.post[t]:
                s[i] += idf * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * self.dl[i] / self.avgdl))
        return s


# --------------------------------------------------------------------------- #
# Identifiers mentioned in an issue
# --------------------------------------------------------------------------- #
_CODE_SPAN = re.compile(r"`([^`\n]+)`")
_DOTTED = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)+\b")
_SNAKE = re.compile(r"(?<![A-Za-z0-9_])_*[a-z0-9]+(?:_[a-z0-9]+)+_*(?![A-Za-z0-9_])")
_CAMEL_WORD = re.compile(r"\b[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*\b|\b[A-Z]{2,}[a-z][A-Za-z0-9]*\b")


def issue_identifiers(text: str) -> list[str]:
    """Code-like identifiers in an issue, in order of first appearance (deduplicated)."""
    found: list[tuple[int, str]] = []
    for m in _CODE_SPAN.finditer(text or ""):
        for t in re.finditer(r"[A-Za-z_][A-Za-z0-9_.]*", m.group(1)):
            found.append((m.start() + t.start(), t.group().strip(".")))
    for rx in (_DOTTED, _SNAKE, _CAMEL_WORD):
        for m in rx.finditer(text or ""):
            found.append((m.start(), m.group()))
    seen, out = set(), []
    for _, t in sorted(found):
        if len(t) < 3 or t.lower() in STOPWORDS or t in seen:
            continue
        if t.startswith(("http", "www.")) or t.endswith((".com", ".org", ".io", ".md", ".txt", ".html")):
            continue
        seen.add(t)
        out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Node-name resolution
# --------------------------------------------------------------------------- #
class Resolver:
    """Replicates swegemma 0.2.7 ``graph_utils.resolve_node_name`` (the harness tool's
    behaviour) and offers a strict variant without the substring fallback."""

    def __init__(self, node_ids: Iterable[str]):
        self.ids = list(node_ids)
        self.idset = set(self.ids)
        self.by_last = defaultdict(list)
        for n in self.ids:
            self.by_last[n.rsplit(".", 1)[-1]].append(n)
        self.lower = [n.lower() for n in self.ids]

    def _suffix(self, q: str) -> list[str]:
        last = q.rsplit(".", 1)[-1]
        return [n for n in self.by_last.get(last, []) if n.endswith("." + q)]

    @staticmethod
    def _pick(c: list[str]) -> str:
        return min(c, key=lambda x: (len(x.split(".")), len(x)))

    def resolve(self, q: str, strict: bool = False) -> str | None:
        if q in self.idset:
            return q
        c = self._suffix(q)
        if c:
            return self._pick(c)
        lq = q.lower()
        c = [n for n, ln in zip(self.ids, self.lower) if ln == lq or ln.endswith("." + lq)]
        if c:
            return self._pick(c)
        if strict:
            return None
        c = [n for n, ln in zip(self.ids, self.lower) if lq in ln]
        return self._pick(c) if c else None


# --------------------------------------------------------------------------- #
# Graph propagation and fusion
# --------------------------------------------------------------------------- #
def personalized_pagerank(nodes: list[str], edges: Iterable[tuple[str, str]], seeds: dict[str, float],
                          alpha: float = 0.85, iters: int = 50) -> np.ndarray:
    """PPR on the undirected, unweighted simple graph (power iteration, scipy sparse)."""
    import scipy.sparse as sp

    idx = {n: i for i, n in enumerate(nodes)}
    pairs = {(idx[u], idx[v]) for u, v in edges if u in idx and v in idx and u != v}
    pairs |= {(v, u) for u, v in pairs}
    n = len(nodes)
    if pairs:
        r, c = zip(*pairs)
        A = sp.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, n))
    else:
        A = sp.csr_matrix((n, n))
    deg = np.asarray(A.sum(1)).ravel()
    p = np.zeros(n)
    for s, w in seeds.items():
        if s in idx:
            p[idx[s]] += w
    if p.sum() == 0:
        return np.zeros(n)
    p /= p.sum()
    inv = np.divide(1.0, deg, out=np.zeros(n), where=deg > 0)
    x = p.copy()
    for _ in range(iters):
        spread = A.T @ (x * inv)
        dangling = x[deg == 0].sum()
        x = alpha * (spread + dangling * p) + (1 - alpha) * p
    return x


def rrf(rankings: Sequence[Sequence[str]], k: int = 60) -> list[str]:
    """Reciprocal rank fusion of several rankings."""
    s: dict[str, float] = defaultdict(float)
    for rk in rankings:
        for r, n in enumerate(rk):
            s[n] += 1.0 / (k + r + 1)
    return sorted(s, key=lambda n: -s[n])


def rank_from_scores(ids: Sequence[str], scores: np.ndarray, top: int = 1000, min_score: float = 0.0) -> list[str]:
    order = np.argsort(-scores, kind="stable")
    return [ids[i] for i in order[:top] if scores[i] > min_score]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def gold_ranks(ranking: Sequence[str], gold: Iterable[str]) -> list[int | None]:
    pos = {n: i + 1 for i, n in enumerate(ranking)}
    return [pos.get(g) for g in gold]


def metrics_from_ranks(ranks: list[int | None], ks=(1, 5, 10)) -> dict:
    hit = {f"hit@{k}": float(any(r is not None and r <= k for r in ranks)) for k in ks}
    rec = {f"recall@{k}": (sum(r is not None and r <= k for r in ranks) / len(ranks)) if ranks else 0.0 for k in ks}
    found = [r for r in ranks if r is not None]
    return {**hit, **rec, "mrr": 1.0 / min(found) if found else 0.0}
