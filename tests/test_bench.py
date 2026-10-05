import numpy as np

from cgl.bench import (BM25, Resolver, gold_ranks, issue_identifiers, metrics_from_ranks,
                       personalized_pagerank, rrf, tokenize)


def test_tokenize_splits_identifiers():
    toks = tokenize("Fix jsonable_encoder in APIRouter.add_api_route")
    assert {"jsonable_encoder", "jsonable", "encoder", "apirouter", "api", "router", "add_api_route"} <= set(toks)
    assert "in" not in toks


def test_bm25_prefers_matching_doc():
    docs = [tokenize("def parse_header(value): ..."), tokenize("class Router: pass"), tokenize("def encode(x): ...")]
    s = BM25(docs).scores(tokenize("header parsing bug in parse_header"))
    assert int(np.argmax(s)) == 0


def test_issue_identifiers():
    text = "Calling `Response.iter_content()` with chunk_size fails in HTTPAdapter; see https://x.com"
    ids = issue_identifiers(text)
    assert "Response.iter_content" in ids and "chunk_size" in ids and "HTTPAdapter" in ids
    assert not any(i.startswith("http") and "://" in i for i in ids)
    assert "_encode_params" in issue_identifiers("the private helper _encode_params breaks")


def test_resolver_matches_harness_order():
    r = Resolver(["pkg.models.Response", "pkg.models.Response.iter_content", "pkg.api.get", "pkg.adapters.HTTPAdapter"])
    assert r.resolve("Response.iter_content") == "pkg.models.Response.iter_content"   # suffix
    assert r.resolve("httpadapter") == "pkg.adapters.HTTPAdapter"                     # case-insensitive
    assert r.resolve("iter_con") == "pkg.models.Response.iter_content"                # substring (harness)
    assert r.resolve("iter_con", strict=True) is None


def test_ppr_ranks_neighbours_above_far_nodes():
    nodes = ["a", "b", "c", "d", "e"]
    edges = [("a", "b"), ("b", "c"), ("c", "d")]
    x = personalized_pagerank(nodes, edges, {"a": 1.0})
    assert x[1] > x[3] > 0 and x[4] == 0


def test_rrf_and_metrics():
    fused = rrf([["x", "y", "z"], ["y", "x", "w"]])
    assert set(fused[:2]) == {"x", "y"}
    ranks = gold_ranks(["a", "b", "c"], ["c", "q"])
    assert ranks == [3, None]
    m = metrics_from_ranks(ranks)
    assert m["hit@1"] == 0 and m["hit@5"] == 1 and m["recall@5"] == 0.5 and abs(m["mrr"] - 1 / 3) < 1e-9
