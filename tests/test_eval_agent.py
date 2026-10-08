import importlib.util
import json
from pathlib import Path

SPEC = importlib.util.spec_from_file_location("eval_agent", Path(__file__).parents[1] / "scripts" / "18_eval_agent.py")
EV = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EV)


def _rec(split, tid, cond, answer, gold, in_graph):
    return {"ok": True, "key": f"{split}|{tid}|{cond}", "split": split, "task_id": tid, "repo": tid.split("_")[0],
            "condition": cond, "answer": answer, "gold": gold, "gold_in_graph": in_graph, "gold_seen": [],
            "n_tool_calls": 3, "n_turns": 4, "tool_counts": {"search_code": 3}, "finish": "submit",
            "cost_usd": 0.001, "n_outside_graph": 0}


def test_hypotheses_and_strata(tmp_path):
    recs = []
    for i in range(30):   # held out: gold in both graphs, both conditions hit -> no difference
        recs += [_rec("swebl", f"a_{i}", c, ["m.f"], ["m.f"], ["m.f"]) for c in ("official", "cgl")]
    for i in range(30):   # public: gold (async) missing from official; only cgl hits
        recs += [_rec("public", f"h_{i}", "official", ["m.g"], ["m.h"], []),
                 _rec("public", f"h_{i}", "cgl", ["m.h"], ["m.h"], ["m.h"])]
    recs.append(dict(recs[-1], ok=False))  # a failed retry must not replace the successful record
    ans = tmp_path / "answers.jsonl"
    ans.write_text("".join(json.dumps(r) + "\n" for r in recs))
    pilot = tmp_path / "pilot.jsonl"
    pilot.write_text(json.dumps({"split": "public", "task_id": "h_0"}) + "\n")
    import sys
    sys.argv = ["x", "--answers", str(ans), "--exclude-tasks-from", str(pilot)]
    EV.main()
    H = json.loads((tmp_path / "summary.json").read_text())["hypotheses"]
    h5, h6, h7, h8 = (H[k] for k in sorted(H))
    assert h5["n"] == 29 and h5["est"][0] == 1.0 and h5["outcome"] == "confirmed"
    assert h6["n"] == 30 and h6["est"][0] == 0.0 and h6["outcome"] == "not confirmed"
    assert h7["n_missing"] == 29 and h7["n_present"] == 30 and h7["est"][0] == 1.0 and h7["outcome"] == "confirmed"
    assert h8["n"] == 30 and h8["outcome"] == "confirmed"
