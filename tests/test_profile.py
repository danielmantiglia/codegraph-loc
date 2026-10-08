import subprocess
from pathlib import Path

from cgl.mining import mine
from cgl.profile import detect_package, main, summarize


def _git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@example.com", *args],
                   check=True, capture_output=True)


def _write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


def _repo(tmp_path: Path) -> Path:
    root = tmp_path / "toy"
    root.mkdir()
    _git(root, "init", "-q")
    _write(root, "pkg/__init__.py", "")
    core = "X = 1\n\n\ndef f():\n    return 1\n\n\nasync def g():\n    return 2\n"
    _write(root, "pkg/core.py", core)
    _write(root, "tests/test_core.py", "def test_a():\n    assert True\n")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "Initial commit")
    for i, (old, new, title) in enumerate([("    return 2", "    return 3", "Fix g result"),
                                            ("X = 1", "X = 2", "Fix default X"),
                                            ("    return 1", "    return 4", "Fix f result")]):
        core = core.replace(old, new)
        _write(root, "pkg/core.py", core)
        _write(root, "tests/test_core.py", f"def test_a():\n    assert {i} >= 0\n")
        _git(root, "add", "-A")
        _git(root, "commit", "-q", "-m", title)
    return root


def test_profile_counts_node_types(tmp_path):
    root = _repo(tmp_path)
    assert detect_package(root) == "pkg"
    rows = mine(root, "pkg", "2000-01-01")
    s = summarize(rows)
    assert s["n_fixes"] == 3
    assert s["edit_locations"] == {"sync functions/methods": 1, "async functions/methods": 1,
                                   "classes (class body)": 0,
                                   "module-level code (incl. new top-level definitions)": 1}
    F = s["share_of_fixes"]
    assert abs(F["touching async code"][0] - 1 / 3) < 1e-3
    assert abs(F["touching module-level code"][0] - 1 / 3) < 1e-3
    assert abs(F["fully representable by a released-style graph"][0] - 1 / 3) < 1e-3


def test_profile_cli(tmp_path, capsys):
    root = _repo(tmp_path)
    out = tmp_path / "p.json"
    assert main([str(root), "--since", "2000-01-01", "--json", str(out)]) == 0
    text = capsys.readouterr().out
    assert "3 fixes (small code+test commits)" in text and out.exists()
