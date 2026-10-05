"""Localization labels: which graph nodes does a reference patch modify?

Given the *pre-fix* source of each file touched by a unified diff, every changed
old line (and every insertion point) is mapped to the innermost definition that
contains it.  This yields gold labels at three granularities:

* ``files``     – paths of modified, pre-existing ``.py`` files
* ``defs``      – innermost function/method/class containing each change
* ``functions`` – innermost *function or method* (class-body / module-level
                  changes fall back to the class or module node)

The same procedure is used for every benchmark split, so labels for the Kaggle
tasks, SWE-bench Lite and the scientific-library split are directly comparable.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FileChange:
    old_path: str | None
    new_path: str | None
    changed_old_lines: set[int] = field(default_factory=set)        # removed / modified
    insertions: dict[int, list[str]] = field(default_factory=dict)  # lines added after old line k

    @property
    def insert_after(self) -> set[int]:
        return set(self.insertions)

    @property
    def is_new_file(self) -> bool:
        return self.old_path is None

    @property
    def path(self) -> str:
        return self.old_path or self.new_path  # type: ignore[return-value]


def parse_unified_diff(diff: str) -> list[FileChange]:
    """Minimal, dependency-free unified-diff parser (git style).

    Hunk line counts are tracked so that content lines starting with ``---``
    or ``+++`` are never mistaken for file headers.
    """
    files: list[FileChange] = []
    cur: FileChange | None = None
    old_ln = 0
    old_left = new_left = 0
    for line in diff.splitlines():
        in_hunk = old_left > 0 or new_left > 0
        if not in_hunk:
            if line.startswith("--- "):
                p = line[4:].split("\t")[0].strip()
                cur = FileChange(old_path=None if p == "/dev/null" else re.sub(r"^a/", "", p), new_path=None)
                files.append(cur)
                continue
            if line.startswith("+++ ") and cur is not None:
                p = line[4:].split("\t")[0].strip()
                cur.new_path = None if p == "/dev/null" else re.sub(r"^b/", "", p)
                continue
            m = _HUNK.match(line)
            if m and cur is not None:
                old_ln = int(m.group(1))
                old_left = int(m.group(2)) if m.group(2) is not None else 1
                new_left = int(m.group(4)) if m.group(4) is not None else 1
                if old_left == 0:  # insertion-only hunk: "-k,0" means after line k
                    old_ln += 1
            continue
        if line.startswith("\\"):
            continue
        tag, rest = (line[0], line[1:]) if line else (" ", "")
        if tag == " ":
            old_ln += 1; old_left -= 1; new_left -= 1
        elif tag == "-":
            cur.changed_old_lines.add(old_ln)
            old_ln += 1; old_left -= 1
        elif tag == "+":
            cur.insertions.setdefault(old_ln - 1, []).append(rest)
            new_left -= 1
    return files


def _def_spans(source: str, module_id: str) -> list[tuple[str, str, int, int, int]]:
    """(id, kind, start, end, depth, col, is_async) for every class/def in ``source``."""
    tree = ast.parse(source)
    out = []

    def walk(body, prefix, depth, in_class):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([node.lineno] + [d.lineno for d in node.decorator_list])
                did = f"{prefix}.{node.name}"
                if isinstance(node, ast.ClassDef):
                    kind = "class"
                else:
                    kind = "method" if in_class else "function"
                out.append((did, kind, start, node.end_lineno, depth, node.col_offset,
                            isinstance(node, ast.AsyncFunctionDef)))
                walk(node.body, did, depth + 1, isinstance(node, ast.ClassDef))
            else:
                # defs nested in if/try/with blocks at the same logical level
                for fld in ("body", "orelse", "finalbody", "handlers"):
                    sub = getattr(node, fld, None)
                    if isinstance(sub, list):
                        walk(sub, prefix, depth, in_class)

    walk(tree.body, module_id, 1, False)
    return out


def locate_changes(fc: FileChange, old_source: str, module_id: str) -> dict[str, set[str]]:
    """Map the changes of one file to node ids (``defs``, ``functions``).

    * a removed/modified line belongs to the innermost def whose span contains it
      (blank old lines are ignored);
    * lines inserted after old line ``k`` belong to the innermost def ``D`` that
      contains ``k`` and either also contains ``k+1`` or is indented less than
      the inserted code (this catches code appended at the end of a body);
      insertions made only of blank lines are ignored.
    """
    spans = _def_spans(old_source, module_id)
    old_lines = old_source.splitlines()
    incidental = _incidental_module_lines(old_source)
    defs, funcs = set(), set()
    module_points = {"incidental": 0, "substantive": 0}

    def pick(cands, functions_only):
        cands = [c for c in cands if not (functions_only and c[1] == "class")]
        return max(cands, key=lambda c: c[4])[0] if cands else None

    def add(cands, is_incidental=False):
        d = pick(cands, False)
        f = pick(cands, True)
        if d is None:  # module scope
            module_points["incidental" if is_incidental else "substantive"] += 1
            if is_incidental:
                return
        defs.add(d or module_id)
        funcs.add(f or d or module_id)

    for ln in fc.changed_old_lines:
        if 1 <= ln <= len(old_lines) and not old_lines[ln - 1].strip():
            continue
        add([c for c in spans if c[2] <= ln <= c[3]], ln in incidental)

    for k, added in fc.insertions.items():
        body = [a for a in added if a.strip()]
        if not body:
            continue
        indent = len(body[0]) - len(body[0].lstrip())
        cands = [c for c in spans
                 if c[2] <= k <= c[3] and (k + 1 <= c[3] or indent > c[5])]
        inc = ({k, k + 1} <= incidental) or all(_IMPORT_LINE.match(a) for a in body)
        add(cands, inc)
    return {"defs": defs, "functions": funcs, "module_points": module_points}


_IMPORT_LINE = re.compile(r"^\s*(import\s|from\s\S+\s+import\b|\)\s*$|#)")


def _incidental_module_lines(source: str) -> set[int]:
    """Lines of top-level import statements, ``__all__`` and ``if TYPE_CHECKING`` import blocks.

    Edits there (e.g. importing a helper used by the actual fix) are not
    counted as module-level localization targets.
    """
    lines: set[int] = set()
    tree = ast.parse(source)

    def only_imports(body):
        return all(isinstance(b, (ast.Import, ast.ImportFrom, ast.Pass)) for b in body)

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            lines.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgts = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(t, ast.Name) and t.id == "__all__" for t in tgts):
                lines.update(range(node.lineno, node.end_lineno + 1))
        elif isinstance(node, (ast.If, ast.Try)) and only_imports(node.body):
            lines.update(range(node.lineno, node.end_lineno + 1))
    return lines


def def_table(source: str, module_id: str) -> dict[str, dict]:
    """id -> {kind, start, end, is_async} for every class/def in ``source``."""
    return {c[0]: {"kind": c[1], "start": c[2], "end": c[3], "is_async": c[6]}
            for c in _def_spans(source, module_id)}


def is_test_path(path: str) -> bool:
    parts = path.replace("\\", "/").split("/")
    name = parts[-1]
    return ("tests" in parts or "test" in parts or "testing" in parts
            or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")
