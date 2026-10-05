"""Static code-graph builder for Python repositories.

Builds a directed multigraph whose nodes are modules, classes, functions and
methods (sync *and* async) and whose edges are typed:

* ``contains`` – module → top-level def, class → method, def → nested def
* ``imports``  – module → internal module or symbol it imports
* ``inherits`` – class → internal base class
* ``calls``    – def (or module/class body) → internal def or class it calls

Every ``calls``/``inherits`` edge carries a ``resolution`` attribute naming the
rule that resolved it, so that ablations can drop heuristic edges.

The JSON export (:func:`to_node_link`) uses the same schema as the graphs
released with the Kaggle "Gemma 4 Developer Agent" competition (``id``,
``name``, ``text`` on nodes; ``source``, ``target``, ``type``, ``key`` on
edges), so a generated graph can be used as a drop-in replacement.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import networkx as nx

NODE_KINDS = ("module", "class", "function", "method")
DEF_KINDS = ("class", "function", "method")
EDGE_TYPES = ("contains", "imports", "inherits", "calls")

DEFAULT_EXCLUDE_DIRS = frozenset(
    {".git", ".hg", ".venv", "venv", "env", "build", "dist", "__pycache__",
     "node_modules", ".tox", ".nox", ".eggs", "site-packages", ".mypy_cache"}
)
_BUILTINS = frozenset(dir(builtins))
# attribute names that are methods of builtin types: never linked by the unique-name heuristic
_BUILTIN_METHODS = frozenset(
    n for t in (object, list, dict, set, frozenset, str, bytes, bytearray, int, float, tuple, type)
    for n in dir(t)
) | {"close", "read", "write", "flush", "seek", "tell", "send", "throw", "get", "items", "keys", "values"}


# --------------------------------------------------------------------------- #
# Data structures
# --------------------------------------------------------------------------- #
@dataclass
class Def:
    """One definition (module, class, function or method)."""

    id: str
    name: str
    kind: str
    module: str
    file: str
    spans: list[tuple[int, int]]          # 1-indexed, inclusive
    is_async: bool = False
    parent: str | None = None
    text: str = ""
    decorators: list[str] = field(default_factory=list)

    @property
    def start_line(self) -> int:
        return min(s for s, _ in self.spans)

    @property
    def end_line(self) -> int:
        return max(e for _, e in self.spans)


@dataclass
class ModuleInfo:
    id: str
    file: str
    is_package: bool
    tree: ast.Module
    lines: list[str]
    imports: dict[str, str] = field(default_factory=dict)   # alias -> qualified name
    star_imports: list[str] = field(default_factory=list)
    import_targets: list[str] = field(default_factory=list)  # qualified names imported


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def module_name_for(path: Path, root: Path, strip_prefixes: Iterable[str] = ("src",),
                    scheme: str = "path") -> tuple[str, bool]:
    """Dotted module name for ``path`` relative to ``root``.

    ``scheme="path"``: full relative path, ``src/`` stripped
    (``src/requests/models.py`` → ``requests.models``).

    ``scheme="package"``: Python import semantics — walk up while the parent
    directory contains ``__init__.py`` (``docs_src/tutorial/app.py`` with a
    package ``tutorial`` but no ``docs_src/__init__.py`` → ``tutorial.app``).
    This is the scheme used by the competition graphs.
    """
    if scheme == "official":  # package import path inside packages, else repo path (see flat_id)
        in_pkg = Path(path).name == "__init__.py" or (Path(path).parent / "__init__.py").exists()
        return module_name_for(path, root, strip_prefixes, "package" if in_pkg else "path")
    if scheme == "package":
        path = Path(path)
        is_package = path.name == "__init__.py"
        parts = [] if is_package else [path.stem]
        d = path.parent
        root = Path(root)
        while d != root and d != d.parent and (d / "__init__.py").exists():
            parts.insert(0, d.name)
            d = d.parent
        if not parts:  # top-level __init__.py outside any package
            parts = [path.parent.name]
        return ".".join(parts), is_package
    rel = path.relative_to(root).with_suffix("")
    parts = list(rel.parts)
    if parts and parts[0] in set(strip_prefixes) and len(parts) > 1:
        parts = parts[1:]
    is_package = parts[-1] == "__init__"
    if is_package:
        parts = parts[:-1]
    return ".".join(parts) if parts else "__init__", is_package


def iter_python_files(root: Path, exclude_dirs=DEFAULT_EXCLUDE_DIRS, include: Iterable[str] | None = None):
    include = [Path(p) for p in include] if include else None
    for p in sorted(root.rglob("*.py")):
        rel = p.relative_to(root)
        if any(part in exclude_dirs for part in rel.parts[:-1]):
            continue
        if include is not None and not any(rel.parts[: len(i.parts)] == i.parts for i in include):
            continue
        yield p


def _dotted(expr: ast.expr) -> str | None:
    """``a.b.c`` for Name/Attribute chains, else ``None``."""
    parts = []
    while isinstance(expr, ast.Attribute):
        parts.append(expr.attr)
        expr = expr.value
    if isinstance(expr, ast.Name):
        parts.append(expr.id)
        return ".".join(reversed(parts))
    return None


def _node_start(node: ast.AST) -> int:
    decos = getattr(node, "decorator_list", None) or []
    return min([node.lineno] + [d.lineno for d in decos])


def _resolve_relative(module: str, is_package: bool, level: int, target: str | None) -> str:
    """Absolute module for ``from <level dots><target> import ...``."""
    if level == 0:
        return target or ""
    base = module.split(".")
    if not is_package:
        base = base[:-1]
    if level > 1:
        base = base[: len(base) - (level - 1)] if level - 1 <= len(base) else []
    if target:
        base = base + target.split(".")
    return ".".join(base)


# --------------------------------------------------------------------------- #
# Pass 1: definitions and import tables
# --------------------------------------------------------------------------- #
class _DefCollector(ast.NodeVisitor):
    def __init__(self, mod: ModuleInfo, defs: dict[str, Def]):
        self.mod = mod
        self.defs = defs
        self.stack: list[Def] = []
        # raw bases for inherits resolution: class id -> list of dotted names
        self.bases: dict[str, list[str]] = {}

    # -- definitions --------------------------------------------------------
    def _add(self, node, kind: str, is_async: bool = False):
        parent = self.stack[-1] if self.stack else None
        prefix = parent.id if parent else self.mod.id
        did = f"{prefix}.{node.name}"
        span = (_node_start(node), node.end_lineno)
        text = "\n".join(self.mod.lines[span[0] - 1: span[1]])
        decos = [_dotted(d.func if isinstance(d, ast.Call) else d) or "" for d in node.decorator_list]
        if did in self.defs:  # property setter, overload, conditional re-definition
            d = self.defs[did]
            d.spans.append(span)
            d.text = d.text + "\n\n" + text
            d.is_async = d.is_async or is_async
        else:
            d = Def(id=did, name=node.name, kind=kind, module=self.mod.id, file=self.mod.file,
                    spans=[span], is_async=is_async, parent=parent.id if parent else self.mod.id,
                    text=text, decorators=decos)
            self.defs[did] = d
        return d

    def visit_ClassDef(self, node: ast.ClassDef):
        d = self._add(node, "class")
        self.bases.setdefault(d.id, []).extend(b for b in (_dotted(x) for x in node.bases) if b)
        self.stack.append(d)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    def _visit_func(self, node, is_async: bool):
        parent = self.stack[-1] if self.stack else None
        kind = "method" if parent is not None and parent.kind == "class" else "function"
        d = self._add(node, kind, is_async)
        self.stack.append(d)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop()

    def visit_FunctionDef(self, node):
        self._visit_func(node, False)

    def visit_AsyncFunctionDef(self, node):
        self._visit_func(node, True)

    # -- imports (module scope and nested; nested imports also bind names) ---
    def visit_Import(self, node: ast.Import):
        for a in node.names:
            if a.asname:
                self.mod.imports[a.asname] = a.name
            else:
                top = a.name.split(".")[0]
                self.mod.imports.setdefault(top, top)
            self.mod.import_targets.append(a.name)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        base = _resolve_relative(self.mod.id, self.mod.is_package, node.level, node.module)
        for a in node.names:
            if a.name == "*":
                self.mod.star_imports.append(base)
                self.mod.import_targets.append(base)
                continue
            q = f"{base}.{a.name}" if base else a.name
            self.mod.imports[a.asname or a.name] = q
            self.mod.import_targets.append(q)


# --------------------------------------------------------------------------- #
# Graph builder
# --------------------------------------------------------------------------- #
class CodeGraphBuilder:
    """Build a :class:`networkx.MultiDiGraph` for a repository checkout."""

    def __init__(self, root: str | Path, *, include: Iterable[str] | None = None,
                 exclude_dirs=DEFAULT_EXCLUDE_DIRS, strip_prefixes=("src",), module_scheme: str = "path",
                 unique_name_fallback: bool = True):
        self.root = Path(root)
        self.include = include
        self.exclude_dirs = exclude_dirs
        self.strip_prefixes = strip_prefixes
        self.module_scheme = module_scheme
        self.unique_name_fallback = unique_name_fallback
        self.modules: dict[str, ModuleInfo] = {}
        self.defs: dict[str, Def] = {}
        self.bases: dict[str, list[str]] = {}
        self.parse_errors: list[str] = []

    # -- pass 1 -------------------------------------------------------------
    def _collect(self):
        for path in iter_python_files(self.root, self.exclude_dirs, self.include):
            rel = str(path.relative_to(self.root))
            mid, is_pkg = module_name_for(path, self.root, self.strip_prefixes, self.module_scheme)
            try:
                src = path.read_text(encoding="utf-8", errors="replace")
                tree = ast.parse(src)
            except (SyntaxError, ValueError) as e:  # pragma: no cover - depends on repo
                self.parse_errors.append(f"{rel}: {e}")
                continue
            lines = src.splitlines()
            mod = ModuleInfo(id=mid, file=rel, is_package=is_pkg, tree=tree, lines=lines)
            if mid in self.modules:  # e.g. src/x.py and x.py – keep the first
                self.parse_errors.append(f"{rel}: duplicate module id {mid}")
                continue
            self.modules[mid] = mod
            coll = _DefCollector(mod, self.defs)
            for stmt in tree.body:
                coll.visit(stmt)
            self.bases.update(coll.bases)
            # module node: text = lines not covered by top-level defs
            covered = set()
            for stmt in tree.body:
                if isinstance(stmt, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    covered.update(range(_node_start(stmt), stmt.end_lineno + 1))
            mtext = "\n".join(l for i, l in enumerate(lines, 1) if i not in covered)
            self.defs[mid] = Def(id=mid, name=mid.split(".")[-1], kind="module", module=mid,
                                 file=rel, spans=[(1, max(1, len(lines)))], text=mtext)

        # indexes
        self._by_name: dict[str, list[str]] = {}
        for d in self.defs.values():
            if d.kind in ("function", "method", "class"):
                self._by_name.setdefault(d.name, []).append(d.id)
        self._children: dict[str, dict[str, str]] = {}
        for d in self.defs.values():
            if d.parent and d.kind != "module":
                self._children.setdefault(d.parent, {})[d.name] = d.id

    # -- name resolution ----------------------------------------------------
    def resolve_qualified(self, q: str, depth: int = 0) -> str | None:
        """Map a qualified name to an internal node id, following re-exports."""
        if depth > 6 or not q:
            return None
        if q in self.defs:
            return q
        # longest internal module prefix
        parts = q.split(".")
        for i in range(len(parts) - 1, 0, -1):
            mod_id = ".".join(parts[:i])
            if mod_id in self.modules:
                rest = parts[i:]
                mod = self.modules[mod_id]
                first = rest[0]
                target = None
                cand = f"{mod_id}.{first}"
                if cand in self.defs:
                    target = cand
                elif first in mod.imports:                       # re-export
                    target = self.resolve_qualified(mod.imports[first], depth + 1)
                else:
                    for star in mod.star_imports:
                        target = self.resolve_qualified(f"{star}.{first}", depth + 1)
                        if target:
                            break
                if target is None:
                    return None
                for r in rest[1:]:
                    target = self._member(target, r)
                    if target is None:
                        return None
                return target
        return None

    def _member(self, owner: str, name: str, depth: int = 0) -> str | None:
        """Look ``name`` up in ``owner`` (class: follow internal bases = MRO approx.)."""
        if depth > 10:
            return None
        child = self._children.get(owner, {}).get(name)
        if child:
            return child
        d = self.defs.get(owner)
        if d is not None and d.kind == "class":
            for b in self._resolved_bases(owner):
                r = self._member(b, name, depth + 1)
                if r:
                    return r
        if d is not None and d.kind == "module":
            return self.resolve_qualified(f"{owner}.{name}")
        return None

    def _resolved_bases(self, cls_id: str) -> list[str]:
        cache = getattr(self, "_base_cache", None)
        if cache is None:
            cache = self._base_cache = {}
        if cls_id in cache:
            return cache[cls_id]
        cache[cls_id] = []  # cycle guard
        out = []
        d = self.defs[cls_id]
        for raw in self.bases.get(cls_id, []):
            r = self.resolve_name(raw, d.module, scope_chain=[d.parent] if d.parent else [])
            if r and r[0] != cls_id and self.defs[r[0]].kind == "class":
                out.append(r[0])
        cache[cls_id] = out
        return out

    def resolve_name(self, dotted: str, module: str, scope_chain: list[str],
                     local_types: dict[str, str] | None = None,
                     enclosing_class: str | None = None) -> tuple[str, str] | None:
        """Resolve a dotted name used in ``module`` to (node id, rule)."""
        head, *rest = dotted.split(".")
        mod = self.modules[module]
        target, rule = None, None
        if head in ("self", "cls") and enclosing_class and rest:
            target, rule = enclosing_class, "self"
        elif local_types and head in local_types:
            target, rule = local_types[head], "typed"
        else:
            for scope in scope_chain:                           # enclosing defs (not class bodies)
                if scope in self.defs and self.defs[scope].kind != "class":
                    c = self._children.get(scope, {}).get(head)
                    if c:
                        target, rule = c, "local"
                        break
            if target is None:
                c = self._children.get(module, {}).get(head)
                if c:
                    target, rule = c, "module"
            if target is None and head in mod.imports:
                target = self.resolve_qualified(mod.imports[head])
                if target is None and head in self.modules:      # `import pkg`
                    target = head
                rule = "import"
            if target is None:
                for star in mod.star_imports:
                    target = self.resolve_qualified(f"{star}.{head}")
                    if target:
                        rule = "import"
                        break
            if target is None and head in self.modules:
                target, rule = head, "import"
        if target is None:
            return None
        for r in rest:
            nxt = self._member(target, r)
            if nxt is None:
                return None
            target = nxt
        return target, rule

    # -- pass 2: edges ------------------------------------------------------
    def build(self) -> nx.MultiDiGraph:
        self._collect()
        G = nx.MultiDiGraph()
        for d in self.defs.values():
            G.add_node(d.id, name=d.name, kind=d.kind, module=d.module, file=d.file,
                       start_line=d.start_line, end_line=d.end_line, is_async=d.is_async,
                       text=d.text, parent=d.parent)
        # contains
        for d in self.defs.values():
            if d.kind != "module" and d.parent in self.defs:
                G.add_edge(d.parent, d.id, key="contains", type="contains")
        # imports
        for mid, mod in self.modules.items():
            for q in dict.fromkeys(mod.import_targets):
                t = self.resolve_qualified(q) or (q if q in self.modules else None)
                if t and t != mid:
                    G.add_edge(mid, t, key=f"imports:{t}", type="imports")
        # inherits
        for cid in self.bases:
            for b in self._resolved_bases(cid):
                G.add_edge(cid, b, key="inherits", type="inherits", resolution="scope")
        # calls
        for mid, mod in self.modules.items():
            _CallVisitor(self, mod, G).run()
        return G


class _CallVisitor(ast.NodeVisitor):
    """Attribute each call to the innermost enclosing def and resolve it."""

    def __init__(self, b: CodeGraphBuilder, mod: ModuleInfo, G: nx.MultiDiGraph):
        self.b, self.mod, self.G = b, mod, G
        self.stack: list[str] = [mod.id]
        self.class_stack: list[str | None] = [None]
        self.types_stack: list[dict[str, str]] = [{}]

    def run(self):
        for stmt in self.mod.tree.body:
            self.visit(stmt)

    def _cur(self):
        return self.stack[-1]

    def visit_ClassDef(self, node):
        for d in node.decorator_list:
            self.visit(d)
        for b in node.bases + [k.value for k in node.keywords]:
            self.visit(b)
        cid = f"{self._cur()}.{node.name}"
        self.stack.append(cid)
        self.class_stack.append(cid)
        self.types_stack.append({})
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop(); self.class_stack.pop(); self.types_stack.pop()

    def _func(self, node):
        for d in node.decorator_list:
            self.visit(d)
        fid = f"{self._cur()}.{node.name}"
        types: dict[str, str] = {}
        for a in node.args.args + node.args.kwonlyargs + node.args.posonlyargs:
            if a.annotation is not None:
                self._bind_annotation(types, a.arg, a.annotation)
        self.stack.append(fid)
        self.class_stack.append(self.class_stack[-1])  # methods and closures inside methods keep `self`
        self.types_stack.append(types)
        for stmt in node.body:
            self.visit(stmt)
        self.stack.pop(); self.class_stack.pop(); self.types_stack.pop()

    visit_FunctionDef = _func
    visit_AsyncFunctionDef = _func

    def _scope_chain(self):
        return list(reversed(self.stack[1:]))

    def _bind_annotation(self, types, var, ann):
        name = _dotted(ann) if not isinstance(ann, ast.Constant) else ann.value if isinstance(ann.value, str) else None
        if isinstance(ann, ast.Subscript):  # Optional[X] / list[X] – take X if simple
            name = _dotted(ann.slice) if isinstance(ann.slice, (ast.Name, ast.Attribute)) else None
        if not name or not isinstance(name, str):
            return
        r = self.b.resolve_name(name, self.mod.id, self._scope_chain())
        if r and self.b.defs[r[0]].kind == "class":
            types[var] = r[0]

    def visit_Assign(self, node):
        self.generic_visit(node)
        if isinstance(node.value, ast.Call) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = _dotted(node.value.func)
            if name:
                r = self.b.resolve_name(name, self.mod.id, self._scope_chain(), self.types_stack[-1], self.class_stack[-1])
                if r and self.b.defs[r[0]].kind == "class":
                    self.types_stack[-1][node.targets[0].id] = r[0]

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if isinstance(node.target, ast.Name) and node.annotation is not None:
            self._bind_annotation(self.types_stack[-1], node.target.id, node.annotation)

    def visit_Call(self, node: ast.Call):
        self.generic_visit(node)
        func = node.func
        src = self._cur()
        target = None
        # super().m(...)
        if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Call)
                and isinstance(func.value.func, ast.Name) and func.value.func.id == "super"
                and self.class_stack[-1]):
            for base in self.b._resolved_bases(self.class_stack[-1]):
                m = self.b._member(base, func.attr)
                if m:
                    target = (m, "super")
                    break
        else:
            name = _dotted(func)
            if name:
                target = self.b.resolve_name(name, self.mod.id, self._scope_chain(),
                                             self.types_stack[-1], self.class_stack[-1])
            if (target is None and isinstance(func, ast.Attribute) and self.b.unique_name_fallback
                    and func.attr not in _BUILTIN_METHODS and not func.attr.startswith("__")):
                cands = [c for c in self.b._by_name.get(func.attr, []) if self.b.defs[c].kind == "method"]
                if len(cands) == 1:
                    target = (cands[0], "unique-name")
        if target and target[0] != src and src in self.G:
            tid, rule = target
            if self.b.defs[tid].kind == "module":
                return
            if not self.G.has_edge(src, tid, key="calls"):
                self.G.add_edge(src, tid, key="calls", type="calls", resolution=rule)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def build_graph(root: str | Path, **kwargs) -> nx.MultiDiGraph:
    return CodeGraphBuilder(root, **kwargs).build()


def to_node_link(G: nx.MultiDiGraph, *, extra_attrs: bool = True) -> dict:
    """Serialise to the competition's node-link JSON schema."""
    nodes = []
    for n, a in G.nodes(data=True):
        rec = {"id": n, "name": a.get("name"), "text": a.get("text", "")}
        if extra_attrs:
            rec.update({k: a[k] for k in ("kind", "file", "start_line", "end_line", "is_async") if k in a})
        nodes.append(rec)
    edges = []
    for u, v, k, a in G.edges(keys=True, data=True):
        rec = {"source": u, "target": v, "type": a.get("type"), "key": k}
        if extra_attrs and "resolution" in a:
            rec["resolution"] = a["resolution"]
        edges.append(rec)
    return {"directed": True, "multigraph": True, "graph": {}, "nodes": nodes, "edges": edges}


def flat_id(node_id: str, kind_of, module_id: str) -> str:
    """Id convention of the competition graphs (reverse-engineered on all 127 released graphs).

    * a class is ``<module>.<ClassName>`` whatever its nesting;
    * a function/method is ``<module>.<innermost enclosing class>.<name>`` (or
      ``<module>.<name>`` when no class encloses it) — enclosing *function* scopes are dropped.

    Distinct definitions can therefore collapse into one id (e.g. every nested
    ``class Config`` of a module, or two closures named ``wrapper``).
    ``kind_of(id)`` returns "module" | "class" | "function" | "method" | None.
    """
    if node_id == module_id or not node_id.startswith(module_id + "."):
        return node_id
    rest = node_id[len(module_id) + 1:].split(".")
    cur, innermost_class = module_id, None
    for part in rest[:-1]:
        cur = f"{cur}.{part}"
        if kind_of(cur) == "class":
            innermost_class = part
    if kind_of(node_id) == "class" or innermost_class is None:
        return f"{module_id}.{rest[-1]}"
    return f"{module_id}.{innermost_class}.{rest[-1]}"


def official_view(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Relabel a cgl graph with the competition id convention (function scopes dropped).

    Nested functions with the same name in the same class/module collapse into one
    node, exactly as in the released graphs.
    """
    kind = {n: a.get("kind") for n, a in G.nodes(data=True)}
    mapping = {n: flat_id(n, kind.get, a.get("module", n)) for n, a in G.nodes(data=True)}
    return nx.relabel_nodes(G, mapping, copy=True)


def restrict(G: nx.MultiDiGraph, *, node_kinds=None, edge_types=None, drop_async: bool = False,
             drop_resolutions: Iterable[str] = ()) -> nx.MultiDiGraph:
    """Ablation helper: sub-graph with selected node kinds / edge types."""
    drop_resolutions = set(drop_resolutions)
    keep = [n for n, a in G.nodes(data=True)
            if (node_kinds is None or a.get("kind") in node_kinds) and not (drop_async and a.get("is_async"))]
    H = nx.MultiDiGraph()
    H.add_nodes_from((n, G.nodes[n]) for n in keep)
    ks = set(keep)
    for u, v, k, a in G.edges(keys=True, data=True):
        if u in ks and v in ks and (edge_types is None or a.get("type") in edge_types) \
                and a.get("resolution") not in drop_resolutions:
            H.add_edge(u, v, key=k, **a)
    return H
