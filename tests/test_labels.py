from pathlib import Path

from cgl.labels import is_test_path, locate_changes, parse_unified_diff

SRC = (Path(__file__).parent / "fixture_repo/src/pkg/core.py").read_text()

DIFF = '''diff --git a/src/pkg/core.py b/src/pkg/core.py
--- a/src/pkg/core.py
+++ b/src/pkg/core.py
@@ -4 +4 @@ from .sub.util import async_helper
-TIMEOUT = 5
+TIMEOUT = 10
@@ -24,3 +24,3 @@ class Client(Base):
     async def send(self, request):
-        await async_helper(request)
+        await async_helper(request, timeout=TIMEOUT)
         return self.ping()
@@ -31,0 +32 @@ class Client(Base):
+        self.n += 1
@@ -37,0 +39,4 @@ def make() -> Client:
+
+
+def new_function():
+    return 0
'''


def test_parse_counts():
    (fc,) = parse_unified_diff(DIFF)
    assert fc.path == "src/pkg/core.py"
    assert fc.changed_old_lines == {4, 25}
    assert fc.insert_after == {4, 25, 31, 37}


def test_locate_changes_async_and_module_level():
    (fc,) = parse_unified_diff(DIFF)
    got = locate_changes(fc, SRC, "pkg.core")
    # async method; code appended at the end of run(); module constant + new def at EOF
    assert got["functions"] == {"pkg.core.Client.send", "pkg.core.Client.run", "pkg.core"}


def test_dash_content_not_header():
    d = ("--- a/x.py\n+++ b/x.py\n@@ -1,2 +1,2 @@\n--- not a header\n+-- replaced\n ctx\n")
    (fc,) = parse_unified_diff(d)
    assert fc.changed_old_lines == {1}


def test_is_test_path():
    assert is_test_path("tests/test_core.py") and is_test_path("pkg/conftest.py")
    assert not is_test_path("src/pkg/core.py")
