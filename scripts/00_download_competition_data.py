"""Download ONLY the competition files needed for the audit (not the 22 GB of repo snapshots).

Usage (on the machine that will hold the data):
    python3 -m pip install kaggle
    # Kaggle API token: kaggle.com -> Settings -> API -> "Create New Token"
    python3 scripts/00_download_competition_data.py ~/Desktop/Kaggle/dati

You must first accept the main competition rules
(https://www.kaggle.com/competitions/gemma-4-developer-agent/rules), otherwise Kaggle answers 403.
Every step is printed; files already present are skipped, so the script can be re-run safely.
The files stay on your machine: the competition rules forbid redistributing them.
"""
print("1) Starting", flush=True)

import os
import sys
import zipfile
from pathlib import Path

COMP = "gemma-4-developer-agent"
WHEELHOUSE = "metric/gemma-4-developer-agent-wheelhouse"
WANT_FILES = {"tasks.jsonl", "HARNESS_README.md"}
WANT_DIRS = ("graphs/", "embeddings/", "sample_submission/")

home = Path.home() / ".kaggle"
print(f"   Python {sys.version.split()[0]}")
print(f"   ~/.kaggle/kaggle.json:  {'present' if (home / 'kaggle.json').exists() else 'MISSING'}")
print(f"   ~/.kaggle/access_token: {'present' if (home / 'access_token').exists() else 'missing'}")
print(f"   KAGGLE_API_TOKEN env var: {'present' if os.environ.get('KAGGLE_API_TOKEN') else 'missing'}", flush=True)

dest = Path(sys.argv[1] if len(sys.argv) > 1 else "~/Desktop/Kaggle/dati").expanduser()
dest.mkdir(parents=True, exist_ok=True)
print(f"2) Destination folder: {dest}", flush=True)

print("3) Importing the kaggle library...", flush=True)
try:
    import kaggle as _k  # noqa: F401  (some versions authenticate on import)
    from kaggle.api.kaggle_api_extended import KaggleApi
except Exception as e:
    print(f"   IMPORT ERROR: {type(e).__name__}: {e}")
    print("   -> Try: python3 -m pip install --upgrade kaggle")
    sys.exit(1)
print(f"   kaggle version: {getattr(_k, '__version__', '?')}", flush=True)

print("4) Authenticating...", flush=True)
api = KaggleApi()
try:
    api.authenticate()
except Exception as e:
    print(f"   AUTHENTICATION ERROR: {type(e).__name__}: {e}")
    sys.exit(1)
print("   ok", flush=True)


def list_comp_files():
    names, token, seen = [], None, set()
    for _ in range(50):  # at most 50 pages: no infinite loops
        try:
            res = api.competition_list_files(COMP, page_token=token, page_size=200)
        except TypeError:  # old client: no paging
            res = api.competition_list_files(COMP)
        files = getattr(res, "files", res) or []
        names += [getattr(f, "name", None) or getattr(f, "ref", None) or str(f) for f in files]
        token = getattr(res, "next_page_token", None)
        print(f"   ...{len(names)} files listed", flush=True)
        if not token or token in seen:
            break
        seen.add(token)
    return names


def unzip_here(folder: Path):
    for z in folder.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(folder)
        z.unlink()


print("5) Listing competition files...", flush=True)
try:
    names = list_comp_files()
except Exception as e:
    print(f"   ERROR: {type(e).__name__}: {e}")
    print("   -> If this is a 403: accept the rules at "
          "https://www.kaggle.com/competitions/gemma-4-developer-agent/rules")
    sys.exit(1)
todo = [n for n in names if n in WANT_FILES or n.startswith(WANT_DIRS)]
print(f"   {len(names)} files in total, {len(todo)} to download", flush=True)

print("6) Downloading...", flush=True)
todo.sort(key=lambda n: (n != "tasks.jsonl", n))  # tasks.jsonl first: acts as an access test
errors = 0
for i, n in enumerate(todo, 1):
    out = dest / "competition" / Path(n).parent
    out.mkdir(parents=True, exist_ok=True)
    if (out / Path(n).name).exists():
        continue
    try:
        api.competition_download_file(COMP, n, path=str(out), quiet=True)
        unzip_here(out)
    except Exception as e:
        errors += 1
        print(f"   error on {n}: {type(e).__name__}: {e}", flush=True)
        if "403" in str(e) and i == 1:
            print("\n   STOP: Kaggle answers 403 (forbidden) on the competition files.")
            print("   This almost always means the rules of the MAIN competition (different from")
            print("   the Paper Track) have not been accepted yet. Open:")
            print("   https://www.kaggle.com/competitions/gemma-4-developer-agent/rules")
            print("   click 'Join Competition' / 'I Understand and Accept', then re-run this script.")
            sys.exit(1)
    if i % 20 == 0 or i == len(todo):
        print(f"   {i}/{len(todo)}", flush=True)
print(f"   finished with {errors} errors", flush=True)

print("7) Harness packages (wheelhouse)...", flush=True)
wh = dest / "wheelhouse"
wh.mkdir(exist_ok=True)
try:
    wnames, token, seen = [], None, set()
    for _ in range(50):  # all pages of the listing
        try:
            res = api.dataset_list_files(WHEELHOUSE, page_token=token, page_size=200)
        except TypeError:
            res = api.dataset_list_files(WHEELHOUSE)
        wnames += [getattr(f, "name", str(f)) for f in (getattr(res, "files", res) or [])]
        token = getattr(res, "next_page_token", None)
        if not token or token in seen:
            break
        seen.add(token)
    print(f"   {len(wnames)} files in the wheelhouse", flush=True)
    for name in wnames:
        base = name.split("/")[-1].lower()
        if base.startswith(("swegemma", "adk_submission", "adk-submission", "adk_eval", "adk-eval")):
            if (wh / name.split("/")[-1]).exists():
                continue
            api.dataset_download_file(WHEELHOUSE, name, path=str(wh), quiet=True)
            unzip_here(wh)
            print("   downloaded", name, flush=True)
except Exception as e:
    print(f"   wheelhouse error (not critical): {type(e).__name__}: {e}")

print("DONE. Folder:", dest)
