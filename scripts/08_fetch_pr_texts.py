"""Fetch title and description of the pull requests behind the mined commits.

The competition's problem statements are "PR title + PR description" (checked on the 129
public tasks), so the public-history split uses the same construction.

Two back-ends:
  * GitHub REST API when a token file is given (fast, 5,000 requests/hour);
  * otherwise the public PR web page, parsed politely (one request per --delay seconds).
Research scraping of public, non-personal data for an open-access publication is allowed
by GitHub's terms; only the PR title and first description are kept.

Resumable: results are appended to the output JSONL and already-fetched PRs are skipped.
Usage:
  python scripts/08_fetch_pr_texts.py --mined results --out results/public/pr_texts.jsonl \
         [--exclude-tasks ../dati/competition/tasks.jsonl] [--token-file ../github_token.txt] [--max-seconds 150]
"""
import argparse
import html
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

OWNER = {"fastapi": "fastapi/fastapi", "rich": "Textualize/rich", "requests": "psf/requests", "httpx": "encode/httpx",
         "pymatgen": "materialsproject/pymatgen"}  # pymatgen: held-out split (Experiment 4)
UA = "codegraph-loc research crawler (open-access paper; contact via GitHub)"
_BODY = re.compile(r'<div class="comment-body markdown-body js-comment-body[^"]*">(.*?)</div>\s*</task-lists>', re.S)
_TITLE = re.compile(r"<title>(.*?)</title>", re.S)


def html_to_text(fragment: str) -> str:
    s = re.sub(r"<br\s*/?>|</p>|</li>|</h\d>|</pre>", "\n", fragment)
    s = re.sub(r"<li[^>]*>", "- ", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    return re.sub(r"\n\s*\n+", "\n\n", s).strip()


def get(url: str, headers: dict) -> tuple[int, str]:
    req = urllib.request.Request(url, headers={"User-Agent": UA, **headers})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, ""
    except (urllib.error.URLError, TimeoutError, ConnectionError):  # transient network error: retried next run
        return 0, ""


def fetch_web(repo: str, pr: int):
    code, page = get(f"https://github.com/{repo}/pull/{pr}", {})
    if code != 200:
        return code, None
    t = _TITLE.search(page)
    title = html.unescape(t.group(1)).split(" by ")[0].split(" · Pull Request")[0].strip() if t else ""
    b = _BODY.search(page)
    return 200, {"title": title, "body": html_to_text(b.group(1)) if b else ""}


def fetch_api(repo: str, pr: int, token: str):
    code, txt = get(f"https://api.github.com/repos/{repo}/pulls/{pr}",
                    {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    if code != 200:
        return code, None
    d = json.loads(txt)
    return 200, {"title": d.get("title") or "", "body": d.get("body") or ""}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mined", default="results")
    ap.add_argument("--out", default="results/public/pr_texts.jsonl")
    ap.add_argument("--exclude-tasks", default=None)
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--delay", type=float, default=1.0)
    ap.add_argument("--max-seconds", type=float, default=150)
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            done.add((r["repo"], r["pr"]))
    excluded = set()
    if a.exclude_tasks and Path(a.exclude_tasks).expanduser().exists():
        for l in open(Path(a.exclude_tasks).expanduser()):
            t = json.loads(l)
            excluded.add((t["instance_id"].split("_")[0], int(t["instance_id"].split("_")[-1])))
    token = Path(a.token_file).expanduser().read_text().strip() if a.token_file and Path(a.token_file).expanduser().exists() else None
    todo = []
    for f in sorted(Path(a.mined).glob("mined_*.jsonl")):
        for l in open(f):
            r = json.loads(l)
            if r.get("pr") and (r["repo"], r["pr"]) not in done and (r["repo"], r["pr"]) not in excluded:
                todo.append((r["repo"], r["pr"]))
    todo = list(dict.fromkeys(todo))
    print(f"{len(todo)} PRs to fetch ({'API' if token else 'web'}); {len(excluded)} competition PRs excluded", flush=True)
    start, n_ok = time.time(), 0
    with open(out, "a") as fo:
        for repo, pr in todo:
            if time.time() - start > a.max_seconds:
                print(f"time budget reached after {n_ok} PRs; run again to continue", flush=True)
                return
            full = OWNER[repo]
            code, d = fetch_api(full, pr, token) if token else fetch_web(full, pr)
            if code == 429 or code == 403:
                print(f"rate limited ({code}); stopping, run again later", flush=True)
                return
            if code == 0:
                print(f"  network error on #{pr}; will retry on the next run", flush=True)
                continue
            rec = {"repo": repo, "pr": pr, "status": code, "url": f"https://github.com/{full}/pull/{pr}",
                   "title": d["title"] if d else None, "body": d["body"] if d else None}
            fo.write(json.dumps(rec) + "\n")
            fo.flush()
            n_ok += 1
            if not token:
                time.sleep(a.delay)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
