"""Send the re-ranking prompts to Gemma 4 through the OpenRouter chat-completions API.

Standard library only (runs with any Python >= 3.9, no packages needed). The API key is read
from a file and never printed or stored in the outputs.

Reproducibility settings: temperature 0, seed 0, reasoning off, providers restricted to
bf16 weights (``--quantization``), provider and token usage logged for every request.
Resumable: requests already answered in <out> are skipped. A spending cap stops the run.

Usage:
  python3 scripts/11_run_llm.py --limit 10          # pilot: first 10 requests (5 tasks x 2 conditions)
  python3 scripts/11_run_llm.py                     # everything left to do
Options: --prompts results/llm/prompts.jsonl --out results/llm/responses.jsonl
         --key-file ~/Desktop/Kaggle/openrouter_key.txt --model google/gemma-4-31b-it
         --workers 4 --budget-usd 3 --quantization bf16
"""
import argparse
import getpass
import json
import re
import ssl
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

URL = "https://openrouter.ai/api/v1/chat/completions"


def ssl_context():
    try:
        import certifi  # present when e.g. the kaggle package is installed
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def call(body: dict, key: str, url: str, ctx, timeout: float = 120.0, tries: int = 6) -> dict:
    data = json.dumps(body).encode()
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json",
               "HTTP-Referer": "https://github.com/codegraph-loc", "X-Title": "codegraph-loc"}
    err = None
    for attempt in range(tries):
        t0 = time.time()
        try:
            req = urllib.request.Request(url, data=data, headers=headers, method="POST")
            with urllib.request.urlopen(req, timeout=timeout, context=ctx) as r:
                out = json.loads(r.read().decode())
            if "error" in out and not out.get("choices"):
                raise RuntimeError(json.dumps(out["error"])[:300])
            out["_latency_s"] = round(time.time() - t0, 2)
            return out
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:300]
            err = f"HTTP {e.code}: {msg}"
            if e.code in (400, 401, 402, 403, 404):  # not retryable: bad request, key, credits, model
                raise RuntimeError(err)
        except (urllib.error.URLError, ssl.SSLError, TimeoutError, ConnectionError, RuntimeError,
                json.JSONDecodeError) as e:
            if isinstance(e, ssl.SSLError) or isinstance(getattr(e, "reason", None), ssl.SSLError):
                raise RuntimeError(f"SSL error ({e}). On a python.org install run "
                                   f"'Install Certificates.command' in /Applications/Python 3.x/ and retry.")
            err = f"{type(e).__name__}: {str(e)[:300]}"
        time.sleep(min(60, 2 ** attempt * 2))
    raise RuntimeError(f"gave up after {tries} tries: {err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="results/llm/prompts.jsonl")
    ap.add_argument("--out", default="results/llm/responses.jsonl")
    ap.add_argument("--key-file", default="~/Desktop/Kaggle/openrouter_key.txt")
    ap.add_argument("--model", default="google/gemma-4-31b-it")
    ap.add_argument("--quantization", default="bf16", help="restrict providers to this weight precision ('' = any)")
    ap.add_argument("--reasoning", choices=["off", "on"], default="off")
    ap.add_argument("--max-tokens", type=int, default=300)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="only the first N requests (pilot)")
    ap.add_argument("--budget-usd", type=float, default=3.0, help="stop when the run's cost exceeds this")
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()

    kf = Path(a.key_file).expanduser()
    key = kf.read_text().strip() if kf.exists() else ""
    if not key.startswith("sk-or-"):
        print(f"No valid OpenRouter key in {kf} (keys start with 'sk-or-').")
        key = getpass.getpass("Paste your OpenRouter API key and press Enter (input is hidden): ").strip()
        if not key.startswith("sk-or-"):
            sys.exit("That does not look like an OpenRouter key (it should start with 'sk-or-'). Nothing saved.")
        kf.write_text(key + "\n")
        kf.chmod(0o600)
        print(f"Key saved to {kf} (readable only by you).")

    prompts = []
    for l in open(a.prompts):
        try:
            prompts.append(json.loads(l))
        except json.JSONDecodeError:  # a line still being written by 10_build_llm_prompts.py
            pass
    out = Path(a.out)
    done = set()
    if out.exists():
        for l in open(out):
            r = json.loads(l)
            if r.get("ok"):
                done.add(r["custom_id"])
    todo = [p for p in prompts if p["custom_id"] not in done]
    if a.limit:
        todo = [p for p in prompts[:a.limit] if p["custom_id"] not in done]
    print(f"{len(prompts)} prompts, {len(done)} already answered, {len(todo)} to send "
          f"(model {a.model}, {a.workers} workers, budget ${a.budget_usd})", flush=True)
    if not todo:
        print("Nothing to do.")
        return

    ctx = ssl_context()
    lock = threading.Lock()
    spent = [0.0]
    stop = threading.Event()
    stats = {"ok": 0, "fail": 0}

    def body_for(p):
        b = {"model": a.model, "messages": p["messages"], "temperature": 0, "seed": 0,
             "max_tokens": a.max_tokens, "usage": {"include": True},
             "reasoning": {"enabled": a.reasoning == "on", "exclude": True}}
        if a.quantization:
            b["provider"] = {"quantizations": [a.quantization]}
        return b

    def work(p):
        if stop.is_set():
            return None
        rec = {"custom_id": p["custom_id"], "model": a.model, "time": time.strftime("%Y-%m-%dT%H:%M:%S")}
        try:
            r = call(body_for(p), key, a.url, ctx)
            ch = (r.get("choices") or [{}])[0]
            u = r.get("usage") or {}
            rec.update(ok=True, content=(ch.get("message") or {}).get("content") or "",
                       finish_reason=ch.get("finish_reason"), provider=r.get("provider"),
                       served_model=r.get("model"), gen_id=r.get("id"), latency_s=r.get("_latency_s"),
                       prompt_tokens=u.get("prompt_tokens"), completion_tokens=u.get("completion_tokens"),
                       cost_usd=u.get("cost"))
        except Exception as e:  # recorded and retried on the next run
            msg = str(e)
            rec.update(ok=False, error=msg)
            if "HTTP 401" in msg or "HTTP 402" in msg or "HTTP 403" in msg or "SSL error" in msg:
                stop.set()
        with lock:
            with open(out, "a") as fo:
                fo.write(json.dumps(rec) + "\n")
            stats["ok" if rec["ok"] else "fail"] += 1
            spent[0] += rec.get("cost_usd") or 0.0
            n = stats["ok"] + stats["fail"]
            if not rec["ok"]:
                print(f"  ! {rec['custom_id']}: {rec['error'][:200]}", flush=True)
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)} done, {stats['fail']} failed, cost so far ${spent[0]:.4f}", flush=True)
            if spent[0] > a.budget_usd:
                stop.set()
        return rec

    out.parent.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as ex:
        futs = [ex.submit(work, p) for p in todo]
        for f in as_completed(futs):
            f.result()
    msg = "STOPPED EARLY (budget, key or credit problem; see messages above)" if stop.is_set() else "ALL DONE"
    print(f"{msg}: {stats['ok']} answered, {stats['fail']} failed, run cost ${spent[0]:.4f}")
    if stats["fail"]:
        print("Run the same command again to retry the failed requests.")
    first = next((json.loads(l) for l in open(out) if json.loads(l).get("ok")), None)
    if first:
        m = re.search(r"\{.*\}", first["content"], re.S)
        print("Example answer:", (m.group(0) if m else first["content"])[:120], "| provider:", first.get("provider"))


if __name__ == "__main__":
    main()
