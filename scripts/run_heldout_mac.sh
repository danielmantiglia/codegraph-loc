#!/bin/bash
# Experiment 4 on a Mac: finish the held-out benchmark in parallel, then run Gemma 4 on its prompts.
# Usage (from anywhere):  caffeinate -i bash ~/Desktop/Kaggle/codegraph-loc/scripts/run_heldout_mac.sh
# Needs: git, python3, ~4 GB of disk for the repository clones, the OpenRouter key file used by 11_run_llm.py.
set -u
cd "$(dirname "$0")/.." || exit 1
REPOS=../repos
OUT=results/heldout/loc
echo "== 1/5 checking tools"
command -v git >/dev/null || { echo "git is missing: run 'xcode-select --install', then run this script again."; exit 1; }
python3 -c "import numpy, scipy, networkx" 2>/dev/null || python3 -m pip install --user --quiet numpy scipy networkx \
  || python3 -m pip install --user --quiet --break-system-packages numpy scipy networkx || { echo "Could not install numpy/scipy/networkx."; exit 1; }
python3 -c "import numpy, scipy, networkx" || exit 1
echo "== 2/5 cloning the public repositories into $REPOS (first time only, ~3 GB)"
mkdir -p "$REPOS"
for r in django/django sympy/sympy scikit-learn/scikit-learn matplotlib/matplotlib pytest-dev/pytest \
         sphinx-doc/sphinx astropy/astropy pydata/xarray pylint-dev/pylint mwaskom/seaborn pallets/flask \
         materialsproject/pymatgen; do
  n=$(basename "$r")
  if [ ! -d "$REPOS/$n/.git" ]; then echo "   cloning $r"; git clone -q "https://github.com/$r" "$REPOS/$n" || { echo "clone of $r failed; run the script again"; exit 1; }; fi
done
echo "== 3/5 building graphs, rankings and prompts (parallel; resumable)"
CORES=$(sysctl -n hw.perflevel0.physicalcpu 2>/dev/null || sysctl -n hw.physicalcpu 2>/dev/null || echo 4)
N=$(( CORES > 6 ? 6 : CORES ))
for k in $(seq 0 $((N-1))); do
  PYTHONPATH=src python3 scripts/13_heldout_benchmark.py --swebench ../dati/swebench_lite_test.jsonl \
    --pymatgen results/heldout/pymatgen --repos "$REPOS" --out "$OUT" --shard "$k/$N" --max-seconds 100000 \
    > "$OUT/run_shard_$k.log" 2>&1 &
done
wait
NT=$(ls "$OUT/prompts" | wc -l | tr -d ' '); NS=$( [ -f "$OUT/skipped.txt" ] && wc -l < "$OUT/skipped.txt" | tr -d ' ' || echo 0)
echo "   tasks with prompts: $NT, skipped (no Python gold edit): $NS (expected total 745)"
grep -h "^  !" "$OUT"/run_shard_*.log | head -5
echo "== 4/5 scoring summary"
PYTHONPATH=src python3 scripts/13_heldout_benchmark.py --out "$OUT" --summarize
PYTHONPATH=src python3 scripts/13_heldout_benchmark.py --out "$OUT" --merge-prompts
mkdir -p results/heldout/llm && cp "$OUT/prompts.jsonl" results/heldout/llm/prompts.jsonl
echo "== 5/5 Gemma 4 on the held-out prompts (resumable; about \$0.40)"
python3 scripts/11_run_llm.py --prompts results/heldout/llm/prompts.jsonl --out results/heldout/llm/responses.jsonl --workers 8
echo "HELDOUT ALL DONE"
