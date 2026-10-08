#!/bin/bash
# Experiment 5 on a Mac: Gemma 4 as a graph-navigating localization agent, released graph vs codegraph-loc.
#
#   caffeinate -i bash ~/Desktop/Kaggle/codegraph-loc/scripts/run_agent_mac.sh pilot        # ~20 tasks, < $0.50
#   caffeinate -i bash ~/Desktop/Kaggle/codegraph-loc/scripts/run_agent_mac.sh full         # all splits, cap $25
#   caffeinate -i bash ~/Desktop/Kaggle/codegraph-loc/scripts/run_agent_mac.sh full 26b     # second model, cap $15
#
# Resumable: run the same command again after an interruption; finished episodes are skipped.
# Needs: git, python3 with numpy/scipy/networkx, the repository clones in ../repos, the competition data in
# ../dati/competition, and the OpenRouter key file ~/Desktop/Kaggle/openrouter_key.txt (never printed).
set -u
cd "$(dirname "$0")/.." || exit 1
MODE=${1:-pilot}
case "${2:-31b}" in
  31b) MODEL=google/gemma-4-31b-it;      PROVIDER=CoreWeave; CAP=25 ;;
  26b) MODEL=google/gemma-4-26b-a4b-it;  PROVIDER=NextBit;   CAP=15 ;;
  *) echo "second argument must be 31b or 26b"; exit 1 ;;
esac
SLUG=${MODEL#google/}
export PYTHONPATH=src
PROTO=$(python3 -c "from cgl.agent import PROTOCOL; print(PROTOCOL)") || exit 1
echo "== 1/3 checking tools and data"
command -v git >/dev/null || { echo "git is missing: run 'xcode-select --install', then run this script again."; exit 1; }
python3 -c "import numpy, scipy, networkx" || exit 1
for r in fastapi/fastapi Textualize/rich psf/requests encode/httpx django/django sympy/sympy \
         scikit-learn/scikit-learn matplotlib/matplotlib pytest-dev/pytest sphinx-doc/sphinx astropy/astropy \
         pydata/xarray pylint-dev/pylint mwaskom/seaborn pallets/flask materialsproject/pymatgen; do
  n=$(basename "$r")
  if [ ! -d "../repos/$n/.git" ]; then echo "   cloning $r"; git clone -q "https://github.com/$r" "../repos/$n" || exit 1; fi
done
[ -f ../dati/competition/tasks.jsonl ] || { echo "competition data not found in ../dati/competition"; exit 1; }
if [ "$MODE" = pilot ]; then
  OUT=results/agent/pilot_$PROTO/$SLUG   # protocol v1's pilot is in results/agent/pilot/
  echo "== 2/3 pilot (protocol $PROTO): 20 random public-history tasks x 2 graphs ($MODEL via $PROVIDER)"
  python3 scripts/17_agent_localize.py --splits public --sample-tasks 20 --model "$MODEL" --provider "$PROVIDER" \
    --workers 4 --budget-usd 2 --out "$OUT" || exit 1
  echo "== 3/3 pilot mechanics (no accuracy is shown on purpose)"
  python3 scripts/18_eval_agent.py --answers "$OUT/answers.jsonl" --mechanics-only
  echo "PILOT DONE. Send the block above to Claude."
elif [ "$MODE" = full ]; then
  OUT=results/agent/$SLUG
  EXCL=$(ls results/agent/pilot*/gemma-4-31b-it/answers.jsonl 2>/dev/null | head -1)  # same 20 tasks in every pilot
  X=$( [ -n "$EXCL" ] && echo --exclude-tasks-from "$EXCL" )
  echo "== 2/3 full run (protocol $PROTO): held-out, public history and competition tasks x 2 graphs, pilot tasks skipped ($MODEL via $PROVIDER)"
  python3 scripts/17_agent_localize.py --splits swebl,pymatgen,public,comp --model "$MODEL" --provider "$PROVIDER" \
    --workers 4 --budget-usd "$CAP" --out "$OUT" $X || exit 1
  echo "== 3/3 scoring (pilot tasks excluded)"
  python3 scripts/18_eval_agent.py --answers "$OUT/answers.jsonl" $X
  python3 scripts/18_eval_agent.py --answers "$OUT/answers.jsonl" --mechanics-only > "$OUT/mechanics.json"
  echo "FULL RUN DONE. Results: $OUT/summary.json"
else
  echo "first argument must be pilot or full"; exit 1
fi
