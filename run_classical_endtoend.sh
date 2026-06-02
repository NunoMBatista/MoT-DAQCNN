#!/bin/bash
# Launch the 6 classical-nonlinear end-to-end HP searches (poly-2/RFF x breast/pneu),
# matched protocol to the quantum table rows: 200 trials, top-1 validated on 10 seeds.
# Keeps ~3 concurrent; idempotent (skips a study whose validation_summary.json exists).
set -u
PY=/home/nuno/python_envs/ML/bin/python
OUT=outputs/paper_results/hp_search/classical_nonlinear
MAXJOBS=3

run_one() {
  local ds=$1 src=$2
  local dir="$OUT/${ds}__${src}"
  if [ -f "$dir/validation/validation_summary.json" ]; then
    echo "[skip] $ds $src (already done)"; return
  fi
  echo "[start] $ds $src -> $dir"
  $PY experiments/hyperparameter_search.py \
    --config configs/${ds}/classical_nonlinear/${src}.yml \
    --search-config configs/${ds}/hp_search/classical_nonlinear_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir "$dir" > "logs/classical_${ds}_${src}.log" 2>&1
  echo "[done] $ds $src"
}

mkdir -p logs "$OUT"
for ds in breast_mnist pneumonia_mnist; do
  for src in poly2 rff_45 rff_180; do
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 30; done
    run_one "$ds" "$src" &
  done
done
wait
echo "ALL_CLASSICAL_ENDTOEND_DONE"
