#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -J brcls
#SBATCH -o logs/brcls_%j.out
#SBATCH -e logs/brcls_%j.err
# BreastMNIST classical-nonlinear end-to-end HP searches (poly2/rff_45/rff_180),
# migrated from local. --resume loads the partial Optuna SQLite DBs rsynced from
# the laptop (poly2 ~76, rff_45 ~62, rff_180 ~54 trials done), so progress is
# preserved; each tops out at >=200 trials, then top-1 is validated on 10 seeds.
cd ~/MoT-DAQCNN
mkdir -p logs
PY=.venv/bin/python
OUT=outputs/paper_results/hp_search/classical_nonlinear
export OMP_NUM_THREADS=16
for src in poly2 rff_45 rff_180; do
  dir=$OUT/breast_mnist__${src}
  $PY experiments/hyperparameter_search.py \
    --config configs/breast_mnist/classical_nonlinear/${src}.yml \
    --search-config configs/breast_mnist/hp_search/classical_nonlinear_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir "$dir" --resume \
    > logs/brcls_${src}.log 2>&1 &
done
wait
echo "ALL_BREAST_CLASSICAL_DONE"
