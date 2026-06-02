#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 48
#SBATCH --mem=64G
#SBATCH -t 2-00:00:00
#SBATCH -J pneucls
#SBATCH -o logs/pneucls_%j.out
#SBATCH -e logs/pneucls_%j.err
# Pneumonia classical-nonlinear end-to-end HP searches (poly2 / rff_45 / rff_180),
# matched protocol to the quantum table rows: 200 trials, top-1 validated on 10
# seeds. Runs through the classical-feature cache -> bypass_quantum -> CNN-head
# path (identical to the quantum rows). One process per source, parallel on one
# node. Idempotent/resumable: --resume loads the per-study Optuna SQLite, so a
# walltime requeue continues where it left off.
cd ~/MoT-DAQCNN
mkdir -p logs
PY=.venv/bin/python
OUT=outputs/paper_results/hp_search/classical_nonlinear
export OMP_NUM_THREADS=16
for src in poly2 rff_45 rff_180; do
  dir=$OUT/pneumonia_mnist__${src}
  $PY experiments/hyperparameter_search.py \
    --config configs/pneumonia_mnist/classical_nonlinear/${src}.yml \
    --search-config configs/pneumonia_mnist/hp_search/classical_nonlinear_search.yml \
    --n-trials 200 --validate-top-k 1 --validation-seeds 0 1 2 3 4 5 6 7 8 9 \
    --output-dir "$dir" --resume \
    > logs/pneucls_${src}.log 2>&1 &
done
wait
echo "ALL_PNEU_CLASSICAL_DONE"
