#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 36:00:00
#SBATCH -o logs/te_%x_%j.out
#SBATCH -e logs/te_%x_%j.err
# One end-to-end TissueMNIST HP search per model. MODEL is passed via
# --export=ALL,MODEL=<name> and selects configs/tissue_mnist/endtoend/<MODEL>.yml.
# Reduced budget (25 trials, 5 validation seeds) for the 165k-image dataset.
cd ~/MoT-DAQCNN
mkdir -p logs
export OMP_NUM_THREADS=8
PY=.venv/bin/python
CFG=configs/tissue_mnist/endtoend/${MODEL}.yml
SRCH=configs/tissue_mnist/hp_search/endtoend_search.yml
OUT=outputs/paper_results/hp_search/tissue_mnist/${MODEL}
$PY experiments/hyperparameter_search.py \
  --config "$CFG" --search-config "$SRCH" \
  --n-trials 25 --validate-top-k 1 --validation-seeds 0 1 2 3 4 \
  --output-dir "$OUT"
echo "TE_${MODEL}_DONE"
