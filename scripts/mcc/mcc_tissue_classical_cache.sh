#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 4:00:00
#SBATCH -J tiss_clscache
#SBATCH -o logs/tiss_clscache_%j.out
#SBATCH -e logs/tiss_clscache_%j.err
# Build the 3 TissueMNIST classical-nonlinear feature caches (poly2_45, rff_45,
# rff_180) in the quantum-cache .npz+.json format, so the end-to-end rows flow
# through the same bypass_quantum path as the quantum/random rows. Pure CPU
# feature math (no quantum sim); slow only because Tissue has 165k images.
cd ~/MoT-DAQCNN
mkdir -p logs
export OMP_NUM_THREADS=8
PY=.venv/bin/python
for S in poly2_45 rff_45 rff_180; do
  echo "=== building $S ==="
  $PY experiments/create_classical_cache.py --dataset tissue_mnist \
      --source "$S" --out-dir data/quantum_datasets
  echo "=== $S exit $? ==="
done
echo TISSUE_CLSCACHE_DONE
