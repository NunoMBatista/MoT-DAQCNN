#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=200G
#SBATCH -t 16:00:00
#SBATCH -J tiss_analog_cap
#SBATCH -o logs/tiss_analog_cap_%j.out
#SBATCH -e logs/tiss_analog_cap_%j.err
# Analog column of the TissueMNIST capacity sweep: the 3 analog sources, same
# protocol as the digital sources (epochs 80, n-trials 20, batch 256, standardized,
# 8-class). Writes into the SAME output dir; the sweep skips existing (digital)
# cells, so this only fills the 3x5 analog grid. Resumable on timeout.
cd ~/MoT-DAQCNN
mkdir -p logs
PY=.venv/bin/python
OUT=outputs/paper_results/capacity_sweep_std/tissue_mnist
export OMP_NUM_THREADS=8
for s in analog_z_1k analog_zz_1k analog_zz_4k; do
  $PY experiments/head_capacity_sweep.py --dataset tissue_mnist --sources $s \
     --output-dir $OUT --device cpu --standardize \
     --num-classes 8 --batch-size 256 --epochs 80 --n-trials 20 \
     > logs/tiss_${s}.log 2>&1 &
done
wait
echo "TISSUE_ANALOG_CAP_DONE"
