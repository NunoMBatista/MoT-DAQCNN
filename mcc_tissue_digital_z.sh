#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 24
#SBATCH --mem=200G
#SBATCH -t 8:00:00
#SBATCH -J tiss_dz
#SBATCH -o logs/tiss_dz_%j.out
#SBATCH -e logs/tiss_dz_%j.err
# Relaunch the two Tissue capacity-sweep sources missing from job 144737:
#   - digital_z_1k : died instantly because no Z-only cache existed (only the ZZ
#                    cache is on disk). Z-only is a free slice of the ZZ cache, so
#                    we derive it first (no quantum recompute), then run the sweep.
#   - random_9     : the dimension-matched random baseline for the 9-dim digital_z,
#                    never included in the original 9-source run. On-the-fly, no cache.
# Writes into the SAME tissue output dir so the grid completes to 11 sources.
cd ~/MoT-DAQCNN
mkdir -p logs
PY=.venv/bin/python
ZZ=data/quantum_datasets/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz.npz
ZCACHE=$(echo "$ZZ" | sed 's/_zz\.npz/_z.npz/')
OUT=outputs/paper_results/capacity_sweep_std/tissue_mnist

# 1) derive the Z-only cache from the ZZ cache if absent (free slice, ~50GB read)
if [ ! -f "$ZCACHE" ]; then
  echo "deriving Z cache from ZZ..."
  $PY experiments/derive_z_from_zz.py --zz "$ZZ" || { echo "DERIVE_FAILED"; exit 1; }
else
  echo "Z cache already present: $ZCACHE"
fi

# 2) run the two missing sources in parallel, identical protocol to cap_std_tissue.sh
export OMP_NUM_THREADS=12
for s in digital_z_1k random_9; do
  $PY experiments/head_capacity_sweep.py --dataset tissue_mnist --sources $s \
     --output-dir $OUT --device cpu --standardize \
     --num-classes 8 --batch-size 256 --epochs 80 --n-trials 20 \
     > logs/tiss_${s}.log 2>&1 &
done
wait
echo "TISSUE_DZ_DONE"
