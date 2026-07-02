#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=2:00:00
#SBATCH -J ta_derivez
#SBATCH --output=logs/ta_derivez_%j.out
# Slice the analog-Z cache (first 9 <Z> channels per topology) out of the merged
# analog-ZZ cache. Free: no quantum recompute.
cd $HOME/MoT-DAQCNN
.venv/bin/python experiments/derive_z_from_zz.py \
  --zz data/quantum_datasets/tissue_mnist__k3_s3_tkin-hor-ver-cro-rin-cha-sta-gri_ev2.50_sc1_gray_zz_analog.npz
echo "TA_DERIVEZ_EXIT $?"
