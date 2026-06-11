#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=8:00:00
#SBATCH -J ta_allenc
#SBATCH --output=logs/ta_allenc_%j.out
# Matched full-dataset linear probe for all 4 TissueMNIST encodings (Dig/Ana x
# Z/ZZ), 8 topologies, macro-OVR test AUC. Reads the digital (49G) and analog
# (23G) ZZ caches once each -> needs the big mem. Writes the CSV the merged
# 3-dataset probing table consumes.
cd $HOME/MoT-DAQCNN
export OMP_NUM_THREADS=16
.venv/bin/python experiments/probe_tissue_all_encodings.py
echo "TA_ALLENC_EXIT $?"
