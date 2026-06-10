#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --time=4:00:00
#SBATCH -J ta_topoprobe
#SBATCH --output=logs/ta_topoprobe_%j.out
# Val-select the 1k topology for Analog-Z and Analog-ZZ on TissueMNIST (8-class
# macro-OVR AUC). Needs the analog-ZZ cache (merged) and analog-Z cache (derived).
cd $HOME/MoT-DAQCNN
export OMP_NUM_THREADS=16
.venv/bin/python experiments/probe_tissue_analog_topology.py
echo "TA_TOPOPROBE_EXIT $?"
