#!/bin/bash
#SBATCH -A <your-account>
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=4:00:00
#SBATCH -J ta_merge
#SBATCH --output=logs/ta_merge_%j.out
# Merge the 1970 analog Tissue chunks into one canonical cache. The merge tool
# runs four integrity gates (identity / exact tiling / label checksum / all
# splits present) and refuses to write unless every one passes.
cd $HOME/MoT-DAQCNN
PY=$HOME/MoT-DAQCNN/.venv/bin/python
$PY experiments/merge_quantum_chunks.py \
  --chunk-dir data/quantum_datasets/chunks_tissue_analog \
  --out-dir   data/quantum_datasets
echo "TA_MERGE_EXIT $?"
