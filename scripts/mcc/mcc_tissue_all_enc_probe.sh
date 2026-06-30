#!/bin/bash
#SBATCH -A mxs42
#SBATCH -p shared
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH -J ta_allenc
#SBATCH --output=logs/ta_allenc_%x_%j.out
# Matched full-dataset (165k) linear probe for ONE TissueMNIST encoding cache,
# selected by $CACHE (digital | analog). Submit twice to run both in parallel.
# Reads the big ZZ cache once (-> 200G), writes per-topology rows INCREMENTALLY
# so a kill never loses completed work. ~5h per cache at full data.
cd $HOME/MoT-DAQCNN
export OMP_NUM_THREADS=16
.venv/bin/python experiments/probe_tissue_all_encodings.py --cache "$CACHE"
echo "TA_ALLENC_${CACHE}_EXIT $?"
