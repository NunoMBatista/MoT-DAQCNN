#!/bin/bash
# Submit the 8 TissueMNIST end-to-end HP searches, one job per model.
# The two quantum-ZZ jobs load the 47 GB ZZ cache and need ~200 GB (the value
# the capacity-sweep Tissue jobs used); digital_z (9 GB Z cache) and the
# classical/raw jobs fit comfortably in 64 GB. --mem here overrides the
# directive in mcc_tissue_e2e.sh; -J sets the job name used in the log path.
cd ~/MoT-DAQCNN
for m in digital_z random_1k random_4k trainable_1k trainable_4k raw; do
  sbatch -J "te_$m" --mem=64G --export=ALL,MODEL="$m" mcc_tissue_e2e.sh
done
for m in digital_zz_1k digital_zz_4k; do
  sbatch -J "te_$m" --mem=200G --export=ALL,MODEL="$m" mcc_tissue_e2e.sh
done
sleep 2
squeue -u $USER -o "%.10i %.14j %.2t %R"
