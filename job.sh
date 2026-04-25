#!/bin/bash
#SBATCH --job-name=quantum_gen
#SBATCH --nodes=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=quantum_gen_%j.log
#SBATCH --error=quantum_gen_%j.err

# Load your environment (replace with your actual env activation)
source /scratch/sp7007/nyuenv/bin/activate

# Move to the project directory
cd /scratch/sp7007/MoT-DAQCNN

# Run the python script
python generate_data.py