#!/bin/bash

# For more information regarding SLURM sbatch options
# and how to set the properties for your job, please
# refer to the SLURM documentation:
#   https://slurm.schedmd.com/pdfs/summary.pdf
#   https://slurm.schedmd.com/documentation.html

#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
1
# The anaconda module provides Python
module load anaconda/anaconda-2023.09

# Activate one of the pre-made ARCC environments (or
# your own environment) on top of the base environment.
conda activate ~/envs/ml

# Run a Python script
python ~/pendulumRNN/scripts/classification/mambaTimeSeriesMassClassification.py --damping 0.5