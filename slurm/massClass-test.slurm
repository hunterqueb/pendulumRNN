#!/bin/bash
#
# – sweep: damping = 0.1, layers ∈ {1, 2}
#
#SBATCH --job-name=massClass_test
#SBATCH --array=0-1%2                  # two elements → layer 1 and layer 2 only
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:2
#SBATCH --output=logs/%x_%A_%a.out

module load anaconda/anaconda-2023.09
conda activate ~/envs/ml

# ----- fixed damping list (single value) -----
damp_values=(0.1)     #  ←–– only 0.1
layer_values=(1 2)

damp_idx=0                            # always 0 because there is only one damping value
layer_idx=$(( SLURM_ARRAY_TASK_ID ))  # 0 → layer 1, 1 → layer 2

DAMPING=${damp_values[$damp_idx]}     # = 0.1
LAYERS=${layer_values[$layer_idx]}

python ~/pendulumRNN/scripts/classification/mambaTimeSeriesMassClassification.py \
       --damping "${DAMPING}" \
       --layers  "${LAYERS}"
