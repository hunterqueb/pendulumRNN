#!/bin/bash
#
#
#
#SBATCH --job-name=orbitMassClass
#SBATCH --array=0-1%2                  # two elements → layer 1 and layer 2 only
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out

module load anaconda/anaconda-2023.09
conda activate ~/envs/ml

# ----- fixed damping list (single value) -----
layer_values=(1 2)

layer_idx=$(( SLURM_ARRAY_TASK_ID ))  # 0 → layer 1, 1 → layer 2

LAYERS=${layer_values[$layer_idx]}

python ~/pendulumRNN/scripts/classification/mambaVLEOMassClassification.py \
       --transformer \
       --layers  "${LAYERS}"

