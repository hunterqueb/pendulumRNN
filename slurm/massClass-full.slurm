#!/bin/bash
#
#
#SBATCH --job-name=massClass
#SBATCH --array=0-17%4                # 9 damping values × 2 layer options = 18 elements - with 4 concurrent jobs
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:4
#SBATCH --output=logs/%x_%A_%a.out  # separate stdout/stderr per element

module load anaconda/anaconda-2023.09
conda activate ~/envs/ml

# ───────────── map array index → (damping, layers) ──────────────
damp_values=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
layer_values=(1 2)

damp_idx=$(( SLURM_ARRAY_TASK_ID / 2 ))
layer_idx=$(( SLURM_ARRAY_TASK_ID % 2 ))

DAMPING=${damp_values[$damp_idx]}
LAYERS=${layer_values[$layer_idx]}

# ───────────── run the model ─────────────────────────────────────
python ~/pendulumRNN/scripts/classification/mambaTimeSeriesMassClassification.py \
       --damping "${DAMPING}" \
       --layers  "${LAYERS}"
