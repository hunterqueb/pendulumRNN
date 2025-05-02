#!/bin/bash
#
# sweep mass × layers × num_classes
#
#SBATCH --job-name=massClass
#SBATCH --array=0-35            # 9×2×2 = 36 combinations
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/mass/%x_%A_%a.out   # one log file per element

module load anaconda/anaconda-2023.09
conda activate ~/envs/ml

# ───────── parameter lists ─────────
mass_values=(1000 100 10 1)
layer_values=(1 2)
num_classes_values=(5 10)

# lengths (avoid hard‑coding)
nm=${#mass_values[@]}
nl=${#layer_values[@]}
nc=${#num_classes_values[@]}

# total combinations must match --array range
# nm*nl*nc = 9*2*2 = 36  ✓

id=$SLURM_ARRAY_TASK_ID      # shorthand

# ───────── index math ─────────
nc_idx=$((  id                % nc          ))           # 0‑1
layer_idx=$(( (id /  nc)      % nl          ))           # 0‑1
damp_idx=$((  id / (nc*nl) ))                             # 0‑8

DAMPING=${damp_values[$damp_idx]}
LAYERS=${layer_values[$layer_idx]}
NUM_CLASSES=${num_classes_values[$nc_idx]}

echo "Running: damping=$DAMPING  layers=$LAYERS  num_classes=$NUM_CLASSES"

# ───────── run the model ─────────
python ~/pendulumRNN/scripts/classification/mambaTimeSeriesMassClassification.py \
       --damping      "$DAMPING" \
       --layers       "$LAYERS" \
       --num_classes  "$NUM_CLASSES" \
       --transformer
