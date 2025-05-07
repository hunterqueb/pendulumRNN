#!/bin/bash
#
#
#SBATCH --job-name=massClass-VLEO
#SBATCH --cpus-per-task=24
#SBATCH --mem=4G
#SBATCH --time=5:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=logs/%x_%A_%a.out

module load anaconda/anaconda-2023.09
conda activate ~/envs/ml

python ~/pendulumRNN/scripts/classification/mambaVLEOMassClassification.py \
       --transformer \
       --numRandSys 100000 \
       --e 0.01 \
       --i 10