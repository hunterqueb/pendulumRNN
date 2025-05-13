#!/bin/bash
#
#
#
#SBATCH --job-name=vleoGen          # appears in `squeue`
#SBATCH --array=0-3                 # 4 array elements
#SBATCH --cpus-per-task=16          # << how many CPU cores you want >>
#SBATCH --mem=20G                   # RAM
#SBATCH --time=5:00:00              # wall‑clock limit
#SBATCH --output=logs/%x_%j.out     # stdout+stderr → logs/gen_vleo_parallel_<jobid>.out

module load anaconda/anaconda-2023.09
conda activate ~/envs/ml

# prevent MKL from overthreading
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

I_LIST=(10 20 30 40 50)
I=${I_LIST[$SLURM_ARRAY_TASK_ID]}

python ~/pendulumRNN/scripts/dataGen/VLEO_datasets_parallel.py \
       --numRandSys 100000 \
       --e 0.01 \
       --i "${I}" \
       --folder "~/pendulumRNN/data/massClassification/VLEO/" \
       --out vleo_i"${I}"_e_0_01 \
       --workers "${SLURM_CPUS_PER_TASK}"
