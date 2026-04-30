#!/bin/bash -l
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=18:00:00
#SBATCH --output=/common/home/dks134/hackprinceton-2026/logs/%x_%j.out
#SBATCH --error=/common/home/dks134/hackprinceton-2026/logs/%x_%j.err

if [ -z "$MODEL_SIZE" ]; then
    echo "ERROR: MODEL_SIZE not set."
    exit 1
fi

echo "Model: $MODEL_SIZE | Job: $SLURM_JOB_ID | Node: $SLURMD_NODENAME | Start: $(date)"

mkdir -p /common/home/dks134/hackprinceton-2026/logs
cd /common/home/dks134/hackprinceton-2026

source /common/home/dks134/hackprinceton/bin/activate

python3 -u probing.py --model_size "$MODEL_SIZE" --permute_n 0 --cv_repeats 1 --shuffle_splits 20

echo "Done: $(date)"