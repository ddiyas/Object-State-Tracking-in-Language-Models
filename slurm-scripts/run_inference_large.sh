#!/bin/bash -l
#SBATCH -G 4
#SBATCH --cpus-per-task=4
#SBATCH --mem=80g
#SBATCH --time=04:00:00
#SBATCH --output=/common/home/dks134/hackprinceton-2026/logs/%j.out
#SBATCH --error=/common/home/dks134/hackprinceton-2026/logs/%j.err

mkdir -p /common/home/dks134/hackprinceton-2026/logs
cd /common/home/dks134/hackprinceton-2026

source /common/home/dks134/hackprinceton/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HOME=/common/users/dks134/hf_cache
export HUGGINGFACE_HUB_CACHE=/common/users/dks134/hf_cache
export TRANSFORMERS_CACHE=/common/users/dks134/hf_cache

python3 inference.py $1