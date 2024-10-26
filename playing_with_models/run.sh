#!/bin/bash
#SBATCH --job-name=GP
#SBATCH --gres gpu:0
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G

if [ -z "$1" ]; then
  echo "Usage: $0 <model_name>"
  exit 1
fi

model_name=$1

nvidia-smi

python /home/abdelrahman.elsayed/GP/playing_with_models/${model_name}_run.py
