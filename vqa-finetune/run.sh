#!/bin/bash
#SBATCH --job-name=lora_without_decoder
#SBATCH --gres gpu:0
#SBATCH --nodes 1
#SBATCH --cpus-per-task=12

# Optionally print an initial snapshot of the GPU state
nvidia-smi

# Start continuous GPU monitoring every 1 second, outputting to a log file.
nvidia-smi -l 1 > gpu_usage.log &
GPU_MONITOR_PID=$!

# Run your training script
python /home/abdelrahman.elsayed/GP/vqa-finetune/finetuning.py

# Once the training script is done, kill the GPU monitoring process
kill $GPU_MONITOR_PID
