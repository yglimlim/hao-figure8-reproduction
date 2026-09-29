#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH -t 00:30:00
#SBATCH --gpus=1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH -J hao_fig8_gpu_test
#SBATCH -o outputs/gpu_test-%j.out

set -euo pipefail

mkdir -p outputs
module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate hao_fig8

python src/check_environment.py
