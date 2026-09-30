#!/bin/bash
#SBATCH -p gpu-preempt
#SBATCH -t 01:00:00
#SBATCH --gpus=1
#SBATCH --mem=24G
#SBATCH --cpus-per-task=4
#SBATCH -J hao_fig8_drq
#SBATCH -o outputs/drq-%j.out
#SBATCH -e outputs/drq-%j.err

set -euo pipefail

cd /nas/cee-water/cjgleason/young/Class/Remote_Sensing/Hao_Figure8/hao-figure8-reproduction
mkdir -p outputs

module load conda/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ewre

echo "Host: $(hostname)"
python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

python src/run_drq_inference.py
