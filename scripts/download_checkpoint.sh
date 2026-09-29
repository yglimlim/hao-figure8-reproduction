#!/bin/bash
set -euo pipefail

mkdir -p checkpoints

URL="https://zenodo.org/records/12747217/files/DRQ_vCloud_0.01_2024_04_23_17_14.pth?download=1"
OUT="checkpoints/DRQ_vCloud_0.01_2024_04_23_17_14.pth"

echo "Downloading Hao et al. pretrained DRQ checkpoint..."
wget -O "$OUT" "$URL"

echo
echo "Saved checkpoint to:"
echo "$OUT"
