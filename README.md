# Hao et al. (2024) Figure 8 Reproduction

Course target:

> Reproduce **Figure 8 — hydrograph for "This study" only** from  
> Hao et al. (2024), *Remote Sensing of River Discharge From Medium-Resolution Satellite Imagery Based on Deep Learning*.

The target case in Figure 8 is **CAMELS-BR_76550000**.

## Recommended workflow

Run this project on the **UMass Unity HPC cluster** rather than on a laptop.

1. Clone this repository on Unity.
2. Create the Python environment in a work/project directory.
3. Download the authors' pretrained DRQ checkpoint from Zenodo.
4. Prepare the Landsat time-series inputs for CAMELS-BR_76550000.
5. Run the pretrained DRQ model.
6. Plot only the predicted hydrograph ("This study").

The official model uses a 20-image time window and a Transformer-based temporal model.

## Unity setup

From a Unity shell:

```bash
module load conda/latest

git clone https://github.com/yglimlim/hao-figure8-reproduction.git
cd hao-figure8-reproduction

conda env create -f environment.yml
conda activate hao_fig8
```

Do not run model inference on the Unity login node. Request a GPU session first, for example:

```bash
salloc -p gpu-preempt -t 02:00:00 --gpus=1 --mem=16G
```

Then verify that PyTorch sees the GPU:

```bash
python src/check_environment.py
```

## Download pretrained model

```bash
bash scripts/download_checkpoint.sh
```

This downloads:

```text
checkpoints/DRQ_vCloud_0.01_2024_04_23_17_14.pth
```

The checkpoint is released by the paper authors on Zenodo.

## Project layout

```text
hao-figure8-reproduction/
├── README.md
├── environment.yml
├── checkpoints/        # ignored by git
├── data/               # ignored by git
├── outputs/            # ignored by git
├── scripts/
│   ├── download_checkpoint.sh
│   └── gpu_test.sbatch
└── src/
    └── check_environment.py
```

## Next milestone

The next step is to prepare the Figure 8 input series for **CAMELS-BR_76550000**:

- Landsat 5–9 TOA bands: Blue, Green, Red, NIR, SWIR1, SWIR2, QA
- centered 64 × 64 pixel river image chips
- chronological Landsat time series
- discharge prior needed to convert relative discharge to absolute discharge

Once the input data are prepared, the released pretrained DRQ model can generate the "This study" hydrograph.
