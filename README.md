# Hao et al. (2024) Figure 8 Reproduction

Course target:

> Reproduce **Figure 8 — hydrograph for "This study" only** from  
> Hao et al. (2024), *Remote Sensing of River Discharge From Medium-Resolution Satellite Imagery Based on Deep Learning*.

The target case in Figure 8 is **CAMELS-BR_75550000**.

## Recommended workflow

Run this project on the **UMass Unity HPC cluster**.

Use the existing conda environment:

```bash
module load conda/latest
conda activate ewre
```

Project path on Unity:

```text
/nas/cee-water/cjgleason/young/Class/Remote_Sensing/Hao_Figure8/hao-figure8-reproduction
```

The workflow is:

1. Verify the current environment and pretrained DRQ checkpoint.
2. Use CAMELS-BR gauge **75550000** metadata and observed streamflow.
3. Prepare Landsat 5–9 TOA time-series inputs.
4. Run the pretrained DRQ model.
5. Plot only the predicted hydrograph ("This study").

The official model uses a 20-image time window and a Transformer-based temporal model.

## Current metadata

The Figure 8 target gauge is **75550000** (not 76550000). Existing downloads under `data/metadata` include CAMELS-BR streamflow and GRWL metadata.

## Download pretrained model

If the checkpoint is not already present:

```bash
bash scripts/download_checkpoint.sh
```

Expected file:

```text
checkpoints/DRQ_vCloud_0.01_2024_04_23_17_14.pth
```

## Next milestone

Prepare the Figure 8 input series for **CAMELS-BR_75550000**:

- Landsat 5–9 TOA bands: Blue, Green, Red, NIR, SWIR1, SWIR2, QA
- centered 64 × 64 pixel river image chips
- chronological Landsat time series
- discharge prior needed to convert relative discharge to absolute discharge

Once the inputs are prepared, the released pretrained DRQ model can generate the requested "This study" hydrograph.
