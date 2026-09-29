# Project instructions for Codex

## Goal

Reproduce the course assignment based on Hao et al. (2024), *Remote Sensing of River Discharge From Medium-Resolution Satellite Imagery Based on Deep Learning*.

The assignment target is:

- Figure 8
- Reproduce **only the hydrograph for "This study"**
- Target case: **CAMELS-BR_76550000**
- Do **not** reproduce geoBAM, the satellite image panels, or the full published Figure 8 unless explicitly asked.

The desired final product is a clean hydrograph of the pretrained DRQ model prediction ("This study"), with observed discharge included only when useful for comparison.

## Compute location

Work on the UMass Unity/CEE system.

Repository path on Unity:

`/nas/cee-water/cjgleason/young/Class/Remote_Sensing/Hao_Figure8/hao-figure8-reproduction`

Use the user's existing conda environment:

`ewre`

Do **not** create a new conda environment unless a verified dependency conflict makes it necessary and the user approves it.

The user is often on a CPU node such as `ceewater-cpu006`. A false result from `torch.cuda.is_available()` on a CPU node is not by itself an error. Use an appropriate GPU compute session for model inference when needed.

## Current state

- Repository exists and is cloned on Unity.
- Existing environment `ewre` is active.
- PyTorch has been installed into `ewre`.
- README, scripts, and basic project scaffolding already exist.
- Check whether the pretrained checkpoint is present before downloading it again.

## Author resources

Official model/code repository:

`https://github.com/haozhen315/DRQ-Deep-learning-based-Remote-sensing-of-Discharge`

Paper code/model release:

`https://zenodo.org/records/12747217`

Expected pretrained checkpoint filename:

`DRQ_vCloud_0.01_2024_04_23_17_14.pth`

The author's reference environment is approximately:

- Python 3.9.12
- torch 2.0.1 + CUDA 11.8
- rasterio 1.3.6
- numpy 1.23.5

Do not force exact versions if the existing `ewre` environment works. Prefer the smallest change needed.

## Model/data facts relevant to reproduction

The released DRQ code:

- takes a chronological time series of Landsat image chips,
- uses 20 images per model window,
- uses 64 x 64 pixel centered image chips,
- expects 8 channels in the model input:
  - Blue
  - Green
  - Red
  - NIR
  - SWIR1
  - SWIR2
  - SRTM placeholder/channel
  - AWEI derived from spectral bands
- predicts relative discharge variation,
- stitches overlapping 20-image windows,
- converts relative predictions to absolute discharge by multiplying by a long-term mean discharge prior.

The paper uses Landsat 5-9 TOA reflectance. The target hydrograph in Figure 8 corresponds to CAMELS-BR gauge `76550000`.

Relevant public data sources named by the paper include:

- Landsat TOA imagery via Google Earth Engine
- Caravan / CAMELS-BR gauge data
- GRADES discharge prior
- GRWL river location/width information

Do not invent missing data. Verify the exact source, gauge metadata, dates, and prior before using them.

## Working approach

Proceed in small, verifiable stages:

1. Inspect the repository and current environment.
2. Verify Python, NumPy, rasterio, PyTorch, and checkpoint availability.
3. Inspect the authors' official code before adapting it.
4. Identify and obtain the exact metadata/location for CAMELS-BR_76550000.
5. Prepare the required Landsat 5-9 TOA time series in chronological order.
6. Verify band order, image dimensions, timestamps, cloud filtering, and river centering.
7. Run the released pretrained DRQ model.
8. Convert relative predictions to discharge using the appropriate long-term mean prior.
9. Plot only the requested "This study" hydrograph.
10. Compare the reproduced hydrograph qualitatively with the published Figure 8 and document any unavoidable discrepancies.

## Safety / reproducibility rules

- Do not delete existing files or overwrite raw data without a clear reason.
- Do not commit large rasters, model checkpoints, or generated bulk data to Git.
- Keep checkpoints, data, and outputs in their existing ignored directories.
- Prefer scripts over manual one-off transformations so the reproduction is repeatable.
- Preserve dates associated with Landsat observations; do not plot predictions against arbitrary sequence indices if observation dates are available.
- Do not fabricate observed discharge, GRADES priors, or Figure 8 values by digitizing the published plot unless explicitly requested as a fallback.
- If exact Figure 8 inputs cannot be recovered from public sources, document what is missing and reproduce the closest defensible result using the authors' released model and public data.

## Interaction style

The user prefers Codex to do the terminal/debugging work directly rather than repeatedly asking them to copy and paste commands.

When safe:
- inspect outputs,
- run diagnostic commands,
- fix small errors,
- rerun tests,
- and continue to the next logical step.

Ask before:
- deleting data,
- replacing the existing conda environment,
- installing a large or potentially conflicting dependency set,
- or making an irreversible change.

At meaningful milestones, briefly report:
- what worked,
- what failed,
- what changed,
- and what the next step is.
