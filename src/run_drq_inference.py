#!/usr/bin/env python3
"""Run the released Hao et al. DRQ checkpoint on the downloaded Landsat chips.

This first pass uses q_mean=1.0, so the saved prediction is a relative
hydrograph shape. Absolute m3/s scaling will be applied after the GRADES
long-term mean discharge for gauge 75550000 is identified.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from tqdm import tqdm

from hao_model import DRQModel


def handle_nan(arr):
    arr = np.asarray(arr)
    arr[np.isnan(arr)] = -1
    arr[arr == -float("inf")] = -1
    arr[np.isinf(arr)] = -1
    return arr


def awei_index(blue, green, nir, swir1, swir2):
    return blue + 2.5 * green - 1.5 * (nir + swir1) - 0.25 * swir2


def load_model(checkpoint: Path, device: torch.device):
    model = DRQModel(
        in_channels=8,
        num_days=20,
        img_size=64,
        num_features=384,
        depth=6,
    )

    state_dict = torch.load(checkpoint, map_location=device)
    clean = OrderedDict()
    for key, value in state_dict.items():
        clean[key[7:] if key.startswith("module.") else key] = value

    model.load_state_dict(clean)
    model.to(device)
    model.eval()
    return model


def build_input(tif_files):
    days = []

    for tif_file in tqdm(tif_files, desc="Reading Landsat chips"):
        with rasterio.open(tif_file) as src:
            arr = src.read()

        if arr.shape != (7, 64, 64):
            raise RuntimeError(f"{tif_file}: expected (7,64,64), found {arr.shape}")

        blue, green, red, nir, swir1, swir2, qa = arr

        blue = handle_nan(blue)
        green = handle_nan(green)
        red = handle_nan(red)
        nir = handle_nan(nir)
        swir1 = handle_nan(swir1)
        swir2 = handle_nan(swir2)

        srtm = np.zeros_like(blue)
        awei = awei_index(blue, green, nir, swir1, swir2)

        day = np.stack(
            [blue, green, red, nir, swir1, swir2, srtm, awei],
            axis=0,
        )
        days.append(day)

    x = np.stack(days, axis=0).astype(np.float32)
    x = handle_nan(x)
    return torch.from_numpy(x).unsqueeze(0)


def transform_pred(pred):
    max_log_of_ratio = 5.096
    min_log_of_ratio = -13.815
    return np.exp(pred * (max_log_of_ratio - min_log_of_ratio) + min_log_of_ratio)


@torch.inference_mode()
def predict_series(model_input, model, device, q_mean=1.0):
    input_days_length = 20
    overlapping_days_length = 19
    step_size = 1

    x = model_input.to(device)
    length = x.shape[1]

    if length < input_days_length:
        raise ValueError(f"Need at least 20 images; found {length}")

    pred_qs = [[] for _ in range(length)]
    prev_pred = None

    starts = range(0, length - input_days_length + 1, step_size)
    for i in tqdm(starts, desc="DRQ sliding windows"):
        tmp = x[:, i:i + input_days_length]
        pred = model(tmp).detach().cpu().numpy().squeeze()
        pred = transform_pred(pred)

        if prev_pred is not None:
            overlap = prev_pred[-overlapping_days_length:]
            denom = pred[:overlapping_days_length].mean()
            scale_ratio = overlap.mean() / denom if denom != 0 else 1.0
            pred *= scale_ratio

        prev_pred = pred.copy()

        for j, value in enumerate(pred):
            pred_qs[i + j].append(value)

    out = np.array([np.mean(values) for values in pred_qs], dtype=float)
    return out * q_mean


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/metadata/landsat_usable_scenes.csv")
    parser.add_argument("--chips", default="data/landsat_chips")
    parser.add_argument("--checkpoint", default="checkpoints/DRQ_vCloud_0.01_2024_04_23_17_14.pth")
    parser.add_argument("--output", default="outputs/drq_relative_predictions.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype=str).sort_values("date").reset_index(drop=True)
    chips_dir = Path(args.chips)

    tif_files = []
    for row in manifest.itertuples(index=False):
        expected = chips_dir / f"{row.date.replace('-', '')}_{row.scene_id}.tif"
        if not expected.exists():
            raise FileNotFoundError(f"Missing chip: {expected}")
        tif_files.append(expected)

    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("CUDA available:", torch.cuda.is_available())
    if device.type != "cuda":
        raise RuntimeError("GPU is required for this inference run. Submit the Slurm GPU job.")

    print("Images:", len(tif_files))
    print("First date:", manifest.loc[0, "date"])
    print("Last date :", manifest.loc[len(manifest)-1, "date"])

    model = load_model(checkpoint, device)
    model_input = build_input(tif_files)

    print("Model input shape:", tuple(model_input.shape))
    pred = predict_series(model_input, model, device, q_mean=1.0)

    out = manifest[["date", "sensor", "scene_id"]].copy()
    out["drq_relative"] = pred

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output, index=False)

    print("Saved:", output)
    print("Prediction count:", len(out))
    print("Relative min:", float(np.nanmin(pred)))
    print("Relative mean:", float(np.nanmean(pred)))
    print("Relative max:", float(np.nanmax(pred)))


if __name__ == "__main__":
    main()
