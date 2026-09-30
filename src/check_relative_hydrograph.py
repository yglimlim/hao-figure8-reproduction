#!/usr/bin/env python3
"""Sanity-check the DRQ relative hydrograph against CAMELS-BR observations.

This does NOT apply the GRADES discharge prior. It compares the shape only:
- DRQ relative prediction from the released model
- observed CAMELS-BR discharge sampled on the selected Landsat dates

Both are normalized by their own mean for shape comparison.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        default="outputs/drq_relative_predictions.csv",
    )
    parser.add_argument(
        "--obs-zip",
        default="data/metadata/02_CAMELS_BR_streamflow_m3s.zip",
    )
    parser.add_argument(
        "--gauge",
        default="75550000",
    )
    parser.add_argument(
        "--csv-out",
        default="outputs/drq_vs_observed_normalized.csv",
    )
    parser.add_argument(
        "--fig-out",
        default="outputs/drq_vs_observed_normalized.png",
    )
    args = parser.parse_args()

    pred = pd.read_csv(args.pred)
    pred["date"] = pd.to_datetime(pred["date"])

    with zipfile.ZipFile(args.obs_zip) as z:
        matches = [n for n in z.namelist() if args.gauge in n and n.endswith(".txt")]
        if not matches:
            raise FileNotFoundError(f"Gauge {args.gauge} not found in {args.obs_zip}")
        obs = pd.read_csv(
            io.StringIO(z.read(matches[0]).decode()),
            sep=r"\s+",
        )

    obs["date"] = pd.to_datetime(obs[["year", "month", "day"]])
    obs = obs[["date", "streamflow_m3s"]].dropna(subset=["streamflow_m3s"])

    merged = pred.merge(obs, on="date", how="left")
    matched = merged.dropna(subset=["streamflow_m3s"]).copy()

    if matched.empty:
        raise RuntimeError("No prediction dates matched observed streamflow dates.")

    matched["drq_norm"] = matched["drq_relative"] / matched["drq_relative"].mean()
    matched["obs_norm"] = matched["streamflow_m3s"] / matched["streamflow_m3s"].mean()

    corr = np.corrcoef(matched["drq_norm"], matched["obs_norm"])[0, 1]

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    matched.to_csv(csv_out, index=False)

    fig_out = Path(args.fig_out)
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.plot(
        matched["date"],
        matched["obs_norm"],
        linewidth=1.2,
        label="Observed (normalized)",
    )
    ax.plot(
        matched["date"],
        matched["drq_norm"],
        linewidth=1.2,
        linestyle="--",
        label="DRQ / This study (normalized)",
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Discharge / mean discharge")
    ax.set_title("CAMELS-BR 75550000: relative hydrograph sanity check")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=180)
    plt.close(fig)

    print("Prediction dates:", len(pred))
    print("Matched observed dates:", len(matched))
    print("Observed mean on matched dates:", matched["streamflow_m3s"].mean(), "m3/s")
    print("DRQ relative mean on matched dates:", matched["drq_relative"].mean())
    print("Pearson correlation (shape):", corr)
    print("Saved CSV:", csv_out)
    print("Saved figure:", fig_out)


if __name__ == "__main__":
    main()
