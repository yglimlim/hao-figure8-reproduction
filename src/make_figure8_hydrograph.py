#!/usr/bin/env python3
"""Scale DRQ relative predictions with the GRADES Qmean prior and plot Figure 8 hydrograph.

The GRADES file used here is the downloaded 2000-2009 mean-discharge product.
This is a reproducible prior candidate, but it may not be identical to the
exact long-term GRADES prior used by Hao et al. (2024).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset


DEFAULT_COMID = 64053766


def read_qmean(nc_path: Path, comid: int) -> tuple[float, str]:
    with Dataset(nc_path) as ds:
        ids = ds.variables["rivid"][:]
        q = ds.variables["Qmean"][:]
        idx = np.where(ids == comid)[0]
        if len(idx) == 0:
            raise RuntimeError(f"COMID {comid} not found in {nc_path}")
        value = float(q[idx[0]])
        units = getattr(ds.variables["Qmean"], "units", "")
    return value, units


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pred",
        default="outputs/drq_relative_predictions.csv",
    )
    parser.add_argument(
        "--grades",
        default="data/grades/GRADES_Qmean_20000101_20091231.nc",
    )
    parser.add_argument("--comid", type=int, default=DEFAULT_COMID)
    parser.add_argument(
        "--csv-out",
        default="outputs/drq_absolute_predictions.csv",
    )
    parser.add_argument(
        "--fig-out",
        default="outputs/figure8_this_study.png",
    )
    args = parser.parse_args()

    pred_path = Path(args.pred)
    grades_path = Path(args.grades)

    pred = pd.read_csv(pred_path)
    pred["date"] = pd.to_datetime(pred["date"])

    qmean, units = read_qmean(grades_path, args.comid)
    pred["discharge_m3s"] = pred["drq_relative"] * qmean

    csv_out = Path(args.csv_out)
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    pred.to_csv(csv_out, index=False)

    fig_out = Path(args.fig_out)
    fig_out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11.5, 4.8))
    ax.plot(
        pred["date"],
        pred["discharge_m3s"],
        linewidth=1.3,
        linestyle="--",
        label="This study",
    )
    ax.set_xlabel("Year")
    ax.set_ylabel("Discharge (m³/s)")
    ax.set_title("CAMELS-BR 75550000")
    ax.legend(frameon=False)
    ax.grid(alpha=0.2)
    ax.set_xlim(pred["date"].min(), pred["date"].max())
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(fig_out, dpi=220)
    plt.close(fig)

    print("COMID:", args.comid)
    print("GRADES Qmean:", qmean, units)
    print("Prediction count:", len(pred))
    print("Absolute mean:", float(pred["discharge_m3s"].mean()), "m3/s")
    print("Absolute max :", float(pred["discharge_m3s"].max()), "m3/s")
    print("Saved CSV:", csv_out)
    print("Saved figure:", fig_out)
    print()
    print("NOTE: This scaling uses the downloaded GRADES 2000-2009 Qmean product.")
    print("It is a reproducible prior candidate, not yet proven identical to the exact Hao et al. Figure 8 prior.")


if __name__ == "__main__":
    main()
