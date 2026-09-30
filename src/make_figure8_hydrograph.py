#!/usr/bin/env python3
"""Scale DRQ relative predictions with the GRADES Qmean prior and plot the
Hao et al. (2024) Figure 8 "This study" hydrograph.

The plotted values are our reproduced DRQ predictions. Plot styling is matched
to the published Figure 8 as closely as practical so the two hydrographs can be
compared directly: red dashed line, fixed 1984-2020 x-axis, and 0-20,000 m3/s
y-axis.

The GRADES file used here is the downloaded 2000-2009 mean-discharge product.
This is a reproducible prior candidate, but it may not be identical to the
exact long-term GRADES prior used by Hao et al. (2024).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from netCDF4 import Dataset


DEFAULT_COMID = 64053766

# Published Figure 8 plotting window for CAMELS_BR_75550000.
FIGURE8_START = "1984-01-01"
FIGURE8_END = "2020-01-01"
FIGURE8_YMIN = 0
FIGURE8_YMAX = 20000


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

    # Match the visual proportions and axes of Hao et al. (2024), Figure 8.
    fig, ax = plt.subplots(figsize=(8.0, 3.0))

    ax.plot(
        pred["date"],
        pred["discharge_m3s"],
        color="#d95f5f",
        linewidth=1.0,
        linestyle="--",
        label="This study",
        zorder=3,
    )

    ax.set_title("CAMELS_BR_75550000", fontsize=9, pad=4)
    ax.set_ylabel(r"Discharge (m$^3$/s)", fontsize=8)

    # The paper labels the x-axis with calendar years rather than an axis title.
    ax.set_xlabel("")
    ax.set_xlim(pd.Timestamp(FIGURE8_START), pd.Timestamp(FIGURE8_END))
    ax.set_ylim(FIGURE8_YMIN, FIGURE8_YMAX)

    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_yticks([0, 5000, 10000, 15000, 20000])

    ax.tick_params(axis="both", which="major", labelsize=7, length=3, width=0.7)
    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=7,
        handlelength=2.4,
        borderaxespad=0.3,
    )

    # Figure 8 uses a clean white background without grid lines.
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    fig.tight_layout(pad=0.7)
    fig.savefig(fig_out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("COMID:", args.comid)
    print("GRADES Qmean:", qmean, units)
    print("Prediction count:", len(pred))
    print("Absolute mean:", float(pred["discharge_m3s"].mean()), "m3/s")
    print("Absolute max :", float(pred["discharge_m3s"].max()), "m3/s")
    print("Saved CSV:", csv_out)
    print("Saved figure:", fig_out)
    print()
    print("Plot style matched to Hao et al. Figure 8:")
    print("  This study = red dashed line")
    print("  x-axis = 1984-2020, 4-year ticks")
    print("  y-axis = 0-20,000 m3/s, 5,000 m3/s ticks")
    print()
    print("NOTE: This scaling uses the downloaded GRADES 2000-2009 Qmean product.")
    print(
        "It is a reproducible prior candidate, not yet proven identical to "
        "the exact Hao et al. Figure 8 prior."
    )


if __name__ == "__main__":
    main()
