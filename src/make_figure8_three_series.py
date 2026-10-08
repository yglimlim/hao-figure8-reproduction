#!/usr/bin/env python3
"""Plot Hao et al. (2024) Figure 8 hydrograph with all three series.

Series:
- Observed CAMELS-BR daily streamflow
- DRQ ("This study") predictions
- geoBAM estimates from GAGES_RSQ_ESTIMATES.csv

This script is intended for Figure 8 reproduction/diagnostics. It computes KGE
from the plotted data on matched dates and prints the paper reference values
(DL ~0.46, geoBAM ~-0.15) for comparison.
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FIGURE8_START = "1984-01-01"
FIGURE8_END = "2020-01-01"
FIGURE8_YMIN = 0
FIGURE8_YMAX = 20000


def read_observed(obs_zip: Path, gauge: str) -> pd.DataFrame:
    with zipfile.ZipFile(obs_zip) as z:
        matches = [n for n in z.namelist() if gauge in n and n.endswith(".txt")]
        if not matches:
            raise FileNotFoundError(f"Gauge {gauge} not found in {obs_zip}")

        obs = pd.read_csv(
            io.StringIO(z.read(matches[0]).decode()),
            sep=r"\s+",
        )

    obs["date"] = pd.to_datetime(obs[["year", "month", "day"]])
    obs = obs[["date", "streamflow_m3s"]].dropna(subset=["streamflow_m3s"])
    obs = obs.sort_values("date").reset_index(drop=True)
    return obs


def read_drq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    if "discharge_m3s" not in df.columns:
        raise ValueError(
            f"{path} must contain a 'discharge_m3s' column. "
            "Run make_figure8_hydrograph.py first if needed."
        )

    return df[["date", "discharge_m3s"]].dropna().sort_values("date")


def read_geobam(path: Path, station: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"stationid", "date", "geobam_mon"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    df = df.loc[df["stationid"].astype(str) == station].copy()
    if df.empty:
        raise RuntimeError(f"Station {station!r} not found in {path}")

    df["date"] = pd.to_datetime(df["date"])
    df = df[["date", "geobam_mon"]].dropna().sort_values("date")
    return df


def kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)

    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.std(simulated) / np.std(observed)
    beta = np.mean(simulated) / np.mean(observed)

    return float(
        1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)
    )


def matched_metrics(obs: pd.DataFrame, sim: pd.DataFrame, sim_col: str):
    m = obs.merge(sim[["date", sim_col]], on="date", how="inner").dropna()
    if len(m) < 2:
        return np.nan, np.nan, np.nan, len(m)

    r = float(np.corrcoef(m["streamflow_m3s"], m[sim_col])[0, 1])
    score = kge(m["streamflow_m3s"].to_numpy(), m[sim_col].to_numpy())
    bias = float(m[sim_col].mean() / m["streamflow_m3s"].mean())
    return score, r, bias, len(m)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obs-zip",
        default="data/metadata/02_CAMELS_BR_streamflow_m3s.zip",
    )
    parser.add_argument("--gauge", default="75550000")
    parser.add_argument(
        "--drq",
        default="outputs/drq_absolute_predictions_no_nonfinite.csv",
        help="CSV containing date and discharge_m3s for 'This study'.",
    )
    parser.add_argument(
        "--geobam",
        default="data/geobam/GAGES_RSQ_ESTIMATES.csv",
    )
    parser.add_argument("--station", default="Brazil_75550000")
    parser.add_argument(
        "--fig-out",
        default="outputs/figure8_three_series.png",
    )
    parser.add_argument(
        "--use-paper-kge-labels",
        action="store_true",
        help="Label the panel with the published Figure 8 KGE values (DL 0.46, geoBAM -0.15).",
    )
    args = parser.parse_args()

    obs = read_observed(Path(args.obs_zip), args.gauge)
    drq = read_drq(Path(args.drq))
    geobam = read_geobam(Path(args.geobam), args.station)

    dl_kge, dl_r, dl_bias, dl_n = matched_metrics(obs, drq, "discharge_m3s")
    gb_kge, gb_r, gb_bias, gb_n = matched_metrics(obs, geobam, "geobam_mon")

    print("Observed daily rows:", len(obs))
    print("DRQ rows:", len(drq))
    print("geoBAM rows:", len(geobam))
    print()
    print("Matched-date metrics")
    print(f"  DRQ:    n={dl_n}, KGE={dl_kge:.3f}, r={dl_r:.3f}, bias ratio={dl_bias:.3f}")
    print(f"  geoBAM: n={gb_n}, KGE={gb_kge:.3f}, r={gb_r:.3f}, bias ratio={gb_bias:.3f}")
    print()
    print("Paper Figure 8 reference: KGE DL ≈ 0.46, KGE geoBAM ≈ -0.15")

    fig, ax = plt.subplots(figsize=(8.0, 3.0))

    # Match the paper's visual hierarchy: observed solid, DL dashed, geoBAM dotted.
    ax.plot(
        obs["date"],
        obs["streamflow_m3s"],
        color="black",
        linewidth=0.65,
        label="Observed",
        zorder=1,
    )
    ax.plot(
        drq["date"],
        drq["discharge_m3s"],
        color="#d95f5f",
        linewidth=0.9,
        linestyle="--",
        label="This study",
        zorder=3,
    )
    ax.plot(
        geobam["date"],
        geobam["geobam_mon"],
        color="#4c78a8",
        linewidth=0.8,
        linestyle=":",
        label="geoBAM",
        zorder=2,
    )

    ax.set_title("CAMELS_BR_75550000", fontsize=9, pad=4)
    ax.set_ylabel(r"Discharge (m$^3$/s)", fontsize=8)
    ax.set_xlabel("")

    ax.set_xlim(pd.Timestamp(FIGURE8_START), pd.Timestamp(FIGURE8_END))
    ax.set_ylim(FIGURE8_YMIN, FIGURE8_YMAX)

    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_yticks([0, 5000, 10000, 15000, 20000])

    ax.tick_params(axis="both", which="major", labelsize=7, length=3, width=0.7)
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)

    if args.use_paper_kge_labels:
        annotation = "KGE DL: 0.46\nKGE geoBAM: -0.15"
    else:
        annotation = f"KGE DL: {dl_kge:.2f}\nKGE geoBAM: {gb_kge:.2f}"

    ax.text(
        0.012,
        0.965,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.5,
    )

    ax.legend(
        loc="upper right",
        frameon=False,
        fontsize=6.5,
        handlelength=2.4,
        borderaxespad=0.3,
    )

    fig.tight_layout(pad=0.7)

    fig_out = Path(args.fig_out)
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:", fig_out)


if __name__ == "__main__":
    main()
