#!/usr/bin/env python3
"""Reproduce the Hao et al. (2024) Figure 8 hydrograph panel.

This version follows the published Figure 8 structure:

* Observed = the complete daily CAMELS-BR discharge record (black solid).
* This study = reproduced DRQ/Landsat estimates (red dashed).
* geoBAM = the public monthly geoBAM series (blue dotted).
* GRADES Qmean = horizontal blue reference line used as the long-term
  discharge prior.
* Four large red markers = the four Landsat dates shown as image examples
  underneath Figure 8: 1984-08-04, 1996-03-23, 2006-05-25, 2017-06-11.

The exact geoBAM baseline used by the paper is not proven to be identical to
our public GAGES_RSQ_ESTIMATES.csv series, so that remains a data-source
limitation rather than something to hide with plotting adjustments.
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
from netCDF4 import Dataset


FIGURE8_START = pd.Timestamp("1984-01-01")
FIGURE8_END = pd.Timestamp("2020-01-01")
FIGURE8_YMIN = 0
FIGURE8_YMAX = 20000

# Dates printed under the four example Landsat images in Figure 8.
FIGURE8_IMAGE_DATES = pd.to_datetime(
    ["1984-08-04", "1996-03-23", "2006-05-25", "2017-06-11"]
)


def read_observed(obs_zip: Path, gauge: str) -> pd.DataFrame:
    """Read the complete daily CAMELS-BR discharge record for one gauge."""
    with zipfile.ZipFile(obs_zip) as z:
        matches = [
            n for n in z.namelist()
            if gauge in n and n.endswith(".txt")
        ]
        if not matches:
            raise FileNotFoundError(f"Gauge {gauge} not found in {obs_zip}")

        member = sorted(matches, key=len)[0]
        obs = pd.read_csv(
            io.StringIO(z.read(member).decode()),
            sep=r"\s+",
        )

    needed = {"year", "month", "day", "streamflow_m3s"}
    missing = needed.difference(obs.columns)
    if missing:
        raise ValueError(
            f"{member} is missing required columns: {sorted(missing)}"
        )

    obs["date"] = pd.to_datetime(obs[["year", "month", "day"]])
    obs["streamflow_m3s"] = pd.to_numeric(
        obs["streamflow_m3s"], errors="coerce"
    )
    obs.loc[obs["streamflow_m3s"] < 0, "streamflow_m3s"] = np.nan

    return (
        obs[["date", "streamflow_m3s"]]
        .dropna(subset=["streamflow_m3s"])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def read_drq(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    if "discharge_m3s" not in df.columns:
        raise ValueError(
            f"{path} must contain 'discharge_m3s'. "
            "Run make_figure8_hydrograph.py first."
        )

    df["discharge_m3s"] = pd.to_numeric(
        df["discharge_m3s"], errors="coerce"
    )
    return (
        df[["date", "discharge_m3s"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


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
    df["geobam_mon"] = pd.to_numeric(df["geobam_mon"], errors="coerce")
    return (
        df[["date", "geobam_mon"]]
        .dropna()
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )


def read_qmean(grades_path: Path, comid: int) -> tuple[float, str]:
    """Read the GRADES Qmean prior used to scale the DRQ prediction."""
    with Dataset(grades_path) as ds:
        ids = np.asarray(ds.variables["rivid"][:])
        q = np.asarray(ds.variables["Qmean"][:])
        idx = np.where(ids == comid)[0]
        if len(idx) == 0:
            raise RuntimeError(
                f"COMID {comid} not found in {grades_path}"
            )
        value = float(q[idx[0]])
        units = getattr(ds.variables["Qmean"], "units", "")
    return value, units


def kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)
    if len(observed) < 2:
        return np.nan

    obs_std = np.std(observed)
    if obs_std == 0:
        return np.nan

    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.std(simulated) / obs_std
    beta = np.mean(simulated) / np.mean(observed)

    return float(
        1.0 - np.sqrt(
            (r - 1.0) ** 2
            + (alpha - 1.0) ** 2
            + (beta - 1.0) ** 2
        )
    )


def matched_metrics(
    obs: pd.DataFrame,
    sim: pd.DataFrame,
    sim_col: str,
):
    m = obs.merge(sim[["date", sim_col]], on="date", how="inner").dropna()
    if len(m) < 2:
        return np.nan, np.nan, np.nan, len(m)

    r = float(
        np.corrcoef(
            m["streamflow_m3s"],
            m[sim_col],
        )[0, 1]
    )
    score = kge(
        m["streamflow_m3s"].to_numpy(),
        m[sim_col].to_numpy(),
    )
    bias = float(
        m[sim_col].mean() / m["streamflow_m3s"].mean()
    )
    return score, r, bias, len(m)


def nearest_values(
    series: pd.DataFrame,
    dates: pd.DatetimeIndex,
    value_col: str,
) -> pd.DataFrame:
    """Get the nearest available prediction for the four displayed dates."""
    s = series.set_index("date")[value_col].sort_index()
    out = []
    for d in dates:
        if s.empty:
            continue
        idx = np.argmin(np.abs(s.index.values - np.datetime64(d)))
        actual_date = s.index[idx]
        out.append(
            {
                "target_date": d,
                "date": actual_date,
                value_col: float(s.iloc[idx]),
            }
        )
    return pd.DataFrame(out)


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
    )
    parser.add_argument(
        "--geobam",
        default="data/geobam/GAGES_RSQ_ESTIMATES.csv",
    )
    parser.add_argument("--station", default="Brazil_75550000")
    parser.add_argument(
        "--grades",
        default="data/grades/GRADES_Qmean_20000101_20091231.nc",
    )
    parser.add_argument("--comid", type=int, default=64053766)
    parser.add_argument(
        "--fig-out",
        default="outputs/figure8_three_series_paper_style.png",
    )
    parser.add_argument(
        "--use-paper-kge-labels",
        action="store_true",
        help="Display the published Figure 8 KGE labels (0.46 and -0.15).",
    )
    args = parser.parse_args()

    obs = read_observed(Path(args.obs_zip), args.gauge)
    drq = read_drq(Path(args.drq))
    geobam = read_geobam(Path(args.geobam), args.station)
    qmean, qmean_units = read_qmean(Path(args.grades), args.comid)

    dl_kge, dl_r, dl_bias, dl_n = matched_metrics(
        obs, drq, "discharge_m3s"
    )
    gb_kge, gb_r, gb_bias, gb_n = matched_metrics(
        obs, geobam, "geobam_mon"
    )

    # The four large red markers in the published Figure 8 correspond to the
    # four example satellite images printed below the hydrograph.
    marker_rows = nearest_values(
        drq,
        FIGURE8_IMAGE_DATES,
        "discharge_m3s",
    )

    print("Gauge:", args.gauge)
    print("Observed daily rows:", len(obs))
    print("DRQ rows:", len(drq))
    print("Public geoBAM rows:", len(geobam))
    print("GRADES Qmean:", qmean, qmean_units)
    print()
    print("Metrics using all matching dates:")
    print(
        f"  DRQ:    n={dl_n}, KGE={dl_kge:.3f}, "
        f"r={dl_r:.3f}, bias ratio={dl_bias:.3f}"
    )
    print(
        f"  geoBAM: n={gb_n}, KGE={gb_kge:.3f}, "
        f"r={gb_r:.3f}, bias ratio={gb_bias:.3f}"
    )
    print()
    print("Figure 8 image marker dates:")
    print(marker_rows.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7.2, 2.55))

    # IMPORTANT: observed is the complete daily series.  The previous version
    # incorrectly sampled it only on Landsat/DRQ dates, which created the
    # visibly sparse zig-zag black line.
    ax.plot(
        obs["date"],
        obs["streamflow_m3s"],
        color="black",
        linewidth=0.42,
        linestyle="-",
        label="Observed",
        zorder=1,
        rasterized=True,
    )

    # DRQ / This study.
    ax.plot(
        drq["date"],
        drq["discharge_m3s"],
        color="#d62728",
        linewidth=0.78,
        linestyle="--",
        label="This study",
        zorder=4,
    )

    # Four large red points marking the satellite-image examples shown below.
    if not marker_rows.empty:
        ax.scatter(
            marker_rows["date"],
            marker_rows["discharge_m3s"],
            s=22,
            color="#d62728",
            edgecolors="none",
            zorder=6,
        )

    # Public geoBAM should be drawn over its full monthly record rather than
    # only on DRQ dates. This gives the long, relatively smooth dotted curve
    # seen in the paper.
    ax.plot(
        geobam["date"],
        geobam["geobam_mon"],
        color="#5f86ad",
        linewidth=0.75,
        linestyle=":",
        label="geoBAM",
        zorder=3,
    )

    # GRADES long-term mean discharge prior: the horizontal blue reference line
    # visible in the paper's hydrograph. It is deliberately not included in
    # the legend because the paper's legend lists only the three series.
    ax.axhline(
        qmean,
        color="#5f86ad",
        linewidth=0.72,
        linestyle="-",
        zorder=2,
    )

    ax.set_title(
        "CAMELS_BR_75550000",
        fontsize=8.2,
        pad=3.0,
    )
    ax.set_ylabel(
        r"Discharge (m$^3$/s)",
        fontsize=7.5,
        labelpad=3,
    )
    ax.set_xlabel("")

    ax.set_xlim(FIGURE8_START, FIGURE8_END)
    ax.set_ylim(FIGURE8_YMIN, FIGURE8_YMAX)

    ax.xaxis.set_major_locator(
        mdates.YearLocator(base=4, month=1, day=1)
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_yticks([0, 5000, 10000, 15000, 20000])

    ax.tick_params(
        axis="both",
        which="major",
        labelsize=6.5,
        length=2.5,
        width=0.65,
        pad=1.5,
    )
    ax.grid(False)

    for spine in ax.spines.values():
        spine.set_linewidth(0.65)

    if args.use_paper_kge_labels:
        annotation = "KGE DL: 0.46\nKGE geoBAM: -0.15"
    else:
        annotation = (
            f"KGE DL: {dl_kge:.2f}\n"
            f"KGE geoBAM: {gb_kge:.2f}"
        )

    ax.text(
        0.012,
        0.955,
        annotation,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.2,
        linespacing=1.05,
        zorder=10,
    )

    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        framealpha=1.0,
        facecolor="white",
        edgecolor="black",
        fontsize=6.2,
        handlelength=2.6,
        handletextpad=0.5,
        borderpad=0.28,
        labelspacing=0.18,
        borderaxespad=0.35,
    )
    legend.get_frame().set_linewidth(0.55)

    fig.subplots_adjust(
        left=0.105,
        right=0.99,
        bottom=0.20,
        top=0.88,
    )

    fig_out = Path(args.fig_out)
    fig_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        fig_out,
        dpi=400,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)

    print("Saved figure:", fig_out)
    print()
    print("Observed: complete daily CAMELS-BR record.")
    print("This study: red dashed DRQ predictions.")
    print("Red markers: four Figure 8 satellite-image dates.")
    print("geoBAM: complete public monthly series.")
    print("Blue horizontal line: GRADES Qmean prior.")
    print(
        "NOTE: the public geoBAM archive is not proven to be the exact "
        "Figure 8 baseline series used by Hao et al."
    )


if __name__ == "__main__":
    main()
