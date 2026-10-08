#!/usr/bin/env python3
"""Reproduce the Hao et al. (2024) Figure 8 hydrograph panel.

The published panel compares three series for CAMELS-BR gauge 75550000:

- Observed discharge
- DRQ ("This study")
- geoBAM

Important plotting detail:
The DRQ model produces estimates only for usable Landsat observations.  The
published hydrograph therefore compares the observed discharge on those
satellite dates, rather than drawing the entire daily CAMELS-BR record as a
dense black line.  Plotting all daily observations makes the black hydrograph
look substantially different from Figure 8.

By default this script:
1. samples CAMELS-BR observations on the DRQ dates,
2. samples the public geoBAM series on the same DRQ dates when dates overlap,
3. uses the paper-like line styles, legend box, axes, and layout.

The public Lin et al. geoBAM archive is useful for visual comparison but is not
guaranteed to be the exact baseline series used by Hao et al. for Figure 8.
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


FIGURE8_START = pd.Timestamp("1984-01-01")
FIGURE8_END = pd.Timestamp("2020-01-01")
FIGURE8_YMIN = 0
FIGURE8_YMAX = 20000


def read_observed(obs_zip: Path, gauge: str) -> pd.DataFrame:
    """Read the CAMELS-BR daily discharge record for one gauge."""
    with zipfile.ZipFile(obs_zip) as z:
        matches = [
            n for n in z.namelist()
            if gauge in n and n.endswith(".txt")
        ]
        if not matches:
            raise FileNotFoundError(f"Gauge {gauge} not found in {obs_zip}")

        # Prefer the shortest matching path if the archive happens to contain
        # duplicated/derived files.
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

    # CAMELS-style archives may use negative sentinels for missing discharge.
    obs.loc[obs["streamflow_m3s"] < 0, "streamflow_m3s"] = np.nan

    obs = (
        obs[["date", "streamflow_m3s"]]
        .dropna(subset=["streamflow_m3s"])
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )
    return obs


def read_drq(path: Path) -> pd.DataFrame:
    """Read reproduced DRQ absolute predictions."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    if "discharge_m3s" not in df.columns:
        raise ValueError(
            f"{path} must contain a 'discharge_m3s' column. "
            "Run make_figure8_hydrograph.py first if needed."
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
    """Read the public Lin et al. geoBAM time series for one station."""
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


def sample_on_dates(
    source: pd.DataFrame,
    target_dates: pd.DataFrame,
    value_col: str,
) -> pd.DataFrame:
    """Keep source values only on dates present in target_dates."""
    dates = (
        target_dates[["date"]]
        .drop_duplicates()
        .sort_values("date")
    )
    out = dates.merge(
        source[["date", value_col]],
        on="date",
        how="left",
    )
    return out.dropna(subset=[value_col]).reset_index(drop=True)


def kge(observed: np.ndarray, simulated: np.ndarray) -> float:
    observed = np.asarray(observed, dtype=np.float64)
    simulated = np.asarray(simulated, dtype=np.float64)

    if len(observed) < 2:
        return np.nan

    obs_std = np.std(observed)
    sim_std = np.std(simulated)
    if obs_std == 0:
        return np.nan

    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = sim_std / obs_std
    beta = np.mean(simulated) / np.mean(observed)

    return float(
        1.0
        - np.sqrt(
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
    m = obs.merge(
        sim[["date", sim_col]],
        on="date",
        how="inner",
    ).dropna()

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
        m[sim_col].mean()
        / m["streamflow_m3s"].mean()
    )
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
        default="outputs/figure8_three_series_paper_style.png",
    )
    parser.add_argument(
        "--obs-sampled-out",
        default="outputs/observed_on_drq_dates.csv",
    )
    parser.add_argument(
        "--use-paper-kge-labels",
        action="store_true",
        help=(
            "Display the published Figure 8 labels "
            "(DL 0.46, geoBAM -0.15) instead of reproduced metrics."
        ),
    )
    parser.add_argument(
        "--plot-daily-observed",
        action="store_true",
        help=(
            "Diagnostic only: plot the full daily CAMELS-BR record. "
            "Default is the Figure-8-like Landsat-date sampled record."
        ),
    )
    parser.add_argument(
        "--plot-geobam-all-dates",
        action="store_true",
        help=(
            "Diagnostic only: plot all dates in the public geoBAM archive. "
            "Default samples geoBAM on the reproduced DRQ dates."
        ),
    )
    args = parser.parse_args()

    obs_daily = read_observed(Path(args.obs_zip), args.gauge)
    drq = read_drq(Path(args.drq))
    geobam_all = read_geobam(Path(args.geobam), args.station)

    # KEY FIX:
    # Figure 8 is a remote-sensing hydrograph.  The black observed line should
    # be compared on the same usable Landsat dates as the red DRQ estimates,
    # not drawn from every daily CAMELS-BR observation.
    obs_on_drq = sample_on_dates(
        obs_daily,
        drq,
        "streamflow_m3s",
    )

    # For a like-for-like visual comparison, keep the public geoBAM values only
    # on dates also present in our DRQ series.  This does not make the public
    # geoBAM archive identical to Hao et al.'s private Figure 8 baseline.
    geobam_on_drq = sample_on_dates(
        geobam_all,
        drq,
        "geobam_mon",
    )

    obs_plot = obs_daily if args.plot_daily_observed else obs_on_drq
    geobam_plot = (
        geobam_all
        if args.plot_geobam_all_dates
        else geobam_on_drq
    )

    sampled_out = Path(args.obs_sampled_out)
    sampled_out.parent.mkdir(parents=True, exist_ok=True)
    obs_on_drq.to_csv(sampled_out, index=False)

    dl_kge, dl_r, dl_bias, dl_n = matched_metrics(
        obs_daily,
        drq,
        "discharge_m3s",
    )
    gb_kge, gb_r, gb_bias, gb_n = matched_metrics(
        obs_daily,
        geobam_plot,
        "geobam_mon",
    )

    print("Gauge:", args.gauge)
    print("Observed daily rows:", len(obs_daily))
    print("DRQ rows:", len(drq))
    print("Observed rows on DRQ dates:", len(obs_on_drq))
    print("Public geoBAM rows:", len(geobam_all))
    print("Public geoBAM rows on DRQ dates:", len(geobam_on_drq))
    print()
    print("Matched-date metrics")
    print(
        f"  DRQ:    n={dl_n}, KGE={dl_kge:.3f}, "
        f"r={dl_r:.3f}, bias ratio={dl_bias:.3f}"
    )
    print(
        f"  geoBAM: n={gb_n}, KGE={gb_kge:.3f}, "
        f"r={gb_r:.3f}, bias ratio={gb_bias:.3f}"
    )
    print()
    print("Paper Figure 8 reference: KGE DL ~= 0.46, KGE geoBAM ~= -0.15")
    print("Saved sampled observed series:", sampled_out)

    # The published hydrograph is a compact, wide panel.
    fig, ax = plt.subplots(figsize=(7.2, 2.55))

    ax.plot(
        obs_plot["date"],
        obs_plot["streamflow_m3s"],
        color="black",
        linewidth=0.72,
        linestyle="-",
        label="Observed",
        zorder=2,
    )
    ax.plot(
        drq["date"],
        drq["discharge_m3s"],
        color="#d62728",
        linewidth=0.86,
        linestyle="--",
        label="This study",
        zorder=4,
    )
    ax.plot(
        geobam_plot["date"],
        geobam_plot["geobam_mon"],
        color="#6f8fae",
        linewidth=0.82,
        linestyle=(0, (1.1, 1.5)),
        label="geoBAM",
        zorder=3,
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
    ax.xaxis.set_major_formatter(
        mdates.DateFormatter("%Y")
    )
    ax.set_yticks(
        [0, 5000, 10000, 15000, 20000]
    )

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
    if not args.plot_daily_observed:
        print(
            "Observed plotting mode: sampled on DRQ/Landsat dates "
            "(paper-like; this fixes the overly dense black line)."
        )
    if not args.plot_geobam_all_dates:
        print(
            "geoBAM plotting mode: public geoBAM values sampled on DRQ dates."
        )
    print(
        "NOTE: the public geoBAM archive is not proven to be the exact "
        "Figure 8 baseline series used by Hao et al."
    )


if __name__ == "__main__":
    main()
