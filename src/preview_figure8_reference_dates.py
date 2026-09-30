#!/usr/bin/env python3
"""Preview Landsat chips for dates shown in Hao et al. (2024) Figure 8.

This is a diagnostic tool, not part of the DRQ inference itself. It renders the
selected 64x64 Landsat TOA chips as the same false-color band combination
described in the Figure 8 caption: NIR, Red, Green.

Use this to check whether our reconstructed Landsat inputs have the same river
geometry/crop as the published Figure 8 examples.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio


def stretch_rgb(rgb: np.ndarray, low: float = 2, high: float = 98) -> np.ndarray:
    """Per-channel percentile stretch for visualization only."""
    rgb = rgb.astype(np.float32)
    out = np.zeros_like(rgb, dtype=np.float32)

    for i in range(3):
        band = rgb[..., i]
        valid = np.isfinite(band)
        if not np.any(valid):
            continue
        lo, hi = np.percentile(band[valid], [low, high])
        if hi > lo:
            out[..., i] = np.clip((band - lo) / (hi - lo), 0, 1)

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/metadata/landsat_usable_scenes.csv",
    )
    parser.add_argument(
        "--chips",
        default="data/landsat_chips",
    )
    parser.add_argument(
        "--dates",
        nargs="+",
        default=["1988-04-08", "2005-05-25"],
        help="Dates to preview, in YYYY-MM-DD format.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figure8_reference_dates_falsecolor.png",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype=str)
    manifest["date"] = pd.to_datetime(manifest["date"])

    chips_dir = Path(args.chips)
    selected = []

    for date_text in args.dates:
        date = pd.Timestamp(date_text)
        rows = manifest.loc[manifest["date"] == date]

        if rows.empty:
            print(f"NOT FOUND: {date.date()}")
            continue

        for _, row in rows.iterrows():
            chip = chips_dir / f"{date.strftime('%Y%m%d')}_{row['scene_id']}.tif"
            if not chip.exists():
                print(f"MISSING CHIP: {chip}")
                continue

            with rasterio.open(chip) as src:
                # Stored band order:
                # 1 blue, 2 green, 3 red, 4 nir, 5 swir1, 6 swir2, 7 qa.
                # Figure 8 caption: false color = NIR, Red, Green.
                false_color = src.read([4, 3, 2])
                profile = {
                    "shape": (src.height, src.width),
                    "crs": str(src.crs),
                    "transform": str(src.transform),
                }

            false_color = np.transpose(false_color, (1, 2, 0))
            false_color = stretch_rgb(false_color)

            selected.append(
                {
                    "date": date,
                    "sensor": row["sensor"],
                    "scene_id": row["scene_id"],
                    "chip": chip,
                    "image": false_color,
                    "profile": profile,
                }
            )

    if not selected:
        raise RuntimeError("No requested reference-date chips were available.")

    fig, axes = plt.subplots(
        1,
        len(selected),
        figsize=(3.2 * len(selected), 3.3),
        squeeze=False,
    )
    axes = axes.ravel()

    for ax, item in zip(axes, selected):
        ax.imshow(item["image"])
        ax.set_title(
            f"{item['date'].date()}\n{item['sensor']}  {item['scene_id']}",
            fontsize=8,
        )
        ax.axis("off")

        print()
        print("DATE:", item["date"].date())
        print("SENSOR:", item["sensor"])
        print("SCENE:", item["scene_id"])
        print("CHIP:", item["chip"])
        print("SHAPE:", item["profile"]["shape"])
        print("CRS:", item["profile"]["crs"])
        print("TRANSFORM:", item["profile"]["transform"])

    fig.suptitle(
        "Hao Figure 8 input check — NIR / Red / Green false color",
        fontsize=10,
    )
    fig.tight_layout()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print()
    print("Saved:", output)
    print("Compare river geometry and crop against the published Figure 8 images.")


if __name__ == "__main__":
    main()
