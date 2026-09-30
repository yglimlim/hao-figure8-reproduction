#!/usr/bin/env python3
"""Download 64x64 Landsat TOA chips for the Hao et al. Figure 8 reproduction.

Reads data/metadata/landsat_usable_scenes.csv and downloads one 7-band
GeoTIFF per selected date in the band order expected by the authors' code:
blue, green, red, nir, swir1, swir2, qa.

By default only one chip is downloaded as a smoke test. Use --all after
verifying the first file.
"""

from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import ee
import pandas as pd

PROJECT = "keen-snow-510214-m9"
LON = -55.64355143522444
LAT = -28.180885089668294
TARGET_CRS = "EPSG:32721"  # UTM zone 21S
HALF_WIDTH_M = 960.0       # 64 px * 30 m / 2

COLLECTIONS = {
    "L5": "LANDSAT/LT05/C02/T1_TOA",
    "L7": "LANDSAT/LE07/C02/T1_TOA",
    "L8": "LANDSAT/LC08/C02/T1_TOA",
}

SOURCE_BANDS = {
    "L5": ["B1", "B2", "B3", "B4", "B5", "B7", "QA_PIXEL"],
    "L7": ["B1", "B2", "B3", "B4", "B5", "B7", "QA_PIXEL"],
    "L8": ["B2", "B3", "B4", "B5", "B6", "B7", "QA_PIXEL"],
}

OUTPUT_BANDS = ["blue", "green", "red", "nir", "swir1", "swir2", "qa"]


def fixed_region() -> ee.Geometry:
    """Return a 1920 m x 1920 m square centered on the GRWL river point."""
    point = ee.Geometry.Point([LON, LAT])
    x, y = point.transform(TARGET_CRS, 1).coordinates().getInfo()
    return ee.Geometry.Rectangle(
        [x - HALF_WIDTH_M, y - HALF_WIDTH_M,
         x + HALF_WIDTH_M, y + HALF_WIDTH_M],
        proj=TARGET_CRS,
        geodesic=False,
    )


def validate_tif(path: Path) -> None:
    try:
        import rasterio
    except ImportError:
        print("  rasterio not installed; skipping local shape check")
        return

    with rasterio.open(path) as src:
        if src.count != 7:
            raise RuntimeError(f"{path}: expected 7 bands, found {src.count}")
        if src.width != 64 or src.height != 64:
            raise RuntimeError(
                f"{path}: expected 64x64, found {src.width}x{src.height}"
            )
        print(
            f"  verified: {src.count} bands, {src.width}x{src.height}, "
            f"CRS={src.crs}"
        )


def download_one(sensor: str, scene_id: str, date: str,
                 region: ee.Geometry, out_dir: Path) -> Path:
    asset_id = f"{COLLECTIONS[sensor]}/{scene_id}"

    # Landsat TOA spectral bands are reflectance values in Earth Engine.
    # Cast all seven bands to float32 so a single multiband GeoTIFF can be
    # written even though QA_PIXEL is natively integer.
    image = (
        ee.Image(asset_id)
        .select(SOURCE_BANDS[sensor], OUTPUT_BANDS)
        .toFloat()
    )

    out_path = out_dir / f"{date.replace('-', '')}_{scene_id}.tif"
    if out_path.exists():
        try:
            validate_tif(out_path)
            print(f"SKIP existing: {out_path}")
            return out_path
        except Exception:
            out_path.unlink()

    params = {
        "bands": OUTPUT_BANDS,
        "region": region,
        "crs": TARGET_CRS,
        "dimensions": "64x64",
        "format": "GEO_TIFF",
    }

    url = image.getDownloadURL(params)

    last_error = None
    for attempt in range(1, 4):
        try:
            print(f"DOWNLOAD {date} {sensor} {scene_id} (attempt {attempt})")
            urllib.request.urlretrieve(url, out_path)
            validate_tif(out_path)
            return out_path
        except Exception as exc:
            last_error = exc
            if out_path.exists():
                out_path.unlink()
            if attempt < 3:
                time.sleep(3 * attempt)

    raise RuntimeError(f"Failed to download {scene_id}: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="data/metadata/landsat_usable_scenes.csv",
    )
    parser.add_argument(
        "--out-dir",
        default="data/landsat_chips",
    )
    parser.add_argument(
        "--project",
        default=PROJECT,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of rows to download (default: 1 smoke-test image).",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all rows in the manifest.",
    )
    args = parser.parse_args()

    ee.Initialize(project=args.project)

    manifest = Path(args.manifest)
    if not manifest.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest}")

    df = pd.read_csv(manifest, dtype={"sensor": str, "scene_id": str, "date": str})
    df = df.sort_values("date").reset_index(drop=True)

    if not args.all:
        df = df.head(args.limit)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    region = fixed_region()

    print(f"Project: {args.project}")
    print(f"Scenes selected: {len(df)}")
    print(f"Output directory: {out_dir}")
    print(f"Center: ({LON}, {LAT})")
    print("Target grid: 64x64 pixels, 1920x1920 m, EPSG:32721")
    print("Band order:", ", ".join(OUTPUT_BANDS))
    print()

    for i, row in df.iterrows():
        sensor = str(row["sensor"])
        if sensor not in COLLECTIONS:
            raise ValueError(f"Unknown sensor {sensor!r}")

        download_one(
            sensor=sensor,
            scene_id=str(row["scene_id"]),
            date=str(row["date"]),
            region=region,
            out_dir=out_dir,
        )

    print("\nDONE")
    print(f"Downloaded/verified {len(df)} chip(s).")


main()
