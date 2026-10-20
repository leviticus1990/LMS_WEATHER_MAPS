#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys
import re

import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = PROJECT_ROOT

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

INPUT_DIR = (
    BASE_DIR
    / "maps"
    / DATE_NOW
    / "ECMWF"
)

OUTPUT_DIR = INPUT_DIR

SHAPEFILE = (
    BASE_DIR
    / "shp_LBR"
    / "gadm41_LBR_1.shp"
)

SERVICE_NAME = "LIBERIA METEOROLOGICAL SERVICE"
ORGANIZATION_NAME = "Ministry of Transport"

FORECAST_HOURS = [24, 48, 72, 96, 120, 144]

# Consistent operational wind-speed scale
WIND_LEVELS = [0, 1, 2, 3, 4, 5, 6, 8]

# Liberia map extent
WEST = -12.0
EAST = -7.0
SOUTH = 4.0
NORTH = 9.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_forecast_label(hours):
    day = hours // 24
    return f"Day {day}"


def read_raster(path):
    with rasterio.open(path) as src:
        data = src.read(1).astype(float)
        transform = src.transform
        bounds = src.bounds

    rows, cols = data.shape

    xs = (
        np.arange(cols) + 0.5
    ) * transform.a + transform.c

    ys = (
        np.arange(rows) + 0.5
    ) * transform.e + transform.f

    lon, lat = np.meshgrid(xs, ys)

    return data, lon, lat, bounds


def reconstruct_uv(speed, direction):
    """
    Convert meteorological wind direction and speed
    back to U and V components.

    Meteorological direction:
    0   = from North
    90  = from East
    180 = from South
    270 = from West
    """

    direction_rad = np.deg2rad(direction)

    u = -speed * np.sin(direction_rad)
    v = -speed * np.cos(direction_rad)

    return u, v


# ============================================================
# LOAD COUNTY BOUNDARIES
# ============================================================

print("=" * 70)
print("LMS 10-METRE WIND MAP RENDERING")
print("=" * 70)

print(f"Input directory : {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"County shapefile : {SHAPEFILE}")

counties = gpd.read_file(SHAPEFILE)

if counties.crs is None:
    counties = counties.set_crs("EPSG:4326")
else:
    counties = counties.to_crs("EPSG:4326")

print(f"County features: {len(counties)}")


# ============================================================
# RENDER EACH FORECAST PERIOD
# ============================================================

for hours in FORECAST_HOURS:

    speed_file = (
        INPUT_DIR
        / f"ECMWF_WIND_SPEED_AFRICA_{DATE_NOW}_{hours:03d}h.tif"
    )

    direction_file = (
        INPUT_DIR
        / f"ECMWF_WIND_DIRECTION_AFRICA_{DATE_NOW}_{hours:03d}h.tif"
    )

    if not speed_file.exists():
        print(f"WARNING: Missing speed file: {speed_file}")
        continue

    if not direction_file.exists():
        print(f"WARNING: Missing direction file: {direction_file}")
        continue

    print()
    print(f"Rendering +{hours} h ...")

    speed, lon, lat, bounds = read_raster(speed_file)
    direction, _, _, _ = read_raster(direction_file)

    # Reconstruct wind components for barbs
    u, v = reconstruct_uv(speed, direction)

    # --------------------------------------------------------
    # Figure
    # --------------------------------------------------------

    fig = plt.figure(figsize=(10, 12), dpi=200)

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    ax.set_extent(
        [WEST, EAST, SOUTH, NORTH],
        crs=ccrs.PlateCarree()
    )

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white",
        zorder=0
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="0.94",
        zorder=0
    )

    ax.coastlines(
        resolution="10m",
        linewidth=0.8,
        color="black",
        zorder=4
    )

    # --------------------------------------------------------
    # Wind-speed field
    # --------------------------------------------------------

    cmap = plt.get_cmap("YlGnBu")

    norm = mcolors.BoundaryNorm(
        WIND_LEVELS,
        cmap.N,
        extend="max"
    )

    mesh = ax.pcolormesh(
        lon,
        lat,
        speed,
        cmap=cmap,
        norm=norm,
        shading="auto",
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    # --------------------------------------------------------
    # County boundaries
    # --------------------------------------------------------

    counties.boundary.plot(
        ax=ax,
        linewidth=0.7,
        edgecolor="black",
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # --------------------------------------------------------
    # County labels
    # --------------------------------------------------------

    for _, row in counties.iterrows():

        geom = row.geometry

        if geom is None or geom.is_empty:
            continue

        point = geom.representative_point()

        name = str(row["NAME_1"])

        ax.text(
            point.x,
            point.y,
            name,
            fontsize=6.5,
            ha="center",
            va="center",
            transform=ccrs.PlateCarree(),
            zorder=6,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.55,
                pad=1.2
            )
        )

    # --------------------------------------------------------
    # Wind barbs
    # --------------------------------------------------------

    # Downsample the 0.25-degree grid for readability
    step = 2

    lon_b = lon[::step, ::step]
    lat_b = lat[::step, ::step]
    u_b = u[::step, ::step]
    v_b = v[::step, ::step]
    speed_b = speed[::step, ::step]

    valid = np.isfinite(speed_b)

    ax.barbs(
        lon_b[valid],
        lat_b[valid],
        u_b[valid],
        v_b[valid],
        length=5.5,
        linewidth=0.45,
        barbcolor="black",
        flagcolor="black",
        sizes={
            "emptybarb": 0.05,
            "spacing": 0.15,
            "height": 0.35,
        },
        pivot="middle",
        transform=ccrs.PlateCarree(),
        zorder=7
    )

    # --------------------------------------------------------
    # Gridlines
    # --------------------------------------------------------

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        color="gray",
        alpha=0.6,
        linestyle="--"
    )

    gl.top_labels = False
    gl.right_labels = False

    gl.xlabel_style = {
        "size": 8
    }

    gl.ylabel_style = {
        "size": 8
    }

    # --------------------------------------------------------
    # Title
    # --------------------------------------------------------

    day_number = max(1, hours // 24)
    valid_date = (
        datetime.strptime(DATE_NOW, "%Y%m%d")
        + timedelta(days=day_number - 1)
    ).strftime("%B %d, %Y").upper().replace(" 0", " ")
    ax.text(
        0.55,
        0.955,
        f"LIBERIA METEOROLOGICAL SERVICE\n{valid_date}\n10-METRE WIND FORECAST",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        family="serif",
        zorder=20,
    )

    # --------------------------------------------------------
    # Colorbar
    # --------------------------------------------------------

    cbar = plt.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        pad=0.055,
        shrink=0.82,
        ticks=WIND_LEVELS
    )

    cbar.set_label(
        "10-metre wind speed (m/s)",
        fontsize=9
    )

    cbar.ax.tick_params(
        labelsize=8
    )

    # --------------------------------------------------------
    # Footer
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.025,
        f"{SERVICE_NAME}  |  {ORGANIZATION_NAME}",
        ha="center",
        fontsize=8.5,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_file = (
        OUTPUT_DIR
        / f"LMS_WIND_AFRICA_{DATE_NOW}_{hours:03d}h.png"
    )

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    print(f"  Created: {output_file.name}")


# ============================================================
# FINAL LMS ARCHIVE
# ============================================================

import shutil

FINAL_LMS_DIR = (
    BASE_DIR
    / "archive"
    / "ECMWF"
    / DATE_NOW
    / "FINAL_LMS"
)

FINAL_LMS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

FINAL_PRODUCTS = {
    "024h": OUTPUT_DIR / f"LMS_WIND_AFRICA_{DATE_NOW}_024h.png",
    "048h": OUTPUT_DIR / f"LMS_WIND_AFRICA_{DATE_NOW}_048h.png",
    "072h": OUTPUT_DIR / f"LMS_WIND_AFRICA_{DATE_NOW}_072h.png",
    "096h": OUTPUT_DIR / f"LMS_WIND_AFRICA_{DATE_NOW}_096h.png",
    "120h": OUTPUT_DIR / f"LMS_WIND_AFRICA_{DATE_NOW}_120h.png",
    "144h": OUTPUT_DIR / f"LMS_WIND_AFRICA_{DATE_NOW}_144h.png",
}

final_count = 0

for label, source_file in FINAL_PRODUCTS.items():

    if source_file.exists():

        destination_file = (
            FINAL_LMS_DIR
            / source_file.name
        )

        shutil.copy2(
            source_file,
            destination_file
        )

        final_count += 1

        print(
            f"  Final LMS product: "
            f"{destination_file.name}"
        )

print()
print(
    f"Final LMS products: "
    f"{final_count}/{len(FINAL_PRODUCTS)}"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 70)
print("LMS 10-METRE WIND MAP RENDERING COMPLETE")
print("=" * 70)
