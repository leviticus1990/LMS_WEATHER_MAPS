#!/usr/bin/env python3

import sys
import shutil
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

BASE_DIR = Path(__file__).resolve().parents[2]

import cartopy.crs as ccrs
import cartopy.feature as cfeature

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ecmwf_config as config
from datetime import datetime, timezone


# ============================================================
# PATHS
# ============================================================

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

INPUT_DIR = (
    config.MAP_DIR
    / DATE_NOW
    / "ECMWF"
)

OUTPUT_DIR = INPUT_DIR

SHAPEFILE = (
    config.BASE_DIR
    / "shp_LBR"
    / "gadm41_LBR_1.shp"
)


# ============================================================
# LIBERIA DOMAIN
# ============================================================

WEST = -12.0
EAST = -7.0
SOUTH = 4.0
NORTH = 9.0


# ============================================================
# PRESSURE LEVELS
# ============================================================

PRESSURE_LEVELS = np.arange(1012, 1019, 1)

# Forecast hour -> forecast day
FORECAST_DAYS = {
    "024h": "Day 1",
    "048h": "Day 2",
    "072h": "Day 3",
    "096h": "Day 4",
    "120h": "Day 5",
    "144h": "Day 6",
}


# ============================================================
# LOAD COUNTY BOUNDARIES
# ============================================================

counties = gpd.read_file(SHAPEFILE)

if counties.crs is None:
    counties = counties.set_crs("EPSG:4326")
else:
    counties = counties.to_crs("EPSG:4326")


# ============================================================
# RENDER FUNCTION
# ============================================================

def render_pressure_map(input_file):

    forecast_code = input_file.stem.split("_")[-1]
    forecast_day = FORECAST_DAYS.get(
        forecast_code,
        forecast_code
    )

    output_file = (
        OUTPUT_DIR
        / f"LMS_PRESSURE_AFRICA_{DATE_NOW}_{forecast_code}.png"
    )

    with rasterio.open(input_file) as src:

        pressure = src.read(1).astype(float)

        transform = src.transform

        height, width = pressure.shape

        west = transform.c
        north = transform.f

        east = west + transform.a * width
        south = north + transform.e * height

        lons = np.linspace(
            west + transform.a / 2,
            east - transform.a / 2,
            width
        )

        lats = np.linspace(
            north + transform.e / 2,
            south - transform.e / 2,
            height
        )

    # --------------------------------------------------------
    # VALID DATA
    # --------------------------------------------------------

    pressure = np.where(
        np.isfinite(pressure),
        pressure,
        np.nan
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig = plt.figure(
        figsize=(11, 9),
        dpi=200
    )

    ax = plt.axes(
        projection=ccrs.PlateCarree()
    )

    ax.set_extent(
        [WEST, EAST, SOUTH, NORTH],
        crs=ccrs.PlateCarree()
    )

    # --------------------------------------------------------
    # BACKGROUND
    # --------------------------------------------------------

    ax.add_feature(
        cfeature.OCEAN,
        facecolor="white",
        zorder=0
    )

    ax.add_feature(
        cfeature.LAND,
        facecolor="white",
        zorder=0
    )

    ax.add_feature(
        cfeature.COASTLINE,
        linewidth=0.8,
        zorder=3
    )

    # --------------------------------------------------------
    # PRESSURE SHADING
    # --------------------------------------------------------

    filled = ax.contourf(
        lons,
        lats,
        pressure,
        levels=PRESSURE_LEVELS,
        cmap="viridis",
        extend="both",
        transform=ccrs.PlateCarree(),
        zorder=1
    )

    # --------------------------------------------------------
    # ISOBARS
    # --------------------------------------------------------

    contours = ax.contour(
        lons,
        lats,
        pressure,
        levels=PRESSURE_LEVELS,
        colors="black",
        linewidths=0.65,
        transform=ccrs.PlateCarree(),
        zorder=4
    )

    ax.clabel(
        contours,
        inline=True,
        fontsize=8,
        fmt="%.0f hPa",
        inline_spacing=5
    )

    # --------------------------------------------------------
    # COUNTY BOUNDARIES
    # --------------------------------------------------------

    counties.boundary.plot(
        ax=ax,
        linewidth=0.55,
        edgecolor="black",
        transform=ccrs.PlateCarree(),
        zorder=5
    )

    # --------------------------------------------------------
    # COUNTY LABELS
    # --------------------------------------------------------

    for _, row in counties.iterrows():

        point = row.geometry.representative_point()

        ax.text(
            point.x,
            point.y,
            row["NAME_1"],
            transform=ccrs.PlateCarree(),
            fontsize=6.5,
            ha="center",
            va="center",
            zorder=6
        )

    # --------------------------------------------------------
    # GRIDLINES
    # --------------------------------------------------------

    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.35,
        alpha=0.5,
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
    # TITLE
    # --------------------------------------------------------

    ax.text(
        0.5,
        1.075,
        "LIBERIA METEOROLOGICAL SERVICE",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=17,
        fontweight="bold"
    )

    ax.text(
        0.5,
        1.045,
        "MEAN SEA-LEVEL PRESSURE FORECAST",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=13,
        fontweight="bold"
    )

    ax.text(
        0.5,
        1.015,
        f"Forecast period: {forecast_day}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=11
    )

    # --------------------------------------------------------
    # LMS LOGO — upper-left inside the actual map axes
    # --------------------------------------------------------
    logo_path = BASE_DIR / "LMS_LOGO_TRANSPARENT.png"

    if logo_path.exists():
        logo_img = plt.imread(logo_path)

        logo = OffsetImage(
            logo_img,
            zoom=0.24,
            interpolation="lanczos"
        )

        logo_box = AnnotationBbox(
            logo,
            (0.015, 0.995),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0, 1),
            pad=0,
            zorder=30,
            annotation_clip=False
        )

        ax.add_artist(logo_box)

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    cbar = fig.colorbar(
        filled,
        ax=ax,
        orientation="horizontal",
        pad=0.055,
        shrink=0.82,
        aspect=38
    )

    cbar.set_label(
        "Mean Sea-Level Pressure (hPa)",
        fontsize=10
    )

    cbar.ax.tick_params(
        labelsize=8
    )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.018,
        "Liberia Meteorological Service (LMS) | Ministry of Transport",
        ha="center",
        va="bottom",
        fontsize=9
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)

    return output_file


# ============================================================
# MAIN
# ============================================================

def main():

    files = sorted(
        INPUT_DIR.glob(
            f"ECMWF_PRESSURE_AFRICA_{DATE_NOW}_*.tif"
        )
    )

    if len(files) != 6:

        print(
            f"WARNING: Expected 6 pressure GeoTIFFs, "
            f"found {len(files)}."
        )

    successful = 0

    for input_file in files:

        try:

            output_file = render_pressure_map(
                input_file
            )

            print(
                f"Created: {output_file.name}"
            )

            successful += 1

        except Exception as exc:

            print(
                f"ERROR: {input_file.name}: {exc}"
            )

    print()
    print("=" * 70)
    print(
        f"Successful pressure maps: {successful}/{len(files)}"
    )
    print("=" * 70)


    # ============================================================
    # FINAL LMS ARCHIVE
    # ============================================================

    FINAL_LMS_DIR = (
        config.BASE_DIR
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
        "024h": OUTPUT_DIR / f"LMS_PRESSURE_AFRICA_{DATE_NOW}_024h.png",
        "048h": OUTPUT_DIR / f"LMS_PRESSURE_AFRICA_{DATE_NOW}_048h.png",
        "072h": OUTPUT_DIR / f"LMS_PRESSURE_AFRICA_{DATE_NOW}_072h.png",
        "096h": OUTPUT_DIR / f"LMS_PRESSURE_AFRICA_{DATE_NOW}_096h.png",
        "120h": OUTPUT_DIR / f"LMS_PRESSURE_AFRICA_{DATE_NOW}_120h.png",
        "144h": OUTPUT_DIR / f"LMS_PRESSURE_AFRICA_{DATE_NOW}_144h.png",
    }

    final_count = 0

    for label, source_file in FINAL_PRODUCTS.items():

        if source_file.exists():

            destination_file = (
                FINAL_LMS_DIR / source_file.name
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

    print(
        f"Final LMS products: "
        f"{final_count}/{len(FINAL_PRODUCTS)}"
    )


if __name__ == "__main__":
    main()
