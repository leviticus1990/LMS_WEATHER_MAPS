#!/usr/bin/env python3

"""
LMS ECMWF 2-METRE MAXIMUM TEMPERATURE MAP RENDERER

Input:
    ECMWF_TMAX_AFRICA_YYYYMMDD_024h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_048h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_072h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_096h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_120h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_144h.tif

Output:
    LMS_TMAX_AFRICA_YYYYMMDD_024h.png
    ...
    LMS_TMAX_AFRICA_YYYYMMDD_144h.png
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import rasterio
from matplotlib.patches import Polygon
from matplotlib.colors import LinearSegmentedColormap, Normalize
from rasterio.features import geometry_mask
from rasterio.transform import from_origin


# ----------------------------------------------------------------------
# PROJECT PATH
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

BASE_DIR = PROJECT_ROOT

DATE_NOW = (
    sys.argv[1]
    if len(sys.argv) > 1
    else datetime.now(timezone.utc).strftime("%Y%m%d")
)

ECMWF_DIR = (
    BASE_DIR
    / "maps"
    / DATE_NOW
    / "ECMWF"
)

ARCHIVE_DIR = (
    BASE_DIR
    / "archive"
    / "ECMWF"
    / DATE_NOW
    / "FINAL_LMS"
)

ARCHIVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# LIBERIA SHAPEFILE
# ----------------------------------------------------------------------

SHAPEFILE = (
    BASE_DIR
    / "shp_LBR"
    / "gadm41_LBR_1.shp"
)


# ----------------------------------------------------------------------
# MAP EXTENT
# ----------------------------------------------------------------------

WEST = -11.7
EAST = -7.1
SOUTH = 4.2
NORTH = 8.7


# ----------------------------------------------------------------------
# FORECAST PERIODS
# ----------------------------------------------------------------------

FORECAST_PERIODS = {
    "024h": "Day 1",
    "048h": "Day 2",
    "072h": "Day 3",
    "096h": "Day 4",
    "120h": "Day 5",
    "144h": "Day 6",
}


# ----------------------------------------------------------------------
# TEMPERATURE LEVELS
# ----------------------------------------------------------------------

TEMP_LEVELS = [19, 21, 23, 25, 27, 29, 31, 33, 35]
TEMP_COLORS = [
    "#313695", "#74ADD1", "#ABDDA4", "#FFFFBF",
    "#FEE090", "#FDAE61", "#F46D43", "#A50026",
]
TEMP_LABELS = [
    "<21°C", "21–23°C", "23–25°C", "25–27°C",
    "27–29°C", "29–31°C", "31–33°C", ">33°C",
]
COLORMAP = LinearSegmentedColormap.from_list(
    "tmax_temperature", TEMP_COLORS, N=256
)
NORM = Normalize(vmin=TEMP_LEVELS[0], vmax=TEMP_LEVELS[-1])


# ----------------------------------------------------------------------
# LOAD COUNTY BOUNDARIES
# ----------------------------------------------------------------------

counties = gpd.read_file(SHAPEFILE)

if counties.crs is None:
    counties = counties.set_crs("EPSG:4326")

counties = counties.to_crs("EPSG:4326")


# ----------------------------------------------------------------------
# RENDER FUNCTION
# ----------------------------------------------------------------------

def render_tmax_map(
    input_file: Path,
    output_file: Path,
    forecast_period: str,
):

    print("\n" + "-" * 70)
    print(f"Rendering: {input_file.name}")
    print("-" * 70)

    with rasterio.open(input_file) as src:

        data = src.read(1).astype(float)
        transform = src.transform

        nodata = src.nodata

        if nodata is not None:
            data[data == nodata] = np.nan

        # Remove clearly invalid temperatures.
        data[data < -80] = np.nan
        data[data > 60] = np.nan

        height, width = data.shape
        lon = np.arange(width) * transform.a + transform.c + transform.a / 2
        lat = np.arange(height) * transform.e + transform.f + transform.e / 2

        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = data[::-1, :]

        lon_mask = (lon >= WEST) & (lon <= EAST)
        lat_mask = (lat >= SOUTH) & (lat <= NORTH)
        if not np.any(lon_mask) or not np.any(lat_mask):
            raise ValueError("Maximum-temperature raster does not overlap Liberia.")

        data = data[lat_mask][:, lon_mask]
        lon = lon[lon_mask]
        lat = lat[lat_mask]

        # Match precipitation's smoothed, country-clipped display layer.
        # Upsample the forecast grid for a smooth display field. Continuous
        # color interpolation below avoids visible bands between class breaks.
        smooth_factor = 12
        smooth_lon = np.linspace(lon[0], lon[-1], (len(lon) - 1) * smooth_factor + 1)
        smooth_lat = np.linspace(lat[0], lat[-1], (len(lat) - 1) * smooth_factor + 1)
        smooth_data_lon = np.vstack([
            np.interp(smooth_lon, lon, row) for row in data
        ])
        smooth_data = np.column_stack([
            np.interp(smooth_lat, lat, smooth_data_lon[:, column])
            for column in range(smooth_data_lon.shape[1])
        ])

        pixel_width = smooth_lon[1] - smooth_lon[0]
        pixel_height = smooth_lat[1] - smooth_lat[0]
        fine_transform = from_origin(
            smooth_lon[0] - pixel_width / 2,
            smooth_lat[-1] + pixel_height / 2,
            pixel_width,
            pixel_height,
        )
        country_mask = geometry_mask(
            counties.geometry,
            out_shape=smooth_data.shape,
            transform=fine_transform,
            invert=True,
        )
        smooth_data = np.where(country_mask[::-1, :], smooth_data, np.nan)

    # ------------------------------------------------------------------
    # FIGURE
    # ------------------------------------------------------------------

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_axes([0.06, 0.09, 0.80, 0.82])
    ax.set_facecolor("white")

    # ------------------------------------------------------------------
    # TEMPERATURE FIELD
    # ------------------------------------------------------------------

    image = ax.pcolormesh(
        smooth_lon,
        smooth_lat,
        smooth_data,
        cmap=COLORMAP,
        norm=NORM,
        shading="auto",
    )

    # ------------------------------------------------------------------
    # COUNTY BOUNDARIES
    # ------------------------------------------------------------------

    counties.boundary.plot(
        ax=ax,
        linewidth=0.55,
        edgecolor="black",
    )

    # ------------------------------------------------------------------
    # MAP EXTENT
    # ------------------------------------------------------------------

    # Match precipitation's county labels and framed nautical-style axes.
    for _, county in counties.iterrows():
        point = county.geometry.representative_point()
        if WEST <= point.x <= EAST and SOUTH <= point.y <= NORTH:
            ax.text(
                point.x, point.y, county["NAME_1"], fontsize=8,
                ha="center", va="center", color="black",
                fontweight="bold", zorder=10,
            )

    ax.set_xlim(-12.3, -6.7)
    ax.set_ylim(4.0, 9.4)
    xticks = np.arange(-12, -6, 1)
    yticks = np.arange(4, 10, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{abs(int(x))}°W" for x in xticks], fontsize=8)
    ax.set_yticklabels([f"{int(y)}°N" for y in yticks], fontsize=8, rotation=90, va="center")
    ax.tick_params(top=True, right=True, labeltop=True, labelright=True,
                   direction="out", length=3, pad=3)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    period_start_hours = {
        "024h": 0, "048h": 24, "072h": 48,
        "096h": 72, "120h": 96, "144h": 120,
    }
    forecast_code = input_file.stem.split("_")[-1]
    valid_date = (
        datetime.strptime(DATE_NOW, "%Y%m%d")
        + timedelta(hours=period_start_hours.get(forecast_code, 0))
    ).strftime("%B %d, %Y").upper().replace(" 0", " ")
    ax.text(0.55, 0.95,
            f"LIBERIA METEOROLOGICAL SERVICE\n{valid_date}\n2-METRE MAXIMUM\nTEMPERATURE FORECAST",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, fontweight="bold", family="serif", zorder=20)
    # Compass rose, matching precipitation's placement and style.
    cx, cy, radius = 0.86, 0.87, 0.073
    ax.add_patch(plt.Circle((cx, cy), radius, transform=ax.transAxes,
                            facecolor="white", edgecolor="black", linewidth=1.2,
                            zorder=21, clip_on=False))
    for angle in np.arange(0, 360, 45):
        theta = np.deg2rad(angle)
        tip = (cx + radius * np.sin(theta), cy + radius * np.cos(theta))
        left = (cx + radius * 0.15 * np.sin(theta + np.pi / 8),
                cy + radius * 0.15 * np.cos(theta + np.pi / 8))
        right = (cx + radius * 0.15 * np.sin(theta - np.pi / 8),
                 cy + radius * 0.15 * np.cos(theta - np.pi / 8))
        ax.add_patch(Polygon([left, tip, right], closed=True,
                             transform=ax.transAxes,
                             facecolor="black" if angle % 90 == 0 else "white",
                             edgecolor="black", linewidth=0.7, zorder=22,
                             clip_on=False))
    for label, dx, dy in (("N", 0, 0.10), ("S", 0, -0.10),
                          ("E", 0.10, 0), ("W", -0.10, 0)):
        ax.text(cx + dx, cy + dy, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, family="serif",
                zorder=23, clip_on=False)

    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.075,
        ticks=[20, 22, 24, 26, 28, 30, 32, 34],
    )
    cbar.ax.set_xticklabels(TEMP_LABELS, fontsize=9)
    cbar.set_label("Maximum Temperature (°C)", fontsize=10, fontweight="bold")

    fig.text(0.46, 0.025,
             "Liberia Meteorological Service | Ministry of Transport | ECMWF",
             ha="center", va="center", fontsize=8, fontweight="bold")

    # ------------------------------------------------------------------
    # LMS LOGO — upper-left inside the actual map axes
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # SAVE
    # ------------------------------------------------------------------

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(
        f"Created: {output_file}"
    )


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

def main():

    print("\n" + "=" * 70)
    print("LMS TMAX MAP RENDERER")
    print("=" * 70)

    print(f"Date: {DATE_NOW}")
    print(f"Input directory: {ECMWF_DIR}")
    print(f"Archive directory: {ARCHIVE_DIR}")

    rendered = 0

    for forecast_code, forecast_period in FORECAST_PERIODS.items():

        input_file = (
            ECMWF_DIR
            / f"ECMWF_TMAX_AFRICA_"
            f"{DATE_NOW}_{forecast_code}.tif"
        )

        output_file = (
            ARCHIVE_DIR
            / f"LMS_TMAX_AFRICA_"
            f"{DATE_NOW}_{forecast_code}.png"
        )

        if not input_file.exists():

            print(
                f"\nMISSING INPUT: "
                f"{input_file.name}"
            )

            continue

        render_tmax_map(
            input_file,
            output_file,
            forecast_period,
        )

        rendered += 1

    # ------------------------------------------------------------------
    # COMPLETION
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print(
        f"TMAX MAP RENDERING COMPLETE: "
        f"{rendered}/6"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
