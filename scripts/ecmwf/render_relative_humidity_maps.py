#!/usr/bin/env python3

"""
LMS RELATIVE HUMIDITY MAP RENDERER

Creates six LMS-branded 2-metre relative humidity forecast maps:

    024h
    048h
    072h
    096h
    120h
    144h

Input:
    ECMWF_RELATIVE_HUMIDITY_AFRICA_YYYYMMDD_*.tif

Output:
    LMS_RH_AFRICA_YYYYMMDD_*.png
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import ListedColormap, BoundaryNorm
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
from matplotlib.patches import Polygon


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path("/home/james/LMS_WEATHER_MAPS")

MAPS_DIR = BASE_DIR / "maps"
ARCHIVE_DIR = BASE_DIR / "archive" / "ECMWF"

SHAPEFILE = (
    BASE_DIR
    / "shp_LBR"
    / "gadm41_LBR_1.shp"
)


# ============================================================
# DATE
# ============================================================

if len(sys.argv) > 1:
    DATE_NOW = sys.argv[1]
else:
    DATE_NOW = datetime.now(
        timezone.utc
    ).strftime("%Y%m%d")


# ============================================================
# DIRECTORIES
# ============================================================

ECMWF_DIR = (
    MAPS_DIR
    / DATE_NOW
    / "ECMWF"
)

FINAL_DIR = (
    ARCHIVE_DIR
    / DATE_NOW
    / "FINAL_LMS"
)

FINAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FORECAST STEPS
# ============================================================

FORECAST_STEPS = [
    24,
    48,
    72,
    96,
    120,
    144,
]


# ============================================================
# MAP EXTENT
# ============================================================

WEST = -11.7
EAST = -7.1
SOUTH = 4.2
NORTH = 8.7


# ============================================================
# RH CLASSIFICATION
# ============================================================

RH_LEVELS = [
    65,
    70,
    75,
    80,
    85,
    90,
    95,
    100,
]

RH_COLORS = [
    "#F7FBFF",
    "#DEEBF7",
    "#C6DBEF",
    "#9ECAE1",
    "#6BAED6",
    "#4292C6",
    "#08519C",
]

RH_LABELS = [
    "65–70%",
    "70–75%",
    "75–80%",
    "80–85%",
    "85–90%",
    "90–95%",
    "95–100%",
]

COLORMAP = ListedColormap(
    RH_COLORS
)

NORM = BoundaryNorm(
    RH_LEVELS,
    COLORMAP.N
)


# ============================================================
# LOAD LIBERIA COUNTY BOUNDARIES
# ============================================================

if not SHAPEFILE.exists():

    raise FileNotFoundError(
        f"Liberia shapefile not found:\n"
        f"{SHAPEFILE}"
    )

liberia = gpd.read_file(
    SHAPEFILE
)

liberia = liberia.to_crs(
    "EPSG:4326"
)


# ============================================================
# RENDER FUNCTION
# ============================================================

def render_rh_map(
    input_tif,
    output_png,
    forecast_hours
):

    print(
        f"\nRendering +{forecast_hours:03d} h"
    )

    # --------------------------------------------------------
    # READ RASTER
    # --------------------------------------------------------

    with rasterio.open(
        input_tif
    ) as src:

        data = src.read(
            1
        ).astype(
            float
        )
        transform = src.transform
        if src.nodata is not None:
            data[data == src.nodata] = np.nan


    # --------------------------------------------------------
    # HANDLE NODATA
    # --------------------------------------------------------

    data[
        ~np.isfinite(data)
    ] = np.nan


    # Keep RH physically valid
    data = np.clip(
        data,
        0,
        100
    )

    height, width = data.shape
    lon = np.arange(width) * transform.a + transform.c + transform.a / 2
    lat = np.arange(height) * transform.e + transform.f + transform.e / 2
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        data = data[::-1, :]

    # Bilinear interpolation at six times the source density smooths the
    # displayed field while keeping the original RH thresholds and palette.
    smooth_factor = 6
    smooth_lon = np.linspace(lon[0], lon[-1], (len(lon) - 1) * smooth_factor + 1)
    smooth_lat = np.linspace(lat[0], lat[-1], (len(lat) - 1) * smooth_factor + 1)

    def interpolate_valid(source_x, target_x, values):
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return np.full(target_x.shape, np.nan)
        return np.interp(target_x, source_x[valid], values[valid])

    smooth_lon_data = np.vstack([
        interpolate_valid(lon, smooth_lon, row) for row in data
    ])
    smooth_data = np.column_stack([
        interpolate_valid(lat, smooth_lat, smooth_lon_data[:, column])
        for column in range(smooth_lon_data.shape[1])
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
        liberia.geometry,
        out_shape=smooth_data.shape,
        transform=fine_transform,
        invert=True,
    )
    # Keep the existing 50% lower display threshold and country mask.
    smooth_data = np.where(
        country_mask[::-1, :] & (smooth_data >= 50),
        smooth_data,
        np.nan,
    )


    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_axes([0.06, 0.11, 0.80, 0.80])
    ax.set_facecolor("white")


    # --------------------------------------------------------
    # RH MAP
    # --------------------------------------------------------

    image = ax.pcolormesh(
        smooth_lon,
        smooth_lat,
        smooth_data,
        cmap=COLORMAP,
        norm=NORM,
        shading="auto",
    )


    # --------------------------------------------------------
    # COUNTY BOUNDARIES
    # --------------------------------------------------------

    liberia.boundary.plot(
        ax=ax,
        linewidth=0.55,
        edgecolor="black"
    )


    # --------------------------------------------------------
    # COUNTY LABELS
    # --------------------------------------------------------

    if "NAME_1" in liberia.columns:

        for _, row in liberia.iterrows():

            if row.geometry is None:
                continue

            point = row.geometry.representative_point()

            ax.text(
                point.x,
                point.y,
                str(row["NAME_1"]),
                fontsize=8,
                ha="center",
                va="center",
                fontweight="bold",
                zorder=10,
            )


    # --------------------------------------------------------
    # MAP EXTENT
    # --------------------------------------------------------

    ax.set_xlim(-12.3, -6.7)
    ax.set_ylim(4.0, 9.4)
    xticks = np.arange(-12, -6, 1)
    yticks = np.arange(4, 10, 1)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f"{abs(int(x))}°W" for x in xticks], fontsize=8)
    ax.set_yticklabels([f"{int(y)}°N" for y in yticks], fontsize=8,
                       rotation=90, va="center")
    ax.tick_params(top=True, right=True, labeltop=True, labelright=True,
                   direction="out", length=3, pad=3)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    day_number = max(1, forecast_hours // 24)
    valid_date = (
        datetime.strptime(DATE_NOW, "%Y%m%d")
        + timedelta(days=day_number - 1)
    ).strftime("%B %d, %Y").upper().replace(" 0", " ")
    ax.text(0.55, 0.955,
            f"LIBERIA METEOROLOGICAL SERVICE\n{valid_date}\nRELATIVE HUMIDITY FORECAST",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, fontweight="bold", family="serif", zorder=20)
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
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.075,
        ticks=[
            67.5,
            72.5,
            77.5,
            82.5,
            87.5,
            92.5,
            97.5,
        ],
    )

    cbar.ax.set_xticklabels(
        RH_LABELS,
        fontsize=9
    )

    cbar.set_label(
        "Relative Humidity (%)",
        fontsize=10,
        fontweight="bold",
    )


    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    fig.text(
        0.46,
        0.025,
        "LIBERIA METEOROLOGICAL SERVICE  |  ECMWF IFS OPEN DATA",
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
    )


    # --------------------------------------------------------
    # LAYOUT
    # --------------------------------------------------------

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    fig.savefig(
        output_png,
        dpi=200,
        bbox_inches=None,
        facecolor="white",
    )

    plt.close(
        fig
    )


    # --------------------------------------------------------
    print(
        f"  Input : {input_tif.name}"
    )

    print(
        f"  Final : {output_png}"
    )


# ============================================================
# PROCESS SIX MAPS
# ============================================================

print("\n" + "=" * 70)
print("LMS RELATIVE HUMIDITY MAP RENDERER")
print("=" * 70)

print(
    f"Date: {DATE_NOW}"
)

print(
    f"Input directory:\n{ECMWF_DIR}"
)

print(
    f"Final directory:\n{FINAL_DIR}"
)

print("=" * 70)


successful = 0


for hours in FORECAST_STEPS:

    input_tif = (
        ECMWF_DIR
        / f"ECMWF_RELATIVE_HUMIDITY_AFRICA_"
        f"{DATE_NOW}_{hours:03d}h.tif"
    )

    output_png = (
        FINAL_DIR
        / f"LMS_RH_AFRICA_"
        f"{DATE_NOW}_{hours:03d}h.png"
    )


    if not input_tif.exists():

        print(
            f"\nWARNING: Missing input:"
            f"\n{input_tif}"
        )

        continue


    try:

        render_rh_map(
            input_tif,
            output_png,
            hours
        )

        successful += 1

    except Exception as e:

        print(
            f"\nERROR rendering +{hours:03d} h:"
        )

        print(e)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print(
    f"RH MAP RENDERING COMPLETE: "
    f"{successful}/{len(FORECAST_STEPS)}"
)
print("=" * 70)

if successful == len(
    FORECAST_STEPS
):

    print(
        "\nAll six LMS RH maps were generated successfully."
    )

    print(
        f"\nFinal LMS maps:\n{FINAL_DIR}"
    )

else:

    print(
        "\nWARNING: Not all RH maps were generated."
    )
