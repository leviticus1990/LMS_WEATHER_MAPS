#!/usr/bin/env python3

from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.patches import Polygon
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path.home() / "LMS_WEATHER_MAPS"
MAP_DIR = BASE_DIR / "maps"

DATE_NOW = (
    sys.argv[1]
    if len(sys.argv) > 1
    else datetime.now(timezone.utc).strftime("%Y%m%d")
)

INPUT_DIR = MAP_DIR / DATE_NOW / "ECMWF"
OUTPUT_DIR = (
    BASE_DIR
    / "archive"
    / "ECMWF"
    / DATE_NOW
    / "FINAL_LMS"
)

SHAPEFILE = BASE_DIR / "shp_LBR" / "gadm41_LBR_1.shp"


# ============================================================
# LIBERIA DOMAIN
# ============================================================

WEST = -12.0
EAST = -7.0
SOUTH = 4.0
NORTH = 9.0


# ============================================================
# TEMPERATURE LEVELS
# ============================================================

TEMP_BOUNDS = [18, 21, 23, 25, 27, 29, 31, 33, 36]
TEMP_LABELS = ["<21", "21-23", "23-25", "25-27", "27-29", "29-31", "31-33", ">33"]
TEMP_CMAP = plt.get_cmap("RdYlBu_r", len(TEMP_LABELS))
TEMP_NORM = BoundaryNorm(TEMP_BOUNDS, TEMP_CMAP.N)


# ============================================================
# FORECAST PERIODS
# ============================================================

FORECAST_PERIODS = {
    "024h": "Day 1",
    "048h": "Day 2",
    "072h": "Day 3",
    "096h": "Day 4",
    "120h": "Day 5",
    "144h": "Day 6",
}

PERIOD_START_HOURS = {
    "024h": 0,
    "048h": 24,
    "072h": 48,
    "096h": 72,
    "120h": 96,
    "144h": 120,
}


# ============================================================
# LOAD COUNTY BOUNDARIES
# ============================================================

if not SHAPEFILE.exists():
    raise FileNotFoundError(
        f"Liberia shapefile not found:\n{SHAPEFILE}"
    )

counties = gpd.read_file(SHAPEFILE)

if counties.crs is None:
    raise ValueError("Liberia shapefile has no CRS.")

counties = counties.to_crs("EPSG:4326")


# ============================================================
# LABEL POSITIONS
# ============================================================

label_points = counties.copy()
label_points["geometry"] = label_points.geometry.representative_point()


# ============================================================
# RENDER FUNCTION
# ============================================================

def render_temperature_map(input_file: Path, output_file: Path):

    print(f"\nProcessing: {input_file.name}")

    with rasterio.open(input_file) as src:

        data = src.read(1).astype(float)
        transform = src.transform
        nodata = src.nodata

        if nodata is not None:
            data[data == nodata] = np.nan

        data[data < -80] = np.nan
        data[data > 60] = np.nan

        height, width = data.shape

        lon = (
            np.arange(width) * transform.a
            + transform.c
            + transform.a / 2
        )
        lat = (
            np.arange(height) * transform.e
            + transform.f
            + transform.e / 2
        )

        # Work in ascending latitude for interpolation and plotting.
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = data[::-1, :]

        lon_mask = (lon >= WEST) & (lon <= EAST)
        lat_mask = (lat >= SOUTH) & (lat <= NORTH)

        if not np.any(lon_mask) or not np.any(lat_mask):
            raise ValueError("Temperature raster does not overlap Liberia.")

        data = data[lat_mask][:, lon_mask]
        lon = lon[lon_mask]
        lat = lat[lat_mask]

        # Bilinear interpolation smooths display boundaries while preserving
        # the source forecast values and the existing temperature scale.
        smooth_factor = 6
        smooth_lon = np.linspace(
            lon[0], lon[-1], (len(lon) - 1) * smooth_factor + 1
        )
        smooth_lat = np.linspace(
            lat[0], lat[-1], (len(lat) - 1) * smooth_factor + 1
        )
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

        print(f"  Minimum: {np.nanmin(data):.2f} °C")
        print(f"  Maximum: {np.nanmax(data):.2f} °C")
        print(f"  Mean:    {np.nanmean(data):.2f} °C")

    fig, ax = plt.subplots(
        figsize=(12, 10),
        dpi=200
    )

    # --------------------------------------------------------
    # TEMPERATURE FIELD
    # --------------------------------------------------------

    image = ax.pcolormesh(
        smooth_lon,
        smooth_lat,
        smooth_data,
        cmap=TEMP_CMAP,
        norm=TEMP_NORM,
        shading="auto",
    )

    # --------------------------------------------------------
    # COUNTY BOUNDARIES
    # --------------------------------------------------------

    counties.boundary.plot(
        ax=ax,
        linewidth=0.8,
        edgecolor="black",
        zorder=5,
    )

    # --------------------------------------------------------
    # COUNTY LABELS
    # --------------------------------------------------------

    for _, row in label_points.iterrows():

        x = row.geometry.x
        y = row.geometry.y

        ax.text(
            x,
            y,
            row["NAME_1"],
            fontsize=7.5,
            ha="center",
            va="center",
            color="black",
            zorder=6,
            bbox=dict(
                facecolor="white",
                edgecolor="none",
                alpha=0.65,
                pad=1.2,
            ),
        )

    # --------------------------------------------------------
    # MAP EXTENT
    # --------------------------------------------------------

    ax.set_xlim(WEST, EAST)
    ax.set_ylim(SOUTH, NORTH)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")

    ax.grid(
        True,
        linewidth=0.4,
        alpha=0.4,
    )

    # --------------------------------------------------------
    # FORECAST PERIOD
    # --------------------------------------------------------

    forecast_code = input_file.stem.split("_")[-1]
    valid_date = (
        datetime.strptime(DATE_NOW, "%Y%m%d")
        + timedelta(hours=PERIOD_START_HOURS.get(forecast_code, 0))
    ).strftime("%B %d, %Y").upper().replace(" 0", " ")

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    fig.suptitle(
        "LIBERIA METEOROLOGICAL SERVICE",
        fontsize=17,
        fontweight="bold",
        y=0.975,
    )

    ax.set_title(
        f"{valid_date}\nTWO-METRE TEMPERATURE FORECAST",
        fontsize=14,
        fontweight="bold",
        pad=22,
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    legend_ticks = [19.5, 22, 24, 26, 28, 30, 32, 34.5]
    cbar = fig.colorbar(
        image,
        ax=ax,
        orientation="horizontal",
        fraction=0.046,
        pad=0.08,
        boundaries=TEMP_BOUNDS,
        ticks=legend_ticks,
    )

    cbar.ax.set_xticklabels(TEMP_LABELS, fontsize=9)

    cbar.set_label(
        "Temperature (°C)",
        fontsize=11,
        fontweight="bold",
    )

    # --------------------------------------------------------
    # LMS LOGO — approved upper-left map position
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
            (0.025, 0.985),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0, 1),
            pad=0,
            zorder=30,
            annotation_clip=False
        )

        ax.add_artist(logo_box)

    # --------------------------------------------------------
    # COMPASS ROSE — aligned vertically with the logo
    # --------------------------------------------------------
    cx, cy, radius = 0.86, 0.87, 0.060
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
    for label, dx, dy in (("N", 0, 0.085), ("S", 0, -0.085),
                          ("E", 0.085, 0), ("W", -0.085, 0)):
        ax.text(cx + dx, cy + dy, label, transform=ax.transAxes,
                ha="center", va="center", fontsize=9, family="serif",
                zorder=23, clip_on=False)

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    fig.text(
        0.5,
        0.018,
        "Liberia Meteorological Service (LMS) | Ministry of Transport",
        ha="center",
        fontsize=9,
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    plt.savefig(
        output_file,
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(f"  PNG: {output_file.name}")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("LMS TEMPERATURE MAP RENDERER")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    success = 0

    for forecast_code in FORECAST_PERIODS:

        input_file = (
            INPUT_DIR
            / f"ECMWF_TEMPERATURE_AFRICA_{DATE_NOW}_{forecast_code}.tif"
        )

        output_file = (
            OUTPUT_DIR
            / f"LMS_TEMPERATURE_AFRICA_{DATE_NOW}_{forecast_code}.png"
        )

        if not input_file.exists():
            print(f"WARNING: Missing {input_file.name}")
            continue

        try:
            render_temperature_map(
                input_file,
                output_file,
            )
            success += 1

        except Exception as exc:
            print(f"ERROR processing {input_file.name}: {exc}")

    print("\n" + "=" * 70)
    print(f"Successful maps: {success}/{len(FORECAST_PERIODS)}")
    print("=" * 70)

    if success == len(FORECAST_PERIODS):
        print("LMS TEMPERATURE MAP RENDERING COMPLETE")
    else:
        print("Temperature map rendering completed with warnings.")


if __name__ == "__main__":
    main()
