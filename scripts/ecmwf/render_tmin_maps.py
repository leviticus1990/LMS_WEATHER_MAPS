#!/usr/bin/env python3
"""
LMS ECMWF Tmin Map Renderer

Renders six 24-hour minimum temperature forecast maps
from ECMWF mn2t3-derived GeoTIFFs.

Product:
    2-metre minimum temperature forecast

Color scale:
    Eight discrete classes matching the LMS_GFS Tmin renderer

Output:
    archive/ECMWF/YYYYMMDD/FINAL_LMS/
"""

from pathlib import Path
from datetime import datetime, timezone, timedelta
import sys

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Polygon
from rasterio.features import geometry_mask
from rasterio.transform import from_origin


# ----------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------

BASE_DIR = Path("/home/james/LMS_WEATHER_MAPS")

DATE_NOW = (
    sys.argv[1]
    if len(sys.argv) > 1
    else datetime.now(timezone.utc).strftime("%Y%m%d")
)

INPUT_DIR = BASE_DIR / "maps" / DATE_NOW / "ECMWF"

ARCHIVE_DIR = (
    BASE_DIR
    / "archive"
    / "ECMWF"
    / DATE_NOW
    / "FINAL_LMS"
)

MAP_OUTPUT_DIR = ARCHIVE_DIR

ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

SHAPEFILE = BASE_DIR / "shp_LBR" / "gadm41_LBR_1.shp"


# ----------------------------------------------------------------------
# LMS MAP EXTENT
# ----------------------------------------------------------------------

WEST = -11.7
EAST = -7.1
SOUTH = 4.2
NORTH = 8.7


# ----------------------------------------------------------------------
# MAP SETTINGS
# ----------------------------------------------------------------------

DPI = 200

FORECAST_PERIODS = {
    "024h": "Day 1",
    "048h": "Day 2",
    "072h": "Day 3",
    "096h": "Day 4",
    "120h": "Day 5",
    "144h": "Day 6",
}


# ----------------------------------------------------------------------
# Match the LMS_GFS minimum-temperature classes and colors.
# ----------------------------------------------------------------------

TMIN_LEVELS = [18, 20, 21, 22, 23, 24, 25, 26, 28]
TMIN_COLORS = [
    "#313695", "#74ADD1", "#ABDDA4", "#FFFFBF",
    "#FEE090", "#FDAE61", "#F46D43", "#A50026",
]
TMIN_LABELS = [
    "<20°C", "20–21°C", "21–22°C", "22–23°C",
    "23–24°C", "24–25°C", "25–26°C", ">26°C",
]
COLORMAP = LinearSegmentedColormap.from_list(
    "tmin_temperature", TMIN_COLORS, N=256
)
NORM = Normalize(vmin=TMIN_LEVELS[0], vmax=TMIN_LEVELS[-1])


# ----------------------------------------------------------------------
# LOAD LIBERIA BOUNDARIES
# ----------------------------------------------------------------------

counties = gpd.read_file(SHAPEFILE)

if counties.crs is None:
    counties = counties.set_crs("EPSG:4326")
else:
    counties = counties.to_crs("EPSG:4326")


# ----------------------------------------------------------------------
# RENDER FUNCTION
# ----------------------------------------------------------------------

def render_tmin_map(code, forecast_period):

    input_file = (
        INPUT_DIR
        / f"ECMWF_TMIN_AFRICA_{DATE_NOW}_{code}.tif"
    )

    output_file = (
        MAP_OUTPUT_DIR
        / f"LMS_TMIN_AFRICA_{DATE_NOW}_{code}.png"
    )

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input GeoTIFF not found: {input_file}"
        )

    print(f"Rendering {forecast_period}: {input_file.name}")

    with rasterio.open(input_file) as src:

        data = src.read(1).astype(float)

        transform = src.transform

        if src.nodata is not None:
            data[data == src.nodata] = np.nan

        height, width = data.shape
        lon = np.arange(width) * transform.a + transform.c + transform.a / 2
        lat = np.arange(height) * transform.e + transform.f + transform.e / 2
        if lat[0] > lat[-1]:
            lat = lat[::-1]
            data = data[::-1, :]

        lon_mask = (lon >= WEST) & (lon <= EAST)
        lat_mask = (lat >= SOUTH) & (lat <= NORTH)
        data = data[lat_mask][:, lon_mask]
        lon = lon[lon_mask]
        lat = lat[lat_mask]

        # Upsample the forecast grid and use continuous colors to soften
        # transitions between the displayed temperature classes.
        smooth_factor = 12
        smooth_lon = np.linspace(lon[0], lon[-1], (len(lon) - 1) * smooth_factor + 1)
        smooth_lat = np.linspace(lat[0], lat[-1], (len(lat) - 1) * smooth_factor + 1)
        data_lon = np.vstack([np.interp(smooth_lon, lon, row) for row in data])
        smooth_data = np.column_stack([
            np.interp(smooth_lat, lat, data_lon[:, column])
            for column in range(data_lon.shape[1])
        ])
        pixel_width = smooth_lon[1] - smooth_lon[0]
        pixel_height = smooth_lat[1] - smooth_lat[0]
        fine_transform = from_origin(
            smooth_lon[0] - pixel_width / 2,
            smooth_lat[-1] + pixel_height / 2,
            pixel_width,
            pixel_height,
        )
        mask = geometry_mask(counties.geometry, out_shape=smooth_data.shape,
                             transform=fine_transform, invert=True)
        smooth_data = np.where(mask[::-1, :], smooth_data, np.nan)

        raster_min = float(np.nanmin(data))
        raster_max = float(np.nanmax(data))
        raster_mean = float(np.nanmean(data))

    print(
        f"  Tmin: min={raster_min:.2f}°C "
        f"max={raster_max:.2f}°C "
        f"mean={raster_mean:.2f}°C"
    )

    fig = plt.figure(figsize=(12, 8), facecolor="white")
    ax = fig.add_axes([0.06, 0.09, 0.80, 0.82])
    ax.set_facecolor("white")

    # --------------------------------------------------------------
    # TEMPERATURE FIELD
    # --------------------------------------------------------------

    mesh = ax.pcolormesh(
        smooth_lon,
        smooth_lat,
        smooth_data,
        cmap=COLORMAP,
        norm=NORM,
        shading="auto",
    )

    # --------------------------------------------------------------
    # LIBERIA COUNTY BOUNDARIES
    # --------------------------------------------------------------

    counties.boundary.plot(
        ax=ax,
        linewidth=0.65,
        color="black",
    )

    # --------------------------------------------------------------
    # MAP EXTENT
    # --------------------------------------------------------------

    for _, county in counties.iterrows():
        point = county.geometry.representative_point()
        if WEST <= point.x <= EAST and SOUTH <= point.y <= NORTH:
            ax.text(point.x, point.y, county["NAME_1"], fontsize=8,
                    ha="center", va="center", color="black",
                    fontweight="bold", zorder=10)

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

    period_start_hours = {"024h": 0, "048h": 24, "072h": 48,
                          "096h": 72, "120h": 96, "144h": 120}
    valid_date = (datetime.strptime(DATE_NOW, "%Y%m%d")
                  + timedelta(hours=period_start_hours.get(code, 0)))
    valid_date = valid_date.strftime("%B %d, %Y").upper().replace(" 0", " ")
    ax.text(0.55, 0.95,
            f"LIBERIA METEOROLOGICAL SERVICE\n{valid_date}\n2-METRE MINIMUM\nTEMPERATURE FORECAST",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=8.5, fontweight="bold", family="serif", zorder=20)
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
        mesh,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.075,
        ticks=[19, 20.5, 21.5, 22.5, 23.5, 24.5, 25.5, 27],
    )
    cbar.ax.set_xticklabels(TMIN_LABELS, fontsize=9)
    cbar.set_label("Minimum Temperature (°C)", fontsize=10, fontweight="bold")

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

    fig.text(0.46, 0.025,
             "Liberia Meteorological Service | Ministry of Transport | ECMWF",
             ha="center", va="center", fontsize=8, fontweight="bold")

    fig.savefig(
        output_file,
        dpi=DPI,
        bbox_inches=None,
        facecolor="white",
    )

    plt.close(fig)

    print(f"  Created: {output_file}")

    return output_file


# ----------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------

print("=" * 70)
print("LMS ECMWF MINIMUM TEMPERATURE MAP RENDERER")
print("=" * 70)
print(f"Forecast date: {DATE_NOW}")
print(f"Input directory: {INPUT_DIR}")
print(f"Archive directory: {ARCHIVE_DIR}")
print("Color scale: Eight LMS_GFS-matched temperature classes")
print()

results = []

for code, forecast_period in FORECAST_PERIODS.items():

    result = render_tmin_map(
        code,
        forecast_period,
    )

    results.append(result)

print()
print("=" * 70)
print(f"TMIN MAP RENDERING COMPLETE: {len(results)}/6")
print("=" * 70)
