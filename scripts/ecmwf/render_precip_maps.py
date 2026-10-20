#!/usr/bin/env python3

"""
LMS ECMWF PROFESSIONAL PRECIPITATION MAP RENDERER

Purpose:
    Convert ECMWF precipitation GeoTIFF products into
    professional Liberia-focused PNG maps for LMS.

Input:
    maps/YYYYMMDD/ECMWF/*.tif

Output:
    maps/YYYYMMDD/ECMWF/*.png

Features:
    - Liberia national boundary
    - 15 county boundaries
    - County labels
    - Rainfall shading in mm
    - Latitude/longitude
    - Forecast valid period
    - ECMWF IFS attribution
    - Liberia Meteorological Service branding
"""

from pathlib import Path
import sys
from datetime import datetime, timezone, timedelta
import shutil

import numpy as np
import geopandas as gpd
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Polygon
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

BASE_DIR = Path.home() / "LMS_WEATHER_MAPS"

if len(sys.argv) > 1:
    DATE_NOW = sys.argv[1]
else:
    DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

ECMWF_DIR = (
    BASE_DIR
    / "maps"
    / DATE_NOW
    / "ECMWF"
)

SHAPEFILE = (
    BASE_DIR
    / "shp_LBR"
    / "gadm41_LBR_1.shp"
)


# Liberia map extent
WEST = -11.7
EAST = -7.1
SOUTH = 4.2
NORTH = 8.7


# ============================================================
# 2. MAP CONFIGURATION
# ============================================================

FIGURE_SIZE = (12, 8)
DPI = 200

# Rainfall intervals in millimetres.
# These are intentionally suitable for operational
# rainfall visualization.
RAINFALL_LEVELS = [
    0,
    10,
    25,
    50,
    75,
    100,
    300,
]

# Match the LMS_GFS precipitation map classes and colors.
RAINFALL_COLORS = [
    "#F7FBFF",
    "#C6DBEF",
    "#6BAED6",
    "#2171B5",
    "#08306B",
    "#54278F",
    "#3F007D",
]
RAINFALL_LABELS = [
    "0–10 mm",
    "10–25 mm",
    "25–50 mm",
    "50–75 mm",
    "75–100 mm",
    "100–300 mm",
]


# ============================================================
# 3. LOAD LIBERIA COUNTY BOUNDARIES
# ============================================================

print("\n" + "=" * 70)
print("LMS ECMWF PRECIPITATION MAP RENDERER")
print("=" * 70)

print("\nProject directory:")
print(BASE_DIR)

print("\nECMWF directory:")
print(ECMWF_DIR)

print("\nLoading Liberia county boundaries...")

if not SHAPEFILE.exists():
    raise FileNotFoundError(
        f"Liberia shapefile not found:\n{SHAPEFILE}"
    )

counties = gpd.read_file(SHAPEFILE)

print(f"Number of counties: {len(counties)}")
print("County-name field: NAME_1")

if len(counties) != 15:
    raise ValueError(
        f"Expected 15 counties, found {len(counties)}."
    )

# Ensure geographic coordinates
if counties.crs is None:
    raise ValueError(
        "Liberia shapefile has no CRS."
    )

counties = counties.to_crs("EPSG:4326")


# ============================================================
# 4. FIND PRECIPITATION GEOTIFF FILES
# ============================================================

if not ECMWF_DIR.exists():
    raise FileNotFoundError(
        f"ECMWF map directory does not exist:\n{ECMWF_DIR}"
    )

geotiff_files = sorted(
    ECMWF_DIR.glob(
        "ECMWF_PRECIP_AFRICA_*.tif"
    )
)

print("\nPrecipitation GeoTIFF files found:")

for tif in geotiff_files:
    print(" ", tif.name)

if not geotiff_files:
    raise FileNotFoundError(
        "No precipitation GeoTIFF files were found."
    )


# ============================================================
# 5. DETERMINE FORECAST PERIOD
# ============================================================

def get_forecast_period(filename):
    """
    Convert the forecast interval in the filename to an
    operational forecast day label.

    Examples:
        00-24h   -> Day 1
        24-48h   -> Day 2
        48-72h   -> Day 3
        72-96h   -> Day 4
        96-120h  -> Day 5
        120-144h -> Day 6
    """

    stem = filename.stem
    parts = stem.split("_")

    if len(parts) < 5:
        return "Forecast day unavailable"

    period = parts[-1]

    day_map = {
        "00-24h": "Day 1",
        "24-48h": "Day 2",
        "48-72h": "Day 3",
        "72-96h": "Day 4",
        "96-120h": "Day 5",
        "120-144h": "Day 6",
    }

    return day_map.get(period, "Forecast day unavailable")


def get_valid_date(filename):
    """Return the date assigned to this forecast day."""
    period = filename.stem.split("_")[-1]
    period_start_hours = {
        "00-24h": 0,
        "24-48h": 24,
        "48-72h": 48,
        "72-96h": 72,
        "96-120h": 96,
        "120-144h": 120,
    }
    if period not in period_start_hours:
        return datetime.strptime(DATE_NOW, "%Y%m%d")
    run_date = datetime.strptime(DATE_NOW, "%Y%m%d")
    return run_date + timedelta(hours=period_start_hours[period])


# ============================================================
# 6. CREATE INDIVIDUAL MAP
# ============================================================

def render_map(tif_file):

    forecast_period = get_forecast_period(
        tif_file
    )

    output_png = tif_file.with_suffix(".png")

    print("\n" + "-" * 70)
    print("Rendering:")
    print(tif_file.name)

    print("Forecast period:")
    print(forecast_period)

    # --------------------------------------------------------
    # Open raster
    # --------------------------------------------------------

    with rasterio.open(tif_file) as src:

        rainfall = src.read(1).astype(
            "float32"
        )

        transform = src.transform
        raster_crs = src.crs

        raster_bounds = src.bounds

        print(
            f"Raster size: "
            f"{src.width} x {src.height}"
        )

        print(
            f"CRS: {src.crs}"
        )

    # --------------------------------------------------------
    # Remove invalid values
    # --------------------------------------------------------

    rainfall = np.where(
        np.isfinite(rainfall),
        rainfall,
        np.nan
    )

    rainfall = np.maximum(
        rainfall,
        0
    )

    # --------------------------------------------------------
    # Calculate raster coordinates
    # --------------------------------------------------------

    height, width = rainfall.shape

    lon = (
        np.arange(width)
        * transform.a
        + transform.c
        + transform.a / 2
    )

    lat = (
        np.arange(height)
        * transform.e
        + transform.f
        + transform.e / 2
    )

    # Raster latitude is normally descending.
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        rainfall = rainfall[::-1, :]

    # --------------------------------------------------------
    # Clip visualization to Liberia
    # --------------------------------------------------------

    lon_mask = (
        (lon >= WEST)
        & (lon <= EAST)
    )

    lat_mask = (
        (lat >= SOUTH)
        & (lat <= NORTH)
    )

    if not np.any(lon_mask):
        raise ValueError(
            "Raster does not overlap Liberia longitude range."
        )

    if not np.any(lat_mask):
        raise ValueError(
            "Raster does not overlap Liberia latitude range."
        )

    rainfall_lbr = rainfall[
        lat_mask
    ][:, lon_mask]

    lon_lbr = lon[
        lon_mask
    ]

    lat_lbr = lat[
        lat_mask
    ]

    # Interpolate between the source grid cells for smoother class boundaries.
    # This improves display quality without changing the underlying forecast
    # values or rainfall thresholds.
    from rasterio.features import geometry_mask
    from rasterio.transform import from_origin

    smooth_factor = 12
    smooth_lon = np.linspace(
        lon_lbr[0], lon_lbr[-1], (len(lon_lbr) - 1) * smooth_factor + 1
    )
    smooth_lat = np.linspace(
        lat_lbr[0], lat_lbr[-1], (len(lat_lbr) - 1) * smooth_factor + 1
    )
    # Separable bilinear interpolation using NumPy only.
    rainfall_lon_smooth = np.vstack([
        np.interp(smooth_lon, lon_lbr, row) for row in rainfall_lbr
    ])
    rainfall_lbr = np.column_stack([
        np.interp(smooth_lat, lat_lbr, rainfall_lon_smooth[:, column])
        for column in range(rainfall_lon_smooth.shape[1])
    ])

    # Apply the Liberia outline at the finer display resolution, keeping the
    # interpolated shading inside the national boundary.
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
        out_shape=rainfall_lbr.shape,
        transform=fine_transform,
        invert=True,
    )
    rainfall_lbr = np.where(country_mask[::-1, :], rainfall_lbr, np.nan)
    lon_lbr = smooth_lon
    lat_lbr = smooth_lat

    print(
        f"Liberia raster size: "
        f"{rainfall_lbr.shape}"
    )

    print(
        f"Minimum rainfall: "
        f"{np.nanmin(rainfall_lbr):.2f} mm"
    )

    print(
        f"Maximum rainfall: "
        f"{np.nanmax(rainfall_lbr):.2f} mm"
    )

    print(
        f"Mean rainfall: "
        f"{np.nanmean(rainfall_lbr):.2f} mm"
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig = plt.figure(figsize=FIGURE_SIZE, facecolor="white")
    ax = fig.add_axes([0.06, 0.11, 0.80, 0.80])
    ax.set_facecolor("white")

    # --------------------------------------------------------
    # Rainfall normalization
    # --------------------------------------------------------

    cmap = ListedColormap(RAINFALL_COLORS)
    norm = BoundaryNorm(RAINFALL_LEVELS, cmap.N, extend="max")

    # --------------------------------------------------------
    # Rainfall raster
    # --------------------------------------------------------

    mesh = ax.pcolormesh(
        lon_lbr,
        lat_lbr,
        rainfall_lbr,
        cmap=cmap,
        norm=norm,
        shading="auto"
    )

    # --------------------------------------------------------
    # County boundaries
    # --------------------------------------------------------

    counties.boundary.plot(
        ax=ax,
        linewidth=0.55,
        edgecolor="black"
    )

    # --------------------------------------------------------
    # County labels
    # --------------------------------------------------------

    for _, county in counties.iterrows():

        centroid = county.geometry.representative_point()

        x = centroid.x
        y = centroid.y

        # Only label counties within the map extent.
        if (
            WEST <= x <= EAST
            and SOUTH <= y <= NORTH
        ):

            ax.text(
                x,
                y,
                county["NAME_1"],
                fontsize=8,
                ha="center",
                va="center",
                color="black",
                fontweight="bold",
                zorder=10
            )

    # --------------------------------------------------------
    # Map extent
    # --------------------------------------------------------

    # Framed coordinate axes with nautical-style degree labels.
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

    # Match the GFS heading placement and typography.
    valid_date = get_valid_date(tif_file).strftime("%B %d, %Y").upper()
    valid_date = valid_date.replace(" 0", " ")
    ax.text(0.55, 0.955,
            f"LIBERIA METEOROLOGICAL SERVICE\n{valid_date}\nPRECIPITATION FORECAST",
            transform=ax.transAxes, ha="center", va="center", fontsize=9,
            fontweight="bold", family="serif", zorder=20)
    # Compact compass rose, positioned in the upper-right white space.
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
            (0.025, 0.985),
            xycoords="axes fraction",
            frameon=False,
            box_alignment=(0, 1),
            pad=0,
            zorder=30,
            annotation_clip=False
        )

        ax.add_artist(logo_box)

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        orientation="horizontal",
        fraction=0.045,
        pad=0.075,
        boundaries=RAINFALL_LEVELS,
        ticks=[5, 17.5, 37.5, 62.5, 87.5, 200],
    )
    cbar.ax.set_xticklabels(RAINFALL_LABELS)
    cbar.ax.text(1.025, 1.25, ">300 mm", transform=cbar.ax.transAxes,
                 ha="center", va="top", fontsize=8)
    cbar.set_label("Precipitation (mm)", fontsize=10, fontweight="bold")

    fig.text(0.46, 0.025,
             "LIBERIA METEOROLOGICAL SERVICE  |  ECMWF IFS OPEN DATA",
             ha="center", va="center", fontsize=8, fontweight="bold")

    fig.savefig(
        output_png,
        dpi=DPI,
        bbox_inches=None,
        facecolor="white"
    )

    plt.close(fig)

    print(
        f"Saved PNG:\n{output_png}"
    )


# ============================================================
# 7. RENDER ALL MAPS
# ============================================================

print("\n" + "=" * 70)
print("RENDERING PRECIPITATION MAPS")
print("=" * 70)

successful = 0

for tif_file in geotiff_files:

    try:

        render_map(tif_file)

        successful += 1

    except Exception as exc:

        print(
            f"\nERROR rendering {tif_file.name}:"
        )

        print(exc)


# ============================================================
# 8. COPY FINAL LMS PRODUCTS
# ============================================================

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

print("\n" + "=" * 70)
print("CREATING FINAL LMS PRECIPITATION PRODUCTS")
print("=" * 70)

period_to_hours = {
    "00-24h": "024h",
    "24-48h": "048h",
    "48-72h": "072h",
    "72-96h": "096h",
    "96-120h": "120h",
    "120-144h": "144h",
}

final_count = 0

for tif_file in geotiff_files:

    period = tif_file.stem.split("_")[-1]

    if period not in period_to_hours:
        print(
            f"Skipping unknown forecast period: "
            f"{tif_file.name}"
        )
        continue

    source_png = tif_file.with_suffix(".png")

    if not source_png.exists():
        print(
            f"Skipping missing PNG: "
            f"{source_png.name}"
        )
        continue

    output_name = (
        f"LMS_PRECIP_AFRICA_"
        f"{DATE_NOW}_"
        f"{period_to_hours[period]}.png"
    )

    final_png = FINAL_LMS_DIR / output_name

    shutil.copy2(
        source_png,
        final_png
    )

    final_count += 1

    print(
        f"Copied:\n"
        f"  {source_png}\n"
        f"     -> {final_png}"
    )

print(
    f"\nFinal LMS products: "
    f"{final_count}/{len(geotiff_files)}"
)

print(
    f"FINAL_LMS directory:\n"
    f"{FINAL_LMS_DIR}"
)


# ============================================================
# 9. SUMMARY
# ============================================================

# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PRECIPITATION MAP RENDERING COMPLETE")
print("=" * 70)

print(
    f"\nSuccessful maps: "
    f"{successful}/{len(geotiff_files)}"
)

print("\nGenerated PNG files:")

for png in sorted(
    ECMWF_DIR.glob(
        "ECMWF_PRECIP_AFRICA_*.png"
    )
):

    print(
        f"  {png.name}"
    )

print("\nOutput directory:")
print(ECMWF_DIR)

print("\nLMS precipitation map rendering finished.")
