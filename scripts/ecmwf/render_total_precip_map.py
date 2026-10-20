#!/usr/bin/env python3

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
import numpy as np
from matplotlib.patches import Polygon
import rasterio
from rasterio.features import geometry_mask
from rasterio.transform import from_origin
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ============================================================
# LMS 6-DAY TOTAL PRECIPITATION MAP RENDERER
# ============================================================

BASE_DIR = Path.home() / "LMS_WEATHER_MAPS"

MAP_DIR = BASE_DIR / "maps"

SHAPEFILE = (
    BASE_DIR
    / "shp_LBR"
    / "gadm41_LBR_1.shp"
)

WEST = -12
EAST = -7
SOUTH = 4
NORTH = 9

RAINFALL_LEVELS = [
    0,
    10,
    25,
    50,
    75,
    100,
    300,
]

COLORMAP = ListedColormap([
    "#E64A0B",
    "#E99413",
    "#F2C94C",
    "#B6CF78",
    "#6B985D",
    "#2F6B47",
])


# ============================================================
# FIND LATEST FORECAST DIRECTORY
# ============================================================

date_dirs = sorted(
    [
        p for p in MAP_DIR.iterdir()
        if p.is_dir() and p.name.isdigit() and len(p.name) == 8
    ]
)

if not date_dirs:
    raise FileNotFoundError(
        "No forecast-date directory found."
    )

DATE_DIR = date_dirs[-1]
ECMWF_DIR = DATE_DIR / "ECMWF"

INPUT_TIF = (
    ECMWF_DIR
    / f"ECMWF_TOTAL_PRECIP_AFRICA_{DATE_DIR.name}_6DAY.tif"
)

OUTPUT_PNG = (
    BASE_DIR
    / "archive"
    / "ECMWF"
    / DATE_DIR.name
    / "FINAL_LMS"
    / f"LMS_TOTAL_PRECIP_AFRICA_{DATE_DIR.name}_6DAY.png"
)


# ============================================================
# CHECK INPUTS
# ============================================================

if not INPUT_TIF.exists():
    raise FileNotFoundError(
        f"6-day precipitation GeoTIFF not found:\n{INPUT_TIF}"
    )

if not SHAPEFILE.exists():
    raise FileNotFoundError(
        f"Liberia shapefile not found:\n{SHAPEFILE}"
    )


# ============================================================
# READ LIBERIA COUNTY BOUNDARIES
# ============================================================

counties = gpd.read_file(SHAPEFILE)

counties = counties.to_crs("EPSG:4326")


# ============================================================
# CREATE MAP
# ============================================================

fig = plt.figure(
    figsize=(10, 8),
    dpi=200
)

ax = plt.axes(
    projection=ccrs.PlateCarree()
)

ax.set_extent(
    [WEST, EAST, SOUTH, NORTH],
    crs=ccrs.PlateCarree()
)

longitude_ticks = np.arange(-12, -6, 1)
latitude_ticks = np.arange(4, 10, 1)
ax.set_xticks(longitude_ticks, crs=ccrs.PlateCarree())
ax.set_yticks(latitude_ticks, crs=ccrs.PlateCarree())
ax.set_xticklabels([f"{abs(int(value))}°W" for value in longitude_ticks], fontsize=8)
ax.set_yticklabels([f"{int(value)}°N" for value in latitude_ticks],
                   fontsize=8, rotation=90, va="center")
ax.tick_params(top=True, right=True, labeltop=False, labelright=False,
               direction="out", length=3, pad=3)


# ============================================================
# BASE FEATURES
# ============================================================

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


# ============================================================
# PLOT PRECIPITATION
# ============================================================

with rasterio.open(INPUT_TIF) as src:

    raster = src.read(1).astype(float)
    transform = src.transform
    if src.nodata is not None:
        raster[raster == src.nodata] = np.nan

raster[~np.isfinite(raster)] = np.nan
raster = np.maximum(raster, 0)

height, width = raster.shape
lon = np.arange(width) * transform.a + transform.c + transform.a / 2
lat = np.arange(height) * transform.e + transform.f + transform.e / 2
if lat[0] > lat[-1]:
    lat = lat[::-1]
    raster = raster[::-1, :]

# Match the RH display layer: interpolate for smoother class transitions,
# then clip the fine display grid to Liberia's boundary.
smooth_factor = 6
smooth_lon = np.linspace(lon[0], lon[-1], (len(lon) - 1) * smooth_factor + 1)
smooth_lat = np.linspace(lat[0], lat[-1], (len(lat) - 1) * smooth_factor + 1)

def interpolate_valid(source_x, target_x, values):
    valid = np.isfinite(values)
    if valid.sum() < 2:
        return np.full(target_x.shape, np.nan)
    return np.interp(target_x, source_x[valid], values[valid])

smooth_lon_data = np.vstack([
    interpolate_valid(lon, smooth_lon, row) for row in raster
])
smooth_raster = np.column_stack([
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
    counties.geometry,
    out_shape=smooth_raster.shape,
    transform=fine_transform,
    invert=True,
)
smooth_raster = np.where(country_mask[::-1, :], smooth_raster, np.nan)
smooth_raster = np.where(smooth_raster >= 0, smooth_raster, np.nan)

# ------------------------------------------------------------
# DISCRETE SIX-CLASS RAINFALL NORMALIZATION
# ------------------------------------------------------------
norm = BoundaryNorm(
    RAINFALL_LEVELS,
    ncolors=COLORMAP.N,
    clip=False
)

img = ax.pcolormesh(
    smooth_lon,
    smooth_lat,
    smooth_raster,
    transform=ccrs.PlateCarree(),
    cmap=COLORMAP,
    norm=norm,
    shading="auto",
    zorder=1,
)


# ============================================================
# COUNTY BOUNDARIES
# ============================================================

counties.boundary.plot(
    ax=ax,
    linewidth=0.7,
    edgecolor="black",
    zorder=4
)


# ============================================================
# COUNTY LABELS
# ============================================================

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
        zorder=5
    )


# ============================================================
# COMPASS ROSE (matching the RH map)
# ============================================================

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
for label, dx, dy in (("N", 0, 0.10), ("S", 0, -0.10),
                      ("E", 0.10, 0), ("W", -0.10, 0)):
    ax.text(cx + dx, cy + dy, label, transform=ax.transAxes,
            ha="center", va="center", fontsize=9, family="serif",
            zorder=23, clip_on=False)


# ============================================================
# TITLE
# ============================================================

ax.set_title(
    "LIBERIA METEOROLOGICAL SERVICE",
    fontsize=16,
    fontweight="bold",
    pad=20
)

ax.text(
    0.5,
    1.015,
    "6-DAY TOTAL PRECIPITATION FORECAST",
    transform=ax.transAxes,
    ha="center",
    va="bottom",
    fontsize=12,
    fontweight="bold"
)


# ============================================================
# ============================================================
# COLORBAR
# ============================================================

# ------------------------------------------------------------
# SIX-CLASS TOTAL PRECIPITATION LEGEND
# Ends at >100 mm with no 100–300 mm visual gap.
# ------------------------------------------------------------
legend_boundaries = [0, 10, 25, 50, 75, 100, 101]

legend_norm = BoundaryNorm(
    legend_boundaries,
    ncolors=COLORMAP.N,
    clip=False
)

sm = plt.cm.ScalarMappable(
    norm=legend_norm,
    cmap=COLORMAP
)
sm.set_array([])

cbar = plt.colorbar(
    sm,
    ax=ax,
    orientation="horizontal",
    pad=0.055,
    shrink=0.85,
    aspect=35,
    boundaries=legend_boundaries,
    spacing="proportional"
)

cbar.set_label(
    "Total precipitation (mm)",
    fontsize=10
)

cbar.set_ticks([
    5,
    17.5,
    37.5,
    62.5,
    87.5,
    100.5
])

cbar.set_ticklabels([
    "0–10",
    "10–25",
    "25–50",
    "50–75",
    "75–100",
    ">100"
])


# ============================================================
# FOOTER
# ============================================================

fig.text(
    0.5,
    0.095,
    "Liberia Meteorological Service (LMS) | Ministry of Transport | ECMWF",
    ha="center",
    fontsize=8
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_PNG,
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
print(
    f"Final LMS product: "
    f"{OUTPUT_PNG.name}"
)

print("\n" + "=" * 70)
print("6-DAY TOTAL PRECIPITATION MAP RENDERING COMPLETE")
print("=" * 70)
print(f"Input : {INPUT_TIF}")
print(f"Final : {OUTPUT_PNG}")
print("=" * 70)
