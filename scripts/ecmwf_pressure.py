#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import xarray as xr
import rioxarray
from ecmwf.opendata import Client

# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ecmwf_config as config


# ============================================================
# DATE AND OUTPUT DIRECTORIES
# ============================================================

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

MAP_DIR = config.MAP_DIR / DATE_NOW / "ECMWF"
MAP_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# ECMWF CONFIGURATION
# ============================================================

MODEL = "ifs"
RESOLUTION = "0p25"
SOURCE = config.SOURCE
PARAMETER = config.VARIABLES["pressure"]

FORECAST_STEPS = [24, 48, 72, 96, 120, 144]


# ============================================================
# ECMWF CLIENT
# ============================================================

client = Client(
    source=SOURCE,
    model=MODEL,
    resol=RESOLUTION,
    maximum_retries=config.MAX_RETRIES,
    retry_after=config.RETRY_AFTER,
)


# ============================================================
# HELPER: FIND FORECAST STEP
# ============================================================

def find_ecmwf_step(available_steps, requested_hours):

    available_steps = np.atleast_1d(
        np.asarray(available_steps)
    )

    requested_step = np.timedelta64(
        requested_hours,
        "h"
    )

    differences = np.abs(
        available_steps - requested_step
    )

    nearest_index = int(
        np.argmin(differences)
    )

    actual_step = available_steps[nearest_index]

    difference = differences[nearest_index]

    difference_hours = (
        difference /
        np.timedelta64(1, "h")
    )

    if difference_hours > 0.01:
        raise ValueError(
            f"Unable to find ECMWF step near "
            f"+{requested_hours} hours."
        )

    return actual_step


# ============================================================
# DOWNLOAD
# ============================================================

global_grib = (
    MAP_DIR /
    f"ECMWF_PRESSURE_GLOBAL_{DATE_NOW}.grib2"
)

print("=" * 70)
print("LMS ECMWF PRESSURE FORECAST PROCESSING")
print("=" * 70)
print(f"Date: {DATE_NOW}")
print(f"Parameter: {PARAMETER}")
print(f"Forecast steps: {FORECAST_STEPS}")
print(f"Output directory: {MAP_DIR}")
print("=" * 70)

print("\nDownloading ECMWF MSLP forecast...")

client.retrieve(
    date=DATE_NOW,
    time=0,
    step=FORECAST_STEPS,
    stream="oper",
    type="fc",
    levtype="sfc",
    param=PARAMETER,
    target=str(global_grib),
)

print(f"Downloaded: {global_grib}")


# ============================================================
# OPEN GRIB
# ============================================================

print("\nOpening GRIB file...")

ds = xr.open_dataset(
    global_grib,
    engine="cfgrib",
    backend_kwargs={
        "indexpath": ""
    }
)

print(ds)


# ============================================================
# IDENTIFY MSLP VARIABLE
# ============================================================

if "msl" not in ds.data_vars:
    raise KeyError(
        "Expected variable 'msl' was not found."
    )

msl = ds["msl"]


# ============================================================
# CONVERT Pa → hPa
# ============================================================

msl_hpa = msl / 100.0

msl_hpa.attrs = msl.attrs.copy()
msl_hpa.attrs["units"] = "hPa"
msl_hpa.attrs["long_name"] = (
    "Mean sea level pressure"
)


# ============================================================
# LIBERIA EXTRACTION
# ============================================================

print("\nExtracting Liberia domain...")

liberia = msl_hpa.sel(
    latitude=slice(
        config.LIBERIA_BBOX["north"],
        config.LIBERIA_BBOX["south"]
    ),
    longitude=slice(
        config.LIBERIA_BBOX["west"],
        config.LIBERIA_BBOX["east"]
    ),
)

print(
    f"  Shape: {liberia.shape}"
)

print(
    f"  Latitude: "
    f"{float(liberia.latitude.min()):.2f} "
    f"to "
    f"{float(liberia.latitude.max()):.2f}"
)

print(
    f"  Longitude: "
    f"{float(liberia.longitude.min()):.2f} "
    f"to "
    f"{float(liberia.longitude.max()):.2f}"
)


# ============================================================
# SAVE SIX FORECAST GEOTIFFS
# ============================================================

print("\nSaving Liberia MSLP GeoTIFFs...")

for requested_hours in FORECAST_STEPS:

    actual_step = find_ecmwf_step(
        ds.step.values,
        requested_hours
    )

    pressure = liberia.sel(
        step=actual_step
    ).squeeze()

    pressure = pressure.rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude",
        inplace=False
    )

    pressure = pressure.rio.write_crs(
        "EPSG:4326",
        inplace=False
    )

    output_file = (
        MAP_DIR /
        f"ECMWF_PRESSURE_AFRICA_"
        f"{DATE_NOW}_"
        f"{requested_hours:03d}h.tif"
    )

    pressure.rio.to_raster(
        output_file
    )

    print(
        f"\n+{requested_hours} h"
    )

    print(
        f"  GeoTIFF: "
        f"{output_file.name}"
    )

    print(
        f"  Minimum: "
        f"{float(pressure.min()):.2f} hPa"
    )

    print(
        f"  Maximum: "
        f"{float(pressure.max()):.2f} hPa"
    )

    print(
        f"  Mean: "
        f"{float(pressure.mean()):.2f} hPa"
    )


print("\n" + "=" * 70)
print("LMS PRESSURE DATA PROCESSING COMPLETE")
print("=" * 70)
print("Output directory:")
print(MAP_DIR)
print("=" * 70)
