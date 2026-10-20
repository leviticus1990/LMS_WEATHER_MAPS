#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np
import xarray as xr
import rioxarray
from ecmwf.opendata import Client

# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ecmwf_config as config


# ============================================================
# OPERATIONAL CONFIGURATION
# ============================================================

BASE_DIR = config.BASE_DIR
DATA_DIR = config.DATA_DIR
GRIB_DIR = config.GRIB_DIR
MAP_DIR = config.MAP_DIR

MODEL = "ifs"
RESOLUTION = "0p25"
SOURCE = config.SOURCE

PARAMETER = config.VARIABLES["temperature"]

FORECAST_STEPS = [
    24,
    48,
    72,
    96,
    120,
    144,
]

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

ECMWF_DIR = (
    MAP_DIR
    / DATE_NOW
    / "ECMWF"
)

ECMWF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRIB_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LIBERIA DOMAIN
# ============================================================

WEST = config.LIBERIA_BBOX["west"]
EAST = config.LIBERIA_BBOX["east"]
SOUTH = config.LIBERIA_BBOX["south"]
NORTH = config.LIBERIA_BBOX["north"]


# ============================================================
# OUTPUT FILE
# ============================================================

GLOBAL_GRIB = (
    ECMWF_DIR
    / f"ECMWF_TEMPERATURE_GLOBAL_{DATE_NOW}.grib2"
)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("LMS ECMWF 2-METRE TEMPERATURE FORECAST")
print("=" * 70)

print(f"Date       : {DATE_NOW}")
print("Cycle      : 00 UTC")
print(f"Source     : {SOURCE}")
print(f"Model      : {MODEL}")
print(f"Resolution : {RESOLUTION}")
print(f"Parameter  : {PARAMETER}")
print(f"Steps      : {FORECAST_STEPS}")
print(
    f"Domain     : "
    f"{WEST}°W to {EAST}°W, "
    f"{SOUTH}°N to {NORTH}°N"
)

print("=" * 70)


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
# DOWNLOAD
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOADING ECMWF 2-METRE TEMPERATURE")
print("=" * 70)

print(f"Output file:\n{GLOBAL_GRIB}")

try:

    client.retrieve(
        date=DATE_NOW,
        time=0,
        step=FORECAST_STEPS,
        stream="oper",
        type="fc",
        levtype="sfc",
        param=PARAMETER,
        target=str(GLOBAL_GRIB)
    )

    print("\nDownload completed successfully.")

except Exception as e:

    print("\nERROR DURING DOWNLOAD:")
    print(e)

    raise


# ============================================================
# CHECK DOWNLOADED FILE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DOWNLOADED FILE")
print("=" * 70)

if not GLOBAL_GRIB.exists():

    raise FileNotFoundError(
        f"GRIB2 file was not created:\n{GLOBAL_GRIB}"
    )

print(
    f"GRIB2 size: "
    f"{GLOBAL_GRIB.stat().st_size / 1024 / 1024:.2f} MB"
)


# ============================================================
# OPEN GRIB
# ============================================================

print("\n" + "=" * 70)
print("OPENING TEMPERATURE GRIB")
print("=" * 70)

ds = xr.open_dataset(
    GLOBAL_GRIB,
    engine="cfgrib",
    backend_kwargs={
        "indexpath": ""
    }
)

print("Variables:", list(ds.data_vars))
print("Dimensions:", dict(ds.sizes))


# ============================================================
# TEMPERATURE VARIABLE
# ============================================================

temperature_k = ds["t2m"]

print(
    "\nGlobal temperature range:"
)

print(
    f"Minimum: {float(temperature_k.min()):.2f} K"
)

print(
    f"Maximum: {float(temperature_k.max()):.2f} K"
)


# ============================================================
# CONVERT KELVIN TO CELSIUS
# ============================================================

temperature_c = (
    temperature_k - 273.15
)

temperature_c.attrs = temperature_k.attrs.copy()

temperature_c.attrs["units"] = "°C"

temperature_c.attrs["long_name"] = (
    "2 metre temperature"
)


# ============================================================
# AVAILABLE FORECAST STEPS
# ============================================================

available_steps = np.atleast_1d(
    np.asarray(ds.step.values)
)

print("\nAvailable forecast steps:")

for step in available_steps:

    hours = int(
        step / np.timedelta64(1, "h")
    )

    print(f"  +{hours:3d} h")


# ============================================================
# FIND ECMWF STEP
# ============================================================

def find_ecmwf_step(
    available_steps,
    requested_hours
):

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

    actual_step = (
        available_steps[nearest_index]
    )

    difference = differences[
        nearest_index
    ]

    difference_hours = (
        difference
        / np.timedelta64(1, "h")
    )

    if difference_hours > 0.01:

        raise ValueError(
            f"Unable to find ECMWF step "
            f"near +{requested_hours} hours."
        )

    return actual_step


# ============================================================
# PROCESS EACH FORECAST DAY
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING LIBERIA TEMPERATURE FORECASTS")
print("=" * 70)

for requested_hours in FORECAST_STEPS:

    actual_step = find_ecmwf_step(
        available_steps,
        requested_hours
    )

    if "step" in temperature_c.dims:

        temperature = (
            temperature_c
            .sel(step=actual_step)
            .squeeze()
        )

    else:

        temperature = (
            temperature_c
            .squeeze()
        )


    # --------------------------------------------------------
    # LIBERIA EXTRACTION
    # --------------------------------------------------------

    liberia = temperature.sel(
        latitude=slice(
            NORTH,
            SOUTH
        ),
        longitude=slice(
            WEST,
            EAST
        )
    )


    liberia = liberia.squeeze()


    # --------------------------------------------------------
    # SPATIAL METADATA
    # --------------------------------------------------------

    liberia.attrs["units"] = "°C"

    liberia.attrs["long_name"] = (
        "2 metre temperature over Liberia"
    )

    liberia = (
        liberia
        .rio.set_spatial_dims(
            x_dim="longitude",
            y_dim="latitude"
        )
        .rio.write_crs(
            "EPSG:4326"
        )
    )


    # --------------------------------------------------------
    # OUTPUT
    # --------------------------------------------------------

    output_tif = (
        ECMWF_DIR
        / f"ECMWF_TEMPERATURE_AFRICA_"
        f"{DATE_NOW}_{requested_hours:03d}h.tif"
    )

    liberia.rio.to_raster(
        output_tif,
        compress="deflate"
    )


    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    print(
        f"\n+{requested_hours:3d} h"
    )

    print(
        f"  GeoTIFF: {output_tif.name}"
    )

    print(
        f"  Minimum: "
        f"{float(liberia.min()):.2f} °C"
    )

    print(
        f"  Maximum: "
        f"{float(liberia.max()):.2f} °C"
    )

    print(
        f"  Mean: "
        f"{float(liberia.mean()):.2f} °C"
    )


# ============================================================
# CLOSE DATASET
# ============================================================

ds.close()


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 70)
print("LMS TEMPERATURE DATA PROCESSING COMPLETE")
print("=" * 70)

print(
    f"Output directory:\n{ECMWF_DIR}"
)

print("=" * 70)
