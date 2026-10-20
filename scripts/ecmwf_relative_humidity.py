#!/usr/bin/env python3

from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np
import xarray as xr
import rioxarray
from ecmwf.opendata import Client


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ecmwf_config as config


BASE_DIR = config.BASE_DIR
DATA_DIR = config.DATA_DIR
GRIB_DIR = config.GRIB_DIR
MAP_DIR = config.MAP_DIR


MODEL = "ifs"
RESOLUTION = "0p25"
SOURCE = config.SOURCE


# ============================================================
# ECMWF PARAMETERS
# ============================================================
# ECMWF Open Data does not provide 2-metre RH directly.
# We calculate RH from:
#
#   2t = 2-metre temperature
#   2d = 2-metre dewpoint temperature
#
# RH = 100 * es(Td) / es(T)
#
# where es is saturation vapour pressure.
# ============================================================

TEMPERATURE_PARAMETER = config.VARIABLES["temperature"]
DEWPOINT_PARAMETER = config.VARIABLES["dewpoint"]


FORECAST_STEPS = [24, 48, 72, 96, 120, 144]

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")


# ============================================================
# DIRECTORIES
# ============================================================

ECMWF_DIR = MAP_DIR / DATE_NOW / "ECMWF"

ECMWF_DIR.mkdir(
    parents=True,
    exist_ok=True
)

GRIB_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DOMAIN
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
    / f"ECMWF_RELATIVE_HUMIDITY_GLOBAL_{DATE_NOW}.grib2"
)


# ============================================================
# HEADER
# ============================================================

print("\n" + "=" * 70)
print("LMS ECMWF 2-METRE RELATIVE HUMIDITY FORECAST")
print("=" * 70)

print(f"Date       : {DATE_NOW}")
print("Cycle      : 00 UTC")
print(f"Source     : {SOURCE}")
print(f"Model      : {MODEL}")
print(f"Resolution : {RESOLUTION}")
print("Parameters : 2t + 2d")
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
# DOWNLOAD 2T + 2D
# ============================================================

print("\n" + "=" * 70)
print("DOWNLOADING ECMWF 2T + 2D")
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
        param=[
            TEMPERATURE_PARAMETER,
            DEWPOINT_PARAMETER,
        ],
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
print("OPENING ECMWF 2T + 2D GRIB")
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
# VERIFY VARIABLES
# ============================================================

if "t2m" not in ds.data_vars:

    raise KeyError(
        "ECMWF 2-metre temperature variable 't2m' "
        "was not found in the GRIB file."
    )

if "d2m" not in ds.data_vars:

    raise KeyError(
        "ECMWF 2-metre dewpoint variable 'd2m' "
        "was not found in the GRIB file."
    )


# ============================================================
# READ TEMPERATURE AND DEWPOINT
# ============================================================

temperature_k = ds["t2m"]

dewpoint_k = ds["d2m"]


print("\n2-metre temperature:")
print(
    f"  Minimum: "
    f"{float(temperature_k.min()):.2f} K"
)
print(
    f"  Maximum: "
    f"{float(temperature_k.max()):.2f} K"
)


print("\n2-metre dewpoint:")
print(
    f"  Minimum: "
    f"{float(dewpoint_k.min()):.2f} K"
)
print(
    f"  Maximum: "
    f"{float(dewpoint_k.max()):.2f} K"
)


# ============================================================
# SATURATION VAPOUR PRESSURE
# ============================================================
#
# Magnus-type formulation over water:
#
# es(T) = 6.112 * exp(
#     17.67 * Tc / (Tc + 243.5)
# )
#
# Temperature is converted from Kelvin to Celsius first.
# ============================================================

def saturation_vapour_pressure(temp_k):

    temp_c = temp_k - 273.15

    return (
        6.112
        * np.exp(
            (17.67 * temp_c)
            / (temp_c + 243.5)
        )
    )


# ============================================================
# CALCULATE RELATIVE HUMIDITY
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING 2-METRE RELATIVE HUMIDITY")
print("=" * 70)

es_temperature = saturation_vapour_pressure(
    temperature_k
)

es_dewpoint = saturation_vapour_pressure(
    dewpoint_k
)

relative_humidity = (
    100.0
    * es_dewpoint
    / es_temperature
)


# ============================================================
# PHYSICAL LIMIT
# ============================================================

relative_humidity = relative_humidity.clip(
    0,
    100
)


relative_humidity.attrs = (
    temperature_k.attrs.copy()
)

relative_humidity.attrs["units"] = "%"

relative_humidity.attrs["long_name"] = (
    "2 metre relative humidity"
)


print(
    "\nGlobal relative humidity range:"
)

print(
    f"Minimum: "
    f"{float(relative_humidity.min()):.2f} %"
)

print(
    f"Maximum: "
    f"{float(relative_humidity.max()):.2f} %"
)

print(
    f"Mean: "
    f"{float(relative_humidity.mean()):.2f} %"
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
        available_steps
        - requested_step
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
# PROCESS EACH FORECAST STEP
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING LIBERIA RELATIVE HUMIDITY FORECASTS")
print("=" * 70)


for requested_hours in FORECAST_STEPS:

    actual_step = find_ecmwf_step(
        available_steps,
        requested_hours
    )


    # --------------------------------------------------------
    # SELECT FORECAST STEP
    # --------------------------------------------------------

    if "step" in relative_humidity.dims:

        relative_humidity_field = (
            relative_humidity
            .sel(step=actual_step)
            .squeeze()
        )

    else:

        relative_humidity_field = (
            relative_humidity
            .squeeze()
        )


    # --------------------------------------------------------
    # EXTRACT LIBERIA
    # --------------------------------------------------------

    liberia = (
        relative_humidity_field
        .sel(
            latitude=slice(
                NORTH,
                SOUTH
            ),
            longitude=slice(
                WEST,
                EAST
            )
        )
    )


    liberia = liberia.squeeze()


    # --------------------------------------------------------
    # SPATIAL METADATA
    # --------------------------------------------------------

    liberia.attrs["units"] = "%"

    liberia.attrs["long_name"] = (
        "2 metre relative humidity over Liberia"
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
    # OUTPUT GEOTIFF
    # --------------------------------------------------------

    output_tif = (
        ECMWF_DIR
        / f"ECMWF_RELATIVE_HUMIDITY_AFRICA_"
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
        f"  GeoTIFF: "
        f"{output_tif.name}"
    )

    print(
        f"  Minimum: "
        f"{float(liberia.min()):.2f} %"
    )

    print(
        f"  Maximum: "
        f"{float(liberia.max()):.2f} %"
    )

    print(
        f"  Mean: "
        f"{float(liberia.mean()):.2f} %"
    )


# ============================================================
# CLOSE DATASET
# ============================================================

ds.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("LMS RELATIVE HUMIDITY DATA PROCESSING COMPLETE")
print("=" * 70)

print(
    f"Output directory:\n{ECMWF_DIR}"
)

print("=" * 70)
