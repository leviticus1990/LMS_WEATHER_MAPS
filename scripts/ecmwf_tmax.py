#!/usr/bin/env python3
"""
LMS ECMWF 2-Metre Maximum Temperature Forecast

Source:
    ECMWF IFS Open Data

Parameter:
    mx2t3 = Maximum temperature at 2 metres in the last 3 hours

Method:
    Six 24-hour Tmax fields are calculated from consecutive
    3-hourly mx2t3 forecasts.

    Day 1: +003 to +024 h
    Day 2: +027 to +048 h
    Day 3: +051 to +072 h
    Day 4: +075 to +096 h
    Day 5: +099 to +120 h
    Day 6: +123 to +144 h

Output:
    ECMWF_TMAX_AFRICA_YYYYMMDD_024h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_048h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_072h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_096h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_120h.tif
    ECMWF_TMAX_AFRICA_YYYYMMDD_144h.tif
"""

from pathlib import Path
from datetime import datetime, timezone
import sys

import numpy as np
import xarray as xr
import rioxarray  # noqa: F401

from ecmwf.opendata import Client


# ----------------------------------------------------------------------
# PROJECT PATH
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ecmwf_config as config


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

BASE_DIR = config.BASE_DIR
DATA_DIR = config.DATA_DIR
GRIB_DIR = config.GRIB_DIR
MAP_DIR = config.MAP_DIR

MODEL = "ifs"
RESOLUTION = "0p25"
SOURCE = config.SOURCE

PARAMETER = "mx2t3"

FORECAST_STEPS = list(range(3, 145, 3))

FORECAST_DAYS = {
    "024h": list(range(3, 25, 3)),
    "048h": list(range(27, 49, 3)),
    "072h": list(range(51, 73, 3)),
    "096h": list(range(75, 97, 3)),
    "120h": list(range(99, 121, 3)),
    "144h": list(range(123, 145, 3)),
}

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")


# ----------------------------------------------------------------------
# OUTPUT DIRECTORIES
# ----------------------------------------------------------------------

ECMWF_DIR = MAP_DIR / DATE_NOW / "ECMWF"

ECMWF_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

GRIB_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ----------------------------------------------------------------------
# LIBERIA DOMAIN
# ----------------------------------------------------------------------

WEST = config.LIBERIA_BBOX["west"]
EAST = config.LIBERIA_BBOX["east"]
SOUTH = config.LIBERIA_BBOX["south"]
NORTH = config.LIBERIA_BBOX["north"]


GLOBAL_GRIB = (
    ECMWF_DIR
    / f"ECMWF_TMAX_GLOBAL_{DATE_NOW}.grib2"
)


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("LMS ECMWF 2-METRE MAXIMUM TEMPERATURE FORECAST")
print("=" * 70)

print(f"Date       : {DATE_NOW}")
print(f"Model      : {MODEL}")
print(f"Resolution : {RESOLUTION}")
print(f"Parameter  : {PARAMETER}")
print(
    f"Forecast   : +{FORECAST_STEPS[0]:03d} "
    f"to +{FORECAST_STEPS[-1]:03d} h"
)

print("\n24-HOUR AGGREGATION:")

for period, steps in FORECAST_DAYS.items():
    print(
        f"  {period}: "
        f"+{steps[0]:03d} to +{steps[-1]:03d} h "
        f"({len(steps)} fields)"
    )


# ----------------------------------------------------------------------
# ECMWF CLIENT
# ----------------------------------------------------------------------

client = Client(
    source=SOURCE,
    model=MODEL,
    resol=RESOLUTION,
    maximum_retries=config.MAX_RETRIES,
    retry_after=config.RETRY_AFTER,
)


# ----------------------------------------------------------------------
# DOWNLOAD
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("DOWNLOADING ECMWF MX2T3 DATA")
print("-" * 70)

TEMP_GRIB = GLOBAL_GRIB.with_name(GLOBAL_GRIB.name + ".part")

if TEMP_GRIB.exists():
    TEMP_GRIB.unlink()

print("Downloading to temporary file:")
print(f"  {TEMP_GRIB}")
print(f"Expected forecast steps: {len(FORECAST_STEPS)}")

try:
    client.retrieve(
        type="fc",
        stream="oper",
        date=DATE_NOW,
        time=0,
        step=FORECAST_STEPS,
        levtype="sfc",
        param=PARAMETER,
        target=str(TEMP_GRIB),
    )

    if not TEMP_GRIB.is_file() or TEMP_GRIB.stat().st_size == 0:
        raise RuntimeError("TMAX download produced no usable GRIB file.")

    print("Validating temporary TMAX GRIB...")

    with xr.open_dataset(
        TEMP_GRIB,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    ) as check_ds:
        if PARAMETER not in check_ds.data_vars:
            raise RuntimeError(
                f"Required variable '{PARAMETER}' not found in temporary GRIB."
            )

        actual_steps = [
            int(step / np.timedelta64(1, "h"))
            for step in check_ds["step"].values
        ]

        if actual_steps != FORECAST_STEPS:
            raise RuntimeError(
                "TMAX forecast-step validation failed.\n"
                f"Expected: {FORECAST_STEPS}\n"
                f"Actual:   {actual_steps}"
            )

    print("TMAX temporary GRIB validation: PASS")
    TEMP_GRIB.replace(GLOBAL_GRIB)
    print(f"Verified GRIB installed: {GLOBAL_GRIB}")

except Exception:
    if TEMP_GRIB.exists():
        print(f"Temporary GRIB retained for diagnosis: {TEMP_GRIB}")
    raise

print("Download and validation complete.")
print()


# ----------------------------------------------------------------------
# OPEN GRIB
# ----------------------------------------------------------------------

print("\n" + "-" * 70)
print("OPENING ECMWF GRIB")
print("-" * 70)

ds = xr.open_dataset(
    GLOBAL_GRIB,
    engine="cfgrib",
    backend_kwargs={
        "indexpath": ""
    },
)


# ----------------------------------------------------------------------
# VERIFY VARIABLE
# ----------------------------------------------------------------------

if PARAMETER not in ds.data_vars:
    ds.close()
    raise RuntimeError(
        f"Required variable '{PARAMETER}' "
        f"not found in ECMWF dataset."
    )

mx2t3 = ds[PARAMETER]

print(f"Variable : {PARAMETER}")
print(f"Units    : {mx2t3.attrs.get('units')}")
print(f"Long name: {mx2t3.attrs.get('long_name')}")
print(f"Steps    : {len(ds.step)}")


# ----------------------------------------------------------------------
# VERIFY ALL 48 STEPS
# ----------------------------------------------------------------------

available_steps = []

for step in ds.step.values:
    hours = int(
        step / np.timedelta64(1, "h")
    )
    available_steps.append(hours)

expected_steps = FORECAST_STEPS

if available_steps != expected_steps:
    ds.close()

    raise RuntimeError(
        "ECMWF forecast-step verification failed.\n"
        f"Expected: {expected_steps}\n"
        f"Available: {available_steps}"
    )

print("\nForecast-step verification: PASS")
print(f"Confirmed {len(available_steps)} fields.")


# ----------------------------------------------------------------------
# CONVERT KELVIN TO CELSIUS
# ----------------------------------------------------------------------

temperature_c = mx2t3 - 273.15

temperature_c.attrs = mx2t3.attrs.copy()
temperature_c.attrs["units"] = "°C"
temperature_c.attrs[
    "long_name"
] = "2 metre maximum temperature in the last 3 hours"


# ----------------------------------------------------------------------
# CREATE SIX 24-HOUR TMAX PRODUCTS
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("CALCULATING 24-HOUR TMAX PRODUCTS")
print("=" * 70)


for forecast_period, requested_hours in FORECAST_DAYS.items():

    print("\n" + "-" * 70)
    print(
        f"TMAX FORECAST PERIOD: {forecast_period}"
    )
    print("-" * 70)

    # --------------------------------------------------------------
    # Select the eight 3-hourly fields
    # --------------------------------------------------------------

    selected = temperature_c.sel(
        step=[
            np.timedelta64(hours, "h")
            for hours in requested_hours
        ]
    )

    # --------------------------------------------------------------
    # Calculate 24-hour maximum
    # --------------------------------------------------------------

    tmax = selected.max(
        dim="step",
        skipna=True,
    )

    # --------------------------------------------------------------
    # Extract Liberia
    # --------------------------------------------------------------

    liberia = tmax.sel(
        latitude=slice(NORTH, SOUTH),
        longitude=slice(WEST, EAST),
    ).squeeze()

    liberia.attrs["units"] = "°C"
    liberia.attrs[
        "long_name"
    ] = "2 metre maximum temperature over Liberia"


    # --------------------------------------------------------------
    # Write CRS
    # --------------------------------------------------------------

    liberia = (
        liberia
        .rio.set_spatial_dims(
            x_dim="longitude",
            y_dim="latitude",
        )
        .rio.write_crs("EPSG:4326")
    )


    # --------------------------------------------------------------
    # Output GeoTIFF
    # --------------------------------------------------------------

    output_tif = (
        ECMWF_DIR
        / f"ECMWF_TMAX_AFRICA_"
        f"{DATE_NOW}_{forecast_period}.tif"
    )

    liberia.rio.to_raster(
        output_tif,
        compress="deflate",
    )


    # --------------------------------------------------------------
    # Statistics
    # --------------------------------------------------------------

    minimum = float(liberia.min())
    maximum = float(liberia.max())
    mean = float(liberia.mean())


    print(
        f"Forecast period : {forecast_period}"
    )

    print(
        f"Source fields    : "
        f"+{requested_hours[0]:03d} "
        f"to +{requested_hours[-1]:03d} h"
    )

    print(
        f"GeoTIFF          : "
        f"{output_tif.name}"
    )

    print(
        f"Minimum          : "
        f"{minimum:.2f} °C"
    )

    print(
        f"Maximum          : "
        f"{maximum:.2f} °C"
    )

    print(
        f"Mean             : "
        f"{mean:.2f} °C"
    )


# ----------------------------------------------------------------------
# CLOSE DATASET
# ----------------------------------------------------------------------

ds.close()


# ----------------------------------------------------------------------
# COMPLETION
# ----------------------------------------------------------------------

print("\n" + "=" * 70)
print("LMS TMAX DATA PROCESSING COMPLETE")
print("=" * 70)

print(
    f"Output directory:\n"
    f"{ECMWF_DIR}"
)

print("\nCreated products:")

for forecast_period in FORECAST_DAYS:

    output_tif = (
        ECMWF_DIR
        / f"ECMWF_TMAX_AFRICA_"
        f"{DATE_NOW}_{forecast_period}.tif"
    )

    if output_tif.exists():
        print(
            f"  ✓ {output_tif.name}"
        )
    else:
        print(
            f"  ✗ MISSING: {output_tif.name}"
        )

print("=" * 70)
