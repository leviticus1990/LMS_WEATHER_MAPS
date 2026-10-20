#!/usr/bin/env python3
"""
LMS ECMWF Tmin Acquisition

Downloads ECMWF IFS 3-hourly minimum temperature at 2 metres (mn2t3),
aggregates the fields into six 24-hour minimum-temperature periods,
extracts the Liberia domain, and writes GeoTIFF outputs.

Products:
    024h = minimum of +003h to +024h
    048h = minimum of +027h to +048h
    072h = minimum of +051h to +072h
    096h = minimum of +075h to +096h
    120h = minimum of +099h to +120h
    144h = minimum of +123h to +144h
"""

from pathlib import Path
from datetime import datetime, timezone
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin

from ecmwf.opendata import Client

import config.ecmwf_config as config


# ----------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------

BASE_DIR = Path("/home/james/LMS_WEATHER_MAPS")

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

OUTPUT_DIR = BASE_DIR / "maps" / DATE_NOW / "ECMWF"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_GRIB = OUTPUT_DIR / f"ECMWF_TMIN_GLOBAL_{DATE_NOW}.grib2"

SOURCE = config.SOURCE
MODEL = "ifs"
RESOLUTION = "0p25"

PARAMETER = "mn2t3"

FORECAST_STEPS = list(range(3, 145, 3))

FORECAST_DAYS = {
    "024h": list(range(3, 25, 3)),
    "048h": list(range(27, 49, 3)),
    "072h": list(range(51, 73, 3)),
    "096h": list(range(75, 97, 3)),
    "120h": list(range(99, 121, 3)),
    "144h": list(range(123, 145, 3)),
}

WEST = -11.7
EAST = -7.1
SOUTH = 4.2
NORTH = 8.7

CRS = "EPSG:4326"


# ----------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------

print("=" * 70)
print("LMS ECMWF MINIMUM TEMPERATURE ACQUISITION")
print("=" * 70)
print(f"Forecast date: {DATE_NOW}")
print(f"Parameter: {PARAMETER}")
print(f"Forecast steps: {len(FORECAST_STEPS)}")
print(f"Output directory: {OUTPUT_DIR}")
print()


# ----------------------------------------------------------------------
# ECMWF DOWNLOAD
# ----------------------------------------------------------------------

print("-" * 70)
print("Downloading ECMWF mn2t3 data")
print("-" * 70)

client = Client(
    source=config.SOURCE,
    model="ifs",
    resol="0p25",
)

TEMP_GRIB = GLOBAL_GRIB.with_name(GLOBAL_GRIB.name + ".part")

if TEMP_GRIB.exists():
    TEMP_GRIB.unlink()

client = Client(
    source=config.SOURCE,
    model="ifs",
    resol="0p25",
)

print("Downloading to temporary GRIB:", TEMP_GRIB)

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
        raise RuntimeError("TMIN download produced no usable file.")

    print("Validating temporary GRIB with cfgrib")

    with xr.open_dataset(
        TEMP_GRIB,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    ) as check_ds:
        if PARAMETER not in check_ds.data_vars:
            raise RuntimeError(
                f"Expected variable '{PARAMETER}' not found in temporary GRIB."
            )

        actual_steps = [
            int(step / np.timedelta64(1, "h"))
            for step in check_ds["step"].values
        ]

        if actual_steps != FORECAST_STEPS:
            raise RuntimeError(
                "TMIN temporary GRIB forecast-step validation failed.\\n"
                f"Expected: {FORECAST_STEPS}\\n"
                f"Actual:   {actual_steps}"
            )

    TEMP_GRIB.replace(GLOBAL_GRIB)
    print("Temporary GRIB validation: PASS")
    print("Verified GRIB promoted to:", GLOBAL_GRIB)

except Exception:
    if TEMP_GRIB.exists():
        print("Temporary download retained for diagnosis:", TEMP_GRIB)
    raise

print()

# ----------------------------------------------------------------------
# OPEN VERIFIED GRIB
# ----------------------------------------------------------------------

print("-" * 70)
print("Opening verified GRIB with cfgrib")
print("-" * 70)

ds = xr.open_dataset(
    GLOBAL_GRIB,
    engine="cfgrib",
    backend_kwargs={"indexpath": ""},
)

if PARAMETER not in ds.data_vars:
    ds.close()
    raise RuntimeError(
        f"Expected variable '{PARAMETER}' not found in GRIB."
    )

tmin_k = ds[PARAMETER]

print(f"Variable: {PARAMETER}")
print(f"Units: {tmin_k.attrs.get('units')}")
print(f"Long name: {tmin_k.attrs.get('long_name')}")
print(f"Number of steps: {tmin_k.sizes['step']}")

actual_steps = [
    int(step / np.timedelta64(1, "h"))
    for step in ds["step"].values
]

if actual_steps != FORECAST_STEPS:
    ds.close()
    raise RuntimeError(
        "Forecast step validation failed.\\n"
        f"Expected: {FORECAST_STEPS}\\n"
        f"Actual:   {actual_steps}"
    )

print("Forecast step validation: PASS")
print()


# ----------------------------------------------------------------------
# CONVERT KELVIN TO CELSIUS
# ----------------------------------------------------------------------

tmin_c = tmin_k - 273.15
tmin_c.attrs["units"] = "°C"
tmin_c.attrs["long_name"] = "2 metre minimum temperature"


# ----------------------------------------------------------------------
# EXTRACT LIBERIA DOMAIN
# ----------------------------------------------------------------------

print("-" * 70)
print("Extracting Liberia domain")
print("-" * 70)

liberia = tmin_c.sel(
    longitude=slice(WEST, EAST),
    latitude=slice(NORTH, SOUTH),
)

print(
    f"Domain: W={WEST}, E={EAST}, "
    f"S={SOUTH}, N={NORTH}"
)
print(
    f"Grid: {liberia.sizes['latitude']} x "
    f"{liberia.sizes['longitude']}"
)
print()


# ----------------------------------------------------------------------
# WRITE SIX 24-HOUR TMIN GEOTIFFS
# ----------------------------------------------------------------------

print("-" * 70)
print("Creating six 24-hour Tmin GeoTIFFs")
print("-" * 70)

created_files = []

for code, steps in FORECAST_DAYS.items():

    print(f"\n{code}: fields +{steps[0]:03d}h to +{steps[-1]:03d}h")

    selected = tmin_c.sel(
        step=[
            np.timedelta64(step, "h")
            for step in steps
        ]
    )

    daily_min = selected.min(
        dim="step",
        skipna=True,
    )

    daily_min = daily_min.sel(
        longitude=slice(WEST, EAST),
        latitude=slice(NORTH, SOUTH),
    )

    data = daily_min.values.astype("float32")

    if not np.isfinite(data).any():
        raise RuntimeError(
            f"No valid Tmin data found for {code}."
        )

    min_value = float(np.nanmin(data))
    max_value = float(np.nanmax(data))
    mean_value = float(np.nanmean(data))

    print(
        f"  Liberia Tmin: "
        f"min={min_value:.2f}°C "
        f"max={max_value:.2f}°C "
        f"mean={mean_value:.2f}°C"
    )

    lons = daily_min.longitude.values
    lats = daily_min.latitude.values

    xres = float(abs(lons[1] - lons[0]))
    yres = float(abs(lats[1] - lats[0]))

    transform = from_origin(
        float(lons.min()) - xres / 2,
        float(lats.max()) + yres / 2,
        xres,
        yres,
    )

    output_tif = (
        OUTPUT_DIR
        / f"ECMWF_TMIN_AFRICA_{DATE_NOW}_{code}.tif"
    )

    with rasterio.open(
        output_tif,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        compress="lzw",
    ) as dst:

        dst.write(data, 1)
        dst.set_band_description(
            1,
            "2 metre minimum temperature (°C)",
        )

    created_files.append(output_tif)

    print(f"  Created: {output_tif}")


# ----------------------------------------------------------------------
# FINAL VALIDATION
# ----------------------------------------------------------------------

ds.close()

print()
print("-" * 70)
print("TMIN OUTPUT VALIDATION")
print("-" * 70)

expected_count = 6

if len(created_files) != expected_count:
    raise RuntimeError(
        f"Expected {expected_count} Tmin GeoTIFFs, "
        f"created {len(created_files)}."
    )

for tif in created_files:
    if not tif.exists():
        raise RuntimeError(f"Missing output: {tif}")

    print(f"PASS: {tif.name}")

print()
print("=" * 70)
print(f"TMIN ACQUISITION COMPLETE: {len(created_files)}/6")
print("=" * 70)
