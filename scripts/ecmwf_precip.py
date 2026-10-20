#!/usr/bin/env python3

# ============================================================
# LMS AUTOMATED WEATHER MAP SYSTEM
# ECMWF IFS PRECIPITATION FORECAST
# ============================================================

import sys
from pathlib import Path
from datetime import datetime, timezone

# Add LMS project root to Python module search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import xarray as xr
import rioxarray
import matplotlib.pyplot as plt

from ecmwf.opendata import Client

# Central LMS ECMWF configuration
from config import ecmwf_config as config


# ============================================================
# 1. SYSTEM CONFIGURATION
# ============================================================

# ============================================================
# 1. SYSTEM CONFIGURATION
# ============================================================

LMS_MAP_ROOT = config.BASE_DIR
MAP_ROOT = config.MAP_DIR

date_now = datetime.now(
    timezone.utc
).strftime("%Y%m%d")

DATE_DIR = MAP_ROOT / date_now

ECMWF_DIR = DATE_DIR / "ECMWF"

ECMWF_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. ECMWF CONFIGURATION
# ============================================================

MODEL = "ifs"
RESOLUTION = "0p25"
SOURCE = config.SOURCE
PARAMETER = config.VARIABLES["precipitation"]

FORECAST_STEPS = config.PRECIP_FORECAST_STEPS


# ============================================================
# 3. AFRICA DOMAIN
# ============================================================

AFRICA_NORTH = 38
AFRICA_WEST = -25
AFRICA_SOUTH = -35
AFRICA_EAST = 55


# ============================================================
# 4. START MESSAGE
# ============================================================

print("=" * 70)
print("LMS AUTOMATED WEATHER MAP SYSTEM")
print("ECMWF IFS PRECIPITATION FORECAST")
print("=" * 70)

print("\nPython executable:")
print(sys.executable)

print("\nPython version:")
print(sys.version)

print("\nForecast date:")
print(date_now)

print("\nModel:")
print(MODEL)

print("\nResolution:")
print(RESOLUTION)

print("\nParameter:")
print(PARAMETER)

print("\nForecast steps:")
print(FORECAST_STEPS)

print("\nOutput directory:")
print(ECMWF_DIR)


# ============================================================
# 5. INITIALIZE ECMWF CLIENT
# ============================================================

print("\n" + "=" * 70)
print("INITIALIZING ECMWF OPEN DATA CLIENT")
print("=" * 70)

client = Client(
    source=SOURCE,
    model=MODEL,
    resol=RESOLUTION,
    maximum_retries=config.MAX_RETRIES,
    retry_after=config.RETRY_AFTER
)

print("\nECMWF Open Data client initialized successfully.")


# ============================================================
# 6. DOWNLOAD ECMWF TOTAL PRECIPITATION
# ============================================================

global_grib = (
    ECMWF_DIR /
    f"ECMWF_{date_now}_PRECIP_GLOBAL.grib2"
)

print("\n" + "=" * 70)
print("DOWNLOADING ECMWF TOTAL PRECIPITATION")
print("=" * 70)

print("\nFinal output file:")
print(global_grib)

TEMP_GRIB = global_grib.with_name(global_grib.name + ".part")
if TEMP_GRIB.exists():
    TEMP_GRIB.unlink()

print("\nDownloading to temporary file:")
print(TEMP_GRIB)
print(f"Expected precipitation steps: {config.PRECIP_FORECAST_STEPS}")

try:
    client.retrieve(
        date=date_now,
        time=0,
        step=FORECAST_STEPS,
        stream="oper",
        type="fc",
        levtype="sfc",
        param=PARAMETER,
        target=str(TEMP_GRIB),
    )

    if not TEMP_GRIB.is_file() or TEMP_GRIB.stat().st_size == 0:
        raise RuntimeError(
            "ECMWF precipitation download produced no usable GRIB file."
        )

    print("Validating temporary precipitation GRIB...")
    with xr.open_dataset(
        TEMP_GRIB,
        engine="cfgrib",
        backend_kwargs={"indexpath": ""},
    ) as check_ds:
        if PARAMETER not in check_ds.data_vars:
            raise RuntimeError(
                f"Required precipitation variable '{PARAMETER}' "
                "not found in temporary GRIB."
            )

        if "step" not in check_ds:
            raise RuntimeError(
                "Temporary precipitation GRIB has no forecast-step coordinate."
            )

        actual_steps = [
            int(step / np.timedelta64(1, "h"))
            for step in check_ds["step"].values
        ]
        expected_steps = list(config.PRECIP_FORECAST_STEPS)

        if actual_steps != expected_steps:
            raise RuntimeError(
                "Precipitation forecast-step validation failed.\n"
                f"Expected: {expected_steps}\n"
                f"Actual:   {actual_steps}"
            )

    print("Precipitation temporary GRIB validation: PASS")
    TEMP_GRIB.replace(global_grib)
    print(f"Verified precipitation GRIB installed: {global_grib}")

except Exception:
    if TEMP_GRIB.exists():
        print(f"Temporary GRIB retained for diagnosis: {TEMP_GRIB}")
    raise

print("Precipitation download and validation complete.")


# ============================================================
# 7. CHECK DOWNLOADED FILE
# ============================================================

print("\n" + "=" * 70)
print("CHECKING DOWNLOADED FILE")
print("=" * 70)

if not global_grib.exists():

    raise FileNotFoundError(
        f"GRIB2 file was not created:\n{global_grib}"
    )

file_size_mb = (
    global_grib.stat().st_size /
    (1024 * 1024)
)

print("\nFile successfully created:")
print(global_grib)

print(
    f"\nFile size: {file_size_mb:.2f} MB"
)


# ============================================================
# 8. OPEN GRIB2 USING CFGRIB
# ============================================================

print("\n" + "=" * 70)
print("OPENING ECMWF GRIB2 FILE")
print("=" * 70)

try:

    ds = xr.open_dataset(
        global_grib,
        engine="cfgrib"
    )

except Exception as e:

    print("\nERROR OPENING GRIB2:")
    print(e)

    raise

print("\nDataset opened successfully.")

print(ds)


# ============================================================
# 9. INSPECT ECMWF FORECAST STEPS
# ============================================================

print("\n" + "=" * 70)
print("ECMWF FORECAST STEP INFORMATION")
print("=" * 70)

print("\nAvailable steps:")
print(ds.step.values)

print("\nStep data type:")
print(ds.step.dtype)

print("\nVariables:")
print(list(ds.data_vars))

# Store ECMWF forecast steps for use throughout the workflow.
# np.atleast_1d() handles both single-step and multi-step files.
available_steps = np.atleast_1d(
    np.asarray(ds.step.values)
)

print("\nStored available steps:")
print(available_steps)


# ============================================================
# 10. ECMWF STEP HANDLING
# ============================================================

def find_ecmwf_step(
    available_steps,
    requested_hours
):
    """
    Find the ECMWF forecast step corresponding
    to the requested number of hours.

    Handles both:
    - multiple ECMWF forecast steps
    - a single scalar ECMWF forecast step
    """

    available_steps = np.atleast_1d(
        np.asarray(available_steps)
    )

    requested_step = np.timedelta64(
        requested_hours,
        "h"
    )

    differences = np.abs(
        available_steps -
        requested_step
    )

    nearest_index = int(
        np.argmin(differences)
    )

    actual_step = available_steps[
        nearest_index
    ]

    difference = differences[
        nearest_index
    ]

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
# 11. EXTRACT TOTAL PRECIPITATION
# ============================================================

print("\n" + "=" * 70)
print("TOTAL PRECIPITATION")
print("=" * 70)

if "tp" not in ds.data_vars:

    raise KeyError(
        "The ECMWF variable 'tp' was not found."
    )

precip = ds["tp"]

print("\nVariable:")
print(precip)

print("\nECMWF units:")

print(
    precip.attrs.get(
        "GRIB_units",
        "Not specified"
    )
)


# ============================================================
# 12. EXTRACT AFRICA
# ============================================================

print("\n" + "=" * 70)
print("EXTRACTING AFRICA")
print("=" * 70)

africa = precip.sel(
    latitude=slice(
        AFRICA_NORTH,
        AFRICA_SOUTH
    ),
    longitude=slice(
        AFRICA_WEST,
        AFRICA_EAST
    )
)

print("\nAfrica extraction successful.")

print("\nLatitude:")
print(
    float(africa.latitude.min()),
    "to",
    float(africa.latitude.max())
)

print("\nLongitude:")
print(
    float(africa.longitude.min()),
    "to",
    float(africa.longitude.max())
)


# ============================================================
# 13. CONVERT PRECIPITATION TO MILLIMETRES
# ============================================================

# ECMWF total precipitation is normally in metres.
# Convert metres to millimetres.

africa_mm = africa * 1000.0

africa_mm.attrs["units"] = "mm"

print("\n" + "=" * 70)
print("PRECIPITATION UNIT CONVERSION")
print("=" * 70)

print("\nConverted units: mm")

print(africa_mm)


# ============================================================
# 14. CREATE CUMULATIVE PRECIPITATION GEOTIFFS
# ============================================================

print("\n" + "=" * 70)
print("CREATING CUMULATIVE PRECIPITATION PRODUCTS")
print("=" * 70)

cumulative_files = {}

for requested_hours in FORECAST_STEPS:

    print(
        f"\nProcessing +{requested_hours} hours..."
    )

    actual_step = find_ecmwf_step(
        available_steps,
        requested_hours
    )

    # Handle both scalar-step and multi-step ECMWF files.
    # A single-step GRIB file stores step as a scalar coordinate,
    # so .sel(step=...) cannot be used in that case.
    if "step" in africa_mm.dims:
        rainfall = africa_mm.sel(
            step=actual_step
        ).squeeze()
    else:
        rainfall = africa_mm.squeeze()

    rainfall = rainfall.rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude"
    )

    rainfall = rainfall.rio.write_crs(
        "EPSG:4326"
    )

    output_file = (
        ECMWF_DIR /
        f"ECMWF_AFRICA_{date_now}_"
        f"{requested_hours}h.tif"
    )

    rainfall.rio.to_raster(
        output_file,
        compress="LZW"
    )

    cumulative_files[
        requested_hours
    ] = output_file

    print(
        "Saved:",
        output_file.name
    )

    print(
        f"Min : {float(rainfall.min()):.2f} mm"
    )

    print(
        f"Max : {float(rainfall.max()):.2f} mm"
    )

    print(
        f"Mean: {float(rainfall.mean()):.2f} mm"
    )


# ============================================================
# 15. CREATE 24-HOUR PRECIPITATION PRODUCTS
# ============================================================

print("\n" + "=" * 70)
print("CREATING 24-HOUR PRECIPITATION PRODUCTS")
print("=" * 70)

daily_rainfall = {}

previous_rainfall = None
previous_hours = 0


for requested_hours in FORECAST_STEPS:

    print(
        f"\nProcessing +{requested_hours} hours..."
    )

    actual_step = find_ecmwf_step(
        available_steps,
        requested_hours
    )

    # Handle both scalar-step and multi-step ECMWF files.
    # A single-step GRIB file stores step as a scalar coordinate,
    # so .sel(step=...) cannot be used in that case.
    if "step" in africa_mm.dims:
        current_rainfall = africa_mm.sel(
            step=actual_step
        ).squeeze()
    else:
        current_rainfall = africa_mm.squeeze()

    if previous_rainfall is None:

        rainfall_24h = current_rainfall

        period_start = 0

    else:

        rainfall_24h = (
            current_rainfall -
            previous_rainfall
        )

        period_start = previous_hours

    # Prevent negative values caused by
    # numerical precision.

    rainfall_24h = rainfall_24h.clip(
        min=0
    )

    rainfall_24h.attrs["units"] = "mm"

    rainfall_24h = rainfall_24h.rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude"
    )

    rainfall_24h = rainfall_24h.rio.write_crs(
        "EPSG:4326"
    )

    output_file = (
        ECMWF_DIR /
        f"ECMWF_PRECIP_AFRICA_{date_now}_"
        f"{period_start:02d}-"
        f"{requested_hours:02d}h.tif"
    )

    rainfall_24h.rio.to_raster(
        output_file,
        compress="LZW"
    )

    daily_rainfall[
        requested_hours
    ] = rainfall_24h

    print(
        f"Period: "
        f"{period_start:02d}-"
        f"{requested_hours:02d}h"
    )

    print(
        "Saved:",
        output_file.name
    )

    print(
        f"Min : {float(rainfall_24h.min()):.2f} mm"
    )

    print(
        f"Max : {float(rainfall_24h.max()):.2f} mm"
    )

    print(
        f"Mean: {float(rainfall_24h.mean()):.2f} mm"
    )

    previous_rainfall = current_rainfall

    previous_hours = requested_hours


# ============================================================
# 16. CREATE 6-DAY TOTAL PRECIPITATION
# ============================================================

print("\n" + "=" * 70)
print("CALCULATING 6-DAY TOTAL PRECIPITATION")
print("=" * 70)

total_file = None
total_6day = None

# The 6-day product requires an ECMWF +144 hour step.
# During controlled single-step tests, only +24 h may be available.
available_hours = np.asarray(
    available_steps / np.timedelta64(1, "h"),
    dtype=float
)

if np.any(np.isclose(available_hours, 144.0)):

    print("ECMWF +144 hour step is available.")

    actual_step_144 = find_ecmwf_step(
        available_steps,
        144
    )

    if "step" in africa_mm.dims:
        total_6day = africa_mm.sel(
            step=actual_step_144
        ).squeeze()
    else:
        total_6day = africa_mm.squeeze()

    total_6day = total_6day.clip(
        min=0
    )

    total_6day.attrs["units"] = "mm"

    total_6day = total_6day.rio.set_spatial_dims(
        x_dim="longitude",
        y_dim="latitude"
    )

    total_6day = total_6day.rio.write_crs(
        "EPSG:4326"
    )

    total_file = (
        ECMWF_DIR /
        f"ECMWF_TOTAL_PRECIP_AFRICA_"
        f"{date_now}_6DAY.tif"
    )

    total_6day.rio.to_raster(
        total_file,
        compress="LZW"
    )

    print("\n6-day rainfall saved:")
    print(total_file)

    print(
        f"\nMinimum: "
        f"{float(total_6day.min()):.2f} mm"
    )

    print(
        f"Maximum: "
        f"{float(total_6day.max()):.2f} mm"
    )

    print(
        f"Mean: "
        f"{float(total_6day.mean()):.2f} mm"
    )

else:

    print(
        "ECMWF +144 hour step is not available."
    )

    print(
        "Skipping 6-day total precipitation "
        "for this controlled test."
    )


# ============================================================
# 17. DISPLAY 24-HOUR RAINFALL MAP
# ============================================================

print("\n" + "=" * 70)
print("DISPLAYING 24-HOUR RAINFALL MAP")
print("=" * 70)

rainfall_24h = daily_rainfall[24]

plt.figure(
    figsize=(12, 8)
)

rainfall_24h.plot()

plt.title(
    f"ECMWF 24-Hour Precipitation - Africa\n"
    f"{date_now} | IFS 0.25°"
)

plt.xlabel("Longitude")
plt.ylabel("Latitude")

plt.tight_layout()

output_png_24h = ECMWF_DIR / f"ECMWF_PRECIP_AFRICA_{date_now}_00-24h.png"
plt.savefig(output_png_24h, dpi=200, bbox_inches="tight")
print(f"Saved PNG: {output_png_24h}")



# ============================================================
# 18. DISPLAY 6-DAY TOTAL RAINFALL MAP
# ============================================================

print("\n" + "=" * 70)
print("DISPLAYING 6-DAY TOTAL RAINFALL MAP")
print("=" * 70)

if total_6day is not None:

    plt.figure(
        figsize=(12, 8)
    )

    total_6day.plot()

    plt.title(
        f"ECMWF 6-Day Total Precipitation - Africa\\n"
        f"{date_now} | IFS 0.25°"
    )

    plt.xlabel("Longitude")
    plt.ylabel("Latitude")

    plt.tight_layout()

    output_png_6day = (
        ECMWF_DIR /
        f"ECMWF_TOTAL_PRECIP_AFRICA_"
        f"{date_now}_6DAY.png"
    )

    plt.savefig(
        output_png_6day,
        dpi=200,
        bbox_inches="tight"
    )

    print(
        f"Saved PNG: {output_png_6day}"
    )


else:

    print(
        "6-day rainfall map skipped because "
        "ECMWF +144 hour data is not available."
    )


# ============================================================
# 19. RAINFALL STATISTICS
# ============================================================

print("\n" + "=" * 70)
print("24-HOUR RAINFALL STATISTICS")
print("=" * 70)

for requested_hours in FORECAST_STEPS:

    rainfall = daily_rainfall[
        requested_hours
    ]

    if requested_hours == 24:

        start_hour = 0

    else:

        start_hour = requested_hours - 24

    print(
        f"\n{start_hour:03d}-{requested_hours:03d} hours"
    )

    print(
        f"Minimum : "
        f"{float(rainfall.min()):.2f} mm"
    )

    print(
        f"Maximum : "
        f"{float(rainfall.max()):.2f} mm"
    )

    print(
        f"Mean    : "
        f"{float(rainfall.mean()):.2f} mm"
    )


# ============================================================
# 20. LIST GENERATED PRODUCTS
# ============================================================

print("\n" + "=" * 70)
print("GENERATED ECMWF PRODUCTS")
print("=" * 70)

for file in sorted(
    ECMWF_DIR.iterdir()
):

    if file.is_file():

        size_mb = (
            file.stat().st_size /
            (1024 * 1024)
        )

        print(
            f"{file.name:<70}"
            f"{size_mb:>8.2f} MB"
        )


# ============================================================
# 21. FINAL SUMMARY
# ============================================================

print("\n")
print("=" * 70)
print("LMS ECMWF PRECIPITATION PROCESSING COMPLETED")
print("=" * 70)

print("\nForecast date:")
print(date_now)

print("\nModel:")
print("ECMWF IFS")

print("\nResolution:")
print("0.25° × 0.25°")

print("\nParameter:")
print("Total precipitation (tp)")

print("\nForecast period:")
print("144 hours / 6 days")

print("\nOutput directory:")
print(ECMWF_DIR)

print("\nCumulative products:")
for hours in FORECAST_STEPS:
    print(f"  +{hours} hours")

print("\n24-hour products:")
print("  00-24h")
print("  24-48h")
print("  48-72h")
print("  72-96h")
print("  96-120h")
print("  120-144h")

print("\n6-day total:")
print("  00-144h")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)