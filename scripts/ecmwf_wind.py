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

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Correct project root for scripts/ecmwf_wind.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import ecmwf_config as config


# ============================================================
# DATE / OUTPUT
# ============================================================

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

OUTPUT_DIR = (
    config.MAP_DIR
    / DATE_NOW
    / "ECMWF"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# ECMWF CONFIGURATION
# ============================================================

MODEL = "ifs"
RESOLUTION = "0p25"
SOURCE = config.SOURCE

FORECAST_STEPS = [
    24,
    48,
    72,
    96,
    120,
    144,
]

U_PARAMETER = config.VARIABLES["u_wind"]
V_PARAMETER = config.VARIABLES["v_wind"]


# ============================================================
# LIBERIA DOMAIN
# ============================================================

WEST = config.LIBERIA_BBOX["west"]
EAST = config.LIBERIA_BBOX["east"]
SOUTH = config.LIBERIA_BBOX["south"]
NORTH = config.LIBERIA_BBOX["north"]


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
# STEP MATCHING
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
            f"Unable to find ECMWF step near "
            f"+{requested_hours} hours."
        )

    return actual_step


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    global_grib = (
        OUTPUT_DIR
        / f"ECMWF_WIND_10M_GLOBAL_{DATE_NOW}.grib2"
    )

    print("=" * 70)
    print("LMS 10-METRE WIND DATA PROCESSING")
    print("=" * 70)
    print(f"Date:       {DATE_NOW}")
    print(f"Source:     {SOURCE}")
    print(f"Resolution: {RESOLUTION}")
    print(
        f"Forecast:   +{FORECAST_STEPS[0]} "
        f"to +{FORECAST_STEPS[-1]} hours"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------------

    print()
    print("Downloading ECMWF 10-metre wind data...")

    client.retrieve(
        date=DATE_NOW,
        time=0,
        step=FORECAST_STEPS,
        stream="oper",
        type="fc",
        levtype="sfc",
        param=[
            U_PARAMETER,
            V_PARAMETER,
        ],
        target=str(global_grib)
    )

    print(
        f"Global GRIB saved: {global_grib.name}"
    )

    # --------------------------------------------------------
    # OPEN GRIB
    # --------------------------------------------------------

    print()
    print("Opening GRIB dataset...")

    ds = xr.open_dataset(
        global_grib,
        engine="cfgrib",
        backend_kwargs={
            "indexpath": ""
        }
    )

    print(
        "Variables:",
        list(ds.data_vars)
    )

    if "u10" not in ds:
        raise KeyError(
            "ECMWF dataset does not contain u10."
        )

    if "v10" not in ds:
        raise KeyError(
            "ECMWF dataset does not contain v10."
        )

    available_steps = np.atleast_1d(
        np.asarray(ds.step.values)
    )

    # --------------------------------------------------------
    # LOOP THROUGH FORECAST PERIODS
    # --------------------------------------------------------

    for requested_hours in FORECAST_STEPS:

        print()
        print(f"+{requested_hours} h")

        actual_step = find_ecmwf_step(
            available_steps,
            requested_hours
        )

        u = ds["u10"].sel(
            step=actual_step
        ).squeeze()

        v = ds["v10"].sel(
            step=actual_step
        ).squeeze()

        # ----------------------------------------------------
        # EXTRACT LIBERIA
        # ----------------------------------------------------

        u = u.sel(
            latitude=slice(NORTH, SOUTH),
            longitude=slice(WEST, EAST)
        )

        v = v.sel(
            latitude=slice(NORTH, SOUTH),
            longitude=slice(WEST, EAST)
        )

        # ----------------------------------------------------
        # WIND SPEED
        # ----------------------------------------------------

        wind_speed = np.sqrt(
            u**2 + v**2
        )

        wind_speed.name = "wind_speed"

        wind_speed.attrs.update({
            "long_name":
                "10 metre wind speed",
            "units":
                "m s-1",
            "forecast_hour":
                requested_hours
        })

        # ----------------------------------------------------
        # METEOROLOGICAL WIND DIRECTION
        #
        # Direction FROM which wind is blowing:
        # 0 = North
        # 90 = East
        # 180 = South
        # 270 = West
        # ----------------------------------------------------

        wind_direction = (
            (270.0 - np.degrees(
                np.arctan2(v, u)
            )) % 360.0
        )

        wind_direction = xr.DataArray(
            wind_direction,
            coords=wind_speed.coords,
            dims=wind_speed.dims,
            name="wind_direction"
        )

        wind_direction.attrs.update({
            "long_name":
                "10 metre meteorological wind direction",
            "units":
                "degrees",
            "forecast_hour":
                requested_hours
        })

        # ----------------------------------------------------
        # SAVE WIND SPEED GEOTIFF
        # ----------------------------------------------------

        speed_output = (
            OUTPUT_DIR
            / (
                f"ECMWF_WIND_SPEED_AFRICA_"
                f"{DATE_NOW}_"
                f"{requested_hours:03d}h.tif"
            )
        )

        wind_speed = (
            wind_speed
            .rio.write_crs("EPSG:4326")
            .rio.set_spatial_dims(
                x_dim="longitude",
                y_dim="latitude"
            )
        )

        wind_speed.rio.to_raster(
            speed_output
        )

        # ----------------------------------------------------
        # SAVE WIND DIRECTION GEOTIFF
        # ----------------------------------------------------

        direction_output = (
            OUTPUT_DIR
            / (
                f"ECMWF_WIND_DIRECTION_AFRICA_"
                f"{DATE_NOW}_"
                f"{requested_hours:03d}h.tif"
            )
        )

        wind_direction = (
            wind_direction
            .rio.write_crs("EPSG:4326")
            .rio.set_spatial_dims(
                x_dim="longitude",
                y_dim="latitude"
            )
        )

        wind_direction.rio.to_raster(
            direction_output
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        speed_values = (
            wind_speed.values
        )

        speed_values = speed_values[
            np.isfinite(speed_values)
        ]

        direction_values = (
            wind_direction.values
        )

        direction_values = direction_values[
            np.isfinite(direction_values)
        ]

        print(
            f"  Speed GeoTIFF: "
            f"{speed_output.name}"
        )

        print(
            f"  Speed minimum: "
            f"{speed_values.min():.2f} m/s"
        )

        print(
            f"  Speed maximum: "
            f"{speed_values.max():.2f} m/s"
        )

        print(
            f"  Speed mean: "
            f"{speed_values.mean():.2f} m/s"
        )

        print(
            f"  Direction range: "
            f"{direction_values.min():.1f}–"
            f"{direction_values.max():.1f}°"
        )

        print(
            f"  Direction GeoTIFF: "
            f"{direction_output.name}"
        )

    ds.close()

    print()
    print("=" * 70)
    print(
        "LMS 10-METRE WIND DATA PROCESSING COMPLETE"
    )
    print("=" * 70)
    print(
        f"Output directory:\n{OUTPUT_DIR}"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
