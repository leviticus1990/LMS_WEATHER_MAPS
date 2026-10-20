#!/usr/bin/env python3

from datetime import datetime, timezone
from pathlib import Path
import shutil
import subprocess
import sys


BASE_DIR = Path.home() / "LMS_WEATHER_MAPS"
SCRIPTS_DIR = BASE_DIR / "scripts"
LOG_DIR = BASE_DIR / "logs"

DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

MAP_DIR = BASE_DIR / "maps" / DATE_NOW
ECMWF_DIR = MAP_DIR / "ECMWF"
FINAL_DIR = (
    BASE_DIR
    / "archive"
    / "ECMWF"
    / DATE_NOW
    / "FINAL_LMS"
)

LOG_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / f"lms_ecmwf_{DATE_NOW}.log"


PIPELINE = [
    ("ECMWF precipitation", SCRIPTS_DIR / "ecmwf_precip.py"),
    ("ECMWF temperature", SCRIPTS_DIR / "ecmwf_temperature.py"),
    ("ECMWF pressure", SCRIPTS_DIR / "ecmwf_pressure.py"),
    ("ECMWF wind", SCRIPTS_DIR / "ecmwf_wind.py"),
    ("ECMWF relative humidity", SCRIPTS_DIR / "ecmwf_relative_humidity.py"),
    ("ECMWF maximum temperature", SCRIPTS_DIR / "ecmwf_tmax.py"),
    ("ECMWF minimum temperature", SCRIPTS_DIR / "ecmwf_tmin.py"),

    ("Precipitation maps", SCRIPTS_DIR / "ecmwf" / "render_precip_maps.py"),
    ("6-day total precipitation map", SCRIPTS_DIR / "ecmwf" / "render_total_precip_map.py"),
    ("Temperature maps", SCRIPTS_DIR / "ecmwf" / "render_temperature_maps.py"),
    ("Pressure maps", SCRIPTS_DIR / "ecmwf" / "render_pressure_maps.py"),
    ("Wind maps", SCRIPTS_DIR / "ecmwf" / "render_wind_maps.py"),
    ("Relative humidity maps", SCRIPTS_DIR / "ecmwf" / "render_relative_humidity_maps.py"),
    ("Maximum temperature maps", SCRIPTS_DIR / "ecmwf" / "render_tmax_maps.py"),
    ("Minimum temperature maps", SCRIPTS_DIR / "ecmwf" / "render_tmin_maps.py"),

    ("LMS ECMWF product quality control", SCRIPTS_DIR / "qc_lms_ecmwf_products.py"),
]


def log(message):
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    line = f"[{timestamp}] {message}"
    print(line)
    with LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def run_script(name, script):
    log("=" * 70)
    log(f"START: {name}")
    log(f"Script: {script}")

    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=BASE_DIR,
    )

    if result.returncode != 0:
        log(f"FAILED: {name} (exit code {result.returncode})")
        raise RuntimeError(f"{name} failed")

    log(f"SUCCESS: {name}")


def publish_maps():
    log("=" * 70)
    log("VERIFYING FINAL LMS MAPS")

    final_dir = (
        BASE_DIR
        / "archive"
        / "ECMWF"
        / DATE_NOW
        / "FINAL_LMS"
    )

    if not final_dir.exists():
        raise FileNotFoundError(
            f"Final LMS archive not found: {final_dir}"
        )

    expected_products = []

    for variable in (
        "PRECIP",
        "TEMPERATURE",
        "PRESSURE",
        "WIND",
        "RH",
        "TMAX",
        "TMIN",
    ):
        for period in (
            "024h",
            "048h",
            "072h",
            "096h",
            "120h",
            "144h",
        ):
            expected_products.append(
                f"LMS_{variable}_AFRICA_{DATE_NOW}_{period}.png"
            )

    expected_products.append(
        f"LMS_TOTAL_PRECIP_AFRICA_{DATE_NOW}_6DAY.png"
    )

    missing = []

    for filename in expected_products:
        product = final_dir / filename

        if product.exists():
            log(f"Verified: {filename}")
        else:
            missing.append(filename)
            log(f"MISSING: {filename}")

    verified_count = len(expected_products) - len(missing)

    log(
        f"Verified final LMS products: "
        f"{verified_count}/{len(expected_products)}"
    )

    if missing:
        raise FileNotFoundError(
            "Missing final LMS products: "
            + ", ".join(missing)
        )

    log(f"Final LMS archive: {final_dir}")
    log("All expected final LMS maps are present.")


def main():
    start = datetime.now(timezone.utc)

    log("=" * 70)
    log("LIBERIA METEOROLOGICAL SERVICE")
    log("ECMWF OPERATIONAL FORECAST-MAP PIPELINE")
    log(f"Forecast date: {DATE_NOW}")
    log("=" * 70)

    try:
        for name, script in PIPELINE:
            if not script.exists():
                raise FileNotFoundError(f"Script not found: {script}")

            run_script(name, script)

        publish_maps()

    except Exception as exc:
        log("=" * 70)
        log(f"PIPELINE FAILED: {exc}")
        log("=" * 70)
        return 1

    elapsed = datetime.now(timezone.utc) - start

    log("=" * 70)
    log("PIPELINE COMPLETED SUCCESSFULLY")
    log(f"Forecast date: {DATE_NOW}")
    log(f"Final maps: {FINAL_DIR}")
    log(f"Elapsed time: {elapsed}")
    log("=" * 70)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
