#!/usr/bin/env python3

"""
LMS ECMWF FINAL PRODUCT QUALITY CONTROL

Validates the final LMS ECMWF PNG archive for:
- Expected product count
- Expected filenames
- Forecast periods
- PNG integrity
- Image dimensions
- Required product groups

This script does NOT modify any forecast products.
"""

from pathlib import Path
from datetime import datetime, timezone
import sys

from PIL import Image


BASE_DIR = Path("/home/james/LMS_WEATHER_MAPS")

if len(sys.argv) > 1:
    DATE_NOW = sys.argv[1]
else:
    DATE_NOW = datetime.now(timezone.utc).strftime("%Y%m%d")

if len(DATE_NOW) != 8 or not DATE_NOW.isdigit():
    raise SystemExit("ERROR: Forecast date must be in YYYYMMDD format.")

try:
    datetime.strptime(DATE_NOW, "%Y%m%d")
except ValueError:
    raise SystemExit("ERROR: Invalid forecast date. Use YYYYMMDD.")

FINAL_DIR = (
    BASE_DIR
    / "archive"
    / "ECMWF"
    / DATE_NOW
    / "FINAL_LMS"
)

FORECAST_PERIODS = (
    "024h",
    "048h",
    "072h",
    "096h",
    "120h",
    "144h",
)

PRODUCT_VARIABLES = (
    "PRECIP",
    "TEMPERATURE",
    "PRESSURE",
    "WIND",
    "RH",
    "TMAX",
    "TMIN",
)

EXPECTED_PRODUCTS = []

for variable in PRODUCT_VARIABLES:
    for period in FORECAST_PERIODS:
        EXPECTED_PRODUCTS.append(
            f"LMS_{variable}_AFRICA_{DATE_NOW}_{period}.png"
        )

EXPECTED_PRODUCTS.append(
    f"LMS_TOTAL_PRECIP_AFRICA_{DATE_NOW}_6DAY.png"
)


def check(condition, message):
    if condition:
        print(f"PASS: {message}")
        return True

    print(f"FAIL: {message}")
    return False


def main():

    print("=" * 72)
    print("LMS ECMWF FINAL PRODUCT QUALITY CONTROL")
    print("=" * 72)
    print(f"Forecast date: {DATE_NOW}")
    print(f"Final archive: {FINAL_DIR}")
    print()

    failures = []

    # ------------------------------------------------------------
    # 1. Final archive
    # ------------------------------------------------------------

    if not FINAL_DIR.exists():
        print(f"FAIL: Final archive does not exist: {FINAL_DIR}")
        return 1

    print("PASS: Final LMS archive exists")
    print()

    # ------------------------------------------------------------
    # 2. Expected product count
    # ------------------------------------------------------------

    png_files = sorted(FINAL_DIR.glob("*.png"))

    print("=== PRODUCT COUNT ===")

    if len(png_files) == 43:
        print("PASS: 43 PNG products found")
    else:
        print(
            f"FAIL: Expected 43 PNG products, found {len(png_files)}"
        )
        failures.append("Product count")

    print()

    # ------------------------------------------------------------
    # 3. Filename verification
    # ------------------------------------------------------------

    print("=== FILENAME VERIFICATION ===")

    actual_names = {f.name for f in png_files}

    for expected in EXPECTED_PRODUCTS:

        if expected in actual_names:
            print(f"PASS: {expected}")
        else:
            print(f"FAIL: MISSING {expected}")
            failures.append(expected)

    print()

    # ------------------------------------------------------------
    # 4. Unexpected files
    # ------------------------------------------------------------

    print("=== UNEXPECTED FILE CHECK ===")

    expected_names = set(EXPECTED_PRODUCTS)

    unexpected = actual_names - expected_names

    if not unexpected:
        print("PASS: No unexpected PNG products found")
    else:
        for filename in sorted(unexpected):
            print(f"FAIL: Unexpected file: {filename}")
            failures.append(filename)

    print()

    # ------------------------------------------------------------
    # 5. PNG integrity and dimensions
    # ------------------------------------------------------------

    print("=== PNG INTEGRITY ===")

    for filename in EXPECTED_PRODUCTS:

        product = FINAL_DIR / filename

        if not product.exists():
            continue

        try:

            with Image.open(product) as image:

                width, height = image.size
                image_format = image.format

                image.verify()

            if image_format != "PNG":
                print(
                    f"FAIL: {filename} is not PNG format"
                )
                failures.append(filename)
                continue

            print(
                f"PASS: {filename} | "
                f"{width}x{height} | PNG"
            )

        except Exception as exc:

            print(
                f"FAIL: {filename} | {exc}"
            )

            failures.append(filename)

    print()

    # ------------------------------------------------------------
    # 6. Product groups
    # ------------------------------------------------------------

    print("=== PRODUCT GROUP VERIFICATION ===")

    expected_groups = {
        "PRECIP": 6,
        "TEMPERATURE": 6,
        "PRESSURE": 6,
        "WIND": 6,
        "RH": 6,
        "TMAX": 6,
        "TMIN": 6,
        "TOTAL_PRECIP": 1,
    }

    for group, expected_count in expected_groups.items():

        if group == "TOTAL_PRECIP":
            pattern = f"LMS_TOTAL_PRECIP_AFRICA_{DATE_NOW}_6DAY.png"
            count = int((FINAL_DIR / pattern).exists())

        else:
            pattern = f"LMS_{group}_AFRICA_{DATE_NOW}_*.png"
            count = len(list(FINAL_DIR.glob(pattern)))

        if count == expected_count:

            print(
                f"PASS: {group}: "
                f"{count}/{expected_count}"
            )

        else:

            print(
                f"FAIL: {group}: "
                f"{count}/{expected_count}"
            )

            failures.append(group)

    print()

    # ------------------------------------------------------------
    # 7. Final result
    # ------------------------------------------------------------

    print("=" * 72)

    if failures:

        print("LMS ECMWF QC RESULT: FAIL")
        print()
        print("Failures:")

        for failure in sorted(set(failures)):
            print(f"  - {failure}")

        print("=" * 72)

        return 1

    print("LMS ECMWF QC RESULT: PASS")
    print()
    print("43/43 final LMS products passed quality control.")
    print("All expected product groups are present.")
    print("All expected PNG files passed integrity verification.")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    sys.exit(main())
