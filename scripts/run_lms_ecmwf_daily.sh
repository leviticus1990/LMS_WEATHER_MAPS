#!/bin/bash

BASE_DIR="/home/james/LMS_WEATHER_MAPS"
PYTHON="/home/james/ecmwf_opendata/venv/bin/python"
SCRIPT="$BASE_DIR/scripts/run_lms_ecmwf.py"

cd "$BASE_DIR" || exit 1

exec "$PYTHON" "$SCRIPT"
