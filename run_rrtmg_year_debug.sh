#!/bin/bash

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: bash run_rrtmg_year_debug.sh YEAR" >&2
    exit 1
fi

# Override the full-year array with the first three six-hourly timestamps.
sbatch --array=0-2 run_rrtmg_year.sh "$1"
