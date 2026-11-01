#!/bin/bash
#SBATCH --job-name=rrtmg
#SBATCH --account=rrg-yihuang-ad
# 365 days x 4 six-hourly timestamps per day = 1,460 tasks.
# need to override the array with 0-1463 when computing values for a leap year
#SBATCH --array=0-1459
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G
#SBATCH --ntasks=1
#SBATCH --time=04:00:00
#SBATCH --output=slurm-%x-%A_%a.out

set -euo pipefail

if [ "$#" -ne 1 ]; then
    echo "Usage: sbatch run_rrtmg_year.sh YEAR" >&2
    exit 1
fi

YEAR="$1"

module purge
module load StdEnv/2023
module load python/3.11
module load cuda/12.2
module load mpi4py/4.1.0
module load scipy-stack
module load cdo

source ~/venv/bin/activate

if ! command -v rrtmg_lw >/dev/null; then
    echo "rrtmg_lw is not on PATH" >&2
    exit 1
fi

python data_generation/main_parallel.py \
    --year "$YEAR"
