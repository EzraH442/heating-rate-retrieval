import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from tqdm import tqdm


def timestamps_for_month(year: int, month: int):
    if not 1 <= month <= 12:
        raise ValueError(f"Month {month} is outside 1-12")
    start = datetime(year, month, 1)
    end = datetime(year + (month == 12), month % 12 + 1, 1)
    return [start + timedelta(hours=6 * index) for index in range((end - start).days * 4)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--task-id", type=int, default=os.getenv("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--max-timestamps", type=int, default=os.getenv("RRTMG_MAX_TIMESTAMPS"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/era5_6hr_2.5"))
    parser.add_argument("--output-root", type=Path, default=Path("data/rrtmg"))
    parser.add_argument(
        "--upper-profile-path", type=Path, default=Path("data_qtqo3_123lvl.nc")
    )
    args = parser.parse_args()
    if args.task_id is None:
        parser.error("--task-id or SLURM_ARRAY_TASK_ID is required")
    month = int(args.task_id) + 1
    try:
        timestamps = timestamps_for_month(args.year, month)
    except ValueError as error:
        parser.error(str(error))
    if args.max_timestamps is not None:
        timestamps = timestamps[:args.max_timestamps]

    for timestamp in tqdm(timestamps, unit="timestamp"):
        time_index = (timestamp.day - 1) * 4 + timestamp.hour // 6
        output_dir = args.output_root / str(args.year) / timestamp.strftime("%Y%m%dT%H")
        marker = output_dir / "_SUCCESS"
        if marker.exists():
            continue

        command = [
            sys.executable,
            str(Path(__file__).with_name("main.py")),
            "--year", str(args.year),
            "--month", str(timestamp.month),
            "--time_index", str(time_index),
            "--input_dir", str(args.input_dir),
            "--output_dir", str(output_dir),
            "--upper_profile_path", str(args.upper_profile_path),
        ]
        subprocess.run(command, check=True)
        marker.touch()


if __name__ == "__main__":
    main()
