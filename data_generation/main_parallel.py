import argparse
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path


def timestamp_for_task(year: int, task_id: int) -> datetime:
    start = datetime(year, 1, 1)
    timestamp = start + timedelta(hours=6 * task_id)
    if timestamp.year != year:
        raise ValueError(f"Task {task_id} is outside {year}")
    return timestamp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, required=True)
    parser.add_argument("--task-id", type=int, default=os.getenv("SLURM_ARRAY_TASK_ID"))
    parser.add_argument("--input-dir", type=Path, default=Path("data/era5_6hr_2.5"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/rrtmg"))
    parser.add_argument(
        "--upper-profile-path", type=Path, default=Path("data_qtqo3_123lvl.nc")
    )
    args = parser.parse_args()
    if args.task_id is None:
        parser.error("--task-id or SLURM_ARRAY_TASK_ID is required")

    timestamp = timestamp_for_task(args.year, int(args.task_id))
    time_index = (timestamp.day - 1) * 4 + timestamp.hour // 6
    output_dir = args.output_dir / str(args.year) / timestamp.strftime("%Y%m%dT%H")
    marker = output_dir / "_SUCCESS"
    if marker.exists():
        print(f"Already complete: {output_dir}")
        return

    command = [
        sys.executable,
        str(Path(__file__).with_name("main.py")),
        "--year",
        str(args.year),
        "--month",
        str(timestamp.month),
        "--time_index",
        str(time_index),
        "--input_dir",
        str(args.input_dir),
        "--output_dir",
        str(output_dir),
        "--upper_profile_path",
        str(args.upper_profile_path),
    ]
    subprocess.run(command, check=True)
    marker.touch()


if __name__ == "__main__":
    main()
