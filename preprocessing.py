from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, date2num
from tap import Tap


class Parser(Tap):
    data_dir: Path
    output_dir: Path
    test_fraction: float = 0.2
    seed: int = 0


def input_files(data_dir: Path) -> dict[int, list[tuple[int, Path, Path]]]:
    years = {}
    prefix = "spectral_kernel_era5_"
    suffix = "_multiple_profile"
    for year_dir in sorted(data_dir.glob(f"{prefix}*{suffix}")):
        year_text = year_dir.name.removeprefix(prefix).removesuffix(suffix)
        if not year_text.isdigit():
            continue

        pairs = []
        for month in range(1, 13):
            mm = f"{month:02d}"
            flux_path = (
                year_dir
                / "instant_TOA_SFC_spectral/lw/no_perturbation"
                / f"TOA_SFC_all_clr_{mm}.nc"
            )
            htr_path = (
                year_dir
                / "generate_hrt/instant/lw/no_perturbation"
                / f"hrt_broadband_all_clr_{mm}.nc"
            )
            if flux_path.exists() and htr_path.exists():
                pairs.append((month, flux_path, htr_path))

        if pairs:
            years[int(year_text)] = pairs
    return years


def create_output(
    path: Path, split: str, year: int, band, level, latitude, longitude, test_fraction, seed
):
    output = Dataset(path, "w", format="NETCDF4")
    output.createDimension("time", None)
    output.createDimension("latitude", len(latitude))
    output.createDimension("longitude", len(longitude))
    output.createDimension("band", len(band))
    output.createDimension("level", len(level))
    time = output.createVariable("time", "f8", ("time",))
    time.units = f"hours since {year}-01-01 00:00:00"
    time.calendar = "standard"
    output.createVariable("latitude", "f4", ("latitude",))[:] = latitude
    output.createVariable("longitude", "f4", ("longitude",))[:] = longitude
    output.createVariable("band", "f4", ("band",))[:] = band
    output.createVariable("level", "f4", ("level",))[:] = level
    output.createVariable(
        "toa_flux_lw",
        "f4",
        ("time", "latitude", "longitude", "band"),
        zlib=True,
        complevel=4,
    )
    output.createVariable(
        "htr_lw",
        "f4",
        ("time", "level", "latitude", "longitude"),
        zlib=True,
        complevel=4,
    )
    output.setncatts(
        {"year": year, "split": split, "test_fraction": test_fraction, "seed": seed}
    )
    return output


def preprocess_year(year, pairs, output_dir, test_fraction, seed):
    _, first_flux, first_htr = pairs[0]
    with Dataset(first_flux) as flux_ds, Dataset(first_htr) as htr_ds:
        band = flux_ds.variables["band"][:]
        level = htr_ds.variables["ilevel"][:]
        latitude = flux_ds.variables["latitude"][:]
        longitude = flux_ds.variables["longitude"][:]

    outputs = {
        split: create_output(
            output_dir / f"{split}_{year}.nc",
            split,
            year,
            band,
            level,
            latitude,
            longitude,
            test_fraction,
            seed,
        )
        for split in ("train", "test")
    }
    positions = {"train": 0, "test": 0}
    rng = np.random.default_rng(seed + year)
    total_times = 0
    for _, flux_path, _ in pairs:
        with Dataset(flux_path) as flux_ds:
            total_times += len(flux_ds.dimensions["time"])
    test_times = set(
        rng.choice(total_times, int(total_times * test_fraction), replace=False)
    )
    year_time_index = 0

    try:
        for month, flux_path, htr_path in pairs:
            with Dataset(flux_path) as flux_ds, Dataset(htr_path) as htr_ds:
                flux = flux_ds.variables["TOA_clr"]
                htr = htr_ds.variables["hrt_clr"]

                for time_index in range(flux.shape[0]):
                    # up_down_net=1 is the upward TOA flux; index 0 in these files.
                    split = "test" if year_time_index in test_times else "train"
                    position = positions[split]
                    output = outputs[split]
                    file_time = flux_ds.variables["time"][time_index]
                    date = datetime(year, month, 1) + timedelta(
                        hours=(float(file_time) - 1) * 6
                    )
                    time = output.variables["time"]
                    time[position] = date2num(date, time.units, time.calendar)
                    output.variables["toa_flux_lw"][position] = flux[time_index, :, :, 0, :]
                    output.variables["htr_lw"][position] = htr[time_index]
                    positions[split] += 1
                    year_time_index += 1
    finally:
        for output in outputs.values():
            output.close()

    print(f"{year}: {positions['train']} training, {positions['test']} testing times")


def main():
    args = Parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    files_by_year = input_files(args.data_dir)
    for year, pairs in sorted(files_by_year.items()):
        preprocess_year(
            year,
            pairs,
            args.output_dir,
            args.test_fraction,
            args.seed,
        )


if __name__ == "__main__":
    main()


