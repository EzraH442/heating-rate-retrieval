import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset
from tqdm import tqdm


def completed_outputs(year_dir: Path, filename: str) -> list[Path]:
    paths = [
        directory / filename
        for directory in sorted(year_dir.iterdir())
        if directory.is_dir() and (directory / "_SUCCESS").exists()
    ]
    missing = [path for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(missing[0])
    if not paths:
        raise FileNotFoundError(f"No completed outputs found in {year_dir}")
    return paths


def combine(paths: list[Path], output_path: Path) -> int:
    dimensions = {}
    destination_kwargs = {}
    variable_attrs = {}
    coordinate_values = {}

    time_variable_names = []
    with Dataset(paths[0]) as first:
        for varname, dimension in first.dimensions.items():
            dimensions[varname] = len(dimension)
        dimensions["time"] = len(paths)

        for varname, variable in first.variables.items():
            if "time" in variable.dimensions:
                time_variable_names.append(varname)

        for varname, variable in first.variables.items():
            destination_kwargs[varname] = {
                "varname": varname,
                "datatype": variable.datatype,
                "dimensions": variable.dimensions,
                "fill_value": (
                    np.nan if np.issubdtype(variable.dtype, np.floating) else False
                ),
                "zlib": variable.ndim > 0,
            }
            variable_attrs[varname] = {
                attr: variable.getncattr(attr)
                for attr in variable.ncattrs()
                if attr != "_FillValue"
            }
            if "time" not in variable.dimensions:
                coordinate_values[varname] = variable[:]

    with Dataset(output_path, "w", format="NETCDF4") as output:
        for varname, size in dimensions.items():
            output.createDimension(varname, size)
        for varname, kwargs in destination_kwargs.items():
            destination = output.createVariable(**kwargs)
            destination.setncatts(variable_attrs[varname])

        # set non-time coordinate variables once
        for varname, values in coordinate_values.items():
            output.variables[varname][:] = values

        # set the time coordinate and data variables one timestamp at a time
        for position, path in enumerate(
            tqdm(paths, desc=output_path.name, unit="file")
        ):
            with Dataset(path) as source:
                for varname in time_variables:
                    output.variables[varname][position] = source.variables[varname][0]

    return len(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("--input-dir", type=Path, default=Path("data/rrtmg"))
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    year_dir = args.input_dir / str(args.year)
    output_dir = args.output_dir or year_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    htr_count = combine(
        completed_outputs(year_dir, "htr_lw.nc"),
        output_dir / f"htr_lw_{args.year}.nc",
    )
    flux_count = combine(
        completed_outputs(year_dir, "toa_flux_lw.nc"),
        output_dir / f"toa_flux_lw_{args.year}.nc",
    )
    if htr_count != flux_count:
        raise ValueError(
            f"Heating rates have {htr_count} timestamps but fluxes have {flux_count}"
        )
    print(f"Combined {htr_count} timestamps into {output_dir}")


if __name__ == "__main__":
    main()
