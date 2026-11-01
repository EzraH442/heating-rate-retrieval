import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
from netCDF4 import Dataset
from rrtmg_lw_input import RRTMG_LW_INPUT, load_upper_profile, patch_upper_layers
from rrtmg_output import read_rrtmg_output
from rrtmg_util import broad, prepare_era5_profile
from tap import Tap
from tqdm import tqdm


class Parser(Tap):
    year: int
    month: int
    time_index: int | None = None
    input_dir: Path = Path("data/era5_6hr_1.0")
    output_dir: Path = Path("data/rrtmg")
    upper_profile_path: Path = Path("data_qtqo3_123lvl.nc")

    def process_args(self):
        self.output_dir.mkdir(exist_ok=True, parents=True)


def open_era5(input_dir: Path, year: int, month: int) -> xr.Dataset:
    datasets = []
    for var in ("o3", "q", "t", "sp", "skt"):
        path = input_dir / var / f"{year}-{month:02d}.nc"
        if not path.exists():
            raise FileNotFoundError(path)
        datasets.append(xr.open_dataset(path))

    ds = xr.merge(datasets)
    return ds.rename(
        {
            name: old
            for name, old in {
                "pressure_level": "level",
                "valid_time": "time",
            }.items()
            if name in ds
        }
    )


def create_output(path, name, ds, sample_dims, output_dim, output_coords):
    output = Dataset(path, "w", format="NETCDF4")
    dimensions = (*sample_dims, output_dim)
    for dim in dimensions:
        values = output_coords if dim == output_dim else ds[dim].values
        output.createDimension(dim, len(values))
        if np.issubdtype(values.dtype, np.datetime64):
            coordinate = output.createVariable(dim, "i8", (dim,))
            coordinate.units = "seconds since 1970-01-01 00:00:00"
            coordinate.calendar = "proleptic_gregorian"
            coordinate[:] = values.astype("datetime64[s]").astype(np.int64)
        else:
            coordinate = output.createVariable(dim, values.dtype, (dim,))
            coordinate[:] = values
        if dim in ds.coords:
            coordinate.setncatts(ds[dim].attrs)

    variable = output.createVariable(
        name, "f4", dimensions, fill_value=np.nan, zlib=True
    )
    return output, variable


def main():
    args = Parser().parse_args()
    ds = open_era5(args.input_dir, args.year, args.month)
    if args.time_index is not None:
        if not 0 <= args.time_index < ds.sizes["time"]:
            raise IndexError(
                f"time_index {args.time_index} is outside 0..{ds.sizes['time'] - 1}"
            )
        ds = ds.isel(time=slice(args.time_index, args.time_index + 1))
    upper_p, upper_t, upper_q, upper_o3, interface_p, interface_t = (
        load_upper_profile(args.upper_profile_path)
    )

    tmpdir = args.output_dir / "_tmp"
    tmpdir.mkdir(exist_ok=True)
    sample_dims = ds.skt.dims
    nband = 16
    levels = np.concatenate((ds.level, upper_p))
    toa_output, toa_flux_lw = create_output(
        args.output_dir / "toa_flux_lw.nc",
        "toa_flux_lw",
        ds,
        sample_dims,
        "band",
        np.arange(1, nband + 1),
    )
    htr_output, htr_lw = create_output(
        args.output_dir / "htr_lw.nc",
        "htr_lw",
        ds,
        sample_dims,
        "level",
        levels,
    )
    failed_cases = []
    sample_shape = tuple(ds.sizes[dim] for dim in sample_dims)

    # ~1.5 iterations per second on 1 cpu with 500M memory on rorqual
    for index in tqdm(np.ndindex(*sample_shape), total=np.prod(sample_shape)):
        isel_arg = dict(zip(sample_dims, index))
        profile = ds.isel(isel_arg)
        clouds = 0
        pave, tave, p0, t0, p1, t1, wkl = prepare_era5_profile(
            profile, profile.sp.item()
        )
        pave, tave, p1, t1, wkl = patch_upper_layers(
            pave,
            tave,
            p1,
            t1,
            wkl,
            upper_p,
            upper_t,
            upper_q,
            upper_o3,
            interface_p,
            interface_t,
        )
        wbroadl = broad(np.r_[p0, p1], wkl)
        input_builder = RRTMG_LW_INPUT(
            cxid="rrtmg input",
            iaer=0,
            iatm=0,
            ixsect=0,
            numangs=0,
            iout=99,
            idrv=0,
            imca=clouds,
            icld=clouds,
            tbound=profile.skt.item(),  # sfc temp set of temp at bottom of first layer
            iemis=0,  # each band has emissivity equal to 1.0
            ireflect=0,  # only choice for RRTMG_LW
            semiss=None,
            dtbound=None,  # only if idrv = 1
        )
        input_builder.set_fields_rrtatm0(
            iform=0,
            nlayrs=len(pave),
            ixmols=7,
            nmol=7,
            pave=pave,
            tave=tave,
            pz0=p0,
            pz1=p1,
            tz0=t0,
            tz1=t1,
            wkl=wkl,
            wbroadl=wbroadl,
        )

        with open(tmpdir / "INPUT_RRTM", "wt") as input_file:
            input_builder.write(input_file)

        try:
            subprocess.run("rrtmg_lw", cwd=tmpdir, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Command failed with exit code: {e.returncode}")
            print(f"Error message: {e.stderr}")
            failed_cases.append(index)
            continue

        # Output has one boundary row followed by one row per layer.
        with open(tmpdir / "OUTPUT_RRTM", "rt") as output_file:
            toa_lw, profile_htr_lw = read_rrtmg_output(
                output_file, nlev=len(pave), nband=nband
            )
            toa_flux_lw[index + (slice(None),)] = toa_lw
            htr_lw[index + (slice(0, len(pave)),)] = profile_htr_lw

    failed_cases_np = np.array(failed_cases)

    toa_output.close()
    htr_output.close()
    np.savetxt(args.output_dir / "failed_cases.nc", failed_cases_np)


if __name__ == "__main__":
    main()
