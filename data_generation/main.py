import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
from rrtmg_lw_input import RRTMG_LW_INPUT
from rrtmg_output import read_rrtmg_output
from rrtmg_util import prepare_era5_profile
from tap import Tap


class Parser(Tap):
    year: int
    month: int
    input_dir: Path = Path("data/era5_6hr_1.0")
    output_dir = Path("data/rrtmg")

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


def main():
    args = Parser().parse_args()
    ds = open_era5(args.input_dir, args.year, args.month)

    tmpdir = args.output_dir / "_tmp"
    tmpdir.mkdir(exist_ok=True)
    sample_dims = ds.skt.dims
    nband = 16
    toa_flux_lw = xr.full_like(
        ds.skt.expand_dims(band=np.arange(1, nband + 1), axis=-1), np.nan
    ).load()
    htr_lw = xr.full_like(ds.t.transpose(*sample_dims, "level"), np.nan).load()
    failed_cases = []
    for index in np.ndindex(*(ds.sizes[dim] for dim in sample_dims)):
        isel_arg = dict(zip(sample_dims, index))
        profile = ds.isel(isel_arg)
        clouds = 0
        pave, tave, p0, t0, p1, t1, wkl, wbroadl = prepare_era5_profile(
            profile, profile.sp.item()
        )
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
            toa_flux_lw.values[index] = toa_lw
            htr_lw.values[index + (slice(0, len(pave)),)] = profile_htr_lw

    failed_cases_np = np.array(failed_cases)

    toa_flux_lw.to_netcdf(args.output_dir / "toa_flux_lw.nc")
    htr_lw.to_netcdf(args.output_dir / "htr_lw.nc")
    np.savetxt(args.output_dir / "failed_cases.nc", failed_cases_np)


if __name__ == "__main__":
    main()
