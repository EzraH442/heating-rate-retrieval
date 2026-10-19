import subprocess
from pathlib import Path

import numpy as np
import xarray as xr
from rrtmg_lw_input import RRTMG_LW_INPUT
from rrtmg_util import prepare_era5_profile
from tap import Tap


class Parser(Tap):
    year: int
    output_dir = Path("data/rrtmg")

    def process_args(self):
        self.ds_atm_path = Path(f"data/era5/era5_plev_qt_monthly_{self.year}.nc")
        self.ds_sfc_path = Path(f"data/era5/era5_plev_qt_monthly_{self.year}.nc")

        self.output_dir.mkdir(exist_ok=True, parents=True)


def main():
    args = Parser().parse_args()
    ds_atm = xr.open_dataset(args.ds_atm_path)
    ds_sfc = xr.open_dataset(args.ds_sfc_path)

    toa_flux_lw: xr.DataArray = xr.zeros_like(ds_sfc.skt)
    sfc_flux_lw: xr.DataArray = xr.zeros_like(ds_sfc.skt)
    htr_lw: xr.DataArray = xr.zeros_like(ds_atm.level)

    tmpdir = args.output_dir / "_tmp"
    N, M = len(ds_sfc.latitude), len(ds_sfc.longitude)
    failed_cases = []
    for i in range(N):
        for j in range(M):
            lat, lon = ds_sfc.latitude[i], ds_sfc.longitude[j]
            clouds = 0
            isel_arg = {"latitude": i, "longitude": j}
            pave, tave, p0, t0, p1, t1, wkl, wbroadl = prepare_era5_profile(
                ds_atm.isel(isel_arg),
                sp=ds_sfc.isel(isel_arg).sp.item(),
                skt=ds_sfc.isel(isel_arg).skt.item(),
            )
            input_builder = RRTMG_LW_INPUT(
                cxid=f"lat={lat}, lon={lon}, t=2020",
                iaer=0,
                iatm=0,
                ixsect=0,
                numangs=0,
                iout=99,
                idrv=0,
                imca=clouds,
                icld=clouds,
                tbound=-1,  # sfc temp set of temp at bottom of first layer
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
                failed_cases.append((i, j))
            # example output:

            # Wavenumbers:   10.0 - 3250.0 cm-1, ATM      1
            # LEVEL    PRESSURE   UPWARD FLUX   DOWNWARD FLUX    NET FLUX       HEATING RATE
            #             mb          W/m2          W/m2           W/m2          degree/day
            #  35        0.5000     220.0961        0.0000       220.0960511            0.00000
            #  34        1.500      219.8816        3.7730       216.1086904          -33.64992
            #  ...
            #   0      964.6        289.8190      144.6184       145.2006054           -0.69800
            #  ...
            with open(tmpdir / "OUTPUT_RRTM", "rt") as output_file:
                data = np.loadtxt(
                    output_file, skiprows=3, max_rows=len(pave), usecols=[4, 5]
                )
                toa_flux_lw[i, j] = data[0, -1]
                sfc_flux_lw[i, j] = data[0, 0]
                htr_lw[i, j] = data[1]

    failed_cases_np = np.array(failed_cases)

    toa_flux_lw.to_netcdf(args.output_dir / "toa_flux_lw.nc")
    sfc_flux_lw.to_netcdf(args.output_dir / "sfc_flux_lw.nc")
    htr_lw.to_netcdf(args.output_dir / "htr_lw.nc")
    np.savetxt(args.output_dir / "failed_cases.nc", failed_cases_np)


if __name__ == "__main__":
    main()
