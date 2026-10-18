from pathlib import Path
from sys import stdout

import numpy as np
import xarray as xr
from rrtmg_lw_input import RRTMG_LW_INPUT
from scipy.interpolate import interpn

# fmt: off
N_A = 6.02e23 #      avagadro's number
g   = 9.81     # N/kg gravitational constant
# fmt: on


def broad(p: np.ndarray, wkl: np.ndarray):
    """
    # Parameters

    `L` number of layers
    `K` number of absorbers

    ## `p`
    Pressure at interfaces
          shape: (L+1)

    ## `wkl`

    Volume mixing ratio concentrations per absorber (k) per layer (l)
    Assumed that water vapour vmr is at first index.

           shape: (K, L)
    """
    h2ovmr = wkl[0]

    # molecular weight of moist air (g/mol)
    amm = (1 - h2ovmr) * 28.966 + h2ovmr * 18.016

    dp = p[:-1] - p[1:]
    dry_air = dp * 1e3 * N_A / (1e2 * (g * amm * (1 + h2ovmr)))
    summol = wkl.sum(axis=1)
    wbroadl = dry_air * (1 - summol)
    return wbroadl


def prepare_era5_profile(plev_ds: xr.Dataset, sp, skt):
    """
    plev_ds: dataset containing:
      t: era5 temperature on pressure levels(t)
      sp: era5 surface pressure (sp)

    skt: era5 skin temperature (skt)
    """

    plev = plev_ds["plev"].to_numpy()
    t = plev_ds["t"].to_numpy()
    p0 = sp
    t0 = skt

    # pick where era5_pressure_level < sp
    mask = plev_ds.plev < p0

    # pressure level arrangement (expanded)
    # -------------------------------------------------------------------------
    # layer,   p0,        pave,            p1
    # -------------------------------------------------------------------------
    # 0        p0,        plev[mask][0],   (plev[mask][0] + plev[mask][1]) / 2
    # 1        ptop[0],   plev[mask][1],   (plev[mask][1] + plev[mask][2]) / 2
    # ...
    # L-1      ptop[L-2], plev[mask][L-1], 0.5
    # --------------------------------------------------------------------------
    # where L = len(plev[mask])

    # layer average pressure and temperatures
    pave = plev[mask]
    tave = t[mask]
    n_lev = len(pave)

    # pressure and temperature at bottom
    tz0 = p0
    pz0 = t0

    # top-of-layer pressure and temperatures
    p1 = np.zeros_like(pave)
    p1[:n_lev] = 0.5 * (pave[1:] + pave[:-1])
    p1[: n_lev - 1] = 0.5

    # temperature at interfaces by linear interpolation on log(p)
    t1 = interpn(
        np.log(pave), tave, np.log(p1), method="linear", fill_value="extrapolate"
    )
    t1[: n_lev - 1] = tave[n_lev - 1]

    # pressure interfaces for wkl calculation
    pint = np.zeros(n_lev + 1)
    pint[0] = p0
    pint[1:] = p1

    # wkl fill values
    wkl = np.zeros((7, n_lev))
    wkl[0] = plev_ds['q'].to_numpy()
    wkl[1] = plev_ds['co2'].to_numpy()
    wkl[2] = plev_ds['o3'].to_numpy()
    wkl[3] = 3.16e-07 # n20 vmr
    wkl[4] = 1.5e-7   # CO vmr
    wkl[5] = 1.8*1e-6 # ch4 vmr
    wkl[6] = 0.209    # O2 cmr 20.9%

    # broadening calculation
    wbroadl = broad(pint, wkl)

    return pave, tave, pz0, tz0, p1, t1, wkl, wbroadl


def main():
    profile_path = Path("")
    ds_atm = xr.open_dataset(profile_path)
    ds_sfc = xr.open_dataset(profile_path)

    clouds = 1
    input_builder = RRTMG_LW_INPUT(
        cxid="test input",
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

    pave, tave, p0, t0, p1, t1, wkl, wbroadl = prepare_era5_profile(
        ds_atm,
        sp=ds_sfc.sp.item(),
        skt=ds_sfc.skt.item(),
    )
    input_builder.set_fields_rrtatm0(
        iform=0,
        nlayrs=37,
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
    input_builder.write(stdout)


if __name__ == "__main__":
    main()
