import numpy as np
from scipy.interpolate import make_interp_spline

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
    summol = wkl.sum(axis=0)
    wbroadl = dry_air * (1 - summol)
    return wbroadl


def vmr(q):
    eps = 0.622
    w = q / (1.0 - q)
    vmr = w / eps
    return vmr


def prepare_era5_profile(plev_ds: xr.Dataset, sp, skt):
    """
    plev_ds: dataset containing:
      t: era5 temperature on pressure levels(t)
      sp: era5 surface pressure (sp)

    skt: era5 skin temperature (skt)
    """

    plev = np.flip(plev_ds["level"].to_numpy()).astype(np.float64)
    t = np.flip(plev_ds["t"].to_numpy())
    p0 = sp / 100
    t0 = skt

    # pick where era5_pressure_level < sp
    mask = plev < p0

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
    tz0 = t0
    pz0 = p0

    # top-of-layer pressure and temperatures
    p1 = np.zeros_like(pave)
    p1[:-1] = 0.5 * (pave[1:] + pave[:-1])
    p1[-1] = 0.5

    # temperature at interfaces by linear interpolation on log(p)
    spl = make_interp_spline(x=-np.log(pave), y=-tave, k=1)  # k=1 linear
    t1 = -spl(-np.log(p1), extrapolate="true")
    t1[-1] = tave[-1]

    # pressure interfaces for wkl calculation
    pint = np.zeros(n_lev + 1)
    pint[0] = p0
    pint[1:] = p1

    # wkl fill values
    wkl = np.zeros((7, n_lev))
    wkl[0] = vmr(plev_ds["q"].to_numpy()[mask])
    wkl[1] = 330e-6  # co2 vmr
    wkl[2] = plev_ds["o3"].to_numpy()[mask]
    wkl[3] = 3.16e-07  # n20 vmr
    wkl[4] = 1.5e-7  # CO vmr
    wkl[5] = 1.8 * 1e-6  # ch4 vmr
    wkl[6] = 0.209  # O2 cmr 20.9%

    # broadening calculation
    wbroadl = broad(pint, wkl)
    print(wkl)
    print(wbroadl)

    return pave, tave, pz0, tz0, p1, t1, wkl, wbroadl
