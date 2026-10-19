from io import TextIOBase

import numpy as np


def read_rrtmg_output(f: TextIOBase, nlev, nband):
    toa_lw = np.zeros(nband)
    htr_lw = np.zeros(nlev)

    lines = f.readlines()

    i = 0
    i += 3 # header
    i += 1 # heating rate not defined at top boundary
    for lev in range(nlev):
        _, _, _, _, _, htr = lines[i].split()
        htr_lw[lev] = htr
        i += 1
    i += 1

    for band in range(nband):
        i += 3
        print(lines[i])
        _, _, f_up_toa, _, _, _ = lines[i].split()
        toa_lw[band] = f_up_toa
        i += (nlev + 1)
        i += 1

    return toa_lw, htr_lw
