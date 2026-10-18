from typing import Literal, TextIO

import numpy as np


class RRTMG_LW_INPUT:
    def __init__(
        self,
        cxid: str,
        iaer: Literal[0, 10],
        iatm: Literal[0, 1],
        ixsect: Literal[0, 1],
        numangs: Literal[0, 1],
        iout: Literal[-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 99],
        idrv: Literal[0, 1],
        imca: Literal[0, 1],
        icld: Literal[0, 1, 2, 3, 4, 5],
        tbound: float,
        iemis: Literal[0, 1, 2],
        ireflect: Literal[0, 1],
        semiss: np.ndarray | None = None,
        dtbound: float | None = None,
    ):
        """
        # Parameters:

        ## `cxid`
        (max) 80 characters of user identification

        ## `iaer`
        flag for aerosols
        = 0 no layers contain aerosols
        = 10 one or more layers contain aerosols
          (only absorption is treated)
          (requires the presence of file IN_AER_RRTM)

        ## `iatm`
        flag for RRTATM; 1=yes

        ## `ixsect`
        flag for cross sections
        = 1 no cross sections
        = 0 cross-sections included in calculation

        currently no cross sections supported.

        ## `numangs`
        = 0 radiace will be computed at 1 angle, the cosine of which equals 1/1.66
        (standard diffusivity angle approximation)
        (default and only option available)

        ## `iout`
        = -1 if no output is to be printed out
        = 0 if the only output is for 10-3250 cm-1
        = n (n = 1-16) if the only output is from band in
        = 99 if output is generated for 17 spectral intervals, one for the full longwave spectrum (10-3250 cm-1), and one for each of the 16 bands.

        ## `idrv`
        flag for applying optional adjustment to the upward flux at each layer based on the derivative of the Planck function with respect to temperature for the change in surface temperature (DTBOUND) provided on record 1.4.1 relative to the surface temperature (TBOUND) provided on record 1.4
          = 0  standard forward calculation; do not apply derivative
               adjustment (default)
          = 1  adjust upward flux at each layer based on derivative of
               the Planck function to account for the change in surface
                           temperature defined by DTBOUND

                                ## `imca`
                                flag for McICA (Monte Carlo Independent Column Approximation) for statistical representation of sub-grid cloud fraction and cloud overlap
                  = 0  standard forward calculation; do not use McICA
                  = 1  use McICA (will perform statistical sample of 200 forward

        ## `icld`
        flag for clouds
          = 0  no cloudy layers in atmosphere
          = 1  one or more cloudy layers present in atmosphere.  Cloud layers are treated using a RANDOM overlap assmption. (requires the presence of file IN_CLD_RRTM for column model) (available for IMCA = 0 or 1)
          = 2  one or more cloudy layers present in atmosphere.  Cloud layers are treated using a MAXIMUM/RANDOM overlap assmption. (requires the presence of file IN_CLD_RRTM for column model) (available for IMCA = 0 or 1)
          = 3  one or more cloudy layers present in atmosphere.  Cloud layers are treated using a MAXIMUM overlap assmption. (requires the presence of file IN_CLD_RRTM for column model) (available only for IMCA = 1)
          = 4  one or more cloudy layers present in atmosphere.  Cloud layers are treated using a EXPONENTIAL overlap assmption. (requires the presence of file IN_CLD_RRTM for column model, and inputs set on Records 1.5 and either 1.5.1 or 1.5.2) (available only for IMCA = 1)
          = 5  one or more cloudy layers present in atmosphere.  Cloud layers are treated using a EXPONENTIAL-RANDOM overlap assmption. (requires the presence of file IN_CLD_RRTM for column model, and inputs set on Records 1.5 and either 1.5.1 or 1.5.2) (available only for IMCA = 1)

        ## `tbound`
        temperature at the surface.  If input value < 0, then surface temperature is set to the temperature at the bottom of the first layer (see Record 2.1.1)

        ## `iemis`
          = 0 each band has surface emissivity equal to 1.0
          = 1 each band has the same surface emissivity (equal to SEMISS(1))
          = 2 each band has different surface emissivity (for band IB, equal to SEMISS(IB))

        ## `ireflect`
          = 0 for Lambertian reflection at surface, i.e. reflected radiance is equal at all angles (default)
          = 1 for specular reflection at surface, i.e. reflected radiance at angle is equal to downward surface radiance at same angle multiplied by the reflectance (option not available in RRTMG_LW)
                  (Note:  for each band, surface reflectance = 1 - surface emissivity)

        ## `semiss`
        the surface emissivity for each band (see Table I).  All values must be greater than 0 and less than or equal to 1. If IEMIS = 1, only SEMISS(1) is considered.  If IEMIS = 2 and no surface emissivity value is given for SEMISS(IB), a value of 1.0 is used for band IB.

        ## `dtbound`
        change in surface temperature relative to TBOUND used to adjust upward flux at each layer based on the derivative of the Planck function with respect to surface temperature.

        """
        assert len(cxid) < 80
        assert iatm == 0, "only supports user-inputed ATM"

        self.cxid = cxid
        self.iaer = iaer
        self.iatm = iatm
        self.ixsect = ixsect
        self.numangs = numangs
        self.iout = iout
        self.idrv = idrv
        self.imca = imca

        assert icld in {0, 1}, "only no cloudy layers and random overlap is implemented"
        assert ixsect == 1, "no cross-sections included in calculation implemented"
        self.icld = icld

        # record 1.4
        if iemis > 0:
            assert semiss != None
            if iemis == 1:
                assert len(semiss) == 1
            if iemis == 2:
                assert len(semiss) == 16

        self.tbound = tbound
        self.iemis = iemis
        self.ireflect = ireflect
        self.semiss = semiss

        # record 1.4.1
        # flag for applying optional adjustment to the upward flux at each layer based on the derivative of the Planck function with respect to temperature for the change in surface temperature (DTBOUND) provided on record 1.4.1 relative to the surface temperature (TBOUND) provided on record 1.4
        if idrv == 1:
            assert dtbound != 0

        self.dtbound = dtbound

    def set_fields_rrtatm0(
        self,
        iform: Literal[0, 1],
        nlayrs: int,
        nmol: int,
        pave: np.ndarray,
        tave: np.ndarray,
        pz0: float,
        tz0: float,
        pz1: np.ndarray,
        tz1: np.ndarray,
        wkl: np.ndarray,
        wbroadl: np.ndarray,
        ixmols: int,
    ):
        """
        ## Parameters
        column amount format flag

        # `iform`
        (0,1) column amount format flag
                          = 0  read PAVE, WKL(M,L), WBROADL(L) in F10.4, E10.3, E10.3 formats (default)
                          = 1  read PAVE, WKL(M,L), WBROADL(L) in E15.7 format

        # `nlayrs`
        number of layers (maximum of 200)

        # `nmol`
        value of highest molecule number used (default = 7; maximum of 35)

        # `pave`
        average pressure of layer (millibars) (**If IFORM=1, then PAVE in E15.7 format**)

        shape: (L)

        # `tave`
        average temperature of layer (K)

        shape: (L)

        # `pz0`
        pressure at bottom of layer L

        shape: (L)

        # `tz0`
        temperature at bottom of layer L  -  used by RRTM for Planck Function Calculation

        shape: (L)

        ** NOTE **  PZ(L-1) and TZ(L-1) are only required for the first layer.  RRTM assumes that
                    these quantites are equal to the top of the previous layer for L > 1.

        # `pz1`
        pressure at top of layer L

        shape: (L)

        # `tz1`
        temperature at top of layer L  -  used by RRTM for Planck Function Calculation

        shape: (L)

        # `wkl`
        column densities or mixing ratios for 7 molecular species (molecules/cm**2)

        shape (M, L)
        M - molecule
        L - layer

        # `wkl`
        column density for broadening gases (molecules/cm**2)

        **NOTE** If IFORM=1, then WKL(M,L) and WBROADL(L) are in 8E15.7 format

        """
        self.iform = iform
        assert nlayrs <= 200
        self.nlayrs = nlayrs
        self.nmol = nmol
        self.pave = pave
        self.tave = tave
        self.pz0 = pz0
        self.tz0 = tz0
        self.pz1 = pz1
        self.tz1 = tz1
        self.wkl = wkl
        self.wbroad = wbroadl

    def set_fields_rrtatm1(
        self,
        model: Literal[0, 1, 2, 3, 4, 5, 6],
        ibmax: int,
        noprnt: int,
        nmol: int,
        hbound: float,
        htoa: float,
        ipunch: int = 0,
        munits=1,
        re=0,
        co2mx=330,
        ref_lat=0,
    ):
        """
        ## Parameters


        # `model`
        selects atmospheric profile
        = 0  user supplied atmospheric profile
        = 1  tropical model
        = 2  midlatitude summer model
        = 3  midlatitude winter model
        = 4  subarctic summer model
        = 5  subarctic winter model
        = 6  U.S. standard 1976

        # `ibmax`
        selects layering for RRTM

        = 0  RRTM layers are generated internally (default)
        > 0  IBMAX is the number of layer boundaries read in on Record 3.3B which are used to define the layers used in RRTM calculation
                         If IBMAX is set to a negative value, the layer boundaries are
                                specified in PRESSURE (mbars).

        # `noprnt`
        = 0  full printout
        = 1  selects short printout

        # `nmol`
        number of molecular species (default = 7; maximum value is 35)

        # `ipunch`
        = 0  layer data not written (default)
        = 1  layer data written to unit IPU (TAPE7)

        # `munits`
        = 0  write molecular column amounts to TAPE7 (if IPUNCH = 1, default)
        = 1  write molecular mixing ratios to TAPE7 (if IPUNCH = 1)

        # `re`
        radius of earth (km)
              defaults for RE=0:
                 a) MODEL 0,2,3,6    RE = 6371.23 km
           b)       1          RE = 6378.39 km
                             c)       4,5        RE = 6356.91 km

        # `co2mx`
        mixing ratio for CO2 (ppm). Default is 330 ppm.

        # `ref_lat`
        latitude of location of calculation (degrees) defaults for REF_LAT = 0:
                     a) MODEL 0,2,3,6    REF_LAT = 45.0 degrees
                     b) MODEL 1          REF_LAT = 15.0
                     c) MODEL 4,5        REF_LAT = 60.0

        # `hbound`
        altitude of the surface (km)

        If ibmax is negative, H1 is provided in pressure units (mbars)

        # `htoa`
        altitude of the top of the atmosphere (km)

        If ibmax is negative, H2 is provided in pressure units (mbars)
        """
        assert nmol <= 35, "maximum number of molecular species is 35"
        self.model = model
        self.ibmax = ibmax
        self.noprnt = noprnt
        self.nmol = nmol
        self.ipunch = ipunch
        self.munits = munits
        self.re = re
        self.co2mx = co2mx
        self.ref_lat = ref_lat
        self.hbound = hbound
        self.htoa = htoa

    def set_fields_rrtatm1a(
        self, avtrat: float, tdiff1: float, tdiff2: float, altd1: float, altd2: float
    ):
        """
        # Parameters
        `AVTRAT`   maximum Voigt width ratio across a layer
                   (if zero, default = 1.5)

        `TDIFF1`   maximum layer temperature difference at
                   ALTD1 (if zero, default =  5 K)

        `TDIFF2`   maximum layer temperature difference at
                   ALTD2 (if zero, default = 8 K)

        `ALTD1`    altitude of TDIFF1 (if zero, default = 0 Km)

        `ALTD2`    altitude of TDIFF2 (if zero, default = 100 Km)
        """
        self.avtrat = avtrat
        self.tdiff1 = tdiff1
        self.tdiff2 = tdiff2
        self.altd1 = altd1
        self.altd2 = altd2

    def set_fields_rrtatm1b(self, zbnd, pbnd):
        pass

    def _write_record_1_1(self, stream: TextIO):
        stream.write(f"${self.cxid}\n")

    def _write_record_1_2(self, stream: TextIO):
        stream.write(
            f"{self.iaer:20d}{self.iatm:30d}{self.ixsect:20d}{self.numangs:15d}{self.iout:5d}{self.idrv:2d}{self.imca:2d}{self.icld:1d}\n"
        )

    def _write_record_1_4(self, stream: TextIO):
        # record 1.4
        semiss_record = ""
        if self.semiss != None:
            self.semiss_str = "".join([f"{self.semiss[i]:5.3f}" for i in range(16)])

        stream.write(
            f"{self.tbound:10.3f}{self.iemis:2d}{self.ireflect:3d}{semiss_record}\n"
        )

        # record 1.4.1
        if self.idrv:
            assert self.dtbound != None
            stream.write(f"{self.dtbound:10.3f}\n")

    def _write_record_1_5(self, stream: TextIO):
        pass

    def _write_record_2_1(self, stream: TextIO):
        # record 2.1
        stream.write(f"{self.iform:2d}{self.nlayrs:3d}{self.nmol:5d}\n")

        # record 2.1.1-2.1.3 for each layer
        if self.iform == 0:
            fmt_fn_pave = lambda x: f"{x:10.4f}"
            fmt_fn_wkl = lambda x: f"{x:10.3e}"
        if self.iform == 1:
            fmt_fn_pave = lambda x: f"{x:15.7e}"
            fmt_fn_wkl = lambda x: f"{x:15.7e}"

        for l in range(self.nlayrs):
            # record 2.1.1
            # bottom-of-layer temerature and pressure only needed for bottom layer
            if l == 0:
                pz0 = f"{self.pz0:8.3f}"
                tz0 = f"{self.tz0:7.2f}"
            else:
                pz0 = " " * 8
                tz0 = " " * 7

            stream.write(
                f"{fmt_fn_pave(self.pave[l])}{self.tave[l]:10.4f}{' ' * 23}{pz0}{tz0}{' ' * 7}{self.pz1[l]:8.3f}{self.tz1[l]:7.2f}\n"
            )

            # record 2.1.2
            wkl_parts = self.wkl[:7, l].tolist() + self.wbroad[l]
            wkl_parts_fmt = [f"{fmt_fn_wkl(x)}" for x in wkl_parts]
            stream.write(f"{' '.join(wkl_parts_fmt)}\n")

            # record 2.1.3 - only if nmol > 7
            if self.nmol <= 7:
                continue

            wkl_parts_fmt = [f"{fmt_fn_wkl(x)}" for x in self.wkl[7:, l]]
            stream.write(f"{' '.join(wkl_parts_fmt)}\n")

    def _write_record_2_2(self, stream: TextIO):
        pass

    def write(self, stream: TextIO):
        self._write_record_1_1(stream)
        self._write_record_1_2(stream)
        self._write_record_1_4(stream)

        if self.iatm == 0:
            self._write_record_2_1(stream)
