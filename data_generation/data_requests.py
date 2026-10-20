import calendar
from pathlib import Path

import cdsapi

# fmt: off
single_level_vars = {
    "t2m" : "2m_temperature",
    "skt" : "skin_temperature",
    "fal" : "forecast_albedo",
    "sp"  : "surface_pressure",
    "tisr": "toa_incident_solar_radiation",
    "tsr" : "top_net_solar_radiation",
    "tsrc": "top_net_solar_radiation_clear_sky",
    "ttr" : "top_net_thermal_radiation",
    "ttrc": "top_net_thermal_radiation_clear_sky",
}

pressure_level_vars = {
    "o3": "ozone_mass_mixing_ratio",
    "q" : "specific_humidity",
    "t" : "temperature"
    # 'fraction_of_cloud_cover',
    # 'geopotential',
    # 'specific_cloud_ice_water_content',
    # 'specific_cloud_liquid',
    # 'specific_rain_water_content',
    # 'specific_snow_water_content'
}



# fmt: off
ALL_PRESSURE_LEVELS = [
  "1",   "2",   "3",   "5",   "7",   "10",
  "20",  "30",  "50",  "70",  "100", "125",
  "150", "175", "200", "225", "250", "300",
  "350", "400", "450", "500", "550", "600",
  "650", "700", "750", "775", "800", "825",
  "850", "875", "900", "925", "950", "975",
  "1000",
]
ALL_TIMES = ["00:00", "06:00", "12:00", "18:00"]
# fmt: on


def all_days(year, month):
    _, n_days = calendar.monthrange(year, month)
    days = [f"{d + 1:02d}" for d in range(n_days)]
    return days


c = cdsapi.Client(
    url="https://cds.climate.copernicus.eu/api",
    key="3dbc229c-81f7-4fd5-be2f-c6f193729ef5",
)


def era5_request_plev(year: str, month: str, var: str, grid: str, out: Path):
    c.retrieve(
        "reanalysis-era5-pressure-levels",
        {
            "product_type": "reanalysis",
            "pressure_level": ALL_PRESSURE_LEVELS,
            "variable": [pressure_level_vars[var]],
            "year": year,
            "month": month,
            "day": all_days(year, month),
            "time": ALL_TIMES,
            "format": "netcdf",
            "grid": grid,
        },
        str(out),
    )


def era5_request_slev(year: str, month: str, var: str, grid: str, out: Path):
    c.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [single_level_vars[var]],
            "year": year,
            "month": month,
            "day": all_days(year, month),
            "time": ALL_TIMES,
            "format": "netcdf",
            "grid": grid,
        },
        str(out),
    )


if __name__ == "__main__":
    base_path = Path("data/era5_6hr_2.5")
    years = ["2011", "2012", "2013", "2014", "2015"]
    months = [f"{m:02d}" for m in range(1, 13)]
    grid = '2.5/2.5'

    for var in single_level_vars:
        (base_path / var).mkdir(parents=True, exist_ok=True)
        for year in years:
            for month in months:
                out_path = base_path / var / f"{year}-{month}.nc"
                if out_path.exists():
                    print(f"{var}/{year}-{month} already exists, skipping")
                    continue

                era5_request_slev(year, month, var, grid, out_path)

    for var in pressure_level_vars:
        (base_path / var).mkdir(parents=True, exist_ok=True)
        for year in years:
            for month in months:
                out_path = base_path / var / f"{year}-{month}.nc"
                if out_path.exists():
                    print(f"{var}/{year}-{month} already exists, skipping")
                    continue

                era5_request_plev(year, month, var, grid, out_path)
