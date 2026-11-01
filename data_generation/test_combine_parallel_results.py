import numpy as np
from netCDF4 import Dataset

from combine_parallel_results import combine, completed_outputs


def test_combines_completed_timestamps_in_order(tmp_path):
    year_dir = tmp_path / "2013"
    for task, value in enumerate((10, 20, 30)):
        directory = year_dir / f"20130101T{task * 6:02d}"
        directory.mkdir(parents=True)
        (directory / "_SUCCESS").touch()
        with Dataset(directory / "htr_lw.nc", "w") as dataset:
            dataset.createDimension("time", 1)
            dataset.createDimension("level", 2)
            dataset.createVariable("time", "i8", ("time",))[:] = task
            dataset.createVariable("level", "f4", ("level",))[:] = [1, 2]
            dataset.createVariable("htr_lw", "f4", ("time", "level"))[:] = value

    paths = completed_outputs(year_dir, "htr_lw.nc")
    output_path = tmp_path / "htr_lw_2013.nc"
    assert combine(paths, output_path) == 3

    with Dataset(output_path) as output:
        assert output.variables["time"][:].tolist() == [0, 1, 2]
        assert np.isnan(output.variables["htr_lw"]._FillValue)
        assert output.variables["htr_lw"][:, 0].tolist() == [10, 20, 30]
