# NOTE: This file previously used PySchism modules which have been removed.
# Consider updating to use PyLibs equivalents or other alternatives.
# PySchism imports have been commented out - functionality may be limited.
import logging
import os
import tarfile
from datetime import datetime
from glob import glob
from pathlib import Path
from time import time

import xarray as xr

from rompy.schism.namelists.generate_models import nml_to_dict

# from rompy.schism.pyschism.forcing.hycom.hycom2schism import DownloadHycom  # TODO: Replace with PyLibs equivalent
# from rompy.schism.pyschism.mesh.hgrid import Hgrid  # TODO: Replace with PyLibs equivalent

"""
Download hycom data for Fortran scripts.
Default is to download data for generating initial condition (use hgrid 
    as parameter to cover the whole domain).
Optionally, download data for generating th.nc (bnd=True) and nu.nc (nudge=True) 
    (use bbox as parameter to cut small domain).
"""
logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def download_hycom(dest=Path("./"), hgrid=Path("./hgrid.gr3")):
    hgrid = Hgrid.open(hgrid, crs="epsg:4326")
    date = datetime(2023, 1, 1)
    hycom = DownloadHycom(hgrid=hgrid)
    t0 = time()
    hycom.fetch_data(date, rnday=2, bnd=False, nudge=False, sub_sample=5, outdir="./")
    print(f"It took {(time()-t0)/60} mins to download")

    files = glob("hycom_*.nc")
    files.sort()
    logging.info("Concatenating files...")
    xr.open_mfdataset(files, concat_dim="time", combine="nested")["surf_el"].to_netcdf(
        dest / "hycom.nc"
    )
    for file in files:
        os.remove(file)


def compare_files(file1, file2):
    with open(file1, "r") as f1:
        with open(file2, "r") as f2:
            for line1, line2 in zip(f1, f2):
                if line1[0] != "$" and line2[0] != "$":
                    if line1.rstrip() != line2.rstrip():
                        print(f"file1': {line1}")
                        print(f"file2': {line2}")
                    assert line1.rstrip() == line2.rstrip()


def untar_file(file, dest=Path("./")):
    logger.info(f"Extracting {file} to {dest}")
    with tarfile.open(file) as tar:
        tar.extractall(dest)


# funcition to step through the namelist and compare the values
def compare_nmls_values(nml1, nml2, raise_missing=False):
    for key, value in nml1.items():
        if key == "description":
            continue
        if key not in nml2:
            if raise_missing:
                raise KeyError(f"Key {key} not found in nml2")
            print(f"Key {key} not found in nml2")
            continue
        if isinstance(value, dict):
            # if size of dictionary is 2, extract value from 'default' key
            if len(value) == 2 and "default" in value:
                var = value["default"]
                print(key, var, nml2[key])
                if var != nml2[key]["default"]:
                    print(key, var, nml2[key]["default"])
            else:
                compare_nmls_values(value, nml2[key], raise_missing=raise_missing)
        else:
            if value != nml2[key]:
                print(key, value, nml2[key])


def compare_nmls(nml1, nml2, raise_missing=False):
    d1 = nml_to_dict(nml1)
    d2 = nml_to_dict(nml2)
    d1.pop("description")
    d2.pop("description")
    compare_nmls_values(d1, d2, raise_missing=raise_missing)


if __name__ == "__main__":
    untar_file("test_data/tpxo9-neaus.tar.gz", "test_data/")
