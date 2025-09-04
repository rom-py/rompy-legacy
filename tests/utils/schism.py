"""
SCHISM-specific utility functions for tests.
"""

import logging
import os
import tarfile
from datetime import datetime
from glob import glob
from pathlib import Path
from time import time

import xarray as xr

# Initialize logger
logging.basicConfig(
    format="[%(asctime)s] %(name)s %(levelname)s: %(message)s",
    force=True,
)
logger = logging.getLogger(__name__)


def download_hycom(dest=Path("./"), hgrid=Path("./hgrid.gr3")):
    """
    Download HYCOM data for use with SCHISM tests.

    This is a PyLibs-compatible version that uses stubs by default.
    To use the actual implementation, the caller should import the necessary modules.
    """
    try:
        from rompy.schism.pyschism.forcing.hycom.hycom2schism import \
            DownloadHycom
        from rompy.schism.pyschism.mesh.hgrid import Hgrid

        hgrid = Hgrid.open(hgrid, crs="epsg:4326")
        date = datetime(2023, 1, 1)
        hycom = DownloadHycom(hgrid=hgrid)
        t0 = time()
        hycom.fetch_data(
            date, rnday=2, bnd=False, nudge=False, sub_sample=5, outdir="./"
        )
        print(f"It took {(time()-t0)/60} mins to download")

        files = glob("hycom_*.nc")
        files.sort()
        logging.info("Concatenating files...")
        xr.open_mfdataset(files, concat_dim="time", combine="nested")[
            "surf_el"
        ].to_netcdf(dest / "hycom.nc")
        for file in files:
            os.remove(file)
    except ImportError:
        logger.warning(
            "Could not import PySchism modules for HYCOM download. Using try using gzipped file."
        )
        if not (dest / "hycom.nc").exists():
            if (dest / "hycom.nc.gz").exists():
                import gzip

                logger.info(f"Unpacking {dest / 'hycom.nc.gz'}")
                with gzip.open(dest / "hycom.nc.gz", "rb") as f_in:
                    with open(dest / "hycom.nc", "wb") as f_out:
                        f_out.write(f_in.read())


def untar_file(file, dest=Path("./")):
    """Extract a tar file."""
    dest = Path(dest)
    dest.mkdir(exist_ok=True, parents=True)
    with tarfile.open(file) as tar:
        tar.extractall(path=dest)
