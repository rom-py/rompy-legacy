# If you are working in a notebook or ipython environment, you may need to uncomment the line below to install the oceanum python library
#!pip install oceanum

import xarray as xr
# Define your datamesh token in evironment variables as DATAMESH_TOKEN or insert into argument below.
from oceanum.datamesh import Connector

datamesh = Connector()

from oceanum.datamesh import Connector

geom = [
    140,
    -26,
    157,
    -15,
]

times = ["2023-01-01T00:00:00.000Z", "2023-01-02T00:00:00.000Z"]

# Put your datamesh token in the Jupyterlab settings, or as argument in the constructor below
datamesh = Connector()

era5_mslp = datamesh.query(
    {
        "id": "00bcb40ecbeea137e7350ae9f9f9a3b3",
        "label": "era5_mslp_00bcb4",
        "geofilter": {
            "geom": geom,
            "type": "bbox",
            "interp": "linear",
        },
        "variables": ["msl"],
        "datasource": "era5_mslp",
        "timefilter": {
            "times": ["2023-01-01T00:00:00.000Z", "2023-01-04T00:00:00.000Z"]
        },
        "description": "ECMWF ERA5 global mean sea leve pressure hindcast",
    }
)
era5_wind10m = datamesh.query(
    {
        "id": "bdaab8e6341f95e35764bcd1a9f737b5",
        "label": "era5_wind10m_bdaab8",
        "geofilter": {
            "geom": geom,
            "type": "bbox",
            "interp": "linear",
        },
        "variables": ["u10", "v10"],
        "datasource": "era5_wind10m",
        "timefilter": {
            "times": ["2023-01-01T00:00:00.000Z", "2023-01-04T00:00:00.000Z"]
        },
        "description": "ECMWF ERA5 global 10m wind reanalysis",
    }
)

xr.merge([era5_mslp, era5_wind10m]).to_netcdf("era5.nc")
