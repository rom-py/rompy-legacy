from dask.diagnostics.progress import ProgressBar
from oceanum.datamesh import Connector

dataset_id = "tpxo9_v5a_cons"  # ~16km tpxo data
# dataset_id = "??"  # atlas
# dataset_id = "oceanum_global_tide_cons_v2" # 400m downscaled tpxo data

datamesh = Connector()
dset = datamesh.load_datasource(dataset_id)
ds = dset.sel(lon=slice(145, 154.5), lat=slice(-25, -16))
with ProgressBar():
    ds = ds.compute()
ds.tide.to_otis_netcdf("./")
