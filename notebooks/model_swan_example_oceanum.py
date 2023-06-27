#!/usr/bin/env python

import numpy as np
import pandas as pd
import wavespectra as ws

import rompy
from rompy.filters import crop_filter, rename_filter, sort_filter
from rompy.swan import SwanModel

new_simulation = SwanModel(
    run_id="test_swan", template="../rompy/templates/swan", output_dir="simulations"
)


new_simulation.settings
grid = new_simulation.grid
# grid.plot()
min_lon, min_lat, max_lon, max_lat = grid.bbox(buffer=0.05)

startdt = pd.to_datetime("2020-12-14 00:00")
enddt = startdt + pd.to_timedelta(1.0, "d")

new_simulation.settings["compute_start"] = startdt.strftime("%Y%m%d.%H%M%S")
new_simulation.settings["compute_stop"] = enddt.strftime("%Y%m%d.%H%M%S")
new_simulation.settings["out_start"] = new_simulation.settings["compute_start"]
new_simulation.settings


ds_wind = rompy.cat.oceanum.era5_wind10m.to_dask()
# ds_wind

ds_wind = sort_filter(ds_wind, coords=["latitude"])
ds_wind = crop_filter(
    ds_wind,
    **{
        "longitude": slice(min_lon, max_lon),
        "latitude": slice(min_lat, max_lat),
        "time": slice(startdt, enddt),
    }
)

ds_wind.load()

# ds_wind

inpwind, readwind = ds_wind.swan.to_inpgrid(
    new_simulation.staging_dir + "/wind.inp", var="WIND", grid=grid, z1="u10", z2="v10"
)
# print(inpwind)
# print(readwind)
new_simulation.settings["wind_grid"] = inpwind
new_simulation.settings["wind_read"] = readwind


ds_spec = rompy.cat.oceanum.oceanum_wave_glob05_era5_v1_spec().to_dask()
# ds_spec = rename_filter(ds_spec, **{'dir':'mean_dir'})
ds_spec = crop_filter(ds_spec, **{"time": slice(startdt, enddt)})


ds_spec = ws.read_dataset(ds_spec)  # Convert it to wavespectra format
ds_spec = ds_spec.sortby("dir")
# ds_spec


ds_boundary = grid.nearby_spectra(ds_spec, dist_thres=1)

# check = [st == wt for st, wt in zip(ds_spec.time.values, ds_wind.time.values)]
# any(check)

ds_boundary.spec.to_swan(
    new_simulation.staging_dir + "/" + new_simulation._default_context["spectra_file"]
)

output_locs = []

bbox = grid.bbox()

output_locs = pd.read_csv("out.loc", delim_whitespace=True).values.tolist()

# from owslib.wfs import WebFeatureService
#
# wfs_url = "http://geoserver-123.aodn.org.au/geoserver/wfs"
# wfs_endpoint = "aodn:aodn_wave_nrt_timeseries_map"
# wfs = WebFeatureService(wfs_url, version="2.0.0")
# response = wfs.getfeature(typename=wfs_endpoint, bbox=tuple(bbox), outputFormat="csv")
# df = pd.read_csv(response)
# df = df.groupby(["wmo_id", "latitude", "longitude"]).mean()
#
# for (wmo_id, yp, xp), data in df.iterrows():
#     output_locs.append([str(round(xp, 2)), str(round(yp, 2))])

spec_int = 0.05
xx = np.arange(round(bbox[0] + spec_int, 2), round(bbox[2] - spec_int, 2), spec_int)
yy = np.arange(round(bbox[1] + spec_int, 2), round(bbox[3] - spec_int, 2), spec_int)
specout_x, specout_y = np.meshgrid(xx, yy)
for i, (xp, yp) in enumerate(zip(specout_x.ravel(), specout_y.ravel())):
    output_locs.append([str(round(xp, 2)), str(round(yp, 2))])


new_simulation.settings["output_locs"] = {"coords": output_locs}


new_simulation.settings


new_simulation.generate()

ds_boundary.to_netcdf(new_simulation.staging_dir + "/downloads/oceanum_ww3_spec.nc")
ds_wind.to_netcdf(new_simulation.staging_dir + "/downloads/era5_wind.nc")

new_simulation.staging_dir
