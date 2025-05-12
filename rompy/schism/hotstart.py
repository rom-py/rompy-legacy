"""
Module for generating SCHISM hotstart files.

This module provides functionality to create hotstart.nc files for SCHISM
by interpolating temperature and salinity data from source datasets to the SCHISM grid.
"""

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import scipy as sp
import xarray as xr
from pydantic import ConfigDict, Field
from pylib import WriteNC, datenum, loadz, zdata

from rompy.core.data import DataBlob, DataGrid
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel
from rompy.schism.grid import SCHISMGrid

logger = logging.getLogger(__name__)


class SCHISMDataHotstart(DataGrid):
    """
    This class is used to generate a hotstart file for SCHISM based on source data.

    It inherits from DataGrid and uses the source dataset to interpolate temperature
    and salinity data to the SCHISM grid and create a hotstart.nc file.
    """

    data_type: Literal["hotstart"] = Field(
        default="hotstart",
        description="Model type discriminator",
    )
    temp_var: str = Field(
        "water_temp", description="Name of temperature variable in source dataset"
    )
    salt_var: str = Field(
        "salinity", description="Name of salinity variable in source dataset"
    )
    time_offset: float = Field(
        0.0, description="Offset to add to source time values (in days)"
    )
    time_base: datetime = Field(
        datetime(2000, 1, 1), description="Base time for source time values"
    )
    output_filename: str = Field(
        "hotstart.nc", description="Name of the output hotstart file"
    )

    def get(
        self,
        destdir: Union[str, Path],
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """
        Generate a hotstart file for SCHISM based on source data.

        Parameters
        ----------
        destdir : str | Path
            Destination directory for the hotstart file.
        grid : SCHISMGrid
            SCHISM grid to interpolate data to.
        time : Optional[TimeRange]
            Time range for the data (not used, as hotstart is for a single time).

        Returns
        -------
        str
            Path to the generated hotstart file.
        """
        logger.info(f"Generating hotstart file in {destdir}")
        destdir = Path(destdir)
        destdir.mkdir(parents=True, exist_ok=True)
        output_path = destdir / self.output_filename

        target_time = time.start

        # Convert to datenum format for pylibs
        start_t = datenum(
            target_time.year,
            target_time.month,
            target_time.day,
            target_time.hour,
            target_time.minute,
            target_time.second,
        )

        # Get dataset directly using DataGrid's ds() method
        ds = self.ds

        # Find the closest time in the dataset
        if self.coords.t in ds.dims or self.coords.t in ds.coords:
            # Convert time to a format we can compare with start_t
            if hasattr(ds[self.coords.t].values[0], "astype"):
                # Numeric time values
                time_values = ds[self.coords.t].values
                # Convert to datenum format for comparison
                if hasattr(time_values[0], "tolist"):
                    time_values = (
                        time_values.astype(float) / 24.0
                        + datenum(
                            self.time_base.year,
                            self.time_base.month,
                            self.time_base.day,
                        )
                        + self.time_offset
                    )
            else:
                # Datetime values
                time_values = np.array(
                    [
                        datenum(t.year, t.month, t.day, t.hour, t.minute, t.second)
                        for t in pd.to_datetime(ds[self.coords.t].values)
                    ]
                )

            # Find closest time index
            time_idx = np.argmin(np.abs(time_values - start_t))
            logger.info(
                f"Using time index {time_idx} (closest to requested start time)"
            )

            # Select the data at this time
            if self.coords.t in ds.dims:
                ds = ds.isel({self.coords.t: time_idx})
        else:
            logger.warning(
                f"Time variable '{self.coords.t}' not found in dataset. Using all data."
            )

        # Read grid information
        # We need to access the pylibs grid objects directly
        if not hasattr(grid, "pylibs_hgrid") or grid.pylibs_hgrid is None:
            # Load the grid if not already loaded
            grid.load()

        # Get grid dimensions from the pylibs objects
        gd = grid.pylibs_hgrid
        vd = grid.pylibs_vgrid

        # Get the number of elements, nodes, and sides
        ne = gd.ne
        np_grid = gd.np

        # For ns (number of sides), we need to check if it's available
        # If not, we can use the length of the side arrays if available
        if hasattr(gd, "ns"):
            ns = gd.ns
        elif hasattr(gd, "isidenode"):
            ns = len(gd.isidenode)
        else:
            # If we can't determine ns, use a reasonable default
            # For triangular meshes, ns is approximately 1.5 * ne
            ns = int(ne * 1.5)
            logger.warning(
                f"Could not determine number of sides, using estimated value: {ns}"
            )

        # Get the number of vertical layers
        nvrt = vd.nvrt

        # Get node coordinates
        lxi = gd.x % 360  # Convert to 0-360 longitude range
        lyi = gd.y
        lzi0 = np.abs(vd.compute_zcor(gd.dp)).T  # Vertical coordinates

        # Get source coordinates
        sx = (
            np.array(ds[self.coords.x].values) % 360
        )  # Convert to 0-360 longitude range
        sy = np.array(ds[self.coords.y].values)
        sz = np.array(ds[self.coords.z].values)

        # Ensure vertical coordinates don't exceed source data range
        fpz = lzi0 >= sz.max()
        lzi0[fpz] = sz.max() - 1e-6

        # Check for NaN values in vertical coordinates
        nan_mask = np.isnan(lzi0)
        if np.any(nan_mask):
            logger.warning(
                f"Found {np.sum(nan_mask)} NaN values in vertical coordinates"
            )
            # Replace NaNs with a valid depth value
            valid_depth = np.nanmean(lzi0)
            if np.isnan(valid_depth):  # If all values are NaN
                valid_depth = 0.0
            lzi0[nan_mask] = valid_depth

        # Initialize data structure for interpolated variables
        S = zdata()
        mvars = ["temp", "salt"]
        svars = [self.temp_var, self.salt_var]

        # Map variable names to dataset variable names using coords
        if hasattr(self, "coords") and hasattr(self.coords, "var"):
            var_mapping = self.coords.var
            if isinstance(var_mapping, dict) and self.temp_var in var_mapping:
                svars[0] = var_mapping[self.temp_var]
            if isinstance(var_mapping, dict) and self.salt_var in var_mapping:
                svars[1] = var_mapping[self.salt_var]

        for var in mvars:
            setattr(S, var, [])

        # Interpolate for each vertical level
        for k in range(nvrt):
            lzi = lzi0[k]
            bxyz = np.c_[lxi, lyi, lzi]

            # Get interpolation indices
            idx = ((lxi[:, None] - sx[None, :]) >= 0).sum(axis=1) - 1
            ratx = (lxi - sx[idx]) / (sx[idx + 1] - sx[idx])

            idy = ((lyi[:, None] - sy[None, :]) >= 0).sum(axis=1) - 1
            raty = (lyi - sy[idy]) / (sy[idy + 1] - sy[idy])

            idz = ((lzi[:, None] - sz[None, :]) >= 0).sum(axis=1) - 1
            ratz = (lzi - sz[idz]) / (sz[idz + 1] - sz[idz])

            # Handle edge cases
            idx = np.clip(idx, 0, len(sx) - 2)
            idy = np.clip(idy, 0, len(sy) - 2)
            idz = np.clip(idz, 0, len(sz) - 2)

            # For each variable
            for m, svar in enumerate(svars):
                logger.info(f"Interpolating {svar} at level {k+1}/{nvrt}")
                mvar = mvars[m]

                # Get source data for this variable
                try:
                    if svar in ds.variables:
                        if len(ds[svar].dims) == 4:  # time, depth, lat, lon
                            cv = ds[svar].values[0]  # First time step
                        elif len(ds[svar].dims) == 3:  # depth, lat, lon
                            cv = ds[svar].values
                        else:
                            raise ValueError(
                                f"Unexpected dimensions for variable {svar}: {ds[svar].dims}"
                            )
                    else:
                        # Try to find the variable using alternative names
                        alt_names = (
                            [
                                v
                                for k, v in self.coords.var.items()
                                if k in [self.temp_var, self.salt_var]
                            ]
                            if hasattr(self, "coords") and hasattr(self.coords, "var")
                            else []
                        )
                        found = False
                        for alt_name in alt_names:
                            if alt_name in ds.variables:
                                if len(ds[alt_name].dims) == 4:  # time, depth, lat, lon
                                    cv = ds[alt_name].values[0]  # First time step
                                    found = True
                                    break
                                elif len(ds[alt_name].dims) == 3:  # depth, lat, lon
                                    cv = ds[alt_name].values
                                    found = True
                                    break
                        if not found:
                            raise ValueError(
                                f"Could not find variable {svar} or any alternative names in dataset"
                            )
                except Exception as e:
                    logger.error(f"Error accessing variable {svar}: {e}")
                    raise

                # Extract values at interpolation points with error handling
                try:
                    v0 = np.array(
                        [
                            cv[idz, idy, idx],
                            cv[idz, idy, idx + 1],
                            cv[idz, idy + 1, idx],
                            cv[idz, idy + 1, idx + 1],
                            cv[idz + 1, idy, idx],
                            cv[idz + 1, idy, idx + 1],
                            cv[idz + 1, idy + 1, idx],
                            cv[idz + 1, idy + 1, idx + 1],
                        ]
                    )

                    # Check for NaN values in the extracted data
                    nan_mask = np.isnan(v0)
                    if np.any(nan_mask):
                        logger.warning(
                            f"Found NaN values in extracted data for {svar} at level {k+1}"
                        )
                        # Replace NaNs with valid values where possible
                        for n in range(8):
                            if np.any(np.isnan(v0[n])):
                                valid_indices = ~np.isnan(v0[n])
                                if np.any(valid_indices):
                                    # Use mean of valid values
                                    v0[n, np.isnan(v0[n])] = np.mean(
                                        v0[n, valid_indices]
                                    )
                                else:
                                    # If all values are NaN, use a default value
                                    v0[n, np.isnan(v0[n])] = 0.0  # Default value

                    # Remove invalid points (very large values)
                    for n in range(8):
                        fpn = np.abs(v0[n]) > 1e3
                        if np.any(fpn):
                            # Use nearest neighbor interpolation for invalid points
                            if np.sum(~fpn) > 0:  # Only if we have valid points
                                try:
                                    v0[n, fpn] = sp.interpolate.griddata(
                                        bxyz[~fpn, :],
                                        v0[n, ~fpn],
                                        bxyz[fpn, :],
                                        "nearest",
                                        rescale=True,
                                    )
                                except Exception as e:
                                    logger.warning(
                                        f"Interpolation failed for {svar} at level {k+1}: {e}"
                                    )
                                    # Use mean of valid values as fallback
                                    if np.sum(~fpn) > 0:
                                        v0[n, fpn] = np.mean(v0[n, ~fpn])
                                    else:
                                        v0[n, fpn] = 0.0  # Default value
                except Exception as e:
                    logger.error(
                        f"Error extracting values for {svar} at level {k+1}: {e}"
                    )
                    # Create a fallback array with zeros
                    v0 = np.zeros((8, len(idx)))
                    logger.warning(f"Using zeros as fallback for {svar} at level {k+1}")

                # Trilinear interpolation
                v11 = v0[0] * (1 - ratx) + v0[1] * ratx
                v12 = v0[2] * (1 - ratx) + v0[3] * ratx
                v1 = v11 * (1 - raty) + v12 * raty

                v21 = v0[4] * (1 - ratx) + v0[5] * ratx
                v22 = v0[6] * (1 - ratx) + v0[7] * ratx
                v2 = v21 * (1 - raty) + v22 * raty

                vi = v1 * (1 - ratz) + v2 * ratz

                # Save interpolated values
                getattr(S, mvar).append(vi)

        # Convert lists to arrays
        for var in mvars:
            setattr(S, var, np.array(getattr(S, var)))

        # Create tracer arrays with NaN handling
        # Convert lists to arrays and check for NaNs
        for var in mvars:
            data_array = np.array(getattr(S, var))
            # Check for NaNs
            if np.any(np.isnan(data_array)):
                logger.warning(
                    f"Found NaN values in {var} data, replacing with interpolated values"
                )
                # Replace NaNs with interpolated values
                for i in range(data_array.shape[0]):
                    layer_data = data_array[i]
                    if np.any(np.isnan(layer_data)):
                        # Get indices of NaN values
                        nan_indices = np.where(np.isnan(layer_data))[0]
                        # Get indices of non-NaN values
                        valid_indices = np.where(~np.isnan(layer_data))[0]
                        if len(valid_indices) > 0:
                            # Use nearest valid values
                            for nan_idx in nan_indices:
                                # Find nearest valid index
                                nearest_idx = valid_indices[
                                    np.argmin(np.abs(valid_indices - nan_idx))
                                ]
                                # Replace NaN with nearest valid value
                                layer_data[nan_idx] = layer_data[nearest_idx]
                        else:
                            # If all values in this layer are NaN, use a default value
                            layer_data[nan_indices] = 0.0
            setattr(S, var, data_array)

        # Create tracer arrays
        tr_nd = np.r_[S.temp[None, ...], S.salt[None, ...]].T
        # Check for NaNs in tr_nd
        if np.any(np.isnan(tr_nd)):
            logger.warning(f"Found NaN values in tracer data, replacing with zeros")
            tr_nd = np.nan_to_num(tr_nd, nan=0.0)

        # Calculate element tracers from node tracers
        try:
            tr_el = tr_nd[gd.elnode[:, :3]].mean(axis=1)
        except Exception as e:
            logger.error(f"Error calculating element tracers: {e}")
            # Create a fallback array with zeros
            tr_el = np.zeros((ne, nvrt, 2))
            logger.warning("Using zeros as fallback for element tracers")

        # Create NetCDF structure
        nd = zdata()
        nd.file_format = "NETCDF4"
        nd.dimname = ["node", "elem", "side", "nVert", "ntracers", "one"]
        nd.dims = [np_grid, ne, ns, nvrt, 2, 1]

        # Define variables
        nd.vars = [
            "time",
            "iths",
            "ifile",
            "idry_e",
            "idry_s",
            "idry",
            "eta2",
            "we",
            "tr_el",
            "tr_nd",
            "tr_nd0",
            "su2",
            "sv2",
            "q2",
            "xl",
            "dfv",
            "dfh",
            "dfq1",
            "dfq2",
            "nsteps_from_cold",
            "cumsum_eta",
        ]

        # Initialize variables
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(0.0)
        nd.time = vi
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(0).astype("int")
        nd.iths = vi
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(1).astype("int")
        nd.ifile = vi
        vi = zdata()
        vi.dimname = ("one",)
        vi.val = np.array(0).astype("int")
        nd.nsteps_from_cold = vi

        vi = zdata()
        vi.dimname = ("elem",)
        vi.val = np.zeros(ne).astype("int32")
        nd.idry_e = vi
        vi = zdata()
        vi.dimname = ("side",)
        vi.val = np.zeros(ns).astype("int32")
        nd.idry_s = vi
        vi = zdata()
        vi.dimname = ("node",)
        vi.val = np.zeros(np_grid).astype("int32")
        nd.idry = vi
        vi = zdata()
        vi.dimname = ("node",)
        vi.val = np.zeros(np_grid)
        nd.eta2 = vi
        vi = zdata()
        vi.dimname = ("node",)
        vi.val = np.zeros(np_grid)
        nd.cumsum_eta = vi

        vi = zdata()
        vi.dimname = ("elem", "nVert")
        vi.val = np.zeros([ne, nvrt])
        nd.we = vi
        vi = zdata()
        vi.dimname = ("side", "nVert")
        vi.val = np.zeros([ns, nvrt])
        nd.su2 = vi
        vi = zdata()
        vi.dimname = ("side", "nVert")
        vi.val = np.zeros([ns, nvrt])
        nd.sv2 = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.q2 = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.xl = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfv = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfh = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfq1 = vi
        vi = zdata()
        vi.dimname = ("node", "nVert")
        vi.val = np.zeros([np_grid, nvrt])
        nd.dfq2 = vi

        vi = zdata()
        vi.dimname = ("elem", "nVert", "ntracers")
        vi.val = tr_el
        nd.tr_el = vi
        vi = zdata()
        vi.dimname = ("node", "nVert", "ntracers")
        vi.val = tr_nd
        nd.tr_nd = vi
        vi = zdata()
        vi.dimname = ("node", "nVert", "ntracers")
        vi.val = tr_nd
        nd.tr_nd0 = vi

        # Write NetCDF file
        WriteNC(str(output_path), nd)
        logger.info(f"Successfully created hotstart file: {output_path}")

        return str(output_path)
