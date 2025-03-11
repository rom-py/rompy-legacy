import logging
import os
from pathlib import Path
from typing import Literal, Optional, Union
import scipy as sp

import numpy as np
import pandas as pd
import xarray as xr
from cloudpathlib import AnyPath
from pydantic import Field, model_validator

from rompy.core import DataGrid, RompyBaseModel
from rompy.core.boundary import BoundaryWaveStation, DataBoundary
from rompy.core.data import DataBlob
from rompy.core.time import TimeRange
from rompy.schism.grid import SCHISMGrid
# from pyschism.forcing.bctides import Bctides
from rompy.schism.pyschism.forcing.bctides import Bctides
# from rompy.schism.pyschism.forcing.hycom.hycom2schism import (
#     interp_to_points_2d, interp_to_points_3d, transform_ll_to_cpp)
from rompy.utils import total_seconds

from .namelists import Sflux_Inputs

logger = logging.getLogger(__name__)


class SfluxSource(DataGrid):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux"] = Field(
        default="sflux",
        description="Model type discriminator",
    )
    id: str = Field("sflux_source", description="id of the source")
    relative_weight: float = Field(
        1.0,
        description="relative weight of the source file if two files are provided",
    )
    max_window_hours: float = Field(
        120.0,
        description="maximum number of hours (offset from start time in each file) in each file of set 1",
    )
    fail_if_missing: bool = Field(
        True, description="Fail if the source file is missing"
    )
    id: str = Field(None, description="id of the source", choices=["air", "rad", "prc"])
    time_buffer: list[int] = Field(
        default=[0, 1],
        description="Number of source data timesteps to buffer the time range if `filter_time` is True",
    )
    _variable_names = []

    def __init__(self, **data):
        super().__init__(**data)
        self._set_variables()

    @property
    def outfile(self) -> str:
        # TODO - filenumber is. Hardcoded to 1 for now.
        return f'{self.id}.{str(1).rjust(4, "0")}.nc'

    def _set_variables(self) -> None:
        for variable in self._variable_names:
            if getattr(self, variable) is not None:
                self.variables.append(getattr(self, variable))

    @property
    def namelist(self) -> dict:
        # ret = self.model_dump()
        ret = {}
        for key, value in self.model_dump().items():
            if key in ["relative_weight", "max_window_hours", "fail_if_missing"]:
                ret.update({f"{self.id}_{key}": value})
        for varname in self._variable_names:
            var = getattr(self, varname)
            if var is not None:
                ret.update({varname: var})
            else:
                ret.update({varname: varname.replace("_name", "")})
        ret.update({f"{self.id}_file": self.id})
        return ret

    @property
    def ds(self):
        """Return the xarray dataset for this data source."""
        ds = self.source.open(
            variables=self.variables, filters=self.filter, coords=self.coords
        )
        # Define a dictionary for potential renaming
        rename_dict = {self.coords.y: "ny_grid", self.coords.x: "nx_grid"}

        # Construct a valid renaming dictionary
        valid_rename_dict = get_valid_rename_dict(ds, rename_dict)

        # Perform renaming if necessary
        if valid_rename_dict:
            ds = ds.rename_dims(valid_rename_dict)

        lon, lat = np.meshgrid(ds[self.coords.x], ds[self.coords.y])
        ds["lon"] = (("ny_grid", "nx_grid"), lon)
        ds["lat"] = (("ny_grid", "nx_grid"), lat)
        basedate = pd.to_datetime(ds.time.values[0])
        unit = f"days since {basedate.strftime('%Y-%m-%d %H:%M:%S')}"
        ds.time.attrs = {
            "long_name": "Time",
            "standard_name": "time",
            "base_date": np.int32(
                np.array(
                    [
                        basedate.year,
                        basedate.month,
                        basedate.day,
                        basedate.hour,
                        basedate.minute,
                        basedate.second,
                    ]
                )
            ),
            # "units": unit,
        }
        ds.time.encoding["units"] = unit
        ds.time.encoding["calendar"] = "proleptic_gregorian"
        # open bad dataset

        # SCHISM doesn't like scale_factor and add_offset attributes and requires Float64 values
        for var in ds.data_vars:
            # If the variable has scale_factor or add_offset attributes, remove them
            if "scale_factor" in ds[var].encoding:
                del ds[var].encoding["scale_factor"]
            if "add_offset" in ds[var].encoding:
                del ds[var].encoding["add_offset"]
            # set the data variable encoding to Float64
            ds[var].encoding["dtype"] = np.dtypes.Float64DType()

        return ds


class SfluxAir(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_air"] = Field(
        default="sflux_air",
        description="Model type discriminator",
    )
    uwind_name: str = Field(
        None,
        description="name of zonal wind variable in source",
    )
    vwind_name: Union[str, None] = Field(
        None,
        description="name of meridional wind variable in source",
    )
    prmsl_name: Union[str, None] = Field(
        None,
        description="name of mean sea level pressure variable in source",
    )
    stmp_name: Union[str, None] = Field(
        None,
        description="name of surface air temperature variable in source",
    )
    spfh_name: Union[str, None] = Field(
        None,
        description="name of specific humidity variable in source",
    )
    _variable_names = [
        "uwind_name",
        "vwind_name",
        "prmsl_name",
        "stmp_name",
        "spfh_name",
    ]

    @property
    def ds(self):
        """Return the xarray dataset for this data source."""
        ds = super().ds
        for variable in self._variable_names:
            data_var = getattr(self, variable)
            if data_var == None:
                proxy_var = variable.replace("_name", "")
                ds[proxy_var] = ds[self.uwind_name].copy()
                if variable == "spfh_name":
                    missing = 0.01
                else:
                    missing = -999
                ds[proxy_var][:, :, :] = missing
                ds.data_vars[proxy_var].attrs["long_name"] = proxy_var
        return ds


class SfluxRad(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_rad"] = Field(
        default="sflux_rad",
        description="Model type discriminator",
    )
    dlwrf_name: str = Field(
        None,
        description="name of downward long wave radiation variable in source",
    )
    dswrf_name: str = Field(
        None,
        description="name of downward short wave radiation variable in source",
    )
    _variable_names = ["dlwrf_name", "dswrf_name"]


class SfluxPrc(SfluxSource):
    """This is a single variable source for and sflux input"""

    data_type: Literal["sflux_prc"] = Field(
        default="sflux_rad",
        description="Model type discriminator",
    )
    prate_name: str = Field(
        None,
        description="name of precipitation rate variable in source",
    )
    _variable_names = ["prate_name"]


class SCHISMDataSflux(RompyBaseModel):
    data_type: Literal["sflux"] = Field(
        default="sflux",
        description="Model type discriminator",
    )
    air_1: Union[DataBlob, SfluxAir, None] = Field(
        None, description="sflux air source 1"
    )
    air_2: Union[DataBlob, SfluxAir, None] = Field(
        None, description="sflux air source 2"
    )
    rad_1: Union[DataBlob, SfluxRad, None] = Field(
        None, description="sflux rad source 1"
    )
    rad_2: Union[DataBlob, SfluxRad, None] = Field(
        None, description="sflux rad source 2"
    )
    prc_1: Union[DataBlob, SfluxPrc, None] = Field(
        None, description="sflux prc source 1"
    )
    prc_2: Union[DataBlob, SfluxPrc, None] = Field(
        None, description="sflux prc source 2"
    )

    def get(
        self,
        destdir: str | Path,
        grid: Optional[SCHISMGrid] = None,
        time: Optional[TimeRange] = None,
    ) -> Path:
        """Writes SCHISM sflux data from a dataset.

        Args:
            destdir (str | Path): The destination directory to write the sflux data.
            grid (Optional[SCHISMGrid], optional): The grid type. Defaults to None.
            time (Optional[TimeRange], optional): The time range. Defaults to None.

        Returns:
            Path: The path to the written sflux data.

        """
        ret = {}
        destdir = Path(destdir) / "sflux"
        destdir.mkdir(parents=True, exist_ok=True)
        namelistargs = {}
        for variable in ["air_1", "air_2", "rad_1", "rad_2", "prc_1", "prc_2"]:
            data = getattr(self, variable)
            if data is None:
                continue
            data.id = variable
            logger.info(f"Fetching {variable}")
            namelistargs.update(data.namelist)
            ret[variable] = data.get(destdir, grid, time)
        ret["nml"] = Sflux_Inputs(**namelistargs).write_nml(destdir)
        return ret

    @model_validator(mode="after")
    def check_weights(v):
        """Check that relative weights for each pair add to 1.

        Args:
            cls: The class.
            v: The variable.

        Raises:
            ValueError: If the relative weights for any variable do not add up to 1.0.

        """
        for variable in ["air", "rad", "prc"]:
            weight = 0
            active = False
            for i in [1, 2]:
                data = getattr(v, f"{variable}_{i}")
                if data is None:
                    continue
                if data.fail_if_missing:
                    continue
                weight += data.relative_weight
                active = True
            if active and weight != 1.0:
                raise ValueError(
                    f"Relative weights for {variable} do not add to 1.0: {weight}"
                )
            return v
        # SCHISM doesn't like scale_factor and add_offset attributes and requires Float64 values
        for var in ds.data_vars:
            # If the variable has scale_factor or add_offset attributes, remove them
            if "scale_factor" in ds[var].encoding:
                del ds[var].encoding["scale_factor"]
            if "add_offset" in ds[var].encoding:
                del ds[var].encoding["add_offset"]
            # set the data variable encoding to Float64
            ds[var].encoding["dtype"] = np.dtypes.Float64DType()


class SCHISMDataWave(BoundaryWaveStation):
    """This class is used to write wave spectral boundary data. Spectral data is extracted
    from the nearest points along the grid boundary"""

    data_type: Literal["wave"] = Field(
        default="wave",
        description="Model type discriminator",
    )
    sel_method: dict = Field(
        default="nearest",
        description="Keyword arguments for sel_method",
    )
    sel_method_kwargs: dict = Field(
        default={"unique": True},
        description="Keyword arguments for sel_method",
    )
    time_buffer: list[int] = Field(
        default=[0, 1],
        description="Number of source data timesteps to buffer the time range if `filter_time` is True",
    )

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write the selected boundary data to a netcdf file.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        logger.info(f"Fetching {self.id}")
        if self.crop_data and time is not None:
            self._filter_time(time)
        ds = self._sel_boundary(grid)
        outfile = Path(destdir) / f"{self.id}.nc"
        ds.spec.to_ww3(outfile)
        logger.info(f"\tSaved to {outfile}")
        return outfile

    @property
    def ds(self):
        """Return the filtered xarray dataset instance."""
        ds = super().ds
        for var in ds.data_vars:
            # If the variable has scale_factor or add_offset attributes, remove them
            if "scale_factor" in ds[var].encoding:
                del ds[var].encoding["scale_factor"]
            if "add_offset" in ds[var].encoding:
                del ds[var].encoding["add_offset"]
            # set the data variable encoding to Float64
            ds[var].encoding["dtype"] = np.dtypes.Float64DType()
        return ds

    def __str__(self):
        return f"SCHISMDataWave"


class SCHISMDataBoundary(DataBoundary):
    """This class is used to extract ocean boundary data from a griddd dataset at all open
    boundary nodes."""

    data_type: Literal["boundary"] = Field(
        default="boundary",
        description="Model type discriminator",
    )
    id: str = Field(
        "bnd",
        description="SCHISM th id of the source",
        choices=["elev2D", "uv3D", "TEM_3D", "SAL_3D", "bnd"],
    )
    variables: list[str] = Field(..., description="variable name in the dataset")
    sel_method: Literal["sel", "interp"] = Field(
        default="interp",
        description=(
            "Xarray method to use for selecting boundary points from the dataset"
        ),
    )
    interpolate_missing_coastal: bool = Field(
        True, description="interpolate_missing coastal data points"
    )
    time_buffer: list[int] = Field(
        default=[0, 1],
        description="Number of source data timesteps to buffer the time range if `filter_time` is True",
    )

    # @model_validator(mode="after")
    # def _set_variables(self) -> "SCHISMDataBoundary":
    #     self.variables = [self.variable]
    #     return self

    # @property
    # def ds(self):
    #     """Return the xarray dataset for this data source."""
    #     # I don't like this approach, as its no longer lazy
    #     ds = super().ds
    #     if self.extrapolate_to_coast:
    #         for var in ds.data_vars:
    #             ds[var] = (
    #                 ds[var].interpolate_na(
    #                     method="linear", fill_value="extrapolate", dim=self.coords.x
    #                 )
    #                 + ds[var].interpolate_na(
    #                     method="linear", fill_value="extrapolate", dim=self.coords.y
    #                 )
    #             ) / 2
    #     return ds

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write the selected boundary data to a netcdf file.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        # prepare xarray.Dataset and save forcing netCDF file
        outfile = Path(destdir) / f"{self.id}.th.nc"
        boundary_ds = self.boundary_ds(grid, time)
        boundary_ds.to_netcdf(outfile, "w", "NETCDF3_CLASSIC", unlimited_dims="time")
        logger.info(f"\tSaved to {outfile}")
        return outfile

    def boundary_ds(self, grid: SCHISMGrid, time: Optional[TimeRange]) -> xr.Dataset:
        """Generate SCHISM boundary dataset from source data.

        This function extracts and formats boundary data for SCHISM from a source dataset.
        For 3D models, it handles vertical interpolation to the SCHISM sigma levels.

        Parameters
        ----------
        grid : SCHISMGrid
            The SCHISM grid to extract boundary data for
        time : Optional[TimeRange]
            The time range to filter data to, if crop_data is True

        Returns
        -------
        xr.Dataset
            Dataset formatted for SCHISM boundary input
        """
        logger.info(f"Fetching {self.id}")
        if self.crop_data and time is not None:
            self._filter_time(time)

        # Extract boundary data from source
        ds = self._sel_boundary(grid)

        # Calculate time step
        if len(ds.time) > 1:
            dt = total_seconds((ds.time[1] - ds.time[0]).values)
        else:
            dt = 3600

        # Get the variable data
        data = ds[self.variables[0]].values

        # Determine if we're working with 3D data
        is_3d_data = grid.is_3d and self.coords.z is not None

        # Handle different data dimensions based on 2D or 3D
        if is_3d_data:
            # Try to determine the dimension order
            if hasattr(ds[self.variables[0]], "dims"):
                # Get dimension names
                dims = list(ds[self.variables[0]].dims)

                # Find indices of time, z, and x dimensions
                time_dim_idx = dims.index(ds.time.dims[0])
                z_dim_idx = (
                    dims.index(ds[self.coords.z].dims[0]) if self.coords.z in ds else 1
                )
                x_dim_idx = (
                    dims.index(ds[self.coords.x].dims[0]) if self.coords.x in ds else 2
                )

                logger.debug(
                    f"Dimension order: time={time_dim_idx}, z={z_dim_idx}, x={x_dim_idx}"
                )

                # Reshape data to expected format if needed (time, x, z)
                if not (time_dim_idx == 0 and x_dim_idx == 1 and z_dim_idx == 2):
                    trans_dims = list(range(data.ndim))
                    trans_dims[time_dim_idx] = 0
                    trans_dims[x_dim_idx] = 1
                    trans_dims[z_dim_idx] = 2

                    data = np.transpose(data, trans_dims)
                    logger.debug(f"Transposed data shape: {data.shape}")

            # Add the component dimension for SCHISM
            time_series = np.expand_dims(data, axis=3)

            # calculate zcor for 3D
            sigma = grid.pyschism_vgrid.sigma

            # Get the boundary node indices
            # Get the boundary points from the grid
            boundary_nodes = grid.pyschism_hgrid.boundaries.open.indexes
            if boundary_nodes.empty:
                raise ValueError("No open boundary nodes found in the grid")
            
            # Extract all boundary node indices
            # Since this is a pandas Series where each item is a list of indices
            boundary_indices = []
            for node_list in boundary_nodes:
                boundary_indices.extend(node_list)
            
            # Get bathymetry for boundary nodes only
            depth = grid.pyschism_hgrid.values
            boundary_depths = depth[boundary_indices]
            
            # Get the sigma values and fixed z levels from the SCHISM grid
            sigma_levels = sigma.copy()  # Copy to avoid modifying the original
            num_sigma_levels = len(sigma_levels)
            
            # Get the fixed z levels
            z_levels = grid.pyschism_vgrid.ztot
            
            # For each boundary point, determine the total number of vertical levels
            # and create appropriate zcor arrays
            all_zcors = []
            all_nvrt = []
            
            for i, (node_idx, depth) in enumerate(zip(boundary_indices, boundary_depths)):
                # Check if we're in deep water (depth > first z level)
                if z_levels.size > 0 and depth > z_levels[0]:
                    # In deep water, find applicable z levels (between first z level and actual depth)
                    first_z_level = z_levels[0]
                    z_mask = (z_levels > first_z_level) & (z_levels < depth)
                    applicable_z = z_levels[z_mask] if np.any(z_mask) else []
                    
                    # Total levels = sigma levels + applicable z levels
                    total_levels = num_sigma_levels + len(applicable_z)
                    
                    # Create zcor for this boundary point
                    node_zcor = np.zeros(total_levels)
                    
                    # First, calculate sigma levels using the first z level as the "floor"
                    for j in range(num_sigma_levels):
                        node_zcor[j] = first_z_level * sigma[j]
                    
                    # Then, add the fixed z levels below the sigma levels
                    for j, z_val in enumerate(applicable_z):
                        node_zcor[num_sigma_levels + j] = z_val
                    
                else:
                    # In shallow water, just use sigma levels scaled to the actual depth
                    total_levels = num_sigma_levels
                    
                    # Create zcor for this boundary point
                    node_zcor = np.zeros(total_levels)
                    
                    for j in range(total_levels):
                        node_zcor[j] = depth * sigma[j]
                
                # Store this boundary point's zcor and number of levels
                all_zcors.append(node_zcor)
                all_nvrt.append(total_levels)
            
            # Now we have a list of zcor arrays with potentially different lengths
            # Find the maximum number of levels across all boundary points
            max_nvrt = max(all_nvrt) if all_nvrt else num_sigma_levels
            
            # Create a uniform zcor array with the maximum number of levels
            zcor = np.zeros((len(boundary_indices), max_nvrt))
            
            # Fill in the values, leaving zeros for levels beyond a particular boundary point's total
            for i, (node_zcor, nvrt_i) in enumerate(zip(all_zcors, all_nvrt)):
                zcor[i, :nvrt_i] = node_zcor
            
            # Get source z-levels and prepare for interpolation
            z_src = ds[self.coords.z].values
            data_shape = data.shape
            
            # Initialize interpolated data array with the maximum number of vertical levels
            interpolated_data = np.zeros((data_shape[0], data_shape[1], max_nvrt))
            
            # For each time step and boundary point
            for t in range(data_shape[0]):  # time
                for n in range(data_shape[1]):  # boundary points
                    # Get z-coordinates for this point
                    z_dest = zcor[n, :]
                    nvrt_n = all_nvrt[n]  # Get the number of vertical levels for this point
                    
                    # Extract vertical profile
                    profile = data[t, n, :]
                    
                    # Create interpolator for this profile
                    interp = sp.interpolate.interp1d(
                        z_src, profile, 
                        kind='linear',
                        bounds_error=False,
                        fill_value='extrapolate'
                    )
                    
                    # Interpolate to SCHISM levels for this boundary point
                    # Only interpolate up to the actual number of levels for this point
                    interpolated_data[t, n, :nvrt_n] = interp(z_dest[:nvrt_n])
            
            # Replace data with interpolated values
            data = interpolated_data
            time_series = np.expand_dims(data, axis=3)
            
            # Handle coastal interpolation
            if self.interpolate_missing_coastal:
                # Apply fill_tails to each vertical slice at each time step
                for i in range(data.shape[0]):  # time
                    for j in range(data.shape[2]):  # depth level
                        data[i, :, j] = fill_tails(data[i, :, j])
            
            # Store the variable vertical levels in the output dataset
            # Create a 2D array where each row contains the vertical levels for a boundary node
            # For nodes with fewer levels, pad with NaN
            vert_levels = np.full((len(boundary_indices), max_nvrt), np.nan)
            for i, (node_zcor, nvrt_i) in enumerate(zip(all_zcors, all_nvrt)):
                vert_levels[i, :nvrt_i] = node_zcor
            
            # Create output dataset
            schism_ds = xr.Dataset(
                coords={
                    "time": ds.time,
                    "nOpenBndNodes": np.arange(data.shape[1]),
                    "nLevels": np.arange(max_nvrt),
                    "nComponents": np.array([1]),
                    "one": np.array([1]),
                },
                data_vars={
                    "time_step": (("one"), np.array([dt])),
                    "time_series": (
                        ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                        time_series,
                    ),
                    "vertical_levels": (
                        ("nOpenBndNodes", "nLevels"),
                        vert_levels,
                    ),
                    "num_levels": (
                        ("nOpenBndNodes"),
                        np.array(all_nvrt),
                    ),
                },
            )
        else:
            # 2D case - simpler handling
            if self.interpolate_missing_coastal:
                for i in range(data.shape[0]):  # time
                    data[i, :] = fill_tails(data[i, :])

            # Add level and component dimensions for SCHISM
            time_series = np.expand_dims(data, axis=(2, 3))

            # Create output dataset
            schism_ds = xr.Dataset(
                coords={
                    "time": ds.time,
                    "nOpenBndNodes": np.arange(data.shape[1]),
                    "nLevels": np.array([0]),  # Single level for 2D
                    "nComponents": np.array([1]),
                    "one": np.array([1]),
                },
                data_vars={
                    "time_step": (("one"), np.array([dt])),
                    "time_series": (
                        ("time", "nOpenBndNodes", "nLevels", "nComponents"),
                        time_series,
                    ),
                },
            )

        # Set attributes and encoding
        schism_ds.time_step.assign_attrs({"long_name": "time_step"})
        basedate = pd.to_datetime(ds.time.values[0])
        unit = f"days since {basedate.strftime('%Y-%m-%d %H:%M:%S')}"
        schism_ds.time.attrs = {
            "long_name": "Time",
            "standard_name": "time",
            "base_date": np.int32(
                np.array(
                    [
                        basedate.year,
                        basedate.month,
                        basedate.day,
                        basedate.hour,
                        basedate.minute,
                        basedate.second,
                    ]
                )
            ),
        }
        schism_ds.time.encoding["units"] = unit
        schism_ds.time.encoding["calendar"] = "proleptic_gregorian"

        # Handle missing values more robustly
        if schism_ds.time_series.isnull().any():
            logger.warning(
                "Some values are null. Attempting to interpolate missing values..."
            )

            # Try interpolating along different dimensions
            for dim in ["nOpenBndNodes", "time", "nLevels"]:
                if dim in schism_ds.dims and len(schism_ds[dim]) > 1:
                    schism_ds["time_series"] = schism_ds.time_series.interpolate_na(
                        dim=dim
                    )
                    if not schism_ds.time_series.isnull().any():
                        logger.info(
                            f"Successfully interpolated all missing values along {dim} dimension"
                        )
                        break

            # If still have NaNs, use more aggressive filling methods
            if schism_ds.time_series.isnull().any():
                logger.warning("Using constant value for remaining missing data points")
                # Find a reasonable fill value (median of non-NaN values)
                valid_values = schism_ds.time_series.values[
                    ~np.isnan(schism_ds.time_series.values)
                ]
                fill_value = np.median(valid_values) if len(valid_values) > 0 else 0.0
                schism_ds["time_series"] = schism_ds.time_series.fillna(fill_value)

        # Clean up encoding
        for var in schism_ds.data_vars:
            if "scale_factor" in schism_ds[var].encoding:
                del schism_ds[var].encoding["scale_factor"]
            if "add_offset" in schism_ds[var].encoding:
                del schism_ds[var].encoding["add_offset"]
            schism_ds[var].encoding["dtype"] = np.dtypes.Float64DType()

        return schism_ds


def fill_tails(arr):
    """If the tails of  1d array are nan, fill with the last non nan value."""
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = arr[idx]
    # repoat the same from the other end
    reverse = out[::-1]
    mask = np.isnan(reverse)
    idx = np.where(~mask, np.arange(mask.shape[0]), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    out = reverse[idx]
    return out[::-1]


class SCHISMDataOcean(RompyBaseModel):
    """This class is used define all ocean boundary forcing"""

    data_type: Literal["ocean"] = Field(
        default="ocean",
        description="Model type discriminator",
    )
    elev2D: Optional[Union[DataBlob, SCHISMDataBoundary]] = Field(
        None,
        description="elev2D",
    )
    uv3D: Optional[Union[DataBlob, SCHISMDataBoundary]] = Field(
        None,
        description="uv3D",
    )
    TEM_3D: Optional[Union[DataBlob, SCHISMDataBoundary]] = Field(
        None,
        description="TEM_3D",
    )
    SAL_3D: Optional[Union[DataBlob, SCHISMDataBoundary]] = Field(
        None,
        description="SAL_3D",
    )

    @model_validator(mode="after")
    def not_yet_implemented(cls, v):
        for variable in ["uv3D"]:
            if getattr(v, variable) is not None:
                raise NotImplementedError(f"Variable {variable} is not yet implemented")
        return v

    @model_validator(mode="after")
    def set_id(cls, v):
        for variable in ["elev2D", "uv3D", "TEM_3D", "SAL_3D"]:
            if getattr(v, variable) is not None:
                getattr(v, variable).id = variable
        return v

    def get(
        self,
        destdir: str | Path,
        grid: SCHISMGrid,
        time: Optional[TimeRange] = None,
    ) -> str:
        """Write all inputs to netcdf files.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid,
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        for variable in ["elev2D", "uv3D", "TEM_3D", "SAL_3D"]:
            data = getattr(self, variable)
            if data is None:
                continue
            data.get(destdir, grid, time)

    def __str__(self):
        return f"SCHISMDataOcean"


class TidalDataset(RompyBaseModel):
    """This class is used to define the tidal dataset"""

    data_type: Literal["tidal_dataset"] = Field(
        default="tidal_dataset",
        description="Model type discriminator",
    )
    elevations: AnyPath = Field(..., description="Path to elevations file")
    velocities: AnyPath = Field(..., description="Path to currents file")

    def get(self, destdir: str | Path) -> str:
        """Write all inputs to netcdf files.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """
        # TODO need to put some smarts in here for remote files
        os.environ["TPXO_ELEVATION"] = self.elevations.as_posix()
        os.environ["TPXO_VELOCITY"] = self.velocities.as_posix()


class SCHISMDataTides(RompyBaseModel):
    """This class is used to define the tidal forcing for SCHISM."""

    data_type: Literal["tides"] = Field(
        default="tide",
        description="Model type discriminator",
    )
    tidal_data: TidalDataset = Field(..., description="tidal dataset")
    cutoff_depth: float = Field(
        50.0,
        description="cutoff depth for tides",
    )
    flags: Optional[list] = Field([[5, 3, 0, 0]], description="nested list of bctypes")
    constituents: Union[str, list] = Field("major", description="constituents")
    database: str = Field("tpxo", description="database", choices=["tpxo", "fes2014"])
    add_earth_tidal: bool = Field(True, description="add_earth_tidal")
    ethconst: Optional[list] = Field(
        [], description="constant elevation value for each open boundary"
    )
    vthconst: Optional[list] = Field(
        [], description="constant discharge value for each open boundary"
    )
    tthconst: Optional[list] = Field(
        [], description="constant temperature value for each open boundary"
    )
    sthconst: Optional[list] = Field(
        [], description="constant salinity value for each open boundary"
    )
    tobc: Optional[list[float]] = Field(
        [1], description="nuding factor of temperature for each open boundary"
    )
    sobc: Optional[list[float]] = Field(
        [1], description="nuding factor of salinity for each open boundary"
    )
    relax: Optional[list[float]] = Field(
        [], description="relaxation constants for inflow and outflow"
    )

    def get(self, destdir: str | Path, grid: SCHISMGrid, time: TimeRange) -> str:
        """Write all inputs to netcdf files.
        Parameters
        ----------
        destdir : str | Path
            Destination directory for the netcdf file.
        grid : SCHISMGrid
            Grid instance to use for selecting the boundary points.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.crop_data` is True.

        Returns
        -------
        outfile : Path
            Path to the netcdf file.

        """

        self.tidal_data.get(destdir)
        logger.info(f"Generating tides")
        bctides = Bctides(
            hgrid=grid.pyschism_hgrid,
            flags=self.flags,
            constituents=self.constituents,
            database=self.database,
            add_earth_tidal=self.add_earth_tidal,
            cutoff_depth=self.cutoff_depth,
            ethconst=self.ethconst,
            vthconst=self.vthconst,
            tthconst=self.tthconst,
            sthconst=self.sthconst,
            tobc=self.tobc,
            sobc=self.sobc,
            relax=self.relax,
        )
        bctides.write(
            destdir,  # +'/bctides.in',
            start_date=time.start,
            rnday=time.end - time.start,
            overwrite=True,
        )


class SCHISMData(RompyBaseModel):
    """
    This class is used to gather all required input forcing for SCHISM
    """

    data_type: Literal["schism"] = Field(
        default="schism",
        description="Model type discriminator",
    )
    atmos: Optional[SCHISMDataSflux] = Field(None, description="atmospheric data")
    ocean: Optional[SCHISMDataOcean] = Field(None, description="ocean data")
    wave: Optional[Union[DataBlob, SCHISMDataWave]] = Field(
        None, description="wave data"
    )
    tides: Optional[Union[DataBlob, SCHISMDataTides]] = Field(
        None, description="tidal data"
    )

    # @model_validator(mode="after")
    # def check_bctides_flags(cls, v):
    #     # TODO Add check fro bc flags in teh event of 3d inputs
    #     # SHould possibly move this these flags out of SCHISMDataTides class as they cover more than
    #     # just tides
    #     return cls

    def get(
        self,
        destdir: str | Path,
        grid: Optional[SCHISMGrid] = None,
        time: Optional[TimeRange] = None,
    ) -> None:
        ret = {}
        # if time:
        #     # Bump enddate by 1 hour to make sure we get the last time step
        #     time = TimeRange(
        #         start=time.start,
        #         end=time.end + timedelta(hours=1),
        #         interval=time.interval,
        #         include_end=time.include_end,
        #     )
        for datatype in ["atmos", "ocean", "wave", "tides"]:
            data = getattr(self, datatype)
            if data is None:
                continue
            if type(data) is DataBlob:
                output = data.get(destdir)
            else:
                output = data.get(destdir, grid, time)
            ret.update({datatype: output})
            # ret[
            #     "wave"
            # ] = "dummy"  # Just to make cookiecutter happy if excluding wave forcing
        return ret


def get_valid_rename_dict(ds, rename_dict):
    """Construct a valid renaming dictionary that only includes names which need renaming."""
    valid_rename_dict = {}
    for old_name, new_name in rename_dict.items():
        if old_name in ds.dims and new_name not in ds.dims:
            valid_rename_dict[old_name] = new_name
    return valid_rename_dict
