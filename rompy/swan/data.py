"""
SWAN Data Module

This module provides data handling functionality for the SWAN model within the ROMPY framework.
"""

import os
import time as time_module
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field, model_validator

from rompy.core.data import DataGrid
from rompy.logging import get_logger
from rompy.core.time import TimeRange
from rompy.formatting import get_formatted_box, log_box
from rompy.swan.grid import SwanGrid
from rompy.swan.types import GridOptions

logger = get_logger(__name__)

FILL_VALUE = -99.0


class SwanDataGrid(DataGrid):
    """This class is used to write SWAN data from a dataset."""

    z1: Optional[str] = Field(
        default=None,
        description=(
            "Name of the data variable in dataset representing either a scaler "
            "parameter or the u-componet of a vector field"
        ),
    )
    z2: Optional[str] = Field(
        default=None,
        description=(
            "Name of the data variable in dataset representing "
            "the v-componet of a vector field"
        ),
    )
    var: GridOptions = Field(description="SWAN input grid name")
    fac: float = Field(
        description=(
            "SWAN multiplies all values that are read from file by `fac`. For "
            "instance if the values are given in unit decimeter, one should make "
            "`fac=0.1` to obtain values in m. To change sign use a negative `fac`"
        ),
        default=1.0,
    )

    @model_validator(mode="after")
    def ensure_z1_in_data_vars(self) -> "SwanDataGrid":
        data_vars = self.variables
        for z in [self.z1, self.z2]:
            if z and z not in data_vars:
                logger.debug(f"Adding {z} to data_vars")
                data_vars.append(z)
        self.variables = data_vars
        return self

    def get(
        self,
        destdir: str | Path,
        grid: Optional[SwanGrid] = None,
        time: Optional[TimeRange] = None,
    ) -> Path:
        """Write the data source to a new location.

        Parameters
        ----------
        destdir : str | Path
            The destination directory to write the netcdf data to.
        grid: SwanGrid, optional
            The grid to filter the data to, only used if `self.filter_grid` is True.
        time: TimeRange, optional
            The times to filter the data to, only used if `self.filter_time` is True.

        Returns
        -------
        cmd: str
            The command line string with the INPGRID/READINP commands ready to be
            written to the SWAN input file.

        Note
        ----
        The data are assumed to not have been rotated. We cannot use the grid.rot attr
        as this is the rotation from the model grid object which is not necessarily the
        same as the rotation of the data.

        """
        if self.crop_data:
            if grid is not None:
                self._filter_grid(grid)
            if time is not None:
                self._filter_time(time)

        output_file = os.path.join(destdir, f"{self.var.value}.grd")

        # Create a formatted box for logging
        log_box(
            title=f"WRITING {self.var.value.upper()} GRID DATA",
            logger=logger,
            add_empty_line=False,
        )

        # Log output file and dataset information using bullet points
        items = [f"Output file: {output_file}"]

        # Add variable information if available
        if self.z1:
            shape_info = f"{self.ds[self.z1].shape}"
            items.append(f"Variable: {self.z1} with shape {shape_info}")
        if self.z2:
            shape_info = f"{self.ds[self.z2].shape}"
            items.append(f"Variable: {self.z2} with shape {shape_info}")

        # Add scaling factor
        items.append(f"Scaling factor: {self.fac}")

        # Log all items as a bulleted list
        logger.bullet_list(items, indent=2)

        start_time = time_module.time()
        if self.var.value == "bottom":
            inpgrid, readgrid = self.ds.swan.to_bottom_grid(
                output_file,
                fmt="%4.2f",
                x=self.coords.x,
                y=self.coords.y,
                z=self.z1,
                fac=self.fac,
                rot=0.0,
                vmin=float("-inf"),
            )
        else:
            inpgrid, readgrid = self.ds.swan.to_inpgrid(
                output_file=output_file,
                x=self.coords.x,
                y=self.coords.y,
                z1=self.z1,
                z2=self.z2,
                fac=self.fac,
                rot=0.0,
                var=self.var.name,
            )

        # Log completion and processing time
        elapsed_time = time_module.time() - start_time
        file_size = Path(output_file).stat().st_size / (1024 * 1024)  # Size in MB

        # Use the centralized functions from rompy package

        # Log completion information as a bulleted list
        logger.bullet_list(
            [
                f"Completed in {elapsed_time:.2f} seconds",
                f"File size: {file_size:.2f} MB",
            ],
            indent=2,
        )

        return f"{inpgrid}\n{readgrid}\n"

    def __str__(self):
        return f"SWANDataGrid {self.var.name}"

    def _format_value(self, obj):
        """Format SwanDataGrid values using the new formatting framework.

        This method provides special formatting for SwanDataGrid objects.

        Args:
            obj: The object to format

        Returns:
            A formatted string or None to use default formatting
        """
        # Only format SwanDataGrid objects
        if not isinstance(obj, SwanDataGrid):
            return None

        # Use the new formatting framework
        from rompy.formatting import format_value

        return format_value(obj)
        lines.append(f"  {bullet} Variable:   {obj.var.name}")

        # Add source information if available
        if hasattr(obj, "source") and obj.source:
            source_type = getattr(obj.source, "model_type", "unknown")
            lines.append(f"  {bullet} Source:     {source_type}")

            # Add dataset information if available
            if hasattr(obj.source, "dataset_id"):
                lines.append(f"  {bullet} Dataset ID: {obj.source.dataset_id}")

        # Add coordinate information if available
        if hasattr(obj, "coords") and obj.coords:
            coords = [f"{k}={v}" for k, v in obj.coords.items()]
            coords_str = ", ".join(coords)
            lines.append(f"  {bullet} Coordinates: {coords_str}")

        # Add scaling factor if available
        if hasattr(obj, "fac"):
            lines.append(f"  {bullet} Scale factor: {obj.fac}")

        # Add z variables information if available
        if hasattr(obj, "z1") and obj.z1:
            lines.append(f"  {bullet} Z1 variable: {obj.z1}")
        if hasattr(obj, "z2") and obj.z2:
            lines.append(f"  {bullet} Z2 variable: {obj.z2}")

        # Close with footer
        lines.append(footer)

        return "\n".join(lines)


def dset_to_swan(
    dset: xr.Dataset,
    output_file: str,
    variables: list,
    fmt: str = "%4.2f",
    fill_value: float = FILL_VALUE,
    time_dim="time",
):
    """Convert xarray Dataset into SWAN ASCII file.

    Parameters
    ----------
    dset: xr.Dataset
        Dataset to write in SWAN ASCII format.
    output_file: str
        Local file name for the ascii output file.
    variables: list
        Variables to write to ascii.
    fmt: str
        String float formatter.
    fill_value: float
        Fill value.
    time_dim: str
        Name of the time dimension if available in the dataset.

    """
    # Input checking
    for data_var in variables:
        if data_var not in dset.data_vars:
            raise ValueError(f"Variable {data_var} not in {dset}")
        if dset[data_var].ndim not in (2, 3):
            raise NotImplementedError(
                "Only 2D and 3D datasets are supported but "
                f"dset.{data_var} has {dset[data_var].ndim} dims"
            )

    # Ensure time is a dimension so iteration will work
    if time_dim not in dset.dims:
        dset = dset.expand_dims(time_dim, 0)

    # Write to ascii
    logger.debug(f"Writing SWAN ASCII file: {output_file}")

    # Use formatting utilities imported at the top of the file

    # Import formatting utilities at function level to avoid scoping issues
    # Create a formatted box for logging
    log_box(title="WRITING SWAN ASCII DATA", logger=logger)

    start_time = time_module.time()
    file_size = 0
    total_times = len(dset[time_dim])

    with open(output_file, "w") as stream:
        for i, t in enumerate(dset[time_dim]):
            time_str = pd.to_datetime(t.values)
            if (
                i % max(1, total_times // 10) == 0 or i == total_times - 1
            ):  # Log progress at 10% intervals
                logger.debug(
                    f"Writing progress: {i+1}/{total_times} times ({(i+1)/total_times*100:.1f}%) - Time: {time_str}"
                )
            else:
                logger.debug(f"Appending Time {time_str} to {output_file}")

            for data_var in variables:
                logger.debug(f"Appending Variable {data_var} to {output_file}")
                data = dset[data_var].sel(time=t).fillna(fill_value).values
                np.savetxt(fname=stream, X=data, fmt=fmt, delimiter="\t")

    elapsed_time = time_module.time() - start_time
    file_size = Path(output_file).stat().st_size / (1024 * 1024)  # Size in MB

    # Format the completion message
    elapsed_str = f"{elapsed_time:.2f}"
    size_str = f"{file_size:.2f}"
    # Get a formatted completion box
    completion_msg = f"COMPLETED: {elapsed_str} seconds, File size: {size_str} MB"
    completion_box = get_formatted_box(completion_msg)
    for line in completion_box.split("\n"):
        logger.debug(line)

    logger.debug(f"SWAN ASCII file written successfully to {output_file}")

    return output_file


@xr.register_dataset_accessor("swan")
class Swan_accessor(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def grid(
        self,
        x: str = "lon",
        y: str = "lat",
        rot: float = 0.0,
        exc: float = FILL_VALUE,
    ):
        """SWAN Grid object for this dataset.

        Parameters
        ----------
        x: str
            Name of the x coordinate variable.
        y: str
            Name of the y coordinate variable.
        rot: float
            Rotation angle in degrees, required if the grid has been previously rotated.
        exc: float
            Exception value for missing data.

        Returns
        -------
        grid: SwanGrid
            SwanGrid object representing this dataset.

        """
        return SwanGrid(
            grid_type="REG",
            x0=float(self._obj[x].min()),
            y0=float(self._obj[y].min()),
            dx=float(np.diff(self._obj[x]).mean()),
            dy=float(np.diff(self._obj[y]).mean()),
            nx=len(self._obj[x]),
            ny=len(self._obj[y]),
            rot=rot,
            exc=exc,
        )

    def to_bottom_grid(
        self,
        output_file,
        fmt="%4.2f",
        x="lon",
        y="lat",
        z="depth",
        fac=1.0,
        rot=0.0,
        vmin=float("-inf"),
        fill_value=FILL_VALUE,
    ):
        """Write SWAN inpgrid BOTTOM file.

        Parameters
        ----------
        output_file: str
            Local file name for the ascii output file.
        fmt: str
            String float formatter.
        x: str
            Name of the x-axis in the dataset.
        y: str
            Name of the y-axis in the dataset.
        z: str
            Name of the bottom variable in the dataset.
        rot: float
            Rotation angle, required if the grid has been previously rotated.
        vmin: float
            Minimum value below which depths are masked.
        fill_value: float
            Fill value.
        fac: float
            Multiplying factor in case data are not in m or should be reversed.

        Returns
        -------
        inpgrid: str
            SWAN INPgrid command instruction.
        readinp: str
            SWAN READinp command instruction.

        TODO: Merge this method with `to_inpgrid`.

        """
        dset_to_swan(
            dset=self._obj.where(self._obj > vmin).transpose(..., y, x),
            output_file=output_file,
            fmt=fmt,
            variables=[z],
            fill_value=fill_value,
        )
        grid = self.grid(x=x, y=y, rot=rot)
        inpgrid = f"INPGRID BOTTOM {grid.inpgrid}"
        readinp = f"READINP BOTTOM {fac} '{Path(output_file).name}' 3 FREE"
        return inpgrid, readinp

    def to_inpgrid(
        self,
        output_file: str,
        var: str = "WIND",
        fmt: str = "%.2f",
        x: str = "lon",
        y: str = "lat",
        z1: str = "u10",
        z2: str | None = None,
        fac: float = 1.0,
        rot: float = 0.0,
        time: str = "time",
    ):
        """This function writes to a SWAN inpgrid format file (i.e. WIND)

        Parameters
        ----------
        output_file: str
            Local file name for the ascii output file.
        var: str
            Type of swan input, used to define the INPGRID/READGRID strings.
        fmt: str
            String float formatter.
        x: str
            Name of the x-axis in the dataset.
        y: str
            Name of the y-axis in the dataset.
        z1: str
            Name of the first variable in the dataset.
        z2: str, optional
            Name of the second variable in the dataset.
        fac: float
            Multiplying factor in case data are not in m or should be reversed.
        rot: float
            Rotation angle, required if the grid has been previously rotated.
        time: str
            Name of the time variable in the dataset

        Returns
        -------
        inpgrid: str
            SWAN INPgrid command instruction.
        readinp: str
            SWAN READinp command instruction.

        """
        ds = self._obj

        # ds = ds.transpose((time,) + ds[x].dims)
        # Calculate time difference in hours
        time_diffs = np.diff(ds[time].values)
        dt = time_diffs.mean() / pd.to_timedelta(1, "h")
        dt_str = f"{dt:.2f}"  # Format as string to avoid formatting issues

        inptimes = []
        with open(output_file, "wt") as f:
            # iterate through time
            for ti, windtime in enumerate(ds[time].values):
                time_str = pd.to_datetime(windtime).strftime("%Y%m%d.%H%M%S")
                logger.debug(time_str)

                # write SWAN time header to file:
                f.write(f"{time_str}\n")

                # Write first component to file
                z1t = np.squeeze(ds[z1].isel(dict(time=ti)).values)
                np.savetxt(f, z1t, fmt=fmt)

                if z2 is not None:
                    z2t = np.squeeze(ds[z2].isel(dict(time=ti)).values)
                    np.savetxt(f, z2t, fmt=fmt)

                inptimes.append(time_str)

        if len(inptimes) < 1:
            os.remove(output_file)
            raise ValueError(
                f"***Error! No times written to {output_file}\n. Check the input data!"
            )

        # Create grid object from this dataset
        grid = self.grid(x=x, y=y, rot=rot)

        inpgrid = f"INPGRID {var} {grid.inpgrid} NONSTATION {inptimes[0]} {dt_str} HR"
        readinp = f"READINP {var} {fac} '{Path(output_file).name}' 3 0 1 0 FREE"

        # Log detailed information about the generated grid
        logger.debug(f"Created {var} grid with:")
        logger.debug(f"  → Grid size: {grid.nx}x{grid.ny} points")
        logger.debug(f"  → Resolution: dx={grid.dx}, dy={grid.dy}")
        logger.debug(f"  → Time points: {len(inptimes)}")
        logger.debug(f"  → Time interval: {dt_str} HR")

        return inpgrid, readinp

    def to_tpar_boundary(
        self,
        dest_path,
        boundary,
        interval,
        x_var="lon",
        y_var="lat",
        hs_var="sig_wav_ht",
        per_var="pk_wav_per",
        dir_var="pk_wav_dir",
        dir_spread=20.0,
    ):
        """This function writes parametric boundary forcing to a set of
        TPAR files at a given distance based on gridded wave output. It returns the string to be included in the Swan INPUT file.

        At present simple nearest neighbour point lookup is used.

        Args:
        TBD
        """
        from shapely.ops import substring

        bound_string = "BOUNDSPEC SEGM XY "
        point_string = "&\n {xp:0.8f} {yp:0.8f} "
        file_string = "&\n {len:0.8f} '{fname}' 1 "

        for xp, yp in boundary.exterior.coords:
            bound_string += point_string.format(xp=xp, yp=yp)

        bound_string += "&\n VAR FILE "

        n_pts = int((boundary.length) / interval)
        splits = np.linspace(0, 1.0, n_pts)
        j = 0
        for i in range(len(splits) - 1):
            segment = substring(
                boundary.exterior, splits[i], splits[i + 1], normalized=True
            )
            xp = segment.coords[1][0]
            yp = segment.coords[1][1]
            logger.debug(f"Extracting point: {xp},{yp}")
            ds_point = self._obj.sel(
                indexers={x_var: xp, y_var: yp}, method="nearest", tolerance=interval
            )
            if len(ds_point.time) == len(self._obj.time):
                if not np.any(np.isnan(ds_point[hs_var])):
                    output_tpar = f"{dest_path}/{j}.TPAR"
                    logger.debug(f"Writing boundary point {j} to {output_tpar}")
                    logger.debug(f"  → Location: ({xp:.5f}, {yp:.5f})")
                    logger.debug(f"  → Time points: {len(ds_point.time)}")

                    with open(output_tpar, "wt") as f:
                        f.write("TPAR\n")
                        for t in range(len(ds_point.time)):
                            ds_row = ds_point.isel(time=t)
                            lf = "{tt} {hs:0.2f} {per:0.2f} {dirn:0.1f} {spr:0.2f}\n"
                            f.write(
                                lf.format(
                                    tt=str(
                                        ds_row["time"]
                                        .dt.strftime("%Y%m%d.%H%M%S")
                                        .values
                                    ),
                                    hs=float(ds_row[hs_var]),
                                    per=float(ds_row[per_var]),
                                    dirn=float(ds_row[dir_var]),
                                    spr=dir_spread,
                                )
                            )
                    bound_string += file_string.format(
                        len=splits[i + 1] * boundary.length, fname=f"{j}.TPAR"
                    )
                    j += 1

        return bound_string
