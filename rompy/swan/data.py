import logging
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import field_validator, Field, model_validator

from rompy.core.data import DataGrid
from rompy.core.time import TimeRange

from rompy.swan.grid import SwanGrid
from rompy.swan.types import GridOptions


logger = logging.getLogger(__name__)

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
        logger.info(f"\tWriting {self.var.value} to {output_file}")
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
        return f"{inpgrid}\n{readgrid}\n"

    def __str__(self):
        return f"SWANDataGrid {self.var.name}"


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
    with open(output_file, "w") as stream:
        for time in dset[time_dim]:
            logger.debug(
                f"Appending Time {pd.to_datetime(time.values)} to {output_file}"
            )
            for data_var in variables:
                logger.debug(f"Appending Variable {data_var} to {output_file}")
                data = dset[data_var].sel(time=time).fillna(fill_value).values
                np.savetxt(fname=stream, X=data, fmt=fmt, delimiter="\t")

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
        dt = np.diff(ds[time].values).mean() / pd.to_timedelta(1, "h")

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

        inpgrid = f"INPGRID {var} {grid.inpgrid} NONSTATION {inptimes[0]} {dt} HR"
        readinp = f"READINP {var} {fac} '{Path(output_file).name}' 3 0 1 0 FREE"

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
        boundary_points = []
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
                    with open(f"{dest_path}/{j}.TPAR", "wt") as f:
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
