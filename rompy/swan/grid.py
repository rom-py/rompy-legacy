"""
SWAN Grid Module

This module provides grid-related functionality for the SWAN model within the ROMPY framework.
"""

# Standard library imports
from typing import Literal, Optional

# Third-party imports
import numpy as np
from shapely.geometry import Polygon
from pydantic import field_validator, Field, model_validator

# Local imports
from rompy.core.grid import RegularGrid
from rompy.core.logging import get_logger
from rompy.swan.subcomponents.readgrid import GRIDREGULAR

# Initialize the logger
logger = get_logger(__name__)


class SwanGrid(RegularGrid):
    """Regular SWAN grid in geographic space."""

    grid_type: Literal["REG", "CURV"] = Field(
        "REG", description="Type of grid (REG=regular, CURV=curvilinear)"
    )
    exc: Optional[float] = Field(None, description="Missing value")
    gridfile: Optional[str] = Field(
        None, description="Name of grid file to load", max_length=36
    )

    @field_validator("grid_type")
    @classmethod
    def validate_grid_type(cls, v):
        if v not in ["REG", "CURV"]:
            raise ValueError("grid_type must be one of REG or CURV")
        return v

    @model_validator(mode="after")
    def validate_curvilinear_grid(self) -> "SwanGrid":
        if self.grid_type == "CURV" and self.gridfile is None:
            raise ValueError("gridfile must be provided for CURV grid")
        return self

    def _regen_grid(self):
        if self.grid_type == "REG":
            _x, _y = self._gen_reg_cgrid()
        elif self.grid_type == "CURV":
            _x, _y = self._gen_curv_cgrid()
        self.x = _x
        self.y = _y

    def _gen_curv_cgrid(self):
        """loads a SWAN curvilinear grid and returns cgrid lat/lons and
        command to be used in SWAN contol file. The Default grid is one I made using
        Deltares' RGFGrid tool and converted to a SWAN-friendly formate using Deltares
        OpenEarth code "swan_io_grd.m"

        """
        # number of grid cells in the 'x' and 'y' directions:
        # (you can get this from d3d_qp.m or other Deltares OpenEarth code)
        nX = self.nx
        nY = self.ny

        grid_Data = open(self.gridpath).readlines()
        ix = grid_Data.index("x-coordinates\n")
        iy = grid_Data.index("y-coordinates\n")
        lons = []
        lats = []
        for idx in np.arange(ix + 1, iy):
            lons.append(re.sub("\n", "", grid_Data[idx]).split())
        for idx in np.arange(iy + 1, len(grid_Data)):
            lats.append(re.sub("\n", "", grid_Data[idx]).split())

        def flatten(l):
            return [item for sublist in l for item in sublist]

        lons = np.array(flatten(lons)).astype(np.float)
        lats = np.array(flatten(lats)).astype(np.float)

        x = np.reshape(lats, (nX, nY))
        y = np.reshape(lons, (nX, nY))

        return x, y

    @property
    def inpgrid(self):
        if self.grid_type == "REG":
            inpstr = f"REG {self.x0} {self.y0} {self.rot} {self.nx-1:0.0f} {self.ny-1:0.0f} {self.dx} {self.dy}"
            if self.exc is not None:
                inpstr += f" EXC {self.exc}"
            return inpstr
        elif self.grid_type == "CURV":
            raise NotImplementedError("Curvilinear grids not supported yet")
            # return f'CURVilinear {self.nx-1:0.0f} {self.ny-1:0.0f} \nREADGRID COOR 1 \'{os.path.basename(self.gridpath)}\' 1 0 1 FREE'

    @property
    def cgrid(self):
        if self.grid_type == "REG":
            return f"REG {self.x0} {self.y0} {self.rot} {self.xlen} {self.ylen} {self.nx-1:0.0f} {self.ny-1:0.0f}"
        elif self.grid_type == "CURV":
            raise NotImplementedError("Curvilinear grids not supported yet")
            # return (f'CURVilinear {self.nx-1:0.0f} {self.ny-1:0.0f}',f'READGRID COOR 1 \'{os.path.basename(self.gridpath)}\' 1 0 1 FREE')

    @property
    def cgrid_read(self):
        if self.grid_type == "REG":
            return ""
        elif self.grid_type == "CURV":
            raise NotImplementedError("Curvilinear grids not supported yet")
            # return f'READGRID COOR 1 \'{os.path.basename(self.gridpath)}\' 1 0 1 FREE'

    @property
    def component(self):
        """Return the respective SWAN component for this grid."""
        if self.grid_type == "REG":
            return GRIDREGULAR(
                xp=self.x0,
                yp=self.y0,
                alp=self.rot,
                xlen=self.xlen,
                ylen=self.ylen,
                mx=self.nx - 1,
                my=self.ny - 1,
            )
        else:
            raise NotImplementedError("Only regular grid is currently supported")

    def __call__(self):
        output = f"CGRID {self.cgrid} CIRCLE 36 0.0464 1. 31\n"
        output += f"{self.cgrid_read}\n"
        return output

    def boundary(self, *args, **kwargs) -> tuple:
        """Returns the grid boundary polygon.

        Override the parent method to use the actual points from the regular grid
        boundary instead of the convex hull which is not always the boundary.

        """
        x = np.concatenate(
            [self.x[0, :], self.x[1:, -1], self.x[-1, -2::-1], self.x[-2::-1, 0]]
        )
        y = np.concatenate(
            [self.y[0, :], self.y[1:, -1], self.y[-1, -2::-1], self.y[-2::-1, 0]]
        )
        return Polygon(zip(x, y))

    def nearby_spectra(self, ds_spec, dist_thres=0.05, plot=True):
        """Find points nearby and project to the boundary

        Parameters
        ----------
        ds_spec: xarray.Dataset
            an XArray dataset of wave spectra at a number of points.
            Dataset variable names standardised using wavespectra.read_*
            functions.

            See https://wavespectra.readthedocs.io/en/latest/api.html#input-functions
        dist_thres: float, optional [Default: 0.05]
            Maximum distance to translate the input spectra to the grid boundary
        plot: boolean, optional [Default: True]
            Generate a plot that shows the applied projections

        Returns
        -------
        xarray.Dataset
            A subset of ds_spec with lat and lon coordinates projected to the boundary
        """

        bbox = self.bbox(buffer=dist_thres)
        minLon, minLat, maxLon, maxLat = bbox

        inds = np.where(
            (ds_spec.lon > minLon)
            & (ds_spec.lon < maxLon)
            & (ds_spec.lat > minLat)
            & (ds_spec.lat < maxLat)
        )[0]
        ds_spec = ds_spec.isel(site=inds)

        # Work out the closest spectral points
        def _nearestPointOnLine(p1, p2, p3):
            # calculate the distance of p3 from the line between p1 and p2 and return
            # the closest point on the line

            from math import fabs, sqrt

            a = p2[1] - p1[1]
            b = -1.0 * (p2[0] - p1[0])
            c = p2[0] * p1[1] - p2[1] * p1[0]

            dist = fabs(a * p3[0] + b * p3[1] + c) / sqrt(a**2 + b**2)
            x = (b * (b * p3[0] - a * p3[1]) - a * c) / (a**2 + b**2)
            y = (a * (-b * p3[0] + a * p3[1]) - b * c) / (a**2 + b**2)

            return dist, x, y

        bx, by = self.boundary_points()
        pol = np.stack([bx, by])

        # Spectra points
        ds_spec.lon.load()
        ds_spec.lat.load()
        ds_spec["lon_original"] = ds_spec["lon"]
        ds_spec["lat_original"] = ds_spec["lat"]
        p3s = list(zip(ds_spec.lon.values, ds_spec.lat.values))

        if plot:
            fig, ax = self.plot()
            ax.scatter(ds_spec.lon, ds_spec.lat)

        specPoints = []
        specPointCoords = []
        for i in range(pol.shape[1] - 1):
            p1 = pol[:, i]
            p2 = pol[:, i + 1]
            line = np.stack((p1, p2))
            output = np.array(
                list(map(lambda xi: _nearestPointOnLine(p1, p2, xi), p3s))
            )
            dists = output[:, 0]
            segmentPoints = output[:, 1:]
            inds = np.where((dists < dist_thres))[0]

            # Loop through the points projected onto the line
            for ind in inds:
                specPoint = ds_spec.isel(site=ind)

                segLon = segmentPoints[ind, 0]
                segLat = segmentPoints[ind, 1]

                if plot:
                    ax.plot(
                        [segLon, specPoint.lon],
                        [segLat, specPoint.lat],
                        color="r",
                        lw=2,
                    )
                    ax.scatter(specPoint.lon, specPoint.lat, marker="o", color="b")
                    ax.scatter(segLon, segLat, marker="x", color="g")

                specPoint["lon"] = segLon
                specPoint["lat"] = segLat
                specPoints.append(specPoint)

            logger.debug(f"Segment {i} - Indices {inds}")

        if plot:
            fig.show()

        ds_boundary = xr.concat(specPoints, dim="site")
        return ds_boundary

    def __repr__(self):
        return f"SwanGrid: {self.grid_type}, {self.nx}x{self.ny}"

    def __str__(self):
        return f"SwanGrid: {self.grid_type}, {self.nx}x{self.ny}"

    @classmethod
    def from_component(cls, component: GRIDREGULAR) -> "SwanGrid":
        """Swan grid from an existing component.

        Parameters
        ----------
        component: GRIDREGULAR
            A GRIDREGULAR SWAN component.

        Returns
        -------
        SwanGrid
            A SwanGrid object.

        """
        return cls(
            x0=component.xp,
            y0=component.yp,
            rot=component.alp,
            dx=component.dx,
            dy=component.dy,
            nx=component.mx + 1,
            ny=component.my + 1,
        )
