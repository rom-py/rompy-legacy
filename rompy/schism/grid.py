import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr, field_validator, model_validator
from shapely.geometry import MultiPoint, Polygon

from rompy.core import DataBlob, RompyBaseModel
from rompy.core.grid import BaseGrid
# from pyschism.mesh import Hgrid
# from pyschism.mesh.prop import Tvdflag
# from pyschism.mesh.vgrid import LSC2, SZ, Vgrid
from rompy.schism.pyschism.mesh import Hgrid
from rompy.schism.pyschism.mesh.base import Gr3
from rompy.schism.pyschism.mesh.prop import Tvdflag
from rompy.schism.pyschism.mesh.vgrid import LSC2, SZ, Vgrid

logger = logging.getLogger(__name__)

import os

from pydantic import BaseModel, field_validator

G3ACCEPT = ["albedo", "diffmin", "diffmax", "watertype", "windrot_geo2proj"]
G3WARN = ["manning", "rough", "drag"]
G3FILES = G3ACCEPT + G3WARN
GRIDLINKS = ["hgridll", "hgrid_WWM"]


class GeneratorBase(RompyBaseModel):
    """Base class for all generators"""

    _copied: str = PrivateAttr(default=None)

    def generate(self, destdir: str | Path) -> Path:
        raise NotImplementedError

    def get(self, destdir: str | Path, name: str = None) -> Path:
        """Alias to maintain api compatibility with DataBlob"""
        return self.generate(destdir)


class GR3Generator(GeneratorBase):
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    gr3_type: str = Field(
        ...,
        description="Type of gr3 file. Must be one of 'albedo', 'diffmin', 'diffmax', 'watertype', 'windrot_geo2proj'",
    )
    value: float = Field(None, description="Constant value to set in gr3 file")
    crs: str = Field("epsg:4326", description="Coordinate reference system")

    @field_validator("gr3_type")
    def gr3_type_validator(cls, v):
        if v not in G3FILES:
            raise ValueError(
                "gr3_type must be one of 'albedo', 'diffmin', 'diffmax', 'watertype', 'windrot_geo2proj'"
            )
        if v in G3WARN:
            logger.warning(
                f"{v} is being set to a constant value, this is not recommended. For best results, please supply friction gr3 files with spatially varying values. Further options are under development."
            )
        return v

    @property
    def id(self):
        return self.gr3_type

    def generate(self, destdir: str | Path, name: str = None) -> Path:
        if isinstance(self.hgrid, DataBlob):
            if not self.hgrid._copied:
                self.hgrid.get(destdir, name="hgrid.gr3")
            ref = self.hgrid._copied
        else:
            ref = self.hgrid
        dest = Path(destdir) / f"{self.gr3_type}.gr3"
        hgrid = Gr3.open(ref, crs=self.crs)
        grd = hgrid.copy()
        grd.description = self.gr3_type
        grd.nodes.values[:] = self.value
        grd.write(dest, overwrite=True)
        logger.info(f"Generated {self.gr3_type} with constant value of {self.value}")
        self._copied = dest
        return dest


# class VgridGeneratorBase(GeneratorBase):


class Vgrid2D(GeneratorBase):
    model_type: Literal["vgrid2D_generator"] = Field(
        "LSC2_generator", description="Model descriminator"
    )

    def generate(self, destdir: str | Path, hgrid=None) -> Path:
        dest = Path(destdir) / "vgrid.in"
        with open(dest, "w") as f:
            f.write("2 !ivcor (1: LSC2; 2: SZ) ; type of mesh you are using\n")
            f.write(
                "2 1 1000000  !nvrt (# of S-levels) (=Nz); kz (# of Z-levels); hs (transition depth between S and Z); large in this case because is 2D implementation\n"
            )
            f.write("Z levels   !Z-levels in the lower portion\n")
            f.write(
                "1 -1000000   !level index, z-coordinates !use very large value for 2D; if 3D would have a list of z-levels here\n"
            )
            f.write("S levels      !S-levels\n")
            f.write(
                "40.0 1.0 0.0001  ! constants used in S-transformation: hc, theta_b, theta_f\n"
            )
            f.write("1 -1.0    !first S-level (sigma-coordinate must be -1)\n")
            f.write("2 0.0     !last sigma-coordinate must be 0\n")
            f.write(
                "!for 3D, would have the levels index and sigma coordinate for each level\n"
            )
        self._copied = dest
        return dest


class Vgrid3D_LSC2(GeneratorBase):
    model_type: Literal["vgrid3D_lsc2"] = Field(
        "LSC2_generator", description="Model descriminator"
    )
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    hsm: list[float] = Field(..., description="Depth for each master grid")
    nv: list[int] = Field(..., description="Total number of vertical levels")
    h_c: float = Field(
        ..., description="Transition depth between sigma and z-coordinates"
    )
    theta_b: float = Field(..., description="Vertical resolution near the surface")
    theta_f: float = Field(..., description="Vertical resolution near the seabed")
    crs: str = Field("epsg:4326", description="Coordinate reference system")
    _vgrid = PrivateAttr(default=None)

    @property
    def vgrid(self):
        if self._vgrid is None:
            self._vgrid = LSC2(
                hsm=self.hsm,
                nv=self.nv,
                h_c=self.h_c,
                theta_b=self.theta_b,
                theta_f=self.theta_f,
            )
            logger.info("Generating LSC2 vgrid")
            self._vgrid.calc_m_grid()
            self._vgrid.calc_lsc2_att(self.hgrid, crs=self.crs)
        return self._vgrid

    def generate(self, destdir: str | Path) -> Path:
        dest = Path(destdir) / "vgrid.in"
        self.vgrid.write(dest)
        return dest


class Vgrid3D_SZ(GeneratorBase):
    model_type: Literal["vgrid3D_sz"] = Field(
        "LSC2_generator", description="Model descriminator"
    )
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    h_s: float = Field(..., description="Depth for each master grid")
    ztot: list[int] = Field(..., description="Total number of vertical levels")
    h_c: float = Field(
        ..., description="Transition depth between sigma and z-coordinates"
    )
    theta_b: float = Field(..., description="Vertical resolution near the surface")
    theta_f: float = Field(..., description="Vertical resolution near the seabed")
    sigma: list[float] = Field(..., description="Sigma levels")
    _vgrid = PrivateAttr(default=None)

    @property
    def vgrid(self):
        if self._vgrid is None:
            self._vgrid = SZ(
                h_s=self.h_s,
                ztot=self.ztot,
                h_c=self.h_c,
                theta_b=self.theta_b,
                theta_f=self.theta_f,
                sigma=self.sigma,
            )
            logger.info("Generating SZ grid")
        return self._vgrid

    def generate(self, destdir: str | Path) -> Path:
        dest = Path(destdir) / "vgrid.in"
        self.vgrid.write(dest)
        return dest


class VgridGenerator(GeneratorBase):
    """
    Generate vgrid.in.
    This is all hardcoded for now, may look at making this more flexible in the future.
    """

    # model_type: Literal["vgrid_generator"] = Field(
    #     "vgrid_generator", description="Model descriminator"
    # )
    vgrid: Union[Vgrid2D, Vgrid3D_LSC2, Vgrid3D_SZ] = Field(
        ...,
        default_factory=Vgrid2D,
        description="Type of vgrid to generate. 2d will create the minimum required for a 2d model. LSC2 will create a full vgrid for a 3d model using pyschsim's LSC2 class",
    )

    def generate(self, destdir: str | Path) -> Path:
        dest = self.vgrid.generate(destdir=destdir)

    def generate_legacy(self, destdir: str | Path) -> Path:
        dest = Path(destdir) / "vgrid.in"
        with open(dest, "w") as f:
            f.write("2 !ivcor (1: LSC2; 2: SZ) ; type of mesh you are using\n")
            f.write(
                "2 1 1000000  !nvrt (# of S-levels) (=Nz); kz (# of Z-levels); hs (transition depth between S and Z); large in this case because is 2D implementation\n"
            )
            f.write("Z levels   !Z-levels in the lower portion\n")
            f.write(
                "1 -1000000   !level index, z-coordinates !use very large value for 2D; if 3D would have a list of z-levels here\n"
            )
            f.write("S levels      !S-levels\n")
            f.write(
                "40.0 1.0 0.0001  ! constants used in S-transformation: hc, theta_b, theta_f\n"
            )
            f.write("1 -1.0    !first S-level (sigma-coordinate must be -1)\n")
            f.write("2 0.0     !last sigma-coordinate must be 0\n")
            f.write(
                "!for 3D, would have the levels index and sigma coordinate for each level\n"
            )
        self._copied = dest
        return dest


class WWMBNDGR3Generator(GeneratorBase):
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    bcflags: list[int] = Field(
        None,
        description="List of boundary condition flags. This replicates the functionality of the gen_wwmbnd.in file. Must be the same length as the number of open boundaries in the hgrid.gr3 file. If not specified, it is assumed that all open hgrid files are open to waves",
    )

    def generate(self, destdir: str | Path, name: str = None) -> Path:
        # Adapted from https://github.com/schism-dev/schism/blob/master/src/Utility/Pre-Processing/gen_wwmbnd.f90
        # Read input files
        if isinstance(self.hgrid, DataBlob):
            if not self.hgrid._copied:
                self.hgrid.get(destdir, name="hgrid.gr3")
            ref = self.hgrid._copied
        else:
            ref = self.hgrid

        with open(ref, "r") as file:
            file.readline()
            ne, nnp = map(int, file.readline().split())
            xnd, ynd, ibnd, nm = (
                np.zeros(nnp),
                np.zeros(nnp),
                np.zeros(nnp),
                np.zeros((ne, 3), dtype=int),
            )
            ibnd.fill(0)

            for i in range(nnp):
                j, xnd[i], ynd[i], tmp = map(float, file.readline().split())

            for i in range(ne):
                j, k, *nm[i, :] = map(int, file.readline().split())

            nope = int(file.readline().split()[0].strip())

            bcflags = self.bcflags or np.ones(nope, dtype=int) * 2
            nope2 = len(bcflags)
            ifl_wwm = np.array(bcflags, dtype=int)

            if nope != nope2:
                raise ValueError(
                    f"List of flags {nope2} must be the same length as the number of open boundaries in the hgrid.gr3 file ({nope})"
                )

            neta = int(file.readline().split()[0].strip())

            for k in range(nope):
                nond = int(file.readline().split()[0].strip())
                for _ in range(nond):
                    iond = int(file.readline().strip())
                    if iond > nnp or iond <= 0:
                        raise ValueError("iond > nnp")

                    ibnd[iond - 1] = ifl_wwm[k]

        # Write output file
        dest = Path(destdir) / "wwmbnd.gr3"
        with open(dest, "w") as file:
            file.write("Generated by rompy\n")
            file.write(f"{ne} {nnp}\n")
            for i in range(nnp):
                file.write(f"{i+1} {xnd[i]} {ynd[i]} {float(ibnd[i])}\n")

            for i in range(ne):
                file.write(f"{i+1} 3 {' '.join(map(str, nm[i, :]))}\n")
        self._copied = dest
        return dest


class GridLinker(GeneratorBase):
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    gridtype: str = Field(..., description="Type of grid to link")

    @field_validator("gridtype")
    @classmethod
    def gridtype_validator(cls, v):
        if v not in GRIDLINKS:
            raise ValueError(f"gridtype must be one of {GRIDLINKS}")
        return v

    def generate(self, destdir: str | Path, name: str = None) -> Path:
        if isinstance(self.hgrid, DataBlob):
            if not self.hgrid._copied:
                self.hgrid.get(destdir, name="hgrid.gr3")
            ref = self.hgrid._copied.name
        else:
            ref = self.hgrid
        if self.gridtype == "hgridll":
            filename = "hgrid.ll"
        elif self.gridtype == "hgrid_WWM":
            filename = "hgrid_WWM.gr3"
        dest = Path(destdir) / f"{filename}"
        logger.info(f"Linking {ref} to {dest}")
        dest.symlink_to(ref)
        return dest


# TODO - check datatypes for gr3 files (int vs float)
class SCHISMGrid(BaseGrid):
    """SCHISM grid in geographic space."""

    grid_type: Literal["schism"] = Field("schism", description="Model descriminator")
    hgrid: DataBlob = Field(..., description="Path to hgrid.gr3 file")
    vgrid: Optional[DataBlob | VgridGenerator] = Field(
        default=None,
        description="Path to vgrid.in file",
        validate_default=True,
    )
    drag: Optional[DataBlob | float | GR3Generator] = Field(
        default=None, description="Path to drag.gr3 file"
    )
    rough: Optional[DataBlob | float | GR3Generator] = Field(
        default=None, description="Path to rough.gr3 file"
    )
    manning: Optional[DataBlob | float | GR3Generator] = Field(
        default=None,
        description="Path to manning.gr3 file",  # TODO treat in the same way as the other gr3 files. Add a warning that this is not advisable
    )
    hgridll: Optional[DataBlob | int | GridLinker] = Field(
        default=None,
        description="Path to hgrid.ll file",
        validate_default=True,
    )
    diffmin: Optional[DataBlob | float | GR3Generator] = Field(
        default=1.0e-6,
        description="Path to diffmax.gr3 file or constant value",
        validate_default=True,
    )
    diffmax: Optional[DataBlob | float | GR3Generator] = Field(
        default=1.0,
        description="Path to diffmax.gr3 file or constant value",
        validate_default=True,
    )
    albedo: Optional[DataBlob | float | GR3Generator] = Field(
        default=0.15,
        description="Path to albedo.gr3 file or constant value",
        validate_default=True,
    )
    watertype: Optional[DataBlob | int | GR3Generator] = Field(
        default=1,
        description="Path to watertype.gr3 file or constant value",
        validate_default=True,
    )
    windrot_geo2proj: Optional[DataBlob | float | GR3Generator] = Field(
        default=0.0,
        description="Path to windrot_geo2proj.gr3 file or constant value",
        validate_default=True,
    )
    hgrid_WWM: Optional[DataBlob | GridLinker] = Field(
        default=None,
        description="Path to hgrid_WWM.gr3 file",
        validate_default=True,
    )
    wwmbnd: Optional[DataBlob | WWMBNDGR3Generator] = Field(
        default=None,
        description="Path to wwmbnd.gr3 file",  # This is generated on the fly. Script sent from Vanessa.
        validate_default=True,
    )
    crs: str = Field("epsg:4326", description="Coordinate reference system")
    _pyschism_hgrid: Optional[Hgrid] = None
    _pyschism_vgrid: Optional[Vgrid] = None

    @model_validator(mode="after")
    def validate_rough_drag_manning(cls, v):
        fric_sum = sum([v.rough is not None, v.drag is not None, v.manning is not None])
        if fric_sum > 1:
            raise ValueError("Only one of rough, drag, manning can be set")
        if fric_sum == 0:
            raise ValueError("At least one of rough, drag, manning must be set")
        return v

    @field_validator(*G3FILES)
    @classmethod
    def gr3_source_validator(cls, v, values):
        if v is not None:
            if not isinstance(v, DataBlob):
                v = GR3Generator(
                    hgrid=values.data["hgrid"], gr3_type=values.field_name, value=v
                )
        return v

    @field_validator(*GRIDLINKS)
    @classmethod
    def gridlink_validator(cls, v, values):
        if v is None:
            v = GridLinker(hgrid=values.data["hgrid"], gridtype=values.field_name)
        return v

    @field_validator("wwmbnd")
    @classmethod
    def wwmbnd_validator(cls, v, values):
        if v is None:
            v = WWMBNDGR3Generator(hgrid=values.data["hgrid"])
        return v

    @field_validator("vgrid")
    @classmethod
    def vgrid_validator(cls, v, values):
        if v is None:
            v = VgridGenerator()
        return v

    @property
    def x(self) -> np.ndarray:
        return self.pyschism_hgrid.x

    @property
    def y(self) -> np.ndarray:
        return self.pyschism_hgrid.y

    @property
    def pyschism_hgrid(self):
        if self._pyschism_hgrid is None:
            self._pyschism_hgrid = Hgrid.open(
                self.hgrid._copied or self.hgrid.source, crs=self.crs
            )
        return self._pyschism_hgrid

    @property
    def pyschism_vgrid(self):
        if self.vgrid is None:
            return None
        if self._pyschism_vgrid is None:
            self._pyschism_vgrid = Vgrid.open(self.vgrid._copied or self.vgrid.source)
        return self._pyschism_vgrid

    @property
    def is_3d(self):
        if self.vgrid is not None:
            return True
        else:
            return False

    def get(self, destdir: Path) -> dict:
        ret = {}
        for filetype in G3FILES + ["hgrid"]:
            source = getattr(self, filetype)
            if source is not None:
                ret[filetype] = source.get(destdir, name=f"{filetype}.gr3")
        for filetype in GRIDLINKS + ["vgrid", "wwmbnd"]:
            source = getattr(self, filetype)
            ret[filetype] = source.get(destdir)
        self.generate_tvprop(destdir)
        return ret

    def generate_tvprop(self, destdir: Path) -> Path:
        """Generate tvprop.in file

        Args:
            destdir (Path): Destination directory

        Returns:
            iath: Path to tvprop.in file
        """
        # TODO - should this be handled in the same way as the gr3 files? i.e. would you
        # ever want to provide a file path to tvprop.in?
        tvdflag = Tvdflag(
            self.pyschism_hgrid, np.array([1] * len(self.pyschism_hgrid.elements))
        )
        dest = destdir / "tvd.prop"
        tvdflag.write(dest)
        return dest

    def boundary(self, tolerance=None) -> Polygon:
        bnd = self.pyschism_hgrid.boundaries.open.get_coordinates()
        polygon = Polygon(zip(bnd.x.values, bnd.y.values))
        if tolerance:
            polygon = polygon.simplify(tolerance=tolerance)
        return polygon

    def plot(self, ax=None, **kwargs):
        import matplotlib.pyplot as plt
        from cartopy import crs as ccrs
        from matplotlib.tri import Triangulation

        if ax is None:
            fig = plt.figure(figsize=(20, 10))
            ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
        else:
            fig = plt.gcf()

        meshtri = Triangulation(
            self.pyschism_hgrid.x,
            self.pyschism_hgrid.y,
            self.pyschism_hgrid.elements.array,
        )
        ax.triplot(meshtri, color="k", alpha=0.3)

        # open boundary nodes/info as geopandas df
        gdf_open_boundary = self.pyschism_hgrid.boundaries.open

        # make a pandas dataframe for easier lon/lat referencing during forcing condition generation
        df_open_boundary = pd.DataFrame(
            {
                "schism_index": gdf_open_boundary.index_id[0],
                "index": gdf_open_boundary.indexes[0],
                "lon": gdf_open_boundary.get_coordinates().x,
                "lat": gdf_open_boundary.get_coordinates().y,
            }
        ).reset_index(drop=True)

        # create sub-sampled wave boundary
        # #wave_boundary = redistribute_vertices(gdf_open_boundary.geometry[0], 0.2)
        #
        # df_wave_boundary = pd.DataFrame(
        #     {"lon": wave_boundary.xy[0], "lat": wave_boundary.xy[1]}
        # ).reset_index(drop=True)
        gdf_open_boundary.plot(ax=ax, color="b")
        ax.add_geometries(
            self.pyschism_hgrid.boundaries.land.geometry.values,
            facecolor="none",
            edgecolor="g",
            linewidth=2,
            crs=ccrs.PlateCarree(),
        )
        ax.plot(
            df_open_boundary["lon"],
            df_open_boundary["lat"],
            "+k",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        ax.plot(
            df_open_boundary["lon"],
            df_open_boundary["lat"],
            "xr",
            transform=ccrs.PlateCarree(),
            zorder=10,
        )
        # ax.plot(
        #     df_wave_boundary["lon"],
        #     df_wave_boundary["lat"],
        #     "+k",
        #     transform=ccrs.PlateCarree(),
        #     zorder=10,
        # )
        # ax.plot(
        #     df_wave_boundary["lon"],
        #     df_wave_boundary["lat"],
        #     "xr",
        #     transform=ccrs.PlateCarree(),
        #     zorder=10,
        # )
        ax.coastlines()
        return fig, ax

    def plot_hgrid(self):
        import matplotlib.pyplot as plt
        from cartopy import crs as ccrs
        from matplotlib.tri import Triangulation

        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(121)
        ax.set_title("Bathymetry")

        hgrid = Hgrid.open(self.hgrid._copied or self.hgrid.source)
        self.pyschism_hgrid.make_plot(axes=ax)

        ax = fig.add_subplot(122, projection=ccrs.PlateCarree())
        self.plot(ax=ax)
        ax.set_title("Mesh")

    def ocean_boundary(self):
        bnd = self.pyschism_hgrid.boundaries.open.get_coordinates()
        return bnd.x.values, bnd.y.values

    def land_boundary(self):
        bnd = self.pyschism_hgrid.boundaries.land.get_coordinates()
        return bnd.x.values, bnd.y.values

    def boundary_points(self, spacing=None) -> tuple:
        return self.ocean_boundary()


if __name__ == "__main__":
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt

    grid = SCHISMGrid(
        hgrid=DataBlob(
            source="../../tests/schism/test_data/hgrid.gr3",
            id="hgrid",
        )
    )
    grid.plot_hgrid()
    # plt.figure()
    # grid._set_xy()
    # bnd = grid.boundary()
    # ax = plt.axes(projection=ccrs.PlateCarree())
    # # plot polygon on cartopy axes
    # ax.add_geometries([bnd], ccrs.PlateCarree(), facecolor="none", edgecolor="red")
    # ax.coastlines()
