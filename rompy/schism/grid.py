import logging
from pathlib import Path
from typing import Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field, PrivateAttr, field_validator, model_validator
from shapely.geometry import MultiPoint, Polygon

from rompy.core import DataBlob, RompyBaseModel
from rompy.core.grid import BaseGrid

# Import PyLibs for SCHISM grid handling
from pylib import read_schism_hgrid, schism_grid, read_schism_vgrid

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
            
        # Determine the output filename
        dest = Path(destdir) / f"{self.gr3_type}.gr3"
        
        # Load the grid with PyLibs
        try:
            gd = schism_grid(ref)
        except Exception:
            gd = read_schism_hgrid(ref)
        
        # Generate a standard gr3 file that matches PySchism format
        # This follows the same format as hgrid.gr3: description, NE NP, node list, element list
        logger.info(f"Generating {self.gr3_type}.gr3 with constant value {self.value}")
        
        with open(dest, 'w') as f:
            # First line: Description
            f.write(f"{self.gr3_type} gr3 file\n")
            
            # Second line: NE NP (# of elements, # of nodes)
            f.write(f"{gd.ne} {gd.np}\n")
            
            # Write node information with the constant value
            # Format: node_id x y value
            for i in range(gd.np):
                f.write(f"{i+1} {gd.x[i]:.8f} {gd.y[i]:.8f} {self.value:.8f}\n")
            
            # Write element connectivity
            # Format: element_id num_vertices vertex1 vertex2 ...
            for i in range(gd.ne):
                if hasattr(gd, 'i34') and gd.i34 is not None:
                    num_vertices = gd.i34[i]
                elif hasattr(gd, 'elnode') and gd.elnode is not None:
                    # Count non-negative values for number of vertices
                    num_vertices = sum(1 for x in gd.elnode[i] if x >= 0)
                else:
                    num_vertices = 3  # Default to triangles
                
                # Write element connectivity line
                if hasattr(gd, 'elnode') and gd.elnode is not None:
                    vertices = ' '.join(str(gd.elnode[i, j]+1) for j in range(num_vertices))
                    f.write(f"{i+1} {num_vertices} {vertices}\n")
            
            # Add empty line at the end (part of PySchism gr3 format)
            f.write("\n")
            self._copied = dest
            return dest
                
        # For all other gr3 files (e.g., albedo, diffmin, diffmax, etc.), use standard node-based gr3 format
        with open(dest, 'w') as f:
            f.write(f"{self.gr3_type} gr3 file\n")
            f.write(f"{gd.np} {gd.ne}\n")
            
            # Write node information with constant value
            for i in range(gd.np):
                f.write(f"{i+1} {gd.x[i]} {gd.y[i]} {self.value}\n")
            
            # Write element connectivity
            for i in range(gd.ne):
                if hasattr(gd, 'i34') and gd.i34 is not None:
                    num_vertices = gd.i34[i]
                elif hasattr(gd, 'elnode') and gd.elnode is not None:
                    # For triangular elements, count non-negative values
                    num_vertices = sum(1 for x in gd.elnode[i] if x >= 0)
                else:
                    num_vertices = 3  # Default to triangles
                
                if hasattr(gd, 'elnode') and gd.elnode is not None:
                    vertices = ' '.join(str(gd.elnode[i, j]+1) for j in range(num_vertices))
                    f.write(f"{i+1} {num_vertices} {vertices}\n")
                
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


# Stub implementations of vertical grid types for PyLibs migration
class LSC2:
    """Stub implementation of the LSC2 vertical grid type from PySchism."""
    
    def __init__(self, hsm, nv, h_c, theta_b, theta_f):
        self.hsm = hsm
        self.nv = nv
        self.h_c = h_c
        self.theta_b = theta_b
        self.theta_f = theta_f
        self.m_grid = None
        self.lsc2_att = None
    
    def calc_m_grid(self):
        """Calculate the master grid."""
        logger.info("Calculating master grid (stub implementation)")
        self.m_grid = {'depth': self.hsm, 'levels': self.nv}
    
    def calc_lsc2_att(self, hgrid, crs="epsg:4326"):
        """Calculate the LSC2 attributes."""
        logger.info("Calculating LSC2 attributes (stub implementation)")
        self.lsc2_att = {'h_c': self.h_c, 'theta_b': self.theta_b, 'theta_f': self.theta_f}
    
    def write(self, path):
        """Write the vgrid.in file."""
        logger.info(f"Writing vgrid.in to {path}")
        with open(path, 'w') as f:
            f.write(f"LSC2 vertical grid with {len(self.hsm)} master grids\n")
            f.write(f"{len(self.hsm)}  !number of master grids\n")
            
            # Write master grid depths
            for i, depth in enumerate(self.hsm):
                f.write(f"{i+1} {depth}  !master grid {i+1} depth\n")
            
            # Write number of levels
            for i, levels in enumerate(self.nv):
                f.write(f"{i+1} {levels}  !master grid {i+1} levels\n")
            
            # Write h_c, theta_b, theta_f
            f.write(f"{self.h_c} {self.theta_b} {self.theta_f}  !h_c, theta_b, theta_f\n")
        return path


class SZ:
    """Stub implementation of the SZ vertical grid type from PySchism."""
    
    def __init__(self, h_s, ztot, h_c, theta_b, theta_f, sigma):
        self.h_s = h_s
        self.ztot = ztot
        self.h_c = h_c
        self.theta_b = theta_b
        self.theta_f = theta_f
        self.sigma = sigma
    
    def write(self, path):
        """Write the vgrid.in file."""
        logger.info(f"Writing vgrid.in to {path}")
        with open(path, 'w') as f:
            f.write(f"SZ vertical grid\n")
            f.write(f"2 {len(self.ztot)} {self.h_s}  !nvrt (# of S-Z levels) (=Nz); kz (# of Z-levels); h_s (transition depth between S and Z)\n")
            
            # Write Z levels
            f.write("Z levels   !Z-levels in the lower portion\n")
            for i, z in enumerate(self.ztot):
                f.write(f"{i+1} {-z}   !level index, z-coordinates\n")
            
            # Write S levels
            f.write("S levels      !S-levels\n")
            f.write(f"{self.h_c} {self.theta_b} {self.theta_f}  !constants used in S-transformation: h_c, theta_b, theta_f\n")
            
            # Write sigma levels
            for i, s in enumerate(self.sigma):
                f.write(f"{i+1} {s}    !sigma-coordinate\n")
        return path


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
    _pylibs_hgrid: Optional[schism_grid] = None
    _pylibs_vgrid: Optional[object] = None

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
        return self.pylibs_hgrid.x

    @property
    def y(self) -> np.ndarray:
        return self.pylibs_hgrid.y

    @property
    def pylibs_hgrid(self):
        if self._pylibs_hgrid is None:
            grid_path = self.hgrid._copied or self.hgrid.source
            try:
                # Try to load as schism_grid first
                self._pylibs_hgrid = schism_grid(grid_path)
            except Exception:
                # Fall back to read_schism_hgrid
                self._pylibs_hgrid = read_schism_hgrid(grid_path)
            
            # Compute all grid properties to ensure they're available
            if hasattr(self._pylibs_hgrid, 'compute_all'):
                self._pylibs_hgrid.compute_all()
            
            # Calculate boundary information
            if hasattr(self._pylibs_hgrid, 'compute_bnd'):
                self._pylibs_hgrid.compute_bnd()
                
        return self._pylibs_hgrid

    @property
    def pylibs_vgrid(self):
        if self.vgrid is None:
            return None
        if self._pylibs_vgrid is None:
            vgrid_path = self.vgrid._copied or self.vgrid.source
            self._pylibs_vgrid = read_schism_vgrid(vgrid_path)
        return self._pylibs_vgrid
        
    # Legacy properties for backward compatibility
    @property
    def pyschism_hgrid(self):
        logger.warning("pyschism_hgrid is deprecated, use pylibs_hgrid instead")
        return self.pylibs_hgrid
        
    @property
    def pyschism_vgrid(self):
        logger.warning("pyschism_vgrid is deprecated, use pylibs_vgrid instead")
        return self.pylibs_vgrid

    @property
    def is_3d(self):
        if self.vgrid == None:
            return False
        elif isinstance(self.vgrid, DataBlob):
            return True
        elif isinstance(self.vgrid, VgridGenerator):
            if isinstance(self.vgrid.vgrid, Vgrid2D):
                return False
            else:
                return True

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
            Path: Path to tvd.prop file
        """
        # With PyLibs, we create a simple tvd.prop file directly
        dest = destdir / "tvd.prop"
        
        # Get number of elements
        num_elements = self.pylibs_hgrid.ne
        
        # Write tvd.prop file with all 1's (upwind TVD)
        with open(dest, 'w') as f:
            f.write(f"{num_elements}\n")
            for i in range(num_elements):
                f.write(f"{i+1} 1\n")
                
        return dest

    def boundary(self, tolerance=None) -> Polygon:
        gd = self.pylibs_hgrid
        
        # Make sure boundaries are computed
        if hasattr(gd, 'compute_bnd') and not hasattr(gd, 'nob'):
            gd.compute_bnd()
            
        if not hasattr(gd, 'nob') or gd.nob is None or gd.nob == 0:
            logger.warning("No open boundaries found in grid")
            # Return an empty polygon
            return Polygon()
            
        # Extract coordinates for the first open boundary
        boundary_nodes = gd.iobn[0]
        x = gd.x[boundary_nodes]
        y = gd.y[boundary_nodes]
        
        # Create a polygon
        polygon = Polygon(zip(x, y))
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

        gd = self.pylibs_hgrid
        
        # Create a triangulation for plotting
        elements_array = gd.i34info.astype(int)
        if hasattr(gd, 'elnode'):
            elements_array = gd.elnode
        
        meshtri = Triangulation(
            gd.x,
            gd.y,
            elements_array,
        )
        ax.triplot(meshtri, color="k", alpha=0.3)
        
        # Make sure boundaries are computed
        if hasattr(gd, 'compute_bnd') and not hasattr(gd, 'nob'):
            gd.compute_bnd()
            
        # Plot open boundaries if they exist
        if hasattr(gd, 'nob') and gd.nob is not None and gd.nob > 0:
            # Plot each open boundary
            for i in range(gd.nob):
                boundary_nodes = gd.iobn[i]
                x_boundary = gd.x[boundary_nodes]
                y_boundary = gd.y[boundary_nodes]
                
                # Plot the line
                ax.plot(x_boundary, y_boundary, '-b', linewidth=2, transform=ccrs.PlateCarree())
                
                # Plot the points
                ax.plot(x_boundary, y_boundary, '+k', markersize=6, transform=ccrs.PlateCarree(), zorder=10)
                
                # Create a dataframe for reference
                df_open_boundary = pd.DataFrame(
                    {
                        "boundary_id": i,
                        "node_index": boundary_nodes,
                        "lon": x_boundary,
                        "lat": y_boundary,
                    }
                )
        
        # Plot land boundaries if they exist
        if hasattr(gd, 'nlb') and gd.nlb is not None and gd.nlb > 0:
            # Plot each land boundary
            for i in range(gd.nlb):
                boundary_nodes = gd.ilbn[i]
                x_boundary = gd.x[boundary_nodes]
                y_boundary = gd.y[boundary_nodes]
                
                # Check if this is an island
                is_island = False
                if hasattr(gd, 'island') and gd.island is not None and i < len(gd.island):
                    is_island = gd.island[i] == 1
                    
                # Plot the land boundary with different color for islands
                color = 'r' if is_island else 'g'  # Red for islands, green for land
                ax.plot(x_boundary, y_boundary, f'-{color}', linewidth=2, transform=ccrs.PlateCarree())
                
        # Add coastlines and borders to the map for context
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        
        # Return the figure and axis for further customization
        return fig, ax
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
        gd = self.pylibs_hgrid
        
        # Make sure boundaries are computed
        if hasattr(gd, 'compute_bnd') and not hasattr(gd, 'nob'):
            gd.compute_bnd()
            
        if not hasattr(gd, 'nob') or gd.nob is None or gd.nob == 0:
            logger.warning("No open boundaries found in grid")
            return np.array([]), np.array([])
            
        # Collect all open boundary coordinates
        x_coords = []
        y_coords = []
        
        for i in range(gd.nob):
            boundary_nodes = gd.iobn[i]
            x_coords.extend(gd.x[boundary_nodes])
            y_coords.extend(gd.y[boundary_nodes])
            
        return np.array(x_coords), np.array(y_coords)

    def land_boundary(self):
        gd = self.pylibs_hgrid
        
        # Make sure boundaries are computed
        if hasattr(gd, 'compute_bnd') and not hasattr(gd, 'nob'):
            gd.compute_bnd()
            
        if not hasattr(gd, 'nlb') or gd.nlb is None or gd.nlb == 0:
            logger.warning("No land boundaries found in grid")
            return np.array([]), np.array([])
            
        # Collect all land boundary coordinates
        x_coords = []
        y_coords = []
        
        for i in range(gd.nlb):
            boundary_nodes = gd.ilbn[i]
            x_coords.extend(gd.x[boundary_nodes])
            y_coords.extend(gd.y[boundary_nodes])
            
        return np.array(x_coords), np.array(y_coords)

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
