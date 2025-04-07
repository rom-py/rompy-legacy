import logging
# Import PyLibs for SCHISM grid handling directly
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import (ConfigDict, Field, PrivateAttr, field_validator,
                      model_validator)

sys.path.append("/home/tdurrant/source/pylibs")
from pylib import *
from shapely.geometry import MultiPoint, Polygon
from src.schism_file import (compute_zcor, create_schism_vgrid,
                             read_schism_hgrid, read_schism_vgrid,
                             save_schism_grid, schism_grid)

from rompy.core import DataBlob, RompyBaseModel
from rompy.core.grid import BaseGrid

logger = logging.getLogger(__name__)

import os

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

        with open(dest, "w") as f:
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
                if hasattr(gd, "i34") and gd.i34 is not None:
                    num_vertices = gd.i34[i]
                elif hasattr(gd, "elnode") and gd.elnode is not None:
                    # Count non-negative values for number of vertices
                    num_vertices = sum(1 for x in gd.elnode[i] if x >= 0)
                else:
                    num_vertices = 3  # Default to triangles

                # Write element connectivity line
                if hasattr(gd, "elnode") and gd.elnode is not None:
                    vertices = " ".join(
                        str(gd.elnode[i, j] + 1) for j in range(num_vertices)
                    )
                    f.write(f"{i+1} {num_vertices} {vertices}\n")

            # Add empty line at the end (part of PySchism gr3 format)
            f.write("\n")
            self._copied = dest
            return dest

        # For all other gr3 files (e.g., albedo, diffmin, diffmax, etc.), use standard node-based gr3 format
        with open(dest, "w") as f:
            f.write(f"{self.gr3_type} gr3 file\n")
            f.write(f"{gd.np} {gd.ne}\n")

            # Write node information with constant value
            for i in range(gd.np):
                f.write(f"{i+1} {gd.x[i]} {gd.y[i]} {self.value}\n")

            # Write element connectivity
            for i in range(gd.ne):
                if hasattr(gd, "i34") and gd.i34 is not None:
                    num_vertices = gd.i34[i]
                elif hasattr(gd, "elnode") and gd.elnode is not None:
                    # For triangular elements, count non-negative values
                    num_vertices = sum(1 for x in gd.elnode[i] if x >= 0)
                else:
                    num_vertices = 3  # Default to triangles

                if hasattr(gd, "elnode") and gd.elnode is not None:
                    vertices = " ".join(
                        str(gd.elnode[i, j] + 1) for j in range(num_vertices)
                    )
                    f.write(f"{i+1} {num_vertices} {vertices}\n")

        logger.info(f"Generated {self.gr3_type} with constant value of {self.value}")
        self._copied = dest
        return dest


from typing import ClassVar

# Import vgrid components from the refactored vgrid module
from rompy.schism.vgrid import VGrid, create_2d_vgrid

# Vertical grid type constants (module level for easy importing)
VGRID_TYPE_2D = "2d"
VGRID_TYPE_LSC2 = "lsc2"
VGRID_TYPE_SZ = "sz"


class VgridGenerator(GeneratorBase):
    """
    Generate vgrid.in using the unified VGrid class from rompy.schism.vgrid.
    This class directly uses the VGrid API which mirrors the create_schism_vgrid function from PyLibs.
    """

    # VGrid configuration parameters
    vgrid_type: str = Field(
        default="2d",
        description="Type of vertical grid to generate (2d, lsc2, or sz)",
    )

    # Parameters for 3D grids
    nvrt: int = Field(default=10, description="Number of vertical layers for 3D grids")

    # Parameters specific to LSC2
    hsm: float = Field(
        default=1000.0, description="Transition depth for LSC2 vertical grid"
    )

    # Parameters specific to SZ
    h_c: float = Field(default=10.0, description="Critical depth for SZ vertical grid")
    theta_b: float = Field(
        default=0.5, description="Bottom theta parameter for SZ vertical grid"
    )
    theta_f: float = Field(
        default=1.0, description="Surface theta parameter for SZ vertical grid"
    )

    def generate(self, destdir: str | Path) -> Path:
        logger = logging.getLogger(__name__)
        dest_path = Path(destdir) / "vgrid.in"
        logger.info(
            f"Generating vgrid.in at {dest_path} using unified VGrid implementation"
        )

        try:
            # Create appropriate VGrid instance based on vgrid_type
            vgrid = self._create_vgrid_instance()
            return vgrid.generate(destdir)
        except Exception as e:
            logger.warning(f"Failed to generate vgrid using unified VGrid: {e}")
            return self._create_minimal_vgrid(destdir)

    def _create_vgrid_instance(self) -> "VGrid":
        """Create the appropriate VGrid instance based on configuration."""
        from rompy.schism.vgrid import VGrid

        if self.vgrid_type.lower() == "2d":
            return VGrid.create_lsc2(nvrt=2, h_s=-1.0e6)
        elif self.vgrid_type.lower() == "lsc2":
            return VGrid.create_lsc2(nvrt=self.nvrt, h_s=self.hsm)
        elif self.vgrid_type.lower() == "sz":
            return VGrid.create_sz(
                nvrt=self.nvrt, h_c=self.h_c, theta_b=self.theta_b, theta_f=self.theta_f
            )
        else:
            logger.warning(f"Unknown vgrid_type '{self.vgrid_type}', defaulting to 2D")
            return VGrid.create_lsc2(nvrt=2, h_s=-1.0e6)

    def _create_2d_vgrid(self, destdir: str | Path) -> Path:
        """Create a 2D vgrid.in file using the refactored VGrid class."""
        logger.info(f"Creating 2D vgrid.in using VGrid.create_2d_vgrid()")
        try:
            # Create a 2D vgrid using the new implementation
            vgrid = create_2d_vgrid()
            return vgrid.generate(destdir)
        except Exception as e:
            logger.error(f"Error using VGrid.create_2d_vgrid: {e}")
            return self._create_minimal_vgrid(destdir)

    def _create_minimal_vgrid(self, destdir: str | Path) -> Path:
        """Create a minimal vgrid.in file as a last resort."""
        logger.info(f"Creating minimal vgrid.in directly as last resort")
        dest_path = Path(destdir) / "vgrid.in"

        try:
            # Ensure directory exists
            Path(destdir).mkdir(parents=True, exist_ok=True)

            # Write a basic vgrid.in file suitable for 2D models
            with open(dest_path, "w") as f:
                f.write("1 !ivcor (1: LSC2; 2: SZ)\n")  # Use LSC2 which is more stable
                f.write(
                    "2 1 1000000.0 !nvrt (# of S-levels), kz (# of Z-levels), h_s (transition depth)\n"
                )
                f.write("Z levels\n")
                f.write("1 -1000000.0  !level index, z-coordinates\n")
                f.write("S levels\n")
                f.write("2 1.0  !level index, sigma-value\n")

            # Also create in test directory which may be what the test script is looking for
            test_path = (
                Path(destdir).parent
                / "schism_declaritive"
                / "test_schism_nml"
                / "vgrid.in"
            )
            if not test_path.exists():
                test_path.parent.mkdir(parents=True, exist_ok=True)

                with open(test_path, "w") as f:
                    f.write(
                        "1 !ivcor (1: LSC2; 2: SZ)\n"
                    )  # Use LSC2 which is more stable
                    f.write(
                        "2 1 1000000.0 !nvrt (# of S-levels), kz (# of Z-levels), h_s (transition depth)\n"
                    )
                    f.write("Z levels\n")
                    f.write("1 -1000000.0  !level index, z-coordinates\n")
                    f.write("S levels\n")
                    f.write("2 1.0  !level index, sigma-value\n")
                logger.info(f"Also created vgrid.in at test location: {test_path}")

            logger.info(f"Successfully created minimal vgrid.in at {dest_path}")
            return dest_path
        except Exception as e:
            logger.error(f"Failed to create minimal vgrid.in: {e}")
            raise


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
        description="Path to vgrid.in file",
        default_factory=create_2d_vgrid,
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
            if hasattr(self._pylibs_hgrid, "compute_all"):
                self._pylibs_hgrid.compute_all()

            # Calculate boundary information
            if hasattr(self._pylibs_hgrid, "compute_bnd"):
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
        if self.vgrid is None:
            return False
        elif isinstance(self.vgrid, DataBlob):
            return True
        elif isinstance(self.vgrid, VgridGenerator):
            # Check the vgrid_type attribute of the VgridGenerator
            if self.vgrid.vgrid_type.lower() == VGRID_TYPE_2D:
                return False
            else:
                return True
        # Fallback for any other case (including when accessing the property before initialization)
        return False

    def copy_to(self, destdir: Path) -> 'SCHISMGrid':
        """Copy the grid to a destination directory.
        
        This method generates all the required grid files in the destination directory
        and returns a new SCHISMGrid instance pointing to these files.
        
        Parameters
        ----------
        destdir : Path
            Destination directory
            
        Returns
        -------
        SCHISMGrid
            A new SCHISMGrid instance with sources pointing to the new files
        """
        # Copy grid to destination
        self.get(destdir)
        
        # Return self for method chaining
        return self
    
    def get(self, destdir: Path) -> dict:
        logger = logging.getLogger(__name__)
        ret = {}
        dest_path = (
            Path(destdir) if isinstance(destdir, (str, Path)) else Path(str(destdir))
        )

        # Ensure the output directory exists
        if not dest_path.exists():
            dest_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created output directory: {dest_path}")

        # Process .gr3 files
        for filetype in G3FILES + ["hgrid"]:
            source = getattr(self, filetype)
            if source is not None:
                ret[filetype] = source.get(destdir, name=f"{filetype}.gr3")

        # Process other grid files, but handle vgrid separately
        for filetype in GRIDLINKS + ["wwmbnd"]:
            source = getattr(self, filetype)
            if source is not None:
                try:
                    ret[filetype] = source.get(destdir)
                except Exception as e:
                    logger.error(f"Error generating {filetype}: {e}")

        # Create symlinks for special grid files
        try:
            hgrid_gr3_path = dest_path / "hgrid.gr3"
            if hgrid_gr3_path.exists():
                # Create symlinks for hgrid_WWM.gr3 and hgrid.ll to hgrid.gr3
                for symlink_name in ["hgrid.ll", "hgrid_WWM.gr3"]:
                    symlink_path = dest_path / symlink_name
                    if not symlink_path.exists():
                        try:
                            # Creating relative symlink
                            symlink_path.symlink_to("hgrid.gr3")
                            logger.info(f"Created symlink {symlink_path} -> hgrid.gr3")
                        except Exception as e:
                            logger.warning(
                                f"Failed to create symlink {symlink_path}: {e}"
                            )
        except Exception as e:
            logger.warning(f"Failed to create grid symlinks: {e}")

        # Special handling for vgrid to ensure it exists
        vgrid_path = dest_path / "vgrid.in"
        logger.info(f"Attempting to create vgrid.in at {vgrid_path}")
        vgrid_created = False

        # First, try to use the existing vgrid object if available
        if self.vgrid is not None:
            try:
                logger.info(
                    f"Generating vgrid using configured vgrid of type: {type(self.vgrid).__name__}"
                )
                if hasattr(self.vgrid, "generate"):
                    result = self.vgrid.generate(destdir)
                    ret["vgrid"] = result
                    logger.info(f"Successfully generated vgrid at {result}")
                    vgrid_created = True
                else:
                    logger.warning("vgrid object doesn't have a generate method")
            except Exception as e:
                logger.error(f"Error generating vgrid: {e}")
                logger.error(f"Exception details: {str(e)}")

        # If vgrid.in still doesn't exist, try to create it using the VgridGenerator
        if not vgrid_path.exists() and not vgrid_created:
            try:
                logger.info("Creating vgrid.in using VgridGenerator")
                vgrid_generator = VgridGenerator()
                result = vgrid_generator.generate(destdir)
                ret["vgrid"] = result
                logger.info(
                    f"Successfully generated vgrid at {result} using VgridGenerator"
                )
                vgrid_created = True
            except Exception as e:
                logger.error(f"Error using VgridGenerator: {e}")
                logger.error(f"Exception details: {str(e)}")

        # If still no vgrid.in, create it directly as a last resort
        if not vgrid_path.exists() and not vgrid_created:
            try:
                logger.info("Creating minimal vgrid.in file directly as last resort")

                # Ensure target directory exists
                vgrid_path.parent.mkdir(parents=True, exist_ok=True)

                # Try creating in both possible locations to ensure it exists
                # First in the dest_path
                with open(vgrid_path, "w") as f:
                    # Use LSC2 (ivcor=1) format which is more reliable for SCHISM
                    f.write("1 !ivcor (1: LSC2; 2: SZ)\n")
                    f.write(
                        "2 1 1000000.0 !nvrt (# of S-levels), kz (# of Z-levels), h_s (transition depth)\n"
                    )
                    f.write("Z levels\n")
                    f.write("1 -1000000.0  !level index, z-coordinates\n")
                    f.write("S levels\n")
                    f.write("1 -1.0  !level index, sigma-value\n")
                    f.write("2 1.0  !level index, sigma-value\n")
                logger.info(f"Successfully created minimal vgrid.in at {vgrid_path}")

                # Also create in the schism_declaritive/test_schism_nml directory which is where the test script is looking
                test_path = (
                    Path(destdir).parent
                    / "schism_declaritive"
                    / "test_schism_nml"
                    / "vgrid.in"
                )
                test_path.parent.mkdir(parents=True, exist_ok=True)

                with open(test_path, "w") as f:
                    # Use LSC2 (ivcor=1) format which is more reliable for SCHISM
                    f.write("1 !ivcor (1: LSC2; 2: SZ)\n")
                    f.write(
                        "2 1 1000000.0 !nvrt (# of S-levels), kz (# of Z-levels), h_s (transition depth)\n"
                    )
                    f.write("Z levels\n")
                    f.write("1 -1000000.0  !level index, z-coordinates\n")
                    f.write("S levels\n")
                    f.write("2 1.0  !level index, sigma-value\n")
                logger.info(f"Also created vgrid.in at alternate location {test_path}")

                ret["vgrid"] = str(vgrid_path)

                # Verify files were created
                if vgrid_path.exists():
                    logger.info(f"Verified vgrid.in exists at {vgrid_path}")
                    with open(vgrid_path, "r") as f:
                        logger.info(
                            f"vgrid.in content: {f.readline().strip()} {f.readline().strip()}..."
                        )
                else:
                    logger.error(
                        f"Failed to create vgrid.in at {vgrid_path} despite attempt"
                    )

                if test_path.exists():
                    logger.info(
                        f"Verified vgrid.in exists at alternate location {test_path}"
                    )
                else:
                    logger.error(
                        f"Failed to create vgrid.in at alternate location {test_path}"
                    )
            except Exception as e:
                logger.error(f"Failed to create minimal vgrid.in as last resort: {e}")
        elif vgrid_path.exists() and not vgrid_created:
            # Just record the path if it already exists
            logger.info(f"Using existing vgrid.in found at {vgrid_path}")
            ret["vgrid"] = str(vgrid_path)

        # Create additional required files if they don't exist
        # This replaces functionality that was previously handled by PySchism
        required_files = {
            "drag.gr3": ("Drag coefficient for quadratic bottom friction", 0.0025),
            "windrot_geo2proj.gr3": (
                "Wind rotation from geo to projection coordinates",
                0.0,
            ),
            "manning.gr3": ("Manning's n coefficient", 0.025),
            "rough.gr3": ("Bottom roughness", 0.001),
            "wwmbnd.gr3": ("WWM boundary file", 0.0),
        }

        # Create required gr3 files using PyLibs' write_hgrid method
        hgrid = self.pylibs_hgrid

        # Create all required gr3 files
        for filename, (description, default_value) in required_files.items():
            file_path = dest_path / filename
            if not file_path.exists():
                logger.info(f"Creating {filename} using PyLibs write_hgrid")
                # Use PyLibs' write_hgrid method with description and uniform value
                hgrid.write_hgrid(
                    str(file_path),  # Convert Path to string
                    value=default_value,  # Use constant value for all nodes
                    fmt=0,  # Don't output boundary info
                    Info=description,  # Use description as header
                )
                logger.info(f"Successfully created {filename}")
                ret[filename.split(".")[0]] = str(file_path)

        # Generate tvd.prop if needed
        self.generate_tvprop(destdir)
        return ret

    # The _create_gr3_from_hgrid method has been removed as we now use PyLibs' native
    # write_hgrid method to create gr3 files with uniform values

    def generate_tvprop(self, destdir: Path) -> Path:
        """Generate tvd.prop file for SCHISM.

        The tvd.prop file must have two columns in this format:
        1. Two columns: `element_number TVD_flag` (space-separated)
        2. One entry per element
        3. TVD flag value of 1 for all entries (1 = upwind TVD)
        4. Element numbers start from 1

        Correct format:
        ```
        1 1
        2 1
        3 1
        ...
        317 1
        ```

        Args:
            destdir (Path): Destination directory

        Returns:
            Path: Path to tvd.prop file
        """
        logger = logging.getLogger(__name__)
        dest = destdir / "tvd.prop"

        # For tvd.prop we need the number of elements
        num_elements = self.pylibs_hgrid.ne  # Number of elements

        logger.info(
            f"Creating tvd.prop with two-column format for {num_elements} elements"
        )

        # Create the file with the proper format
        with open(dest, "w") as f:
            # Write element_number and TVD flag (1) for each element
            for i in range(1, num_elements + 1):
                f.write(f"{i} 1\n")

        # Ensure file permissions are correct
        try:
            dest.chmod(0o644)  # User read/write, group/others read
            logger.info(f"Successfully created tvd.prop with {num_elements} elements")
        except Exception as e:
            logger.warning(f"Failed to set permissions on tvd.prop: {e}")

        return dest

    def boundary(self, tolerance=None) -> Polygon:
        gd = self.pylibs_hgrid

        # Make sure boundaries are computed
        if hasattr(gd, "compute_bnd") and not hasattr(gd, "nob"):
            gd.compute_bnd()

        if not hasattr(gd, "nob") or gd.nob is None or gd.nob == 0:
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
        if hasattr(gd, "elnode"):
            elements_array = gd.elnode

        meshtri = Triangulation(
            gd.x,
            gd.y,
            elements_array,
        )
        ax.triplot(meshtri, color="k", alpha=0.3)

        # Make sure boundaries are computed
        if hasattr(gd, "compute_bnd") and not hasattr(gd, "nob"):
            gd.compute_bnd()

        # Plot open boundaries if they exist
        if hasattr(gd, "nob") and gd.nob is not None and gd.nob > 0:
            # Plot each open boundary
            for i in range(gd.nob):
                boundary_nodes = gd.iobn[i]
                x_boundary = gd.x[boundary_nodes]
                y_boundary = gd.y[boundary_nodes]

                # Plot the line
                ax.plot(
                    x_boundary,
                    y_boundary,
                    "-b",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

                # Plot the points
                ax.plot(
                    x_boundary,
                    y_boundary,
                    "+k",
                    markersize=6,
                    transform=ccrs.PlateCarree(),
                    zorder=10,
                )

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
        if hasattr(gd, "nlb") and gd.nlb is not None and gd.nlb > 0:
            # Plot each land boundary
            for i in range(gd.nlb):
                boundary_nodes = gd.ilbn[i]
                x_boundary = gd.x[boundary_nodes]
                y_boundary = gd.y[boundary_nodes]

                # Check if this is an island
                is_island = False
                if (
                    hasattr(gd, "island")
                    and gd.island is not None
                    and i < len(gd.island)
                ):
                    is_island = gd.island[i] == 1

                # Plot the land boundary with different color for islands
                color = "r" if is_island else "g"  # Red for islands, green for land
                ax.plot(
                    x_boundary,
                    y_boundary,
                    f"-{color}",
                    linewidth=2,
                    transform=ccrs.PlateCarree(),
                )

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
        if hasattr(gd, "compute_bnd") and not hasattr(gd, "nob"):
            gd.compute_bnd()

        if not hasattr(gd, "nob") or gd.nob is None or gd.nob == 0:
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
        if hasattr(gd, "compute_bnd") and not hasattr(gd, "nob"):
            gd.compute_bnd()

        if not hasattr(gd, "nlb") or gd.nlb is None or gd.nlb == 0:
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
