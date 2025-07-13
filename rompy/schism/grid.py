from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import (
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_serializer,
    model_validator,
)
from pylib import (
    compute_zcor,
    create_schism_vgrid,
    read_schism_hgrid,
    read_schism_vgrid,
    save_schism_grid,
    schism_grid,
)
from shapely.geometry import MultiPoint, Polygon

from rompy.core.data import DataBlob
from rompy.core.grid import BaseGrid
from rompy.core.logging import get_logger
from rompy.core.types import RompyBaseModel

from .vgrid import VGrid, create_2d_vgrid

logger = get_logger(__name__)


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
    model_type: Literal["gr3_generator"] = Field(
        "gr3_generator", description="Model discriminator"
    )
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    gr3_type: str = Field(
        ...,
        description="Type of gr3 file. Must be one of 'albedo', 'diffmin', 'diffmax', 'watertype', 'windrot_geo2proj'",
    )
    value: float = Field(None, description="Constant value to set in gr3 file")
    crs: str = Field("epsg:4326", description="Coordinate reference system")

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        """Add discriminator to JSON schema"""
        schema = super().model_json_schema(*args, **kwargs)
        schema["discriminator"] = {"propertyName": "model_type"}
        return schema

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to ensure proper serialization when used in other models."""
        result = {}
        result["model_type"] = self.model_type

        # Explicitly include the value field to ensure it's correctly serialized
        if hasattr(self, "value") and self.value is not None:
            result["value"] = self.value

        # Include other non-private fields
        for field_name in self.model_fields:
            if (
                not field_name.startswith("_")
                and getattr(self, field_name, None) is not None
            ):
                if field_name not in ["value", "model_type"]:  # Already handled above
                    result[field_name] = getattr(self, field_name)

        # Remove any private attributes
        for key in list(result.keys()):
            if key.startswith("_"):
                del result[key]

        return result

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
    model_type: Literal["vgridgenerator"] = Field(
        "vgridgenerator", description="Model discriminator"
    )
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
        dest_path = Path(destdir) / "vgrid.in"
        logger.info(
            f"Generating vgrid.in at {dest_path} using unified VGrid implementation"
        )

        vgrid = self._create_vgrid_instance()
        return vgrid.generate(destdir)

    def _create_vgrid_instance(self) -> "VGrid":
        """Create the appropriate VGrid instance based on configuration."""
        from rompy.schism.vgrid import VGrid

        if self.vgrid_type.lower() == "2d":
            return create_2d_vgrid()
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


class WWMBNDGR3Generator(GeneratorBase):
    model_type: Literal["wwmbnd_generator"] = Field(
        "wwmbnd_generator", description="Model discriminator"
    )
    hgrid: DataBlob | Path = Field(..., description="Path to hgrid.gr3 file")
    bcflags: list[int] = Field(
        None,
        description="List of boundary condition flags. This replicates the functionality of the gen_wwmbnd.in file. Must be the same length as the number of open boundaries in the hgrid.gr3 file. If not specified, it is assumed that all open hgrid files are open to waves",
    )

    @classmethod
    def model_json_schema(cls, *args, **kwargs):
        """Add discriminator to JSON schema"""
        schema = super().model_json_schema(*args, **kwargs)
        schema["discriminator"] = {"propertyName": "model_type"}
        return schema

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to ensure proper serialization when used in other models."""
        result = {}
        result["model_type"] = self.model_type

        # Include other non-private fields
        for field_name in self.model_fields:
            if (
                not field_name.startswith("_")
                and field_name != "model_type"
                and getattr(self, field_name, None) is not None
            ):
                result[field_name] = getattr(self, field_name)

        return result

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
    vgrid: Optional[DataBlob | VgridGenerator | VGrid] = Field(
        description="Path to vgrid.in file",
        default_factory=create_2d_vgrid,
    )

    @model_validator(mode="after")
    def validate_gr3_fields(self):
        """Custom validator to handle GR3Generator conversion during deserialization."""
        # Convert GR3Generator fields that might have been serialized as simple values
        gr3_fields = [
            "drag",
            "diffmin",
            "diffmax",
            "albedo",
            "watertype",
            "windrot_geo2proj",
        ]

        for field in gr3_fields:
            value = getattr(self, field, None)
            if (
                value is not None
                and not isinstance(value, (DataBlob, GR3Generator))
                and isinstance(value, (int, float))
            ):
                # Create a GR3Generator instance instead of using the raw value
                gr3_generator = GR3Generator(
                    hgrid=self.hgrid,
                    gr3_type=field,
                    value=value,
                )
                setattr(self, field, gr3_generator)

        return self

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to handle proper serialization."""
        result = {}

        # Add fields to the result dictionary
        for field_name in self.model_fields:
            value = getattr(self, field_name, None)

            # Special handling for GR3Generator fields
            if value is not None and isinstance(value, GR3Generator):
                # For GR3Generator objects, just use the value field
                result[field_name] = value.value
            # Skip wwmbnd field if it's a WWMBNDGR3Generator (similar to TimeRange behavior in memory)
            elif (
                field_name == "wwmbnd"
                and value is not None
                and isinstance(value, WWMBNDGR3Generator)
            ):
                # Skip this field to prevent validation errors as it's complex to serialize/deserialize
                pass
            elif value is not None and not field_name.startswith("_"):
                result[field_name] = value

        return result

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
    def ne(self) -> int:
        return self.pylibs_hgrid.ne

    @property
    def np(self) -> int:
        return self.pylibs_hgrid.np

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

    @property
    def nob(self):
        if not hasattr(self.pylibs_hgrid, "nobn"):
            self.pylibs_hgrid.compute_bnd()
        return self.pylibs_hgrid.nob

    @property
    def nobn(self):
        if not hasattr(self.pylibs_hgrid, "nobn"):
            self.pylibs_hgrid.compute_bnd()
        return self.pylibs_hgrid.nobn

    @property
    def nvrt(self):
        if self.is_3d:
            return self.pylibs_vgrid.nvrt
        else:
            return None

    def copy_to(self, destdir: Path) -> "SCHISMGrid":
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
        from rompy.formatting import ARROW

        ret = {}
        dest_path = (
            Path(destdir) if isinstance(destdir, (str, Path)) else Path(str(destdir))
        )

        # Ensure the output directory exists
        if not dest_path.exists():
            dest_path.mkdir(parents=True, exist_ok=True)

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

        # Generate vertical grid
        logger.info(f"{ARROW} Generating vertical grid configuration")
        ret["vgrid"] = self.vgrid.get(destdir)

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

    def plot_bnd(self, ax=None, add_coastlines=True, **kwargs):

        # Make sure boundaries are computed if needed
        if hasattr(self.pylibs_hgrid, "compute_bnd") and not hasattr(
            self.pylibs_hgrid, "nob"
        ):
            self.pylibs_hgrid.compute_bnd()

        # Use the native pylibs plotting function
        return self.plot(fmt=3, **kwargs)

    def plot_grid(self, ax=None, add_coastlines=True, **kwargs):
        """
        Plot just the grid triangulation.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure is created.
        add_coastlines : bool, optional:w

            Whether to add coastlines to the plot (requires cartopy).
        **kwargs : dict
            Additional keyword arguments to pass to the pylibs plot functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.plot(fmt=0, **kwargs)

    def plot_bathymetry(self, ax=None, add_coastlines=True, **kwargs):
        """
        Plot filled contours of depth/bathymetry.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure is created.
        add_coastlines : bool, optional
            Whether to add coastlines to the plot (requires cartopy).
        **kwargs : dict
            Additional keyword arguments to pass to the pylibs plot functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.plot(fmt=1, **kwargs)

    def plot_contours(self, ax=None, add_coastlines=True, **kwargs):
        """
        Plot contour lines of depth/bathymetry.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure is created.
        add_coastlines : bool, optional
            Whether to add coastlines to the plot (requires cartopy).
        **kwargs : dict
            Additional keyword arguments to pass to the pylibs plot functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.
        """
        return self.plot(fmt=2, **kwargs)

    def plot(self, ax=None, plot_type="domain", add_coastlines=True, **kwargs):
        """
        Plot the SCHISM grid using native pylibs plotting functionality.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            The axes to plot on. If None, a new figure is created.
        plot_type : str, optional
            Type of plot to create. Options are:
            - 'domain': Plot the full domain with boundaries (default)
            - 'grid': Plot just the grid triangulation
            - 'bnd': Plot just the boundaries
        add_coastlines : bool, optional
            Whether to add coastlines to the plot (requires cartopy).
        fmt : int, optional
            Plotting format. Options are:
            - 0: Plot grid only (default)
            - 1: Plot filled contours of depth/bathymetry
            - 2: Plot contour lines of depth/bathymetry
            - 3: Plot boundaries only
        value : numpy.ndarray, optional
            Color values for plotting. If None, grid depth is used.
        levels : int or array-like, optional
            If int, number of contour levels to use (default 51).
            If array-like, specific levels to plot.
        clim : [min, max], optional
            Value range for plot/colorbar. If None, determined from data.
        cmap : str, optional
            Colormap to use for depth visualization (default 'jet').
        cb : bool, optional
            Whether to add a colorbar to the plot (default True).
        **kwargs : dict
            Additional keyword arguments to pass to the pylibs plot functions.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object.
        ax : matplotlib.axes.Axes
            The axes object.

        Examples
        --------
        # Plot grid with bathymetry as filled contours
        >>> grid.plot(fmt=1, cmap='viridis', levels=20)

        # Plot specific depth contours with custom range
        >>> grid.plot(fmt=2, clim=[-100, 0], levels=[-100, -50, -20, -10, 0])
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig = plt.figure(figsize=(12, 10))
            try:
                # Try to create a cartopy axis if available
                from cartopy import crs as ccrs

                ax = fig.add_subplot(111, projection=ccrs.PlateCarree())
                if add_coastlines:
                    ax.coastlines()
                    ax.gridlines(draw_labels=True)
            except ImportError:
                # Fall back to regular axis if cartopy isn't available
                ax = fig.add_subplot(111)
        else:
            fig = plt.gcf()
        self.pylibs_hgrid.plot(**kwargs)
        self.pylibs_hgrid.plot_bnd()
        return fig, ax

    def plot_hgrid(self, figsize=(20, 10)):
        """
        Create a comprehensive two-panel visualization of the SCHISM grid.

        Left panel shows bathymetry/depth, right panel shows the mesh and boundaries.

        Parameters
        ----------
        figsize : tuple, optional
            Size of the figure (width, height). Default is (20, 10).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing both panels.
        (ax1, ax2) : tuple of matplotlib.axes.Axes
            The axes objects for the bathymetry and mesh panels.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.tri import Triangulation

        # Create figure with two subplots
        fig = plt.figure(figsize=figsize)

        # Left panel: Bathymetry
        try:
            from cartopy import crs as ccrs

            ax1 = fig.add_subplot(121, projection=ccrs.PlateCarree())
            ax1.coastlines()
        except ImportError:
            ax1 = fig.add_subplot(121)

        # Plot bathymetry using pylibs
        gd = self.pylibs_hgrid
        tri = Triangulation(gd.x, gd.y, triangles=gd.i34)
        depth = -gd.dp  # Convert to positive for depth
        cs = ax1.tricontourf(tri, depth, cmap="viridis")
        plt.colorbar(cs, ax=ax1, label="Depth (m)")
        ax1.set_title("Bathymetry")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")

        # Right panel: Mesh and boundaries
        try:
            ax2 = fig.add_subplot(122, projection=ccrs.PlateCarree())
            ax2.coastlines()
        except ImportError:
            ax2 = fig.add_subplot(122)

        # Use the main plot method for the mesh
        _, ax2 = self.plot(ax=ax2, plot_type="domain", linewidth=0.5, color="gray")
        ax2.set_title("Grid Mesh and Boundaries")
        ax2.set_xlabel("Longitude")
        ax2.set_ylabel("Latitude")

        plt.tight_layout()
        return fig, (ax1, ax2)

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

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to handle optional fields properly during serialization."""
        # Start with all fields that have values
        result = {
            field_name: getattr(self, field_name)
            for field_name in self.model_fields
            if getattr(self, field_name, None) is not None
            or field_name in ["grid_type", "hgrid", "crs"]
        }

        # Remove private attributes that shouldn't be in the serialized output
        for key in list(result.keys()):
            if key.startswith("_"):
                del result[key]

        return result

    def _format_value(self, obj):
        """Custom formatter for SCHISMGrid values.

        This method provides special formatting for specific types used in
        SCHISMGrid such as grid components and data sources.

        Args:
            obj: The object to format

        Returns:
            A formatted string or None to use default formatting
        """
        # Import specific types and formatting utilities
        from rompy.core.logging import LoggingConfig
        from rompy.formatting import get_formatted_header_footer

        # Get ASCII mode setting from LoggingConfig
        logging_config = LoggingConfig()
        USE_ASCII_ONLY = logging_config.use_ascii

        # Format SCHISMGrid (self-formatting)
        if isinstance(obj, SCHISMGrid):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM GRID CONFIGURATION", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Add horizontal grid information
            if hasattr(obj, "hgrid") and obj.hgrid is not None:
                hgrid_path = str(obj.hgrid.uri if hasattr(obj.hgrid, "uri") else obj.hgrid)
                if len(hgrid_path) > 50:
                    hgrid_path = "..." + hgrid_path[-47:]
                lines.append(f"  {bullet} Horizontal Grid: {hgrid_path}")

                # Try to get grid statistics
                try:
                    np_points = obj.np
                    ne_elements = obj.ne
                    lines.append(f"      Points: {np_points}, Elements: {ne_elements}")
                except:
                    pass

            # Add vertical grid information
            if hasattr(obj, "vgrid") and obj.vgrid is not None:
                vgrid_type = type(obj.vgrid).__name__
                lines.append(f"  {bullet} Vertical Grid: {vgrid_type}")

                # Add 2D/3D information
                try:
                    is_3d = obj.is_3d
                    grid_dim = "3D" if is_3d else "2D"
                    lines.append(f"      Dimension: {grid_dim}")

                    if is_3d and hasattr(obj, "nvrt") and obj.nvrt is not None:
                        lines.append(f"      Vertical levels: {obj.nvrt}")
                except:
                    pass

            # Add boundary information
            try:
                nob = obj.nob
                if nob and nob > 0:
                    lines.append(f"  {bullet} Open Boundaries: {nob}")
            except:
                pass

            # Add friction information
            friction_types = []
            if hasattr(obj, "drag") and obj.drag is not None:
                friction_types.append("Drag")
            if hasattr(obj, "rough") and obj.rough is not None:
                friction_types.append("Roughness")
            if hasattr(obj, "manning") and obj.manning is not None:
                friction_types.append("Manning")

            if friction_types:
                lines.append(f"  {bullet} Friction: {', '.join(friction_types)}")

            # Add other grid files
            other_files = []
            for attr in ["diffmin", "diffmax", "albedo", "watertype", "windrot_geo2proj"]:
                if hasattr(obj, attr) and getattr(obj, attr) is not None:
                    other_files.append(attr)

            if other_files:
                lines.append(f"  {bullet} Additional files: {', '.join(other_files)}")

            # Add coordinate system
            if hasattr(obj, "crs") and obj.crs is not None:
                lines.append(f"  {bullet} CRS: {obj.crs}")

            lines.append(footer)
            return "\n".join(lines)

        # Format GR3Generator
        if isinstance(obj, GR3Generator):
            header, footer, _ = get_formatted_header_footer(
                title="GR3 GENERATOR", use_ascii=USE_ASCII_ONLY
            )

            gr3_type = getattr(obj, "gr3_type", "unknown")
            value = getattr(obj, "value", "unknown")

            return (
                f"{header}\n"
                f"  Type:  {gr3_type}\n"
                f"  Value: {value}\n"
                f"{footer}"
            )

        # Format VgridGenerator
        if isinstance(obj, VgridGenerator):
            header, footer, bullet = get_formatted_header_footer(
                title="VGRID GENERATOR", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            if hasattr(obj, "vgrid_type"):
                lines.append(f"  {bullet} Type: {obj.vgrid_type}")

            if hasattr(obj, "nlayer") and obj.nlayer is not None:
                lines.append(f"  {bullet} Layers: {obj.nlayer}")

            lines.append(footer)
            return "\n".join(lines)

        # Use the new formatting framework
        from rompy.formatting import format_value
        return format_value(obj)


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
