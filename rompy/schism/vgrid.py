"""
SCHISM Vertical Grid Module

This module provides a unified interface for creating SCHISM vertical grid files
that aligns with the PyLibs API.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

# Try importing from pylibs or pylib depending on what's available
try:
    from pylibs.schism_file import create_schism_vgrid
except ImportError:
    try:
        from pylib import create_schism_vgrid
    except ImportError:
        # Will handle this gracefully in the implementation
        pass

logger = logging.getLogger(__name__)


class VGrid(BaseModel):
    """
    Base class for SCHISM vertical grid generation.
    This class directly mirrors the PyLibs create_schism_vgrid API.
    """

    # Type of vertical coordinate: 1=LSC2, 2=SZ
    ivcor: int = Field(default=1, description="Vertical coordinate type (1=LSC2, 2=SZ)")

    # Number of vertical layers
    nvrt: int = Field(default=2, description="Number of vertical layers")

    # Z levels or transition depth (h_s) if a single number
    zlevels: Union[List[float], float, np.ndarray] = Field(
        default=-1.0e6, description="Z levels or transition depth (h_s)"
    )

    # Parameters for SZ coordinate (used when ivcor=2)
    h_c: float = Field(default=10.0, description="Critical depth for SZ coordinate")
    theta_b: float = Field(default=0.5, description="Bottom theta parameter for SZ")
    theta_f: float = Field(default=1.0, description="Surface theta parameter for SZ")

    class Config:
        arbitrary_types_allowed = True

    def generate(self, destdir: Union[str, Path]) -> Path:
        """
        Generate vgrid.in file in the specified output directory.

        Parameters
        ----------
        destdir : str or Path
            Directory where vgrid.in will be created

        Returns
        -------
        Path
            Path to the created vgrid.in file
        """
        destdir = Path(destdir)
        destdir.mkdir(parents=True, exist_ok=True)

        vgrid_path = destdir / "vgrid.in"

        # Convert zlevels to numpy array if it's a single value or list
        if isinstance(self.zlevels, (float, int)):
            zlevels_param = float(self.zlevels)
        elif isinstance(self.zlevels, list):
            zlevels_param = np.array(self.zlevels)
        else:
            zlevels_param = self.zlevels

        try:
            logger.info(
                f"Creating vgrid.in with ivcor={self.ivcor}, nvrt={self.nvrt}, "
                f"zlevels={zlevels_param}, h_c={self.h_c}, theta_b={self.theta_b}, "
                f"theta_f={self.theta_f}"
            )

            # LSC2 (ivcor=1) is not yet supported by PyLibs' create_schism_vgrid
            # Always use manual creation for LSC2
            if self.ivcor == 1:
                logger.info(
                    "Using manual creation for LSC2 grid (ivcor=1) as PyLibs doesn't support it yet"
                )
                self._create_manually(vgrid_path)
            else:  # For SZ and other types, try to use PyLibs first
                try:
                    # Check if we have the create_schism_vgrid function
                    if "create_schism_vgrid" in globals():
                        # Call PyLibs function with our parameters
                        create_schism_vgrid(
                            fname=str(vgrid_path),
                            ivcor=self.ivcor,
                            nvrt=self.nvrt,
                            zlevels=zlevels_param,
                            h_c=self.h_c,
                            theta_b=self.theta_b,
                            theta_f=self.theta_f,
                        )
                        logger.info(
                            f"Successfully used create_schism_vgrid to create {vgrid_path}"
                        )
                    else:
                        # Fallback to manual creation
                        logger.warning(
                            "create_schism_vgrid function not available, using fallback method"
                        )
                        self._create_manually(vgrid_path)
                except Exception as e:
                    logger.warning(
                        f"Error using create_schism_vgrid: {e}, falling back to manual method"
                    )
                    self._create_manually(vgrid_path)

            # Verify the file was created
            if vgrid_path.exists():
                logger.info(f"Successfully created vgrid.in at {vgrid_path}")
                with open(vgrid_path, "r") as f:
                    content = f.read(100)  # Read first 100 chars for logging
                    logger.debug(f"vgrid.in content begins with: {content}...")
                return vgrid_path
            else:
                raise FileNotFoundError(f"vgrid.in was not created at {vgrid_path}")

        except Exception as e:
            logger.error(f"Error creating vgrid.in: {e}")
            raise

    def _create_manually(self, vgrid_path: Path) -> None:
        """Create vgrid.in file manually as a fallback method.

        SCHISM vgrid.in format for LSC2 (ivcor=1):
            Line 1: 1 (ivcor value)
            Line 2: nvrt kz h_s (e.g., "2 1 1000000.0")
            Line 3: "Z levels"
            Line 4: "1 -1000000.0"
            Line 5: "S levels"
            Line 6+: "i sigma_value" for each level i=2...nvrt

        SCHISM vgrid.in format for SZ (ivcor=2):
            Line 1: 2 (ivcor value)
            Line 2: nvrt (e.g., "10")
            Line 3: h_c theta_b theta_f (e.g., "10.0 0.5 1.0")
            Line 4+: "i sigma_value" for each level i=1...nvrt
        """
        logger.info(f"Creating vgrid.in manually at {vgrid_path}")

        with open(vgrid_path, "w") as f:
            if self.ivcor == 1:  # LSC2
                # Use LSC2 format with strict formatting for Fortran readability
                f.write("1\n")

                # Calculate h_s value
                h_s = (
                    float(self.zlevels)
                    if isinstance(self.zlevels, (float, int))
                    else 1000000.0
                )
                f.write(f"{self.nvrt} 1 {h_s}\n")

                f.write("Z levels\n")
                f.write("1 -1000000.0\n")
                f.write("S levels\n")

                # Add S levels
                for i in range(2, self.nvrt + 1):
                    sigma = 1.0 - (i - 2) / max(
                        1, self.nvrt - 2
                    )  # Linear distribution from 1.0 to 0.0
                    f.write(f"{i} {sigma:.6f}\n")
            else:  # SZ
                # Use SZ format with strict formatting for Fortran readability
                f.write("2\n")
                f.write(f"{self.nvrt}\n")
                f.write(f"{self.h_c} {self.theta_b} {self.theta_f}\n")

                # Add levels with strict formatting
                for i in range(1, self.nvrt + 1):
                    sigma = 1.0 - (i - 1) / max(
                        1, self.nvrt - 1
                    )  # Linear distribution from 1.0 to 0.0
                    f.write(f"{i} {sigma:.6f}\n")

        logger.info(f"Manually created vgrid.in at {vgrid_path}")

    @classmethod
    def create_lsc2(cls, nvrt: int = 2, h_s: float = -1.0e6) -> "VGrid":
        """
        Create an LSC2 vertical grid configuration.

        Parameters
        ----------
        nvrt : int, optional
            Number of vertical layers, by default 2 for 2D model
        h_s : float, optional
            Transition depth, by default -1.0e6 (very deep)

        Returns
        -------
        VGrid
            Configured VGrid instance
        """
        return cls(ivcor=1, nvrt=nvrt, zlevels=h_s)  # LSC2

    @classmethod
    def create_sz(
        cls,
        nvrt: int = 10,
        h_c: float = 10.0,
        theta_b: float = 0.5,
        theta_f: float = 1.0,
        zlevels: Optional[List[float]] = None,
    ) -> "VGrid":
        """
        Create an SZ vertical grid configuration.

        Parameters
        ----------
        nvrt : int, optional
            Number of vertical layers, by default 10
        h_c : float, optional
            Critical depth, by default 10.0
        theta_b : float, optional
            Bottom theta parameter, by default 0.5
        theta_f : float, optional
            Surface theta parameter, by default 1.0
        zlevels : list of float, optional
            Z levels, by default None (will use default in PyLibs)

        Returns
        -------
        VGrid
            Configured VGrid instance
        """
        return cls(
            ivcor=2,  # SZ
            nvrt=nvrt,
            h_c=h_c,
            theta_b=theta_b,
            theta_f=theta_f,
            zlevels=zlevels if zlevels is not None else -1.0e6,
        )


# For 2D models, we primarily use LSC2 with 2 layers
def create_2d_vgrid() -> VGrid:
    """Create a standard 2D vertical grid configuration."""
    return VGrid.create_sz(nvrt=2, h_c=40, theta_b=0.5, theta_f=1)
