"""
SCHISM Vertical Grid Module

This module provides a unified interface for creating SCHISM vertical grid files
that aligns with the PyLibs API.
"""

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

from rompy.core.logging import get_logger

logger = get_logger(__name__)


class VGrid(BaseModel):
    """
    Base class for SCHISM vertical grid generation.
    This class directly mirrors the PyLibs create_schism_vgrid API.
    """

    model_type: Literal["schism_vgrid"] = Field(
        "schism_vgrid", description="Model discriminator"
    )

    # Type of vertical coordinate: 1=LSC2, 2=SZ
    ivcor: int = Field(default=1, description="Vertical coordinate type (1=LSC2, 2=SZ)")

    # Number of vertical layers
    nvrt: int = Field(default=2, description="Number of vertical layers")

    # Z levels or transition depth (h_s) if a single number
    zlevels: Union[List[float], float] = Field(
        default=-1.0e6, description="Z levels or transition depth (h_s)"
    )

    # Parameters for SZ coordinate (used when ivcor=2)
    h_c: float = Field(default=10.0, description="Critical depth for SZ coordinate")
    theta_b: float = Field(default=0.5, description="Bottom theta parameter for SZ")
    theta_f: float = Field(default=1.0, description="Surface theta parameter for SZ")

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
        # TODO cleanup
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
                raise ValueError("LSC2 grid (ivcor=1) not supported in pylibs")
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
            logger.info(f"Successfully used create_schism_vgrid to create {vgrid_path}")
            return vgrid_path

        except Exception as e:
            logger.error(f"Error creating vgrid.in: {e}")
            raise

    def get(self, destdir: Union[str, Path]) -> Path:
        """Compatitibilty helper function"""
        return self.generate(destdir)

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
