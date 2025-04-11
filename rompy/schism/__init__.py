from .config import Inputs, SCHISMConfig, SchismCSIROConfig, SchismCSIROMigrationConfig
from .data import SCHISMDataOcean, SCHISMDataSflux, SCHISMDataWave

# Import grid implementations directly from the grid module
from .grid import SCHISMGrid

# Legacy compatibility imports
from importlib import import_module as _import_module
import warnings


# Add deprecation warning for the old grid module
class LegacyGridAccess:
    """Proxy for the legacy grid module to provide deprecation warnings."""

    def __getattr__(self, name):
        warnings.warn(
            "rompy.schism.grid module is deprecated. Use rompy.schism.SCHISMGrid, "
            "SchismHGrid, SchismVGrid directly instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if name == "SCHISMGrid":
            return SCHISMGrid
        return getattr(_import_module("rompy.schism.grid"), name)


# Expose the legacy module with deprecation warnings
grid = LegacyGridAccess()
