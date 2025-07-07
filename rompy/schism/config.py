from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import ConfigDict, Field, model_serializer, model_validator

from rompy.core.config import BaseConfig
from rompy.core.data import DataBlob
from rompy.core.logging import get_logger
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel, Spectrum

from .config_legacy import SchismCSIROConfig as _LegacySchismCSIROConfig

# Import new unified plotting infrastructure
from .plotting import SchismPlotter
from .data import SCHISMData
from .grid import SCHISMGrid
from .interface import TimeInterface
from .namelists import NML
from .namelists.param import Param

logger = get_logger(__name__)

HERE = Path(__file__).parent

SCHISM_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "schism")


class SCHISMConfig(BaseConfig):
    model_type: Literal["schism"] = Field(
        "schism", description="The model type for SCHISM."
    )
    grid: SCHISMGrid = Field(description="The model grid")
    data: Optional[SCHISMData] = Field(None, description="Model inputs")
    nml: Optional[NML] = Field(
        default_factory=lambda: NML(param=Param()), description="The namelist"
    )
    template: Optional[str] = Field(
        description="The path to the model template",
        default=SCHISM_TEMPLATE,
    )

    # add a validator that checks that nml.param.ihot is 1 if data.hotstart is not none
    @model_validator(mode="after")
    def check_hotstart(self):
        if (
            self.data is not None
            and hasattr(self.data, "hotstart")
            and self.data.hotstart is not None
        ):
            self.nml.param.opt.ihot = 1
        return self

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to handle proper serialization of nested components."""
        from rompy.schism.grid import GR3Generator

        result = {}

        # Explicitly handle required fields
        result["model_type"] = self.model_type

        # Handle grid separately to process GR3Generator objects
        if self.grid is not None:
            grid_dict = {}
            for field_name in self.grid.model_fields:
                value = getattr(self.grid, field_name, None)

                # Special handling for GR3Generator objects
                if value is not None and isinstance(value, GR3Generator):
                    # For GR3Generator objects, extract just the value field
                    grid_dict[field_name] = value.value
                elif value is not None and not field_name.startswith("_"):
                    grid_dict[field_name] = value

            result["grid"] = grid_dict

        # Add optional fields that are not None
        if self.data is not None:
            result["data"] = self.data

        if self.nml is not None:
            result["nml"] = self.nml

        if self.template is not None:
            result["template"] = self.template

        return result

    # Enable arbitrary types and validation from instances in Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    def _get_plotter(self):
        """Get SchismPlotter instance for this configuration."""
        return SchismPlotter(config=self)

    # Add data visualization methods using new plotting infrastructure
    # Atmospheric (sflux) plotting
    def plot_sflux_spatial(self, **kwargs):
        """Plot spatial distribution of atmospheric forcing data."""
        return self._get_plotter().plot_atmospheric_data(**kwargs)
    
    def plot_sflux_timeseries(self, **kwargs):
        """Plot time series of atmospheric data at a specific location."""
        return self._get_plotter().data_plotter.plot_atmospheric_timeseries(**kwargs)

    # Boundary data plotting
    def plot_boundary_points(self, **kwargs):
        """Plot boundary node points."""
        return self._get_plotter().plot_boundaries(**kwargs)
    
    def plot_boundary_timeseries(self, file_path, **kwargs):
        """Plot time series of boundary data."""
        return self._get_plotter().plot_boundary_data(file_path, **kwargs)
    
    def plot_boundary_profile(self, file_path, **kwargs):
        """Plot vertical profile of boundary data."""
        return self._get_plotter().plot_boundary_data(file_path, plot_type="profile", **kwargs)

    # Tidal data plotting
    def plot_tidal_boundaries(self, **kwargs):
        """Plot boundaries with tidal forcing."""
        return self._get_plotter().plot_tidal_data(**kwargs)
    
    def plot_tidal_stations(self, **kwargs):
        """Plot tidal stations."""
        return self._get_plotter().plot_tidal_data(**kwargs)
    
    def plot_tidal_rose(self, **kwargs):
        """Plot tidal rose diagram."""
        return self._get_plotter().plot_tidal_data(**kwargs)
    
    def plot_tidal_dataset(self, **kwargs):
        """Plot tidal dataset."""
        return self._get_plotter().plot_tidal_data(**kwargs)

    # Grid plotting methods
    def plot_grid(self, **kwargs):
        """Plot SCHISM grid."""
        return self._get_plotter().plot_grid(**kwargs)
    
    def plot_bathymetry(self, **kwargs):
        """Plot SCHISM bathymetry."""
        return self._get_plotter().plot_bathymetry(**kwargs)
    
    def plot_overview(self, **kwargs):
        """Create comprehensive overview plot of SCHISM model setup."""
        return self._get_plotter().plot_overview(**kwargs)
    
    def plot_gr3_file(self, file_path, **kwargs):
        """Plot .gr3 property files with appropriate colormaps."""
        return self._get_plotter().plot_gr3_file(file_path, **kwargs)
    
    def plot_bctides_file(self, file_path, **kwargs):
        """Plot bctides.in configuration file."""
        return self._get_plotter().plot_bctides_file(file_path, **kwargs)

    def __call__(self, runtime) -> str:

        logger.info(f"Generating grid files using {type(self.grid).__name__}")
        self.grid.get(runtime.staging_dir)

        if self.data is not None:
            self.nml.update_data_sources(
                self.data.get(
                    destdir=runtime.staging_dir, grid=self.grid, time=runtime.period
                )
            )
        self.nml.update_times(period=runtime.period)

        self.nml.write_nml(runtime.staging_dir)

        return str(runtime.staging_dir)


class SchismCSIROConfig(_LegacySchismCSIROConfig):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The SchismCSIROMigrationConfig class from config.py is deprecated. "
        )
        super().__init__(*args, **kwargs)
