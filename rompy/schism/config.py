import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import ConfigDict, Field, model_serializer, model_validator

from rompy.core.config import BaseConfig
from rompy.core.data import DataBlob
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel, Spectrum

from .config_legacy import SchismCSIROConfig as _LegacySchismCSIROConfig
# Import plotting functions
from .config_plotting import plot_sflux_spatial, plot_sflux_timeseries
from .config_plotting_boundary import (plot_boundary_points,
                                       plot_boundary_profile,
                                       plot_boundary_timeseries)
from .config_plotting_tides import (plot_tidal_boundaries, plot_tidal_dataset,
                                    plot_tidal_rose, plot_tidal_stations)
from .data import SCHISMData
from .grid import SCHISMGrid
from .interface import TimeInterface
from .namelists import NML
from .namelists.param import Param

logger = logging.getLogger(__name__)

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

    # Add data visualization methods
    # Atmospheric (sflux) plotting
    plot_sflux_spatial = plot_sflux_spatial
    plot_sflux_timeseries = plot_sflux_timeseries

    # Boundary data plotting
    plot_boundary_points = plot_boundary_points
    plot_boundary_timeseries = plot_boundary_timeseries
    plot_boundary_profile = plot_boundary_profile

    # Tidal data plotting
    plot_tidal_boundaries = plot_tidal_boundaries
    plot_tidal_stations = plot_tidal_stations
    plot_tidal_rose = plot_tidal_rose
    plot_tidal_dataset = plot_tidal_dataset

    def __call__(self, runtime) -> str:
        logger = logging.getLogger(__name__)

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
