from pathlib import Path
from typing import Any, Literal, Optional, Union
import warnings

from pydantic import ConfigDict, Field, model_serializer, model_validator

from rompy.core.config import BaseConfig
from rompy.core.data import DataBlob
from rompy.core.logging import get_logger
from rompy.core.time import TimeRange
from rompy.core.types import RompyBaseModel, Spectrum

from .config_legacy import SchismCSIROConfig as _LegacySchismCSIROConfig

# Import plotting functions
from .config_plotting import plot_sflux_spatial, plot_sflux_timeseries
from .config_plotting_boundary import (
    plot_boundary_points,
    plot_boundary_profile,
    plot_boundary_timeseries,
)
from .config_plotting_tides import (
    plot_tidal_boundaries,
    plot_tidal_dataset,
    plot_tidal_rose,
    plot_tidal_stations,
)
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
            and getattr(self.data, "hotstart", None) is not None
            and self.nml is not None
            and self.nml.param is not None
            and self.nml.param.opt is not None
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

        logger.info(f"Generating grid files using {type(self.grid).__name__}")
        self.grid.get(runtime.staging_dir)

        if self.data is not None and self.nml is not None:
            self.nml.update_data_sources(
                self.data.get(
                    destdir=runtime.staging_dir, grid=self.grid, time=runtime.period
                )
            )
        if self.nml is not None:
            self.nml.update_times(period=runtime.period)
            self.nml.write_nml(runtime.staging_dir)

        return str(runtime.staging_dir)

    def _format_value(self, obj):
        """Custom formatter for SCHISMConfig values.

        This method provides special formatting for specific types used in
        SCHISMConfig such as grid, data, and namelist components.

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

        # Format SCHISMConfig (self-formatting)
        if isinstance(obj, SCHISMConfig):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM MODEL CONFIGURATION", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Add grid information
            if obj.grid is not None:
                grid_type = type(obj.grid).__name__
                lines.append(f"  {bullet} Grid: {grid_type}")

                # Try to get grid details
                if hasattr(obj.grid, "hgrid"):
                    hgrid = obj.grid.hgrid
                    hgrid_path = str(getattr(hgrid, "uri", getattr(hgrid, "source", hgrid)))
                    if len(hgrid_path) > 50:
                        hgrid_path = "..." + hgrid_path[-47:]
                    lines.append(f"      Horizontal grid: {hgrid_path}")

                if hasattr(obj.grid, "vgrid") and obj.grid.vgrid is not None:
                    vgrid_type = type(obj.grid.vgrid).__name__
                    lines.append(f"      Vertical grid: {vgrid_type}")

            # Add data information
            if obj.data is not None:
                data_type = type(obj.data).__name__
                lines.append(f"  {bullet} Data: {data_type}")

                # Count data components
                data_components = []
                if hasattr(obj.data, "atmos") and obj.data.atmos is not None:
                    data_components.append("Atmospheric")
                if hasattr(obj.data, "wave") and obj.data.wave is not None:
                    data_components.append("Wave")
                if hasattr(obj.data, "boundary_conditions") and obj.data.boundary_conditions is not None:
                    data_components.append("Boundary Conditions")

                if data_components:
                    lines.append(f"      Components: {', '.join(data_components)}")

            # Add namelist information
            if obj.nml is not None:
                nml_type = type(obj.nml).__name__
                lines.append(f"  {bullet} Namelist: {nml_type}")

                # Count active namelist components
                nml_components = []
                if hasattr(obj.nml, "param") and obj.nml.param is not None:
                    nml_components.append("Parameters")
                if hasattr(obj.nml, "ice") and obj.nml.ice is not None:
                    nml_components.append("Ice")
                if hasattr(obj.nml, "icm") and obj.nml.icm is not None:
                    nml_components.append("ICM")
                if hasattr(obj.nml, "sediment") and obj.nml.sediment is not None:
                    nml_components.append("Sediment")
                if hasattr(obj.nml, "wwminput") and obj.nml.wwminput is not None:
                    nml_components.append("Wave")

                if nml_components:
                    lines.append(f"      Active modules: {', '.join(nml_components)}")

            # Add template information
            if obj.template is not None:
                template_path = obj.template
                if len(template_path) > 50:
                    template_path = "..." + template_path[-47:]
                lines.append(f"  {bullet} Template: {template_path}")

            lines.append(footer)
            return "\n".join(lines)

        # Format SCHISMGrid
        from .grid import SCHISMGrid
        if isinstance(obj, SCHISMGrid):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM GRID", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            if hasattr(obj, "hgrid"):
                hgrid = obj.hgrid
                hgrid_path = str(getattr(hgrid, "uri", getattr(hgrid, "source", hgrid)))
                if len(hgrid_path) > 50:
                    hgrid_path = "..." + hgrid_path[-47:]
                lines.append(f"  {bullet} Horizontal Grid: {hgrid_path}")

            if hasattr(obj, "vgrid") and obj.vgrid is not None:
                vgrid_type = type(obj.vgrid).__name__
                lines.append(f"  {bullet} Vertical Grid: {vgrid_type}")

                # Try to get vertical grid details
                if hasattr(obj.vgrid, "nlayer") and hasattr(obj.vgrid, "nlayer"):
                    nlayer = getattr(obj.vgrid, "nlayer", None)
                    if nlayer is not None:
                        lines.append(f"      Layers: {nlayer}")

            lines.append(footer)
            return "\n".join(lines)

        # Format SCHISMData
        from .data import SCHISMData
        if isinstance(obj, SCHISMData):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM DATA", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Count and list data components
            components = {}
            if hasattr(obj, "atmos") and obj.atmos is not None:
                components["Atmospheric"] = type(obj.atmos).__name__
            if hasattr(obj, "wave") and obj.wave is not None:
                components["Wave"] = type(obj.wave).__name__
            if hasattr(obj, "boundary_conditions") and obj.boundary_conditions is not None:
                components["Boundary Conditions"] = type(obj.boundary_conditions).__name__

            for comp_name, comp_type in components.items():
                lines.append(f"  {bullet} {comp_name}: {comp_type}")

            if not components:
                lines.append(f"  {bullet} No data components configured")

            lines.append(footer)
            return "\n".join(lines)

        # Format NML
        from .namelists import NML
        if isinstance(obj, NML):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM NAMELIST", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # List active namelist components
            components = {}
            if hasattr(obj, "param") and obj.param is not None:
                components["Parameters"] = type(obj.param).__name__
            if hasattr(obj, "ice") and obj.ice is not None:
                components["Ice"] = type(obj.ice).__name__
            if hasattr(obj, "icm") and obj.icm is not None:
                components["ICM"] = type(obj.icm).__name__
            if hasattr(obj, "sediment") and obj.sediment is not None:
                components["Sediment"] = type(obj.sediment).__name__
            if hasattr(obj, "wwminput") and obj.wwminput is not None:
                components["Wave"] = type(obj.wwminput).__name__
            if hasattr(obj, "cosine") and obj.cosine is not None:
                components["CoSiNE"] = type(obj.cosine).__name__
            if hasattr(obj, "mice") and obj.mice is not None:
                components["MICE"] = type(obj.mice).__name__

            for comp_name, comp_type in components.items():
                lines.append(f"  {bullet} {comp_name}: {comp_type}")

            if not components:
                lines.append(f"  {bullet} No modules configured")

            lines.append(footer)
            return "\n".join(lines)

        # Use the new formatting framework
        from rompy.formatting import format_value
        return format_value(obj)


class SchismCSIROConfig(_LegacySchismCSIROConfig):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "The SchismCSIROMigrationConfig class from config.py is deprecated. "
        )
        super().__init__(*args, **kwargs)
