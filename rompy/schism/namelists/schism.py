import logging
from pathlib import Path
from typing import Optional

from pydantic import Field, model_serializer

from rompy.core.time import TimeRange
from rompy.schism.namelists.basemodel import NamelistBaseModel

from .cosine import Cosine
from .ice import Ice
from .icm import Icm
from .mice import Mice
from .param import Param
from .sediment import Sediment
from .wwminput import Wwminput

logger = logging.getLogger(__name__)


class NML(NamelistBaseModel):
    param: Optional[Param] = Field(description="Model paramaters", default=None)
    ice: Optional[Ice] = Field(description="Ice model parameters", default=None)
    icm: Optional[Icm] = Field(description="Icm model parameters", default=None)
    mice: Optional[Mice] = Field(description="Mice model parameters", default=None)
    sediment: Optional[Sediment] = Field(
        description="Sediment model parameters", default=None
    )
    cosine: Optional[Cosine] = Field(
        description="Cosine model parameters", default=None
    )
    wwminput: Optional[Wwminput] = Field(
        description="Wave model input parameters", default=None
    )

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to handle proper serialization of namelist components."""
        result = {}

        # Include only non-None fields in the serialized output
        for field_name in self.model_fields:
            value = getattr(self, field_name, None)
            if value is not None:
                # Ensure we're returning the model object, not a dict
                if hasattr(value, "model_dump"):
                    # This ensures we maintain the model instance for proper serialization
                    result[field_name] = value
                else:
                    result[field_name] = value

        return result

    def update_times(self, period=TimeRange):
        """
        This class is used to set consistent time parameters in a group component by
        redefining existing `times` component attribute based on the `period` field.

        """

        update = {
            "param": {
                "core": {
                    "rnday": period.duration.total_seconds() / 86400,
                },
                "opt": {
                    "start_year": period.start.year,
                    "start_month": period.start.month,
                    "start_day": period.start.day,
                    "start_hour": period.start.hour,
                },
            }
        }

        date_format = "%Y%m%d.%H%M%S"
        if hasattr(self, "wwminput"):  # TODO change this check to the actual flag value
            # TODO these are currently all the same, but they could be different
            update.update(
                {
                    "wwminput": {
                        "proc": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "wind": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "curr": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "walv": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "history": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "bouc": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "station": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                        "hotfile": {
                            "begtc": period.start.strftime(date_format),
                            "endtc": period.end.strftime(date_format),
                        },
                    }
                }
            )
        self.update(update)

    def update_data_sources(self, datasources: dict):
        """Update the data sources in the namelist based on rompy data preparation."""
        update = {}
        if ("wave" in datasources) and (datasources["wave"] is not None):
            if hasattr(
                self, "wwminput"
            ):  # TODO change this check to the actual flag value
                if self.wwminput.bouc is not None:
                    logger.warn(
                        "Overwriting existing wave data source specified in namelist with rompy generated data"
                    )
                update.update(
                    {
                        "wwminput": {
                            "bouc": {
                                "filewave": datasources["wave"].name,
                            },
                        }
                    }
                )
        if ("atmos" in datasources) and (datasources["atmos"] is not None):
            if self.param.opt.nws is not 2:
                logger.warn(
                    f"Overwriting param nws value of {self.param.opt.nws} to 2 to use rompy generated sflux data"
                )
                update.update(
                    {
                        "param": {
                            "opt": {"nws": 2},
                        }
                    }
                )
        self.update(update)

    def write_nml(self, workdir: Path):
        for nml in [
            "param",
            "ice",
            "icm",
            "mice",
            "sediment",
            "cosine",
            "wwminput",
        ]:
            attr = getattr(self, nml)
            if attr is not None:
                attr.write_nml(workdir)

    def _format_value(self, obj):
        """Custom formatter for NML values.

        This method provides special formatting for specific types used in
        NML such as namelist components and parameters.

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

        # Format NML (self-formatting)
        if isinstance(obj, NML):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM NAMELIST CONFIGURATION", use_ascii=USE_ASCII_ONLY
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
            if hasattr(obj, "mice") and obj.mice is not None:
                components["MICE"] = type(obj.mice).__name__
            if hasattr(obj, "sediment") and obj.sediment is not None:
                components["Sediment"] = type(obj.sediment).__name__
            if hasattr(obj, "cosine") and obj.cosine is not None:
                components["CoSiNE"] = type(obj.cosine).__name__
            if hasattr(obj, "wwminput") and obj.wwminput is not None:
                components["Wave"] = type(obj.wwminput).__name__

            for comp_name, comp_type in components.items():
                lines.append(f"  {bullet} {comp_name}: {comp_type}")

            if not components:
                lines.append(f"  {bullet} No modules configured")

            lines.append(footer)
            return "\n".join(lines)

        # Format Param class
        from .param import Param
        if isinstance(obj, Param):
            header, footer, bullet = get_formatted_header_footer(
                title="SCHISM PARAMETERS", use_ascii=USE_ASCII_ONLY
            )

            lines = [header]

            # Add key parameter information
            if hasattr(obj, "core") and obj.core is not None:
                if hasattr(obj.core, "rnday"):
                    lines.append(f"  {bullet} Run duration: {obj.core.rnday} days")
                if hasattr(obj.core, "dt"):
                    lines.append(f"  {bullet} Time step: {obj.core.dt} seconds")

            if hasattr(obj, "opt") and obj.opt is not None:
                if hasattr(obj.opt, "nws"):
                    lines.append(f"  {bullet} Wind forcing: NWS={obj.opt.nws}")
                if hasattr(obj.opt, "ihot"):
                    hotstart = "Enabled" if obj.opt.ihot == 1 else "Disabled"
                    lines.append(f"  {bullet} Hotstart: {hotstart}")

            lines.append(footer)
            return "\n".join(lines)

        # Use the new formatting framework
        from rompy.formatting import format_value
        return format_value(obj)


if __name__ == "__main__":
    nml = NML()
    nml.write_nml(Path("test"))
