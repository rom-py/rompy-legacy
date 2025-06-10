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
    icm: Optional[Icm] = Field(description="Ice model parameters", default=None)
    mice: Optional[Mice] = Field(description="Ice model parameters", default=None)
    sediment: Optional[Sediment] = Field(
        description="Sediment model parameters", default=None
    )
    cosine: Optional[Cosine] = Field(
        description="Sediment model parameters", default=None
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
        if datasources["wave"] is not None:
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
        if datasources["atmos"] is not None:
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


if __name__ == "__main__":
    nml = NML()
    nml.write_nml(Path("test"))
