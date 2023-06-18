"""Spectrum subcomponents."""
import logging
from typing import Literal, Optional
from pydantic import Field, root_validator, conint, confloat

from rompy.swan.subcomponents.base import BaseSubComponent

logger = logging.getLogger(__name__)


class SPECTRUM(BaseSubComponent):
    """SWAN spectrum specification.

    `CIRCLE|SECTOR ([dir1] [dir2]) [mdc] [flow] [fhigh] [msc]`

    Notes
    -----
    Directions in the spectra are defined either as a CIRCLE or as a SECTOR. In the
    case of a SECTOR, both `dir1` and `dir2` must be specified. In the case of a
    CIRCLE, neither `dir1` nor `dir2` should be specified.

    At least two of `flow`, `fhigh` and `msc` must be specified in which case the third
    parameter will be calculated by SWAN such that the frequency resolution
    $df/f$ = 0.1 (10% increments).

    """

    model_type: Literal["spectrum"] = Field(
        default="spectrum", description="Model type discriminator"
    )
    mdc: int = Field(
        description=(
            "Number of meshes in θ-space. In the case of CIRCLE, this is the number "
            "of subdivisions of the 360 degrees of a circle so ∆θ = [360]/[mdc] is "
            "the spectral directional resolution. In the case of SECTOR, "
            "∆θ = ([dir2] - [dir1])/[mdc]. The minimum number of directional bins is "
            "3 per directional quadrant."
        )
    )
    flow: Optional[float] = Field(
        description=(
            "Lowest discrete frequency that is used in the calculation (in Hz)."
        ),
    )
    fhigh: Optional[float] = Field(
        description=(
            "Highest discrete frequency that is used in the calculation (in Hz)."
        ),
    )
    msc: Optional[conint(ge=3)] = Field(
        description=(
            "One less than the number of frequencies. This defines the grid "
            "resolution in frequency-space between the lowest discrete frequency "
            "`flow` and the highest discrete frequency `fhigh`. This resolution is "
            "not constant, since the frequencies are distributed logarithmical: "
            "fi+1 = yfi with y is a constant. The minimum number of frequencies is 4."
        ),
    )
    dir1: Optional[float] = Field(
        description=(
            "The direction of the right-hand boundary of the sector when looking "
            "outward from the sector (required for option SECTOR) in degrees."
        ),
    )
    dir2: Optional[float] = Field(
        description=(
            "The direction of the left-hand boundary of the sector when looking "
            "outward from the sector (required for option SECTOR) in degrees."
        ),
    )

    @root_validator
    def check_direction_definition(cls, values: dict) -> dict:
        """Check that dir1 and dir2 are specified together."""
        dir1 = values.get("dir1")
        dir2 = values.get("dir2")
        if None in [dir1, dir2] and dir1 != dir2:
            raise ValueError("dir1 and dir2 must be specified together")
        return values

    @root_validator
    def check_frequency_definition(cls, values: dict) -> dict:
        """Check spectral frequencies are prescribed correctly."""
        flow = values.get("flow")
        fhigh = values.get("fhigh")
        msc = values.get("msc")
        args = [flow, fhigh, msc]
        if None in args:
            args = [arg for arg in args if arg is not None]
            if len(args) != 2:
                raise ValueError("You must specify at least 2 of [flow, fhigh, msc]")
        if flow is not None and fhigh is not None and flow >= fhigh:
            raise ValueError("flow must be less than fhigh")
        return values

    @property
    def dir_sector(self):
        if self.dir1 is None and self.dir2 is None:
            return "CIRCLE"
        else:
            return f"SECTOR {self.dir1} {self.dir2}"

    def cmd(self) -> str:
        repr = f"{self.dir_sector} mdc={self.mdc}"
        if self.flow is not None:
            repr += f" flow={self.flow}"
        if self.fhigh is not None:
            repr += f" fhigh={self.fhigh}"
        if self.msc is not None:
            repr += f" msc={self.msc}"
        return repr


class JONSWAP(BaseSubComponent):
    """Jonswap spectral shape.

    `JONSWAP [gamma]`

    """

    model_type: Literal["jonswap"] = Field(
        default="jonswap", description="Model type discriminator"
    )
    gamma: confloat(gt=0.0) = Field(
        default=3.3,
        description="Peak enhancement parameter of the JONSWAP spectrum.",
    )

    def cmd(self) -> str:
        return f"{super().cmd()} gamma={self.gamma}"


class TMA(JONSWAP):
    """TMA spectral shape.

    `TMA [gamma] [d]`

    """

    model_type: Literal["tma"] = Field(
        default="tma", description="Model type discriminator"
    )
    d: confloat(gt=0.0) = Field(
        description="The reference depth at the wave maker in meters.",
    )

    def cmd(self) -> str:
        return f"{super().cmd()} d={self.d}"


class GAUSS(BaseSubComponent):
    """Gaussian spectral shape.

    `GAUSS [sigfr]`

    """

    model_type: Literal["gauss"] = Field(
        default="gauss", description="Model type discriminator"
    )
    sigfr: confloat(gt=0.0) = Field(
        description=(
            "Width of the Gaussian frequency spectrum expressed "
            "as a standard deviation in Hz."
        ),
    )

    def cmd(self) -> str:
        return f"{super().cmd()} sigfr={self.sigfr}"


class PM(BaseSubComponent):
    """Pearson-Moskowitz spectral shape.

    `PM`

    """

    model_type: Literal["pm"] = Field(
        default="pm", description="Model type discriminator"
    )


class BIN(BaseSubComponent):
    """Single frequency bin spectral shape.

    `BIN`

    """

    model_type: Literal["bin"] = Field(
        default="bin", description="Model type discriminator"
    )


class SHAPESPEC(BaseSubComponent):
    """Spectral shape specification.

    `BOUND SHAPESPEC JONSWAP|PM|GAUSS|BIN|TMA PEAK|MEAN DSPR [POWER|DEGREES]`

    This command BOUND SHAPESPEC defines the shape of the spectra (both in frequency
    and direction) at the boundary of the computational grid in case of parametric
    spectral input.

    Notes
    -----
    While technically a component `BOUND SHAPESPEC`, this is only intended to be used
    as a subcomponent of the `BOUNDSPEC` component.

    """

    model_type: Literal["shapespec"] = Field(
        default="shapespec", description="Model type discriminator"
    )
    shape: JONSWAP | PM | GAUSS | BIN | TMA = Field(
        default=JONSWAP(),
        description="The spectral shape",
    )
    per_type: Literal["peak", "mean"] = Field(
        default="peak",
        description="The type of characteristic wave period",
    )
    dspr_type: Literal["power", "degrees"] = Field(
        default="power",
        description="The type of directional spreading",
    )

    def cmd(self) -> str:
        repr = (
            f"BOUND SHAPESPEC {self.shape.render()} {self.per_type.upper()} "
            f"DSPR {self.dspr_type.upper()}"
        )
        return repr
