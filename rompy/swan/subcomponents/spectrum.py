"""Spectrum subcomponents."""

import logging
from typing import Any, Literal, Optional

from pydantic import Field, model_validator

from rompy.swan.subcomponents.base import BaseSubComponent

logger = logging.getLogger(__name__)


class SPECTRUM(BaseSubComponent):
    """SWAN spectrum specification.

    .. code-block:: text

        ->CIRCLE|SECTOR ([dir1] [dir2]) [mdc] [flow] [fhigh] [msc]

    Notes
    -----

    Directions in the spectra are defined either as a CIRCLE or as a SECTOR. In the
    case of a SECTOR, both `dir1` and `dir2` must be specified. In the case of a
    CIRCLE, neither `dir1` nor `dir2` should be specified.

    At least two of `flow`, `fhigh` and `msc` must be specified in which case the
    third parameter will be calculated by SWAN such that the frequency resolution
    `df/f = 0.1` (10% increments).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.spectrum import SPECTRUM
        spec = SPECTRUM(mdc=36, flow=0.04, fhigh=1.0)
        print(spec.render())
        spec = SPECTRUM(mdc=36, dir1=0, dir2=180, flow=0.04, msc=31)
        print(spec.render())

    """

    model_type: Literal["spectrum", "SPECTRUM"] = Field(
        default="spectrum", description="Model type discriminator"
    )
    mdc: int = Field(
        description=(
            "Number of meshes in theta-space. In the case of CIRCLE, this is the "
            "number of subdivisions of the 360 degrees of a circle so "
            "`dtheta = [360]/[mdc]` is the spectral directional resolution. In the "
            "case of SECTOR, `dtheta = ([dir2] - [dir1])/[mdc]`. The minimum number "
            "of directional bins is 3 per directional quadrant."
        )
    )
    flow: Optional[float] = Field(
        default=None,
        description=(
            "Lowest discrete frequency that is used in the calculation (in Hz)."
        ),
    )
    fhigh: Optional[float] = Field(
        default=None,
        description=(
            "Highest discrete frequency that is used in the calculation (in Hz)."
        ),
    )
    msc: Optional[int] = Field(
        default=None,
        description=(
            "One less than the number of frequencies. This defines the grid "
            "resolution in frequency-space between the lowest discrete frequency "
            "`flow` and the highest discrete frequency `fhigh`. This resolution is "
            "not constant, since the frequencies are distributed logarithmical: "
            "`fi+1 = yfi` where `y` is a constant. The minimum number of frequencies "
            "is 4"
        ),
        ge=3,
    )
    dir1: Optional[float] = Field(
        default=None,
        description=(
            "The direction of the right-hand boundary of the sector when looking "
            "outward from the sector (required for option SECTOR) in degrees."
        ),
    )
    dir2: Optional[float] = Field(
        default=None,
        description=(
            "The direction of the left-hand boundary of the sector when looking "
            "outward from the sector (required for option SECTOR) in degrees."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def check_direction_definition(cls, data: Any) -> Any:
        """Check that dir1 and dir2 are specified together."""
        dir1 = data.get("dir1")
        dir2 = data.get("dir2")
        if None in [dir1, dir2] and dir1 != dir2:
            raise ValueError("dir1 and dir2 must be specified together")
        return data

    @model_validator(mode="after")
    def check_frequency_definition(self) -> "SPECTRUM":
        """Check spectral frequencies are prescribed correctly."""
        args = [self.flow, self.fhigh, self.msc]
        if None in args:
            args = [arg for arg in args if arg is not None]
            if len(args) != 2:
                raise ValueError("You must specify at least 2 of [flow, fhigh, msc]")
        if self.flow is not None and self.fhigh is not None and self.flow >= self.fhigh:
            raise ValueError("flow must be less than fhigh")
        return self

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

    .. code-block:: text

        JONSWAP [gamma]

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.spectrum import JONSWAP
        shape = JONSWAP(gamma=3.3)
        print(shape.render())

    """

    model_type: Literal["jonswap", "JONSWAP"] = Field(
        default="jonswap", description="Model type discriminator"
    )
    gamma: float = Field(
        default=3.3,
        description="Peak enhancement parameter of the JONSWAP spectrum.",
        gt=0.0,
    )

    def cmd(self) -> str:
        return f"{super().cmd()} gamma={self.gamma}"


class TMA(JONSWAP):
    """TMA spectral shape.

    .. code-block:: text

        TMA [gamma] [d]

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.spectrum import TMA
        shape = TMA(gamma=2.0, d=18)
        print(shape.render())

    """

    model_type: Literal["tma", "TMA"] = Field(
        default="tma", description="Model type discriminator"
    )
    d: float = Field(
        description="The reference depth at the wave maker in meters.",
        gt=0.0,
    )

    def cmd(self) -> str:
        return f"{super().cmd()} d={self.d}"


class GAUSS(BaseSubComponent):
    """Gaussian spectral shape.

    .. code-block:: text

        GAUSS [sigfr]

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.spectrum import GAUSS
        shape = GAUSS(sigfr=0.02)
        print(shape.render())

    """

    model_type: Literal["gauss", "GAUSS"] = Field(
        default="gauss", description="Model type discriminator"
    )
    sigfr: float = Field(
        description=(
            "Width of the Gaussian frequency spectrum expressed "
            "as a standard deviation in Hz."
        ),
        gt=0.0,
    )

    def cmd(self) -> str:
        return f"{super().cmd()} sigfr={self.sigfr}"


class PM(BaseSubComponent):
    """Pearson-Moskowitz spectral shape.

    .. code-block:: text

        PM

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.spectrum import PM
        shape = PM()
        print(shape.render())

    """

    model_type: Literal["pm", "PM"] = Field(
        default="pm", description="Model type discriminator"
    )


class BIN(BaseSubComponent):
    """Single frequency bin spectral shape.

    .. code-block:: text

        BIN

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.spectrum import BIN
        shape = BIN()
        print(shape.render())

    """

    model_type: Literal["bin", "BIN"] = Field(
        default="bin", description="Model type discriminator"
    )


class SHAPESPEC(BaseSubComponent):
    """Spectral shape specification.

    .. code-block:: text

        BOUND SHAPESPEC JONSWAP|PM|GAUSS|BIN|TMA PEAK|MEAN DSPR [POWER|DEGREES]

    This command BOUND SHAPESPEC defines the shape of the spectra (both in frequency
    and direction) at the boundary of the computational grid in case of parametric
    spectral input.

    Notes
    -----
    While technically a component `BOUND SHAPESPEC`, this is only intended to be used
    as a subcomponent of the `BOUNDSPEC` component.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.spectrum import SHAPESPEC
        shapespec = SHAPESPEC()
        print(shapespec.render())
        shapespec = SHAPESPEC(
            shape=dict(model_type="tma", gamma=3.1, d=12),
            per_type="mean",
            dspr_type="degrees",
        )
        print(shapespec.render())

    """

    model_type: Literal["shapespec", "SHAPESPEC"] = Field(
        default="shapespec", description="Model type discriminator"
    )
    shape: JONSWAP | PM | GAUSS | BIN | TMA = Field(
        default_factory=JONSWAP,
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
