"""
SWAN Output Subcomponents

This module contains subcomponents for defining output specifications in SWAN,
including spectral output types and frequency representations.
"""

# Standard library imports
from typing import Literal

# Third-party imports
from pydantic import Field

# Local imports
from rompy.swan.subcomponents.base import BaseSubComponent


class SPEC1D(BaseSubComponent):
    """Frequency 1D spectra.

    .. code-block:: text

        SPEC1D

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.output import SPEC1D
        spec = SPEC1D()
        print(spec.render())

    """

    model_type: Literal["spec1d", "SPEC1D"] = Field(
        default="spec1d", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        return "SPEC2D"


class SPEC2D(BaseSubComponent):
    """Frequency-direction 2D spectra.

    .. code-block:: text

        SPEC2D

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.output import SPEC2D
        spec = SPEC2D()
        print(spec.render())

    """

    model_type: Literal["spec2d", "SPEC2D"] = Field(
        default="spec2d", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        return "SPEC2D"


class ABS(BaseSubComponent):
    """Absolute frequency spectra.

    .. code-block:: text

        ABS

    Spectra are computed as a function of absolute frequency, i.e., the frequency as
    measured in a fixed point.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.output import ABS
        freq = ABS()
        print(freq.render())

    """

    model_type: Literal["abs", "ABS"] = Field(
        default="abs", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        return "ABS"


class REL(BaseSubComponent):
    """Relative frequency spectra.

    .. code-block:: text

        REL

    Spectra are computed as a function of relative frequency, i.e., the frequency as
    measured when moving with current.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.output import REL
        freq = REL()
        print(freq.render())

    """

    model_type: Literal["rel", "REL"] = Field(
        default="rel", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        return "REL"
