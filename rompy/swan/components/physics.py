"""
SWAN Physics Components

This module contains components for configuring the physical processes in SWAN,
including wind generation, whitecapping, quadruplet interactions, and wave breaking.
"""

# Standard library imports
import logging
from typing import Any, Literal, Optional, Union, Annotated

# Third-party imports
from pydantic import field_validator, model_validator, Field, ValidationInfo

# Local imports
from rompy.core.logging import get_logger
from rompy.swan.components.base import BaseComponent
from rompy.swan.types import IDLA, PhysicsOff
from rompy.swan.subcomponents.physics import (
    JANSSEN,
    KOMEN,
    WESTHUYSEN,
    ST6,
    ST6C1,
    ST6C2,
    ST6C3,
    ST6C4,
    ST6C5,
    ELDEBERKY,
    DEWIT,
    TRANSM,
    TRANS1D,
    TRANS2D,
    GODA,
    DANGREMOND,
    REFL,
    RSPEC,
    RDIFF,
    FREEBOARD,
    LINE,
)

# Initialize the logger
logger = get_logger(__name__)


SOURCE_TERMS = Union[
    JANSSEN,
    KOMEN,
    WESTHUYSEN,
    ST6,
    ST6C1,
    ST6C2,
    ST6C3,
    ST6C4,
    ST6C5,
]


# =====================================================================================
# Wave generation GEN1 | GEN2 | GEN3
# =====================================================================================
class GEN1(BaseComponent):
    """First generation source terms GEN1.

    .. code-block:: text

        GEN1 [cf10] [cf20] [cf30] [cf40] [edmlpm] [cdrag] [umin] [cfpm]

    With this command the user indicates that SWAN should run in first-generation mode
    (see Scientific/Technical documentation).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import GEN1
        gen = GEN1()
        print(gen.render())
        kwargs = dict(
            cf10=188.0,
            cf20=0.59,
            cf30=0.12,
            cf40=250.0,
            edmlpm=0.0036,
            cdrag=0.0012,
            umin=1.0,
            cfpm=0.13
        )
        gen = GEN1(**kwargs)
        print(gen.render())

    """

    model_type: Literal["gen1", "GEN1"] = Field(
        default="gen1", description="Model type discriminator"
    )
    cf10: Optional[float] = Field(
        default=None,
        description="Controls the linear wave growth (SWAN default: 188.0)",
    )
    cf20: Optional[float] = Field(
        default=None,
        description="Controls the exponential wave growth (SWAN default: 0.59)",
    )
    cf30: Optional[float] = Field(
        default=None,
        description="Controls the exponential wave growth (SWAN default: 0.12)",
    )
    cf40: Optional[float] = Field(
        default=None,
        description=(
            "Controls the dissipation rate, i.e., the time decay scale "
            "(SWAN default: 250.0)"
        ),
    )
    edmlpm: Optional[float] = Field(
        default=None,
        description=(
            "Maximum non-dimensionless energy density of the wind sea part of the "
            "spectrum according to Pierson Moskowitz (SWAN default: 0.0036)"
        ),
    )
    cdrag: Optional[float] = Field(
        default=None, description="Drag coefficient (SWAN default: 0.0012)"
    )
    umin: Optional[float] = Field(
        default=None,
        description=(
            "Minimum wind velocity (relative to current; all wind speeds "
            "are taken at 10 m above sea level) (SWAN default: 1)"
        ),
    )
    cfpm: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient which determines the Pierson Moskowitz frequency: "
            "`delta_PM = 2pi g / U_10` (SWAN default: 0.13)"
        ),
    )

    def cmd(self):
        """Command line string for this component."""
        repr = "GEN1"
        if self.cf10 is not None:
            repr += f" cf10={self.cf10}"
        if self.cf20 is not None:
            repr += f" cf20={self.cf20}"
        if self.cf30 is not None:
            repr += f" cf30={self.cf30}"
        if self.cf40 is not None:
            repr += f" cf40={self.cf40}"
        if self.edmlpm is not None:
            repr += f" edmlpm={self.edmlpm}"
        if self.cdrag is not None:
            repr += f" cdrag={self.cdrag}"
        if self.umin is not None:
            repr += f" umin={self.umin}"
        if self.cfpm is not None:
            repr += f" cfpm={self.cfpm}"
        return repr


class GEN2(GEN1):
    """Second generation source terms GEN2.

    .. code-block:: text

        GEN2 [cf10] [cf20] [cf30] [cf40] [cf50] [cf60] [edmlpm] [cdrag] [umin] [cfpm]

    With this command the user indicates that SWAN should run in second-generation mode
    (see Scientific/Technical documentation).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import GEN2
        gen = GEN2()
        print(gen.render())
        kwargs = dict(
            cf10=188.0,
            cf20=0.59,
            cf30=0.12,
            cf40=250.0,
            cf50=0.0023,
            cf60=-0.223,
            edmlpm=0.0036,
            cdrag=0.0012,
            umin=1.0,
            cfpm=0.13
        )
        gen = GEN2(**kwargs)
        print(gen.render())

    """

    model_type: Literal["gen2", "GEN2"] = Field(
        default="gen2", description="Model type discriminator"
    )
    cf50: Optional[float] = Field(
        default=None,
        description=(
            "Controls the spectral energy scale of the limit spectrum "
            "(SWAN default: 0.0023)"
        ),
    )
    cf60: Optional[float] = Field(
        default=None,
        description=(
            "Ccontrols the spectral energy scale of the limit spectrum "
            "(SWAN default: -0.223"
        ),
    )

    def cmd(self):
        """Command line string for this component."""
        repr = "GEN2"
        if self.cf10 is not None:
            repr += f" cf10={self.cf10}"
        if self.cf20 is not None:
            repr += f" cf20={self.cf20}"
        if self.cf30 is not None:
            repr += f" cf30={self.cf30}"
        if self.cf40 is not None:
            repr += f" cf40={self.cf40}"
        if self.cf50 is not None:
            repr += f" cf50={self.cf50}"
        if self.cf60 is not None:
            repr += f" cf60={self.cf60}"
        if self.edmlpm is not None:
            repr += f" edmlpm={self.edmlpm}"
        if self.cdrag is not None:
            repr += f" cdrag={self.cdrag}"
        if self.umin is not None:
            repr += f" umin={self.umin}"
        if self.cfpm is not None:
            repr += f" cfpm={self.cfpm}"
        return repr


class GEN3(BaseComponent):
    """Third generation source terms GEN3.

    .. code-block:: text

        GEN3 JANSSEN|KOMEN|->WESTHUYSEN|ST6 AGROW [a]

    With this command the user indicates that SWAN should run in third-generation mode
    for wind input, quadruplet interactions and whitecapping.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import GEN3
        gen = GEN3(
            source_terms=dict(
                model_type="westhuysen",
                wind_drag="wu",
                agrow=True,
            ),
        )
        print(gen.render())
        from rompy.swan.subcomponents.physics import ST6C1
        gen = GEN3(source_terms=ST6C1())
        print(gen.render())

    """

    model_type: Literal["gen3", "GEN3"] = Field(
        default="gen3", description="Model type discriminator"
    )
    source_terms: SOURCE_TERMS = Field(
        default_factory=WESTHUYSEN,
        description="SWAN source terms to be used (SWAN default: WESTHUYSEN)",
        discriminator="model_type",
    )

    def cmd(self):
        """Command line string for this component."""
        repr = f"GEN3 {self.source_terms.render()}"
        return repr


# =====================================================================================
# Swell dissipation SSWELL
# =====================================================================================
class NEGATINP(BaseComponent):
    """Negative wind input.

    .. code-block:: text

        NEGATINP [rdcoef]

    With this optional command the user activates negative wind input. **This is
    intended only for use with non-breaking swell dissipation SSWELL ZIEGER**.
    Parameter `rdcoef` is a fraction between 0 and 1, representing the strength of
    negative wind input. As an example, with [rdcoef]=0.04, for a spectral bin that is
    opposed to the wind direction, the wind input factor W(k, θ) is negative, and its
    magnitude is 4% of the corresponding value of the spectral bin that is in the
    opposite direction (i.e. in the wind direction). See Zieger et al. (2015) eq. 11,
    where a0 is their notation for [rdcoef]. Default [rdcoef]=0.0 and `rdcoef=0.04` is
    recommended, though as implied by Zieger et al. (2015), this value is not
    well-established, so the user is encouraged to experiment with other values.

    References
    ----------
    Zieger, S., Babanin, A.V., Rogers, W.E. and Young, I.R., 2015. Observation-based
    source terms in the third-generation wave model WAVEWATCH. Ocean Modelling, 96,
    pp.2-25.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import NEGATINP
        negatinp = NEGATINP()
        print(negatinp.render())
        negatinp = NEGATINP(rdcoef=0.04)
        print(negatinp.render())

    """

    model_type: Literal["negatinp", "NEGATINP"] = Field(
        default="negatinp", description="Model type discriminator"
    )
    rdcoef: Optional[float] = Field(
        default=None,
        description="Coefficient representing the strength of negative wind input",
        ge=0.0,
        le=1.0,
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "NEGATINP"
        if self.rdcoef is not None:
            repr += f" rdcoef={self.rdcoef}"
        return repr


class SSWELL_ROGERS(BaseComponent):
    """Nonbreaking dissipation of Rogers et al. (2012).

    .. code-block:: text

        SSWELL ROGERS [cdsv] [feswell]

    References
    ----------
    Rogers, W.E., Babanin, A.V. and Wang, D.W., 2012. Observation-consistent input and
    whitecapping dissipation in a model for wind-generated surface waves: Description
    and simple calculations. Journal of Atmospheric and Oceanic Technology, 29(9),
    pp.1329-1346.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SSWELL_ROGERS
        sswell = SSWELL_ROGERS()
        print(sswell.render())
        sswell = SSWELL_ROGERS(cdsv=1.2, feswell=0.5)
        print(sswell.render())

    """

    model_type: Literal["rogers", "ROGERS"] = Field(
        default="rogers", description="Model type discriminator"
    )
    cdsv: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient related to laminar atmospheric boundary layer "
            "(SWAN default: 1.2)"
        ),
    )
    feswell: Optional[float] = Field(
        default=None, description="Swell dissipation factor"
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SSWELL ROGERS"
        if self.cdsv is not None:
            repr += f" cdsv={self.cdsv}"
        if self.feswell is not None:
            repr += f" feswell={self.feswell}"
        return repr


class SSWELL_ARDHUIN(BaseComponent):
    """Nonbreaking dissipation of Ardhuin et al. (2010).

    .. code-block:: text

        SSWELL ARDHUIN [cdsv]

    References
    ----------
    Ardhuin, F., Rogers, E., Babanin, A.V., Filipot, J.F., Magne, R., Roland, A.,
    Van Der Westhuysen, A., Queffeulou, P., Lefevre, J.M., Aouf, L. and Collard, F.,
    2010. Semiempirical dissipation source functions for ocean waves. Part I:
    Definition, calibration, and validation. Journal of Physical Oceanography, 40(9),
    pp.1917-1941.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SSWELL_ARDHUIN
        sswell = SSWELL_ARDHUIN()
        print(sswell.render())
        sswell = SSWELL_ARDHUIN(cdsv=1.2)
        print(sswell.render())

    """

    model_type: Literal["ardhuin", "ARDHUIN"] = Field(
        default="ardhuin", description="Model type discriminator"
    )
    cdsv: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient related to laminar atmospheric boundary layer "
            "(SWAN default: 1.2)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SSWELL ARDHUIN"
        if self.cdsv is not None:
            repr += f" cdsv={self.cdsv}"
        return repr


class SSWELL_ZIEGER(BaseComponent):
    """Nonbreaking dissipation of Zieger et al. (2015).

    .. code-block:: text

        SSWELL ZIEGER [b1]

    Swell dissipation of Young et al. (2013) updated by Zieger et al. (2015). The
    Zieger option is intended for use with negative wind input via the NEGATINP
    command. Zieger non-breaking dissipation follows the method used in WAVEWATCH III
    version 4 and does not include the steepness-dependent swell coefficient introduced
    in WAVEWATCH III version 5.

    References
    ----------
    Zieger, S., Babanin, A.V., Rogers, W.E. and Young, I.R., 2015. Observation-based
    source terms in the third-generation wave model WAVEWATCH. Ocean Modelling, 96,
    pp.2-25.

    Young, I.R., Babanin, A.V. and Zieger, S., 2013. The decay rate of ocean swell
    observed by altimeter. Journal of physical oceanography, 43(11), pp.2322-2333.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SSWELL_ZIEGER
        sswell = SSWELL_ZIEGER()
        print(sswell.render())
        sswell = SSWELL_ZIEGER(b1=0.00025)
        print(sswell.render())

    """

    model_type: Literal["zieger", "ZIEGER"] = Field(
        default="zieger", description="Model type discriminator"
    )
    b1: Optional[float] = Field(
        default=None,
        description="Non-dimensional proportionality coefficient "
        "(SWAN default: 0.00025)",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SSWELL ZIEGER"
        if self.b1 is not None:
            repr += f" b1={self.b1}"
        return repr


# =====================================================================================
# WCAPPING
# =====================================================================================
class WCAPPING_KOMEN(BaseComponent):
    """Whitecapping according to Komen (1984).

    .. code-block:: text

        WCAPPING KOMEN [cds2] [stpm] [powst] [delta] [powk]

    Notes
    -----
    The SWAN default for `delta` has been changed since version 40.91A. The setting
    `delta = 1` will improve the prediction of the wave energy at low frequencies, and
    hence the mean wave period. The original default was `delta = 0`, which corresponds
    to WAM Cycle 3. See the Scientific/Technical documentation for further details.

    References
    ----------
    Komen, G.J., Hasselmann, S. and Hasselmann, K., 1984. On the existence of a fully
    developed wind-sea spectrum. Journal of physical oceanography, 14(8), pp.1271-1285.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import WCAPPING_KOMEN
        wcapping = WCAPPING_KOMEN()
        print(wcapping.render())
        wcapping = WCAPPING_KOMEN(cds2=2.36e-5, stpm=3.02e-3, powst=2, delta=1, powk=2)
        print(wcapping.render())

    """

    model_type: Literal["komen", "KOMEN"] = Field(
        default="komen", description="Model type discriminator"
    )
    cds2: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient for determining the rate of whitecapping dissipation ($Cds$) "
            "(SWAN default: 2.36e-5)"
        ),
    )
    stpm: Optional[float] = Field(
        default=None,
        description=(
            "Value of the wave steepness for a Pierson-Moskowitz spectrum "
            "($s^2_{PM}$) (SWAN default: 3.02e-3)"
        ),
    )
    powst: Optional[float] = Field(
        default=None,
        description=(
            "Power of steepness normalized with the wave steepness "
            "of a Pierson-Moskowitz spectrum (SWAN default: 2)"
        ),
    )
    delta: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient which determines the dependency of the whitecapping "
            "on wave number (SWAN default: 1)"
        ),
    )
    powk: Optional[float] = Field(
        default=None,
        description=(
            "power of wave number normalized with the mean wave number "
            "(SWAN default: 1)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"WCAPPING KOMEN"
        if self.cds2 is not None:
            repr += f" cds2={self.cds2}"
        if self.stpm is not None:
            repr += f" stpm={self.stpm}"
        if self.powst is not None:
            repr += f" powst={self.powst}"
        if self.delta is not None:
            repr += f" delta={self.delta}"
        if self.powk is not None:
            repr += f" powk={self.powk}"
        return repr


class WCAPPING_AB(BaseComponent):
    """Whitecapping according to Alves and Banner (2003).

    .. code-block:: text

        WCAPPING AB [cds2] [br] CURRENT [cds3]

    References
    ----------
    Alves, J.H.G. and Banner, M.L., 2003. Performance of a saturation-based
    dissipation-rate source term in modeling the fetch-limited evolution of wind waves.
    Journal of Physical Oceanography, 33(6), pp.1274-1298.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import WCAPPING_AB
        wcapping = WCAPPING_AB()
        print(wcapping.render())
        wcapping = WCAPPING_AB(cds2=5.0e-5, br=1.75e-3, current=True, cds3=0.8)
        print(wcapping.render())

    """

    model_type: Literal["ab", "AB"] = Field(
        default="ab", description="Model type discriminator"
    )
    cds2: Optional[float] = Field(
        default=None,
        description=(
            "proportionality coefficient due to Alves and Banner (2003) "
            "(SWAN default: 5.0e-5)"
        ),
    )
    br: Optional[float] = Field(
        default=None, description="Threshold saturation level	(SWAN default: 1.75e-3)"
    )
    current: bool = Field(
        default=False,
        description=(
            "Indicates that enhanced current-induced dissipation "
            "as proposed by Van der Westhuysen (2012) is to be added"
        ),
    )
    cds3: Optional[float] = Field(
        default=None, description="Proportionality coefficient (SWAN default: 0.8)"
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"WCAPPING AB"
        if self.cds2 is not None:
            repr += f" cds2={self.cds2}"
        if self.br is not None:
            repr += f" br={self.br}"
        if self.current:
            repr += f" CURRENT"
        if self.cds3 is not None:
            repr += f" cds3={self.cds3}"
        return repr


# =====================================================================================
# QUADRUPL
# =====================================================================================
class QUADRUPL(BaseComponent):
    """Nonlinear quadruplet wave interactions.

    .. code-block:: text

        QUADRUPL [iquad] [lambda] [cnl4] [Csh1] [Csh2] [Csh3]

    With this option the user can influence the computation of nonlinear quadruplet
    wave interactions which are usually included in the computations. Can be
    de-activated with command OFF QUAD. Note that the DIA approximation of the
    quadruplet interactions is a poor approximation for long-crested waves and
    frequency resolutions that are deviating much more than 10% (see command CGRID).
    Note that DIA is usually updated per sweep, either semi-implicit (`iquad = 1`) or
    explicit (`iquad = 2`). However, when ambient current is included, the bounds of
    the directional sector within a sweep may be different for each frequency bin
    (particularly the higher frequencies are modified by the current). So there may be
    some overlap of frequency bins between the sweeps, implying non-conservation of
    wave energy. To prevent this the user is advised to choose the integration of DIA
    per iteration instead of per sweep, i.e. `iquad = 3`. If you want to speed up your
    computation a bit more, than the choice `iquad = 8` is a good choice.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import QUADRUPL
        quadrupl = QUADRUPL()
        print(quadrupl.render())
        kwargs = dict(
            iquad=3, lambd=0.25, cnl4=3.0e7, csh1=5.5, csh2=0.833333, csh3=-1.25
        )
        quadrupl = QUADRUPL(**kwargs)
        print(quadrupl.render())

    """

    model_type: Literal["quadrupl", "QUADRUPL"] = Field(
        default="quadrupl", description="Model type discriminator"
    )
    iquad: Optional[Literal[1, 2, 3, 8, 4, 51, 52, 53]] = Field(
        default=None,
        description=(
            "Numerical procedures for integrating the quadruplets: 1 = semi-implicit "
            "per sweep, 2 = explicit per sweep, 3 = explicit per iteration, "
            "8 = explicit per iteration, but with a more efficient implementation, "
            "4 = multiple DIA, 51 = XNL (deep water transfer), 52 = XNL (deep water "
            "transfer with WAM depth scaling), 53  XNL (finite depth transfer) (SWAN "
            "default: 2)"
        ),
    )
    lambd: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient for quadruplet configuration in case of DIA "
            "(SWAN default: 0.25)"
        ),
    )
    cnl4: Optional[float] = Field(
        default=None,
        description=(
            "Proportionality coefficient for quadruplet interactions in case of DIA "
            "(SWAN default: 3.0e7"
        ),
    )
    csh1: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient for shallow water scaling in case of DIA (SWAN default: 5.5)"
        ),
    )
    csh2: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient for shallow water scaling in case of DIA "
            "(SWAN default: 0.833333)"
        ),
    )
    csh3: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient for shallow water scaling in case of DIA "
            "(SWAN default: -1.25)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"QUADRUPL"
        if self.iquad is not None:
            repr += f" iquad={self.iquad}"
        if self.lambd is not None:
            repr += f" lambda={self.lambd}"
        if self.cnl4 is not None:
            repr += f" cnl4={self.cnl4}"
        if self.csh1 is not None:
            repr += f" csh1={self.csh1}"
        if self.csh2 is not None:
            repr += f" csh2={self.csh2}"
        if self.csh3 is not None:
            repr += f" csh3={self.csh3}"
        return repr


# =====================================================================================
# BREAKING
# =====================================================================================
class BREAKING_CONSTANT(BaseComponent):
    """Constant wave breaking index.

    .. code-block:: text

        BREAKING CONSTANT [alpha] [gamma]

    Indicates that a constant breaker index is to be used.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import BREAKING_CONSTANT
        breaking = BREAKING_CONSTANT()
        print(breaking.render())
        breaking = BREAKING_CONSTANT(alpha=1.0, gamma=0.73)
        print(breaking.render())

    """

    model_type: Literal["constant", "CONSTANT"] = Field(
        default="constant", description="Model type discriminator"
    )
    alpha: Optional[float] = Field(
        default=None,
        description=(
            "Proportionality coefficient of the rate of dissipation "
            "(SWAN default: 1.0)"
        ),
    )
    gamma: Optional[float] = Field(
        default=None,
        description=(
            "The breaker index, i.e. the ratio of maximum individual wave height "
            "over depth (SWAN default: 0.73)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "BREAKING CONSTANT"
        if self.alpha is not None:
            repr += f" alpha={self.alpha}"
        if self.gamma is not None:
            repr += f" gamma={self.gamma}"
        return repr


class BREAKING_BKD(BaseComponent):
    """Variable wave breaking index.

    .. code-block:: text

        BREAKING BKD [alpha] [gamma0] [a1] [a2] [a3]

    Indicates that the breaker index scales with both the bottom slope (`beta`)
    and the dimensionless depth (kd).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import BREAKING_BKD
        breaking = BREAKING_BKD()
        print(breaking.render())
        breaking = BREAKING_BKD(alpha=1.0, gamma0=0.54, a1=7.59, a2=-8.06, a3=8.09)
        print(breaking.render())

    """

    model_type: Literal["bkd", "BKD"] = Field(
        default="bkd", description="Model type discriminator"
    )
    alpha: Optional[float] = Field(
        default=None,
        description=(
            "Proportionality coefficient of the rate of dissipation "
            "(SWAN default: 1.0)"
        ),
    )
    gamma0: Optional[float] = Field(
        default=None,
        description="The reference $gamma$ for horizontal slopes (SWAN default: 0.54)",
    )
    a1: Optional[float] = Field(
        default=None,
        description=(
            "First tunable coefficient for the breaker index (SWAN default: 7.59)"
        ),
    )
    a2: Optional[float] = Field(
        default=None,
        description=(
            "Second tunable coefficient for the breaker index (SWAN default: -8.06)"
        ),
    )
    a3: Optional[float] = Field(
        default=None,
        description=(
            "Third tunable coefficient for the breaker index (SWAN default: 8.09)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "BREAKING BKD"
        if self.alpha is not None:
            repr += f" alpha={self.alpha}"
        if self.gamma0 is not None:
            repr += f" gamma0={self.gamma0}"
        if self.a1 is not None:
            repr += f" a1={self.a1}"
        if self.a2 is not None:
            repr += f" a2={self.a2}"
        if self.a3 is not None:
            repr += f" a3={self.a3}"
        return repr


# =====================================================================================
# FRICTION
# =====================================================================================
class FRICTION_JONSWAP(BaseComponent):
    """Hasselmann et al. (1973) Jonswap friction.

    .. code-block:: text

        FRICTION JONSWAP CONSTANT [cfjon]

    Indicates that the semi-empirical expression derived from the JONSWAP results for
    bottom friction dissipation (Hasselmann et al., 1973, JONSWAP) should be activated.
    This option is default.

    References
    ----------
    Hasselmann, K., Barnett, T.P., Bouws, E., Carlson, H., Cartwright, D.E., Enke, K.,
    Ewing, J.A., Gienapp, A., Hasselmann, D.E., Kruseman, P. and Meerburg, A., 1973.
    Measurements of wind-wave growth and swell decay during the Joint North Sea Wave
    Project (JONSWAP). Deutches Hydrographisches Institut, Hamburg, Germany,
    Rep. No. 12, 95 pp.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import FRICTION_JONSWAP
        friction = FRICTION_JONSWAP()
        print(friction.render())
        friction = FRICTION_JONSWAP(cfjon=0.038)
        print(friction.render())

    TODO: Implement VARIABLE option?

    """

    model_type: Literal["jonswap", "JONSWAP"] = Field(
        default="jonswap", description="Model type discriminator"
    )
    cfjon: Optional[float] = Field(
        default=None,
        description="Coefficient of the JONSWAP formulation (SWAN default: 0.038)",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "FRICTION JONSWAP CONSTANT"
        if self.cfjon is not None:
            repr += f" cfjon={self.cfjon}"
        return repr


class FRICTION_COLLINS(BaseComponent):
    """Collins (1972) friction.

    .. code-block:: text

        FRICTION COLLINS [cfw]

    Note that `cfw` is allowed to vary over the computational region; in that case use
    the commands INPGRID FRICTION and READINP FRICTION to define and read the friction
    data. This command FRICTION is still required to define the type of friction
    expression. The value of `cfw` in this command is then not required (it will be
    ignored).

    References
    ----------
    Collins, J.I., 1972. Prediction of shallow-water spectra. Journal of Geophysical
    Research, 77(15), pp.2693-2707.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import FRICTION_COLLINS
        friction = FRICTION_COLLINS()
        print(friction.render())
        friction = FRICTION_COLLINS(cfw=0.038)
        print(friction.render())

    """

    model_type: Literal["collins", "COLLINS"] = Field(
        default="collins", description="Model type discriminator"
    )
    cfw: Optional[float] = Field(
        default=None,
        description="Collins bottom friction coefficient (SWAN default: 0.015)",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "FRICTION COLLINS"
        if self.cfw is not None:
            repr += f" cfw={self.cfw}"
        return repr


class FRICTION_MADSEN(BaseComponent):
    """Madsen et al (1988) friction.

    .. code-block:: text

        FRICTION MADSEN [kn]

    Note that `kn` is allowed to vary over the computational region; in that case use
    the commands INPGRID FRICTION and READINP FRICTION to define and read the friction
    data. This command FRICTION is still required to define the type of friction
    expression. The value of `kn` in this command is then not required (it will be
    ignored).

    References
    ----------
    Madsen, O.S., Poon, Y.K. and Graber, H.C., 1988. Spectral wave attenuation by
    bottom friction: Theory. In Coastal engineering 1988 (pp. 492-504).

    Madsen, O.S. and Rosengaus, M.M., 1988. Spectral wave attenuation by bottom
    friction: Experiments. In Coastal Engineering 1988 (pp. 849-857).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import FRICTION_MADSEN
        friction = FRICTION_MADSEN()
        print(friction.render())
        friction = FRICTION_MADSEN(kn=0.038)
        print(friction.render())

    """

    model_type: Literal["madsen", "MADSEN"] = Field(
        default="madsen", description="Model type discriminator"
    )
    kn: Optional[float] = Field(
        default=None,
        description=(
            "equivalent roughness length scale of the bottom (in m) "
            "(SWAN default: 0.05)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "FRICTION MADSEN"
        if self.kn is not None:
            repr += f" kn={self.kn}"
        return repr


class FRICTION_RIPPLES(BaseComponent):
    """Smith et al. (2011) Ripples friction.

    .. code-block:: text

        FRICTION RIPPLES [S] [D]

    Indicates that the expression of Smith et al. (2011) should be activated. Here
    friction depends on the formation of bottom ripples and sediment size.

    References
    ----------
    Smith, G.A., Babanin, A.V., Riedel, P., Young, I.R., Oliver, S. and Hubbert, G.,
    2011. Introduction of a new friction routine into the SWAN model that evaluates
    roughness due to bedform and sediment size changes. Coastal Engineering, 58(4),
    pp.317-326.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import FRICTION_RIPPLES
        friction = FRICTION_RIPPLES()
        print(friction.render())
        friction = FRICTION_RIPPLES(s=2.65, d=0.0001)
        print(friction.render())

    """

    model_type: Literal["ripples", "RIPPLES"] = Field(
        default="ripples", description="Model type discriminator"
    )
    s: Optional[float] = Field(
        default=None,
        description="The specific gravity of the sediment (SWAN default: 2.65)",
    )
    d: Optional[float] = Field(
        default=None, description="The sediment diameter (in m) (SWAN default: 0.0001)"
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "FRICTION RIPPLES"
        if self.s is not None:
            repr += f" S={self.s}"
        if self.d is not None:
            repr += f" D={self.d}"
        return repr


# =====================================================================================
# TRIAD
# =====================================================================================
class TRIAD(BaseComponent):
    """Wave triad interactions.

    .. code-block:: text

        TRIAD [itriad] [trfac] [cutfr] [a] [b] [urcrit] [urslim]

    With this command the user can activate the triad wave-wave interactions. If this
    command is not used, SWAN will not account for triads.

    Note
    ----
    This is the TRIAD specification in SWAN < 41.45.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import TRIAD
        triad = TRIAD()
        print(triad.render())
        triad = TRIAD(
            itriad=1,
            trfac=0.8,
            cutfr=2.5,
            a=0.95,
            b=-0.75,
            ucrit=0.2,
            urslim=0.01,
        )
        print(triad.render())

    """

    model_type: Literal["triad", "TRIAD"] = Field(
        default="triad", description="Model type discriminator"
    )
    itriad: Optional[Literal[1, 2]] = Field(
        default=None,
        description=(
            "Approximation method for the triad computation: \n\n* 1: the LTA method "
            "of Eldeberky (1996) \n* 2: the SPB method of Becq-Girard et al. (1999) "
            "(SWAN default: 1)"
        ),
    )
    trfac: Optional[float] = Field(
        default=None,
        description=(
            "Proportionality coefficient (SWAN default: 0.8 in case of LTA method, "
            "0.9 in case of SPB method)"
        ),
    )
    cutfr: Optional[float] = Field(
        default=None,
        description=(
            "Controls the maximum frequency that is considered in the LTA "
            "computation. The value of `cutfr` is the ratio of this maximum "
            "frequency over the mean frequency (SWAN default: 2.5)"
        ),
    )
    a: Optional[float] = Field(
        default=None,
        description=(
            "First calibration parameter for tuning K in Eq. (5.1) of Becq-Girard et "
            "al. (1999). This parameter is associated with broadening of the "
            "resonance condition (SWAN default: 0.95)"
        ),
    )
    b: Optional[float] = Field(
        default=None,
        description=(
            "Second calibration parameter for tuning K in Eq. (5.1) of Becq-Girard "
            "et al. (1999). This parameter is associated with broadening of the "
            "resonance condition (SWAN default: -0.75 for 1D, 0.0 for 2D"
        ),
    )
    ucrit: Optional[float] = Field(
        default=None,
        description=(
            "The critical Ursell number appearing in the expression for the biphase "
            "(SWAN default: 0.2)"
        ),
    )
    urslim: Optional[float] = Field(
        default=None,
        description=(
            "The lower threshold for Ursell number, if the actual Ursell number is "
            "below this value triad interactions are be computed (SWAN default: 0.01)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "TRIAD"
        if self.itriad is not None:
            repr += f" itriad={self.itriad}"
        if self.trfac is not None:
            repr += f" trfac={self.trfac}"
        if self.cutfr is not None:
            repr += f" cutfr={self.cutfr}"
        if self.a is not None:
            repr += f" a={self.a}"
        if self.b is not None:
            repr += f" b={self.b}"
        if self.ucrit is not None:
            repr += f" urcrit={self.ucrit}"
        if self.urslim is not None:
            repr += f" urslim={self.urslim}"
        return repr


class TRIAD_DCTA(BaseComponent):
    """Triad interactions with the DCTA method of Booij et al. (2009).

    .. code-block:: text

        TRIAD DCTA [trfac] [p] COLL|NONC BIPHHASE ELDEBERKY|DEWIT

    References
    ----------
    Booij, N., Holthuijsen, L.H. and Bénit, M.P., 2009. A distributed collinear triad
    approximation in SWAN. In Proceedings Of Coastal Dynamics 2009: Impacts of Human
    Activities on Dynamic Coastal Processes (With CD-ROM) (pp. 1-10).

    Note
    ----
    This is the default method to compute the triad interactions in SWAN >= 41.45, it
    is not supported in earlier versions of the model.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import TRIAD_DCTA
        triad = TRIAD_DCTA()
        print(triad.render())
        triad = TRIAD_DCTA(
            trfac=4.4,
            p=1.3,
            noncolinear=True,
            biphase={"model_type": "dewit", "lpar": 0.0},
        )
        print(triad.render())

    """

    model_type: Literal["dcta", "DCTA"] = Field(
        default="dcta", description="Model type discriminator"
    )
    trfac: Optional[float] = Field(
        default=None,
        description=(
            "Scaling factor that controls the intensity of "
            "the triad interaction due to DCTA (SWAN default: 4.4)"
        ),
    )
    p: Optional[float] = Field(
        default=None,
        description=(
            "Shape coefficient to force the high-frequency tail" "(SWAN default: 4/3)"
        ),
    )
    noncolinear: bool = Field(
        default=False,
        description=(
            "If True, the noncolinear triad interactions "
            "with the DCTA framework are accounted for"
        ),
    )
    biphase: Optional[Union[ELDEBERKY, DEWIT]] = Field(
        default=None,
        description=(
            "Defines the parameterization of biphase (self-self interaction) "
            "(SWAN default: ELDEBERKY)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "TRIAD DCTA"
        if self.trfac is not None:
            repr += f" trfac={self.trfac}"
        if self.p is not None:
            repr += f" p={self.p}"
        if self.noncolinear:
            repr += " NONC"
        else:
            repr += " COLL"
        if self.biphase is not None:
            repr += f" {self.biphase.render()}"
        return repr


class TRIAD_LTA(BaseComponent):
    """Triad interactions with the LTA method of Eldeberky (1996).

    .. code-block:: text

        TRIAD LTA [trfac] [cutfr] BIPHHASE ELDEBERKY|DEWIT

    References
    ----------
    Eldeberky, Y., Polnikov, V. and Battjes, J.A., 1996. A statistical approach for
    modeling triad interactions in dispersive waves. In Coastal Engineering 1996
    (pp. 1088-1101).

    Note
    ----
    This method to compute the triad interactions is only supported in SWAN >= 41.45.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import TRIAD_LTA
        triad = TRIAD_LTA()
        print(triad.render())
        triad = TRIAD_LTA(
            trfac=0.8,
            cutfr=2.5,
            biphase={"model_type": "eldeberky", "urcrit": 0.63},
        )
        print(triad.render())

    """

    model_type: Literal["lta", "LTA"] = Field(
        default="lta", description="Model type discriminator"
    )
    trfac: Optional[float] = Field(
        default=None,
        description=(
            "Scaling factor that controls the intensity of "
            "the triad interaction due to LTA (SWAN default: 0.8)"
        ),
    )
    cutfr: Optional[float] = Field(
        default=None,
        description=(
            "Controls the maximum frequency that is considered in the LTA "
            "computation. The value of `cutfr` is the ratio of this maximum "
            "frequency over the mean frequency (SWAN default: 2.5)"
        ),
    )
    biphase: Optional[Union[ELDEBERKY, DEWIT]] = Field(
        default=None,
        description=(
            "Defines the parameterization of biphase (self-self interaction) "
            "(SWAN default: ELDEBERKY)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "TRIAD LTA"
        if self.trfac is not None:
            repr += f" trfac={self.trfac}"
        if self.cutfr is not None:
            repr += f" cutfr={self.cutfr}"
        if self.biphase is not None:
            repr += f" {self.biphase.render()}"
        return repr


class TRIAD_SPB(BaseComponent):
    """Triad interactions with the SPB method of Becq-Girard et al. (1999).

    .. code-block:: text

        TRIAD SPB [trfac] [a] [b] BIPHHASE ELDEBERKY|DEWIT

    References
    ----------
    Becq-Girard, F., Forget, P. and Benoit, M., 1999. Non-linear propagation of
    unidirectional wave fields over varying topography. Coastal Engineering, 38(2),
    pp.91-113.

    Note
    ----
    This method to compute the triad interactions is only supported in SWAN >= 41.45.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import TRIAD_SPB
        triad = TRIAD_SPB()
        print(triad.render())
        triad = TRIAD_SPB(
            trfac=0.9,
            a=0.95,
            b=0.0,
            biphase={"model_type": "eldeberky", "urcrit": 0.63},
        )
        print(triad.render())

    """

    model_type: Literal["spb", "SPB"] = Field(
        default="spb", description="Model type discriminator"
    )
    trfac: Optional[float] = Field(
        default=None,
        description=(
            "Scaling factor that controls the intensity of "
            "the triad interaction due to SPB (SWAN default: 0.9)"
        ),
    )
    a: Optional[float] = Field(
        default=None,
        description=(
            "First calibration parameter for tuning K in Eq. (5.1) of "
            "Becq-Girard et al. (1999). This parameter is associated with broadening "
            "of the resonance condition. The default value is 0.95 and is calibrated "
            "by means of laboratory experiments (SWAN default: 0.95)"
        ),
    )
    b: Optional[float] = Field(
        default=None,
        description=(
            "Second calibration parameter for tuning K in Eq. (5.1) of "
            "Becq-Girard et al. (1999). This parameter is associated with broadening "
            "of the resonance condition. The default value is -0.75 and is calibrated "
            "by means of laboratory experiments. However, it may not be appropriate "
            "for true 2D field cases as it does not scale with the wave field "
            "characteristics. Hence, this parameter is set to zero (SWAN default: 0.0)"
        ),
    )
    biphase: Optional[Union[ELDEBERKY, DEWIT]] = Field(
        default=None,
        description=(
            "Defines the parameterization of biphase (self-self interaction) "
            "(SWAN default: ELDEBERKY)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "TRIAD SPB"
        if self.trfac is not None:
            repr += f" trfac={self.trfac}"
        if self.a is not None:
            repr += f" a={self.a}"
        if self.b is not None:
            repr += f" b={self.b}"
        if self.biphase is not None:
            repr += f" {self.biphase.render()}"
        return repr


# =====================================================================================
# VEGETATION
# =====================================================================================
class VEGETATION(BaseComponent):
    """Vegetation dumping.

    .. code-block:: text

        VEGETATION [iveg] < [height] [diamtr] [nstems] [drag] >

    With this command the user can activate wave damping due to vegetation based on the
    Dalrymple's formula (1984) as implemented by Suzuki et al. (2011). This damping is
    uniform over the wave frequencies. An alternative is the frequency-dependent
    (canopy) dissipation model of Jacobsen et al. (2019). If this command is not used,
    SWAN will not account for vegetation effects.

    The vegetation (rigid plants) can be divided over a number of vertical segments and
    so, the possibility to vary the vegetation vertically is included. Each vertical
    layer represents some characteristics of the plants. These variables as indicated
    below can be repeated as many vertical layers to be chosen.

    References
    ----------
    Dalrymple, R.A., Kirby, J.T. and Hwang, P.A., 1984. Wave diffraction due to areas
    of energy dissipation. Journal of waterway, port, coastal, and ocean engineering,
    110(1), pp.67-79.

    Jacobsen, N.G., Bakker, W., Uijttewaal, W.S. and Uittenbogaard, R., 2019.
    Experimental investigation of the wave-induced motion of and force distribution
    along a flexible stem. Journal of Fluid Mechanics, 880, pp.1036-1069.

    Suzuki, T., Zijlema, M., Burger, B., Meijer, M.C. and Narayan, S., 2012. Wave
    dissipation by vegetation with layer schematization in SWAN. Coastal Engineering,
    59(1), pp.64-71.

    Notes
    -----
    Vertical layering of the vegetation is not yet implemented for the
    Jacobsen et al. (2019) method.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import VEGETATION
        # Single layer
        vegetation = VEGETATION(
            height=1.2,
            diamtr=0.1,
            drag=0.5,
            nstems=10,
        )
        print(vegetation.render())
        # 2 vertical layers
        vegetation = VEGETATION(
            iveg=1,
            height=[1.2, 0.8],
            diamtr=[0.1, 0.1],
            drag=[0.5, 0.5],
            nstems=[10, 5],
        )
        print(vegetation.render())

    """

    model_type: Literal["vegetation", "VEGETATION"] = Field(
        default="vegetation", description="Model type discriminator"
    )
    iveg: Literal[1, 2] = Field(
        default=1,
        description=(
            "Indicates the method for the vegetation computation (SWAN default: 1):\n"
            "\n* 1: Suzuki et al. (2011)\n* 2: Jacobsen et al. (2019)\n"
        ),
    )
    height: Union[float, list[float]] = Field(
        description="The plant height per layer (in m)"
    )
    diamtr: Union[float, list[float]] = Field(
        description="The diameter of each plant stand per layer (in m)"
    )
    drag: Union[float, list[float]] = Field(
        description="The drag coefficient per layer"
    )
    nstems: Union[int, list[int]] = Field(
        default=1,
        description=(
            "The number of plant stands per square meter for each layer. Note that "
            "`nstems` is allowed to vary over the computational region to account for "
            "the zonation of vegetation. In that case use the commands "
            "`IMPGRID NPLANTS` and `READINP NPLANTS` to define and read the "
            "vegetation density. The (vertically varying) value of `nstems` in this "
            "command will be multiplied by this horizontally varying plant density "
            "(SWAN default: 1)"
        ),
        validate_default=True,
    )

    @field_validator("height", "diamtr", "drag", "nstems")
    @classmethod
    def number_of_layers(cls, v: Any, info: ValidationInfo) -> Any:
        if v is None:
            return v
        elif not isinstance(v, list):
            v = [v]
        sizes = {k: len(v) for k, v in info.data.items() if isinstance(v, list)}
        if len(set(sizes.values())) > 1:
            raise ValueError(
                "The number of layers must be the same for all variables. "
                f"Got these number of layers: {sizes}"
            )
        return v

    @model_validator(mode="after")
    def jacomsen_layering_not_implemented(self) -> "VEGETATION":
        if self.iveg == 2 and len(self.nstems) > 1:
            raise NotImplementedError(
                "Vertical layering of the vegetation is not yet implemented for the "
                "Jacobsen et al. (2019) method, please define single layer"
            )
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"VEGETATION iveg={self.iveg}"
        for h, d, dr, n in zip(self.height, self.diamtr, self.drag, self.nstems):
            repr += f" height={h} diamtr={d} nstems={n} drag={dr}"
        return repr


# =====================================================================================
# MUD
# =====================================================================================
class MUD(BaseComponent):
    """Mud dumping.

    .. code-block:: text

        MUD [layer] [rhom] [viscm]

    With this command the user can activate wave damping due to mud based on Ng (2000).
    If this command or the commands INPGRID MUDLAY and READINP MUDLAY are not used,
    SWAN will not account for muddy bottom effects.

    References
    ----------
    Ng, C., 2000, Water waves over a muddy bed: A two layer Stokes' boundary layer
    model, Coastal Eng., 40, 221-242.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import MUD
        mud = MUD()
        print(mud.render())
        mud = MUD(
            layer=2.0,
            rhom=1300,
            viscm=0.0076,
        )
        print(mud.render())

    TODO: Validate `layer` must be prescribed if `INPGRID MUDLAY` isn't used.

    """

    model_type: Literal["mud", "MUD"] = Field(
        default="mud", description="Model type discriminator"
    )
    layer: Optional[float] = Field(
        default=None,
        description=(
            "The thickness of the mud layer (in m). Note that `layer` is allowed to "
            "vary over the computational region to account for the zonation of muddy "
            "bottom. In that case use the commands `INPGRID MUDLAY` and `READINP "
            "MUDLAY` to define and read the layer thickness of mud. The value of "
            "`layer` in this command is then not required (it will be ignored)"
        ),
    )
    rhom: Optional[float] = Field(
        default=None,
        description="The density of the mud layer (in kg/m3) (SWAN default: 1300)",
    )
    viscm: Optional[float] = Field(
        default=None,
        description=(
            "The kinematic viscosity of the mud layer (in m2/s) (SWAN default: 0.0076)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "MUD"
        if self.layer is not None:
            repr += f" layer={self.layer}"
        if self.rhom is not None:
            repr += f" rhom={self.rhom}"
        if self.viscm is not None:
            repr += f" viscm={self.viscm}"
        return repr


# =====================================================================================
# SICE
# =====================================================================================
class SICE(BaseComponent):
    """Sea ice dissipation.

    .. code-block:: text

        SICE [aice]

    Using this command, the user activates a sink term to represent the dissipation of
    wave energy by sea ice. The default method is R19 empirical/parametric: a
    polynomial based on wave frequency (Rogers, 2019). This polynomial (in 1/m) has
    seven dimensional coefficients; see Scientific/Technical documentation for details.
    If this command is not used, SWAN will not account for sea ice effects.

    References
    ----------
    Doble, M.J., De Carolis, G., Meylan, M.H., Bidlot, J.R. and Wadhams, P., 2015.
    Relating wave attenuation to pancake ice thickness, using field measurements and
    model results. Geophysical Research Letters, 42(11), pp.4473-4481.

    Meylan, M.H., Bennetts, L.G. and Kohout, A.L., 2014. In situ measurements and
    analysis of ocean waves in the Antarctic marginal ice zone. Geophysical Research
    Letters, 41(14), pp.5046-5051.

    Rogers, W.E., Meylan, M.H. and Kohout, A.L., 2018. Frequency distribution of
    dissipation of energy of ocean waves by sea ice using data from Wave Array 3 of
    the ONR “Sea State” field experiment. Nav. Res. Lab. Memo. Rep, pp.18-9801.

    Rogers, W.E., Meylan, M.H. and Kohout, A.L., 2021. Estimates of spectral wave
    attenuation in Antarctic sea ice, using model/data inversion. Cold Regions Science
    and Technology, 182, p.103198.

    Notes
    -----
    Iis also necessary to describe the ice, using the `ICE` command (for uniform and
    stationary ice) or `INPGRID`/`READINP` commands (for variable ice).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SICE
        sice = SICE()
        print(sice.render())
        sice = SICE(aice=0.5)
        print(sice.render())

    TODO: Verify if the `aice` parameter should be used with SICE command, it is not
    shown in the command tree but it is described as an option in the description.

    """

    model_type: Literal["sice", "SICE"] = Field(
        default="sice", description="Model type discriminator"
    )
    aice: Optional[float] = Field(
        default=None,
        description=(
            "Ice concentration as a fraction from 0 to 1. Note that `aice` is allowed "
            "to vary over the computational region to account for the zonation of ice "
            "concentration. In that case use the commands `INPGRID AICE` and `READINP "
            "AICE` to define and read the sea concentration. The value of `aice` in "
            "this command is then not required (it will be ignored)"
        ),
        ge=0.0,
        le=1.0,
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SICE"
        if self.aice is not None:
            repr += f" aice={self.aice}"
        return repr


class SICE_R19(SICE):
    """Sea ice dissipation based on the method of Rogers et al (2019).

    .. code-block:: text

        SICE [aice] R19 [c0] [c1] [c2] [c3] [c4] [c5] [c6]

    The default options recover the polynomial of Meylan et al. (2014), calibrated for
    a case of ice floes, mostly 10 to 25 m in diameter, in the marginal ice zone near
    Antarctica. Examples for other calibrations can be found in the
    Scientific/Technical documentation.

    References
    ----------
    Meylan, M.H., Bennetts, L.G. and Kohout, A.L., 2014. In situ measurements and
    analysis of ocean waves in the Antarctic marginal ice zone. Geophysical Research
    Letters, 41(14), pp.5046-5051.

    Rogers, W.E., Meylan, M.H. and Kohout, A.L., 2018. Frequency distribution of
    dissipation of energy of ocean waves by sea ice using data from Wave Array 3 of
    the ONR “Sea State” field experiment. Nav. Res. Lab. Memo. Rep, pp.18-9801.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SICE_R19
        sice = SICE_R19()
        print(sice.render())
        kwargs = dict(
            aice=0.5,
            c0=0.0,
            c1=0.0,
            c2=1.06e-3,
            c3=0.0,
            c4=0.0,
            c5=0.0,
            c6=0.0,
        )
        sice = SICE_R19(**kwargs)
        print(sice.render())

    """

    model_type: Literal["r19", "R19"] = Field(
        default="r19", description="Model type discriminator"
    )
    c0: Optional[float] = Field(
        default=None,
        description=(
            "Polynomial coefficient (in 1/m) for determining the rate of sea ice "
            "dissipation (SWAN default: 0.0)"
        ),
    )
    c1: Optional[float] = Field(
        default=None,
        description=(
            "Polynomial coefficient (in s/m) for determining the rate of sea ice "
            "dissipation (SWAN default: 0.0)"
        ),
    )
    c2: Optional[float] = Field(
        default=None,
        description=(
            "Polynomial coefficient (in s2/m) for determining the rate of sea ice "
            "dissipation (SWAN default: 1.06E-3)"
        ),
    )
    c3: Optional[float] = Field(
        default=None,
        description=(
            "Polynomial coefficient (in s3/m) for determining the rate of sea ice "
            "dissipation (SWAN default: 0.0)"
        ),
    )
    c4: Optional[float] = Field(
        default=None,
        description=(
            "Polynomial coefficient (in s4/m) for determining the rate of sea ice "
            "dissipation (SWAN default: 2.3E-2)"
        ),
    )
    c5: Optional[float] = Field(
        default=None,
        description=(
            "Polynomial coefficient (in s5/m) for determining the rate of sea ice "
            "dissipation (SWAN default: 0.0)"
        ),
    )
    c6: Optional[float] = Field(
        default=None,
        description=(
            "Polynomial coefficient (in s6/m) for determining the rate of sea ice "
            "dissipation (SWAN default: 0.0)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} {self.model_type.upper()}"
        if self.c0 is not None:
            repr += f" c0={self.c0}"
        if self.c1 is not None:
            repr += f" c1={self.c1}"
        if self.c2 is not None:
            repr += f" c2={self.c2}"
        if self.c3 is not None:
            repr += f" c3={self.c3}"
        if self.c4 is not None:
            repr += f" c4={self.c4}"
        if self.c5 is not None:
            repr += f" c5={self.c5}"
        if self.c6 is not None:
            repr += f" c6={self.c6}"
        return repr


class SICE_D15(SICE):
    """Sea ice dissipation based on the method of Doble et al. (2015).

    .. code-block:: text

        SICE [aice] D15 [chf]

    References
    ----------
    Doble, M.J., De Carolis, G., Meylan, M.H., Bidlot, J.R. and Wadhams, P., 2015.
    Relating wave attenuation to pancake ice thickness, using field measurements and
    model results. Geophysical Research Letters, 42(11), pp.4473-4481.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SICE_D15
        sice = SICE_D15()
        print(sice.render())
        sice = SICE_D15(aice=0.2, chf=0.1)
        print(sice.render())

    """

    model_type: Literal["d15", "D15"] = Field(
        default="d15", description="Model type discriminator"
    )
    chf: Optional[float] = Field(
        default=None,
        description="A simple coefficient of proportionality (SWAN default: 0.1)",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} {self.model_type.upper()}"
        if self.chf is not None:
            repr += f" chf={self.chf}"
        return repr


class SICE_M18(SICE):
    """Sea ice dissipation based on the method of Meylan et al. (2018).

    .. code-block:: text

        SICE [aice] M18 [chf]

    References
    ----------
    Meylan, M.H., Bennetts, L.G. and Kohout, A.L., 2014. In situ measurements and
    analysis of ocean waves in the Antarctic marginal ice zone. Geophysical Research
    Letters, 41(14), pp.5046-5051.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SICE_M18
        sice = SICE_M18()
        print(sice.render())
        sice = SICE_M18(aice=0.8, chf=0.059)
        print(sice.render())

    """

    model_type: Literal["m18", "M18"] = Field(
        default="m18", description="Model type discriminator"
    )
    chf: Optional[float] = Field(
        default=None,
        description="A simple coefficient of proportionality (SWAN default: 0.059)",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} {self.model_type.upper()}"
        if self.chf is not None:
            repr += f" chf={self.chf}"
        return repr


class SICE_R21B(SICE):
    """Sea ice dissipation based on the method of Rogers et al. (2021).

    .. code-block:: text

        SICE [aice] R21B [chf] [npf]

    References
    ----------
    Rogers, W.E., Meylan, M.H. and Kohout, A.L., 2021. Estimates of spectral wave
    attenuation in Antarctic sea ice, using model/data inversion. Cold Regions Science
    and Technology, 182, p.103198.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SICE_R21B
        sice = SICE_R21B()
        print(sice.render())
        sice = SICE_R21B(aice=0.8, chf=2.9, npf=4.5)
        print(sice.render())

    """

    model_type: Literal["r21b", "R21B"] = Field(
        default="r21b", description="Model type discriminator"
    )
    chf: Optional[float] = Field(
        default=None,
        description="A simple coefficient of proportionality (SWAN default: 2.9)",
    )
    npf: Optional[float] = Field(
        default=None,
        description=(
            "Controls the degree of dependence on frequency and ice thickness "
            "(SWAN default: 4.5)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} {self.model_type.upper()}"
        if self.chf is not None:
            repr += f" chf={self.chf}"
        if self.npf is not None:
            repr += f" npf={self.npf}"
        return repr


# =====================================================================================
# TURBULENCE
# =====================================================================================
class TURBULENCE(BaseComponent):
    """Turbulent viscosity.

    .. code-block:: text

        TURBULENCE [ctb] (CURRENT [tbcur])

    With this optional command the user can activate turbulent viscosity. This physical
    effect is also activated by reading values of the turbulent viscosity using the
    `READGRID TURB` command, but then with the default value of `ctb`. The command
    `READGRID TURB` is necessary if this command `TURB` is used since the value of the
    viscosity is assumed to vary over space.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import TURBULENCE
        turbulence = TURBULENCE(current=False)
        print(turbulence.render())
        turbulence = TURBULENCE(ctb=0.01, current=True, tbcur=0.004)
        print(turbulence.render())

    """

    model_type: Literal["turbulence", "TURBULENCE"] = Field(
        default="turbulence", description="Model type discriminator"
    )
    ctb: Optional[float] = Field(
        default=None,
        description=(
            "The value of the proportionality coefficient appearing in the energy "
            "dissipation term (SWAN default: 0.01)"
        ),
    )
    current: Optional[bool] = Field(
        default=True,
        description=(
            "If this keyword is present the turbulent viscosity will be derived from "
            "the product of the depth and the absolute value of the current velocity. "
            "If the command `READGRID TURB` is used, this option is ignored; "
            "the values read from file will prevail"
        ),
    )
    tbcur: Optional[float] = Field(
        default=None,
        description=(
            "The factor by which depth x current velocity is multiplied in order to "
            "get the turbulent viscosity (SWAN default: 0.004)"
        ),
    )

    @model_validator(mode="after")
    def tbcur_only_with_current(self) -> "TURBULENCE":
        if self.current == False and self.tbcur is not None:
            raise ValueError("`tbcur` can only be defined if `current` is True")
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "TURBULENCE"
        if self.ctb is not None:
            repr += f" ctb={self.ctb}"
        if self.current:
            repr += " CURRENT"
        if self.tbcur is not None:
            repr += f" tbcur={self.tbcur}"
        return repr


# =====================================================================================
# BRAGG
# =====================================================================================
class BRAGG(BaseComponent):
    """Bragg scattering.

    .. code-block:: text

        BRAGG [ibrag] [nreg] [cutoff]

    Using this optional command, the user activates a source term to represent the
    scattering of waves due to changes in the small-scale bathymetry based on the
    theory of Ardhuin and Herbers (2002). If this command is not used, SWAN will not
    account for Bragg scattering.

    The underlying process is related to the bed elevation spectrum that describes the
    random variability of the bathymetry at the scale of the wave length on top of a
    slowly varying depth. To input this spectrum in the model, two options are
    available. One option is to read a spectrum from a file. This single bottom
    spectrum will subsequently be applied in all active grid points. The assumption
    being made here is that the inputted bottom is gently sloping. Note that the bottom
    spectrum must be given as a function of the wave number `k`.

    Another option is to compute the spectrum by a Fourier transform from `x` to `k` of
    the bed modulations around a computational grid point. First, one must define a
    square region with a fixed size around the grid point in order to perform the
    Fourier transform. The size should correspond to a multiple of the wave length at
    which refraction is resolved (i.e. consistent with the mild slope assumption).
    Next, the amplitude modulation of the small-scale bathymetry is obtained by
    substracting a slowly varying bed level from the inputted high-resolution
    bathymetric data within this square region. Here, the smooth bed level is achieved
    using a bilinear fit. During the computation, however, SWAN employs the gently
    sloping bed as the mean of the original bathymetry within the given square around
    each computational grid point. Finally, the corresponding bottom spectrum is
    computed with an FFT.

    Notes
    -----
    The Bragg scattering source term to the action balance equation gives rise to a
    fairly stiff equation. The best remedy is to run SWAN in the nonstationary mode
    with a relatively small time step or in the stationary mode with some under
    relaxation (see command `NUM STAT [alfa]`).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import BRAGG
        bragg = BRAGG(nreg=200)
        print(bragg.render())
        bragg = BRAGG(ibrag=1, nreg=200, cutoff=5.0)
        print(bragg.render())

    """

    model_type: Literal["bragg", "BRAGG"] = Field(
        default="bragg", description="Model type discriminator"
    )
    ibrag: Optional[Literal[1, 2, 3]] = Field(
        default=None,
        description=(
            "Indicates the computation of Bragg scattering term:\n\n* 1: source term "
            "is calculated per sweep and bottom spectrum is interpolated at the "
            "difference wave number a priori (thus requiring storage)\n* 2: source "
            "term is calculated per sweep and bottom spectrum is interpolated at the "
            "difference wave number per sweep (no storage)\n* 3: source term is "
            "calculated per iteration and bottom spectrum is interpolated at the "
            "difference wave number per iteration (no storage)\n\n(SWAN default: 1)"
        ),
    )
    nreg: int = Field(
        description=(
            "Size of square region around computational grid point (centered) for "
            "computing the mean depth and, if desired, the bed elevation spectrum. It "
            "is expressed in terms of the number of grid points (per direction) "
            "of the inputted bottom grid"
        ),
    )
    cutoff: Optional[float] = Field(
        default=None,
        description=(
            "Cutoff to the ratio between surface and bottom wave numbers. Note: see"
            "the Scientific/Technical documentation for details (SWAN default: 5.0)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "BRAGG"
        if self.ibrag is not None:
            repr += f" ibrag={self.ibrag}"
        if self.nreg is not None:
            repr += f" nreg={self.nreg}"
        if self.cutoff is not None:
            repr += f" cutoff={self.cutoff}"
        return repr


class BRAGG_FT(BRAGG):
    """Bragg scattering with bottom spectrum computed from FFT.

    .. code-block:: text

        BRAGG [ibrag] [nreg] [cutoff] FT

    If this keyword is present the bottom spectrum will be computed in each active
    grid point using a Fast Fourier Transform (FFT).

    Notes
    -----
    The depth in each computational grid point is computed as the average of the
    inputted (high-resolution) bed levels within the square region.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import BRAGG_FT
        bragg = BRAGG_FT(nreg=350)
        print(bragg.render())
        bragg = BRAGG_FT(ibrag=2, nreg=350, cutoff=5.0)
        print(bragg.render())

    """

    model_type: Literal["ft", "FT"] = Field(
        default="ft", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        return f"{super().cmd()} FT"


class BRAGG_FILE(BRAGG):
    """Bragg scattering with bottom spectrum from file.

    .. code-block:: text

        BRAGG [ibrag] [nreg] [cutoff] FILE 'fname' [idla] [mkx] [mky] [dkx] [dky]

    The bed elevation spectrum `FB(kx, ky)` is read from a file.

    Notes
    -----
    This spectrum is taken to be uniform over the entire computational domain.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import BRAGG_FILE
        bragg = BRAGG_FILE(fname="bottom_spectrum.txt", nreg=500, mkx=99, dkx=0.1)
        print(bragg.render())
        kwargs = dict(
            ibrag=3,
            nreg=500,
            cutoff=5.0,
            fname="bottom_spectrum.txt",
            mkx=99,
            mky=149,
            dkx=0.1,
            dky=0.1,
        )
        bragg = BRAGG_FILE(**kwargs)
        print(bragg.render())

    """

    model_type: Literal["file", "FILE"] = Field(
        default="file", description="Model type discriminator"
    )
    fname: str = Field(
        description="Name of file containing the bottom spectrum",
        max_length=36,
    )
    idla: Optional[IDLA] = Field(
        default=None,
        description=("Order in which the values should be given in the input files"),
    )
    mkx: int = Field(
        description=(
            "Number of cells in x-direction of the wave number grid related to bottom "
            "spectrum (this is one less than the number of points in this direction)"
        ),
    )
    mky: Optional[int] = Field(
        default=None,
        description=(
            "Number of cells in y-direction of the wave number grid related to bottom "
            "spectrum (this is one less than the number of points in this direction)"
            "(SWAN default: `mky = mkx`)"
        ),
    )
    dkx: float = Field(
        description=(
            "Mesh size in x-direction of the wave number grid related to bottom "
            "spectrum (1/m)"
        ),
    )
    dky: Optional[float] = Field(
        default=None,
        description=(
            "Mesh size in y-direction of the wave number grid related to bottom "
            "spectrum (1/m) (SWAN default: `dky = dkx`)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"{super().cmd()} FILE fname='{self.fname}'"
        if self.idla is not None:
            repr += f" idla={self.idla.value}"
        repr += f" mkx={self.mkx}"
        if self.mky is not None:
            repr += f" mky={self.mky}"
        repr += f" dkx={self.dkx}"
        if self.dky is not None:
            repr += f" dky={self.dky}"
        return repr


# =====================================================================================
# LIMITER
# =====================================================================================
class LIMITER(BaseComponent):
    """Physics limiter.

    .. code-block:: text

        LIMITER [ursell] [qb]

    With this command the user can de-activate permanently the quadruplets when
    the actual Ursell number exceeds `ursell`. Moreover, as soon as the actual
    fraction of breaking waves exceeds `qb` then the action limiter will not be
    used in case of decreasing action density.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import LIMITER
        limiter = LIMITER()
        print(limiter.render())
        limiter = LIMITER(ursell=10.0, qb=1.0)
        print(limiter.render())

    """

    model_type: Literal["limiter", "LIMITER"] = Field(
        default="limiter", description="Model type discriminator"
    )
    ursell: Optional[float] = Field(
        default=None,
        description=("The upper threshold for Ursell number (SWAN default: 10.0)"),
    )
    qb: Optional[float] = Field(
        default=None,
        description="The threshold for fraction of breaking waves (SWAN default: 1.0)",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "LIMITER"
        if self.ursell is not None:
            repr += f" ursell={self.ursell}"
        if self.qb is not None:
            repr += f" qb={self.qb}"
        return repr


# =====================================================================================
# OBSTACLE
# =====================================================================================
TRANSMISSION_TYPE = Annotated[
    Union[TRANSM, TRANS1D, TRANS2D, GODA, DANGREMOND],
    Field(description="Wave transmission", discriminator="model_type"),
]
REFLECTION_TYPE = Annotated[
    Union[RSPEC, RDIFF],
    Field(description="Wave reflection type", discriminator="model_type"),
]


class OBSTACLE(BaseComponent):
    """Subgrid obstacle.

    .. code-block:: text

        OBSTACLE ->TRANSM|TRANS1D|TRANS2D|GODA|DANGREMOND REFL [reflc] ->RSPEC|RDIFF &
            (FREEBOARD [hgt] [gammat] [gammar] QUAY) LINE < [xp] [yp] >

    With this optional command the user provides the characteristics of a (line
    of) sub-grid obstacle(s) through which waves are transmitted or against which
    waves are reflected (possibly both at the same time). The obstacle is sub-grid
    in the sense that it is narrow compared to the spatial meshes; its length should
    be at least one mesh length.

    The location of the obstacle is defined by a sequence of corner points of a line.
    The obstacles interrupt the propagation of the waves from one grid point to the
    next wherever this obstacle line is located between two neighbouring grid points
    (of the computational grid; the resolution of the obstacle is therefore equal to
    the computational grid spacing). This implies that an obstacle to be effective must
    be located such that it crosses at least one grid line. This is always the case
    when an obstacle is larger than one mesh length.

    Notes
    -----

    * The advise is to define obstacles with the least amount of points possible.
    * SWAN checks if the criterion `reflc^2 + trcoef^2 LE 1` is fulfilled.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import OBSTACLE
        obs = OBSTACLE(
            transmission=dict(model_type="transm", trcoef=0.5),
            reflection=dict(reflc=0.5),
            line=dict(xp=[174.1, 174.2, 174.3], yp=[-39.1, -39.1, -39.1]),
        )
        print(obs.render())

    """

    model_type: Literal["obstacle", "OBSTACLE"] = Field(
        default="obstacle", description="Model type discriminator"
    )
    transmission: Optional[TRANSMISSION_TYPE] = Field(default=None)
    reflection: Optional[REFL] = Field(default=None, description="Wave reflection")
    reflection_type: Optional[REFLECTION_TYPE] = Field(default=None)
    freeboard: Optional[FREEBOARD] = Field(default=None, description="Freeboard")
    line: LINE = Field(default=None, description="Line of obstacle")

    @model_validator(mode="after")
    def hgt_consistent(self) -> "OBSTACLE":
        """Warns if `hgt` has different values in DAM and FREEBOARD specifications."""
        if self.transmission is not None and self.freeboard is not None:
            is_dam = self.transmission.model_type.upper() in ["GODA", "DANGREMOND"]
            if is_dam and self.freeboard.hgt != self.transmission.hgt:
                logger.warning("hgt in FREEBOARD and DAM specifications are not equal")
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "OBSTACLE"
        if self.transmission is not None:
            repr += f" {self.transmission.render()}"
        if self.reflection:
            repr += f" {self.reflection.render()}"
        if self.reflection_type is not None:
            repr += f" {self.reflection_type.render()}"
        if self.freeboard is not None:
            repr += f" {self.freeboard.render()}"
        repr += f" {self.line.render()}"
        return repr


class OBSTACLE_FIG(BaseComponent):
    """Obstacle for free infragravity radiation.

    .. code-block:: text

        OBSTACLE FIG [alpha1] [hss] [tss] (REFL [reflc]) LINE <[xp] [yp]>

    With this optional command the user specifies the obstacles along which the
    free infra-gravity (FIG) energy is radiated. By placing the obstacles close to
    the shorelines SWAN will include the FIG source term along the coastlines
    according to the parametrization of Ardhuin et al. (2014).

    The location of the obstacle is defined by a sequence of corner points of a line.
    For an obstacle line to be effective its length is at least one mesh size large. It
    is recommended to place the obstacles at the inner area of the computational grid,
    not at or through the boundaries. In particular, each obstacle line must be
    bordered by wet points on both sides.

    In addition, the orientation of the obstacle line determines from which side of the
    obstacle the FIG wave energy is radiated away. If the begin point of the line is
    below or left of the end point, that is, pointing upwards/to the right, then FIG
    energy is radiated from the west/north side of the line. If the begin point is
    above or right of the end point (pointing downwards/to the left), then FIG energy
    is radiated away from the east/south side of the obstacle line.

    References
    ----------
    Ardhuin, F., Rawat, A. and Aucan, J., 2014. A numerical model for free
    infragravity waves: Definition and validation at regional and global scales.
    Ocean Modelling, 77, pp.20-32.

    Notes
    -----
    Either `hss` or `tss` or both are allowed to vary over the computational domain.
    In that case use the commands `INPGRID HSS` and `READINP HSS` and/or the commands
    `INPGRID TSS` and `READINP TSS` to define and read the sea-swell wave height/period
    It is permissible to have constant sea-swell height and non-constant sea-swell
    period, or vice versa. The command `OBST FIG` is still required to define the
    obstacles. The values of `hss` and/or `tss` in this command are then not required
    (they will be ignored).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import OBSTACLE_FIG
        obs = OBSTACLE_FIG(
            alpha1=5e-4,
            hss=2.5,
            tss=10.3,
            line=dict(xp=[174.1, 174.2, 174.3], yp=[-39.1, -39.1, -39.1]),
        )
        print(obs.render())
        obs = OBSTACLE_FIG(
            alpha1=5e-4,
            hss=2.5,
            tss=10.3,
            reflection=dict(reflc=0.5),
            line=dict(xp=[174.1, 174.2, 174.3], yp=[-39.1, -39.1, -39.1]),
        )
        print(obs.render())

    """

    model_type: Literal["fig", "FIG"] = Field(
        default="fig", description="Model type discriminator"
    )
    alpha1: float = Field(
        description=(
            "Calibration parameter (in 1/s) for determining the rate of radiating FIG "
            "energy from the shorelines, values in Table 1 of Ardhuin et al. (2014) "
            "are between 4e-4 and 8.1e-4"
        ),
    )
    hss: float = Field(
        description="The sea-swell significant wave height (in m)",
        ge=0.0,
    )
    tss: float = Field(
        description="The sea-swell mean wave period (in s)",
        ge=0.0,
    )
    reflection: Optional[REFL] = Field(default=None, description="Wave reflection")
    line: LINE = Field(description="Line of obstacle")

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"OBSTACLE FIG alpha1={self.alpha1} hss={self.hss} tss={self.tss}"
        if self.reflection:
            repr += f" {self.reflection.render()}"
        repr += f" {self.line.render()}"
        return repr


OBSTACLES_TYPE = Annotated[
    Union[OBSTACLE, OBSTACLE_FIG],
    Field(discriminator="model_type"),
]


class OBSTACLES(BaseComponent):
    """List of swan obstacles.

    .. code-block:: text

        OBSTACLE ... LINE < [xp] [yp] >
        OBSTACLE ... LINE < [xp] [yp] >
        .

    This group component is a convenience to allow defining and rendering
    a list of obstacle components.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import OBSTACLES, OBSTACLE, OBSTACLE_FIG
        obst1 = dict(
            model_type="obstacle",
            reflection=dict(reflc=1.0),
            line=dict(xp=[174.1, 174.2, 174.3], yp=[-39.1, -39.1, -39.1]),
        )
        obst2 = OBSTACLE(
            transmission=dict(model_type="transm"),
            line=dict(xp=[174.3, 174.3], yp=[-39.1, -39.2]),
        )
        obst3 = OBSTACLE_FIG(
            alpha1=5e-4,
            hss=2.5,
            tss=10.3,
            line=dict(xp=[174.1, 174.2, 174.3], yp=[-39.1, -39.1, -39.1]),
        )
        obstacles = OBSTACLES(obstacles=[obst1, obst2, obst3])
        for obst in obstacles.render():
            print(obst)

    """

    model_type: Literal["obstacles", "OBSTACLES"] = Field(
        default="obstacles", description="Model type discriminator"
    )
    obstacles: list[OBSTACLES_TYPE] = Field(description="List of obstacles")

    def cmd(self) -> list:
        """Command file strings for this component."""
        repr = []
        for obstacle in self.obstacles:
            repr += [obstacle.cmd()]
        return repr

    def render(self) -> str:
        """Override base class to allow rendering list of components."""
        cmds = []
        for cmd in self.cmd():
            cmds.append(super().render(cmd))
        return cmds


# =====================================================================================
# SETUP
# =====================================================================================
class SETUP(BaseComponent):
    """Wave setup.

    .. code-block:: text

        SETUP [supcor]

    If this command is given, the wave-induced set-up is computed and accounted for in
    the wave computations (during the computation it is added to the depth that is
    obtained from the `READ BOTTOM` and `READ WLEVEL` commands). This approximation in
    SWAN can only be applied to open coast (unlimited supply of water from outside the
    domain, e.g. nearshore coasts) in contrast to closed basin, e.g. lakes and
    estuaries, where this option should not be used. Note that set-up is not computed
    correctly with spherical coordinates.

    Notes
    -----

    * The SETUP command cannot be used in case of unstructured grids.
    * Set-up is not supported in case of parallel runs using either MPI or OpenMP.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import SETUP
        setup = SETUP()
        print(setup.render())
        setup = SETUP(supcor=0.5)
        print(setup.render())

    """

    model_type: Literal["setup", "SETUP"] = Field(
        default="setup", description="Model type discriminator"
    )
    supcor: Optional[float] = Field(
        default=None,
        description=(
            "By default the wave-induced set-up is computed with a constant added "
            "such that the set-up is zero in the deepest point in the computational "
            "grid. The user can modify this constant by the value of `supcor`. The "
            "user can thus impose a set-up in any one point (and only one) in the "
            "computational grid by first running SWAN, then reading the set-up in "
            "that point and adding or subtracting the required value of `supcor` "
            "(in m; positive if the set-up has to rise) (SWAN default: 0.0)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SETUP"
        if self.supcor is not None:
            repr += f" supcor={self.supcor}"
        return repr


# =====================================================================================
# DIFFRACTION
# =====================================================================================
class DIFFRACTION(BaseComponent):
    """Wave diffraction.

    .. code-block:: text

        DIFFRACTION [idiffr] [smpar] [smnum] [cgmod]

    If this optional command is given, the diffraction is included in the wave
    computation. But the diffraction approximation in SWAN does not properly handle
    diffraction in harbours or in front of reflecting obstacles (see
    Scientific/Technical documentation). Behind breakwaters with a down-wave beach, the
    SWAN results seem reasonable. The spatial resolution near (the tip of) the
    diffraction obstacle should be 1/5 to 1/10 of the dominant wave length.

    Notes
    -----
    Without extra measures, the diffraction computations with SWAN often converge
    poorly or not at all. Two measures can be taken:

    1. (RECOMMENDED) The user can request under-relaxation. See command `NUMERIC`
    parameter `alpha` and Scientific/Technical documentation (Eq. (3.31)). Very limited
    experience suggests `alpha = 0.01`.

    2. Alternatively, the user can request smoothing of the wave field for the
    computation of the diffraction parameter (the wave field remains intact for all
    other computations and output). This is done with a repeated convolution filtering.

    Examples
    --------

    .. ipython:: python

        from rompy.swan.components.physics import DIFFRACTION
        diffraction = DIFFRACTION()
        print(diffraction.render())
        diffraction = DIFFRACTION(idiffr=True, smpar=0.0, smnum=1.0)
        print(diffraction.render())

    """

    model_type: Literal["diffraction", "DIFFRACTION"] = Field(
        default="diffraction", description="Model type discriminator"
    )
    idiffr: Optional[bool] = Field(
        default=None,
        description=(
            "Indicates the use of diffraction. If `idiffr=0` then no diffraction is "
            "taken into account (SWAN default: 1)"
        ),
    )
    smpar: Optional[float] = Field(
        default=None,
        description=(
            "Smoothing parameter for the calculation of ∇ · √Etot. During every "
            "smoothing step all grid points exchange `smpar` times the energy with "
            "their neighbours. Note that `smpar` is parameter a in the above text "
            "(SWAN default: 0.0)"
        ),
    )
    smnum: Optional[int] = Field(
        default=None,
        description="Number of smoothing steps relative to `smpar` (SWAN default: 0)",
    )
    cgmod: Optional[float] = Field(
        default=None,
        description=(
            "Adaption of propagation velocities in geographic space due to "
            "diffraction. If `cgmod=0` then no adaption (SWAN default: 1.0)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "DIFFRACTION"
        if self.idiffr is not None:
            repr += f" idiffr={int(self.idiffr)}"
        if self.smpar is not None:
            repr += f" smpar={self.smpar}"
        if self.smnum is not None:
            repr += f" smnum={self.smnum}"
        if self.cgmod is not None:
            repr += f" cgmod={self.cgmod}"
        return repr


# =====================================================================================
# SURFBEAT
# =====================================================================================
class SURFBEAT(BaseComponent):
    """Surfbeat.

    .. code-block:: text

        SURFBEAT [df] [nmax] [emin] UNIFORM/LOGARITHMIC

    Using this optional command, the user activates the Infragravity Energy Module
    (IEM) of Reniers and Zijlema (2022). Besides the energy balance equation for a
    sea-swell wave field, another energy balance is included to account for the
    transfer of sea-swell energy to the bound infragravity (BIG) wave. This
    infragravity energy balance also involves a nonlinear transfer, expressed by the
    biphase, through the phase coupling between the radiation stress forcing and the
    BIG wave. For the prediction of the biphase for obliquely incident waves, an
    evolution equation is provided under the assumption that the bottom slopes are mild
    and alongshore uniform.

    References
    ----------
    Reniers, A. and Zijlema, M., 2022. Swan surfbeat-1d. Coastal Engineering, 172,
    p.104068.

    Examples:
    ---------

    .. ipython:: python

        from rompy.swan.components.physics import SURFBEAT
        surfbeat = SURFBEAT()
        print(surfbeat.render())
        surfbeat = SURFBEAT(df=0.01, nmax=50000, emin=0.05, spacing="logarithmic")
        print(surfbeat.render())

    """

    model_type: Literal["surfbeat", "SURFBEAT"] = Field(
        default="surfbeat", description="Model type discriminator"
    )
    df: Optional[float] = Field(
        default=None,
        description=(
            "The constant size of BIG frequency bin (in Hz) (SWAN default: 0.01)"
        ),
        ge=0.0,
    )
    nmax: Optional[int] = Field(
        default=None,
        description=(
            "The maximum number of short-wave pairs for creating bichromatic wave "
            "groups (SWAN default: 50000)"
        ),
        ge=0,
    )
    emin: Optional[float] = Field(
        default=None,
        description=(
            "The energy threshold in fraction of energy spectrum peak. With this "
            "threshold one takes into account those short wave components to create "
            "bichromatic wave groups while their energy levels are larger than "
            "`emin x E_max` with `E_max` the peak of the spectrum (SWAN default: 0.05)"
        ),
    )
    spacing: Optional[Literal["uniform", "logarithmic"]] = Field(
        default=None,
        description=(
            "Define if frequencies for reflected ig waves are uniformly or "
            "logarithmically distributed"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SURFBEAT"
        if self.df is not None:
            repr += f" df={self.df}"
        if self.nmax is not None:
            repr += f" nmax={self.nmax}"
        if self.emin is not None:
            repr += f" emin={self.emin}"
        if self.spacing is not None:
            repr += f" {self.spacing.upper()}"
        return repr


# =====================================================================================
# SCAT
# =====================================================================================
class SCAT(BaseComponent):
    """Scattering.

    .. code-block:: text

        SCAT [iqcm] (GRID [rfac]) (TRUNC [alpha] [qmax])

    Using this optional command, the user activates a source term that allows for the
    generation and propagation of cross correlations between scattered waves due to
    variations in the bathymetry and mean currents. Such variations are rapid compared
    to the distancebetween the crossing waves (at the scale of 100-1000 m) and is
    particularly relevant for cases involving narrowband waves (swells) in coastal
    regions with shallow water and ambient currents. In turn, the immediate spatial
    effects of coherent scattering, interference, refraction and diffraction can cause
    large-scale changes in the wave parameters.

    References
    ----------
    Smit, P.B. and Janssen, T.T., 2013. The evolution of inhomogeneous wave statistics
    through a variable medium. Journal of Physical Oceanography, 43(8), pp.1741-1758.

    Smit, P.B., Janssen, T.T. and Herbers, T.H.C., 2015. Stochastic modeling of
    inhomogeneous ocean waves. Ocean Modelling, 96, pp.26-35.

    Smit, P.B., Janssen, T.T. and Herbers, T.H.C., 2015. Stochastic modeling of
    coherent wave fields over variable depth. Journal of Physical Oceanography, 45(4),
    pp.1139-1154.

    Akrish, G., Smit, P., Zijlema, M. and Reniers, A., 2020. Modelling statistical wave
    interferences over shear currents. Journal of Fluid Mechanics, 891, p.A2.

    Notes
    -----
    Implemented in SWAN 41.41.

    If both `alpha` and `qmax` options are provided to truncate the infinite
    convolution sum their mimimum is considered as the final limit on the sum.

    Examples:
    ---------

    .. ipython:: python

        from rompy.swan.components.physics import SCAT
        scat = SCAT()
        print(scat.render())
        scat = SCAT(iqcm=2, rfac=1.0, alpha=1.0)
        print(scat.render())

    """

    model_type: Literal["scat", "SCAT"] = Field(
        default="scat", description="Model type discriminator"
    )
    iqcm: Optional[Literal[0, 1, 2]] = Field(
        default=None,
        description=(
            "Indicates the modelling and computation of QC scattering:\n\n* 0: no "
            "scattering\n* 1: scattering due to non-uniform bathymetry and currents "
            "(the latter only if applicable; see command `INPGRID CURRENT`)\n* 2: "
            "wave-current interaction under the assumption of a slowly varying "
            "bathymetry\n\n(SWAN default: 1)"
        ),
    )
    rfac: Optional[float] = Field(
        default=None,
        description=(
            "The resolution factor through which the incident spectral width is"
            "multiplied (SWAN default: 1.0)"
        ),
        ge=1.0,
    )
    alpha: Optional[float] = Field(
        default=None,
        description=(
            "The coefficient by which the mean wave number is multiplied to set the"
            "limit on the convolution sum (SWAN default: 1.0)"
        ),
    )
    qmax: Optional[float] = Field(
        default=None, description="The maximum scattering wave number (in 1/m)"
    )

    @model_validator(mode="after")
    def warn_if_qmax_and_alpha(self) -> "SCAT":
        if self.qmax is not None and self.alpha is not None:
            logger.warning(
                "Both `alpha` and `qmax` options are provided to truncate the "
                "infinite convolution sum. Their mimimum is considered in SWAN as the "
                "final limit on the sum"
            )
        return self

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SCAT"
        if self.iqcm is not None:
            repr += f" iqcm={self.iqcm}"
        if self.rfac is not None:
            repr += f" GRID rfac={self.rfac}"
        if self.alpha is not None or self.qmax is not None:
            repr += " TRUNC"
            if self.alpha is not None:
                repr += f" alpha={self.alpha}"
            if self.qmax is not None:
                repr += f" qmax={self.qmax}"
        return repr


# =====================================================================================
# OFF
# =====================================================================================
class OFF(BaseComponent):
    """Deactivate physics commands.

    .. code-block:: text

        OFF WINDGROWTH|QUADRUPL|WCAPPING|BREAKING|REFRAC|FSHIFT|BNDCHK

    This command deactivates physics commands. The command can be used to switch off
    the computation of a certain physics component without having to remove the command
    from the input file. This is useful for testing purposes.

    Examples:
    ---------

    .. ipython:: python

        from rompy.swan.components.physics import OFF
        off = OFF(physics="windgrowth")
        print(off.render())

    """

    model_type: Literal["off", "OFF"] = Field(
        default="off", description="Model type discriminator"
    )
    physics: PhysicsOff = Field(description="Physics command to be switched off")

    def cmd(self) -> str:
        """Command file string for this component."""
        return f"OFF {self.physics.value.upper()}"


class OFFS(BaseComponent):
    """Deactivate multiple physics commands.

    .. code-block:: text

        OFF WINDGROWTH|QUADRUPL|WCAPPING|BREAKING|REFRAC|FSHIFT|BNDCHK
        OFF WINDGROWTH|QUADRUPL|WCAPPING|BREAKING|REFRAC|FSHIFT|BNDCHK
        .

    This group component is a convenience to allow defining and rendering
    a list of `OFF` components.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.physics import OFFS
        off1 = dict(physics="windgrowth")
        off2 = dict(physics="wcapping")
        offs = OFFS(offs=[off1, off2])
        for off in offs.render():
            print(off)

    """

    model_type: Literal["offs", "OFFS"] = Field(
        default="offs", description="Model type discriminator"
    )
    offs: list[OFF] = Field(description="Physics commands to deactivate")

    def cmd(self) -> list:
        """Command file strings for this component."""
        repr = []
        for off in self.offs:
            repr += [off.cmd()]
        return repr
