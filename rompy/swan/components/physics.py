"""Model physics components."""
import logging
from typing import Literal, Optional, Union
from pydantic import validator, root_validator, Field, confloat

from rompy.swan.components.base import BaseComponent
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
)


logger = logging.getLogger(__name__)


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


#======================================================================================
# Wave generation GEN1 | GEN2 | GEN3
#======================================================================================
class GEN1(BaseComponent):
    """First generation source terms GEN1.

    `GEN1 [cf10] [cf20] [cf30] [cf40] [edmlpm] [cdrag] [umin] [cfpm]`

    With this command the user indicates that SWAN should run in first-generation mode
    (see Scientific/Technical documentation).

    """

    model_type: Literal["gen1"] = Field(
        default="gen1", description="Model type discriminator"
    )
    cf10: Optional[float] = Field(
        description="Controls the linear wave growth (SWAN default: 188.0)"
    )
    cf20: Optional[float] = Field(
        description="Controls the exponential wave growth (SWAN default: 0.59)"
    )
    cf30: Optional[float] = Field(
        description="Controls the exponential wave growth (SWAN default: 0.12)"
    )
    cf40: Optional[float] = Field(
        description="Controls the dissipation rate, i.e., the time decay scale (SWAN default: 250.0)"
    )
    edmlpm: Optional[float] = Field(
        description="Maximum non-dimensionless energy density of the wind sea part of the spectrum according to Pierson Moskowitz (SWAN default: 0.0036)"
    )
    cdrag: Optional[float] = Field(
        description="Drag coefficient (SWAN default: 0.0012)"
    )
    umin: Optional[float] = Field(
        description="Minimum wind velocity (relative to current; all wind speeds are taken at 10 m above sea level) (SWAN default: 1)"
    )
    cfpm: Optional[float] = Field(
        description="Coefficient which determines the Pierson Moskowitz frequency: $delta_{PM} = 2pi g / U_{10}$ (SWAN default: 0.13)"
    )

    def cmd(self):
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

    `GEN2 [cf10] [cf20] [cf30] [cf40] [cf50] [cf60] [edmlpm] [cdrag] [umin] [cfpm]`

    With this command the user indicates that SWAN should run in second-generation mode
    (see Scientific/Technical documentation).

    """

    model_type: Literal["gen2"] = Field(
        default="gen2", description="Model type discriminator"
    )
    cf50: Optional[float] = Field(
        description="Controls the spectral energy scale of the limit spectrum (SWAN default: 0.0023)"
    )
    cf60: Optional[float] = Field(
        description="Ccontrols the spectral energy scale of the limit spectrum (SWAN default: -0.223"
    )

    def cmd(self):
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

    `GEN3 JANSSEN|KOMEN|WESTHUYSEN|ST6 (...) AGROW [a]`

    With this command the user indicates that SWAN should run in third-generation mode
    for wind input, quadruplet interactions and whitecapping.

    """

    model_type: Literal["gen3"] = Field(
        default="gen3", description="Model type discriminator"
    )
    source_terms: SOURCE_TERMS = Field(
        default=WESTHUYSEN(),
        description="SWAN source terms to be used",
        discriminator="model_type",
    )

    @root_validator
    def check_source_terms(cls, values):
        return values

    def cmd(self):
        repr = f"GEN3 {self.source_terms.render()}"
        return repr


#======================================================================================
# Swell dissipation SSWELL
#======================================================================================
class NEGATINP(BaseComponent):
    """Negative wind input.

    With this optional command the user activates negative wind input. **This is intended
    only for use with non-breaking swell dissipation SSWELL ZIEGER**. Parameter `rdcoef`
    is a fraction between 0 and 1, representing the strength of negative wind input. As
    an example, with [rdcoef]=0.04, for a spectral bin that is opposed to the wind
    direction, the wind input factor W(k, θ) is negative, and its magnitude is 4% of
    the corresponding value of the spectral bin that is in the opposite direction
    (i.e. in the wind direction). See Zieger et al. (2015) eq. 11, where a0 is their
    notation for [rdcoef]. Default [rdcoef]=0.0 and `rdcoef=0.04` is recommended,
    though as implied by Zieger et al. (2015), this value is not well-established, so
    the user is encouraged to experiment with other values.

    References
    ----------
    Zieger, S., A.V. Babanin, W. E. Rogers and I.R. Young, 2015: Observation-based
    source terms in the third-generation wave model WAVEWATCH, Ocean Model., 96, 2-25.

    """

    model_type: Literal["negatinp"] = Field(
        default="negatinp", description="Model type discriminator"
    )
    rdcoef: confloat(ge=0.0, le=1.0) = Field(
        default=0.0,
        description="Coefficient representing the strength of negative wind input",
    )

    def cmd(self) -> str:
        return f"NEGATINP rdcoef={self.rdcoef}"


class SSWELL(BaseComponent):
    """Swell dissipation sub-component.

    `SSWELL base component.`

    With this command the user can influence the type of swell dissipation included in
    the computations. The Zieger option is intended for use with negative wind input
    via the NEGATINP command. Zieger non-breaking dissipation follows the method used
    in WAVEWATCH III version 4 and does not include the steepness-dependent swell
    coefficient introduced in WAVEWATCH III version 5. As noted already, if
    `GEN3 ST6...` command is used, this SSWELL command should be provided.

    """

    model_type: Literal["sswell"] = Field(
        default="sswell", description="Model type discriminator"
    )

    def cmd(self) -> str:
        return "SSWELL"


class ROGERS(SSWELL):
    """Nonbreaking dissipation of Rogers et al. (2012).

    References
    ----------
    Rogers, W.E., A.V. Babanin and D.W. Wang, 2012: Observation-Consistent Input and
    Whitecapping Dissipation in a Model for Wind-Generated Surface Waves: Description
    and Simple Calculations, J. Atmos. Oceanic Technol., 29:9, 1329-1346.

    """

    model_type: Literal["rogers"] = Field(
        default="rogers", description="Model type discriminator"
    )
    cdsv: Optional[float] = Field(
        description="Coefficient related to laminar atmospheric boundary layer (SWAN default: 1.2)"
    )
    feswell: Optional[float] = Field(
        description="Swell dissipation factor"
    )

    def cmd(self) -> str:
        repr = f"{super().cmd()} ROGERS"
        if self.cdsv is not None:
            repr += f" cdsv={self.cdsv}"
        if self.feswell is not None:
            repr += f" feswell={self.feswell}"
        return repr


class ARDHUIN(SSWELL):
    """Nonbreaking dissipation of Ardhuin et al. (2010).

    References
    ----------
    Ardhuin, F., A. Rogers, A. Babanin, J. Filipot, J. Magne, A. Roland, and P. van
    der Westhuysen, 2010: Semiempirical dissipation source functions for ocean waves.
    Part I: Definition, calibration, and validation, J. Phys. Oceanogr., 40, 1917-1941.

    """

    model_type: Literal["ardhuin"] = Field(
        default="ardhuin", description="Model type discriminator"
    )
    cdsv: Optional[float] = Field(
        description="Coefficient related to laminar atmospheric boundary layer (SWAN default: 1.2)"
    )

    def cmd(self) -> str:
        repr = f"{super().cmd()} ARDHUIN"
        if self.cdsv is not None:
            repr += f" cdsv={self.cdsv}"
        return repr


class ZIEGER(SSWELL):
    """Nonbreaking dissipation of Zieger et al. (2015).

    Swell dissipation of Young et al. (2013) updated by Zieger et al. (2015).

    References
    ----------
    Zieger, S., A.V. Babanin, W. E. Rogers and I.R. Young, 2015: Observation-based
    source terms in the third-generation wave model WAVEWATCH, Ocean Model., 96, 2-25.

    Young, I.R., A.V.Babanin and S. Zieger, 2013: The Decay Rate of Ocean Swell
    Observed by Altimeter, J. Phys. Oceanogr., 43, 3233-3233.

    """

    model_type: Literal["zieger"] = Field(
        default="zieger", description="Model type discriminator"
    )
    b1: Optional[float] = Field(
        description="Non-dimensional proportionality coefficient"
    )

    def cmd(self) -> str:
        repr = f"{super().cmd()} ZIEGER"
        if self.b1 is not None:
            repr += f" b1={self.b1}"
        return repr


#======================================================================================
# WCAPPING
#======================================================================================
class WCAPKOMEN(BaseComponent):
    """Whitecapping according to Komen (1984).

    `WCAPPING KOMEN [cds2] [stpm] [powst] [delta] [powk]`

    Notes
    -----
    The SWAN default for `delta` has been changed since version 40.91A. The setting
    `delta = 1` will improve the prediction of the wave energy at low frequencies, and
    hence the mean wave period. The original default was `delta = 0`, which corresponds
    to WAM Cycle 3. See the Scientific/Technical documentation for further details.

    References
    ----------
    Komen, G.J., S. Hasselmann, and K. Hasselmann, 1984: On the existence of a fully
    developed wind-sea spectrum, J. Phys. Oceanogr., 14, 1271-1285.

    """
    model_type: Literal["wcapkomen"] = Field(
        default="wcapkomen", description="Model type discriminator"
    )
    cds2: Optional[float] = Field(
        description="Coefficient for determining the rate of whitecapping dissipation ($Cds$) (SWAN default: 2.36e-5)"
    )
    stpm: Optional[float] = Field(
        description="Value of the wave steepness for a Pierson-Moskowitz spectrum ($s^2_{PM}$) (SWAN default: 3.02e-3)"
    )
    powst: Optional[float] = Field(
        description="Power of steepness normalized with the wave steepness of a Pierson-Moskowitz spectrum (SWAN default: 2)"
    )
    delta: Optional[float] = Field(
        description="Coefficient which determines the dependency of the whitecapping on wave number (SWAN default: 1)"
    )
    powk: Optional[float] = Field(
        description="power of wave number normalized with the mean wave number (SWAN default: 1)",
    )

    def cmd(self) -> str:
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


class WCAPAB(BaseComponent):
    """Whitecapping according to Alves and Banner (2003).

    `WCAPPING AB [cds2] [br] CURRENT [cds3]`

    References
    ----------
    Alves, J.H.G.M. and M.L. Banner, 2003: A unified formulation of the wave-action
    balance equation, J. Phys. Oceanogr., 33, 2343-2356.

    """
    model_type: Literal["wcapab"] = Field(
        default="wcapab", description="Model type discriminator"
    )
    cds2: Optional[float] = Field(
        description="proportionality coefficient due to Alves and Banner (2003) (SWAN default: 5.0e-5)."
    )
    br: Optional[float] = Field(
        description="Threshold saturation level	(SWAN default: 1.75e-3)"
    )
    current: bool = Field(
        default=False,
        description="Indicates that enhanced current-induced dissipation as proposed by Van der Westhuysen (2012) is to be added",
    )
    cds3: Optional[float] = Field(
        description="Proportionality coefficient (SWAN default: 0.8)"
    )

    def cmd(self) -> str:
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


#======================================================================================
# QUADRUPL
#======================================================================================
class QUADRUPL(BaseComponent):
    """Nonlinear quadruplet wave interactions.

    `QUADRUPL [iquad] [lambda] [Cnl4] [Csh1] [Csh2] [Csh3]`

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

    """

    model_type: Literal["quadrupl"] = Field(
        default="quadrupl", description="Model type discriminator"
    )
    iquad: Literal[1, 2, 3, 8, 4, 51, 52, 53] = Field(
        default=2,
        description=(
            "Numerical procedures for integrating the quadruplets: 1 = semi-implicit "
            "per sweep, 2 = explicit per sweep, 3 = explicit per iteration, "
            "8 = explicit per iteration, but with a more efficient implementation, "
            "4 = multiple DIA, 51 = XNL (deep water transfer), 52 = XNL (deep water "
            "transfer with WAM depth scaling), 53  XNL (finite depth transfer)"
        ),
    )
    lambd: Optional[float] = Field(
        description="Coefficient for quadruplet configuration in case of DIA (SWAN default: 0.25)",
    )
    cn14: Optional[float] = Field(
        description="Proportionality coefficient for quadruplet interactions in case of DIA (SWAN default: 3.0e7",
    )
    csh1: Optional[float] = Field(
        description="Coefficient for shallow water scaling in case of DIA (SWAN default: 5.5",
    )
    csh2: Optional[float] = Field(
        description="Coefficient for shallow water scaling in case of DIA (SWAN default: 0.833333",
    )
    csh3: Optional[float] = Field(
        description="Coefficient for shallow water scaling in case of DIA (SWAN default: -1.25",
    )

    def cmd(self) -> str:
        repr = f"QUADRUPL iquad={self.iquad}"
        if self.lambd is not None:
            repr += f" lambda={self.lambd}"
        if self.cn14 is not None:
            repr += f" Cn14={self.cn14}"
        if self.csh1 is not None:
            repr += f" Csh1={self.csh1}"
        if self.csh2 is not None:
            repr += f" Csh2={self.csh2}"
        if self.csh3 is not None:
            repr += f" Csh3={self.csh3}"
        return repr


#======================================================================================
# BREAKING
#======================================================================================
class BREAKCONSTANT(BaseComponent):
    """Constant wave breaking index.

    `BREAKING CONSTANT [alpha] [gamma]`

    Indicates that a constant breaker index is to be used.

    """
    model_type: Literal["constant"] = Field(
        default="constant", description="Model type discriminator"
    )
    alpha: Optional[float] = Field(
        description="Proportionality coefficient of the rate of dissipation (SWAN default: 1.0)"
    )
    gamma: Optional[float] = Field(
        description="The breaker index, i.e. the ratio of maximum individual wave height over depth (SWAN default: 0.73)"
    )

    def cmd(self) -> str:
        repr = "BREAKING CONSTANT"
        if self.alpha is not None:
            repr += f" alpha={self.alpha}"
        if self.gamma is not None:
            repr += f" gamma={self.gamma}"
        return repr


class BREAKBKD(BaseComponent):
    """Variable wave breaking index.

    `BREAKING BKD [alpha] [gamma0] [a1] [a2] [a3]`

    Indicates that the breaker index scales with both the bottom slope ($beta$) and the
    dimensionless depth (kd).

    """
    model_type: Literal["bkd"] = Field(
        default="bkd", description="Model type discriminator"
    )
    alpha: Optional[float] = Field(
        description="Proportionality coefficient of the rate of dissipation (SWAN default: 1.0)"
    )
    gamma0: Optional[float] = Field(
        description="The reference $gamma$ for horizontal slopes (SWAN default: 0.54)"
    )
    a1: Optional[float] = Field(
        description="First tunable coefficient for the breaker index (SWAN default: 7.59)"
    )
    a2: Optional[float] = Field(
        description="Second tunable coefficient for the breaker index (SWAN default: -8.06)"
    )
    a3: Optional[float] = Field(
        description="Third tunable coefficient for the breaker index (SWAN default: 8.09)"
    )

    def cmd(self) -> str:
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


#======================================================================================
# FRICTION
#======================================================================================
class JONSWAP(BaseComponent):
    """Hasselmann et al. (1973) Jonswap friction.

    `FRICTION JONSWAP CONSTANT [cfjon]`

    Indicates that the semi-empirical expression derived from the JONSWAP results for
    bottom friction dissipation (Hasselmann et al., 1973, JONSWAP) should be activated.
    This option is default.

    References
    ----------
    Hasselmann, K., Barnett, T.P., Bouws, E., Carlson, H., Cartwright, D.E., Enke, K.,
    Ewing, J.A., Gienapp, H., Hasselmann, D.E., Kruseman, P., Meerburg, A., Müller, P.,
    Olbers, D.J., Richter, K., Sell, W., Walden, H., 1973. Measurements of wind-wave
    growth and swell decay during the Joint North Sea Wave Project (JONSWAP). Deutches
    Hydrographisches Institut, Hamburg, Germany, Rep. No. 12, 95 pp.

    TODO: Implement VARIABLE option?

    """
    model_type: Literal["jonswap"] = Field(
        default="jonswap", description="Model type discriminator"
    )
    cfjon: Optional[float] = Field(
        description="Coefficient of the JONSWAP formulation (SWAN default: 0.038)"
    )

    def cmd(self) -> str:
        repr = "FRICTION JONSWAP CONSTANT"
        if self.cfjon is not None:
            repr += f" cfjon={self.cfjon}"
        return repr


class COLLINS(BaseComponent):
    """Collins (1972) friction.

    `FRICTION COLLINS [cfw]`

    Note that `cfw` is allowed to vary over the computational region; in that case use
    the commands INPGRID FRICTION and READINP FRICTION to define and read the friction
    data. This command FRICTION is still required to define the type of friction
    expression. The value of `cfw` in this command is then not required (it will be
    ignored).

    References
    ----------
    Collins, M.B., 1972. The effect of bottom friction on the propagation of long
    waves. Coastal Engineering, 1972, 1, 163-181.

    """
    model_type: Literal["collins"] = Field(
        default="collins", description="Model type discriminator"
    )
    cfw: Optional[float] = Field(
        description="Collins bottom friction coefficient (SWAN default: 0.015)"
    )

    def cmd(self) -> str:
        repr = "FRICTION COLLINS"
        if self.cfw is not None:
            repr += f" cfw={self.cfw}"
        return repr


class MADSEN(BaseComponent):
    """Madsen et al (1988) friction.

    `FRICTION MADSEN [kn]`

    Note that `kn` is allowed to vary over the computational region; in that case use
    the commands INPGRID FRICTION and READINP FRICTION to define and read the friction
    data. This command FRICTION is still required to define the type of friction
    expression. The value of `kn` in this command is then not required (it will be
    ignored).

    References
    ----------
    Madsen, O.S., Sørensen, O.R., Schäffer, H.A., 1988. Surf zone dynamics simulated by
    a Boussinesq type model. Part I. Model description and cross-shore motion of
    regular waves. Coastal Engineering, 1988, 12, 115-145.

    """
    model_type: Literal["madsen"] = Field(
        default="madsen", description="Model type discriminator"
    )
    kn: Optional[float] = Field(
        description="equivalent roughness length scale of the bottom (in m) (SWAN default: 0.05)"
    )

    def cmd(self) -> str:
        repr = "FRICTION MADSEN"
        if self.kn is not None:
            repr += f" kn={self.kn}"
        return repr


class RIPPLES(BaseComponent):
    """Smith et al. (2011) Ripples friction.

    `FRICTION RIPPLES [S] [D]`

    Indicates that the expression of Smith et al. (2011) should be activated. Here
    friction depends on the formation of bottom ripples and sediment size.

    References
    ----------
    Smith, J.M., McCall, R.T., 2011. A model for wave dissipation in the surf zone
    based on field observations. Coastal Engineering, 2011, 58, 917-928.

    """
    model_type: Literal["ripples"] = Field(
        default="ripples", description="Model type discriminator"
    )
    s: Optional[float] = Field(
        description="The specific gravity of the sediment (SWAN default: 2.65)"
    )
    d: Optional[float] = Field(
        description="The sediment diameter (in m) (SWAN default: 0.0001)"
    )

    def cmd(self) -> str:
        repr = "FRICTION RIPPLES"
        if self.s is not None:
            repr += f" S={self.s}"
        if self.d is not None:
            repr += f" D={self.d}"
        return repr


#======================================================================================
# TRIAD
#======================================================================================
class TRIAD(BaseComponent):
    """Wave triad interactions.

    `TRIAD`

    With this command the user can activate the triad wave-wave interactions. If this
    command is not used, SWAN will not account for triads.

    """
    model_type: Literal["triad"] = Field(
        default="triad", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"TRIAD"


#======================================================================================
# VEGETATION
#======================================================================================
class VEGETATION(BaseComponent):
    """Vegetation dumping.

    `VEGETATION`

    With this command the user can activate wave damping due to vegetation based on the
    Dalrymple's formula (1984) as implemented by Suzuki et al. (2011). This damping is
    uniform over the wave frequencies. An alternative is the frequency-dependent
    (canopy) dissipation model of Jacobsen et al. (2019). If this command is not used,
    SWAN will not account for vegetation effects.

    References
    ----------
    Dalrymple, R.A., 1984. Water wave interactions with vegetation: Part 1. A fully
    nonlinear theory. Journal of Fluid Mechanics, 1984, 140, 197-229.

    """
    model_type: Literal["vegetation"] = Field(
        default="vegetation", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"VEGETATION"


#======================================================================================
# MUD
#======================================================================================
class MUD(BaseComponent):
    """Mud dumping.

    `MUD`

    With this command the user can activate wave damping due to mud based on Ng (2000).
    If this command or the commands INPGRID MUDLAY and READINP MUDLAY are not used,
    SWAN will not account for muddy bottom effects.

    References
    ----------
    Ng, C.O., 2000. Wave attenuation in mudflat and salt marsh vegetation. Journal of
    Coastal Research, 2000, 16, 379-389.

    """
    model_type: Literal["mud"] = Field(
        default="mud", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"MUD"


#======================================================================================
# SICE
#======================================================================================
class SICE(BaseComponent):
    """Sea ice dissipation.

    `SICE`

    Using this command, the user activates a sink term to represent the dissipation of
    wave energy by sea ice. The R19 method is empirical/parametric: a polynomial based
    on wave frequency (Rogers, 2019). This polynomial (in 1/m) has seven dimensional
    coefficients; see Scientific/Technical documentation for details. If this command
    is not used, SWAN will not account for sea ice effects.

    References
    ----------
    Doble, M.J., Bidlot, J.R., Wadhams, P., 2015. Sea ice and ocean wave modelling in
    the Arctic: Interactions between waves and ice in a coupled atmosphere-wave-ice
    model. Ocean Modelling, 2015, 96, 141-155.

    Meylan, M.H., Bennetts, L.G., Thomas, T., 2014. Dissipation of ocean waves by sea
    ice. Geophysical Research Letters, 2014, 41, 7561-7568.

    Meylan, M.H., Bennetts, L.G., Williams, T.D., 2018. Wave attenuation by pancake
    ice. Journal of Geophysical Research: Oceans, 2018, 123, 1-16.

    Rogers, W.E., 2019. Wave attenuation by sea ice: A new parametric model. Journal of
    Geophysical Research: Oceans, 2019, 124, 1-15.

    """
    model_type: Literal["sice"] = Field(
        default="sice", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"SICE"


#======================================================================================
# TURBULENCE
#======================================================================================
class TURBULENCE(BaseComponent):
    """Turbulent viscosity.

    `TURBULENCE`

    With this optional command the user can activate turbulent viscosity. This physical
    effect is also activated by reading values of the turbulent viscosity using the
    `READGRID TURB` command, but then with the default value of `ctb`. The command
    `READGRID TURB` is necessary if this command `TURB` is used since the value of the
    viscosity is assumed to vary over space.

    """
    model_type: Literal["turbulence"] = Field(
        default="turbulence", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"TURBULENCE"


#======================================================================================
# BRAGG
#======================================================================================
class BRAGG(BaseComponent):
    """Bragg scattering.

    `BRAGG`

    Using this optional command, the user activates a source term to represent the
    scattering of waves due to changes in the small-scale bathymetry based on the
    theory of Ardhuin and Herbers (2002). If this command is not used, SWAN will not
    account for Bragg scattering.

    """
    model_type: Literal["bragg"] = Field(
        default="bragg", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"BRAGG"


#======================================================================================
# LIMITER
#======================================================================================
class LIMITER(BaseComponent):
    """Physics limiter.

    `LIMITER`

    With this command the user can de-activate permanently the quadruplets when
    the actual Ursell number exceeds `ursell`. Moreover, as soon as the actual
    fraction of breaking waves exceeds `qb` then the action limiter will not be
    used in case of decreasing action density.

    """
    model_type: Literal["limiter"] = Field(
        default="limiter", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"LIMITER"


#======================================================================================
# OBSTACLE
#======================================================================================
class OBSTACLE(BaseComponent):
    """Sub-grid obstacle.

    `OBSTACLE`

    With this optional command the user provides the characteristics of a (line
    of) sub-grid obstacle(s) through which waves are transmitted or against which
    waves are reflected (possibly both at the same time). The obstacle is sub-grid
    in the sense that it is narrow compared to the spatial meshes; its length should
    be at least one mesh length.

    """
    model_type: Literal["obstacle"] = Field(
        default="obstacle", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"OBSTACLE"


#======================================================================================
# OBSTACLE FIG
#======================================================================================
class OBSTACLE_FIG(BaseComponent):
    """Obstacle for free infragravity radiation.

    `OBSTACLE FIG`

    With this optional command the user specifies the obstacles along which the
    free infra-gravity (FIG) energy is radiated. By placing the obstacles close to
    the shorelines SWAN will include the FIG source term along the coastlines
    according to the parametrization of Ardhuin et al. (2014).

    References
    ----------
    Ardhuin, F., A. Rogers, A. Babanin, L. Filipot, A. Magne, A. Roland, and P.
    van der Westhuysen, Semiempirical Dissipation Source Functions for Ocean Waves.
    Part I: Definition, Calibration, and Validation, J. Phys. Oceanogr., 44(9),
    2352-2368, 2014.

    """
    model_type: Literal["obstacle_fig"] = Field(
        default="obstacle_fig", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"OBSTACLE FIG"


#======================================================================================
# SETUP
#======================================================================================
class SETUP(BaseComponent):
    """Wave setup.

    `SETUP`

    """
    model_type: Literal["setup"] = Field(
        default="setup", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"SETUP"


#======================================================================================
# DIFFRACTION
#======================================================================================
class DIFFRACTION(BaseComponent):
    """Wave diffraction.

    `DIFFRACTION`

    """
    model_type: Literal["diffraction"] = Field(
        default="diffraction", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"DIFFRACTION"


#======================================================================================
# SURFBEAT
#======================================================================================
class SURFBEAT(BaseComponent):
    """Surfbeat.

    `SURFBEAT`

    """
    model_type: Literal["surfbeat"] = Field(
        default="surfbeat", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"SURFBEAT"


#======================================================================================
# SCAT
#======================================================================================
class SCAT(BaseComponent):
    """Scattering.

    `SCAT`

    """
    model_type: Literal["scat"] = Field(
        default="scat", description="Model type discriminator"
    )
    def cmd(self) -> str:
        return f"SCAT"


#======================================================================================
# Physics group component
#======================================================================================
class PHYSICS(BaseComponent):
    """Physics group component.

    The physics group component is a convenience to allow specifying several individual
    components in a single command and check for consistency between them.

    TODO: Implement OFF command

    """

    model_type: Literal["physics"] = Field(
        default="physics", description="Model type discriminator"
    )
    gen: GEN1 | GEN2 | GEN3 = Field(
        default=GEN3(),
        description="Wave generation specification",
        discriminator="model_type",
    )
    sswell: ROGERS | ARDHUIN | ZIEGER | None = Field(
        default=None,
        description="Swell dissipation specification",
        discriminator="model_type",
    )
    negatinp: Optional[NEGATINP] = Field(
        default=None,
        description="Negative wind input specification",
    )
    wcapping: WCAPKOMEN | WCAPAB | None = Field(
        default=None,
        description="Whitecapping specification",
        discriminator="model_type",
    )
    quadrupl: Optional[QUADRUPL] = Field(
        description="Quadruplet interactions specification",
    )
    breaking: BREAKCONSTANT | BREAKBKD | None = Field(
        description="Wave breaking specification",
        discriminator="model_type",
    )
    friction: JONSWAP | COLLINS | MADSEN | RIPPLES | None = Field(
        default=None,
        description="Bottom friction specification",
    )

    @root_validator
    def deactivate_physics(cls, values):
        return values

    @validator("negatinp", pre=False)
    def negatinp_only_with_zieger(cls, value, values):
        """Log a warning if NEGATINP is used with a non-ZIEGER SSWELL."""
        if values["sswell"] is None:
            logger.warning(
                "The negative wind input NEGATINP is only intended to use with the "
                "swell dissipation SSWELL ZIEGER but no SSWELL has been specified."
            )
        elif values["sswell"].model_type != "zieger":
            logger.warning(
                "The negative wind input NEGATINP is only intended to use with the "
                "swell dissipation SSWELL ZIEGER but "
                f"SSWELL {values['sswell'].model_type.upper()} has been specified."
            )
        return value

    def cmd(self):
        repr = [self.gen.render()]
        if self.sswell is not None:
            repr += [f"{self.sswell.render()}"]
        if self.negatinp is not None:
            repr += [self.negatinp.render()]
        if self.wcapping is not None:
            repr += [self.wcapping.render()]
        if self.quadrupl is not None:
            repr += [self.quadrupl.render()]
        if self.breaking is not None:
            repr += [self.breaking.render()]
        if self.friction is not None:
            repr += [self.friction.render()]
        return repr
