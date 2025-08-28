"""SWAN physics subcomponents."""

from typing import Annotated, Literal, Optional
from pydantic import field_validator, Field, model_validator
from abc import ABC
from pydantic_numpy.typing import Np2DArray

from rompy.swan.subcomponents.base import BaseSubComponent


# ======================================================================================
# Source terms
# ======================================================================================
class SourceTerms(BaseSubComponent, ABC):
    """Source term subcomponent base class."""

    wind_drag: Literal["wu", "fit"] = Field(
        default="wu",
        description="Indicates the wind drag formulation",
    )
    agrow: bool = Field(
        default=False,
        description="Activate the Cavaleri and Malanotte (1981) wave growth term",
    )
    a: Optional[float] = Field(
        default=None,
        description=(
            "Proportionality coefficient when activating the Cavaleri and Malanotte "
            "(1981) wave growth term (SWAN default: 0.0015)"
        ),
    )


class JANSSEN(SourceTerms):
    """Janssen source terms subcomponent.

    .. code-block:: text

        JANSSEN [cds1] [delta] (AGROW [a])

    References
    ----------
    Janssen, P.A., 1989. Wave-induced stress and the drag of air flow over sea waves.
    Journal of Physical Oceanography, 19(6), pp.745-754.

    Janssen, P.A.E.M., Lionello, P. and Zambresky, L., 1989. On the interaction of wind
    and waves. Philosophical transactions of the royal society of London. Series A,
    Mathematical and Physical Sciences, 329(1604), pp.289-301.

    Janssen, P.A., 1991. Quasi-linear theory of wind-wave generation applied to wave
    forecasting. Journal of physical oceanography, 21(11), pp.1631-1642.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import JANSSEN
        janssen = JANSSEN()
        print(janssen.render())
        janssen = JANSSEN(cds1=4.5, delta=0.5, agrow=True)
        print(janssen.render())

    """

    model_type: Literal["janssen", "JANSSEN"] = Field(
        default="janssen", description="Model type discriminator"
    )
    cds1: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient for determining the rate of whitecapping dissipation "
            "($Cds / s^4_{PM}$) (SWAN default: 4.5)"
        ),
    )
    delta: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient which determines the dependency of the whitecapping on wave "
            "number (mix with Komen et al. formulation) (SWAN default: 0.5)"
        ),
    )

    def cmd(self) -> str:
        repr = "JANSSEN"
        if self.cds1 is not None:
            repr += f" cds1={self.cds1}"
        if self.delta is not None:
            repr += f" delta={self.delta}"
        repr += f" DRAG {self.wind_drag.upper()}"
        if self.agrow:
            repr += " AGROW"
        if self.a is not None and self.agrow:
            repr += f" a={self.a}"
        return repr


class KOMEN(SourceTerms):
    """Komen source terms subcomponent.

    .. code-block:: text

        KOMEN [cds2] [stpm] (AGROW [a])

    References
    ----------
    Komen, G.J., Hasselmann, S. and Hasselmann, K., 1984. On the existence of a fully
    developed wind-sea spectrum. Journal of physical oceanography, 14(8), pp.1271-1285.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import KOMEN
        komen = KOMEN()
        print(komen.render())
        komen = KOMEN(cds2=2.36e-5, stpm=3.02e-3, agrow=True, a=0.0015)
        print(komen.render())

    """

    model_type: Literal["komen", "KOMEN"] = Field(
        default="komen", description="Model type discriminator"
    )
    cds2: Optional[float] = Field(
        default=None,
        description=(
            "Coefficient for determining the rate of whitecapping dissipation "
            "(`Cds`) (SWAN default: 2.36e-5)"
        ),
    )
    stpm: Optional[float] = Field(
        default=None,
        description=(
            "Value of the wave steepness for a Pierson-Moskowitz spectrum "
            "(`s^2_PM`) (SWAN default: 3.02e-3)"
        ),
    )

    def cmd(self) -> str:
        repr = "KOMEN"
        if self.cds2 is not None:
            repr += f" cds2={self.cds2}"
        if self.stpm is not None:
            repr += f" stpm={self.stpm}"
        repr += f" DRAG {self.wind_drag.upper()}"
        if self.agrow:
            repr += " AGROW"
        if self.a is not None and self.agrow:
            repr += f" a={self.a}"
        return repr


class WESTHUYSEN(SourceTerms):
    """Westhuysen source terms subcomponent.

    .. code-block:: text

        WESTHUYSEN [cds2] [br] (AGROW [a])

    Nonlinear saturation-based whitecapping combined with wind input of Yan (1987).

    Notes
    -----
    The two arguments are specified in the Appendix C of the User manual but not in the
    command description for WESTH in Section 4.5.4. They are also options in the
    WCAPPING command. It is not entirely clear if they should/could be specified here.

    References
    ----------
    van der Westhuysen, A.J., Zijlema, M. and Battjes, J.A., 2007. Nonlinear
    saturation-based whitecapping dissipation in SWAN for deep and shallow water.
    Coastal Engineering, 54(2), pp.151-170.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import WESTHUYSEN
        westhuysen = WESTHUYSEN()
        print(westhuysen.render())
        westhuysen = WESTHUYSEN(cds2=5.0e-5, br=1.75e-3)
        print(westhuysen.render())

    """

    model_type: Literal["westhuysen", "WESTHUYSEN"] = Field(
        default="westhuysen", description="Model type discriminator"
    )
    cds2: Optional[float] = Field(
        default=None,
        description=(
            "proportionality coefficient due to Alves and Banner (2003) "
            "(SWAN default: 5.0e-5)."
        ),
    )
    br: Optional[float] = Field(
        default=None, description="Threshold saturation level	(SWAN default: 1.75e-3)"
    )

    def cmd(self) -> str:
        repr = "WESTHUYSEN"
        if self.cds2 is not None:
            repr += f" cds2={self.cds2}"
        if self.br is not None:
            repr += f" br={self.br}"
        repr += f" DRAG {self.wind_drag.upper()}"
        if self.agrow:
            repr += " AGROW"
        if self.a is not None and self.agrow:
            repr += f" a={self.a}"
        return repr


class ST6(SourceTerms):
    """St6 source terms subcomponent.

    .. code-block:: text

        ST6 [a1sds] [a2sds] [p1sds] [p2sds] UP|DOWN HWANG|FAN|ECMWF VECTAU|SCATAU &
            TRUE10|U10PROXY [windscaling] DEBIAS [cdfac] (AGROW [a])

    wind input and whitecapping from Rogers et al. (2012) (RBW12).

    Notes
    -----
    The two arguments are specified in the Appendix C of the User manual but not in the
    command description for WESTH in Section 4.5.4. They are also options in the
    WCAPPING command. It is not entirely clear if they should/could be specified here.

    References
    ----------
    Fan, Y., Lin, S.J., Held, I.M., Yu, Z. and Tolman, H.L., 2012. Global ocean surface
    wave simulation using a coupled atmosphere–wave model. Journal of Climate, 25(18),
    pp.6233-6252.

    Rogers, W.E., Babanin, A.V. and Wang, D.W., 2012. Observation-consistent input and
    whitecapping dissipation in a model for wind-generated surface waves: Description
    and simple calculations. Journal of Atmospheric and Oceanic Technology, 29(9),
    pp.1329-1346.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import ST6
        st6 = ST6(a1sds=4.7e-7, a2sds=6.6e-6)
        print(st6.render())
        kwargs = dict(
            a1sds=2.8e-6,
            a2sds=3.5e-5,
            normalization="up",
            wind_drag="hwang",
            tau="vectau",
            u10="u10proxy",
            windscaling=32.0,
            cdfac=0.89,
            agrow=True,
            a=0.0015,
        )
        st6 = ST6(**kwargs)
        print(st6.render())

    """

    model_type: Literal["st6"] = Field(
        default="st6", description="Model type discriminator"
    )
    a1sds: float = Field(
        description="Coefficient related to local dissipation term T1 (a1 in RBW12)"
    )
    a2sds: float = Field(
        description="Coefficient related to local dissipation term T2 (a2 in RBW12)"
    )
    p1sds: Optional[float] = Field(
        default=None,
        description=(
            "Power coefficient controlling strength of dissipation term T1 "
            "(L in RBW12, SWAN default: 4)"
        ),
    )
    p2sds: Optional[float] = Field(
        default=None,
        description=(
            "Power coefficient controlling strength of dissipation term T2 "
            "(M in RBW12, SWAN default: 4)"
        ),
    )
    normalization: Literal["up", "down"] = Field(
        default="up",
        description=(
            "Selection of normalization of exceedance level by ET(f) (`up`) or E(f) "
            "(`down`) as in RBW12 (right column, page 1333), `up` is default and "
            "strongly recommended"
        ),
    )
    wind_drag: Literal["hwang", "fan", "ecmwf"] = Field(
        default="hwang",
        description=(
            "Wind drag formula, `hwang` is the default and is unchanged from RBW12, "
            "`fan` is from Fan et al. (2012), `ecmwf` follows WAM Cycle 4 methodology"
        ),
    )
    tau: Literal["vectau", "scatau"] = Field(
        default="vectau",
        description=(
            "Use vector (vectau) or scalar (scatau) calculation for the wind strerss "
            "(Eq. 12 in RBW12), `vectau` is the default and strongly recommended"
        ),
    )
    u10: Literal["u10proxy", "true10"] = Field(
        default="u10proxy",
        description="Wind velocity definition",
    )
    windscaling: Optional[float] = Field(
        default=32.0,
        description="Factor to scale U10 with U* when using U10PROXY",
    )
    cdfac: Optional[float] = Field(
        default=None,
        description=(
            "Counter bias in the input wind fields by providing a multiplier "
            "on the drag coefficient"
        ),
        gt=0.0,
    )

    @model_validator(mode="after")
    def debias_only_with_hwang(self) -> "ST6":
        if self.cdfac is not None and self.wind_drag != "hwang":
            raise ValueError(
                f"Debias is only supported with hwang wind drag, not {self.wind_drag}"
            )
        return self

    @property
    def u10_cmd(self) -> str:
        if self.u10 == "true10":
            return "TRUE10"
        else:
            return f"U10PROXY windscaling={self.windscaling}"

    def cmd(self) -> str:
        repr = f"ST6 a1sds={self.a1sds} a2sds={self.a2sds}"
        if self.p1sds is not None:
            repr += f" p1sds={self.p1sds}"
        if self.p2sds is not None:
            repr += f" p2sds={self.p2sds}"
        repr += f" {self.normalization.upper()}"
        repr += f" {self.wind_drag.upper()}"
        repr += f" {self.tau.upper()}"
        repr += f" {self.u10_cmd}"
        if self.cdfac is not None:
            repr += f" DEBIAS cdfac={self.cdfac}"
        if self.agrow:
            repr += " AGROW"
        if self.a is not None and self.agrow:
            repr += f" a={self.a}"
        return repr


class ST6C1(ST6):
    """First ST6 calibration in the SWAN user manual.

    .. code-block:: text

        ST6 4.7e-7 6.6e-6 4.0 4.0 UP HWANG VECTAU U10PROXY 28.0 AGROW

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import ST6C1
        st6 = ST6C1()
        print(st6.render())

    """

    model_type: Literal["st6c1"] = Field(default="st6c1")
    a1sds: Literal[4.7e-7] = Field(default=4.7e-7)
    a2sds: Literal[6.6e-6] = Field(default=6.6e-6)
    p1sds: Literal[4.0] = Field(default=4.0)
    p2sds: Literal[4.0] = Field(default=4.0)
    windscaling: Literal[28.0] = Field(default=28.0)
    agrow: Literal[True] = Field(default=True)


class ST6C2(ST6C1):
    """Second ST6 calibration in the SWAN user manual.

    .. code-block:: text

        ST6 4.7e-7 6.6e-6 4.0 4.0 UP FAN VECTAU U10PROXY 28.0 AGROW

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import ST6C2
        st6 = ST6C2()
        print(st6.render())

    TODO: Ensure validator is reused here so fan and debias are not used together.

    """

    model_type: Literal["st6c2"] = Field(default="st6c2")
    wind_drag: Literal["fan"] = Field(default="fan")


class ST6C3(ST6C1):
    """Third ST6 calibration in the SWAN user manual.

    .. code-block:: text

        ST6 2.8e-6 3.5e-5 4.0 4.0 UP HWANG VECTAU U10PROXY 32.0 AGROW

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import ST6C3
        st6 = ST6C3()
        print(st6.render())

    """

    model_type: Literal["st6c3"] = Field(default="st6c3")
    a1sds: Literal[2.8e-6] = Field(default=2.8e-6)
    a2sds: Literal[3.5e-5] = Field(default=3.5e-5)
    windscaling: Literal[32.0] = Field(default=32.0)


class ST6C4(ST6C3):
    """Fourth ST6 calibration in the SWAN user manual.

    .. code-block:: text

        ST6 2.8e-6 3.5e-5 4.0 4.0 UP HWANG VECTAU U10PROXY 32.0 DEBIAS 0.89 AGROW

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import ST6C4
        st6 = ST6C4()
        print(st6.render())

    """

    model_type: Literal["st6c4"] = Field(default="st6c4")
    cdfac: Literal[0.89] = Field(default=0.89)


class ST6C5(ST6C1):
    """Fifth ST6 calibration in the SWAN user manual.

    .. code-block:: text

        ST6 4.7e-7 6.6e-6 4.0 4.0 UP HWANG VECTAU U10PROXY 28.0 AGROW

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import ST6C5
        st6 = ST6C5()
        print(st6.render())

    """

    model_type: Literal["st6c5"] = Field(default="st6c5")
    cdfac: Literal[0.89] = Field(default=0.89)
    a1sds: Literal[6.5e-6] = Field(default=6.5e-6)
    a2sds: Literal[8.5e-5] = Field(default=8.5e-5)
    windscaling: Literal[35.0] = Field(default=35.0)


# =====================================================================================
# Biphase
# =====================================================================================
class ELDEBERKY(BaseSubComponent):
    """Biphase of Eldeberky (1999).

    .. code-block:: text

        BIPHASE ELDEBERKY [urcrit]

    Biphase parameterisation as a funtion of the Ursell number of Eldeberky (1999).

    References
    ----------
    Eldeberky, Y., Polnikov, V. and Battjes, J.A., 1996. A statistical approach for
    modeling triad interactions in dispersive waves. In Coastal Engineering 1996
    (pp. 1088-1101).

    Eldeberky, Y. and Madsen, P.A., 1999. Deterministic and stochastic evolution
    equations for fully dispersive and weakly nonlinear waves. Coastal Engineering,
    38(1), pp.1-24.

    Doering, J.C. and Bowen, A.J., 1995. Parametrization of orbital velocity
    asymmetries of shoaling and breaking waves using bispectral analysis. Coastal
    engineering, 26(1-2), pp.15-33.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import ELDEBERKY
        biphase = ELDEBERKY()
        print(biphase.render())
        biphase = ELDEBERKY(urcrit=0.63)
        print(biphase.render())

    """

    model_type: Literal["eldeberky"] = Field(
        default="eldeberky", description="Model type discriminator"
    )
    urcrit: Optional[float] = Field(
        default=None,
        description=(
            "The critical Ursell number appearing in the parametrization. Note: the "
            "value of `urcrit` is setted by Eldeberky (1996) at 0.2 based on a "
            "laboratory experiment, whereas Doering and Bowen (1995) employed the "
            "value of 0.63 based on the field experiment data (SWAN default: 0.63)"
        ),
    )

    def cmd(self) -> str:
        repr = "BIPHASE ELDEBERKY"
        if self.urcrit is not None:
            repr += f" urcrit={self.urcrit}"
        return repr


class DEWIT(BaseSubComponent):
    """Biphase of De Wit (2022).

    .. code-block:: text

        BIPHASE DEWIT [lpar]

    Biphase parameterization based on bed slope and peak period of De Wit (2022).

    References
    ----------
    De Wit, F.P., 2022. Wave shape prediction in complex coastal systems (Doctoral
    dissertation, PhD. thesis. Delft University of Technology. https://repository.
    tudelft. nl/islandora/object/uuid% 3A0fb850a4-4294-4181-9d74-857de21265c2).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import DEWIT
        biphase = DEWIT()
        print(biphase.render())
        biphase = DEWIT(lpar=0.0)
        print(biphase.render())

    """

    model_type: Literal["dewit"] = Field(
        default="dewit", description="Model type discriminator"
    )
    lpar: Optional[float] = Field(
        default=None,
        description=(
            "Scales spatial averaging of the De Wit's biphase in terms of a multiple "
            "of peak wave length of the incident wave field. Note: `lpar` = 0` means "
            "no averaging (SWAN default: 0)"
        ),
    )

    def cmd(self) -> str:
        repr = "BIPHASE DEWIT"
        if self.lpar is not None:
            repr += f" lpar={self.lpar}"
        return repr


# =====================================================================================
# Transmission and reflection
# =====================================================================================
class TRANSM(BaseSubComponent):
    """Constant transmission coefficient.

    .. code-block:: text

        TRANSM [trcoef]

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import TRANSM
        transm = TRANSM()
        print(transm.render())
        transm = TRANSM(trcoef=0.5)
        print(transm.render())

    """

    model_type: Literal["transm", "TRANSM"] = Field(
        default="transm", description="Model type discriminator"
    )
    trcoef: Optional[float] = Field(
        default=None,
        description=(
            "Constant transmission coefficient (ratio of transmitted over incoming "
            "significant wave height) (SWAN default: 0.0) (no transmission = complete "
            "blockage)"
        ),
        ge=0.0,
        le=1.0,
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = "TRANSM"
        if self.trcoef is not None:
            repr += f" trcoef={self.trcoef}"
        return repr


class TRANS1D(BaseSubComponent):
    """Frequency dependent transmission.

    .. code-block:: text

        TRANS1D < [trcoef] >

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import TRANS1D
        transm = TRANS1D(trcoef=[0.0, 0.0, 0.2, 0.5, 0.2, 0.0, 0.0])
        print(transm.render())

    """

    model_type: Literal["trans1d", "TRANS1D"] = Field(
        default="trans1d", description="Model type discriminator"
    )
    trcoef: list[Annotated[float, Field(ge=0.0, le=1.0)]] = Field(
        description=(
            "Transmission coefficient (ratio of transmitted over incoming significant "
            "wave height) per frequency. The number of these transmission values must "
            "be equal to the number of frequencies, i.e. `msc` + 1"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        return f"TRANS1D {' '.join(str(v) for v in self.trcoef)}"


class TRANS2D(BaseSubComponent):
    """Frequency-direction dependent transmission.

    .. code-block:: text

        TRANS2D < [trcoef] >

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import TRANS2D
        trcoef = np.array([[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
        transm = TRANS2D(trcoef=trcoef)
        print(transm.render())

    """

    model_type: Literal["trans2d", "TRANS2D"] = Field(
        default="trans2d", description="Model type discriminator"
    )
    trcoef: Np2DArray = Field(
        description=(
            "Transmission coefficient (ratio of transmitted over incoming significant "
            "wave height) per frequency and direction, rows represent directions and "
            "columns represent frequencies"
        ),
    )

    @field_validator("trcoef")
    @classmethod
    def constrained_0_1(cls, value: float) -> float:
        """Ensure all directions have the same number of frequencies."""
        if value.min() < 0 or value.max() > 1:
            raise ValueError("Transmission coefficients must be between 0.0 and 1.0")
        return value

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = "TRANS2D"
        for coef in self.trcoef:
            repr += f" &\n\t{' '.join(str(v) for v in coef)}"
        return f"{repr} &\n\t"


class GODA(BaseSubComponent):
    """DAM transmission of Goda/Seelig (1979).

    .. code-block:: text

        DAM GODA [hgt] [alpha] [beta]

    This option specified transmission coefficients dependent on the incident wave
    conditions at the obstacle and on the obstacle height (which may be submerged).

    References
    ----------
    Goda, Y. and Suzuki, Y., 1976. Estimation of incident and reflected waves in random
    wave experiments. In Coastal Engineering 1976 (pp. 828-845).

    Seelig, W.N., 1979. Effects of breakwaters on waves: Laboratory test of wave
    transmission by overtopping. In Proc. Conf. Coastal Structures, 1979
    (Vol. 79, No. 2, pp. 941-961).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import GODA
        transm = GODA(hgt=3.0)
        print(transm.render())
        transm = GODA(hgt=3.0, alpha=2.6, beta=0.15)
        print(transm.render())

    """

    model_type: Literal["goda", "GODA"] = Field(
        default="goda", description="Model type discriminator"
    )
    hgt: float = Field(
        description=(
            "The elevation of the top of the obstacle above reference level (same "
            "reference level as for bottom etc.); use a negative value if the top is "
            "below that reference level"
        ),
    )
    alpha: Optional[float] = Field(
        default=None,
        description=(
            "coefficient determining the transmission coefficient for Goda's "
            "transmission formula (SWAN default: 2.6)"
        ),
    )
    beta: Optional[float] = Field(
        default=None,
        description=(
            "Another coefficient determining the transmission coefficient for Goda's "
            "transmission formula (SWAN default: 0.15)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = f"DAM {self.model_type.upper()} hgt={self.hgt}"
        if self.alpha is not None:
            repr += f" alpha={self.alpha}"
        if self.beta is not None:
            repr += f" beta={self.beta}"
        return repr


class DANGREMOND(BaseSubComponent):
    """DAM transmission of d'Angremond et al. (1996).

    .. code-block:: text

        DAM DANGREMOND [hgt] [slope] [Bk]

    This option specifies transmission coefficients dependent on the incident wave
    conditions at the obstacle and on the obstacle height (which may be submerged).

    References
    ----------
    d'Angremond, K., Van Der Meer, J.W. and De Jong, R.J., 1996. Wave transmission at
    low-crested structures. In Coastal Engineering 1996 (pp. 2418-2427).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import DANGREMOND
        transm = DANGREMOND(hgt=3.0, slope=60, Bk=10.0)
        print(transm.render())

    """

    model_type: Literal["dangremond", "DANGREMOND"] = Field(
        default="dangremond", description="Model type discriminator"
    )
    hgt: float = Field(
        description=(
            "The elevation of the top of the obstacle above reference level (same "
            "reference level as for bottom etc.); use a negative value if the top is "
            "below that reference level"
        ),
    )
    slope: float = Field(
        description="The slope of the obstacle (in degrees)", ge=0.0, le=90.0
    )
    Bk: float = Field(description="The crest width of the obstacle")

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = f"DAM {self.model_type.upper()}"
        repr += f" hgt={self.hgt} slope={self.slope} Bk={self.Bk}"
        return repr


class REFL(BaseSubComponent):
    """Obstacle reflections.

    .. code-block:: text

        REFL [reflc]

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import REFL
        refl = REFL()
        print(refl.render())
        refl = REFL(reflc=0.5)
        print(refl.render())

    """

    model_type: Literal["refl", "REFL"] = Field(
        default="refl", description="Model type discriminator"
    )
    reflc: Optional[float] = Field(
        default=None,
        description=(
            "Constant reflection coefficient (ratio of reflected over incoming "
            "significant wave height) (SWAN default: 1.0)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = "REFL"
        if self.reflc is not None:
            repr += f" reflc={self.reflc}"
        return repr


class RSPEC(BaseSubComponent):
    """Specular reflection.

    .. code-block:: text

        RSPEC

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import RSPEC
        refl = RSPEC()
        print(refl.render())

    """

    model_type: Literal["rspec", "RSPEC"] = Field(
        default="rspec", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        return "RSPEC"


class RDIFF(BaseSubComponent):
    """Diffuse reflection.

    .. code-block:: text

        RDIFF [pown]

    Specular reflection where incident waves are scattered over reflected direction.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import RDIFF
        refl = RDIFF()
        print(refl.render())
        refl = RDIFF(pown=1.0)
        print(refl.render())

    """

    model_type: Literal["rdiff", "RDIFF"] = Field(
        default="rdiff", description="Model type discriminator"
    )
    pown: Optional[float] = Field(
        default=None,
        description=(
            "Each incoming direction θ is scattered over reflected direction θ_refl "
            "according to cos^pown(θ-θ_refl). The parameter `pown` indicates the width"
            "of the redistribution function (SWAN default: 1.0)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = "RDIFF"
        if self.pown is not None:
            repr += f" pown={self.pown}"
        return repr


class FREEBOARD(BaseSubComponent):
    """Freeboard dependent transmission and reflection.

    .. code-block:: text

        FREEBOARD [hgt] [gammat] [gammar] [QUAY]

    With this option the user indicates that the fixed transmission `trcoef` and
    reflection `reflc` coefficients are freeboard dependent. The freeboard dependency
    has no effect on the transmission coefficient as computed using the DAM option.

    Notes
    -----
    See the Scientific/Technical documentation for background information on the
    `gammat` and `gammar` shape parameters.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import FREEBOARD
        freeboard = FREEBOARD(hgt=2.0)
        print(freeboard.render())
        freeboard = FREEBOARD(hgt=2.0, gammat=1.0, gammar=1.0, quay=True)
        print(freeboard.render())

    """

    model_type: Literal["freeboard", "FREEBOARD"] = Field(
        default="freeboard", description="Model type discriminator"
    )
    hgt: float = Field(
        description=(
            "The elevation of the top of the obstacle or height of the quay above the "
            "reference level (same reference level as for the bottom). Use a negative "
            "value if the top is below that reference level. In case `hgt` is also "
            "specified in the DAM option, both values of `hgt` should be equal for "
            "consistency"
        ),
    )
    gammat: Optional[float] = Field(
        default=None,
        description=(
            "Shape parameter of relative freeboard dependency of transmission "
            "coefficient. This parameter should be higher than zero (SWAN default 1.0)"
        ),
        gt=0.0,
    )
    gammar: Optional[float] = Field(
        default=None,
        description=(
            "Shape parameter of relative freeboard dependency of reflection "
            "coefficient. This parameter should be higher than zero (SWAN default 1.0)"
        ),
        gt=0.0,
    )
    quay: bool = Field(
        default=False,
        description=(
            "With this option the user indicates that the freeboard dependency of the "
            "transmission and reflection coefficients also depends on the relative "
            "position of an obstacle-linked grid point with respect to the position "
            "of the obstacle line representing the edge of a quay. In case the active "
            "grid point is on the deeper side of the obstacle, then the correction "
            "factors are applied using the parameters `hgt`, `gammat` and `gammar`."
            "In case the active grid point is on the shallower side of the obstacle, "
            "the reflection coefficient is set to 0 and the transmission coefficient "
            "to 1."
        ),
    )

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = "FREEBOARD"
        if self.hgt is not None:
            repr += f" hgt={self.hgt}"
        if self.gammat is not None:
            repr += f" gammat={self.gammat}"
        if self.gammar is not None:
            repr += f" gammar={self.gammar}"
        if self.quay:
            repr += " QUAY"
        return repr


class LINE(BaseSubComponent):
    """Line of points to define obstacle location.

    .. code-block:: text

        LINE < [xp] [yp] >

    With this option the user indicates that the fixed transmission `trcoef` and
    reflection `reflc` coefficients are freeboard dependent. The freeboard dependency
    has no effect on the transmission coefficient as computed using the DAM option.

    Notes
    -----
    Points coordinates should be provided in m If Cartesian coordinates are used or in
    degrees if spherical coordinates are used (see command `COORD`). At least two
    corner points must be provided.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.physics import LINE
        line = LINE(xp=[174.1, 174.2, 174.3], yp=[-39.1, -39.1, -39.1])
        print(line.render())

    """

    model_type: Literal["line", "LINE"] = Field(
        default="line", description="Model type discriminator"
    )
    xp: list[float] = Field(
        description="The x-coordinates of the points defining the line", min_length=2
    )
    yp: list[float] = Field(
        description="The y-coordinates of the points defining the line", min_length=2
    )

    @model_validator(mode="after")
    def check_length(self) -> "LINE":
        """Check that the length of xp and yp are the same."""
        if len(self.xp) != len(self.yp):
            raise ValueError("xp and yp must be the same length")
        return self

    def cmd(self) -> str:
        """Command file string for this subcomponent."""
        repr = "LINE"
        for xp, yp in zip(self.xp, self.yp):
            repr += f" {xp} {yp}"
        return repr
