"""SWAN numerics subcomponents."""
from typing import Annotated, Literal, Optional, Union
from pydantic import field_validator, Field, model_validator
from abc import ABC
from pydantic_numpy.typing import Np2DArray
import numpy as np

from rompy.swan.subcomponents.base import BaseSubComponent


class BSBT(BaseSubComponent):
    """BSBT first order propagation scheme.

    .. code-block:: text

        BSTB

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import BSBT
        scheme = BSBT()
        print(scheme.render())

    """

    model_type: Literal["bsbt", "BSBT"] = Field(
        default="bsbt", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        return "BSBT"


class GSE(BaseSubComponent):
    """Garden-sprinkler effect.

    .. code-block:: text

        GSE [waveage] Sec|MIn|HR|DAy

    Garden-sprinkler effect is to be counteracted in the S&L propagation scheme
    (default for nonstationary regular grid computations) or in the propagation
    scheme for unstructured grids by adding a diffusion term to the basic equation.
    This may affect the numerical stability of SWAN.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import GSE
        scheme = GSE(waveage=1, units="day")
        print(scheme.render())

    """

    model_type: Literal["gse", "GSE"] = Field(
        default="gse", description="Model type discriminator"
    )
    waveage: Optional[float] = Field(
        default=None,
        description=(
            "The time interval used to determine the diffusion which counteracts the "
            "so-called garden-sprinkler effect. The default value of `waveage` is "
            "zero, i.e. no added diffusion. The value of `waveage` should correspond "
            "to the travel time of the waves over the computational region."
        ),
    )
    units: Literal["sec", "min", "hr", "day"] = Field(
        default="hr", description="Units for waveage",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "GSE"
        if self.waveage is not None:
            repr += f" waveage={self.waveage} {self.units.upper()}"
        return repr


class STAT(BaseSubComponent):
    """Iteration termination for stationary computations.

    .. code-block:: text

        STAT [mxitst] [alfa]

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import STAT
        stat = STAT()
        print(stat.render())
        stat = STAT(mxitst=10, alfa=0.1)
        print(stat.render())

    """
    model_type: Literal["stat", "STAT"] = Field(
        default="stat", description="Model type discriminator"
    )
    mxitst: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of iterations for stationary computations, the "
            "computation stops when this number is exceeded (SWAN default: 50)"
        ),
    )
    alfa: Optional[float] = Field(
        default=None,
        description=(
            "Proportionality constant used in the frequency-dependent under-"
            "relaxation technique. Based on experiences, a suggestion for this "
            "parameter is `alfa = 0.01`. In case of diffraction computations, the use "
            "of this parameter is recommended (SWAN default: 0.00)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "STAT"
        if self.mxitst is not None:
            repr += f" mxitst={self.mxitst}"
        if self.alfa is not None:
            repr += f" alfa={self.alfa}"
        return repr


class NONSTAT(BaseSubComponent):
    """Iteration termination for nonstationary computations.

    .. code-block:: text

        NONSTAT [mxitst]

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import NONSTAT
        nonstat = NONSTAT()
        print(nonstat.render())
        nonstat = NONSTAT(mxitst=3)
        print(nonstat.render())

    """
    model_type: Literal["nonstat", "NONSTAT"] = Field(
        default="nonstat", description="Model type discriminator"
    )
    mxitst: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of iterations for stationary computations, the "
            "computation moves to the next time step when this number is exceeded "
            "(SWAN default: 1)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "NONSTAT"
        if self.mxitst is not None:
            repr += f" mxitst={self.mxitst}"
        return repr


class STOPC(BaseSubComponent):
    """Stop the iterative procedure.

    .. code-block:: text

        STOPC [dabs] [drel] [curvat] [npnts] ->STAT|NONSTAT [limiter]

    With this option the user can influence the criterion for terminating the iterative
    procedure in the SWAN computations (both stationary and nonstationary). The
    criterion makes use of the second derivative, or curvature, of the iteration curve
    of the significant wave height. As the solution of a simulation approaches full
    convergence, the curvature of the iteration curve will tend to zero. SWAN stops the
    process if the relative change in Hs from one iteration to the next is less than
    `drel` and the curvature of the iteration curve of Hs normalized with Hs is less
    than `curvat` or the absolute change in Hs from one iteration to the next is less
    than `dabs`. Both conditions need to be fulfilled in more than fraction `npnts`
    percent of all wet grid points.

    With respect to the QC modelling, another stopping criteria will be employed.
    Namely, SWAN stops the iteration process if the absolute change in Hs from one
    iterate to another is less than `dabs` * Hinc, where Hinc is the representative
    incident wave height, or the relative change in Hs from one to the next iteration
    is less than `drel`. These criteria must be fulfilled in more than `npnts`
    percent of all active, well-defined points.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import STOPC
        stop = STOPC()
        print(stop.render())
        stop=STOPC(
            dabs=0.005,
            drel=0.01,
            curvat=0.005,
            npnts=99.5,
            mode=dict(model_type="nonstat", mxitst=1),
        )
        print(stop.render())

    """
    model_type: Literal["stopc", "STOPC"] = Field(
        default="stopc", description="Model type discriminator"
    )
    dabs: Optional[float] = Field(
        default=None,
        description=(
            "Maximum absolute change in Hs from one iteration to the next "
            "(SWAN default: 0.005 [m] or 0.05 [-] in case of QC model)"
        ),
    )
    drel: Optional[float] = Field(
        default=None,
        description=(
            "Maximum relative change in Hs from one iteration to the next "
            "(SWAN default: 0.01 [-])"
        ),
    )
    curvat: Optional[float] = Field(
        default=None,
        description=(
            "Maximum curvature of the iteration curve of Hs normalised with Hs "
            "(SWAN default: 0.005 [-] (not used in the QC model))"
        ),
    )
    npnts: Optional[float] = Field(
        default=None,
        description=(
            "Percentage of points in the computational grid above which the stopping "
            "criteria needs to be satisfied (SWAN default: 99.5 [-])"
        ),
    )
    mode: Optional[Union[STAT, NONSTAT]] = Field(
        default=None,
        description=(
            "Iteration termination criteria for stationary or nonstationary runs"
        ),
        discriminator="model_type",
    )
    limiter: Optional[float] = Field(
        default=None,
        description=(
            "Determines the maximum change per iteration of the energy density per "
            "spectral-bin given in terms of a fraction of the omni-directional "
            "Phillips level (SWAN default: 0.1)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "STOPC"
        if self.dabs is not None:
            repr += f" dabs={self.dabs}"
        if self.drel is not None:
            repr += f" drel={self.drel}"
        if self.curvat is not None:
            repr += f" curvat={self.curvat}"
        if self.npnts is not None:
            repr += f" npnts={self.npnts}"
        if self.mode is not None:
            repr += f" {self.mode.render()}"
        if self.limiter is not None:
            repr += f" limiter={self.limiter}"
        return repr


class DIRIMPL(BaseSubComponent):
    """Numerical scheme for refraction.

    .. code-block:: text

        DIRIMPL [cdd]

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import DIRIMPL
        dirimpl = DIRIMPL()
        print(dirimpl.render())
        dirimpl = DIRIMPL(cdd=0.5)
        print(dirimpl.render())

    """
    model_type: Literal["dirimpl", "DIRIMPL"] = Field(
        default="dirimpl", description="Model type discriminator"
    )
    cdd: Optional[float] = Field(
        default=None,
        description=(
            "A value of `cdd=0` corresponds to a central scheme and has the largest "
            "accuracy (diffusion ≈ 0) but the computation may more easily generate"
            "spurious fluctuations. A value of `cdd=1` corresponds to a first order"
            "upwind scheme and it is more diffusive and therefore preferable if "
            "(strong) gradients in depth or current are present (SWAN default: 0.5)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "DIRIMPL"
        if self.cdd is not None:
            repr += f" cdd={self.cdd}"
        return repr


class SIGIMPL(BaseSubComponent):
    """Frequency shifting accuracy.

    .. code-block:: text

        SIGIMpl [cfl] [eps2] [outp] [niter]

    Controls the accuracy of computing the frequency shifting and the stopping
    criterion and amount of output for the SIP solver (used in the computations in the
    presence of currents or time varying depth)

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import SIGIMPL
        sigimpl = SIGIMPL()
        print(sigimpl.render())
        sigimpl = SIGIMPL(css=0.5, eps2=1e-4, outp=0, niter=20)
        print(sigimpl.render())

    """
    model_type: Literal["sigimpl", "SIGIMPL"] = Field(
        default="sigimpl", description="Model type discriminator"
    )
    css: Optional[float] = Field(
        default=None,
        description=(
            "A value of `css=0` corresponds to a central scheme and has the largest "
            "accuracy (diffusion ≈ 0) but the computation may more easily generate "
            "spurious fluctuations. A value of `css=1` corresponds to a first order "
            "upwind scheme and it is more diffusive and therefore preferable if "
            "(strong) gradients in depth or current are present (SWAN default: 0.5)"
        ),
    )
    eps2: Optional[float] = Field(
        default=None,
        description=(
            "Relative stopping criterion to terminate the linear solver (SIP or SOR). "
            "(SWAN default: 1.e-4 in case of SIP and 1.e-6 in case of SOR)"
        ),
    )
    outp: Optional[Literal[0, 1, 2, 3]] = Field(
        default=None,
        description=(
            "Output for the iterative solver:\n\n* 0 = no output\n* 1 = additional "
            "information about the iteration process is written to the PRINT file "
            "\n* 2 = gives a maximal amount of output concerning the iteration "
            "process\n* 3 = summary of the iteration process\n\n(SWAN default: 0)"
        ),
    )
    niter: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of iterations for the linear solver (SWAN default: 20 in "
            "case of SIP and 1000 in case of SOR)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SIGIMPL"
        if self.css is not None:
            repr += f" css={self.css}"
        if self.eps2 is not None:
            repr += f" eps2={self.eps2}"
        if self.outp is not None:
            repr += f" outp={self.outp}"
        if self.niter is not None:
            repr += f" niter={self.niter}"
        return repr


class CTHETA(BaseSubComponent):
    """Prevents excessive directional turning.

    .. code-block:: text

        CTheta [cfl]

    This option prevents an excessive directional turning at a single grid point or
    vertex due to a very coarse bathymetry or current locally. This option limits the
    directional turning rate cθ based on the CFL restriction. (See Eq. 3.41 of
    Scientific/Technical documentation). See also the final remark in Section 2.6.3.
    Note that if this command is not specified, then the limiter is not activated.

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import CTHETA
        ctheta = CTHETA()
        print(ctheta.render())
        ctheta = CTHETA(cfl=0.9)
        print(ctheta.render())

    """
    model_type: Literal["ctheta", "CTHETA"] = Field(
        default="ctheta", description="Model type discriminator"
    )
    cfl: Optional[float] = Field(
        default=None,
        description=(
            "Upper limit for the CFL restriction for ctheta. A suggestion for this "
            "parameter is `cfl = 0.9` (SWAN default: 0.9 when CTHETA is activated)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "CTHETA"
        if self.cfl is not None:
            repr += f" cfl={self.cfl}"
        return repr


class CSIGMA(BaseSubComponent):
    """Prevents excessive directional turning.

    .. code-block:: text

        CSigma [cfl]

    This option prevents an excessive frequency shifting at a single grid point or
    vertex due to a very coarse bathymetry or current locally. This option limits the
    frequency shifting rate csigma based on the CFL restriction. See also the final
    remark in Section 2.6.3. Note that if this command is not specified, then the
    limiter is not activated.

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import CSIGMA
        csigma = CSIGMA()
        print(csigma.render())
        csigma = CSIGMA(cfl=0.9)
        print(csigma.render())

    """
    model_type: Literal["ctheta", "CTHETA"] = Field(
        default="ctheta", description="Model type discriminator"
    )
    cfl: Optional[float] = Field(
        default=None,
        description=(
            "Upper limit for the CFL restriction for csigma. A suggestion for this "
            "parameter is `cfl = 0.9` (SWAN default: 0.9 when CSIGMA is activated)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "CSIGMA"
        if self.cfl is not None:
            repr += f" cfl={self.cfl}"
        return repr


class SETUP(BaseSubComponent):
    """Stop criteria in the computation of wave setup.

    .. code-block:: text

        SETUP [eps2] [outp] [niter]

    Controls the stopping criterion and amount of output for the SOR solver in the
    computation of the wave-induced set-up.

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.subcomponents.numerics import SETUP
        setup = SETUP()
        print(setup.render())
        setup = SETUP(eps2=1e-4, outp=0, niter=20)
        print(setup.render())

    """
    model_type: Literal["setup", "SETUP"] = Field(
        default="setup", description="Model type discriminator"
    )
    eps2: Optional[float] = Field(
        default=None,
        description=(
            "Relative stopping criterion to terminate the linear solver (SIP or SOR). "
            "(SWAN default: 1.e-4 in case of SIP and 1.e-6 in case of SOR)"
        ),
    )
    outp: Optional[Literal[0, 1, 2, 3]] = Field(
        default=None,
        description=(
            "Output for the iterative solver:\n\n* 0 = no output\n* 1 = additional "
            "information about the iteration process is written to the PRINT file "
            "\n* 2 = gives a maximal amount of output concerning the iteration process "
            "\n* 3 = summary of the iteration process\n\n(SWAN default: 0)"
        ),
    )
    niter: Optional[int] = Field(
        default=None,
        description=(
            "Maximum number of iterations for the linear solver (SWAN default: 20 in "
            "case of SIP and 1000 in case of SOR)"
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "SETUP"
        if self.eps2 is not None:
            repr += f" eps2={self.eps2}"
        if self.outp is not None:
            repr += f" outp={self.outp}"
        if self.niter is not None:
            repr += f" niter={self.niter}"
        return repr
