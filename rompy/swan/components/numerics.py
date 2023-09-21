"""Model numerics components."""
import logging
from typing import Any, Literal, Optional, Union, Annotated
from pydantic import field_validator, model_validator, Field, FieldValidationInfo

from rompy.swan.components.base import BaseComponent
from rompy.swan.types import IDLA, PhysicsOff
from rompy.swan.subcomponents.numerics import (
    BSBT,
    GSE,
    STOPC,
    DIRIMPL,
    SIGIMPL,
    CTHETA,
    CSIGMA,
    SETUP,
)


logger = logging.getLogger(__name__)


PROP_TYPE = Annotated[
    Union[BSBT, GSE],
    Field(description="Propagation scheme", discriminator="model_type"),
]


class PROP(BaseComponent):
    """Propagation scheme.

    .. code-block:: text

        PROP BSTB|GSE

    Notes
    -----
    * The scheme defaults to `S&L` and `SORDUP` for nonstationary and stationary
      simulations if not specified.
    * All schemes (BSBT, SORDUP and S&L) can be used in combination with curvilinear
      grids. With the higher order schemes (S&L and SORDUP) it is important to use a
      gradually varying grid otherwise there may be a severe loss of accuracy. If sharp
      transitions in the grid cannot be avoided it is safer to use the BSBT scheme.
    * In the computation with unstructured meshes, a lowest order upwind scheme will be
      employed. This scheme is very robust but rather diffusive. This may only be
      significant for the case when swell waves propagate over relative large distances
      (in the order of thousands of kilometers) within the model domain. However and
      most fortunately, in such a case this will alleviate the garden-sprinkler effect.
    * Alleviating the garden-sprinkler effect by adding some diffusion makes the SWAN
      computation conditionally stable. You can either use (i) a smaller time step,
      (ii) a lower value of `waveage`, (iii) better resolution in the directional
      space, or (iv) worse resolution in the geographic space, in order of preference,
      to make the model stable when necessary.

    Examples
    --------

    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.numerics import PROP
        prop = PROP()
        print(prop.render())
        prop = PROP(scheme=dict(model_type="bsbt"))
        print(prop.render())
        prop = PROP(scheme=dict(model_type="gse", waveage=5, units="hr"))
        print(prop.render())

    """
    model_type: Literal["prop", "PROP"] = Field(
        default="prop", description="Model type discriminator"
    )
    scheme: Optional[PROP_TYPE] = Field(
        default=None,
        description=(
            "Propagation scheme, by default S&L for nonstationary and SORDUP for "
            "stationary computation."
        ),
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "PROP"
        if self.scheme is not None:
            repr += f" {self.scheme.render()}"
        return repr


class NUMERIC(BaseComponent):
    """Numerical properties.

    .. code-block:: text

        NUMeric ( STOPC [dabs] [drel] [curvat] [npnts] ->STAT|NONSTAT [limiter] ) &
            ( DIRimpl [cdd] ) ( SIGIMpl [css] [eps2] [outp] [niter] ) &
            ( CTheta [cfl] ) ( CSigma [cfl] ) ( SETUP [eps2] [outp] [niter] )

    Examples
    --------
    .. ipython:: python
        :okwarning:
        :okexcept:

        from rompy.swan.components.numerics import NUMERIC
        numeric = NUMERIC()
        print(numeric.render())
        numeric = NUMERIC(
            stopc=dict(dabs=0.05, drel=0.01, curvat=0.05, npnts=99.5),
            dirimpl=dict(cdd=0.5),
            sigimpl=dict(css=0.5, eps2=1e-4, outp=0, niter=20),
            ctheta=dict(cfl=0.9),
            csigma=dict(cfl=0.9),
            setup=dict(eps2=1e-4, outp=0, niter=20),
        )
        print(numeric.render())

    """
    model_type: Literal["numeric", "NUMERIC"] = Field(
        default="numeric", description="Model type discriminator"
    )
    stopc: Optional[STOPC] = Field(
        default=None, description="Iteration termination criteria",
    )
    dirimpl: Optional[DIRIMPL] = Field(
        default=None, description="Numerical scheme for refraction",
    )
    sigimpl: Optional[SIGIMPL] = Field(
        default=None, description="Frequency shifting accuracy",
    )
    ctheta: Optional[CTHETA] = Field(
        default=None, description="Prevents excessive directional turning",
    )
    csigma: Optional[CSIGMA] = Field(
        default=None, description="Prevents excessive frequency shifting",
    )
    setup: Optional[SETUP] = Field(
        default=None, description="Stop criteria in the computation of wave setup",
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "NUMERIC"
        if self.stopc is not None:
            repr += f" {self.stopc.render()}"
        if self.dirimpl is not None:
            repr += f" {self.dirimpl.render()}"
        if self.sigimpl is not None:
            repr += f" {self.sigimpl.render()}"
        if self.ctheta is not None:
            repr += f" {self.ctheta.render()}"
        return repr
