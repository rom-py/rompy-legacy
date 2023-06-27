"""SWAN startup subcomponents."""
from typing import Literal
from pydantic import Field

from rompy.swan.subcomponents.base import BaseSubComponent


class CARTESIAN(BaseSubComponent):
    """Cartesian coordinates.

    `CARTESIAN`

    All locations and distances are in m. Coordinates are given with respect
    to x- and y-axes chosen by the user in the various commands.

    """

    model_type: Literal["cartesian"] = Field(
        default="cartesian", description="Model type discriminator"
    )


class SPHERICAL(BaseSubComponent):
    """Spherical coordinates.

    `SPHERICAL [CCM|QC]`

    Notes
    -----
    projection options:
    - CCM: central conformal Mercator. The horizontal and vertical scales are
      uniform in terms of cm/degree over the area shown. In the centre of the scale
      is identical to that of the conventional Mercator projection (but only at
      that centre). The area in the projection centre is therefore exactly conformal.
    - QC: the projection method is quasi-cartesian, i.e. the horizontal and vertical
      scales are equal to one another in terms of cm/degree.

    All coordinates of locations and geographical grid sizes are given in degrees;`x`
    is longitude with `x = 0` being the Greenwich meridian and `x > 0` is East of this
    meridian; `y` is latitude with `y > 0` being the Northern hemisphere. Input and
    output grids have to be oriented with their x-axis to the East; mesh sizes are in
    degrees. All other distances are in meters.

    Note that spherical coordinates can also be used for relatively small areas, say 10
    or 20 km horizontal dimension. This may be useful if one obtains the boundary
    conditions by nesting in an oceanic model which is naturally formulated in
    spherical coordinates. Note that in case of spherical coordinates regular grids
    must always be oriented E-W, N-S, i.e. `alpc`=0`, `alpinp`=0`, `alpfr`=0`
    (see commands CGRID, INPUT GRID and FRAME, respectively).

    """

    model_type: Literal["spherical"] = Field(
        default="spherical", description="Model type discriminator"
    )
    projection: Literal["ccm", "qc"] = Field(
        default="ccm",
        description="Defines the projection method in case of spherical coordinates"
    )

    def cmd(self) -> str:
        """Render subcomponent cmd."""
        repr = super().cmd()
        if self.projection is not None:
            repr += f" {self.projection.upper()}"
        return repr
