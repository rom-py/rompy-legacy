"""
SWAN Components Group Module

This module provides group components for organizing SWAN model configurations in the ROMPY framework.
"""

import logging
from typing import Annotated, Literal, Optional, Union

from pydantic import Field, field_validator, model_validator

from rompy.swan.components.base import BaseComponent
from rompy.swan.components.inpgrid import (CURVILINEAR, ICE, REGULAR,
                                           UNSTRUCTURED, WIND)
from rompy.swan.components.lockup import COMPUTE_NONSTAT, COMPUTE_STAT, STOP
from rompy.swan.components.output import (BLOCK, BLOCKS, CURVES, FRAME, GROUP,
                                          ISOLINE, NESTOUT, NGRID,
                                          NGRID_UNSTRUCTURED, OUTPUT_OPTIONS,
                                          POINTS, POINTS_FILE, QUANTITIES, RAY,
                                          SPECIAL_NAMES, SPECOUT, TABLE, TEST)
from rompy.swan.components.physics import (BRAGG, BRAGG_FILE, BRAGG_FT,
                                           BREAKING_BKD, BREAKING_CONSTANT,
                                           DIFFRACTION, FRICTION_COLLINS,
                                           FRICTION_JONSWAP, FRICTION_MADSEN,
                                           FRICTION_RIPPLES, GEN1, GEN2, GEN3,
                                           LIMITER, MUD, NEGATINP, OBSTACLES,
                                           OFFS, QUADRUPL, SCAT, SETUP, SICE,
                                           SICE_D15, SICE_M18, SICE_R19,
                                           SICE_R21B, SSWELL_ARDHUIN,
                                           SSWELL_ROGERS, SSWELL_ZIEGER,
                                           SURFBEAT, TRIAD, TRIAD_DCTA,
                                           TRIAD_LTA, TRIAD_SPB, TURBULENCE,
                                           VEGETATION, WCAPPING_AB,
                                           WCAPPING_KOMEN)
from rompy.swan.components.startup import COORDINATES, MODE, PROJECT, SET
from rompy.swan.types import PhysicsOff

logger = logging.getLogger(__name__)


class BaseGroupComponent(BaseComponent):
    """Base group component.

    Base class for SWAN group components. This base class modifies the `render()`
    method to allow rendering of a list of components. It is not intended to be used
    directly but to be subclassed by other group components.

    """

    model_type: Literal["group", "GROUP"] = Field(
        default="group", description="Model type discriminator"
    )

    def render(self, *args, **kwargs) -> str:
        """Override base class to allow rendering list of components."""
        cmds = []
        for cmd in self.cmd():
            cmds.append(super().render(cmd))
        return "\n\n".join(cmds)


# =====================================================================================
# Startup
# =====================================================================================
PROJECT_TYPE = Annotated[PROJECT, Field(description="Project component")]
SET_TYPE = Annotated[SET, Field(description="Set component")]
MODE_TYPE = Annotated[MODE, Field(description="Mode component")]
COORDINATES_TYPE = Annotated[COORDINATES, Field(description="Coordinates component")]


class STARTUP(BaseGroupComponent):
    """Startup group component.

    .. code-block:: text

        PROJECT ...
        SET ...
        MODE ...
        COORDINATES ...

    This group component is used to group individual startup components. Only fields
    that are explicitly prescribed are rendered by this group component.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.startup import PROJECT, SET, MODE, COORDINATES
        from rompy.swan.components.group import STARTUP
        proj = PROJECT(nr="01")
        set = SET(level=0.5, direction_convention="nautical")
        mode = MODE(kind="nonstationary", dim="twodimensional")
        coords = COORDINATES(kind=dict(model_type="spherical", projection="ccm"))
        startup = STARTUP(
            project=proj,
            set=set,
            mode=mode,
            coordinates=coords,
        )
        print(startup.render())

    """

    model_type: Literal["startup", "STARTUP"] = Field(
        default="startup", description="Model type discriminator"
    )
    project: Optional[PROJECT_TYPE] = Field(default=None)
    set: Optional[SET_TYPE] = Field(default=None)
    mode: Optional[MODE_TYPE] = Field(default=None)
    coordinates: Optional[COORDINATES_TYPE] = Field(default=None)

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = []
        if self.project is not None:
            repr += [f"{self.project.cmd()}"]
        if self.set is not None:
            repr += [f"{self.set.cmd()}"]
        if self.mode is not None:
            repr += [f"{self.mode.cmd()}"]
        if self.coordinates is not None:
            repr += [f"{self.coordinates.cmd()}"]
        return repr


# =====================================================================================
# Inpgrid
# =====================================================================================
INPGRID_TYPE = Annotated[
    Union[REGULAR, CURVILINEAR, UNSTRUCTURED, WIND, ICE],
    Field(discriminator="model_type"),
]


class INPGRIDS(BaseGroupComponent):
    """SWAN input grids group component.

    .. code-block:: text

        INPGRID ...
        READGRID ...

        INPGRID ...
        READGRID ...

        ...

    This group component is a convenience to allow defining and rendering
    a list of input grid components.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.inpgrid import REGULAR, ICE
        from rompy.swan.components.group import INPGRIDS
        inpgrid_bottom = REGULAR(
            grid_type="bottom",
            excval=-99.0,
            xpinp=172.0,
            ypinp=-41.0,
            alpinp=0.0,
            mxinp=99,
            myinp=99,
            dxinp=0.005,
            dyinp=0.005,
            readinp=dict(fname1="bottom.txt"),
        )
        inpgrid_wind = REGULAR(
            grid_type="wind",
            excval=-99.0,
            xpinp=172.0,
            ypinp=-41.0,
            alpinp=0.0,
            mxinp=99,
            myinp=99,
            dxinp=0.005,
            dyinp=0.005,
            readinp=dict(fname1="wind.txt"),
            nonstationary=dict(
                tbeg="2019-01-01T00:00:00",
                tend="2019-01-07 00:00:00",
                delt=3600,
                dfmt="hr",
            ),
        )
        inpgrid_ice_cte = ICE(aice=0.8, hice=2.0)
        inpgrids = INPGRIDS(inpgrids=[inpgrid_bottom, inpgrid_wind, inpgrid_ice_cte])
        print(inpgrids.render())

    """

    model_type: Literal["inpgrids"] = Field(
        default="inpgrids", description="Model type discriminator"
    )
    inpgrids: list[INPGRID_TYPE] = Field(
        min_length=1,
        description="List of input grid components",
    )

    @field_validator("inpgrids")
    @classmethod
    def ensure_unique_grid_type(cls, inpgrids: INPGRID_TYPE) -> INPGRID_TYPE:
        """Ensure that each grid type is unique."""
        grid_types = [inp.grid_type for inp in inpgrids if hasattr(inp, "grid_type")]
        if len(grid_types) != len(set(grid_types)):
            raise ValueError("Each grid type must be unique")
        return inpgrids

    def cmd(self) -> str | list:
        repr = []
        for inpgrid in self.inpgrids:
            repr += [inpgrid.cmd()]
        return repr


# =====================================================================================
# Physics
# =====================================================================================
NEGATINP_TYPE = Annotated[NEGATINP, Field(description="Negative wind input component")]
QUADRUPL_TYPE = Annotated[QUADRUPL, Field(description="Quadruplets component")]
VEGETATION_TYPE = Annotated[VEGETATION, Field(description="Vegetation component")]
MUD_TYPE = Annotated[MUD, Field(description="Mud component")]
TURBULENCE_TYPE = Annotated[TURBULENCE, Field(description="Turbulence component")]
LIMITER_TYPE = Annotated[LIMITER, Field(description="Limiter component")]
OBSTACLE_TYPE = Annotated[OBSTACLES, Field(description="Obstacle group component")]
SETUP_TYPE = Annotated[SETUP, Field(description="Setup component")]
DIFFRACTION_TYPE = Annotated[DIFFRACTION, Field(description="Diffraction component")]
SURFBEAT_TYPE = Annotated[SURFBEAT, Field(description="Surfbeat component")]
SCAT_TYPE = Annotated[SCAT, Field(description="Scattering component")]
OFF_TYPE = Annotated[OFFS, Field(description="Deactivate components")]
GEN_TYPE = Annotated[
    Union[GEN1, GEN2, GEN3],
    Field(description="Wave generation component", discriminator="model_type"),
]
SSWELL_TYPE = Annotated[
    Union[SSWELL_ROGERS, SSWELL_ARDHUIN, SSWELL_ZIEGER],
    Field(description="Swell dissipation component", discriminator="model_type"),
]
WCAPPING_TYPE = Annotated[
    Union[WCAPPING_KOMEN, WCAPPING_AB],
    Field(description="Whitecapping component", discriminator="model_type"),
]
BREAKING_TYPE = Annotated[
    Union[BREAKING_CONSTANT, BREAKING_BKD],
    Field(description="Wave breaking component", discriminator="model_type"),
]
FRICTION_TYPE = Annotated[
    Union[FRICTION_JONSWAP, FRICTION_COLLINS, FRICTION_MADSEN, FRICTION_RIPPLES],
    Field(description="Bottom friction component", discriminator="model_type"),
]
TRIAD_TYPE = Annotated[
    Union[TRIAD, TRIAD_DCTA, TRIAD_LTA, TRIAD_SPB],
    Field(description="Triad interactions component", discriminator="model_type"),
]
SICE_TYPE = Annotated[
    Union[SICE, SICE_R19, SICE_D15, SICE_M18, SICE_R21B],
    Field(description="Sea ice component", discriminator="model_type"),
]
BRAGG_TYPE = Annotated[
    Union[BRAGG, BRAGG_FT, BRAGG_FILE],
    Field(description="Bragg scattering component", discriminator="model_type"),
]


class PHYSICS(BaseGroupComponent):
    """Physics group component.

    The physics group component is a convenience to allow specifying several individual
    components in a single command and check for consistency between them.

    Exemples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.group import PHYSICS
        gen = {"model_type": "gen3", "source_terms": {"model_type": "komen"}}
        phys = PHYSICS(gen=gen)
        print(phys.render())
        phys = PHYSICS(
            gen=dict(model_type="gen3", source_terms={"model_type": "st6c1"}),
            negatinp={"model_type": "negatinp", "rdcoef": 0.04},
            sswell={"model_type": "zieger"},
            breaking={"model_type": "constant", "alpha": 1.0, "gamma": 0.73},
            friction={"model_type": "jonswap", "cfjon": 0.038},
        )
        print(phys.render())

    """

    model_type: Literal["physics", "PHYSICS"] = Field(
        default="physics", description="Model type discriminator"
    )
    gen: Optional[GEN_TYPE] = Field(default=None)
    sswell: Optional[SSWELL_TYPE] = Field(default=None)
    negatinp: Optional[NEGATINP_TYPE] = Field(default=None)
    wcapping: Optional[WCAPPING_TYPE] = Field(default=None)
    quadrupl: Optional[QUADRUPL_TYPE] = Field(default=None)
    breaking: Optional[BREAKING_TYPE] = Field(default=None)
    friction: Optional[FRICTION_TYPE] = Field(default=None)
    triad: Optional[TRIAD_TYPE] = Field(default=None)
    vegetation: Optional[VEGETATION_TYPE] = Field(default=None)
    mud: Optional[MUD_TYPE] = Field(default=None)
    sice: Optional[SICE_TYPE] = Field(default=None)
    turbulence: Optional[TURBULENCE_TYPE] = Field(default=None)
    bragg: Optional[BRAGG_TYPE] = Field(default=None)
    limiter: Optional[LIMITER_TYPE] = Field(default=None)
    obstacle: Optional[OBSTACLE_TYPE] = Field(default=None)
    setup: Optional[SETUP_TYPE] = Field(default=None)
    diffraction: Optional[DIFFRACTION_TYPE] = Field(default=None)
    surfbeat: Optional[SURFBEAT_TYPE] = Field(default=None)
    scat: Optional[SCAT_TYPE] = Field(default=None)
    deactivate: Optional[OFF_TYPE] = Field(default=None)

    @field_validator("deactivate")
    @classmethod
    def deactivate_physics(cls, off: OFF_TYPE) -> OFF_TYPE:
        """Convert OFF to OFFS so list is rendered."""
        for phys in PhysicsOff:
            print(phys.value)
        return off

    @model_validator(mode="after")
    def negatinp_only_with_zieger(self) -> "PHYSICS":
        """Log a warning if NEGATINP is used with a non-ZIEGER SSWELL."""
        if self.negatinp is None:
            return self
        elif self.sswell is None:
            logger.warning(
                "The negative wind input NEGATINP is only intended to use with the "
                "swell dissipation SSWELL ZIEGER but no SSWELL has been specified."
            )
        elif self.sswell.model_type != "zieger":
            logger.warning(
                "The negative wind input NEGATINP is only intended to use with the "
                "swell dissipation SSWELL ZIEGER but the SSWELL "
                f"{self.sswell.model_type.upper()} has been specified."
            )
        return self

    def cmd(self):
        repr = []
        if self.gen is not None:
            repr += [self.gen.cmd()]
        if self.sswell is not None:
            repr += [f"{self.sswell.cmd()}"]
        if self.negatinp is not None:
            repr += [self.negatinp.cmd()]
        if self.wcapping is not None:
            repr += [self.wcapping.cmd()]
        if self.quadrupl is not None:
            repr += [self.quadrupl.cmd()]
        if self.breaking is not None:
            repr += [self.breaking.cmd()]
        if self.friction is not None:
            repr += [self.friction.cmd()]
        if self.triad is not None:
            repr += [self.triad.cmd()]
        if self.vegetation is not None:
            repr += [self.vegetation.cmd()]
        if self.mud is not None:
            repr += [self.mud.cmd()]
        if self.sice is not None:
            repr += [self.sice.cmd()]
        if self.turbulence is not None:
            repr += [self.turbulence.cmd()]
        if self.bragg is not None:
            repr += [self.bragg.cmd()]
        if self.limiter is not None:
            repr += [self.limiter.cmd()]
        if self.obstacle is not None:
            repr += self.obstacle.cmd()  # Object returns a list of components
        if self.setup is not None:
            repr += [self.setup.cmd()]
        if self.diffraction is not None:
            repr += [self.diffraction.cmd()]
        if self.surfbeat is not None:
            repr += [self.surfbeat.cmd()]
        if self.scat is not None:
            repr += [self.scat.cmd()]
        if self.deactivate is not None:
            repr += self.deactivate.cmd()  # Object returns a list of components
        return repr


# =====================================================================================
# Output
# =====================================================================================
FRAME_TYPE = Annotated[FRAME, Field(description="Frame locations component")]
GROUP_TYPE = Annotated[GROUP, Field(description="Group locations component")]
CURVE_TYPE = Annotated[CURVES, Field(description="Curve locations component")]
RAY_TYPE = Annotated[RAY, Field(description="Ray locations component")]
ISOLINE_TYPE = Annotated[ISOLINE, Field(description="Isoline locations component")]
QUANTITY_TYPE = Annotated[QUANTITIES, Field(description="Quantity component")]
OUTOPT_TYPE = Annotated[OUTPUT_OPTIONS, Field(description="Output options component")]
TABLE_TYPE = Annotated[TABLE, Field(description="Table write component")]
SPECOUT_TYPE = Annotated[SPECOUT, Field(description="Spectra write component")]
NESTOUT_TYPE = Annotated[NESTOUT, Field(description="Nest write component")]
TEST_TYPE = Annotated[TEST, Field(description="Intermediate write component")]
BLOCK_TYPE = Annotated[
    Union[BLOCK, BLOCKS],
    Field(description="Block write component", discriminator="model_type"),
]
POINTS_TYPE = Annotated[
    Union[POINTS, POINTS_FILE],
    Field(description="Points locations component", discriminator="model_type"),
]
NGRID_TYPE = Annotated[
    Union[NGRID, NGRID_UNSTRUCTURED],
    Field(description="Ngrid locations component", discriminator="model_type"),
]


class OUTPUT(BaseGroupComponent):
    """Output group component.

    .. code-block:: text

        FRAME 'sname' ...
        GROUP 'sname' ...
        CURVE 'sname' ...
        RAY 'rname' ...
        ISOLINE 'sname' 'rname' ...
        POINTS 'sname ...
        NGRID 'sname' ...
        QUANTITY ...
        OUTPUT OPTIONS ...
        BLOCK 'sname' ...
        TABLE 'sname' ...
        SPECOUT 'sname' ...
        NESTOUT 'sname ...

    This group component is used to define multiple types of output locations and
    write components in a single model. Only fields that are explicitly prescribed are
    rendered by this group component.

    Note
    ----
    The components prescribed are validated according to some constraints as defined
    in the SWAN manual:

    - The name `'sname'` of each Locations component must be unique.
    - The Locations `'sname'` assigned to each write component must be defined.
    - The BLOCK component must be associated with either a `FRAME` or `GROUP`.
    - The ISOLINE write component must be associated with a `RAY` component.
    - The NGRID and NESTOUT components must be defined together.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.output import POINTS, BLOCK, QUANTITIES, TABLE
        from rompy.swan.components.group import OUTPUT
        points = POINTS(sname="outpts", xp=[172.3, 172.4], yp=[-39, -39])
        quantity = QUANTITIES(
            quantities=[
                dict(output=["depth", "hsign", "tps", "dir", "tm01"], excv=-9),
            ]
        )
        times = dict(tbeg="2012-01-01T00:00:00", delt="PT30M", tfmt=1, dfmt="min")
        block = BLOCK(
            model_type="block",
            sname="COMPGRID",
            fname="./swangrid.nc",
            output=["depth", "hsign", "tps", "dir"],
            times=times,
        )
        table = TABLE(
            sname="outpts",
            format="noheader",
            fname="./swantable.nc",
            output=["hsign", "hswell", "dir", "tps", "tm01", "watlev", "qp"],
            times=times,
        )
        out = OUTPUT(
            points=points,
            quantity=quantity,
            block=block,
            table=table,
        )
        print(out.render())

    """

    model_type: Literal["output", "OUTPUT"] = Field(
        default="output", description="Model type discriminator"
    )
    frame: Optional[FRAME_TYPE] = Field(default=None)
    group: Optional[GROUP_TYPE] = Field(default=None)
    curve: Optional[CURVE_TYPE] = Field(default=None)
    ray: Optional[RAY_TYPE] = Field(default=None)
    isoline: Optional[ISOLINE_TYPE] = Field(default=None)
    points: Optional[POINTS_TYPE] = Field(default=None)
    ngrid: Optional[NGRID_TYPE] = Field(default=None)
    quantity: Optional[QUANTITY_TYPE] = Field(default=None)
    output_options: Optional[OUTOPT_TYPE] = Field(default=None)
    block: Optional[BLOCK_TYPE] = Field(default=None)
    table: Optional[TABLE_TYPE] = Field(default=None)
    specout: Optional[SPECOUT_TYPE] = Field(default=None)
    nestout: Optional[NESTOUT_TYPE] = Field(default=None)
    test: Optional[TEST_TYPE] = Field(default=None)
    _location_fields: list = ["frame", "group", "curve", "isoline", "points", "ngrid"]
    _write_fields: list = ["block", "table", "specout", "nestout"]

    @model_validator(mode="after")
    def write_locations_exists(self) -> "OUTPUT":
        """Ensure the location component requested by a write component exists."""
        for write in self.write_set:
            obj = getattr(self, write)
            if obj is None:
                continue
            snames = obj.sname if isinstance(obj.sname, list) else [obj.sname]
            for sname in snames:
                if sname in SPECIAL_NAMES:
                    return self
                try:
                    self._filter_location(sname)
                except ValueError as err:
                    raise ValueError(
                        f"Write component '{write}' specified with sname='{sname}' but "
                        f"no location component with sname='{sname}' has been defined"
                    ) from err
        return self

    @model_validator(mode="after")
    def locations_sname_unique(self) -> "OUTPUT":
        """Ensure same `sname` isn't used in more than one set of output locations."""
        duplicates = {x for x in self.snames if self.snames.count(x) > 1}
        if duplicates:
            raise ValueError(
                "The following snames are used to define more than one set of output "
                f"components: {duplicates}, please ensure each location component has "
                "a unique `sname`"
            )
        return self

    @model_validator(mode="after")
    def block_with_frame_or_group(self) -> "OUTPUT":
        """Ensure Block is only defined for FRAME or GROUP locations."""
        if self.block is not None:
            snames = self.block.sname
            if isinstance(snames, str):
                snames = [self.block.sname]
            for sname in snames:
                if sname not in ["BOTTGRID", "COMPGRID"]:
                    location = self._filter_location(sname)
                    component = location.model_type.upper().split("_")[0]
                    if component not in ["FRAME", "GROUP"]:
                        raise ValueError(
                            f"Block sname='{sname}' specified with {component} "
                            "location component but only only FRAME or GROUP "
                            "components are supported"
                        )
        return self

    @model_validator(mode="after")
    def isoline_ray_defined(self) -> "OUTPUT":
        """Ensure the isoline ray has been defined."""
        if self.isoline is not None:
            if self.ray is None:
                raise ValueError(
                    f"Isoline {self.isoline} requires RAY rname='{self.isoline.rname}'"
                    " but no RAY component has been defined"
                )
            elif self.ray.rname != self.isoline.rname:
                raise ValueError(
                    f"Isoline rname='{self.isoline.rname}' does not match "
                    f"the ray rname='{self.ray.rname}'"
                )
        return self

    @model_validator(mode="after")
    def ngrid_and_nestout(self) -> "OUTPUT":
        """Ensure NGRID and NESTOUT are specified together."""
        if self.ngrid is not None and self.nestout is None:
            raise ValueError(
                "NGRID component specified but no NESTOUT component has been defined"
            )
        elif self.ngrid is None and self.nestout is not None:
            raise ValueError(
                "NESTOUT component specified but no NGRID component has been defined"
            )
        elif self.ngrid is not None and self.nestout is not None:
            if self.ngrid.sname != self.nestout.sname:
                raise ValueError(
                    f"NGRID sname='{self.ngrid.sname}' does not match "
                    f"the NESTOUT sname='{self.nestout.sname}'"
                )
        return self

    @property
    def locations_set(self):
        """List of specified location fields."""
        return [fld for fld in self.model_fields_set if fld in self._location_fields]

    @property
    def write_set(self):
        """List of specified write fields."""
        return [fld for fld in self.model_fields_set if fld in self._write_fields]

    @property
    def snames(self):
        """List of snames from specified location components."""
        snames = []
        for field in self.locations_set:
            obj = getattr(self, field)
            if obj is None:
                continue
            sname = obj.sname
            if isinstance(sname, str):
                sname = [sname]
            snames.extend(sname)
        return snames

    def _filter_location(self, sname):
        """Filter the location component defined with the specified sname."""
        for field in self.locations_set:
            obj = getattr(self, field)
            if obj is None:
                continue
            obj_snames = obj.sname if isinstance(obj.sname, list) else [obj.sname]
            for obj_sname in obj_snames:
                if obj_sname == sname:
                    return obj
        raise ValueError(f"Location component with sname='{sname}' not found")

    def cmd(self) -> list:
        """Command file string for this component."""
        repr = []
        if self.frame is not None:
            repr += [f"{self.frame.cmd()}"]
        if self.group is not None:
            repr += [f"{self.group.cmd()}"]
        if self.curve is not None:
            # Component renders a list
            repr += self.curve.cmd()
        if self.ray is not None:
            repr += [f"{self.ray.cmd()}"]
        if self.isoline is not None:
            repr += [f"{self.isoline.cmd()}"]
        if self.points is not None:
            repr += [f"{self.points.cmd()}"]
        if self.ngrid is not None:
            repr += [f"{self.ngrid.cmd()}"]
        if self.quantity is not None:
            # Component renders a list
            repr += self.quantity.cmd()
        if self.output_options is not None:
            repr += [f"{self.output_options.cmd()}"]
        if self.block is not None:
            # Component may or may not render a list, handles both
            cmds = self.block.cmd()
            if not isinstance(cmds, list):
                cmds = [cmds]
            for cmd in cmds:
                repr += [f"{cmd}"]
        if self.table is not None:
            repr += [f"{self.table.cmd()}"]
        if self.specout is not None:
            repr += [f"{self.specout.cmd()}"]
        if self.nestout is not None:
            repr += [f"{self.nestout.cmd()}"]
        if self.test is not None:
            repr += [f"{self.test.cmd()}"]
        return repr


# =====================================================================================
# Lockup
# =====================================================================================
COMPUTE_TYPE = Annotated[
    Union[COMPUTE_STAT, COMPUTE_NONSTAT],
    Field(description="Compute components", discriminator="model_type"),
]


class LOCKUP(BaseComponent):
    """Lockup group component.

    .. code-block:: text

        COMPUTE ...
        HOTFILE ...
        COMPUTE ...
        HOTFILE ...
        ...
        STOP

    This is a group component to specify SWAN "Lockup" commands including multiple
    `COMPUTE` commands that may or may not be interleaved with `HOTFILE` commands,
    and a final `STOP` command.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.group import LOCKUP
        lockup = LOCKUP(
            compute=dict(
                model_type="stat",
                times=dict(
                    model_type="nonstationary",
                    tbeg="1990-01-01T00:00:00",
                    tend="1990-01-01T03:00:00",
                    delt="PT1H",
                    dfmt="hr",
                ),
                hotfile=dict(fname="hotfile"),
                hottimes=[-1],
            ),
        )
        print(lockup.render())

    """

    model_type: Literal["lockup", "LOCKUP"] = Field(
        default="lockup", description="Model type discriminator"
    )
    compute: COMPUTE_TYPE = Field(description="Compute components")

    def cmd(self) -> list:
        """Command file strings for this component."""
        return self.compute.cmd() + [STOP().render()]
