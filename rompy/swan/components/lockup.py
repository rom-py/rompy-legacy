"""
SWAN Lockup Components

This module contains components for controlling SWAN model execution flow,
including computation, hotfile output, and program termination.
"""

# Standard library imports
from pathlib import Path
from typing import Literal, Optional, Union
from datetime import datetime

# Third-party imports
from pydantic import field_validator, model_validator, Field
from pandas import Timestamp
from numpy import inf

# Local imports
from rompy.core.logging import get_logger
from rompy.swan.components.base import BaseComponent
from rompy.swan.subcomponents.time import STATIONARY, NONSTATIONARY

# Initialize the logger
logger = get_logger(__name__)

TIMES_TYPE = Union[STATIONARY, NONSTATIONARY]
HOTTIMES_TYPE = Union[list[datetime], list[int]]


class COMPUTE(BaseComponent):
    """Start SWAN computation.

    .. code-block:: text

        COMPUTE STATIONARY|NONSTATIONARY

    If the SWAN mode is stationary (see command `MODE`), then only the command
    `COMPUTE` should be given here.

    If the SWAN mode is nonstationary (see command `MODE`), then the computation can
    be:

    * stationary (at the specified time: option STATIONARY here).
    * nonstationary (over the specified period of time.

    To verify input to SWAN (e.g., all input fields such as water depth, wind fields,
    etc), SWAN can be run without computations (that is: zero iterations by using
    command `NUM ACCUR MXITST=0`).

    In the case `MODE NONSTATIONARY` several commands COMPUTE can appear, where the
    wave state at the end of one computation is used as initial state for the next one,
    unless a command `INIT` appears in between the two COMPUTE commands. This enables
    the user to make a stationary computation to obtain the initial state for a
    nonstationary computation and/or to change the computational time step during a
    computation, to change a boundary condition etc. This also has the advantage of not
    using a hotfile since, it can be very large in size.

    For small domains, i.e. less than 100 km or 1 deg, a stationary computation is
    recommended. Otherwise, a nonstationary computation is advised.

    For a nonstationary computation, a time step of at most 10 minutes is advised (when
    you are choosing a time step larger than 10 minutes, the action density limiter
    (see command `NUM`) becomes probably a part of the physics).

    Also, the time step should be chosen such that the Courant number is smaller than
    10 for the fastest (or dominant) wave. Otherwise, a first order upwind scheme is
    recommended in that case; see command `PROP BSBT`. If you want to run a high
    resolution model with a very large time step, e.g. 1 hour, you may apply multiple
    COMPUT STAT commands. For a small time step (<= 10 minutes), no more than 1
    iteration per time step is recommended (see command `NUM ... NONSTAT mxitns`).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.lockup import COMPUTE
        comp = COMPUTE()
        print(comp.render())
        comp = COMPUTE(
            times=dict(model_type="stationary", time="1990-01-01T00:00:00", tfmt=2)
        )
        print(comp.render())
        comp = COMPUTE(
            times=dict(
                model_type="nonstationary",
                tbeg="1990-01-01T00:00:00",
                tend="1990-02-01T00:00:00",
                delt="PT1H",
                tfmt=1,
                dfmt="hr",
            ),
        )
        print(comp.render())

    """

    model_type: Literal["compute", "COMPUTE"] = Field(
        default="compute", description="Model type discriminator"
    )
    times: Optional[TIMES_TYPE] = Field(
        default=None,
        description="Times for the stationary or nonstationary computation",
        discriminator="model_type",
    )
    i0: Optional[int] = Field(
        default=None,
        description="Time index of the initial time step",
    )
    i1: Optional[int] = Field(
        default=None,
        description="Time index of the final time step",
    )

    @field_validator("times")
    @classmethod
    def times_suffix(cls, times: TIMES_TYPE) -> TIMES_TYPE:
        if isinstance(times, NONSTATIONARY):
            times.suffix = "c"
        return times

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = "COMPUTE"
        if self.times is not None:
            repr += f" {self.times.render()}"
        return repr


class HOTFILE(BaseComponent):
    """Write intermediate results.

    .. code-block:: text

        HOTFILE 'fname' ->FREE|UNFORMATTED

    This command can be used to write the entire wave field at the end of a computation
    to a so-called hotfile, to be used as initial condition in a subsequent SWAN run
    (see command `INITIAL HOTSTART`). This command must be entered immediately after a
    `COMPUTE` command.

    The user may choose the format of the hotfile to be written either as free or
    unformatted. If the free format is chosen, then this format is identical to the
    format of the files written by the `SPECOUT` command (option `SPEC2D`). This
    hotfile is therefore an ASCII file which is human readable.

    An unformatted (or binary) file usually requires less space on your computer than
    an ASCII file. Moreover, it can be readed by a subsequent SWAN run much faster than
    an ASCII file. Especially, when the hotfile might become a big file, the choice for
    unformatted is preferable. Note that your compiler and OS should follow the same
    ABI (Application Binary Interface) conventions (e.g. word size, endianness), so
    that unformatted hotfiles may transfer properly between different OS or platforms.
    This implies that the present and subsequent SWAN runs do not have to be carried
    out on the same operating system (e.g. Windows, Linux) or on the same computer,
    provided that distinct ABI conventions have been followed.

    Note
    ----
    For parallel MPI runs, more than one hotfile will be generated depending on the
    number of processors (`fname-001`, `fname-002`, etc).

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.lockup import HOTFILE
        hotfile = HOTFILE(fname="hotfile.swn")
        print(hotfile.render())
        hotfile = HOTFILE(fname="hotfile.dat", format="unformatted")
        print(hotfile.render())

    """

    model_type: Literal["hotfile", "HOTFILE"] = Field(
        default="hotfile", description="Model type discriminator"
    )
    fname: Path = Field(
        description="Name of the file to which the wave field is written",
    )
    format: Optional[Literal["free", "unformatted"]] = Field(
        default=None,
        description=("Choose between free (SWAN ASCII) or unformatted (binary) format"),
    )

    @field_validator("fname")
    @classmethod
    def max_length(cls, fname: Path) -> Path:
        if len(str(fname)) > 36:
            raise ValueError("fname must be less than 36 characters")
        return fname

    def cmd(self) -> str:
        """Command file string for this component."""
        repr = f"HOTFILE fname='{self.fname}'"
        if self.format is not None:
            repr += f" {self.format.upper()}"
        return repr


class COMPUTE_STAT(BaseComponent):
    """Multiple SWAN stationary computations.

    .. code-block:: text

        COMPUTE STATIONARY [time]
        HOTFILE 'fname' ->FREE|UNFORMATTED
        COMPUTE STATIONARY [time]
        COMPUTE STATIONARY [time]
        HOTFILE 'fname' ->FREE|UNFORMATTED
        .
        .

    This component can be used to define multiple stationary compute commands and
    write intermediate results as hotfiles between then.

    Note
    ----
    The field `times` is optional to allow for the case where the user wants to set
    times dynamically after instantiating this component.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import STATIONARY, NONSTATIONARY
        from rompy.swan.components.lockup import COMPUTE_STAT
        time = STATIONARY(time="1990-01-01T00:00:00")
        comp = COMPUTE_STAT(times=time)
        print(comp.render())
        times = NONSTATIONARY(
            tbeg="1990-01-01T00:00:00",
            tend="1990-01-01T03:00:00",
            delt="PT1H",
        )
        comp = COMPUTE_STAT(times=times)
        print(comp.render())
        hotfile = dict(fname="./hotfile.swn")
        hottimes=["1990-01-01T03:00:00"]
        comp = COMPUTE_STAT(times=times, hotfile=hotfile, hottimes=hottimes)
        print(comp.render())
        comp = COMPUTE_STAT(times=times, hotfile=hotfile, hottimes=[2, -1])
        print(comp.render())

    """

    model_type: Literal["stat", "STAT"] = Field(
        default="stat", description="Model type discriminator"
    )
    times: TIMES_TYPE = Field(
        default_factory=STATIONARY,
        description="Compute times",
        discriminator="model_type",
    )
    hotfile: Optional[HOTFILE] = Field(
        default=None,
        description="Write results to restart files",
    )
    hottimes: HOTTIMES_TYPE = Field(
        default=[],
        description=(
            "Times to write hotfiles, can be a list of datetimes or times indices"
        ),
    )
    suffix: str = Field(
        default="_%Y%m%dT%H%M%S",
        description=("Time-template suffix to add to hotfile fname"),
    )

    @field_validator("hottimes")
    @classmethod
    def timestamp_to_datetime(cls, hottimes: TIMES_TYPE) -> TIMES_TYPE:
        """Ensure pandas.Timestamp entries are coerced into datatime."""
        if hottimes and isinstance(hottimes[0], Timestamp):
            hottimes = [t.to_pydatetime() for t in hottimes]
        return hottimes

    @model_validator(mode="after")
    def hotfile_with_hottimes(self) -> "COMPUTE_NONSTAT":
        if self.hottimes and self.hotfile is None:
            logger.warning("hotfile not specified, hottimes will be ignored")
        elif self.hotfile is not None and not self.hottimes:
            logger.warning("hottimes not specified, hotfile will be ignored")
        return self

    @property
    def hotids(self) -> list:
        """List time ids at which to write hotfiles."""
        if self.hottimes and isinstance(self.hottimes[0], datetime):
            ids = []
            for t in self.hottimes:
                try:
                    ids.append(self.times().index(t))
                except ValueError as e:
                    raise ValueError(f"hottime {t} not in times {self.times}") from e
        else:
            ids = [i if i >= 0 else i + len(self.times) for i in self.hottimes]
            for i in ids:
                if i >= len(self.times):
                    raise ValueError(
                        f"Hotfile requested for time {i} but times have "
                        f"only {len(self.times)} values: {self.times} "
                    )
        return ids

    def _hotfile(self, time):
        """Set timestamp to hotfile fname."""
        timestamp = time.strftime(self.suffix)
        fname = self.hotfile.fname.parent / (
            f"{self.hotfile.fname.stem}{timestamp}" f"{self.hotfile.fname.suffix}"
        )
        return HOTFILE(fname=fname, format=self.hotfile.format)

    def cmd(self) -> list:
        """Command file string for this component."""
        repr = []
        for ind, time in enumerate(self.times()):
            repr += [f"COMPUTE {STATIONARY(time=time, tfmt=self.times.tfmt).render()}"]
            if ind in self.hotids and self.hotfile is not None:
                repr += [f"{self._hotfile(time).render()}"]
        return repr


class COMPUTE_NONSTAT(COMPUTE_STAT):
    """Multiple SWAN nonstationary computations.

    .. code-block:: text

        COMPUTE NONSTATIONARY [tbegc] [deltc] SEC|MIN|HR|DAY [tendc]
        HOTFILE 'fname' ->FREE|UNFORMATTED
        COMPUTE NONSTATIONARY [tbegc] [deltc] SEC|MIN|HR|DAY [tendc]
        HOTFILE 'fname' ->FREE|UNFORMATTED
        .
        .

    This component can be used to define multiple nonstationary compute commands and
    write intermediate results as hotfiles between then.

    Note
    ----
    The field `times` is optional to allow for the case where the user wants to set
    times dynamically after instantiating this component.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.subcomponents.time import NONSTATIONARY
        from rompy.swan.components.lockup import COMPUTE_NONSTAT
        times = NONSTATIONARY(
            tbeg="1990-01-01T00:00:00",
            tend="1990-02-01T00:00:00",
            delt="PT1H",
            dfmt="hr",
        )
        comp = COMPUTE_NONSTAT(times=times)
        print(comp.render())
        comp = COMPUTE_NONSTAT(
            times=times,
            hotfile=dict(fname="hotfile.swn", format="free"),
            hottimes=["1990-02-01T00:00:00"],
        )
        print(comp.render())
        comp = COMPUTE_NONSTAT(
            times=times,
            initstat=True,
            hotfile=dict(fname="hotfile", format="free"),
            hottimes=[6, 12, 18, -1],
        )
        print(comp.render())

    """

    model_type: Literal["nonstat", "NONSTAT"] = Field(
        default="nonstat", description="Model type discriminator"
    )
    times: NONSTATIONARY = Field(
        default_factory=NONSTATIONARY, description="Compute times"
    )
    initstat: bool = Field(
        default=False,
        description=(
            "Run a STATIONARY computation at the initial time prior to the "
            "NONSTATIONARY computation(s) to prescribe initial conditions"
        ),
    )

    @field_validator("times")
    @classmethod
    def times_suffix(cls, times: NONSTATIONARY) -> NONSTATIONARY:
        times.suffix = "c"
        return times

    def _times(self, tbeg, tend):
        return NONSTATIONARY(
            tbeg=tbeg,
            tend=tend,
            delt=self.times.delt,
            tfmt=self.times.tfmt,
            dfmt=self.times.dfmt,
            suffix=self.times.suffix,
        )

    def cmd(self) -> list:
        """Command file string for this component."""
        repr = []
        ind = -inf
        tbeg = self.times.tbeg
        if self.initstat:
            repr += [f"COMPUTE {STATIONARY(time=tbeg, tfmt=self.times.tfmt).render()}"]
        for ind in self.hotids:
            tend = self.times()[ind]
            times = self._times(tbeg, tend)
            repr += [f"COMPUTE {times.render()}"]
            if self.hotfile is not None:
                repr += [f"{self._hotfile(tend).render()}"]
            tbeg = tend
        if ind < len(self.times) - 1:
            times = self._times(tbeg, self.times.tend)
            repr += [f"COMPUTE {times.render()}"]
        return repr


class STOP(BaseComponent):
    """End of commands.

    .. code-block:: text

        STOP

    This required command marks the end of the commands in the command file. Note that
    the command `STOP` may be the last command in the input file; any information in
    the input file beyond this command is ignored.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.swan.components.lockup import STOP
        stop = STOP()
        print(stop.render())

    """

    model_type: Literal["stop", "STOP"] = Field(
        default="stop", description="Model type discriminator"
    )

    def cmd(self) -> str:
        """Command file string for this component."""
        return "STOP"
