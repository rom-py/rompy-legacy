import logging
from typing import Optional, Literal
import numpy as np
from pydantic import Field, model_validator
from pydantic_numpy.typing import Np1DArray

from rompy.core.types import RompyBaseModel


logger = logging.getLogger(__name__)


class Frequency(RompyBaseModel):
    """Wave frequency."""

    model_type: Literal["frequency", "FREQUENCY"] = Field(
        default="frequency", description="Model type discriminator"
    )
    freq: Np1DArray = Field(description="Frequency array")

    @property
    def f0(self):
        return self.freq.min()

    @property
    def f1(self):
        return self.freq.max()

    @property
    def nf(self):
        return self.freq.size

    @property
    def flen(self):
        return self.f1 - self.f0


class LogFrequency(RompyBaseModel):
    """Logarithmic wave frequencies.

    Frequencies are defined according to:

    :math:`f_{i+1} = \gamma * f_{i}`

    Note
    ----
    The number of frequency bins `nbin` is always kept unchanged when provided. This
    implies other parameters may be adjusted so `nbin` bins can be defined. Specify
    `f0`, `f1` and `finc` and let `nbin` be calculated to avoid those values changing.

    Note
    ----
    Choose `finc=0.1` for a 10% increment between frequencies that satisfies the DIA.

    Examples
    --------

    .. ipython:: python
        :okwarning:

        from rompy.core.spectrum import LogFrequency

        LogFrequency(f0=0.04, f1=1.0, nbin=34)
        LogFrequency(f0=0.04, f1=1.0, finc=0.1)
        LogFrequency(f0=0.04, nbin=34, finc=0.1)
        LogFrequency(f1=1.0, nbin=34, finc=0.1)

    """

    model_type: Literal["log", "LOG"] = Field(
        default="log", description="Model type discriminator"
    )
    f0: Optional[float] = Field(
        default=None, description="Lower frequency boundary (Hz)", gt=0.0
    )
    f1: Optional[float] = Field(
        default=None, description="Upper frequency boundary (Hz)"
    )
    finc: Optional[float] = Field(
        default=None, description="Log frequency increment", gt=0.0
    )
    nbin: Optional[int] = Field(
        default=None,
        description="Number of frequency bins, one less the size of frequency array",
        gt=0,
    )

    @model_validator(mode="after")
    def init_options(self) -> "LogFrequency":
        """Set the missing frequency parameters."""
        if sum([v is not None for v in [self.f0, self.f1, self.finc, self.nbin]]) != 3:
            raise ValueError("Three (only) of (f0, f1, finc, nbin) must be provided")

        # Calculate the missing frequency parameters
        if self.finc is None:
            self.finc = self._finc()
        elif self.nbin is None:
            self.nbin = self._nbin(self.f0, self.f1, self.finc)
        elif self.f1 is None:
            self.f1 = self.f0 * self.gamma**self.nbin
        else:
            self.f0 = self._f0(self.f1, self.nbin, self.gamma)

        # Redefine parameters based on the calculated values
        self.f0 = self()[0]
        self.f1 = self()[-1]
        self.finc = self._finc()
        self.nbin = len(self()) - 1

        return self

    def __call__(self) -> Np1DArray:
        """Frequency array."""
        return np.geomspace(self.f0, self.f1, self.nf)

    def __getitem__(self, index) -> float | list[float]:
        """Slicing from the frequency array."""
        return self.__call__()[index]

    def __len__(self):
        """Returns the length of the frequency array."""
        return len(self())

    def _finc(self):
        return (self()[1] - self()[0]) / self()[0]

    def _nbin(self, f0, f1, finc):
        return np.round(np.log(f1 / f0) / np.log(1 + finc)).astype("int")

    def _f0(self, f1, nbin, gamma):
        """Returns f0 given f1, nbin and gamma."""
        freqs = [f1]
        for n in range(nbin):
            freqs.append(freqs[-1] / gamma)
        return freqs[-1]

    @property
    def nf(self):
        return self.nbin + 1

    @property
    def gamma(self):
        return self.finc + 1

    @property
    def flen(self):
        return self.f1 - self.f0
