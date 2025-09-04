# This file was auto generated from a SCHISM namelist file on 2025-01-24.

from typing import List, Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Marco(NamelistBaseModel):
    idelay: Optional[int] = Field(
        0, description="Switch for 7-day delay in zooplankton predation (0: off, 1: on)"
    )
    ndelay: Optional[int] = Field(
        7,
        description="Number of days for zooplankton predation delay when idelay is active",
    )
    ibgraze: Optional[int] = Field(
        0, description="Switch for bottom grazing function (0: off, 1: on)"
    )
    idapt: Optional[int] = Field(
        0, description="Switch for light adaptation (0: off, 1: on)"
    )
    alpha_corr: Optional[float] = Field(
        1.25, description="Correction factor for light adaptation when idapt is active"
    )
    zeptic: Optional[float] = Field(
        10.0,
        description="Euphotic zone depth parameter for light adaptation when idapt is active",
    )
    iz2graze: Optional[int] = Field(
        1, description="Switch for Z2 grazing on S2, Z1, and DN (0: off, 1: on)"
    )
    iout_cosine: Optional[int] = Field(
        0,
        description="CoSiNE model station output option (0: off, 1-5: various output levels)",
    )
    nspool_cosine: Optional[int] = Field(
        30,
        description="Output interval (number of time steps) for CoSiNE model station output",
    )
    ico2s: Optional[int] = Field(
        0,
        description="Switch for CO2 limitation on phytoplankton growth (0: off, 1: on)",
    )
    ispm: Optional[int] = Field(
        0,
        description="Suspended Particulate Matter (SPM) calculation method (0: constant, 1: spatial varying, 2: SED model)",
    )
    spm0: Optional[float] = Field(
        20.0, description="Constant SPM value used when ispm is 0"
    )
    ised: Optional[int] = Field(
        1, description="Switch for sediment flux model (0: off, 1: on)"
    )

    @field_validator("idelay")
    @classmethod
    def validate_idelay(cls, v):
        if v not in [0, 1]:
            raise ValueError("idelay must be 0 or 1")
        return v

    @field_validator("ndelay")
    @classmethod
    def validate_ndelay(cls, v):
        if v < 0:
            raise ValueError("ndelay must be non-negative")
        return v

    @field_validator("ibgraze")
    @classmethod
    def validate_ibgraze(cls, v):
        if v not in [0, 1]:
            raise ValueError("ibgraze must be 0 or 1")
        return v

    @field_validator("idapt")
    @classmethod
    def validate_idapt(cls, v):
        if v not in [0, 1]:
            raise ValueError("idapt must be 0 or 1")
        return v

    @field_validator("alpha_corr")
    @classmethod
    def validate_alpha_corr(cls, v):
        if v <= 0:
            raise ValueError("alpha_corr must be positive")
        return v

    @field_validator("zeptic")
    @classmethod
    def validate_zeptic(cls, v):
        if v <= 0:
            raise ValueError("zeptic must be positive")
        return v

    @field_validator("iz2graze")
    @classmethod
    def validate_iz2graze(cls, v):
        if v not in [0, 1]:
            raise ValueError("iz2graze must be 0 or 1")
        return v

    @field_validator("iout_cosine")
    @classmethod
    def validate_iout_cosine(cls, v):
        if v not in range(6):
            raise ValueError("iout_cosine must be between 0 and 5")
        return v

    @field_validator("nspool_cosine")
    @classmethod
    def validate_nspool_cosine(cls, v):
        if v <= 0:
            raise ValueError("nspool_cosine must be positive")
        return v

    @field_validator("ico2s")
    @classmethod
    def validate_ico2s(cls, v):
        if v not in [0, 1]:
            raise ValueError("ico2s must be 0 or 1")
        return v

    @field_validator("ispm")
    @classmethod
    def validate_ispm(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("ispm must be 0, 1, or 2")
        return v

    @field_validator("spm0")
    @classmethod
    def validate_spm0(cls, v):
        if v < 0:
            raise ValueError("spm0 must be non-negative")
        return v

    @field_validator("ised")
    @classmethod
    def validate_ised(cls, v):
        if v not in [0, 1]:
            raise ValueError("ised must be 0 or 1")
        return v


class Core(NamelistBaseModel):
    gmaxs: Optional[list] = Field(
        [2.0, 2.5], description="Maximum growth rates for two phytoplankton types"
    )
    gammas: Optional[list] = Field(
        [0.2, 0.075], description="Mortality rates for two phytoplankton types"
    )
    pis: Optional[list] = Field(
        [1.5, 1.5],
        description="Ammonium inhibition parameters for two phytoplankton types",
    )
    kno3s: Optional[list] = Field(
        [1.0, 3.0],
        description="NO3 half-saturation constants for two phytoplankton types",
    )
    knh4s: Optional[list] = Field(
        [0.15, 0.45],
        description="NH4 half-saturation constants for two phytoplankton types",
    )
    kpo4s: Optional[list] = Field(
        [0.1, 0.1],
        description="PO4 half-saturation constants for two phytoplankton types",
    )
    kco2s: Optional[list] = Field(
        [50.0, 50.0],
        description="CO2 half-saturation constants for two phytoplankton types",
    )
    ksio4: Optional[float] = Field(
        4.5, description="SiO4 half-saturation constant for diatoms"
    )
    kns: Optional[list] = Field(
        [0.0, 0.0],
        description="Nighttime uptake rates of NH4 for two phytoplankton types",
    )
    alphas: Optional[list] = Field(
        [0.1, 0.1],
        description="Initial slopes of P-I curve for two phytoplankton types",
    )
    betas: Optional[list] = Field(
        [0.0, 0.0],
        description="Slopes for photo-inhibition for two phytoplankton types",
    )
    aks: Optional[list] = Field(
        [0.75, 0.03, 0.066],
        description="Light extinction coefficients for phytoplankton and suspended particulate matter",
    )
    betaz: Optional[list] = Field(
        [1.35, 0.4], description="Maximum grazing rates for two zooplankton types"
    )
    alphaz: Optional[list] = Field(
        [0.75, 0.75], description="Assimilation rates for two zooplankton types"
    )
    gammaz: Optional[list] = Field(
        [0.2, 0.2], description="Mortality rates for two zooplankton types"
    )
    kez: Optional[list] = Field(
        [0.2, 0.2], description="Excretion rates for two zooplankton types"
    )
    kgz: Optional[list] = Field(
        [0.5, 0.25],
        description="Reference prey concentrations for grazing for two zooplankton types",
    )
    rhoz: Optional[list] = Field(
        [0.6, 0.3, 0.1], description="Prey preference factors of Z2 on (S2, Z1, DN)"
    )
    ipo4: Optional[int] = Field(
        1, description="Flag to add additional PO4 from biogenic silica dissolution"
    )
    tr: Optional[float] = Field(
        20.0,
        description="Reference temperature for temperature adjustment for CoSiNE sink and source",
    )
    kox: Optional[float] = Field(
        30.0, description="Reference oxygen concentration for oxidation"
    )
    wss2: Optional[float] = Field(
        0.2, description="Settling velocity of S2 (phytoplankton type 2)"
    )
    wsdn: Optional[float] = Field(
        1.0, description="Settling velocity of DN (detrital nitrogen)"
    )
    wsdsi: Optional[float] = Field(
        1.0, description="Settling velocity of DSi (dissolved silica)"
    )
    si2n: Optional[float] = Field(
        1.2, description="Silica to nitrogen conversion coefficient"
    )
    p2n: Optional[float] = Field(
        0.0625, description="Phosphorus to nitrogen conversion coefficient"
    )
    o2no: Optional[float] = Field(
        8.625, description="Oxygen to nitrogen (NO3) conversion coefficient"
    )
    o2nh: Optional[float] = Field(
        6.625, description="Oxygen to nitrogen (NH4) conversion coefficient"
    )
    c2n: Optional[float] = Field(
        7.3, description="Carbon to nitrogen conversion coefficient"
    )
    gamman: Optional[float] = Field(0.07, description="Nitrification coefficient")
    pco2a: Optional[float] = Field(391.63, description="Atmospheric CO2 concentration")
    kmdn: Optional[list] = Field(
        [0.009, 0.075],
        description="Remineralization coefficients for DN (detrital nitrogen)",
    )
    kmdsi: Optional[list] = Field(
        [0.0114, 0.015],
        description="Remineralization coefficients for DSi (dissolved silica)",
    )

    @field_validator("gmaxs")
    @classmethod
    def check_gmaxs(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("gmaxs must be a list of 2 positive floats")
        return v

    @field_validator("gammas")
    @classmethod
    def check_gammas(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("gammas must be a list of 2 floats between 0 and 1")
        return v

    @field_validator("pis")
    @classmethod
    def check_pis(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("pis must be a list of 2 positive floats")
        return v

    @field_validator("kno3s")
    @classmethod
    def check_kno3s(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("kno3s must be a list of 2 positive floats")
        return v

    @field_validator("knh4s")
    @classmethod
    def check_knh4s(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("knh4s must be a list of 2 positive floats")
        return v

    @field_validator("kpo4s")
    @classmethod
    def check_kpo4s(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("kpo4s must be a list of 2 positive floats")
        return v

    @field_validator("kco2s")
    @classmethod
    def check_kco2s(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("kco2s must be a list of 2 positive floats")
        return v

    @field_validator("ksio4")
    @classmethod
    def check_ksio4(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("ksio4 must be a positive float")
        return v

    @field_validator("kns")
    @classmethod
    def check_kns(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x >= 0 for x in v)
        ):
            raise ValueError("kns must be a list of 2 non-negative floats")
        return v

    @field_validator("alphas")
    @classmethod
    def check_alphas(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("alphas must be a list of 2 positive floats")
        return v

    @field_validator("betas")
    @classmethod
    def check_betas(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x >= 0 for x in v)
        ):
            raise ValueError("betas must be a list of 2 non-negative floats")
        return v

    @field_validator("aks")
    @classmethod
    def check_aks(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 3
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("aks must be a list of 3 positive floats")
        return v

    @field_validator("betaz")
    @classmethod
    def check_betaz(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("betaz must be a list of 2 positive floats")
        return v

    @field_validator("alphaz")
    @classmethod
    def check_alphaz(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and 0 < x <= 1 for x in v)
        ):
            raise ValueError("alphaz must be a list of 2 floats between 0 and 1")
        return v

    @field_validator("gammaz")
    @classmethod
    def check_gammaz(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and 0 < x <= 1 for x in v)
        ):
            raise ValueError("gammaz must be a list of 2 floats between 0 and 1")
        return v

    @field_validator("kez")
    @classmethod
    def check_kez(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and 0 < x <= 1 for x in v)
        ):
            raise ValueError("kez must be a list of 2 floats between 0 and 1")
        return v

    @field_validator("kgz")
    @classmethod
    def check_kgz(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x > 0 for x in v)
        ):
            raise ValueError("kgz must be a list of 2 positive floats")
        return v

    @field_validator("rhoz")
    @classmethod
    def check_rhoz(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 3
            and all(isinstance(x, float) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("rhoz must be a list of 3 floats between 0 and 1")
        return v

    @field_validator("ipo4")
    @classmethod
    def check_ipo4(cls, v):
        if v not in [0, 1]:
            raise ValueError("ipo4 must be either 0 or 1")
        return v

    @field_validator("tr")
    @classmethod
    def check_tr(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("tr must be a positive float")
        return v

    @field_validator("kox")
    @classmethod
    def check_kox(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("kox must be a positive float")
        return v

    @field_validator("wss2")
    @classmethod
    def check_wss2(cls, v):
        if not (isinstance(v, float) and v >= 0):
            raise ValueError("wss2 must be a non-negative float")
        return v

    @field_validator("wsdn")
    @classmethod
    def check_wsdn(cls, v):
        if not (isinstance(v, float) and v >= 0):
            raise ValueError("wsdn must be a non-negative float")
        return v

    @field_validator("wsdsi")
    @classmethod
    def check_wsdsi(cls, v):
        if not (isinstance(v, float) and v >= 0):
            raise ValueError("wsdsi must be a non-negative float")
        return v

    @field_validator("si2n")
    @classmethod
    def check_si2n(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("si2n must be a positive float")
        return v

    @field_validator("p2n")
    @classmethod
    def check_p2n(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("p2n must be a positive float")
        return v

    @field_validator("o2no")
    @classmethod
    def check_o2no(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("o2no must be a positive float")
        return v

    @field_validator("o2nh")
    @classmethod
    def check_o2nh(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("o2nh must be a positive float")
        return v

    @field_validator("c2n")
    @classmethod
    def check_c2n(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("c2n must be a positive float")
        return v

    @field_validator("gamman")
    @classmethod
    def check_gamman(cls, v):
        if not (isinstance(v, float) and 0 <= v <= 1):
            raise ValueError("gamman must be a float between 0 and 1")
        return v

    @field_validator("pco2a")
    @classmethod
    def check_pco2a(cls, v):
        if not (isinstance(v, float) and v > 0):
            raise ValueError("pco2a must be a positive float")
        return v

    @field_validator("kmdn")
    @classmethod
    def check_kmdn(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x >= 0 for x in v)
        ):
            raise ValueError("kmdn must be a list of 2 non-negative floats")
        return v

    @field_validator("kmdsi")
    @classmethod
    def check_kmdsi(cls, v):
        if not (
            isinstance(v, list)
            and len(v) == 2
            and all(isinstance(x, float) and x >= 0 for x in v)
        ):
            raise ValueError("kmdsi must be a list of 2 non-negative floats")
        return v

    @model_validator(mode="after")
    def check_rhoz_sum(self):
        if sum(self.rhoz) != 1:
            raise ValueError("Sum of rhoz values must equal 1")
        return self


class Misc(NamelistBaseModel):
    iws: Optional[int] = Field(
        0,
        description="Flag to enable or disable diatom sinking velocity dependence on NO3 concentration",
    )
    no3c: Optional[float] = Field(
        2.0,
        description="Critical NO3 concentration (mmol/m3) for diatom sinking velocity calculation",
    )
    ws1: Optional[float] = Field(
        2.5,
        description="Diatom sinking velocity (m/day) when NO3 concentration is below no3c",
    )
    ws2: Optional[float] = Field(
        2.0,
        description="Diatom sinking velocity (m/day) when NO3 concentration is above no3c",
    )
    iclam: Optional[int] = Field(
        0, description="Flag to enable or disable clam grazing model"
    )
    deltaz: Optional[int] = Field(
        1, description="Vertical grid size (meter) for clam grazing model"
    )
    kcex: Optional[float] = Field(0.002, description="Clam excretion rate (day^-1)")
    nperclam: Optional[float] = Field(
        0.39032, description="Nitrogen content per clam (mmol[N])"
    )
    wclam: Optional[float] = Field(0.00545, description="Clam weight (g)")
    fclam: Optional[int] = Field(
        40, description="Clam filtration rate (L.g[AFDW]^-1.day^-1)"
    )
    nclam0: Optional[int] = Field(2000, description="Initial number of clams")
    fs2: Optional[list] = Field(
        [0.1, 0.1, 0.8],
        description="Partitioning coefficients for S2 from water column into sediment (3 values)",
    )
    rks2: Optional[list] = Field(
        [0.004, 0.0001, 0.0],
        description="Changing rates of remineralization for sediment S2 (3 values, day^-1)",
    )
    mks2: Optional[list] = Field(
        [0.1, 0.01, 0.0],
        description="Maximum remineralization rates for sediment S2 (3 values, day^-1)",
    )
    fdn: Optional[list] = Field(
        [0.15, 0.1, 0.75],
        description="Partitioning coefficients for DN from water column into sediment (3 values)",
    )
    rkdn: Optional[list] = Field(
        [0.004, 0.0001, 0.0],
        description="Changing rates of remineralization for sediment DN (3 values, day^-1)",
    )
    mkdn: Optional[list] = Field(
        [0.1, 0.01, 0.0],
        description="Maximum remineralization rates for sediment DN (3 values, day^-1)",
    )
    fdsi: Optional[list] = Field(
        [0.3, 0.3, 0.4],
        description="Partitioning coefficients for DSi from water column into sediment (3 values)",
    )
    rkdsi: Optional[list] = Field(
        [0.004, 0.0001, 0.0],
        description="Changing rates of remineralization for sediment DSi (3 values, day^-1)",
    )
    mkdsi: Optional[list] = Field(
        [0.1, 0.01, 0.0],
        description="Maximum remineralization rates for sediment DSi (3 values, day^-1)",
    )

    @field_validator("iws")
    @classmethod
    def validate_iws(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError("iws must be either 0 or 1")
        return v

    @field_validator("no3c")
    @classmethod
    def validate_no3c(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("no3c must be positive")
        return v

    @field_validator("ws1")
    @classmethod
    def validate_ws1(cls, v: float) -> float:
        if v < 0:
            raise ValueError("ws1 must be non-negative")
        return v

    @field_validator("ws2")
    @classmethod
    def validate_ws2(cls, v: float) -> float:
        if v < 0:
            raise ValueError("ws2 must be non-negative")
        return v

    @field_validator("iclam")
    @classmethod
    def validate_iclam(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError("iclam must be either 0 or 1")
        return v

    @field_validator("deltaz")
    @classmethod
    def validate_deltaz(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("deltaz must be positive")
        return v

    @field_validator("kcex")
    @classmethod
    def validate_kcex(cls, v: float) -> float:
        if v < 0 or v > 1:
            raise ValueError("kcex must be between 0 and 1")
        return v

    @field_validator("nperclam")
    @classmethod
    def validate_nperclam(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("nperclam must be positive")
        return v

    @field_validator("wclam")
    @classmethod
    def validate_wclam(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("wclam must be positive")
        return v

    @field_validator("fclam")
    @classmethod
    def validate_fclam(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("fclam must be positive")
        return v

    @field_validator("nclam0")
    @classmethod
    def validate_nclam0(cls, v: int) -> int:
        if v < 0:
            raise ValueError("nclam0 must be non-negative")
        return v

    @field_validator("fs2")
    @classmethod
    def validate_fs2(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(0 <= x <= 1 for x in v) or abs(sum(v) - 1) > 1e-6:
            raise ValueError(
                "fs2 must be a list of 3 values between 0 and 1, summing to 1"
            )
        return v

    @field_validator("rks2")
    @classmethod
    def validate_rks2(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(x >= 0 for x in v):
            raise ValueError("rks2 must be a list of 3 non-negative values")
        return v

    @field_validator("mks2")
    @classmethod
    def validate_mks2(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(x >= 0 for x in v):
            raise ValueError("mks2 must be a list of 3 non-negative values")
        return v

    @field_validator("fdn")
    @classmethod
    def validate_fdn(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(0 <= x <= 1 for x in v) or abs(sum(v) - 1) > 1e-6:
            raise ValueError(
                "fdn must be a list of 3 values between 0 and 1, summing to 1"
            )
        return v

    @field_validator("rkdn")
    @classmethod
    def validate_rkdn(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(x >= 0 for x in v):
            raise ValueError("rkdn must be a list of 3 non-negative values")
        return v

    @field_validator("mkdn")
    @classmethod
    def validate_mkdn(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(x >= 0 for x in v):
            raise ValueError("mkdn must be a list of 3 non-negative values")
        return v

    @field_validator("fdsi")
    @classmethod
    def validate_fdsi(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(0 <= x <= 1 for x in v) or abs(sum(v) - 1) > 1e-6:
            raise ValueError(
                "fdsi must be a list of 3 values between 0 and 1, summing to 1"
            )
        return v

    @field_validator("rkdsi")
    @classmethod
    def validate_rkdsi(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(x >= 0 for x in v):
            raise ValueError("rkdsi must be a list of 3 non-negative values")
        return v

    @field_validator("mkdsi")
    @classmethod
    def validate_mkdsi(cls, v: List[float]) -> List[float]:
        if len(v) != 3 or not all(x >= 0 for x in v):
            raise ValueError("mkdsi must be a list of 3 non-negative values")
        return v


class Cosine(NamelistBaseModel):
    marco: Optional[Marco] = Field(default_factory=Marco)
    core: Optional[Core] = Field(default_factory=Core)
    misc: Optional[Misc] = Field(default_factory=Misc)
