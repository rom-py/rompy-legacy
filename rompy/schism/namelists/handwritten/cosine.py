from pydantic import Field

from rompy.core.types import RompyBaseModel


class MARCO(RompyBaseModel):
    idelay: int = Field(0, description="A 7-day delay for zooplankton predation.")
    ndelay: int = Field(7, description="The number of days for delay.")
    ibgraze: int = Field(0, description="Bottom grazing function switch.")
    idapt: int = Field(0, description="Light adaptation switch.")
    alpha_corr: float = Field(1.25, description="Correction factor for alpha.")
    zeptic: float = Field(10.0, description="Value for zeptic.")
    iz2graze: int = Field(1, description="Z2 grazing switch.")
    iout_cosine: int = Field(
        0, description="CoSiNE model station output option (0 to turn off)."
    )
    nspool_cosine: int = Field(30, description="Output interval for CoSiNE model.")
    ico2s: int = Field(0, description="CO2 limitation on phytoplankton growth switch.")
    ispm: int = Field(0, description="Suspended Particulate Matter (SPM) switch.")
    spm0: float = Field(20.0, description="Value for spm0.")
    ised: int = Field(1, description="Sediment flux model switch.")


class CORE(RompyBaseModel):
    gmaxs: float = Field(2.0, description="Maximum growth rate for phytoplankton.")
    gammas: float = Field(0.2, description="Mortality rate for phytoplankton.")
    pis: float = Field(1.5, description="Ammonium inhibition for phytoplankton.")
    kno3s: float = Field(1.0, description="NO3 half saturation for phytoplankton.")
    knh4s: float = Field(0.15, description="NH4 half saturation for phytoplankton.")
    kpo4s: float = Field(0.1, description="PO4 half saturation for phytoplankton.")
    kco2s: float = Field(50.0, description="CO2 half saturation for phytoplankton.")
    ksio4: float = Field(4.5, description="SiO4 half saturation for diatom.")
    kns: float = Field(
        0.0, description="Nighttime uptake rate of NH4 for phytoplankton."
    )
    alphas: float = Field(
        0.1, description="Initial slopes of P-I curve for phytoplankton."
    )
    betas: float = Field(
        0.0, description="Slope for photo-inhibition for phytoplankton."
    )
    aks: list[float] = Field(
        [0.75, 0.03, 0.066],
        description="Light extinction coefficients for phytoplankton.",
    )
    betaz: float = Field(1.35, description="Maximum grazing rate for zooplankton.")
    alphaz: float = Field(0.75, description="Assimilation rate for zooplankton.")
    gammaz: float = Field(0.2, description="Mortality rate for zooplankton.")
    kez: float = Field(0.2, description="Excretion rate for zooplankton.")
    kgz: float = Field(
        0.5, description="Reference prey concentration for grazing for zooplankton."
    )
    rhoz: list[float] = Field(
        [0.6, 0.3, 0.1], description="Prey preference factors for zooplankton."
    )
    ipo4: int = Field(1, description="Additional PO4 switch.")
    TR: float = Field(20.0, description="Reference temperature for temperature adjust.")
    kox: float = Field(
        30.0, description="Reference oxygen concentration for oxidation."
    )
    wss2: float = Field(0.2, description="Settling velocity of S2.")
    wsdn: float = Field(1.0, description="Settling velocity of DN.")
    wsdsi: float = Field(1.0, description="Settling velocity of DSi.")
    si2n: float = Field(1.2, description="Silica to nitrogen conversion coefficient.")
    p2n: float = Field(
        0.0625, description="Phosphorus to nitrogen conversion coefficient."
    )
    o2no: float = Field(
        8.625, description="Oxygen to nitrogen (NO3) conversion coefficient."
    )
    o2nh: float = Field(
        6.625, description="Oxygen to nitrogen (NH4) conversion coefficient."
    )
    c2n: float = Field(7.3, description="Carbon to nitrogen conversion coefficient.")
    gamman: float = Field(0.07, description="Nitrification coefficient.")
    pco2a: float = Field(391.63, description="Atmospheric CO2 concentration.")
    kmdn: list[float] = Field(
        [0.009, 0.075], description="Remineralization coefficients for DN."
    )
    kmdsi: list[float] = Field(
        [0.0114, 0.015], description="Remineralization coefficients for DSi."
    )


class MISC(RompyBaseModel):
    iws: int = Field(0, description="Diatom sinking velocity switch.")
    NO3c: float = Field(
        2.0, description="NO3 concentration for diatom sinking velocity."
    )
    ws1: float = Field(2.5, description="Diatom sinking velocity for ws1.")
    ws2: float = Field(2.0, description="Diatom sinking velocity for ws2.")
    iclam: int = Field(0, description="Clam grazing model switch.")
    deltaZ: int = Field(1, description="Meter value for clam grazing model.")
    kcex: float = Field(0.002, description="Clam grazing rate.")
    Nperclam: float = Field(0.39032, description="Clam grazing N concentration.")
    Wclam: float = Field(5.45e-3, description="Clam weight.")
    Fclam: int = Field(40, description="Clam filtration rate.")
    nclam0: int = Field(2000, description="Initial clam population.")
    fS2: list[float] = Field(
        [0.1, 0.1, 0.8],
        description="Partitioning coefficient from S2 in water column into sediment S2.",
    )
    rkS2: list[float] = Field(
        [4e-3, 1.0e-4, 0.0],
        description="Changing rate of remineralization rate for sediment S2.",
    )
    mkS2: list[float] = Field(
        [0.1, 0.01, 0.0], description="Maximum remineralization rate for sediment S2."
    )
    fDN: list[float] = Field(
        [0.15, 0.10, 0.75],
        description="Partitioning coefficient from DN in water column into sediment DN.",
    )
    rkDN: list[float] = Field(
        [4e-3, 1.0e-4, 0.0],
        description="Changing rate of remineralization rate for sediment DN.",
    )
    mkDN: list[float] = Field(
        [0.1, 0.01, 0.0], description="Maximum remineralization rate for sediment DN."
    )
    fDSi: list[float] = Field(
        [0.3, 0.3, 0.4],
        description="Partitioning coefficient from DSi in water column into sediment DSi.",
    )
    rkDSi: list[float] = Field(
        [0.004, 1e-4, 0.0],
        description="Changing rate of remineralization rate for sediment DSi.",
    )
    mkDSi: list[float] = Field(
        [0.1, 0.01, 0.0], description="Maximum remineralization rate for sediment DSi."
    )


class COSINE(RompyBaseModel):
    marco: MARCO = Field(description="MARCO parameters")
    core: CORE = Field(description="CORE parameters")
    misc: MISC = Field(description="MISC parameters")
