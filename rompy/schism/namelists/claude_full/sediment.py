# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import List, Literal, Optional

from pydantic import (
    Field,
    field_validator,
    ConfigDict,
)

from rompy.schism.namelists.basemodel import NamelistBaseModel


class SedCore(NamelistBaseModel):
    """
    Core parameters for non-cohesive sediment model.
    """

    sd50: List[float] = Field(
        ..., description="D50 MEDIAN SEDIMENT GRAIN DIAMETER (mm) - [1:Ntracers]"
    )
    erate: List[float] = Field(
        ...,
        description="SURFACE EROSION RATE, E0 - [1:Ntracers]. If ierosion=0, dimension is kg/m/m/s. If ierosion=1, dimension is s/m",
    )

    @field_validator("sd50", "erate")
    def validate_list_length(cls, v):
        if len(v) != 5:
            raise ValueError("Must have 5 values")
        return v


class SedOpt(NamelistBaseModel):
    """
    Optional parameters for non-cohesive sediment model.
    """

    ised_type: List[int] = Field(
        [1, 1, 1, 1, 1],
        description="SEDIMENT TYPE - [1:Ntracers]. 0 = MUD-like, 1 = SAND-like, 2 = GRAVEL-like",
    )
    srho: List[float] = Field(
        [2650.0, 2650.0, 2650.0, 2650.0, 2650.0],
        description="SEDIMENT GRAIN DENSITY (kg/m3) - [1:Ntracers]",
    )
    comp_ws: Literal[0, 1] = Field(
        0,
        description="COMPUTATION OF SEDIMENT SETTLING VELOCITY. 0 = Disabled, 1 = Enabled",
    )
    comp_tauce: Literal[0, 1] = Field(
        0,
        description="COMPUTATION OF SEDIMENT CRITICAL SHEAR STRESS. 0 = Disabled, 1 = Enabled",
    )
    wsed: List[float] = Field(
        [1.06, 3.92, 5.43, 10.19, 28.65],
        description="PATICLES SETTLING VELOCITY (mm/s) - [1:Ntracers]",
    )
    tau_ce: List[float] = Field(
        [0.15, 0.17, 0.23, 0.3, 0.6],
        description="CRITICAL SHEAR STRESS FOR EROSION (Pa) - [1:Ntracers]",
    )
    sed_debug: Literal[0, 1] = Field(
        0, description="DEBUG. 0 = silent, 1 = output lots of variables"
    )
    ised_dump: Literal[0, 1] = Field(
        0, description="Dumping/dredging option. 0 = no, 1 = needs input sed_dump.in"
    )
    bedload: Literal[0, 1, 2, 3, 4] = Field(
        1,
        description="BEDLOAD. 0 = Disabled, 1 = van rijn (2007), 2 = Meyer-Peter and Mueller (1948), 3 = Soulsby and Damgaard (2005), 4 = Wu and Lin (2014)",
    )
    bedload_filter: Literal[0, 1] = Field(
        0, description="FILTER for bedload fluxes. 0 = Disabled, 1 = Enabled"
    )
    bedload_limiter: Literal[0, 1] = Field(
        0, description="LIMITER for bedload fluxes. 0 = Disabled, 1 = Enabled"
    )
    suspended_load: Literal[0, 1] = Field(
        1, description="SUSPENDED LOAD. 0 = Disabled, 1 = Enabled"
    )
    iasym: Literal[0, 1] = Field(
        0, description="WAVE ASYMMETRY. 0 = Disabled, 1 = Enabled"
    )
    w_asym_max: float = Field(
        0.4, description="Maximum asymmetry coefficient considered for waves"
    )
    elfrink_filter: Literal[0, 1] = Field(
        0, description="Elfrink filter. 0 = Disabled, 1 = Enabled"
    )
    ech_uorb: int = Field(
        200,
        description="Number of bins considered to reconstitute orbital velocity temporal series along a wave period",
    )
    bedload_acc: Literal[0, 1, 2] = Field(
        0,
        description="Methods to compute bedload transport caused by acceleration-skewness Qacc",
    )
    bedload_acc_filter: Literal[0, 1] = Field(
        0, description="Bedload acceleration filter. 0 = Disabled, 1 = Enabled"
    )
    kacc_hoe: float = Field(
        1.4e-4, description="Constant [m.s] used in Hoefel and Elgar formulation"
    )
    kacc_dub: float = Field(
        0.631e-4, description="Constant [m.s] used in Dubarbier et al. formulation"
    )
    thresh_acc_opt: Literal[0, 1, 2] = Field(
        2,
        description="Method to compute the critical mobility parameter for acceleration-skewness",
    )
    acrit: float = Field(0.2, description="Critical acceleration [m.s-2] for Qacc")
    tau_option: Literal[1, 2, 3] = Field(
        1,
        description="Controls at which height above the bed the current-induced bottom shear stress is derived",
    )
    tau_max: float = Field(
        10.0,
        description="Maximum shear stress value authorised for current and waves (in Pa)",
    )
    zstress: float = Field(
        0.2, description="Height above bed for shear stress calculation (m)"
    )
    ierosion: Literal[0, 1] = Field(
        0,
        description="Erosional formulations. 0 = Ariathurai & Arulanandan (1978), 1 = Winterwerp et al. (2012)",
    )
    slope_formulation: Literal[0, 1, 2, 3, 4] = Field(
        4, description="SLOPE FORMULATION"
    )
    alpha_bs: float = Field(1.0, description="Coefficient for longitudinal slopes")
    alpha_bn: float = Field(1.5, description="Coefficient for transversal slopes")
    ised_bc_bot: Literal[1, 2] = Field(
        1,
        description="BOTTOM BOUNDARY CONDITION OPTION. 1 = Warner (2008), 2 = Tsinghua Univ group",
    )
    alphd: float = Field(1.00, description="SED DEPOSIT CORRECTION COEFFICIENT")
    refht: float = Field(0.75, description="REFERENCE HEIGHT for pick-up flux")
    tbp: float = Field(100.0, description="BURSTING PERIOD")
    im_pick_up: Literal[0, 1, 2, 3, 4] = Field(4, description="BOTTOM PICK-UP OPTION")
    sed_morph: Literal[0, 1, 2] = Field(0, description="MORPHOLOGY")
    sed_morph_time: float = Field(
        1.0, description="Time after which active morphology is turned on (days)"
    )
    morph_fac: float = Field(1.0, description="Morphological time-scale factor")
    drag_formulation: Literal[1, 2, 3] = Field(1, description="DRAG FORMULATION")
    ddensed: Literal[0, 1] = Field(0, description="SEDIMENT DENSITY IN STATE EQUATION")
    bedforms_rough: Literal[0, 1, 2] = Field(
        0, description="ROUGHNESS LENGTH PREDICTION FROM BEDFORMS"
    )
    iwave_ripple: Literal[0, 1] = Field(
        1, description="Wave ripples computation method"
    )
    irough_bdld: Literal[0, 1] = Field(
        1, description="Roughness induced by sediment transport"
    )
    slope_avalanching: Literal[0, 1] = Field(
        1, description="SLUMPING OF SEDIMENTS (AVALANCHING)"
    )
    dry_slope_cr: float = Field(1.0, description="Critical slope for dry element")
    wet_slope_cr: float = Field(0.3, description="Critical slope for wet element")
    bedmass_filter: Literal[0, 1, 2] = Field(0, description="BED MASS FILTER")
    bedmass_threshold: float = Field(
        0.025, description="Threshold value for D50 instabilities [mm]"
    )
    bdldiffu: float = Field(0.5, description="BEDLOAD DIFFUSION COEFFICIENT")
    bedload_coeff: float = Field(1.0, description="BEDLOAD TRANSPORT RATE COEFFICIENT")
    cdb_min: float = Field(
        1e-6, description="MINIMUM THRESHOLD FOR bottom drag coefficient"
    )
    cdb_max: float = Field(
        0.01, description="MAXIMUM THRESHOLD FOR bottom drag coefficient"
    )
    actv_max: float = Field(
        0.05,
        description="ACTIVE LAYER THICKNESS: Maximum thickness of sediment remobilised during one time step (m)",
    )
    nbed: int = Field(1, description="NUMBER OF BED LAYERS")
    sedlay_ini_opt: Literal[0, 1] = Field(
        0, description="INITIAL VERTICAL DISCRETISATION OF SEDIMENT LAYERS"
    )
    toplay_inithick: float = Field(0.1, description="Initial top layer thickness (m)")
    newlayer_thick: float = Field(
        0.001, description="BED LAYER THICKNESS THRESHOLD (m)"
    )
    imeth_bed_evol: Literal[1, 2] = Field(
        2,
        description="EXNER EQUATION: Numerical method for the resolution of the sediment continuity equation",
    )
    poro_option: Literal[1, 2] = Field(1, description="SEDIMENT POROSITY option")
    porosity: float = Field(0.4, description="Constant porosity value")
    awooster: float = Field(
        0.42, description="Wooster et al. (2008) porosity parameter A"
    )
    bwooster: float = Field(
        -0.458, description="Wooster et al. (2008) porosity parameter B"
    )

    @field_validator("ised_type", "srho", "wsed", "tau_ce")
    def validate_list_length(cls, v):
        if len(v) != 5:
            raise ValueError("Must have 5 values")
        return v


class Sediment(NamelistBaseModel):
    """
    Master model for non-cohesive sediment model parameters.
    """

    sed_core: Optional[SedCore] = Field(
        None, description="Core parameters for non-cohesive sediment model"
    )
    sed_opt: Optional[SedOpt] = Field(
        None, description="Optional parameters for non-cohesive sediment model"
    )

    model_config = ConfigDict(title="Non-cohesive Sediment Model Parameters")
