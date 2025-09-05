# This file was auto generated from a SCHISM namelist file on 2025-01-24.

from typing import Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Sed_core(NamelistBaseModel):
    sd50: Optional[list] = Field(
        [0.12, 0.18, 0.39, 0.60, 1.2],
        description="Median sediment grain diameter (D50) for each sediment tracer, specified in millimeters. This is a list of values corresponding to the number of sediment tracers (Ntracers).",
    )
    erate: Optional[list] = Field(
        [1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3, 1.6e-3],
        description="Surface erosion rate for each sediment tracer. The interpretation and units depend on the 'ierosion' parameter. If ierosion=0, the units are kg/m²/s. If ierosion=1, the units are s/m (as per M_E in Winterwerp et al. 2012, JGR, vol 117).",
    )

    @field_validator("sd50")
    @classmethod
    def validate_sd50(cls, v):
        if not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("All Sd50 values must be positive numbers")
        return v

    @field_validator("erate")
    @classmethod
    def validate_erate(cls, v):
        if not all(isinstance(x, (int, float)) and x >= 0 for x in v):
            raise ValueError("All Erate values must be non-negative numbers")
        return v

    @model_validator(mode="after")
    def validate_erate_length(self):
        if len(self.erate) != len(self.sd50):
            raise ValueError("Erate and Sd50 must have the same number of elements")
        return self


class Sed_opt(NamelistBaseModel):
    isedtype: Optional[list] = Field(
        [1, 1, 1, 1, 1],
        description="Sediment type for each class. 0: MUD-like (suspension only), 1: SAND-like (suspension + bedload), 2: GRAVEL-like (not available)",
    )
    srho: Optional[list] = Field(
        [2650.0, 2650.0, 2650.0, 2650.0, 2650.0],
        description="Sediment grain density (kg/m3) for each sediment class",
    )
    comp_ws: Optional[int] = Field(
        0,
        description="Flag to enable/disable computation of sediment settling velocity. 0: Disabled (user-defined), 1: Enabled (computed from SAND_SD50 and SAND_SRHO)",
    )
    comp_tauce: Optional[int] = Field(
        0,
        description="Flag to enable/disable computation of sediment critical shear stress. 0: Disabled (user-defined), 1: Enabled (computed from SAND_SD50 and SAND_SRHO)",
    )
    wsed: Optional[list] = Field(
        [1.06, 3.92, 5.43, 10.19, 28.65],
        description="Particle settling velocity (mm/s) for each sediment class",
    )
    tau_ce: Optional[list] = Field(
        [0.15, 0.17, 0.23, 0.3, 0.6],
        description="Critical shear stress for erosion (Pa) for each sediment class",
    )
    sed_debug: Optional[int] = Field(
        0,
        description="Debug flag. 0: silent, 1: output variables to outputs/nonfatal_*",
    )
    ised_dump: Optional[int] = Field(
        0, description="Dumping/dredging option. 0: no, 1: needs input sed_dump.in"
    )
    bedload: Optional[int] = Field(
        1,
        description="Bedload transport formula. 0: Disabled, 1: van Rijn (2007), 3: Soulsby and Damgaard (2005), 4: Wu and Lin (2014)",
    )
    bedload_filter: Optional[int] = Field(
        0,
        description="Flag to enable/disable diffusive filter for bedload fluxes. 0: Disabled, 1: Enabled",
    )
    bedload_limiter: Optional[int] = Field(
        0,
        description="Flag to enable/disable limiting of bedload flux components. 0: Disabled, 1: Enabled",
    )
    suspended_load: Optional[int] = Field(
        1,
        description="Flag to enable/disable suspended load transport. 0: Disabled, 1: Enabled",
    )
    iasym: Optional[int] = Field(0, description="")
    w_asym_max: Optional[float] = Field(0.4, description="")
    elfrink_filter: Optional[int] = Field(0, description="")
    ech_uorb: Optional[int] = Field(200, description="")
    bedload_acc: Optional[int] = Field(0, description="")
    bedload_acc_filter: Optional[int] = Field(0, description="")
    kacc_hoe: Optional[float] = Field(1.4e-4, description="")
    kacc_dub: Optional[float] = Field(6.31e-5, description="")
    thresh_acc_opt: Optional[int] = Field(2, description="")
    acrit: Optional[float] = Field(0.2, description="")
    tau_option: Optional[int] = Field(1, description="")
    tau_max: Optional[float] = Field(10.0, description="[Pa]")
    zstress: Optional[float] = Field(0.2, description="[m]; only used if tau_option/=1")
    ierosion: Optional[int] = Field(0, description="")
    slope_formulation: Optional[int] = Field(4, description="")
    alpha_bs: Optional[float] = Field(
        1.0, description="only used if slope_formulation=4"
    )
    alpha_bn: Optional[float] = Field(
        1.5, description="only used if slope_formulation=4"
    )
    ised_bc_bot: Optional[int] = Field(1, description="")
    alphd: Optional[float] = Field(1.0, description="")
    refht: Optional[float] = Field(0.75, description="suggested value: 0.75;")
    tbp: Optional[float] = Field(100.0, description="suggested value: 100;")
    im_pick_up: Optional[int] = Field(4, description="")
    sed_morph: Optional[int] = Field(0, description="")
    sed_morph_time: Optional[float] = Field(1.0, description="")
    morph_fac: Optional[float] = Field(1.0, description="for all classes")
    drag_formulation: Optional[int] = Field(1, description="")
    ddensed: Optional[int] = Field(0, description="")
    bedforms_rough: Optional[int] = Field(0, description="")
    iwave_ripple: Optional[int] = Field(1, description="")
    irough_bdld: Optional[int] = Field(1, description="")
    slope_avalanching: Optional[int] = Field(1, description="")
    dry_slope_cr: Optional[float] = Field(1.0, description="")
    wet_slope_cr: Optional[float] = Field(0.3, description="")
    bedmass_filter: Optional[int] = Field(0, description="")
    bedmass_threshold: Optional[float] = Field(0.025, description="")
    bdldiffu: Optional[float] = Field(0.5, description="")
    bedload_coeff: Optional[float] = Field(1.0, description="")
    cdb_min: Optional[float] = Field(1e-6, description="")
    cdb_max: Optional[float] = Field(0.01, description="")
    actv_max: Optional[float] = Field(0.05, description="")
    nbed: Optional[int] = Field(1, description="")
    sedlay_ini_opt: Optional[int] = Field(0, description="")
    toplay_inithick: Optional[float] = Field(0.1, description="")
    newlayer_thick: Optional[float] = Field(0.001, description="")
    imeth_bed_evol: Optional[int] = Field(2, description="")
    poro_option: Optional[int] = Field(1, description="")
    porosity: Optional[float] = Field(0.4, description="")
    awooster: Optional[float] = Field(0.42, description="")
    bwooster: Optional[float] = Field(-0.458, description="")

    @field_validator("isedtype")
    @classmethod
    def validate_isedtype(cls, v):
        if not all(0 <= x <= 1 for x in v):
            raise ValueError("isedtype values must be 0 or 1")
        return v

    @field_validator("srho")
    @classmethod
    def validate_srho(cls, v):
        if any(x <= 0 for x in v):
            raise ValueError("srho values must be positive")
        return v

    @field_validator("comp_ws")
    @classmethod
    def validate_comp_ws(cls, v):
        if v not in [0, 1]:
            raise ValueError("comp_ws must be 0 or 1")
        return v

    @field_validator("comp_tauce")
    @classmethod
    def validate_comp_tauce(cls, v):
        if v not in [0, 1]:
            raise ValueError("comp_tauce must be 0 or 1")
        return v

    @field_validator("wsed")
    @classmethod
    def validate_wsed(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("wsed values must be non-negative")
        return v

    @field_validator("tau_ce")
    @classmethod
    def validate_tau_ce(cls, v):
        if any(x < 0 for x in v):
            raise ValueError("tau_ce values must be non-negative")
        return v

    @field_validator("sed_debug")
    @classmethod
    def validate_sed_debug(cls, v):
        if v not in [0, 1]:
            raise ValueError("sed_debug must be 0 or 1")
        return v

    @field_validator("ised_dump")
    @classmethod
    def validate_ised_dump(cls, v):
        if v not in [0, 1]:
            raise ValueError("ised_dump must be 0 or 1")
        return v

    @field_validator("bedload")
    @classmethod
    def validate_bedload(cls, v):
        if v not in [0, 1, 3, 4]:
            raise ValueError("bedload must be 0, 1, 3, or 4")
        return v

    @field_validator("bedload_filter")
    @classmethod
    def validate_bedload_filter(cls, v):
        if v not in [0, 1]:
            raise ValueError("bedload_filter must be 0 or 1")
        return v

    @field_validator("bedload_limiter")
    @classmethod
    def validate_bedload_limiter(cls, v):
        if v not in [0, 1]:
            raise ValueError("bedload_limiter must be 0 or 1")
        return v

    @field_validator("suspended_load")
    @classmethod
    def validate_suspended_load(cls, v):
        if v not in [0, 1]:
            raise ValueError("suspended_load must be 0 or 1")
        return v

    @model_validator(mode="after")
    def check_wsed_comp_ws(self):
        if self.comp_ws == 1 and any(
            self.isedtype[i] == 1 for i in range(len(self.isedtype))
        ):
            print(
                "Warning: wsed values will be overwritten for SAND-like classes when comp_ws=1"
            )
        return self

    @model_validator(mode="after")
    def check_tau_ce_comp_tauce(self):
        if self.comp_tauce == 1 and any(
            self.isedtype[i] == 1 for i in range(len(self.isedtype))
        ):
            print(
                "Warning: tau_ce values will be overwritten for SAND-like classes when comp_tauce=1"
            )
        return self


class Sediment(NamelistBaseModel):
    sed_core: Optional[Sed_core] = Field(default_factory=Sed_core)
    sed_opt: Optional[Sed_opt] = Field(default_factory=Sed_opt)
