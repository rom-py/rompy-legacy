# This file was auto generated from a SCHISM namelist file on 2025-01-24.

from typing import Optional

from pydantic import Field, field_validator, model_validator
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Ice_in(NamelistBaseModel):
    ice_tests: Optional[int] = Field(
        0, description="Flag for box test. 0 disables, 1 enables."
    )
    ice_advection: Optional[int] = Field(
        1, description="Flag to enable/disable ice advection. 1 enables, 0 disables."
    )
    ice_therm_on: Optional[int] = Field(
        1,
        description="Flag to enable/disable ice thermodynamics. 1 enables, 0 disables.",
    )
    ievp: Optional[int] = Field(
        2,
        description="Selects the rheology model. 1 for EVP (Elastic-Viscous-Plastic), 2 for mEVP (modified EVP).",
    )
    ice_cutoff: Optional[float] = Field(
        0.001,
        description="Cut-off thickness [m] or fraction for ice. No ice velocity if ice thickness is less than or equal to ice_cutoff.",
    )
    evp_rheol_steps: Optional[int] = Field(
        200, description="Number of subcycling steps in EVP rheology model."
    )
    mevp_rheol_steps: Optional[int] = Field(
        200, description="Number of iterations in mEVP rheology model."
    )
    ice_atmos_stress_form: Optional[int] = Field(
        1,
        description="Form of atmospheric stress calculation. 0 for constant Cd, 1 for FESOM formulation.",
    )
    cdwin0: Optional[float] = Field(
        0.002,
        description="Constant drag coefficient for wind stress, used if ice_atmos_stress_form=0.",
    )
    delta_min: Optional[float] = Field(
        2e-09,
        description="Limit for minimum divergence (1/s). Used in both VP and EVP rheology models.",
    )
    theta_io: Optional[float] = Field(
        0.0,
        description="Ice/ocean rotation angle in degrees. Usually 0 unless vertical grid is too coarse.",
    )
    mevp_coef: Optional[int] = Field(
        0,
        description="Options for specifying 2 relax coefficients in mEVP. 0 for constant, 1 for variable coefficients.",
    )
    mevp_alpha1: Optional[float] = Field(
        200.0,
        description="Constant used in mEVP for constitutive equation if mevp_coef=0.",
    )
    mevp_alpha2: Optional[float] = Field(
        200.0, description="Constant used in mEVP for momentum equation if mevp_coef=0."
    )
    mevp_alpha3: Optional[float] = Field(
        200.0, description="Minimum value for variable coefficients if mevp_coef=1."
    )
    mevp_alpha4: Optional[float] = Field(
        0.02,
        description="Coefficient used in variable coefficient calculation if mevp_coef=1.",
    )
    pstar: Optional[float] = Field(
        15000.0, description="Ice strength parameter [N/m^2]."
    )
    ellipse: Optional[float] = Field(2.0, description="Ellipticity of the yield curve.")
    c_pressure: Optional[float] = Field(
        20.0, description="Ice concentration parameter C [-]."
    )
    ncyc_fct: Optional[int] = Field(
        1, description="Number of subcycling steps in transport for FCT scheme."
    )
    niter_fct: Optional[int] = Field(
        3, description="Number of iterations in higher-order solve for FCT scheme."
    )
    ice_gamma_fct: Optional[float] = Field(
        0.25,
        description="Smoothing parameter for FCT scheme. 1 for maximum positivity preserving.",
    )
    depth_ice_fct: Optional[float] = Field(
        5.0, description="Cut-off depth (m) for non-FCT zone in ice_fct.gr3."
    )
    h_ml0: Optional[float] = Field(
        0.1, description="Ocean mixed layer depth [m] for thermodynamics calculations."
    )
    salt_ice: Optional[float] = Field(
        5.0, description="Salinity for ice [PSU] (must be non-negative)."
    )
    salt_water: Optional[float] = Field(
        34.0, description="Salinity for water [PSU] (must be non-negative)."
    )
    lead_closing: Optional[float] = Field(
        0.5,
        description="Lead closing parameter [m]. Larger values slow down freezing-up but increase sea ice thickness.",
    )
    saterm: Optional[float] = Field(
        0.5, description="Semter constant. Smaller values could slow down melting."
    )
    albsn: Optional[float] = Field(0.85, description="Albedo for frozen snow.")
    albsnm: Optional[float] = Field(
        0.75,
        description="Albedo for melting snow (must be less than or equal to albsn).",
    )
    albi: Optional[float] = Field(
        0.75, description="Albedo for frozen ice (must be less than or equal to albsn)."
    )
    albm: Optional[float] = Field(
        0.66, description="Albedo for melting ice (must be less than or equal to albi)."
    )

    @field_validator("ice_tests")
    @classmethod
    def validate_ice_tests(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError("ice_tests must be 0 or 1")
        return v

    @field_validator("ice_advection")
    @classmethod
    def validate_ice_advection(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError("ice_advection must be 0 or 1")
        return v

    @field_validator("ice_therm_on")
    @classmethod
    def validate_ice_therm_on(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError("ice_therm_on must be 0 or 1")
        return v

    @field_validator("ievp")
    @classmethod
    def validate_ievp(cls, v):
        if not isinstance(v, int) or v not in [1, 2]:
            raise ValueError("ievp must be 1 or 2")
        return v

    @field_validator("ice_cutoff")
    @classmethod
    def validate_ice_cutoff(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("ice_cutoff must be a positive float")
        return v

    @field_validator("evp_rheol_steps")
    @classmethod
    def validate_evp_rheol_steps(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("evp_rheol_steps must be a positive integer")
        return v

    @field_validator("mevp_rheol_steps")
    @classmethod
    def validate_mevp_rheol_steps(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("mevp_rheol_steps must be a positive integer")
        return v

    @field_validator("ice_atmos_stress_form")
    @classmethod
    def validate_ice_atmos_stress_form(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError("ice_atmos_stress_form must be 0 or 1")
        return v

    @field_validator("cdwin0")
    @classmethod
    def validate_cdwin0(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("cdwin0 must be a positive float")
        return v

    @field_validator("delta_min")
    @classmethod
    def validate_delta_min(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("delta_min must be a positive float")
        return v

    @field_validator("theta_io")
    @classmethod
    def validate_theta_io(cls, v):
        if not isinstance(v, float) or v < 0 or v >= 360:
            raise ValueError("theta_io must be a float between 0 and 360")
        return v

    @field_validator("mevp_coef")
    @classmethod
    def validate_mevp_coef(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError("mevp_coef must be 0 or 1")
        return v

    @field_validator("mevp_alpha1")
    @classmethod
    def validate_mevp_alpha1(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("mevp_alpha1 must be a positive float")
        return v

    @field_validator("mevp_alpha2")
    @classmethod
    def validate_mevp_alpha2(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("mevp_alpha2 must be a positive float")
        return v

    @field_validator("mevp_alpha3")
    @classmethod
    def validate_mevp_alpha3(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("mevp_alpha3 must be a positive float")
        return v

    @field_validator("mevp_alpha4")
    @classmethod
    def validate_mevp_alpha4(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("mevp_alpha4 must be a positive float")
        return v

    @field_validator("pstar")
    @classmethod
    def validate_pstar(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("pstar must be a positive float")
        return v

    @field_validator("ellipse")
    @classmethod
    def validate_ellipse(cls, v):
        if not isinstance(v, float) or v <= 1:
            raise ValueError("ellipse must be a float greater than 1")
        return v

    @field_validator("c_pressure")
    @classmethod
    def validate_c_pressure(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("c_pressure must be a positive float")
        return v

    @field_validator("ncyc_fct")
    @classmethod
    def validate_ncyc_fct(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("ncyc_fct must be a positive integer")
        return v

    @field_validator("niter_fct")
    @classmethod
    def validate_niter_fct(cls, v):
        if not isinstance(v, int) or v <= 0:
            raise ValueError("niter_fct must be a positive integer")
        return v

    @field_validator("ice_gamma_fct")
    @classmethod
    def validate_ice_gamma_fct(cls, v):
        if not isinstance(v, float) or v < 0 or v > 1:
            raise ValueError("ice_gamma_fct must be a float between 0 and 1")
        return v

    @field_validator("depth_ice_fct")
    @classmethod
    def validate_depth_ice_fct(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("depth_ice_fct must be a positive float")
        return v

    @field_validator("h_ml0")
    @classmethod
    def validate_h_ml0(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("h_ml0 must be a positive float")
        return v

    @field_validator("salt_ice")
    @classmethod
    def validate_salt_ice(cls, v):
        if not isinstance(v, float) or v < 0:
            raise ValueError("salt_ice must be a non-negative float")
        return v

    @field_validator("salt_water")
    @classmethod
    def validate_salt_water(cls, v):
        if not isinstance(v, float) or v < 0:
            raise ValueError("salt_water must be a non-negative float")
        return v

    @field_validator("lead_closing")
    @classmethod
    def validate_lead_closing(cls, v):
        if not isinstance(v, float) or v <= 0:
            raise ValueError("lead_closing must be a positive float")
        return v

    @field_validator("saterm")
    @classmethod
    def validate_saterm(cls, v):
        if not isinstance(v, float) or v <= 0 or v > 1:
            raise ValueError("saterm must be a float between 0 and 1")
        return v

    @field_validator("albsn")
    @classmethod
    def validate_albsn(cls, v):
        if not isinstance(v, float) or v <= 0 or v > 1:
            raise ValueError("albsn must be a float between 0 and 1")
        return v

    @field_validator("albsnm")
    @classmethod
    def validate_albsnm(cls, v):
        if not isinstance(v, float) or v <= 0 or v > 1:
            raise ValueError("albsnm must be a float between 0 and 1")
        return v

    @field_validator("albi")
    @classmethod
    def validate_albi(cls, v):
        if not isinstance(v, float) or v <= 0 or v > 1:
            raise ValueError("albi must be a float between 0 and 1")
        return v

    @field_validator("albm")
    @classmethod
    def validate_albm(cls, v):
        if not isinstance(v, float) or v <= 0 or v > 1:
            raise ValueError("albm must be a float between 0 and 1")
        return v

    @model_validator(mode="after")
    def validate_rheol_steps(self):
        if self.ievp == 1 and self.evp_rheol_steps <= 0:
            raise ValueError("evp_rheol_steps must be positive when ievp is 1")
        return self

    @model_validator(mode="after")
    def validate_rheol_steps(self):
        if self.ievp == 2 and self.mevp_rheol_steps <= 0:
            raise ValueError("mevp_rheol_steps must be positive when ievp is 2")
        return self

    @model_validator(mode="after")
    def validate_cdwin0_usage(self):
        if self.ice_atmos_stress_form == 0 and self.cdwin0 <= 0:
            raise ValueError("cdwin0 must be positive when ice_atmos_stress_form is 0")
        return self

    @model_validator(mode="after")
    def validate_mevp_alpha1_usage(self):
        if self.mevp_coef == 0 and self.mevp_alpha1 <= 0:
            raise ValueError("mevp_alpha1 must be positive when mevp_coef is 0")
        return self

    @model_validator(mode="after")
    def validate_mevp_alpha2_usage(self):
        if self.mevp_coef == 0 and self.mevp_alpha2 <= 0:
            raise ValueError("mevp_alpha2 must be positive when mevp_coef is 0")
        return self

    @model_validator(mode="after")
    def validate_mevp_alpha3_usage(self):
        if self.mevp_coef == 1 and self.mevp_alpha3 <= 0:
            raise ValueError("mevp_alpha3 must be positive when mevp_coef is 1")
        return self

    @model_validator(mode="after")
    def validate_mevp_alpha4_usage(self):
        if self.mevp_coef == 1 and self.mevp_alpha4 <= 0:
            raise ValueError("mevp_alpha4 must be positive when mevp_coef is 1")
        return self

    @model_validator(mode="after")
    def validate_albsnm_relation(self):
        if self.albsnm > self.albsn:
            raise ValueError("albsnm must be less than or equal to albsn")
        return self

    @model_validator(mode="after")
    def validate_albi_relation(self):
        if self.albi > self.albsn:
            raise ValueError("albi must be less than or equal to albsn")
        return self

    @model_validator(mode="after")
    def validate_albm_relation(self):
        if self.albm > self.albi:
            raise ValueError("albm must be less than or equal to albi")
        return self


class Ice(NamelistBaseModel):
    ice_in: Optional[Ice_in] = Field(default_factory=Ice_in)
