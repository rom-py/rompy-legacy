# This file was auto generated from a SCHISM namelist file on 2025-01-24.

from typing import Optional

from pydantic import Field, field_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Mice_in(NamelistBaseModel):
    ice_tests: Optional[int] = Field(
        0, description="Flag for box test. 0 indicates no box test."
    )
    ihot_mice: Optional[int] = Field(
        1,
        description="Start mode for ice model. 0: cold start, 1: restart, 2: hotstart from HYCOM.",
    )
    ice_advection: Optional[int] = Field(
        6,
        description="Ice advection scheme. 3: upwind, 4: center-difference, 5: tvd, 6: tvd-up, 7: TVD_Casulli.",
    )
    ice_therm_on: Optional[int] = Field(
        1, description="Flag for ice thermodynamics. 1: on, 0: off."
    )
    ievp: Optional[int] = Field(
        2,
        description="Elastic-Viscous-Plastic (EVP) model selection. 1: EVP, 2: modified EVP (mEVP).",
    )
    ice_cutoff: Optional[float] = Field(
        0.001,
        description="Cut-off thickness [m] or fraction for ice. No ice velocity if ice thickness <= ice_cutoff.",
    )
    evp_rheol_steps: Optional[int] = Field(
        500, description="Number of subcycling steps in EVP."
    )
    mevp_rheol_steps: Optional[int] = Field(
        500, description="Number of iterations in modified EVP (mEVP)."
    )
    delta_min: Optional[float] = Field(
        1e-11,
        description="Limit for minimum divergence (1/s). Used in both VP and EVP.",
    )
    theta_io: Optional[float] = Field(
        0.0, description="Ice/ocean rotation angle in degrees."
    )
    mevp_alpha1: Optional[float] = Field(
        200.0, description="Constant used in mEVP for constitutive equation."
    )
    mevp_alpha2: Optional[float] = Field(
        200.0, description="Constant used in mEVP for momentum equation."
    )
    pstar: Optional[float] = Field(
        27500.0, description="Ice strength parameter [N/m^2]."
    )
    ellipse: Optional[float] = Field(
        2.0, description="Ellipticity parameter for ice rheology."
    )
    c_pressure: Optional[float] = Field(
        20.0, description="Ice pressure coefficient [-]."
    )
    niter_fct: Optional[int] = Field(
        3,
        description="Number of iterations in higher-order solve for FCT (Flux-Corrected Transport).",
    )
    ice_gamma_fct: Optional[float] = Field(
        0.25, description="Smoothing parameter for FCT."
    )
    h_ml0: Optional[float] = Field(
        0.1, description="Initial ocean mixed layer depth [m]."
    )
    salt_ice: Optional[float] = Field(5.0, description="Salinity for ice [PSU].")
    salt_water: Optional[float] = Field(34.0, description="Salinity for water [PSU].")

    @field_validator("ice_tests")
    @classmethod
    def validate_ice_tests(cls, v):
        if not isinstance(v, int) or v not in [0, 1]:
            raise ValueError("ice_tests must be 0 or 1")
        return v

    @field_validator("ihot_mice")
    @classmethod
    def validate_ihot_mice(cls, v):
        if not isinstance(v, int) or v not in [0, 1, 2]:
            raise ValueError("ihot_mice must be 0, 1, or 2")
        return v

    @field_validator("ice_advection")
    @classmethod
    def validate_ice_advection(cls, v):
        if not isinstance(v, int) or v not in range(3, 8):
            raise ValueError("ice_advection must be an integer between 3 and 7")
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
        if not isinstance(v, float) or v < 0:
            raise ValueError("ice_cutoff must be a non-negative float")
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


class Mice(NamelistBaseModel):
    mice_in: Optional[Mice_in] = Field(default_factory=Mice_in)
