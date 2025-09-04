# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class MiceIn(NamelistBaseModel):
    """
    Configuration for ice-related parameters in the model.
    """

    ice_tests: int = Field(0, description="Box test flag")
    ihot_mice: Literal[0, 1, 2] = Field(
        1, description="0: cold start 1: restart 2: hotstart_from_HYCOM"
    )
    ice_advection: Literal[3, 4, 5, 6, 7] = Field(
        6,
        description="Advection scheme: 3: upwind 4: center-difference 5: tvd 6: tvd-up 7: TVD_Casulli",
    )
    ice_therm_on: Literal[0, 1] = Field(1, description="Ice thermodynamics on/off flag")
    ievp: Literal[1, 2] = Field(2, description="1: EVP; 2: mEVP")
    ice_cutoff: float = Field(
        0.001,
        description="Cut-off thickness [m] or fraction for ice. No ice velocity if *<=ice_cuttoff",
    )
    evp_rheol_steps: int = Field(
        500, description="The number of subcycling steps in EVP"
    )
    mevp_rheol_steps: int = Field(500, description="The number of iterations in mEVP")
    delta_min: float = Field(
        1.0e-11,
        description="(1/s) Limit for minimum divergence (Hibler, Hunke normally use 2.0e-9, which does much stronger limiting; valid for both VP and EVP)",
    )
    theta_io: float = Field(0.0, description="Ice/ocean rotation angle [degr]")
    mevp_alpha1: float = Field(
        200.0, description="Const used in mEVP (constitutive eq)"
    )
    mevp_alpha2: float = Field(200.0, description="Const used in mEVP for momentum eq")
    pstar: float = Field(27500.0, description="[N/m^2]")
    ellipse: float = Field(2.0, description="Ellipticity")
    c_pressure: float = Field(20.0, description="C [-]")
    niter_fct: int = Field(3, description="# of iterations in higher-order solve")
    ice_gamma_fct: float = Field(0.25, description="Smoothing parameter")
    h_ml0: float = Field(0.1, description="Ocean mixed layer depth [m]")
    salt_ice: float = Field(5.0, description="Salinity for ice [PSU] (>=0)")
    salt_water: float = Field(34.0, description="Salinity for water [PSU] (>=0)")

    @field_validator("salt_ice", "salt_water")
    def validate_salinity(cls, v):
        if v < 0:
            raise ValueError("Salinity must be >= 0")
        return v

    @model_validator(mode="after")
    def validate_salinity_difference(self):
        if self.salt_water <= self.salt_ice:
            raise ValueError("salt_water must be greater than salt_ice")
        return self


class Mice(NamelistBaseModel):
    """
    Master configuration for the ice model parameters.
    """

    mice_in: Optional[MiceIn] = Field(None, description="Ice-related parameters")
