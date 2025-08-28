# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class IceIn(NamelistBaseModel):
    """
    Configuration for ice model parameters and settings.
    """

    ice_tests: int = Field(0, description="Box test flag")
    ice_advection: int = Field(1, description="Advection on/off")
    ice_therm_on: int = Field(1, description="Ice thermodynamics on/off flag")
    ievp: Literal[1, 2] = Field(2, description="1: EVP; 2: mEVP")
    ice_cutoff: float = Field(
        1e-3,
        description="Cut-off thickness [m] or fraction for ice. No ice velocity if *<=ice_cuttoff",
    )
    evp_rheol_steps: int = Field(
        200, description="The number of subcycling steps in EVP"
    )
    mevp_rheol_steps: int = Field(200, description="The number of iterations in mEVP")
    ice_atmos_stress_form: Literal[0, 1] = Field(
        1, description="0-const Cd; 1: FESOM formulation"
    )
    cdwin0: float = Field(
        2e-3, description="Needed if ice_atmos_stress_form=0 (const Cdw)"
    )
    delta_min: float = Field(
        2.0e-9,
        description="(1/s) Limit for minimum divergence (Hibler, Hunke normally use 2.0e-9, which does much stronger limiting; valid for both VP and EVP",
    )
    theta_io: float = Field(
        0.0,
        description="Ice/ocean rotation angle. [degr]. Usually 0 unless vgrid is too coarse",
    )
    mevp_coef: Literal[0, 1] = Field(
        0, description="Options for specifying 2 relax coefficients in mEVP only"
    )
    mevp_alpha1: float = Field(
        200.0, description="Const used in mEVP (constitutive eq), if mevp_coef=0"
    )
    mevp_alpha2: float = Field(
        200.0, description="Const used in mEVP for momentum eq, if mevp_coef=0"
    )
    mevp_alpha3: float = Field(200.0, description="Used if mevp_coef=1")
    mevp_alpha4: float = Field(2e-2, description="Used if mevp_coef=1")
    pstar: float = Field(15000.0, description="[N/m^2]")
    ellipse: float = Field(2.0, description="Ellipticity")
    c_pressure: float = Field(20.0, description="C [-]")
    ncyc_fct: int = Field(1, description="# of subcycling in transport")
    niter_fct: int = Field(3, description="# of iterations in higher-order solve")
    ice_gamma_fct: float = Field(
        0.25, description="Smoothing parameter; 1 for max positivity preserving"
    )
    depth_ice_fct: float = Field(5.0, description="Cut off depth (m) for non-FCT")
    h_ml0: float = Field(0.1, description="Ocean mixed layer depth [m]")
    salt_ice: float = Field(5.0, description="Salinity for ice [PSU] (>=0)")
    salt_water: float = Field(34.0, description="Salinity for water [PSU] (>=0)")
    lead_closing: float = Field(
        0.5,
        description="Lead closing parameter [m] - larger values slow down freezing-up but increase sea ice thickness",
    )
    Saterm: float = Field(
        0.5, description="Semter const -smaller value could slow down melting"
    )
    albsn: float = Field(0.85, description="Albedo: frozen snow")
    albsnm: float = Field(0.75, description="Melting snow (<=albsn)")
    albi: float = Field(0.75, description="Frozen ice (<=albsn)")
    albm: float = Field(0.66, description="Melting ice (<=albi)")

    @field_validator("salt_ice", "salt_water")
    def check_salinity(cls, v):
        if v < 0:
            raise ValueError("Salinity must be >= 0")
        return v

    @model_validator(mode="after")
    def check_albedo_relationships(cls, values):
        if values.albsnm > values.albsn:
            raise ValueError("albsnm must be <= albsn")
        if values.albi > values.albsn:
            raise ValueError("albi must be <= albsn")
        if values.albm > values.albi:
            raise ValueError("albm must be <= albi")
        return values


class Ice(NamelistBaseModel):
    """
    Master model for ice configuration.
    """

    ice_in: Optional[IceIn] = Field(
        None, description="Ice model parameters and settings"
    )
