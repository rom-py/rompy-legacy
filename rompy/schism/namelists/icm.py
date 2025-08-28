# This file was auto generated from a SCHISM namelist file on 2025-01-24.

from typing import Optional

from pydantic import Field, field_validator, model_validator
from rompy.schism.namelists.basemodel import NamelistBaseModel


class Marco(NamelistBaseModel):
    nsub: Optional[int] = Field(1, description="Number of subcycles in ICM kinetics")
    ike: Optional[int] = Field(
        0,
        description="Option for computing light attenuation coefficients. 0: Ke=Ke0+KeC*Chl+KeS*(tss2c*POC), 1: Ke=Ke0+KeC*Chl+KeS*TSS, 2: Ke=Ke0+KeC*Chl+KeSalt*Salt",
    )
    ke0: Optional[float] = Field(
        0.26, description="Background light extinction coefficient (1/m)"
    )
    kec: Optional[float] = Field(
        0.017, description="Light attenuation due to chlorophyll"
    )
    kes: Optional[float] = Field(0.07, description="Light attenuation due to TSS")
    kesalt: Optional[float] = Field(
        -0.02, description="Light attenuation due to CDOM (related to salinity)"
    )
    tss2c: Optional[float] = Field(6.0, description="TSS to carbon ratio")
    ilight: Optional[int] = Field(
        0,
        description="Option for computing light limitation factor. 0: Carl Cerco method (unit: E/m^2)",
    )
    alpha: Optional[list] = Field(
        [8.0, 8.0, 8.0],
        description="Initial slope of P-I curve (g[C]*m2/g[Chl]/E) for each phytoplankton group",
    )
    ipr: Optional[int] = Field(
        1,
        description="Option for phytoplankton predation term. 0: linear formulation, 1: quadratic",
    )
    prr: Optional[list] = Field(
        [0.1, 0.2, 0.05],
        description="Predation rate by higher trophic level (day-1 or day-1.g-1.m3) for each phytoplankton group",
    )
    wqc0: Optional[list] = Field(
        [
            1.0,
            0.5,
            0.05,
            1.0,
            0.5,
            0.5,
            0.15,
            0.15,
            0.05,
            0.01,
            0.05,
            0.005,
            0.005,
            0.01,
            0.05,
            0.0,
            12.0,
        ],
        description="Initial values for ICM state variables",
    )
    wsp: Optional[list] = Field(
        [
            0.3,
            0.1,
            0.0,
            0.25,
            0.25,
            0.0,
            0.25,
            0.25,
            0.0,
            0.0,
            0.0,
            0.25,
            0.25,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        description="Settling velocity for ICM state variables (m.day-1)",
    )
    wspn: Optional[list] = Field(
        [
            0.3,
            0.1,
            0.0,
            0.25,
            0.25,
            0.0,
            0.25,
            0.25,
            0.0,
            0.0,
            0.0,
            0.25,
            0.25,
            0.0,
            1.0,
            0.0,
            0.0,
        ],
        description="Net settling velocity for ICM state variables (m.day-1)",
    )
    isilica: Optional[int] = Field(0, description="Silica model switch. 0: OFF, 1: ON")
    izb: Optional[int] = Field(
        0,
        description="Zooplankton dynamics switch. 0: don't use, 1: use zooplankton dynamics",
    )
    iph: Optional[int] = Field(0, description="PH model switch. 0: OFF, 1: ON")
    icbp: Optional[int] = Field(
        0, description="Chesapeake Bay Program Model switch. 0: OFF, 1: ON"
    )
    isav_icm: Optional[int] = Field(
        0, description="Submerged Aquatic Vegetation switch. 0: OFF, 1: ON"
    )
    iveg_icm: Optional[int] = Field(
        0, description="Intertidal vegetation switch. 0: OFF, 1: ON"
    )
    ised: Optional[int] = Field(
        1, description="Sediment module switch. 0: OFF, 1: Use sediment flux model"
    )
    iba: Optional[int] = Field(0, description="Benthic Algae switch. 0: OFF, 1: ON")
    irad: Optional[int] = Field(
        0,
        description="Solar radiation option. 0: short wave from sflux, 1: short wave from ICM_rad.th.nc",
    )
    isflux: Optional[int] = Field(
        0,
        description="Atmospheric fluxes option. 0: OFF, 1: additional nutrient fluxes from ICM_sflux.th.nc",
    )
    ibflux: Optional[int] = Field(
        0,
        description="Bottom fluxes option. 0: OFF, 1: additional nutrient fluxes from ICM_bflux.th.nc",
    )
    iout_icm: Optional[int] = Field(
        0,
        description="ICM station outputs switch. 0: OFF, 1: ON (requires istation.in with *.bp format)",
    )
    nspool_icm: Optional[int] = Field(
        24, description="Output frequency for ICM station outputs"
    )
    ilimit: Optional[int] = Field(
        0,
        description="Option for nutrient limitation on phytoplankton growth. 0: f=min[f(N),f(P)]*f(I), 1: f=min[f(N),f(P),f(I)]",
    )
    idry_icm: Optional[int] = Field(
        0,
        description="Shallow kinetic biochemical process option. 0: jump dry elements, keep last wet value, 1: turn on shallow kinetic biochemical process",
    )

    @field_validator("nsub")
    @classmethod
    def validate_nsub(cls, v):
        if v < 1:
            raise ValueError("nsub must be a positive integer")
        return v

    @field_validator("ike")
    @classmethod
    def validate_ike(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("ike must be 0, 1, or 2")
        return v

    @field_validator("ke0")
    @classmethod
    def validate_ke0(cls, v):
        if v < 0:
            raise ValueError("ke0 must be non-negative")
        return v

    @field_validator("kec")
    @classmethod
    def validate_kec(cls, v):
        if v < 0:
            raise ValueError("kec must be non-negative")
        return v

    @field_validator("kes")
    @classmethod
    def validate_kes(cls, v):
        if v < 0:
            raise ValueError("kes must be non-negative")
        return v

    @field_validator("kesalt")
    @classmethod
    def validate_kesalt(cls, v):
        return v

    @field_validator("tss2c")
    @classmethod
    def validate_tss2c(cls, v):
        if v <= 0:
            raise ValueError("tss2c must be positive")
        return v

    @field_validator("ilight")
    @classmethod
    def validate_ilight(cls, v):
        if v != 0:
            raise ValueError("ilight must be 0")
        return v

    @field_validator("alpha")
    @classmethod
    def validate_alpha(cls, v):
        if len(v) != 3 or any(x <= 0 for x in v):
            raise ValueError("alpha must be a list of 3 positive values")
        return v

    @field_validator("ipr")
    @classmethod
    def validate_ipr(cls, v):
        if v not in [0, 1]:
            raise ValueError("ipr must be 0 or 1")
        return v

    @field_validator("prr")
    @classmethod
    def validate_prr(cls, v):
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("prr must be a list of 3 non-negative values")
        return v

    @field_validator("wqc0")
    @classmethod
    def validate_wqc0(cls, v):
        if len(v) != 16 or any(x < 0 for x in v):
            raise ValueError("wqc0 must be a list of 16 non-negative values")
        return v

    @field_validator("wsp")
    @classmethod
    def validate_wsp(cls, v):
        if len(v) != 16:
            raise ValueError("wsp must be a list of 16 values")
        return v

    @field_validator("wspn")
    @classmethod
    def validate_wspn(cls, v):
        if len(v) != 16:
            raise ValueError("wspn must be a list of 16 values")
        return v

    @field_validator("isilica")
    @classmethod
    def validate_isilica(cls, v):
        if v not in [0, 1]:
            raise ValueError("isilica must be 0 or 1")
        return v

    @field_validator("izb")
    @classmethod
    def validate_izb(cls, v):
        if v not in [0, 1]:
            raise ValueError("izb must be 0 or 1")
        return v

    @field_validator("iph")
    @classmethod
    def validate_iph(cls, v):
        if v not in [0, 1]:
            raise ValueError("iph must be 0 or 1")
        return v

    @field_validator("icbp")
    @classmethod
    def validate_icbp(cls, v):
        if v not in [0, 1]:
            raise ValueError("icbp must be 0 or 1")
        return v

    @field_validator("isav_icm")
    @classmethod
    def validate_isav_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("isav_icm must be 0 or 1")
        return v

    @field_validator("iveg_icm")
    @classmethod
    def validate_iveg_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("iveg_icm must be 0 or 1")
        return v

    @field_validator("ised")
    @classmethod
    def validate_ised(cls, v):
        if v not in [0, 1]:
            raise ValueError("ised must be 0 or 1")
        return v

    @field_validator("iba")
    @classmethod
    def validate_iba(cls, v):
        if v not in [0, 1]:
            raise ValueError("iba must be 0 or 1")
        return v

    @field_validator("irad")
    @classmethod
    def validate_irad(cls, v):
        if v not in [0, 1]:
            raise ValueError("irad must be 0 or 1")
        return v

    @field_validator("isflux")
    @classmethod
    def validate_isflux(cls, v):
        if v not in [0, 1]:
            raise ValueError("isflux must be 0 or 1")
        return v

    @field_validator("ibflux")
    @classmethod
    def validate_ibflux(cls, v):
        if v not in [0, 1]:
            raise ValueError("ibflux must be 0 or 1")
        return v

    @field_validator("iout_icm")
    @classmethod
    def validate_iout_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("iout_icm must be 0 or 1")
        return v

    @field_validator("nspool_icm")
    @classmethod
    def validate_nspool_icm(cls, v):
        if v <= 0:
            raise ValueError("nspool_icm must be positive")
        return v

    @field_validator("ilimit")
    @classmethod
    def validate_ilimit(cls, v):
        if v not in [0, 1]:
            raise ValueError("ilimit must be 0 or 1")
        return v

    @field_validator("idry_icm")
    @classmethod
    def validate_idry_icm(cls, v):
        if v not in [0, 1]:
            raise ValueError("idry_icm must be 0 or 1")
        return v


class Core(NamelistBaseModel):
    gpm: Optional[list] = Field(
        [2.5, 2.8, 3.5],
        description="Phytoplankton growth rates for three different species (day^-1)",
    )
    tgp: Optional[list] = Field(
        [15.0, 22.0, 27.0],
        description="Optimal temperatures for phytoplankton growth for three different species (°C)",
    )
    ktgp: Optional[list] = Field(
        [0.005, 0.004, 0.003, 0.008, 0.006, 0.004],
        description="Temperature dependence for phytoplankton growth, dimensioned as (PB=1:3,1:2) (°C^-2)",
    )
    mtr: Optional[list] = Field(
        [0.0, 0.0, 0.0],
        description="Phytoplankton photorespiration coefficients for three species (0 < MTR < 1)",
    )
    mtb: Optional[list] = Field(
        [0.01, 0.02, 0.03],
        description="Phytoplankton metabolism rates for three species (day^-1)",
    )
    tmt: Optional[list] = Field(
        [20.0, 20.0, 20.0],
        description="Reference temperatures for phytoplankton metabolism for three species (°C)",
    )
    ktmt: Optional[list] = Field(
        [0.0322, 0.0322, 0.0322],
        description="Temperature dependence for phytoplankton metabolism for three species (°C^-1)",
    )
    fcp: Optional[list] = Field(
        [0.35, 0.3, 0.2, 0.55, 0.5, 0.5, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
        description="Fractions of phytoplankton carbon into (RPOC,LPOC,DOC,SRPOC) for three species",
    )
    fnp: Optional[list] = Field(
        [
            0.35,
            0.35,
            0.35,
            0.5,
            0.5,
            0.5,
            0.1,
            0.1,
            0.1,
            0.05,
            0.05,
            0.05,
            0.0,
            0.0,
            0.0,
        ],
        description="Fractions of phytoplankton nitrogen into (RPON,LPON,DON,NH4,SRPON) for three species",
    )
    fpp: Optional[list] = Field(
        [0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0],
        description="Fractions of phytoplankton Phosphorus into (RPOP,LPOP,DOP,PO4,SRPOP) for three species",
    )
    fcm: Optional[list] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        description="Fractions of phytoplankton metabolism carbon into (RPOC,LPOC,DOC,SRPOC) for three species",
    )
    fnm: Optional[list] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Fractions of phytoplankton metabolism nitrogen into (RPON,LPON,DON,NH4,SRPON) for three species",
    )
    fpm: Optional[list] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Fractions of phytoplankton metabolism phosphorus into (RPOP,LPOP,DOP,PO4,SRPOP) for three species",
    )
    nit: Optional[float] = Field(
        0.07, description="Maximum nitrification rate (day^-1)"
    )
    tnit: Optional[float] = Field(
        27.0, description="Optimal temperature for nitrification (°C)"
    )
    ktnit: Optional[list] = Field(
        [0.0045, 0.0045],
        description="Temperature dependence for nitrification (T<=TNit & T>TNit) (°C^-2)",
    )
    khdon: Optional[float] = Field(
        1.0, description="Dissolved Oxygen half saturation for nitrification (mg/L)"
    )
    khdoox: Optional[float] = Field(
        0.5,
        description="Dissolved Oxygen half saturation for denitrification & DOC's oxic respiration (mg/L)",
    )
    khno3dn: Optional[float] = Field(
        0.1, description="Nitrate half saturation for denitrification (mg/L)"
    )
    kc0: Optional[list] = Field(
        [0.005, 0.075, 0.2],
        description="Minimum decay rates of RPOC, LPOC, DOC (day^-1)",
    )
    kn0: Optional[list] = Field(
        [0.005, 0.075, 0.2],
        description="Minimum decay rates of RPON, LPON, DON (day^-1)",
    )
    kp0: Optional[list] = Field(
        [0.005, 0.075, 0.2],
        description="Minimum decay rates of RPOP, LPOP, DOP (day^-1)",
    )
    kcalg: Optional[list] = Field(
        [0.0, 0.0, 0.0],
        description="Algae effect on RPOC, LPOC, DOC decay (day^-1.m3.g[C]^-1)",
    )
    knalg: Optional[list] = Field(
        [0.0, 0.0, 0.0],
        description="Algae effect on RPON, LPON, DON decay (day^-1.m3.g[C]^-1)",
    )
    kpalg: Optional[list] = Field(
        [0.0, 0.0, 0.0],
        description="Algae effect on RPOP, LPOP, DOP decay (day^-1.m3.g[C]^-1)",
    )
    trm: Optional[list] = Field(
        [20.0, 20.0, 20.0],
        description="Reference temperatures for (RPOM,LPOM,DOM) decay (°C)",
    )
    ktrm: Optional[list] = Field(
        [0.069, 0.069, 0.069],
        description="Temperature dependence for (RPOM,LPOM,DOM) decay (°C^-1)",
    )
    ksr0: Optional[list] = Field(
        [0.001, 0.001, 0.001], description="Decay rates of SRPOC, SRPON, SRPOP (day^-1)"
    )
    trsr: Optional[list] = Field(
        [20.0, 20.0, 20.0],
        description="Reference temperatures for (SRPOC,SRPON,SRPOP) decay (°C)",
    )
    ktrsr: Optional[list] = Field(
        [0.069, 0.069, 0.069],
        description="Temperature dependence for (SRPOC,SRPON,SRPOP) decay (°C^-1)",
    )
    kpip: Optional[float] = Field(0.0, description="Dissolution rate of PIP (day^-1)")
    kcd: Optional[float] = Field(
        1.0, description="Oxidation rate of COD at TRCOD (day^-1)"
    )
    trcod: Optional[float] = Field(
        20.0, description="Reference temperature for COD oxidation (°C)"
    )
    ktrcod: Optional[float] = Field(
        0.041, description="Temperature dependence for COD oxidation (°C^-1)"
    )
    khcod: Optional[float] = Field(
        1.5, description="COD half saturation for COD oxidation (mg[O2]/L)"
    )
    khn: Optional[list] = Field(
        [0.01, 0.01, 0.01],
        description="Nitrogen half saturation for three phytoplankton species (mg/L)",
    )
    khp: Optional[list] = Field(
        [0.001, 0.001, 0.001],
        description="Phosphorus half saturation for three phytoplankton species (mg/L)",
    )
    khsal: Optional[list] = Field(
        [1000000.0, 1000000.0, 0.1],
        description="Salinity when phytoplankton growth is halved for three species (PSU); (1e6: no salinity stress)",
    )
    c2chl: Optional[list] = Field(
        [0.059, 0.059, 0.059],
        description="Carbon to chlorophyll ratio for three phytoplankton species (g[C]/mg[Chl])",
    )
    n2c: Optional[list] = Field(
        [0.167, 0.167, 0.167],
        description="Nitrogen to carbon ratio for three phytoplankton species",
    )
    p2c: Optional[list] = Field(
        [0.02, 0.02, 0.02],
        description="Phosphorus to carbon ratio for three phytoplankton species",
    )
    o2c: Optional[float] = Field(
        2.67, description="Oxygen to carbon ratio in respiration"
    )
    o2n: Optional[float] = Field(
        4.33, description="Oxygen to ammonium ratio (g[O2]/g[NH4])"
    )
    dn2c: Optional[float] = Field(
        0.933,
        description="Mass of NO3 consumed per mass of DOC oxidized in denitrification (g[N]/g[C])",
    )
    an2c: Optional[float] = Field(
        0.5, description="Ratio of denitrification rate to oxic DOC respiration rate"
    )
    khdo: Optional[list] = Field(
        [0.5, 0.5, 0.5],
        description="DO half saturation for phytoplankton's DOC excretion (mg/L) for three species",
    )
    kpo4p: Optional[float] = Field(
        0.0, description="Coefficient relating PO4 sorption to TSS"
    )
    wrea: Optional[float] = Field(
        0.0, description="Baseline wind-induced reaeration coefficient for DO (day^-1)"
    )
    pbmin: Optional[list] = Field(
        [0.01, 0.01, 0.01],
        description="Minimum phytoplankton concentration (mg[C]/L) for three species",
    )
    dz_flux: Optional[list] = Field(
        [1.0, 1.0],
        description="Surface/bottom thickness (m) within which surface flux/bottom flux are redistributed",
    )

    @field_validator("gpm")
    @classmethod
    def validate_gpm(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("GPM must be a list of 3 positive numbers")
        return v

    @field_validator("tgp")
    @classmethod
    def validate_tgp(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 <= x <= 40 for x in v)
        ):
            raise ValueError("TGP must be a list of 3 numbers between 0 and 40")
        return v

    @field_validator("ktgp")
    @classmethod
    def validate_ktgp(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 6
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KTGP must be a list of 6 positive numbers")
        return v

    @field_validator("mtr")
    @classmethod
    def validate_mtr(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 <= x < 1 for x in v)
        ):
            raise ValueError("MTR must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("mtb")
    @classmethod
    def validate_mtb(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("MTB must be a list of 3 positive numbers")
        return v

    @field_validator("tmt")
    @classmethod
    def validate_tmt(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 <= x <= 40 for x in v)
        ):
            raise ValueError("TMT must be a list of 3 numbers between 0 and 40")
        return v

    @field_validator("ktmt")
    @classmethod
    def validate_ktmt(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KTMT must be a list of 3 positive numbers")
        return v

    @field_validator("fcp")
    @classmethod
    def validate_fcp(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 12
            or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("FCP must be a list of 12 numbers between 0 and 1")
        return v

    @field_validator("fnp")
    @classmethod
    def validate_fnp(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 15
            or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("FNP must be a list of 15 numbers between 0 and 1")
        return v

    @field_validator("fpp")
    @classmethod
    def validate_fpp(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 15
            or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("FPP must be a list of 15 numbers between 0 and 1")
        return v

    @field_validator("fcm")
    @classmethod
    def validate_fcm(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 12
            or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("FCM must be a list of 12 numbers between 0 and 1")
        return v

    @field_validator("fnm")
    @classmethod
    def validate_fnm(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 15
            or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("FNM must be a list of 15 numbers between 0 and 1")
        return v

    @field_validator("fpm")
    @classmethod
    def validate_fpm(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 15
            or not all(isinstance(x, (int, float)) and 0 <= x <= 1 for x in v)
        ):
            raise ValueError("FPM must be a list of 15 numbers between 0 and 1")
        return v

    @field_validator("nit")
    @classmethod
    def validate_nit(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("Nit must be a positive number")
        return v

    @field_validator("tnit")
    @classmethod
    def validate_tnit(cls, v):
        if not isinstance(v, (int, float)) or v < 0 or v > 40:
            raise ValueError("TNit must be a number between 0 and 40")
        return v

    @field_validator("ktnit")
    @classmethod
    def validate_ktnit(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 2
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KTNit must be a list of 2 positive numbers")
        return v

    @field_validator("khdon")
    @classmethod
    def validate_khdon(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("KhDOn must be a positive number")
        return v

    @field_validator("khdoox")
    @classmethod
    def validate_khdoox(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("KhDOox must be a positive number")
        return v

    @field_validator("khno3dn")
    @classmethod
    def validate_khno3dn(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("KhNO3dn must be a positive number")
        return v

    @field_validator("kc0")
    @classmethod
    def validate_kc0(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KC0 must be a list of 3 positive numbers")
        return v

    @field_validator("kn0")
    @classmethod
    def validate_kn0(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KN0 must be a list of 3 positive numbers")
        return v

    @field_validator("kp0")
    @classmethod
    def validate_kp0(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KP0 must be a list of 3 positive numbers")
        return v

    @field_validator("kcalg")
    @classmethod
    def validate_kcalg(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x >= 0 for x in v)
        ):
            raise ValueError("KCalg must be a list of 3 non-negative numbers")
        return v

    @field_validator("knalg")
    @classmethod
    def validate_knalg(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x >= 0 for x in v)
        ):
            raise ValueError("KNalg must be a list of 3 non-negative numbers")
        return v

    @field_validator("kpalg")
    @classmethod
    def validate_kpalg(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x >= 0 for x in v)
        ):
            raise ValueError("KPalg must be a list of 3 non-negative numbers")
        return v

    @field_validator("trm")
    @classmethod
    def validate_trm(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 <= x <= 40 for x in v)
        ):
            raise ValueError("TRM must be a list of 3 numbers between 0 and 40")
        return v

    @field_validator("ktrm")
    @classmethod
    def validate_ktrm(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KTRM must be a list of 3 positive numbers")
        return v

    @field_validator("ksr0")
    @classmethod
    def validate_ksr0(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KSR0 must be a list of 3 positive numbers")
        return v

    @field_validator("trsr")
    @classmethod
    def validate_trsr(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 <= x <= 40 for x in v)
        ):
            raise ValueError("TRSR must be a list of 3 numbers between 0 and 40")
        return v

    @field_validator("ktrsr")
    @classmethod
    def validate_ktrsr(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KTRSR must be a list of 3 positive numbers")
        return v

    @field_validator("kpip")
    @classmethod
    def validate_kpip(cls, v):
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("KPIP must be a non-negative number")
        return v

    @field_validator("kcd")
    @classmethod
    def validate_kcd(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("KCD must be a positive number")
        return v

    @field_validator("trcod")
    @classmethod
    def validate_trcod(cls, v):
        if not isinstance(v, (int, float)) or v < 0 or v > 40:
            raise ValueError("TRCOD must be a number between 0 and 40")
        return v

    @field_validator("ktrcod")
    @classmethod
    def validate_ktrcod(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("KTRCOD must be a positive number")
        return v

    @field_validator("khcod")
    @classmethod
    def validate_khcod(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("KhCOD must be a positive number")
        return v

    @field_validator("khn")
    @classmethod
    def validate_khn(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KhN must be a list of 3 positive numbers")
        return v

    @field_validator("khp")
    @classmethod
    def validate_khp(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KhP must be a list of 3 positive numbers")
        return v

    @field_validator("khsal")
    @classmethod
    def validate_khsal(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KhSal must be a list of 3 positive numbers")
        return v

    @field_validator("c2chl")
    @classmethod
    def validate_c2chl(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("c2chl must be a list of 3 positive numbers")
        return v

    @field_validator("n2c")
    @classmethod
    def validate_n2c(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v)
        ):
            raise ValueError("n2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("p2c")
    @classmethod
    def validate_p2c(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v)
        ):
            raise ValueError("p2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("o2c")
    @classmethod
    def validate_o2c(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("o2c must be a positive number")
        return v

    @field_validator("o2n")
    @classmethod
    def validate_o2n(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("o2n must be a positive number")
        return v

    @field_validator("dn2c")
    @classmethod
    def validate_dn2c(cls, v):
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("dn2c must be a positive number")
        return v

    @field_validator("an2c")
    @classmethod
    def validate_an2c(cls, v):
        if not isinstance(v, (int, float)) or v <= 0 or v > 1:
            raise ValueError("an2c must be a number between 0 and 1")
        return v

    @field_validator("khdo")
    @classmethod
    def validate_khdo(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("KhDO must be a list of 3 positive numbers")
        return v

    @field_validator("kpo4p")
    @classmethod
    def validate_kpo4p(cls, v):
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("KPO4p must be a non-negative number")
        return v

    @field_validator("wrea")
    @classmethod
    def validate_wrea(cls, v):
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("WRea must be a non-negative number")
        return v

    @field_validator("pbmin")
    @classmethod
    def validate_pbmin(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 3
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("PBmin must be a list of 3 positive numbers")
        return v

    @field_validator("dz_flux")
    @classmethod
    def validate_dz_flux(cls, v):
        if (
            not isinstance(v, list)
            or len(v) != 2
            or not all(isinstance(x, (int, float)) and x > 0 for x in v)
        ):
            raise ValueError("dz_flux must be a list of 2 positive numbers")
        return v

    @model_validator(mode="after")
    def validate_fcp_sum(self):
        for i in range(3):
            if sum(self.fcp[i::3]) != 1:
                raise ValueError(f"Sum of FCP fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fnp_sum(self):
        for i in range(3):
            if sum(self.fnp[i::3]) != 1:
                raise ValueError(f"Sum of FNP fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fpp_sum(self):
        for i in range(3):
            if sum(self.fpp[i::3]) != 1:
                raise ValueError(f"Sum of FPP fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fcm_sum(self):
        for i in range(3):
            if sum(self.fcm[i::3]) != 0.1:
                raise ValueError(
                    f"Sum of FCM fractions for species {i+1} must equal 0.1"
                )
        return self

    @model_validator(mode="after")
    def validate_fnm_sum(self):
        for i in range(5):
            if sum(self.fnm[i::3]) != 1:
                raise ValueError(f"Sum of FNM fractions for species {i+1} must equal 1")
        return self

    @model_validator(mode="after")
    def validate_fpm_sum(self):
        for i in range(5):
            if sum(self.fpm[i::3]) != 1:
                raise ValueError(f"Sum of FPM fractions for species {i+1} must equal 1")
        return self


class Sfm(NamelistBaseModel):
    btemp0: Optional[float] = Field(
        5.0, description="Initial temperature of the sediment layer in degrees Celsius"
    )
    bstc0: Optional[float] = Field(
        0.1, description="Initial surface transfer coefficient for the sediment layer"
    )
    bstr0: Optional[float] = Field(0.0, description="Initial benthic stress in days")
    bthp0: Optional[float] = Field(
        0.0, description="Initial consecutive days of hypoxia"
    )
    btox0: Optional[float] = Field(
        0.0,
        description="Initial consecutive days of oxic condition after hypoxia event",
    )
    bnh40: Optional[float] = Field(
        4.0, description="Initial NH4 concentration in the sediment layer (g/m3)"
    )
    bno30: Optional[float] = Field(
        1.0, description="Initial NO3 concentration in the sediment layer (g/m3)"
    )
    bpo40: Optional[float] = Field(
        5.0, description="Initial PO4 concentration in the sediment layer (g/m3)"
    )
    bh2s0: Optional[float] = Field(
        250.0, description="Initial H2S concentration in the sediment layer (g/m3)"
    )
    bch40: Optional[float] = Field(
        40.0, description="Initial CH4 concentration in the sediment layer (g/m3)"
    )
    bpos0: Optional[float] = Field(
        500.0,
        description="Initial POS (Particulate Organic Silica) concentration in the sediment layer (g/m3)",
    )
    bsa0: Optional[float] = Field(
        500.0,
        description="Initial SA (Salinity) concentration in the sediment layer (g/m3)",
    )
    bpoc0: Optional[list] = Field(
        [1000.0, 3000.0, 5000.0],
        description="Initial POC (Particulate Organic Carbon) concentrations for 3 classes (G1, G2, G3) in the sediment layer (g/m3)",
    )
    bpon0: Optional[list] = Field(
        [150.0, 500.0, 1500.0],
        description="Initial PON (Particulate Organic Nitrogen) concentrations for 3 classes (G1, G2, G3) in the sediment layer (g/m3)",
    )
    bpop0: Optional[list] = Field(
        [30.0, 300.0, 500.0],
        description="Initial POP (Particulate Organic Phosphorus) concentrations for 3 classes (G1, G2, G3) in the sediment layer (g/m3)",
    )
    bdz: Optional[float] = Field(0.1, description="Sediment thickness (m)")
    bvb: Optional[float] = Field(1.37e-05, description="Burial rate (m/day)")
    bsolid: Optional[list] = Field(
        [0.5, 0.5],
        description="Sediment solid concentrations in Layer 1 and Layer 2 (Kg/L)",
    )
    bdiff: Optional[float] = Field(
        1.8e-07, description="Diffusion coefficient for sediment temperature (m2/s)"
    )
    btr: Optional[int] = Field(
        20, description="Reference temperature for sediment processes (°C)"
    )
    bvpmin: Optional[float] = Field(
        3e-06, description="minimum particle mixing velocity coefficient (m.day-1)"
    )
    bvp: Optional[float] = Field(
        0.00012, description="particle mixing velocity coefficient (m.day-1)"
    )
    bvd: Optional[float] = Field(
        0.001, description="diffusion velocity coefficient (m.day-1)"
    )
    bktvp: Optional[float] = Field(
        1.117, description="temp. dependece of particle mixing velocity"
    )
    bktvd: Optional[float] = Field(
        1.08, description="temp. dependece of diffusion velocity"
    )
    bkst: Optional[float] = Field(
        0.03, description="1st order decay rate of benthic stress  (day-1)"
    )
    bstmax: Optional[float] = Field(
        20.0,
        description="maximum value of benthic stress (day) (note: smaller than 1/bKST)",
    )
    bkhdo_vp: Optional[float] = Field(
        4.0, description="DO half-saturation of particle mixing (mg/L)"
    )
    bdoc_st: Optional[float] = Field(
        1.0, description="DO criteria for benthic stress (mg/L)"
    )
    banoxic: Optional[float] = Field(
        10.0,
        description="consective days of hypoxia causing maximum benthic stress (day)",
    )
    boxic: Optional[float] = Field(
        45.0, description="time lag for bethos recovery from hypoxia event (day)"
    )
    bp2d: Optional[float] = Field(
        0.0,
        description="ratio from mixing coef. to diffusion coef. (benthos enhanced effect)",
    )
    bkc: Optional[list] = Field(
        [0.035, 0.0018, 0.0], description="decay rate of POC (3G class) at bTR (day-1)"
    )
    bkn: Optional[list] = Field(
        [0.035, 0.0018, 0.0], description="decay rate of PON (3G class) at bTR (day-1)"
    )
    bkp: Optional[list] = Field(
        [0.035, 0.0018, 0.0], description="decay rate of POP (3G class) at bTR (day-1)"
    )
    bktc: Optional[list] = Field(
        [1.1, 1.15, 1.17], description="temp. dependence of POC decay (oC-1)"
    )
    bktn: Optional[list] = Field(
        [1.1, 1.15, 1.17], description="temp. dependence of PON decay (oC-1)"
    )
    bktp: Optional[list] = Field(
        [1.1, 1.15, 1.17], description="temp. dependence of POP decay (oC-1)"
    )
    bfcp: Optional[list] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto POC into sed POC (G3,PB=1:3)",
    )
    bfnp: Optional[list] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto PON into sed PON (G3,PB=1:3)",
    )
    bfpp: Optional[list] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto POP into sed POP (G3,PB=1:3)",
    )
    bfcm: Optional[list] = Field(
        [0.0, 0.43, 0.57], description="refractory POC into sed POC(3G)"
    )
    bfnm: Optional[list] = Field(
        [0.0, 0.54, 0.46], description="refractory PON into sed PON(3G)"
    )
    bfpm: Optional[list] = Field(
        [0.0, 0.43, 0.57], description="refractory POP into sed POP(3G)"
    )
    bknh4f: Optional[float] = Field(
        0.2, description="NH4 reaction rate in freshwater  at bTR (1st layer) (m/day)"
    )
    bknh4s: Optional[float] = Field(
        0.14, description="NH4 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bktnh4: Optional[float] = Field(
        1.08, description="temp. dependency for NH4 reaction (oC-1)"
    )
    bkhnh4: Optional[float] = Field(
        1.5, description="half-stauration NH4 for nitrification (g/m3)"
    )
    bkhdo_nh4: Optional[float] = Field(
        2.0, description="half-stauration DO for nitrification (g/m3)"
    )
    bpienh4: Optional[float] = Field(
        1.0, description="partition coefficients of NH4 in Layer 1 & 2 (Kg-1.L)"
    )
    bsaltn: Optional[float] = Field(
        1.0,
        description="salinity criteria of fresh/salty water for NH4/NO3 reaction (PSU)",
    )
    bkno3f: Optional[float] = Field(
        0.3, description="NO3 reaction rate in freshwater at bTR (1st layer) (m/day)"
    )
    bkno3s: Optional[float] = Field(
        0.125, description="NO3 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bkno3: Optional[float] = Field(
        0.25, description="NO3 reaction rate (2nd layer) (m/day)"
    )
    bktno3: Optional[float] = Field(
        1.08, description="temp. dependency for NO3 reaction (oC-1)"
    )
    bkh2sd: Optional[float] = Field(
        0.2, description="dissolved H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bkh2sp: Optional[float] = Field(
        0.4, description="particulate H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bkth2s: Optional[float] = Field(
        1.08, description="temp. dependency for H2S reaction (oC-1)"
    )
    bpieh2ss: Optional[float] = Field(
        100.0, description="partition coefficient of NH4 in Layer 1 (Kg-1.L)"
    )
    bpieh2sb: Optional[float] = Field(
        100.0, description="partition coefficient of NH4 in Layer 2 (Kg-1.L)"
    )
    bkhdo_h2s: Optional[float] = Field(
        8.0, description="O2 constant to normalize H2S oxidation (g[O2]/m3)"
    )
    bsaltc: Optional[float] = Field(
        1.0,
        description="salinity criteria of fresh/salty water for carbon reaction (PSU)",
    )
    bkch4: Optional[float] = Field(
        0.2, description="CH4 reaction rate at bTR (1st layer) (m/day)"
    )
    bktch4: Optional[float] = Field(
        1.08, description="temp. dependency for CH4 reaction"
    )
    bkhdo_ch4: Optional[float] = Field(
        0.2, description="half-saturation DO for CH4 oxidation (g[O2]/m3)"
    )
    bo2n: Optional[float] = Field(
        2.86, description="oxygen to nitrogen ratio in sediment (denitrification)"
    )
    bpiepo4: Optional[float] = Field(
        50.0, description="partition coefficient of PO4 in Layer 2 (Kg-1.L)"
    )
    bkopo4f: Optional[float] = Field(
        3000.0,
        description="oxygen dependency for PO4 sorption in freshwater in Layer 1",
    )
    bkopo4s: Optional[float] = Field(
        300.0,
        description="oxygen dependency for PO4 sorption in salty water in Layer 1",
    )
    bdoc_po4: Optional[float] = Field(
        1.0, description="DO criteria for PO4 sorptiona (g[O2]/m3)"
    )
    bsaltp: Optional[float] = Field(
        1.0,
        description="salinity criteria of fresh/salty water for PO4 partition (PSU)",
    )
    bks: Optional[float] = Field(0.5, description="decay rate of POS (3G class) at bTR")
    bkts: Optional[float] = Field(1.1, description="temp. dependence of POS decay")
    bsisat: Optional[float] = Field(
        40.0, description="silica saturation conc. in pore water (g[Si]/m3)"
    )
    bpiesi: Optional[float] = Field(
        100.0, description="partition coefficient of silica in Layer 2 (Kg-1.L)"
    )
    bkosi: Optional[float] = Field(
        10.0, description="oxygen dependency for silica sorption in Layer 1"
    )
    bkhpos: Optional[float] = Field(
        50000.0, description="POS half saturation for POS dissolution (g/m3)"
    )
    bdoc_si: Optional[float] = Field(
        1.0, description="DO criteria for silica sorptiona (g[O2]/m3)"
    )
    bjposa: Optional[float] = Field(
        0.0,
        description="additional POS flux associated with POM detrius beside algea (g.m-2.day-1)",
    )
    bfcs: Optional[list] = Field(
        [0.65, 0.255, 0.095], description="SAV POC into 3G sed 3G POC"
    )
    bfns: Optional[list] = Field(
        [0.65, 0.3, 0.05], description="SAV PON into 3G sed 3G PON"
    )
    bfps: Optional[list] = Field(
        [0.65, 0.255, 0.095], description="SAV POP into 3G sed 3G POP"
    )
    bfcv: Optional[list] = Field(
        [0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095],
        description="VEG POC into sed POC (G3,PB=1:3)",
    )
    bfnv: Optional[list] = Field(
        [0.65, 0.3, 0.05, 0.65, 0.3, 0.05, 0.65, 0.3, 0.05],
        description="VEG PON into sed PON (G3,PB=1:3)",
    )
    bfpv: Optional[list] = Field(
        [0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095],
        description="VEG POP into sed POP (G3,PB=1:3)",
    )

    @field_validator("btemp0")
    @classmethod
    def validate_btemp0(cls, v):
        if v < -2 or v > 40:
            raise ValueError("btemp0 must be between -2 and 40")
        return v

    @field_validator("bstc0")
    @classmethod
    def validate_bstc0(cls, v):
        if v < 0:
            raise ValueError("bstc0 must be non-negative")
        return v

    @field_validator("bstr0")
    @classmethod
    def validate_bstr0(cls, v):
        if v < 0:
            raise ValueError("bstr0 must be non-negative")
        return v

    @field_validator("bthp0")
    @classmethod
    def validate_bthp0(cls, v):
        if v < 0:
            raise ValueError("bthp0 must be non-negative")
        return v

    @field_validator("btox0")
    @classmethod
    def validate_btox0(cls, v):
        if v < 0:
            raise ValueError("btox0 must be non-negative")
        return v

    @field_validator("bnh40")
    @classmethod
    def validate_bnh40(cls, v):
        if v < 0:
            raise ValueError("bnh40 must be non-negative")
        return v

    @field_validator("bno30")
    @classmethod
    def validate_bno30(cls, v):
        if v < 0:
            raise ValueError("bno30 must be non-negative")
        return v

    @field_validator("bpo40")
    @classmethod
    def validate_bpo40(cls, v):
        if v < 0:
            raise ValueError("bpo40 must be non-negative")
        return v

    @field_validator("bh2s0")
    @classmethod
    def validate_bh2s0(cls, v):
        if v < 0:
            raise ValueError("bh2s0 must be non-negative")
        return v

    @field_validator("bch40")
    @classmethod
    def validate_bch40(cls, v):
        if v < 0:
            raise ValueError("bch40 must be non-negative")
        return v

    @field_validator("bpos0")
    @classmethod
    def validate_bpos0(cls, v):
        if v < 0:
            raise ValueError("bpos0 must be non-negative")
        return v

    @field_validator("bsa0")
    @classmethod
    def validate_bsa0(cls, v):
        if v < 0:
            raise ValueError("bsa0 must be non-negative")
        return v

    @field_validator("bpoc0")
    @classmethod
    def validate_bpoc0(cls, v):
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("bpoc0 must be a list of 3 non-negative values")
        return v

    @field_validator("bpon0")
    @classmethod
    def validate_bpon0(cls, v):
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("bpon0 must be a list of 3 non-negative values")
        return v

    @field_validator("bpop0")
    @classmethod
    def validate_bpop0(cls, v):
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("bpop0 must be a list of 3 non-negative values")
        return v

    @field_validator("bdz")
    @classmethod
    def validate_bdz(cls, v):
        if v <= 0:
            raise ValueError("bdz must be positive")
        return v

    @field_validator("bvb")
    @classmethod
    def validate_bvb(cls, v):
        if v < 0:
            raise ValueError("bvb must be non-negative")
        return v

    @field_validator("bsolid")
    @classmethod
    def validate_bsolid(cls, v):
        if len(v) != 2 or any(x <= 0 or x > 1 for x in v):
            raise ValueError("bsolid must be a list of 2 values between 0 and 1")
        return v

    @field_validator("bdiff")
    @classmethod
    def validate_bdiff(cls, v):
        if v <= 0:
            raise ValueError("bdiff must be positive")
        return v

    @field_validator("btr")
    @classmethod
    def validate_btr(cls, v):
        if v < 0 or v > 40:
            raise ValueError("btr must be between 0 and 40")
        return v


class Silica(NamelistBaseModel):
    fsp: Optional[list] = Field(
        [0.9, 0.1], description="fractions of diatom silica into (SU,SA)"
    )
    fsm: Optional[list] = Field(
        [0.5, 0.5], description="fractions of diatom metabolism Si into (SU,SA)"
    )
    ks: Optional[float] = Field(
        0.03, description="dissolution rate of SU at TRS (day-1)"
    )
    trs: Optional[float] = Field(
        20.0, description="reference temp. for SU dissolution (oC)"
    )
    ktrs: Optional[float] = Field(
        0.092, description="temp. dependence for SU dissolution (oC-1)"
    )
    khs: Optional[list] = Field(
        [0.05, 0.0, 0.0],
        description="silica half saturation (mg/L); (0.0: no Si limitation)",
    )
    s2c: Optional[list] = Field(
        [0.5, 0.0, 0.0],
        description="silica to carbon ratio for phytolankton; (0.0: no Si uptake)",
    )
    ksap: Optional[float] = Field(
        0.0, description="coefficient relating Silicate(SA) sorption to TSS"
    )


class Zb(NamelistBaseModel):
    zgpm: Optional[list] = Field(
        [
            0.0,
            0.0,
            1.75,
            1.75,
            1.75,
            1.75,
            1.75,
            1.75,
            1.0,
            0.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
        ],
        description="Zooplankton predation rate (day^-1) for different prey types and zooplankton groups. Dimension: (prey=1:8, ZB=1:2)",
    )
    zkhg: Optional[list] = Field(
        [
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
            0.175,
        ],
        description="Reference prey concentration (mg/L) for zooplankton growth. Dimension: (prey=1:8, ZB=1:2)",
    )
    ztgp: Optional[list] = Field(
        [25.0, 25.0], description="Optimal temperature for zooplankton growth (°C)"
    )
    zktgp: Optional[list] = Field(
        [0.0035, 0.008, 0.025, 0.03],
        description="Temperature dependence for zooplankton growth (°C^-2). Dimension: (ZB=1:2, 1:2) for T<=zTGP & T>zTGP",
    )
    zag: Optional[float] = Field(
        0.75, description="Zooplankton assimilation efficiency ratio (0-1)"
    )
    zrg: Optional[float] = Field(
        0.1, description="Zooplankton respiration ratio when grazing (0-1)"
    )
    zmrt: Optional[list] = Field(
        [0.02, 0.02], description="Zooplankton mortality rates (day^-1)"
    )
    zmtb: Optional[list] = Field(
        [0.254, 0.186], description="Zooplankton metabolism rates (day^-1)"
    )
    ztmt: Optional[list] = Field(
        [20.0, 20.0],
        description="Reference temperature for zooplankton metabolism (°C)",
    )
    zktmt: Optional[list] = Field(
        [0.0693, 0.0693],
        description="Temperature dependence for zooplankton metabolism (°C^-1)",
    )
    zfcp: Optional[list] = Field(
        [0.35, 0.55, 0.1],
        description="Fractions of zooplankton carbon partitioned into (RPOC, LPOC, DOC)",
    )
    zfnp: Optional[list] = Field(
        [0.35, 0.5, 0.1, 0.05],
        description="Fractions of zooplankton nitrogen partitioned into (RPON, LPON, DON, NH4)",
    )
    zfpp: Optional[list] = Field(
        [0.1, 0.2, 0.5, 0.2],
        description="Fractions of zooplankton phosphorus partitioned into (RPOP, LPOP, DOP, PO4)",
    )
    zfsp: Optional[list] = Field(
        [0.7, 0.25],
        description="Fractions of zooplankton silica partitioned into (SU, SA)",
    )
    zfcm: Optional[list] = Field(
        [0.1, 0.0],
        description="Fractions of zooplankton metabolism carbon partitioned into DOC. Dimension: (ZB=1:2)",
    )
    zfnm: Optional[list] = Field(
        [0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1],
        description="Fractions of zooplankton metabolism nitrogen partitioned into (RPON, LPON, DON, NH4). Dimension: (ZB=1:2, 4)",
    )
    zfpm: Optional[list] = Field(
        [0.35, 0.3, 0.5, 0.4, 0.1, 0.2, 0.05, 0.1],
        description="Fractions of zooplankton metabolism phosphorus partitioned into (RPOP, LPOP, DOP, PO4). Dimension: (ZB=1:2, 4)",
    )
    zfsm: Optional[list] = Field(
        [0.5, 0.4, 0.5, 0.6],
        description="Fractions of zooplankton metabolism silica partitioned into (SU, SA). Dimension: (ZB=1:2, 2)",
    )
    zkhdo: Optional[list] = Field(
        [0.5, 0.5],
        description="Dissolved oxygen half-saturation for zooplankton's DOC excretion (mg/L)",
    )
    zn2c: Optional[list] = Field(
        [0.2, 0.2], description="Nitrogen to carbon ratio for zooplankton"
    )
    zp2c: Optional[list] = Field(
        [0.02, 0.02], description="Phosphorus to carbon ratio for zooplankton"
    )
    zs2c: Optional[list] = Field(
        [0.5, 0.5], description="Silica to carbon ratio for zooplankton"
    )
    z2pr: Optional[list] = Field(
        [0.5, 0.5],
        description="Ratio converting zooplankton and phytoplankton biomass to predation rates on zooplankton (L.mg^-1.day^-1)",
    )
    p2pr: Optional[float] = Field(
        0.25,
        description="Ratio converting zooplankton and phytoplankton biomass to predation rates on phytoplankton (L.mg^-1.day^-1)",
    )

    @field_validator("zgpm")
    @classmethod
    def validate_zgpm(cls, v):
        if not isinstance(v, list) or len(v) != 16:
            raise ValueError("zgpm must be a list of 16 float values")
        return [float(x) for x in v]

    @field_validator("zkhg")
    @classmethod
    def validate_zkhg(cls, v):
        if not isinstance(v, list) or len(v) != 16:
            raise ValueError("zkhg must be a list of 16 float values")
        return [float(x) for x in v]

    @field_validator("ztgp")
    @classmethod
    def validate_ztgp(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("ztgp must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zktgp")
    @classmethod
    def validate_zktgp(cls, v):
        if not isinstance(v, list) or len(v) != 4:
            raise ValueError("zktgp must be a list of 4 float values")
        return [float(x) for x in v]

    @field_validator("zag")
    @classmethod
    def validate_zag(cls, v):
        if not 0 <= float(v) <= 1:
            raise ValueError("zag must be between 0 and 1")
        return float(v)

    @field_validator("zrg")
    @classmethod
    def validate_zrg(cls, v):
        if not 0 <= float(v) <= 1:
            raise ValueError("zrg must be between 0 and 1")
        return float(v)

    @field_validator("zmrt")
    @classmethod
    def validate_zmrt(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zmrt must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zmtb")
    @classmethod
    def validate_zmtb(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zmtb must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("ztmt")
    @classmethod
    def validate_ztmt(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("ztmt must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zktmt")
    @classmethod
    def validate_zktmt(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zktmt must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zfcp")
    @classmethod
    def validate_zfcp(cls, v):
        if not isinstance(v, list) or len(v) != 3 or sum(v) != 1.0:
            raise ValueError("zfcp must be a list of 3 float values summing to 1.0")
        return [float(x) for x in v]

    @field_validator("zfnp")
    @classmethod
    def validate_zfnp(cls, v):
        if not isinstance(v, list) or len(v) != 4 or sum(v) != 1.0:
            raise ValueError("zfnp must be a list of 4 float values summing to 1.0")
        return [float(x) for x in v]

    @field_validator("zfpp")
    @classmethod
    def validate_zfpp(cls, v):
        if not isinstance(v, list) or len(v) != 4 or sum(v) != 1.0:
            raise ValueError("zfpp must be a list of 4 float values summing to 1.0")
        return [float(x) for x in v]

    @field_validator("zfsp")
    @classmethod
    def validate_zfsp(cls, v):
        if not isinstance(v, list) or len(v) != 2 or sum(v) != 1.0:
            raise ValueError("zfsp must be a list of 2 float values summing to 1.0")
        return [float(x) for x in v]

    @field_validator("zfcm")
    @classmethod
    def validate_zfcm(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zfcm must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zfnm")
    @classmethod
    def validate_zfnm(cls, v):
        if not isinstance(v, list) or len(v) != 8:
            raise ValueError("zfnm must be a list of 8 float values")
        return [float(x) for x in v]

    @field_validator("zfpm")
    @classmethod
    def validate_zfpm(cls, v):
        if not isinstance(v, list) or len(v) != 8:
            raise ValueError("zfpm must be a list of 8 float values")
        return [float(x) for x in v]

    @field_validator("zfsm")
    @classmethod
    def validate_zfsm(cls, v):
        if not isinstance(v, list) or len(v) != 4:
            raise ValueError("zfsm must be a list of 4 float values")
        return [float(x) for x in v]

    @field_validator("zkhdo")
    @classmethod
    def validate_zkhdo(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zkhdo must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zn2c")
    @classmethod
    def validate_zn2c(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zn2c must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zp2c")
    @classmethod
    def validate_zp2c(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zp2c must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("zs2c")
    @classmethod
    def validate_zs2c(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("zs2c must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("z2pr")
    @classmethod
    def validate_z2pr(cls, v):
        if not isinstance(v, list) or len(v) != 2:
            raise ValueError("z2pr must be a list of 2 float values")
        return [float(x) for x in v]

    @field_validator("p2pr")
    @classmethod
    def validate_p2pr(cls, v):
        return float(v)


class Ph_icm(NamelistBaseModel):
    ppatch0: Optional[int] = Field(
        -999,
        description="Region flag for pH modeling. If set to 1, pH modeling is enabled for all elements. If set to -999, spatial pH modeling is used.",
    )
    pkcaco3: Optional[float] = Field(
        60.0,
        description="Dissolution rate constant between calcium carbonate (CaCO3) and calcium ions (Ca++)",
    )
    pkca: Optional[float] = Field(
        60.0,
        description="Sediment surface transfer coefficient from calcium carbonate (CaCO3) to calcium ions (Ca++)",
    )
    prea: Optional[float] = Field(
        1.0, description="Reaeration rate for carbon dioxide (CO2)"
    )
    inu_ph: Optional[int] = Field(
        0,
        description="Nudge option for pH model. Controls whether pH values are adjusted during simulation.",
    )

    @field_validator("ppatch0")
    @classmethod
    def validate_ppatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("ppatch0 must be either 1 or -999")
        return v

    @field_validator("pkcaco3")
    @classmethod
    def validate_pkcaco3(cls, v):
        if v <= 0:
            raise ValueError("pkcaco3 must be positive")
        return v

    @field_validator("pkca")
    @classmethod
    def validate_pkca(cls, v):
        if v <= 0:
            raise ValueError("pkca must be positive")
        return v

    @field_validator("prea")
    @classmethod
    def validate_prea(cls, v):
        if v <= 0:
            raise ValueError("prea must be positive")
        return v

    @field_validator("inu_ph")
    @classmethod
    def validate_inu_ph(cls, v):
        if v not in [0, 1]:
            raise ValueError("inu_ph must be either 0 or 1")
        return v


class Sav(NamelistBaseModel):
    spatch0: Optional[int] = Field(
        -999,
        description="Region flag for SAV. 1 indicates SAV is active on all elements, -999 indicates spatial distribution",
    )
    stleaf0: Optional[int] = Field(
        -999, description="Initial concentration of total SAV leaf biomass"
    )
    ststem0: Optional[int] = Field(
        -999, description="Initial concentration of total SAV stem biomass"
    )
    stroot0: Optional[int] = Field(
        -999, description="Initial concentration of total SAV root biomass"
    )
    sgpm: Optional[float] = Field(0.1, description="Maximum growth rate of SAV per day")
    stgp: Optional[int] = Field(
        32, description="Optimal growth temperature for SAV in degrees Celsius"
    )
    sktgp: Optional[list] = Field(
        [0.003, 0.005],
        description="Temperature dependence coefficients for SAV growth (for T<=sTGP and T>sTGP)",
    )
    sfam: Optional[float] = Field(
        0.2,
        description="Fraction of SAV leaf production allocated to active metabolism",
    )
    sfcp: Optional[list] = Field(
        [0.6, 0.3, 0.1],
        description="Fractions of SAV production allocated to leaf, stem, and root biomass",
    )
    smtb: Optional[list] = Field(
        [0.02, 0.02, 0.02], description="Metabolism rates of SAV leaf, stem, and root"
    )
    stmt: Optional[list] = Field(
        [20, 20, 20],
        description="Reference temperatures for SAV leaf, stem, and root metabolism in degrees Celsius",
    )
    sktmt: Optional[list] = Field(
        [0.069, 0.069, 0.069],
        description="Temperature dependence coefficients for SAV leaf, stem, and root metabolism",
    )
    sfcm: Optional[list] = Field(
        [0.05, 0.15, 0.3, 0.5],
        description="Fractions of SAV metabolism carbon allocated to RPOC, LPOC, DOC, and CO2",
    )
    sfnm: Optional[list] = Field(
        [0.05, 0.15, 0.3, 0.5],
        description="Fractions of SAV metabolism nitrogen allocated to RPON, LPON, DON, and NH4",
    )
    sfpm: Optional[list] = Field(
        [0.05, 0.1, 0.35, 0.5],
        description="Fractions of SAV metabolism phosphorus allocated to RPOP, LPOP, DOP, and PO4",
    )
    skhnw: Optional[float] = Field(
        0.01, description="Nitrogen half-saturation constant for SAV in water column"
    )
    skhns: Optional[float] = Field(
        0.1, description="Nitrogen half-saturation constant for SAV in sediments"
    )
    skhnh4: Optional[float] = Field(
        0.1, description="Ammonium half-saturation constant for SAV"
    )
    skhpw: Optional[float] = Field(
        0.001, description="Phosphorus half-saturation constant for SAV in water column"
    )
    skhps: Optional[float] = Field(
        0.01, description="Phosphorus half-saturation constant for SAV in sediments"
    )
    salpha: Optional[float] = Field(
        0.006, description="Initial slope of the SAV photosynthesis-irradiance curve"
    )
    ske: Optional[float] = Field(
        0.045, description="Light attenuation coefficient due to SAV absorption"
    )
    shtm: Optional[list] = Field(
        [0.054, 2.0], description="Minimum (base) and maximum SAV canopy height"
    )
    s2ht: Optional[list] = Field(
        [0.0036, 0.0036, 0.0],
        description="Coefficients for converting SAV leaf, stem, and root biomass to canopy height",
    )
    sc2dw: Optional[float] = Field(
        0.38, description="Carbon to dry weight ratio of SAV"
    )
    s2den: Optional[int] = Field(
        10, description="Coefficient for computing SAV density from leaf biomass"
    )

    @field_validator("spatch0")
    @classmethod
    def validate_spatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("spatch0 must be either 1 or -999")
        return v

    @field_validator("stleaf0")
    @classmethod
    def validate_stleaf0(cls, v):
        if v < 0 and v != -999:
            raise ValueError("stleaf0 must be non-negative or -999")
        return v

    @field_validator("ststem0")
    @classmethod
    def validate_ststem0(cls, v):
        if v < 0 and v != -999:
            raise ValueError("ststem0 must be non-negative or -999")
        return v

    @field_validator("stroot0")
    @classmethod
    def validate_stroot0(cls, v):
        if v < 0 and v != -999:
            raise ValueError("stroot0 must be non-negative or -999")
        return v

    @field_validator("sgpm")
    @classmethod
    def validate_sgpm(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("sgpm must be between 0 and 1")
        return v

    @field_validator("stgp")
    @classmethod
    def validate_stgp(cls, v):
        if v < 0 or v > 50:
            raise ValueError("stgp must be between 0 and 50")
        return v

    @field_validator("sktgp")
    @classmethod
    def validate_sktgp(cls, v):
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError("sktgp must be a list of two positive numbers")
        return v

    @field_validator("sfam")
    @classmethod
    def validate_sfam(cls, v):
        if v < 0 or v > 1:
            raise ValueError("sfam must be between 0 and 1")
        return v

    @field_validator("sfcp")
    @classmethod
    def validate_sfcp(cls, v):
        if len(v) != 3 or sum(v) != 1 or any(x < 0 for x in v):
            raise ValueError(
                "sfcp must be a list of three non-negative numbers that sum to 1"
            )
        return v

    @field_validator("smtb")
    @classmethod
    def validate_smtb(cls, v):
        if len(v) != 3 or any(x <= 0 for x in v):
            raise ValueError("smtb must be a list of three positive numbers")
        return v

    @field_validator("stmt")
    @classmethod
    def validate_stmt(cls, v):
        if len(v) != 3 or any(x < 0 or x > 50 for x in v):
            raise ValueError("stmt must be a list of three numbers between 0 and 50")
        return v

    @field_validator("sktmt")
    @classmethod
    def validate_sktmt(cls, v):
        if len(v) != 3 or any(x <= 0 for x in v):
            raise ValueError("sktmt must be a list of three positive numbers")
        return v

    @field_validator("sfcm")
    @classmethod
    def validate_sfcm(cls, v):
        if len(v) != 4 or sum(v) != 1 or any(x < 0 for x in v):
            raise ValueError(
                "sfcm must be a list of four non-negative numbers that sum to 1"
            )
        return v

    @field_validator("sfnm")
    @classmethod
    def validate_sfnm(cls, v):
        if len(v) != 4 or sum(v) != 1 or any(x < 0 for x in v):
            raise ValueError(
                "sfnm must be a list of four non-negative numbers that sum to 1"
            )
        return v

    @field_validator("sfpm")
    @classmethod
    def validate_sfpm(cls, v):
        if len(v) != 4 or sum(v) != 1 or any(x < 0 for x in v):
            raise ValueError(
                "sfpm must be a list of four non-negative numbers that sum to 1"
            )
        return v

    @field_validator("skhnw")
    @classmethod
    def validate_skhnw(cls, v):
        if v <= 0:
            raise ValueError("skhnw must be positive")
        return v

    @field_validator("skhns")
    @classmethod
    def validate_skhns(cls, v):
        if v <= 0:
            raise ValueError("skhns must be positive")
        return v

    @field_validator("skhnh4")
    @classmethod
    def validate_skhnh4(cls, v):
        if v <= 0:
            raise ValueError("skhnh4 must be positive")
        return v

    @field_validator("skhpw")
    @classmethod
    def validate_skhpw(cls, v):
        if v <= 0:
            raise ValueError("skhpw must be positive")
        return v

    @field_validator("skhps")
    @classmethod
    def validate_skhps(cls, v):
        if v <= 0:
            raise ValueError("skhps must be positive")
        return v

    @field_validator("salpha")
    @classmethod
    def validate_salpha(cls, v):
        if v <= 0:
            raise ValueError("salpha must be positive")
        return v

    @field_validator("ske")
    @classmethod
    def validate_ske(cls, v):
        if v <= 0:
            raise ValueError("ske must be positive")
        return v

    @field_validator("shtm")
    @classmethod
    def validate_shtm(cls, v):
        if len(v) != 2 or v[0] >= v[1] or any(x <= 0 for x in v):
            raise ValueError(
                "shtm must be a list of two positive numbers with the first less than the second"
            )
        return v

    @field_validator("s2ht")
    @classmethod
    def validate_s2ht(cls, v):
        if len(v) != 3 or any(x < 0 for x in v):
            raise ValueError("s2ht must be a list of three non-negative numbers")
        return v

    @field_validator("sc2dw")
    @classmethod
    def validate_sc2dw(cls, v):
        if v <= 0 or v > 1:
            raise ValueError("sc2dw must be between 0 and 1")
        return v

    @field_validator("s2den")
    @classmethod
    def validate_s2den(cls, v):
        if v <= 0:
            raise ValueError("s2den must be positive")
        return v


class Stem(NamelistBaseModel):
    sn2c: Optional[float] = Field(
        0.09,
        description="Nitrogen to carbon ratio of submerged aquatic vegetation (SAV)",
    )
    sp2c: Optional[float] = Field(0.01, description="Phosphorus to carbon ratio")
    so2c: Optional[float] = Field(2.67, description="Oxygen to carbon ratio")

    @field_validator("sn2c")
    @classmethod
    def validate_sn2c(cls, v):
        if v < 0 or v > 1:
            raise ValueError("sn2c must be between 0 and 1")
        return v

    @field_validator("sp2c")
    @classmethod
    def validate_sp2c(cls, v):
        if v < 0 or v > 1:
            raise ValueError("sp2c must be between 0 and 1")
        return v

    @field_validator("so2c")
    @classmethod
    def validate_so2c(cls, v):
        if v <= 0:
            raise ValueError("so2c must be greater than 0")
        return v


class Veg(NamelistBaseModel):
    vpatch0: Optional[int] = Field(
        -999,
        description="Region flag for VEG. (1: ON for all elements; -999: spatial distribution)",
    )
    vtleaf0: Optional[list] = Field(
        [100.0, 100.0, 100.0],
        description="Initial concentration for total vegetation leaf biomass (3 values for different vegetation types)",
    )
    vtstem0: Optional[list] = Field(
        [100.0, 100.0, 100.0],
        description="Initial concentration for total vegetation stem biomass (3 values for different vegetation types)",
    )
    vtroot0: Optional[list] = Field(
        [30.0, 30.0, 30.0],
        description="Initial concentration for total vegetation root biomass (3 values for different vegetation types)",
    )
    vgpm: Optional[list] = Field(
        [0.1, 0.1, 0.1],
        description="Maximum growth rate for vegetation (day^-1) (3 values for different vegetation types)",
    )
    vfam: Optional[list] = Field(
        [0.2, 0.2, 0.2],
        description="Fractions of leaf production allocated to active metabolism (3 values for different vegetation types)",
    )
    vtgp: Optional[list] = Field(
        [32.0, 32.0, 32.0],
        description="Optimal growth temperature for vegetation (°C) (3 values for different vegetation types)",
    )
    vktgp: Optional[list] = Field(
        [0.003, 0.003, 0.003, 0.005, 0.005, 0.005],
        description="Temperature dependence coefficients for growth (6 values: 3 for T<=vTGP and 3 for T>vTGP, for different vegetation types)",
    )
    vfcp: Optional[list] = Field(
        [0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1],
        description="Fractions of production allocated to leaf/stem/root biomass (9 values: 3 for each vegetation type, 3 for each biomass component)",
    )
    vmtb: Optional[list] = Field(
        [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01],
        description="Metabolism rates for leaf/stem/root (9 values: 3 for each vegetation type, 3 for each biomass component)",
    )
    vtmt: Optional[list] = Field(
        [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        description="Reference temperatures for leaf/stem/root metabolism (9 values: 3 for each vegetation type, 3 for each biomass component)",
    )
    vktmt: Optional[list] = Field(
        [0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069],
        description="Temperature dependence coefficients for leaf/stem/root metabolism (9 values: 3 for each vegetation type, 3 for each biomass component)",
    )
    vfnm: Optional[list] = Field(
        [0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
        description="Fractions of metabolism N into RPON, LPON, DON, NH4 (12 values: 3 for each vegetation type, 4 for each N form)",
    )
    vfpm: Optional[list] = Field(
        [0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5],
        description="Fractions of metabolism P into RPOP, LPOP, DOP, PO4 (12 values: 3 for each vegetation type, 4 for each P form)",
    )
    vfcm: Optional[list] = Field(
        [0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
        description="Fractions of metabolism C into RPOC, LPOC, DOC, CO2 (12 values: 3 for each vegetation type, 4 for each C form)",
    )
    ivnc: Optional[int] = Field(
        1,
        description="Flag for recycled vegetation N destination (0: sediment; 1: water)",
    )
    ivpc: Optional[int] = Field(
        1,
        description="Flag for recycled vegetation P destination (0: sediment; 1: water)",
    )
    vkhns: Optional[list] = Field(
        [0.1, 0.1, 0.1],
        description="Nitrogen half-saturation constant in sediments (3 values for different vegetation types)",
    )
    vkhps: Optional[list] = Field(
        [0.01, 0.01, 0.01],
        description="Phosphorus half-saturation constant in sediments (3 values for different vegetation types)",
    )
    vscr: Optional[list] = Field(
        [35.0, 35.0, 35.0],
        description="Reference salinity for computing vegetation growth (3 values for different vegetation types)",
    )
    vsopt: Optional[list] = Field(
        [35.0, 15.0, 0.0],
        description="Optimal salinity for vegetation growth (3 values for different vegetation types)",
    )
    vinun: Optional[list] = Field(
        [1.0, 1.0, 1.0],
        description="Reference value for inundation stress (nondimensional) (3 values for different vegetation types)",
    )
    ivns: Optional[int] = Field(
        1, description="Flag for N limitation on vegetation growth (0: OFF; 1: ON)"
    )
    ivps: Optional[int] = Field(
        1, description="Flag for P limitation on vegetation growth (0: OFF; 1: ON)"
    )
    ivmrt: Optional[int] = Field(
        0, description="Flag for vegetation mortality term (0: OFF; 1: ON)"
    )
    vtmr: Optional[list] = Field(
        [17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
        description="Reference temperatures for leaf/stem mortality (6 values: 3 for each vegetation type, 2 for leaf/stem)",
    )
    vktmr: Optional[list] = Field(
        [4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        description="Temperature dependence coefficients for leaf/stem mortality (6 values: 3 for each vegetation type, 2 for leaf/stem)",
    )
    vmr0: Optional[list] = Field(
        [12.8, 12.8, 12.8, 12.8, 12.8, 12.8],
        description="Base value of temperature effect on mortality (6 values: 3 for each vegetation type, 2 for leaf/stem)",
    )
    vmrcr: Optional[list] = Field(
        [15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        description="Reference value for computing mortality (6 values: 3 for each vegetation type, 2 for leaf/stem)",
    )
    valpha: Optional[list] = Field(
        [0.006, 0.006, 0.006],
        description="Initial slope of P-I curve (3 values for different vegetation types)",
    )
    vke: Optional[list] = Field(
        [0.045, 0.045, 0.045],
        description="Light attenuation coefficient from vegetation absorption (3 values for different vegetation types)",
    )
    vht0: Optional[list] = Field(
        [0.054, 0.054, 0.054],
        description="Base vegetation canopy height (3 values for different vegetation types)",
    )
    vcrit: Optional[list] = Field(
        [250.0, 250.0, 250.0],
        description="Critical mass for computing vegetation height (3 values for different vegetation types)",
    )
    v2ht: Optional[list] = Field(
        [0.0036, 0.0036, 0.0036, 0.001, 0.001, 0.001],
        description="Coefficients to convert mass to canopy height (6 values: 3 for each vegetation type, 2 for different conditions)",
    )
    vc2dw: Optional[list] = Field(
        [0.38, 0.38, 0.38],
        description="Carbon to dry weight ratio of vegetation (3 values for different vegetation types)",
    )
    v2den: Optional[list] = Field(
        [10, 10, 10],
        description="Coefficient for computing vegetation density (3 values for different vegetation types)",
    )
    vp2c: Optional[list] = Field(
        [0.01, 0.01, 0.01],
        description="Phosphorus to carbon ratio in vegetation (3 values for different vegetation types)",
    )
    vn2c: Optional[list] = Field(
        [0.09, 0.09, 0.09],
        description="Nitrogen to carbon ratio in vegetation (3 values for different vegetation types)",
    )
    vo2c: Optional[list] = Field(
        [2.67, 2.67, 2.67],
        description="Oxygen to carbon ratio in vegetation (3 values for different vegetation types)",
    )

    @field_validator("vpatch0")
    @classmethod
    def check_vpatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("vpatch0 must be either 1 or -999")
        return v

    @field_validator("vtleaf0")
    @classmethod
    def check_vtleaf0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vtleaf0 must be a list of 3 positive numbers")
        return v

    @field_validator("vtstem0")
    @classmethod
    def check_vtstem0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vtstem0 must be a list of 3 positive numbers")
        return v

    @field_validator("vtroot0")
    @classmethod
    def check_vtroot0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vtroot0 must be a list of 3 positive numbers")
        return v

    @field_validator("vgpm")
    @classmethod
    def check_vgpm(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vgpm must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("vfam")
    @classmethod
    def check_vfam(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vfam must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("vtgp")
    @classmethod
    def check_vtgp(cls, v):
        if len(v) != 3 or not all(
            isinstance(x, (int, float)) and 0 < x < 50 for x in v
        ):
            raise ValueError("vtgp must be a list of 3 numbers between 0 and 50")
        return v

    @field_validator("vktgp")
    @classmethod
    def check_vktgp(cls, v):
        if len(v) != 6 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vktgp must be a list of 6 positive numbers")
        return v

    @field_validator("vfcp")
    @classmethod
    def check_vfcp(cls, v):
        if len(v) != 9 or not all(
            isinstance(x, (int, float)) and 0 <= x <= 1 for x in v
        ):
            raise ValueError("vfcp must be a list of 9 numbers between 0 and 1")
        return v

    @field_validator("vmtb")
    @classmethod
    def check_vmtb(cls, v):
        if len(v) != 9 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vmtb must be a list of 9 positive numbers")
        return v

    @field_validator("vtmt")
    @classmethod
    def check_vtmt(cls, v):
        if len(v) != 9 or not all(
            isinstance(x, (int, float)) and 0 < x < 50 for x in v
        ):
            raise ValueError("vtmt must be a list of 9 numbers between 0 and 50")
        return v

    @field_validator("vktmt")
    @classmethod
    def check_vktmt(cls, v):
        if len(v) != 9 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vktmt must be a list of 9 positive numbers")
        return v

    @field_validator("vfnm")
    @classmethod
    def check_vfnm(cls, v):
        if len(v) != 12 or not all(
            isinstance(x, (int, float)) and 0 <= x <= 1 for x in v
        ):
            raise ValueError("vfnm must be a list of 12 numbers between 0 and 1")
        return v

    @field_validator("vfpm")
    @classmethod
    def check_vfpm(cls, v):
        if len(v) != 12 or not all(
            isinstance(x, (int, float)) and 0 <= x <= 1 for x in v
        ):
            raise ValueError("vfpm must be a list of 12 numbers between 0 and 1")
        return v

    @field_validator("vfcm")
    @classmethod
    def check_vfcm(cls, v):
        if len(v) != 12 or not all(
            isinstance(x, (int, float)) and 0 <= x <= 1 for x in v
        ):
            raise ValueError("vfcm must be a list of 12 numbers between 0 and 1")
        return v

    @field_validator("ivnc")
    @classmethod
    def check_ivnc(cls, v):
        if v not in [0, 1]:
            raise ValueError("ivnc must be either 0 or 1")
        return v

    @field_validator("ivpc")
    @classmethod
    def check_ivpc(cls, v):
        if v not in [0, 1]:
            raise ValueError("ivpc must be either 0 or 1")
        return v

    @field_validator("vkhns")
    @classmethod
    def check_vkhns(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vkhns must be a list of 3 positive numbers")
        return v

    @field_validator("vkhps")
    @classmethod
    def check_vkhps(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vkhps must be a list of 3 positive numbers")
        return v

    @field_validator("vscr")
    @classmethod
    def check_vscr(cls, v):
        if len(v) != 3 or not all(
            isinstance(x, (int, float)) and 0 <= x <= 40 for x in v
        ):
            raise ValueError("vscr must be a list of 3 numbers between 0 and 40")
        return v

    @field_validator("vsopt")
    @classmethod
    def check_vsopt(cls, v):
        if len(v) != 3 or not all(
            isinstance(x, (int, float)) and 0 <= x <= 40 for x in v
        ):
            raise ValueError("vsopt must be a list of 3 numbers between 0 and 40")
        return v

    @field_validator("vinun")
    @classmethod
    def check_vinun(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vinun must be a list of 3 positive numbers")
        return v

    @field_validator("ivns")
    @classmethod
    def check_ivns(cls, v):
        if v not in [0, 1]:
            raise ValueError("ivns must be either 0 or 1")
        return v

    @field_validator("ivps")
    @classmethod
    def check_ivps(cls, v):
        if v not in [0, 1]:
            raise ValueError("ivps must be either 0 or 1")
        return v

    @field_validator("ivmrt")
    @classmethod
    def check_ivmrt(cls, v):
        if v not in [0, 1]:
            raise ValueError("ivmrt must be either 0 or 1")
        return v

    @field_validator("vtmr")
    @classmethod
    def check_vtmr(cls, v):
        if len(v) != 6 or not all(
            isinstance(x, (int, float)) and 0 < x < 50 for x in v
        ):
            raise ValueError("vtmr must be a list of 6 numbers between 0 and 50")
        return v

    @field_validator("vktmr")
    @classmethod
    def check_vktmr(cls, v):
        if len(v) != 6 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vktmr must be a list of 6 positive numbers")
        return v

    @field_validator("vmr0")
    @classmethod
    def check_vmr0(cls, v):
        if len(v) != 6 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vmr0 must be a list of 6 positive numbers")
        return v

    @field_validator("vmrcr")
    @classmethod
    def check_vmrcr(cls, v):
        if len(v) != 6 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vmrcr must be a list of 6 positive numbers")
        return v

    @field_validator("valpha")
    @classmethod
    def check_valpha(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("valpha must be a list of 3 positive numbers")
        return v

    @field_validator("vke")
    @classmethod
    def check_vke(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vke must be a list of 3 positive numbers")
        return v

    @field_validator("vht0")
    @classmethod
    def check_vht0(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vht0 must be a list of 3 positive numbers")
        return v

    @field_validator("vcrit")
    @classmethod
    def check_vcrit(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vcrit must be a list of 3 positive numbers")
        return v

    @field_validator("v2ht")
    @classmethod
    def check_v2ht(cls, v):
        if len(v) != 6 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("v2ht must be a list of 6 positive numbers")
        return v

    @field_validator("vc2dw")
    @classmethod
    def check_vc2dw(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vc2dw must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("v2den")
    @classmethod
    def check_v2den(cls, v):
        if len(v) != 3 or not all(isinstance(x, int) and x > 0 for x in v):
            raise ValueError("v2den must be a list of 3 positive integers")
        return v

    @field_validator("vp2c")
    @classmethod
    def check_vp2c(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vp2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("vn2c")
    @classmethod
    def check_vn2c(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and 0 < x < 1 for x in v):
            raise ValueError("vn2c must be a list of 3 numbers between 0 and 1")
        return v

    @field_validator("vo2c")
    @classmethod
    def check_vo2c(cls, v):
        if len(v) != 3 or not all(isinstance(x, (int, float)) and x > 0 for x in v):
            raise ValueError("vo2c must be a list of 3 positive numbers")
        return v

    @model_validator(mode="after")
    def check_vfcp_sum(self):
        for i in range(3):
            if abs(sum(self.vfcp[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfcp values for vegetation type {i+1} must be 1"
                )
        return self

    @model_validator(mode="after")
    def check_vfnm_sum(self):
        for i in range(3):
            if abs(sum(self.vfnm[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfnm values for vegetation type {i+1} must be 1"
                )
        return self

    @model_validator(mode="after")
    def check_vfpm_sum(self):
        for i in range(3):
            if abs(sum(self.vfpm[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfpm values for vegetation type {i+1} must be 1"
                )
        return self

    @model_validator(mode="after")
    def check_vfcm_sum(self):
        for i in range(3):
            if abs(sum(self.vfcm[i::3]) - 1) > 1e-6:
                raise ValueError(
                    f"Sum of vfcm values for vegetation type {i+1} must be 1"
                )
        return self


class Bag(NamelistBaseModel):
    gpatch0: Optional[int] = Field(
        -999,
        description="Region flag for Benthic Algae (BA). 1 indicates BA is ON for all elements, -999 indicates spatial distribution.",
    )
    ba0: Optional[float] = Field(
        5.0,
        description="Initial Benthic Algae (BA) concentration in grams of carbon per square meter.",
    )
    ggpm: Optional[float] = Field(
        2.25, description="Maximum growth rate for Benthic Algae (BA) per day."
    )
    gtgp: Optional[float] = Field(
        20.0,
        description="Optimal temperature for Benthic Algae (BA) growth in degrees Celsius.",
    )
    gktgp: Optional[list] = Field(
        [0.004, 0.006],
        description="Temperature dependence coefficients for Benthic Algae (BA) growth in inverse degrees Celsius squared.",
    )
    gmtb: Optional[float] = Field(
        0.05,
        description="Respiration rate for Benthic Algae (BA) at the reference temperature gTR, per day.",
    )
    gprr: Optional[float] = Field(
        0.1,
        description="Predation rate for Benthic Algae (BA) at the reference temperature gTR, per day.",
    )
    gtr: Optional[float] = Field(
        20.0,
        description="Reference temperature for Benthic Algae (BA) respiration in degrees Celsius.",
    )
    gktr: Optional[float] = Field(
        0.069,
        description="Temperature dependence coefficient for Benthic Algae (BA) respiration in inverse degrees Celsius.",
    )
    galpha: Optional[float] = Field(
        0.1,
        description="Initial slope of the Photosynthesis-Irradiance (P-I) curve for Benthic Algae (BA) in square meters per Einstein.",
    )
    gksed: Optional[float] = Field(
        0.0,
        description="Light attenuation due to sediment for Benthic Algae (BA) growth (dimensionless).",
    )
    gkba: Optional[float] = Field(
        0.01,
        description="Light attenuation coefficient due to Benthic Algae (BA) self-shading in square meters per gram of carbon.",
    )
    gkhn: Optional[float] = Field(
        0.01,
        description="Nitrogen half-saturation constant for Benthic Algae (BA) growth in grams of nitrogen per square meter.",
    )
    gkhp: Optional[float] = Field(
        0.001,
        description="Phosphorus half-saturation constant for Benthic Algae (BA) growth in grams of phosphorus per square meter.",
    )
    gp2c: Optional[float] = Field(
        0.0167, description="Phosphorus to carbon ratio in Benthic Algae (BA) biomass."
    )
    gn2c: Optional[float] = Field(
        0.167, description="Nitrogen to carbon ratio in Benthic Algae (BA) biomass."
    )
    go2c: Optional[float] = Field(
        2.67, description="Oxygen to carbon ratio for Benthic Algae (BA) respiration."
    )
    gfcp: Optional[list] = Field(
        [0.5, 0.45, 0.05],
        description="Fractions of predated Benthic Algae (BA) carbon distributed into three classes in sediment.",
    )
    gfnp: Optional[list] = Field(
        [0.5, 0.45, 0.05],
        description="Fractions of predated Benthic Algae (BA) nitrogen distributed into three classes in sediment.",
    )
    gfpp: Optional[list] = Field(
        [0.5, 0.45, 0.05],
        description="Fractions of predated Benthic Algae (BA) phosphorus distributed into three classes in sediment.",
    )

    @field_validator("gpatch0")
    @classmethod
    def check_gpatch0(cls, v):
        if v not in [1, -999]:
            raise ValueError("gpatch0 must be either 1 or -999")
        return v

    @field_validator("ba0")
    @classmethod
    def check_ba0(cls, v):
        if v <= 0:
            raise ValueError("BA0 must be positive")
        return v

    @field_validator("ggpm")
    @classmethod
    def check_ggpm(cls, v):
        if v <= 0:
            raise ValueError("gGPM must be positive")
        return v

    @field_validator("gtgp")
    @classmethod
    def check_gtgp(cls, v):
        if v < 0 or v > 100:
            raise ValueError("gTGP must be between 0 and 100")
        return v

    @field_validator("gktgp")
    @classmethod
    def check_gktgp(cls, v):
        if len(v) != 2 or any(x <= 0 for x in v):
            raise ValueError("gKTGP must be a list of two positive values")
        return v

    @field_validator("gmtb")
    @classmethod
    def check_gmtb(cls, v):
        if v < 0 or v > 1:
            raise ValueError("gMTB must be between 0 and 1")
        return v

    @field_validator("gprr")
    @classmethod
    def check_gprr(cls, v):
        if v < 0 or v > 1:
            raise ValueError("gPRR must be between 0 and 1")
        return v

    @field_validator("gtr")
    @classmethod
    def check_gtr(cls, v):
        if v < 0 or v > 100:
            raise ValueError("gTR must be between 0 and 100")
        return v

    @field_validator("gktr")
    @classmethod
    def check_gktr(cls, v):
        if v <= 0:
            raise ValueError("gKTR must be positive")
        return v

    @field_validator("galpha")
    @classmethod
    def check_galpha(cls, v):
        if v <= 0:
            raise ValueError("galpha must be positive")
        return v

    @field_validator("gksed")
    @classmethod
    def check_gksed(cls, v):
        if v < 0:
            raise ValueError("gKSED must be non-negative")
        return v

    @field_validator("gkba")
    @classmethod
    def check_gkba(cls, v):
        if v <= 0:
            raise ValueError("gKBA must be positive")
        return v

    @field_validator("gkhn")
    @classmethod
    def check_gkhn(cls, v):
        if v <= 0:
            raise ValueError("gKhN must be positive")
        return v

    @field_validator("gkhp")
    @classmethod
    def check_gkhp(cls, v):
        if v <= 0:
            raise ValueError("gKhP must be positive")
        return v

    @field_validator("gp2c")
    @classmethod
    def check_gp2c(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("gp2c must be between 0 and 1")
        return v

    @field_validator("gn2c")
    @classmethod
    def check_gn2c(cls, v):
        if v <= 0 or v >= 1:
            raise ValueError("gn2c must be between 0 and 1")
        return v

    @field_validator("go2c")
    @classmethod
    def check_go2c(cls, v):
        if v <= 0:
            raise ValueError("go2c must be positive")
        return v

    @field_validator("gfcp")
    @classmethod
    def check_gfcp(cls, v):
        if len(v) != 3 or any(x < 0 or x > 1 for x in v) or sum(v) != 1:
            raise ValueError(
                "gFCP must be a list of 3 values between 0 and 1, summing to 1"
            )
        return v

    @field_validator("gfnp")
    @classmethod
    def check_gfnp(cls, v):
        if len(v) != 3 or any(x < 0 or x > 1 for x in v) or sum(v) != 1:
            raise ValueError(
                "gFNP must be a list of 3 values between 0 and 1, summing to 1"
            )
        return v

    @field_validator("gfpp")
    @classmethod
    def check_gfpp(cls, v):
        if len(v) != 3 or any(x < 0 or x > 1 for x in v) or sum(v) != 1:
            raise ValueError(
                "gFPP must be a list of 3 values between 0 and 1, summing to 1"
            )
        return v


class Ero(NamelistBaseModel):
    ierosion: Optional[int] = Field(
        0,
        description="Determines the type of benthic erosion flux to be modeled. 0: No erosion, 1: H2S flux, 2: POC (Particulate Organic Carbon) flux, 3: H2S flux (alternative method).",
    )

    @field_validator("ierosion")
    @classmethod
    def validate_ierosion(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("ierosion must be 0, 1, 2, or 3")
        return v


class Poc(NamelistBaseModel):
    erosion: Optional[int] = Field(
        864, description="Erosion rate in kilograms per square meter per day"
    )
    etau: Optional[float] = Field(
        1e-06, description="Critical bottom shear stress in Pascals"
    )
    eporo: Optional[float] = Field(
        0.8, description="Coefficient in erosion formula (see code in icm_sfm.F90)"
    )
    efrac: Optional[float] = Field(
        0.5,
        description="Fraction coefficient in erosion formula (see code in icm_sfm.F90)",
    )
    ediso: Optional[float] = Field(
        2.5, description="H2S erosion coefficient (see code in icm_sfm.F90)"
    )
    dfrac: Optional[list] = Field(
        [0.02, 0.02],
        description="Deposition fraction of POC (Particulate Organic Carbon). If negative, dfrac will be computed",
    )
    dws_poc: Optional[list] = Field(
        [3.0, 3.0], description="Coefficient in POC erosion (see code in icm_sfm.F90)"
    )

    @field_validator("erosion")
    @classmethod
    def validate_erosion(cls, v):
        if v <= 0:
            raise ValueError("Erosion rate must be positive")
        return v

    @field_validator("etau")
    @classmethod
    def validate_etau(cls, v):
        if v <= 0:
            raise ValueError("Critical bottom shear stress must be positive")
        return v

    @field_validator("eporo")
    @classmethod
    def validate_eporo(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Eporo must be between 0 and 1")
        return v

    @field_validator("efrac")
    @classmethod
    def validate_efrac(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("Efrac must be between 0 and 1")
        return v

    @field_validator("ediso")
    @classmethod
    def validate_ediso(cls, v):
        if v <= 0:
            raise ValueError("Ediso must be positive")
        return v

    @field_validator("dfrac")
    @classmethod
    def validate_dfrac(cls, v):
        if not (isinstance(v, list) and len(v) == 2):
            raise ValueError("Dfrac must be a list of two float values")
        return v

    @field_validator("dws_poc")
    @classmethod
    def validate_dws_poc(cls, v):
        if not (isinstance(v, list) and len(v) == 2 and all(x > 0 for x in v)):
            raise ValueError("DWS_POC must be a list of two positive float values")
        return v


class Icm(NamelistBaseModel):
    marco: Optional[Marco] = Field(default_factory=Marco)
    core: Optional[Core] = Field(default_factory=Core)
    sfm: Optional[Sfm] = Field(default_factory=Sfm)
    silica: Optional[Silica] = Field(default_factory=Silica)
    zb: Optional[Zb] = Field(default_factory=Zb)
    ph_icm: Optional[Ph_icm] = Field(default_factory=Ph_icm)
    sav: Optional[Sav] = Field(default_factory=Sav)
    stem: Optional[Stem] = Field(default_factory=Stem)
    veg: Optional[Veg] = Field(default_factory=Veg)
    bag: Optional[Bag] = Field(default_factory=Bag)
    ero: Optional[Ero] = Field(default_factory=Ero)
    poc: Optional[Poc] = Field(default_factory=Poc)
