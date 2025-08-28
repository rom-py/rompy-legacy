# This file was auto generated from a schism namelist file on 2024-09-11.

from typing import Literal, Optional

from pydantic import Field

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Marco(NamelistBaseModel):
    """
    Parameters for the Marco module in ICM.
    """

    nsub: int = Field(1, description="Number of subcycles in ICM kinetics")
    iKe: Literal[0, 1, 2] = Field(
        0, description="Option for computing light attenuation coefficients"
    )
    Ke0: float = Field(
        0.26, description="Background light extinction coefficient (1/m)"
    )
    KeC: float = Field(0.017, description="Light attenuation due to chlorophyll")
    KeS: float = Field(0.07, description="Light attenuation due to TSS")
    KeSalt: float = Field(
        -0.02, description="Light attenuation due to CDOM (related to salinity)"
    )
    tss2c: float = Field(6.0, description="TSS to carbon ratio")
    iLight: Literal[0, 1] = Field(
        0, description="Option for computing light limitation factor"
    )
    alpha: list[float] = Field(
        [8.0, 8.0, 8.0], description="Initial slope of P-I curve (g[C]*m2/g[Chl]/E)"
    )
    iPR: Literal[0, 1] = Field(1, description="Option for phytoplankton predation term")
    PRR: list[float] = Field(
        [0.1, 0.2, 0.05],
        description="Predation rate by higher trophic level (day-1, or day-1.g-1.m3)",
    )
    wqc0: list[float] = Field(
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
        description="ICM initial values",
    )
    WSP: list[float] = Field(
        [
            0.3,
            0.10,
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
        description="Settling velocity (m.day-1)",
    )
    WSPn: list[float] = Field(
        [
            0.3,
            0.10,
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
        description="Net settling velocity (m.day-1)",
    )
    iSilica: Literal[0, 1] = Field(0, description="Silica model switch")
    iZB: Literal[0, 1] = Field(0, description="Zooplankton dynamics switch")
    iPh: Literal[0, 1] = Field(0, description="PH model switch")
    iCBP: Literal[0, 1] = Field(0, description="ChesBay Program Model switch")
    isav_icm: Literal[0, 1] = Field(
        0, description="Submerged Aquatic Vegetation switch"
    )
    iveg_icm: Literal[0, 1] = Field(0, description="Intertidal vegetation switch")
    iSed: Literal[0, 1] = Field(1, description="Sediment module switch")
    iBA: Literal[0, 1] = Field(0, description="Benthic Algae switch")
    iRad: Literal[0, 1] = Field(0, description="Solar radiation option")
    isflux: Literal[0, 1] = Field(
        0, description="Additional nutrient fluxes from surface"
    )
    ibflux: Literal[0, 1] = Field(
        0, description="Additional nutrient fluxes from bottom"
    )
    iout_icm: Literal[0, 1] = Field(0, description="ICM station outputs switch")
    nspool_icm: int = Field(24, description="ICM output frequency")
    iLimit: Literal[0, 1] = Field(
        0, description="Option for nutrient limitation on phytoplankton growth"
    )
    idry_icm: Literal[0, 1] = Field(
        0, description="Shallow kinetic biochemical process switch"
    )


class Core(NamelistBaseModel):
    """
    Parameters for the Core module in ICM.
    """

    GPM: list[float] = Field([2.5, 2.8, 3.5], description="PB growth rates (day-1)")
    TGP: list[float] = Field(
        [15.0, 22.0, 27.0], description="Optimal temp. for PB growth (oC)"
    )
    KTGP: list[float] = Field(
        [0.005, 0.004, 0.003, 0.008, 0.006, 0.004],
        description="Temp. dependence for PB growth (oC-2)",
    )
    MTR: list[float] = Field(
        [0.0, 0.0, 0.0], description="PB photorespiration coefficient (0<MTR<1)"
    )
    MTB: list[float] = Field(
        [0.01, 0.02, 0.03], description="PB metabolism rates (day-1)"
    )
    TMT: list[float] = Field(
        [20.0, 20.0, 20.0], description="Reference temp. for PB metabolism (oC)"
    )
    KTMT: list[float] = Field(
        [0.0322, 0.0322, 0.0322],
        description="Temp. dependence for PB metabolism (oC-1)",
    )
    FCP: list[float] = Field(
        [0.35, 0.30, 0.20, 0.55, 0.50, 0.50, 0.10, 0.20, 0.30, 0.0, 0.0, 0.0],
        description="Fractions of PB carbon into (RPOC,LPOC,DOC,SRPOC)",
    )
    FNP: list[float] = Field(
        [
            0.35,
            0.35,
            0.35,
            0.50,
            0.50,
            0.50,
            0.10,
            0.10,
            0.10,
            0.05,
            0.05,
            0.05,
            0.0,
            0.0,
            0.0,
        ],
        description="Fractions of PB nitrogen into (RPON,LPON,DON,NH4,SRPON)",
    )
    FPP: list[float] = Field(
        [
            0.10,
            0.10,
            0.10,
            0.20,
            0.20,
            0.20,
            0.50,
            0.50,
            0.50,
            0.20,
            0.20,
            0.20,
            0.0,
            0.0,
            0.0,
        ],
        description="Fractions of PB Phosphorus into (RPOP,LPOP,DOP,PO4,SRPOP)",
    )
    FCM: list[float] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
        description="Fractions of PB metabolism carbon into (RPOC,LPOC,DOC,SRPOC)",
    )
    FNM: list[float] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Fractions of PB metabolism N. into (RPON,LPON,DON,NH4,SRPON)",
    )
    FPM: list[float] = Field(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        description="Fractions of PB metabolism P. into (RPOP,LPOP,DOP,PO4,SRPOP)",
    )
    Nit: float = Field(0.07, description="Maximum nitrification rate (day-1)")
    TNit: float = Field(27.0, description="Optimal temp. for nitrification (oC)")
    KTNit: list[float] = Field(
        [0.0045, 0.0045], description="Temp. dependence for nitrification (oC-2)"
    )
    KhDOn: float = Field(1.0, description="DO half saturation for nitrification (mg/L)")
    KhDOox: float = Field(
        0.5,
        description="DO half saturation for dentrification & DOC's oxic respiration (mg/L)",
    )
    KhNO3dn: float = Field(
        0.1, description="NO3 half saturation for denitrification (mg/L)"
    )
    KC0: list[float] = Field(
        [0.005, 0.075, 0.2], description="Minimum decay rate of RPOC,LPOC,DOC (day-1)"
    )
    KN0: list[float] = Field(
        [0.005, 0.075, 0.2], description="Minimum decay rate of RPON,LPON,DON (day-1)"
    )
    KP0: list[float] = Field(
        [0.005, 0.075, 0.2], description="Minimum decay rate of RPOP,LPOP,DOP (day-1)"
    )
    KCalg: list[float] = Field(
        [0.0, 0.0, 0.0],
        description="Algae effect on RPOC,LPOC,DOC decay (day-1.m3.g[C]-1)",
    )
    KNalg: list[float] = Field(
        [0.0, 0.0, 0.0],
        description="Algae effect on RPON,LPON,DON decay (day-1.m3.g[C]-1)",
    )
    KPalg: list[float] = Field(
        [0.0, 0.0, 0.0],
        description="Algae effect on RPOP,LPOP,DOP decay (day-1.m3.g[C]-1)",
    )
    TRM: list[float] = Field(
        [20.0, 20.0, 20.0], description="Reference temp. for (RPOM,LPOM,DOM) decay (oC)"
    )
    KTRM: list[float] = Field(
        [0.069, 0.069, 0.069],
        description="Temp. dependence for (RPOM,LPOM,DOM) decay (oC-1)",
    )
    KSR0: list[float] = Field(
        [0.001, 0.001, 0.001], description="Decay rates of SRPOC,SRPON,SRPOP (day-1)"
    )
    TRSR: list[float] = Field(
        [20.0, 20.0, 20.0],
        description="Reference temp. for (SRPOC,SRPON,SRPOP) decay (oC)",
    )
    KTRSR: list[float] = Field(
        [0.069, 0.069, 0.069],
        description="Temp. dependence for (SRPOC,SRPON,SRPOP) decay (oC-1)",
    )
    KPIP: float = Field(0.0, description="Dissolution rate of PIP (day-1)")
    KCD: float = Field(1.0, description="Oxidation rate of COD at TRCOD (day-1)")
    TRCOD: float = Field(20.0, description="Reference temp. for COD oxidation (oC)")
    KTRCOD: float = Field(
        0.041, description="Temp. dependence for COD oxidation (oC-1)"
    )
    KhCOD: float = Field(
        1.5, description="COD half saturation for COD oxidation (mg[O2]/L)"
    )
    KhN: list[float] = Field(
        [0.01, 0.01, 0.01], description="Nitrogen half saturation (mg/L)"
    )
    KhP: list[float] = Field(
        [0.001, 0.001, 0.001], description="Phosphorus half saturation (mg/L)"
    )
    KhSal: list[float] = Field(
        [1e6, 1e6, 0.1], description="Salinity when PB growth is halved (PSU)"
    )
    c2chl: list[float] = Field(
        [0.059, 0.059, 0.059], description="Carbon to chlorophyll ratio (g[C]/mg[Chl])"
    )
    n2c: list[float] = Field(
        [0.167, 0.167, 0.167], description="Nitrogen to carbon ratio for phytoplankton"
    )
    p2c: list[float] = Field(
        [0.02, 0.02, 0.02], description="Phosphorus to carbon ratio for phytoplankton"
    )
    o2c: float = Field(2.67, description="Oxygen to carbon ratio in respiration")
    o2n: float = Field(4.33, description="Oxygen to ammonium ratio (g[O2]/g[NH4])")
    dn2c: float = Field(
        0.933,
        description="Mass of NO3 consumed per mass of DOC oxidized in denit. (g[N]/g[C])",
    )
    an2c: float = Field(
        0.5, description="Ratio of denit. rate to oxic DOC respiration rate"
    )
    KhDO: list[float] = Field(
        [0.5, 0.5, 0.5], description="DO half saturation for PB's DOC excretion (mg/L)"
    )
    KPO4p: float = Field(0.0, description="Coefficient relating PO4 sorption to TSS")
    WRea: float = Field(
        0.0, description="Baseline wind-induced reaeration coefficient for DO (day-1)"
    )
    PBmin: list[float] = Field(
        [0.01, 0.01, 0.01], description="Minimum PB concentration (mg[C]/L)"
    )
    dz_flux: list[float] = Field(
        [1.0, 1.0],
        description="Surface/bottom thickness (m) within which sflux/bflux are redistributed",
    )


class Sfm(NamelistBaseModel):
    """
    Parameters for the Sediment Flux Model (SFM) in ICM.
    """

    btemp0: float = Field(5.0, description="Initial temperature (oC)")
    bstc0: float = Field(0.1, description="Surface transfer coefficient")
    bSTR0: float = Field(0.0, description="Benthic stress (day)")
    bThp0: float = Field(0.0, description="Consecutive days of hypoxia (day)")
    bTox0: float = Field(
        0.0, description="Consecutive days of oxic condition after hypoxia event (day)"
    )
    bNH40: float = Field(4.0, description="Initial NH4 concentration")
    bNO30: float = Field(1.0, description="Initial NO3 concentration")
    bPO40: float = Field(5.0, description="Initial PO4 concentration")
    bH2S0: float = Field(250.0, description="Initial H2S concentration")
    bCH40: float = Field(40.0, description="Initial CH4 concentration")
    bPOS0: float = Field(500.0, description="Initial POS concentration")
    bSA0: float = Field(500.0, description="Initial SA concentration")
    bPOC0: list[float] = Field(
        [1000.0, 3000.0, 5000.0], description="Initial POC concentrations (G=1:3)"
    )
    bPON0: list[float] = Field(
        [150.0, 500.0, 1500.0], description="Initial PON concentrations (G=1:3)"
    )
    bPOP0: list[float] = Field(
        [30.0, 300.0, 500.0], description="Initial POP concentrations (G=1:3)"
    )
    bdz: float = Field(0.1, description="Sediment thickness (m)")
    bVb: float = Field(1.37e-5, description="Burial rate (m.day-1)")
    bsolid: list[float] = Field(
        [0.5, 0.5], description="Sediment solid conc. in Layer 1 and Layer 2 (Kg.L-1)"
    )
    bdiff: float = Field(
        1.8e-7, description="Diffusion coefficient for sediment temp. (m2/s)"
    )
    bTR: int = Field(20, description="Reference temp. for sediment processes")
    bVpmin: float = Field(
        3.0e-6, description="Minimum particle mixing velocity coefficient (m.day-1)"
    )
    bVp: float = Field(
        1.2e-4, description="Particle mixing velocity coefficient (m.day-1)"
    )
    bVd: float = Field(1.0e-3, description="Diffusion velocity coefficient (m.day-1)")
    bKTVp: float = Field(
        1.117, description="Temp. dependence of particle mixing velocity"
    )
    bKTVd: float = Field(1.08, description="Temp. dependence of diffusion velocity")
    bKST: float = Field(
        0.03, description="1st order decay rate of benthic stress (day-1)"
    )
    bSTmax: float = Field(20.0, description="Maximum value of benthic stress (day)")
    bKhDO_Vp: float = Field(
        4.0, description="DO half-saturation of particle mixing (mg/L)"
    )
    bDOc_ST: float = Field(1.0, description="DO criteria for benthic stress (mg/L)")
    banoxic: float = Field(
        10.0,
        description="Consecutive days of hypoxia causing maximum benthic stress (day)",
    )
    boxic: float = Field(
        45.0, description="Time lag for benthos recovery from hypoxia event (day)"
    )
    bp2d: float = Field(
        0.0,
        description="Ratio from mixing coef. to diffusion coef. (benthos enhanced effect)",
    )
    bKC: list[float] = Field(
        [0.035, 0.0018, 0.00], description="Decay rate of POC (3G class) at bTR (day-1)"
    )
    bKN: list[float] = Field(
        [0.035, 0.0018, 0.00], description="Decay rate of PON (3G class) at bTR (day-1)"
    )
    bKP: list[float] = Field(
        [0.035, 0.0018, 0.00], description="Decay rate of POP (3G class) at bTR (day-1)"
    )
    bKTC: list[float] = Field(
        [1.10, 1.150, 1.17], description="Temp. dependence of POC decay (oC-1)"
    )
    bKTN: list[float] = Field(
        [1.10, 1.150, 1.17], description="Temp. dependence of PON decay (oC-1)"
    )
    bKTP: list[float] = Field(
        [1.10, 1.150, 1.17], description="Temp. dependence of POP decay (oC-1)"
    )
    bFCP: list[float] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto POC into sed POC (G3,PB=1:3)",
    )
    bFNP: list[float] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto PON into sed PON (G3,PB=1:3)",
    )
    bFPP: list[float] = Field(
        [0.35, 0.55, 0.01, 0.35, 0.55, 0.01, 0.35, 0.55, 0.01],
        description="Phyto POP into sed POP (G3,PB=1:3)",
    )
    bFCM: list[float] = Field(
        [0.0, 0.43, 0.57], description="Refractory POC into sed POC(3G)"
    )
    bFNM: list[float] = Field(
        [0.0, 0.54, 0.46], description="Refractory PON into sed PON(3G)"
    )
    bFPM: list[float] = Field(
        [0.0, 0.43, 0.57], description="Refractory POP into sed POP(3G)"
    )
    bKNH4f: float = Field(
        0.20, description="NH4 reaction rate in freshwater at bTR (1st layer) (m/day)"
    )
    bKNH4s: float = Field(
        0.140, description="NH4 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bKTNH4: float = Field(1.08, description="Temp. dependency for NH4 reaction (oC-1)")
    bKhNH4: float = Field(
        1.5, description="Half-saturation NH4 for nitrification (g/m3)"
    )
    bKhDO_NH4: float = Field(
        2.00, description="Half-saturation DO for nitrification (g/m3)"
    )
    bpieNH4: float = Field(
        1.0, description="Partition coefficients of NH4 in Layer 1 & 2 (Kg-1.L)"
    )
    bsaltn: float = Field(
        1.0,
        description="Salinity criteria of fresh/salty water for NH4/NO3 reaction (PSU)",
    )
    bKNO3f: float = Field(
        0.30, description="NO3 reaction rate in freshwater at bTR (1st layer) (m/day)"
    )
    bKNO3s: float = Field(
        0.125, description="NO3 reaction rate in salty water at bTR (1st layer) (m/day)"
    )
    bKNO3: float = Field(0.25, description="NO3 reaction rate (2nd layer) (m/day)")
    bKTNO3: float = Field(1.08, description="Temp. dependency for NO3 reaction (oC-1)")
    bKH2Sd: float = Field(
        0.2, description="Dissolved H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bKH2Sp: float = Field(
        0.4, description="Particulate H2S reaction rate at bTR (1st layer) (m/day)"
    )
    bKTH2S: float = Field(1.08, description="Temp. dependency for H2S reaction (oC-1)")
    bpieH2Ss: float = Field(
        100.0, description="Partition coefficient of NH4 in Layer 1 (Kg-1.L)"
    )
    bpieH2Sb: float = Field(
        100.0, description="Partition coefficient of NH4 in Layer 2 (Kg-1.L)"
    )
    bKhDO_H2S: float = Field(
        8.0, description="O2 constant to normalize H2S oxidation (g[O2]/m3)"
    )
    bsaltc: float = Field(
        1.0,
        description="Salinity criteria of fresh/salty water for carbon reaction (PSU)",
    )
    bKCH4: float = Field(
        0.2, description="CH4 reaction rate at bTR (1st layer) (m/day)"
    )
    bKTCH4: float = Field(1.08, description="Temp. dependency for CH4 reaction")
    bKhDO_CH4: float = Field(
        0.2, description="Half-saturation DO for CH4 oxidation (g[O2]/m3)"
    )
    bo2n: float = Field(
        2.86, description="Oxygen to nitrogen ratio in sediment (denitrification)"
    )
    bpiePO4: float = Field(
        50.0, description="Partition coefficient of PO4 in Layer 2 (Kg-1.L)"
    )
    bKOPO4f: float = Field(
        3000.0,
        description="Oxygen dependency for PO4 sorption in freshwater in Layer 1",
    )
    bKOPO4s: float = Field(
        300.0,
        description="Oxygen dependency for PO4 sorption in salty water in Layer 1",
    )
    bDOc_PO4: float = Field(1.0, description="DO criteria for PO4 sorptiona (g[O2]/m3)")
    bsaltp: float = Field(
        1.0,
        description="Salinity criteria of fresh/salty water for PO4 partition (PSU)",
    )
    bKS: float = Field(0.50, description="Decay rate of POS (3G class) at bTR")
    bKTS: float = Field(1.10, description="Temp. dependence of POS decay")
    bSIsat: float = Field(
        40.0, description="Silica saturation conc. in pore water (g[Si]/m3)"
    )
    bpieSI: float = Field(
        100.0, description="Partition coefficient of silica in Layer 2 (Kg-1.L)"
    )
    bKOSI: float = Field(
        10.0, description="Oxygen dependency for silica sorption in Layer 1"
    )
    bKhPOS: float = Field(
        5.0e4, description="POS half saturation for POS dissolution (g/m3)"
    )
    bDOc_SI: float = Field(
        1.0, description="DO criteria for silica sorptiona (g[O2]/m3)"
    )
    bJPOSa: float = Field(
        0.0,
        description="Additional POS flux associated with POM detrius beside algea (g.m-2.day-1)",
    )
    bFCs: list[float] = Field(
        [0.65, 0.255, 0.095], description="SAV POC into 3G sed 3G POC"
    )
    bFNs: list[float] = Field(
        [0.65, 0.300, 0.050], description="SAV PON into 3G sed 3G PON"
    )
    bFPs: list[float] = Field(
        [0.65, 0.255, 0.095], description="SAV POP into 3G sed 3G POP"
    )
    bFCv: list[float] = Field(
        [0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095],
        description="VEG POC into sed POC (G3,PB=1:3)",
    )
    bFNv: list[float] = Field(
        [0.65, 0.300, 0.050, 0.65, 0.300, 0.050, 0.65, 0.300, 0.050],
        description="VEG PON into sed PON (G3,PB=1:3)",
    )
    bFPv: list[float] = Field(
        [0.65, 0.255, 0.095, 0.65, 0.255, 0.095, 0.65, 0.255, 0.095],
        description="VEG POP into sed POP (G3,PB=1:3)",
    )


class Silica(NamelistBaseModel):
    """
    Parameters for the Silica module in ICM.
    """

    FSP: list[float] = Field(
        [0.90, 0.10], description="Fractions of diatom silica into (SU,SA)"
    )
    FSM: list[float] = Field(
        [0.50, 0.50], description="Fractions of diatom metabolism Si into (SU,SA)"
    )
    KS: float = Field(0.03, description="Dissolution rate of SU at TRS (day-1)")
    TRS: float = Field(20.0, description="Reference temp. for SU dissolution (oC)")
    KTRS: float = Field(0.092, description="Temp. dependence for SU dissolution (oC-1)")
    KhS: list[float] = Field(
        [0.05, 0.0, 0.0], description="Silica half saturation (mg/L)"
    )
    s2c: list[float] = Field(
        [0.50, 0.0, 0.0], description="Silica to carbon ratio for phytolankton"
    )
    KSAp: float = Field(
        0.0, description="Coefficient relating Silicate(SA) sorption to TSS"
    )


class Zb(NamelistBaseModel):
    """
    Zooplankton (ZB) parameters for the ICM model.
    """

    zGPM: list[float] = Field(
        default=[
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
        description="ZB predation rate(day-1); dim(prey=1:8, ZB=1:2)",
    )
    zKhG: list[float] = Field(
        default=[
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
        description="Reference prey conc.(mg/L); dim(prey=1:8, ZB=1:2)",
    )
    zTGP: list[float] = Field(
        default=[25.0, 25.0], description="Optimal temp. for ZB growth (oC)"
    )
    zKTGP: list[float] = Field(
        default=[0.0035, 0.008, 0.025, 0.030],
        description="Temp. dependence for ZB growth (T<=zTGP & T>zTGP); dim(ZB=1:2,1:2) (oC-2)",
    )
    zAG: float = Field(
        default=0.75, description="ZB assimilation efficiency ratio (0-1)"
    )
    zRG: float = Field(
        default=0.1, description="ZB respiration ratio when it grazes (0-1)"
    )
    zMRT: list[float] = Field(
        default=[0.02, 0.02], description="ZB mortality rates (day-1)"
    )
    zMTB: list[float] = Field(
        default=[0.254, 0.186], description="ZB metabolism rates (day-1)"
    )
    zTMT: list[float] = Field(
        default=[20.0, 20.0], description="Reference temp. for ZB metabolism (oC)"
    )
    zKTMT: list[float] = Field(
        default=[0.0693, 0.0693],
        description="Temp. dependence for ZB metabolism (oC-1)",
    )
    zFCP: list[float] = Field(
        default=[0.35, 0.55, 0.10],
        description="Fractions of ZB carbon into (RPOC,LPOC,DOC)",
    )
    zFNP: list[float] = Field(
        default=[0.35, 0.50, 0.10, 0.05],
        description="Fractions of ZB nitrogen into (RPON,LPON,DON,NH4)",
    )
    zFPP: list[float] = Field(
        default=[0.10, 0.20, 0.50, 0.20],
        description="Fractions of ZB Phosphorus into (RPOP,LPOP,DOP,PO4)",
    )
    zFSP: list[float] = Field(
        default=[0.70, 0.25], description="Fractions of ZB silica into (SU,SA)"
    )
    zFCM: list[float] = Field(
        default=[0.10, 0.0],
        description="Fractions of ZB metabolism carbon into DOC; dim(ZB=1:2)",
    )
    zFNM: list[float] = Field(
        default=[0.35, 0.30, 0.50, 0.40, 0.10, 0.20, 0.05, 0.10],
        description="Fractions of ZB metabolism N. into (RPON,LPON,DON,NH4); dim(ZB=1:2,4)",
    )
    zFPM: list[float] = Field(
        default=[0.35, 0.30, 0.50, 0.40, 0.10, 0.20, 0.05, 0.10],
        description="Fractions of ZB metabolism P. into (RPOP,LPOP,DOP,PO4); dim(ZB=1:2,4)",
    )
    zFSM: list[float] = Field(
        default=[0.50, 0.40, 0.50, 0.60],
        description="Fractions of ZB metabolism Si into (SU,SA); dim(ZB=1:2,2)",
    )
    zKhDO: list[float] = Field(
        default=[0.5, 0.5],
        description="DO half saturation for ZB's DOC excretion (mg/L)",
    )
    zn2c: list[float] = Field(
        default=[0.20, 0.20], description="Nitrogen to carbon ratio for zooplankton"
    )
    zp2c: list[float] = Field(
        default=[0.02, 0.02], description="Phosphorus to carbon ratio for zooplankton"
    )
    zs2c: list[float] = Field(
        default=[0.50, 0.50], description="Silica to carbon ratio for zooplankton"
    )
    z2pr: list[float] = Field(
        default=[0.5, 0.5],
        description="Ratio converting ZB+PB biomass to predation rates on ZB (L.mg-1.day-1)",
    )
    p2pr: float = Field(
        default=0.25,
        description="Ratio converting ZB+PB biomass to predation rates on PB (L.mg-1.day-1)",
    )


class PhIcm(NamelistBaseModel):
    """
    PH model parameters for the ICM model.
    """

    ppatch0: Literal[-999] = Field(
        default=-999, description="Region flag for pH (1: ON all elem.; -999: spatial)"
    )
    pKCACO3: float = Field(
        default=60.0, description="Dissulution between CaCO3 and Ca++"
    )
    pKCA: float = Field(
        default=60.0,
        description="Sediment surface transfer coefficient from CaCO3 to Ca++",
    )
    pRea: float = Field(default=1.0, description="Reaeration rate for CO2")
    inu_ph: Literal[0] = Field(default=0, description="Nudge option for pH model")


class Sav(NamelistBaseModel):
    """
    Submerged Aquatic Vegetation (SAV) parameters for the ICM model.
    """

    spatch0: Literal[-999] = Field(
        default=-999,
        description="Region flag for SAV. (1: ON all elem.; -999: spatial)",
    )
    stleaf0: Literal[-999] = Field(
        default=-999, description="Init conc of total sav leaf"
    )
    ststem0: Literal[-999] = Field(
        default=-999, description="Init conc of total sav stem"
    )
    stroot0: Literal[-999] = Field(
        default=-999, description="Init conc of total sav root"
    )
    sGPM: float = Field(default=0.1, description="Maximum growth rate (day-1)")
    sTGP: int = Field(default=32, description="Optimal growth temperature (oC)")
    sKTGP: list[float] = Field(
        default=[0.003, 0.005],
        description="Temp. dependence for growth (T<=sTGP & T>sTGP)",
    )
    sFAM: float = Field(
        default=0.2, description="Fraction of leaf production to active metabolism"
    )
    sFCP: list[float] = Field(
        default=[0.6, 0.3, 0.1],
        description="Fractions of production to leaf/stem/root biomass",
    )
    sMTB: list[float] = Field(
        default=[0.02, 0.02, 0.02], description="Metabolism rates of leaf/stem/root"
    )
    sTMT: list[int] = Field(
        default=[20, 20, 20],
        description="Reference temp. for leaf/stem/root metabolism",
    )
    sKTMT: list[float] = Field(
        default=[0.069, 0.069, 0.069],
        description="Temp. dependence of leaf/stem/root metabolism",
    )
    sFCM: list[float] = Field(
        default=[0.05, 0.15, 0.3, 0.5],
        description="Fractions of metabolism C into (RPOC,LPOC,DOC,CO2)",
    )
    sFNM: list[float] = Field(
        default=[0.05, 0.15, 0.3, 0.5],
        description="Fractions of metabolism N into (RPON,LPON,DON,NH4)",
    )
    sFPM: list[float] = Field(
        default=[0.05, 0.1, 0.35, 0.5],
        description="Fractions of metabolism P into (RPOP,LPOP,DOP,PO4)",
    )
    sKhNw: float = Field(
        default=0.01, description="Nitrogen half saturation in water column"
    )
    sKhNs: float = Field(
        default=0.1, description="Nitrogen half saturation in sediments"
    )
    sKhNH4: float = Field(default=0.1, description="Ammonium half saturation")
    sKhPw: float = Field(
        default=0.001, description="Phosphorus half saturation in water column"
    )
    sKhPs: float = Field(
        default=0.01, description="Phosphorus half saturation in sediments"
    )
    salpha: float = Field(
        default=0.006, description="Init. slope of P-I curve (g[C]*m2/g[Chl]/E)"
    )
    sKe: float = Field(
        default=0.045, description="Light attenuation due to sav absorption"
    )
    shtm: list[float] = Field(
        default=[0.054, 2.0], description="Minimum (base) and maximium canopy height"
    )
    s2ht: list[float] = Field(
        default=[0.0036, 0.0036, 0.0],
        description="Coeffs. converting (leaf,stem,root) to canopy height",
    )
    sc2dw: float = Field(default=0.38, description="Carbon to dry weight ratio of SAV")
    s2den: int = Field(
        default=10, description="Coeff. computing SAV density from leaf&stem"
    )
    sn2c: float = Field(default=0.09, description="Nitrogen to carbon ratio of sav")
    sp2c: float = Field(default=0.01, description="Phosphorus to carbon ratio")
    so2c: float = Field(default=2.67, description="Oxygen to carbon ratio")


class Veg(NamelistBaseModel):
    """
    Intertidal Vegetation (VEG) parameters for the ICM model.
    """

    vpatch0: Literal[-999] = Field(
        default=-999,
        description="Region flag for VEG. (1: ON all elem.; -999: spatial)",
    )
    vtleaf0: list[float] = Field(
        default=[100.0, 100.0, 100.0], description="Init conc to total veg leaf"
    )
    vtstem0: list[float] = Field(
        default=[100.0, 100.0, 100.0], description="Init conc to total veg stem"
    )
    vtroot0: list[float] = Field(
        default=[30.0, 30.0, 30.0], description="Init conc to total veg root"
    )
    vGPM: list[float] = Field(
        default=[0.1, 0.1, 0.1], description="Maximum growth rate (day-1)"
    )
    vFAM: list[float] = Field(
        default=[0.2, 0.2, 0.2],
        description="Fractions of leaf production to active metabolism",
    )
    vTGP: list[float] = Field(
        default=[32.0, 32.0, 32.0], description="Optimal growth temperature (oC)"
    )
    vKTGP: list[float] = Field(
        default=[0.003, 0.003, 0.003, 0.005, 0.005, 0.005],
        description="Temp. dependence(3,2) for growth (T<=vTGP & T>vTGP)",
    )
    vFCP: list[float] = Field(
        default=[0.6, 0.6, 0.6, 0.3, 0.3, 0.3, 0.1, 0.1, 0.1],
        description="Fractions of production to leaf/stem/root biomass; dim(veg,leaf/stem/root)",
    )
    vMTB: list[float] = Field(
        default=[0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.01, 0.01, 0.01],
        description="Metabolism rates of leaf/stem/root; dim(veg,leaf/stem/root)",
    )
    vTMT: list[float] = Field(
        default=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        description="Reference temp. for leaf/stem/root metabolism",
    )
    vKTMT: list[float] = Field(
        default=[0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069, 0.069],
        description="Temp. depdenpence for leaf/stem/root metabolism",
    )
    vFNM: list[float] = Field(
        default=[0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
        description="Fractions(3,4) of metabolism N into (RPON,LPON,DON,NH4)",
    )
    vFPM: list[float] = Field(
        default=[0.05, 0.05, 0.05, 0.1, 0.1, 0.1, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5],
        description="Fractions(3,4) of metabolism P into (RPOP,LPOP,DOP,PO4)",
    )
    vFCM: list[float] = Field(
        default=[0.05, 0.05, 0.05, 0.15, 0.15, 0.15, 0.3, 0.3, 0.3, 0.5, 0.5, 0.5],
        description="Fractions(3,4) of metabolism N into (RPOC,LPOC,DOC,CO2)",
    )
    ivNc: Literal[1] = Field(
        default=1, description="Recycled veg N goes to (0: sediment; 1: water)"
    )
    ivPc: Literal[1] = Field(
        default=1, description="Recycled veg P goes to (0: sediment; 1: water)"
    )
    vKhNs: list[float] = Field(
        default=[0.1, 0.1, 0.1], description="Nitrogen half saturation in sediments"
    )
    vKhPs: list[float] = Field(
        default=[0.01, 0.01, 0.01],
        description="Phosphorus half saturation in sediments",
    )
    vScr: list[float] = Field(
        default=[35.0, 35.0, 35.0],
        description="Reference sality for computing veg growth",
    )
    vSopt: list[float] = Field(
        default=[35.0, 15.0, 0.0], description="Optimal salinity for veg growth"
    )
    vInun: list[float] = Field(
        default=[1.0, 1.0, 1.0],
        description="Reference value for inundation stress (nondimensional)",
    )
    ivNs: Literal[1] = Field(
        default=1, description="N limitation on veg growth(0: OFF; 1: ON)"
    )
    ivPs: Literal[1] = Field(
        default=1, description="P limitation on veg growth(0: OFF; 1: ON)"
    )
    ivMRT: Literal[0] = Field(
        default=0, description="Veg mortality term (0: OFF;  1: ON)"
    )
    vTMR: list[float] = Field(
        default=[17.0, 17.0, 17.0, 17.0, 17.0, 17.0],
        description="Reference temp(3,2) for leaf/stem mortality",
    )
    vKTMR: list[float] = Field(
        default=[4.0, 4.0, 4.0, 4.0, 4.0, 4.0],
        description="Temp dependence(3,2) for leaf/stem mortality",
    )
    vMR0: list[float] = Field(
        default=[12.8, 12.8, 12.8, 12.8, 12.8, 12.8],
        description="Base value(3,2) of temp effect on mortality (unit: None)",
    )
    vMRcr: list[float] = Field(
        default=[15.0, 15.0, 15.0, 15.0, 15.0, 15.0],
        description="Reference value(3,2) for computing mortality (unit: None)",
    )
    valpha: list[float] = Field(
        default=[0.006, 0.006, 0.006], description="Init. slope of P-I curve"
    )
    vKe: list[float] = Field(
        default=[0.045, 0.045, 0.045],
        description="Light attenuation from veg absorption",
    )
    vht0: list[float] = Field(
        default=[0.054, 0.054, 0.054], description="Base veg canopy hgt (vht)"
    )
    vcrit: list[float] = Field(
        default=[250.0, 250.0, 250.0], description="Critical mass for computing vht"
    )
    v2ht: list[float] = Field(
        default=[0.0036, 0.0036, 0.0036, 0.001, 0.001, 0.001],
        description="Coefs. convert mass(3,2) to canopy height",
    )
    vc2dw: list[float] = Field(
        default=[0.38, 0.38, 0.38], description="Carbon to dry weight ratio of VEG"
    )
    v2den: list[int] = Field(
        default=[10, 10, 10], description="Coeff. computing veg density"
    )
    vp2c: list[float] = Field(
        default=[0.01, 0.01, 0.01], description="Phosphorus to carbon ratio"
    )
    vn2c: list[float] = Field(
        default=[0.09, 0.09, 0.09], description="Nitrogen to carbon ratio"
    )
    vo2c: list[float] = Field(
        default=[2.67, 2.67, 2.67], description="Oxygen to carbon ratio"
    )


class Bag(NamelistBaseModel):
    """
    Benthic Algae parameters for the ICM model.
    """

    gpatch0: Literal[-999] = Field(
        default=-999, description="Region flag for BA. (1: ON all elem.; -999: spatial)"
    )
    BA0: float = Field(default=5.0, description="Initial BA concentration (g[C].m-2)")
    gGPM: float = Field(default=2.25, description="BA maximum growth rate (day-1)")
    gTGP: float = Field(default=20.0, description="Optimal temp. for BA growth (oC)")
    gKTGP: list[float] = Field(
        default=[0.004, 0.006], description="Temp. dependence for BA growth (oC-2)"
    )
    gMTB: float = Field(
        default=0.05, description="Respiration rate at temp. of gTR (day-1)"
    )
    gPRR: float = Field(
        default=0.1, description="Predation rate at temp. of gTR (day-1)"
    )
    gTR: float = Field(
        default=20.0, description="Reference temperature for BA respiration (oC)"
    )
    gKTR: float = Field(
        default=0.069, description="Temp. dependence for BA respiration (oC-1)"
    )
    galpha: float = Field(default=0.1, description="Init. slope of P-I curve (m2/E)")
    gKSED: float = Field(
        default=0.0, description="Light attenuation due to sediment (None)"
    )
    gKBA: float = Field(
        default=0.01,
        description="Light attenuation coef. due to BA self-shading (g[C]-1.m2)",
    )
    gKhN: float = Field(
        default=0.01, description="Nitrogen half saturation for BA growth (g[N]/m2)"
    )
    gKhP: float = Field(
        default=0.001, description="Phosphorus half saturation for BA growth (g[P]/2)"
    )
    gp2c: float = Field(default=0.0167, description="Phosphorus to carbon ratio")
    gn2c: float = Field(default=0.167, description="Nitrogen to carbon ratio")
    go2c: float = Field(default=2.67, description="Oxygen to carbon ratio")
    gFCP: list[float] = Field(
        default=[0.5, 0.45, 0.05],
        description="Fraction of predation BA C into 3G classes in sediment",
    )
    gFNP: list[float] = Field(
        default=[0.5, 0.45, 0.05],
        description="Fraction of predation BA N into 3G classes in sediment",
    )
    gFPP: list[float] = Field(
        default=[0.5, 0.45, 0.05],
        description="Fraction of predation BA P into 3G classes in sediment",
    )


class Ero(NamelistBaseModel):
    """
    Benthic erosion parameters for the ICM model.
    """

    ierosion: Literal[0, 1, 2, 3] = Field(
        default=0, description="1: H2S flux; 2: POC flux; 3: H2S&POC flux"
    )
    erosion: int = Field(default=864, description="Erosion rate (kg.m-2.day)")
    etau: float = Field(default=1e-6, description="Critical bottom shear stress (Pa)")
    eporo: float = Field(default=0.8, description="Coefficient in erosion formula")
    efrac: float = Field(
        default=0.5, description="Fraction coefficient in erosion formula"
    )
    ediso: float = Field(default=2.5, description="H2S erosion coefficient")
    dfrac: list[float] = Field(
        default=[0.02, 0.02],
        description="Deposition fraction of POC (negative value: dfrac will be computed)",
    )
    dWS_POC: list[float] = Field(
        default=[3.0, 3.0], description="Coefficient in POC erosion"
    )


class Icm(NamelistBaseModel):
    """
    Master model for ICM (Integrated Compartment Model) parameters.
    """

    marco: Optional[Marco] = Field(default=None)
    core: Optional[Core] = Field(default=None)
    sfm: Optional[Sfm] = Field(default=None)
    silica: Optional[Silica] = Field(default=None)
    zb: Optional[Zb] = Field(default=None)
    ph_icm: Optional[PhIcm] = Field(default=None)
    sav: Optional[Sav] = Field(default=None)
    veg: Optional[Veg] = Field(default=None)
    bag: Optional[Bag] = Field(default=None)
    ero: Optional[Ero] = Field(default=None)
