# This file was auto generated from a SCHISM namelist file on 2025-01-29.

import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Proc(NamelistBaseModel):
    procname: Optional[str] = Field(
        "ACS", description="Project name for the simulation"
    )
    dimmode: Optional[int] = Field(
        2, description="Mode of run: 2 for 2D (always 2D when coupled to SCHISM)"
    )
    lstea: Optional[bool] = Field(
        False, description="Steady mode flag (under development)"
    )
    lqstea: Optional[bool] = Field(
        False,
        description="Quasi-Steady Mode flag. If True, WWM-II performs subiterations defined as DELTC/NQSITER unless QSCONVI is reached",
    )
    lsphe: Optional[bool] = Field(
        True, description="Flag for spherical coordinates (lon/lat)"
    )
    lnautin: Optional[bool] = Field(
        True,
        description="Flag for nautical convention in input angles (degrees). Recommended to be True",
    )
    lnautout: Optional[bool] = Field(
        True,
        description="Flag for nautical convention in output angles. If True, 0 is from north, 90 is from east. If False, mathematical convention is used (0: to east, 90: to north)",
    )
    lmono_in: Optional[bool] = Field(
        False,
        description="Flag for prescribing monochromatic wave height Hmono as boundary conditions. Incident wave is defined as Hmono = sqrt(2) * Hs",
    )
    lmono_out: Optional[bool] = Field(
        False, description="Flag for outputting wave heights in terms of Lmono"
    )
    begtc: Optional[str] = Field(
        "20200101.000000",
        description="Start time of the simulation in format 'yyyymmdd.hhmmss'",
    )
    deltc: Optional[int] = Field(
        600, description="Time step in seconds (must match dt*nstep_wwm in SCHISM)"
    )
    unitc: Optional[str] = Field(
        "SEC", description="Unit of time step (SEC for seconds)"
    )
    endtc: Optional[str] = Field(
        "20200201.000000",
        description="End time of the simulation in format 'yyyymmdd.hhmmss'",
    )
    dmin: Optional[float] = Field(
        0.01, description="Minimum water depth. Must be the same as h0 in SCHISM"
    )

    @field_validator("procname")
    def validate_procname(cls, v):
        if not v.strip():
            raise ValueError("Project name cannot be empty")
        return v

    @field_validator("dimmode")
    def validate_dimmode(cls, v):
        if v != 2:
            raise ValueError("DIMMODE must be 2 when coupled to SCHISM")
        return v

    @field_validator("lstea")
    def validate_lstea(cls, v):
        return v

    @field_validator("lqstea")
    def validate_lqstea(cls, v):
        return v

    @field_validator("lsphe")
    def validate_lsphe(cls, v):
        return v

    @field_validator("lnautin")
    def validate_lnautin(cls, v):
        return v

    @field_validator("lnautout")
    def validate_lnautout(cls, v):
        return v

    @field_validator("lmono_in")
    def validate_lmono_in(cls, v):
        return v

    @field_validator("lmono_out")
    def validate_lmono_out(cls, v):
        return v

    @field_validator("begtc")
    def validate_begtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
        except ValueError:
            raise ValueError("Invalid date format for BEGTC")
        return v

    @field_validator("deltc")
    def validate_deltc(cls, v):
        if v <= 0:
            raise ValueError("DELTC must be positive")
        return v

    @field_validator("unitc")
    def validate_unitc(cls, v):
        if v.upper() != "SEC":
            raise ValueError("UNITC must be SEC")
        return v.upper()

    @field_validator("endtc")
    def validate_endtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
        except ValueError:
            raise ValueError("Invalid date format for ENDTC")
        return v

    @field_validator("dmin")
    def validate_dmin(cls, v):
        if v <= 0:
            raise ValueError("DMIN must be positive")
        return v


class Coupl(NamelistBaseModel):
    lcpl: Optional[bool] = Field(
        True,
        description="Main switch to enable coupling with the current model. Should be kept on for SCHISM-WWM coupling.",
    )
    radflag: Optional[str] = Field(
        "LON",
        description="Determines the formulation for wave-induced forces. 'LON' for Longuet-Higgins formulation, 'VOR' for vortex formulation. Usually set to 'LON'.",
    )
    letot: Optional[bool] = Field(
        False,
        description="Option to compute wave-induced radiation stress. If True, radiation stress is based on the integrated wave spectrum. If False (recommended), it's estimated using the directional spectra as described in Roland et al. (2008). False is preferred as it preserves spectral information.",
    )
    nlvt: Optional[int] = Field(
        10, description="Number of vertical layers. Not used when coupled with SCHISM."
    )
    dtcoup: Optional[float] = Field(
        600.0,
        description="Coupling time step in seconds. Not used when coupled to SCHISM.",
    )

    @field_validator("lcpl")
    @classmethod
    def validate_lcpl(cls, v):
        return v

    @field_validator("radflag")
    @classmethod
    def validate_radflag(cls, v):
        if v not in ["LON", "VOR"]:
            raise ValueError("radflag must be either 'LON' or 'VOR'")
        return v

    @field_validator("letot")
    @classmethod
    def validate_letot(cls, v):
        return v

    @field_validator("nlvt")
    @classmethod
    def validate_nlvt(cls, v):
        if v <= 0:
            raise ValueError("nlvt must be a positive integer")
        return v

    @field_validator("dtcoup")
    @classmethod
    def validate_dtcoup(cls, v):
        if v <= 0:
            raise ValueError("dtcoup must be a positive number")
        return v


class Grid(NamelistBaseModel):
    lcird: Optional[bool] = Field(
        True,
        description="Flag to indicate if a full circle in directional space is used. If True, MINDIR and MAXDIR are ignored.",
    )
    lstag: Optional[bool] = Field(
        False,
        description="Flag to stagger directional bins with a half Dtheta. Can only be True for regular grids to avoid characteristic line aligning with grid line.",
    )
    mindir: Optional[float] = Field(
        0.0,
        description="Minimum direction for simulation in degrees (nautical convention; 0: from N; 90: from E). Not used if LCIRD is True.",
    )
    maxdir: Optional[float] = Field(
        360.0,
        description="Maximum direction for simulation in degrees. May be less than MINDIR. Not used if LCIRD is True.",
    )
    mdc: Optional[int] = Field(24, description="Number of directional bins")
    frlow: Optional[float] = Field(
        0.04,
        description="Low frequency limit of the discrete wave period in Hz (1/period)",
    )
    frhigh: Optional[float] = Field(
        1.0, description="High frequency limit of the discrete wave period in Hz"
    )
    msc: Optional[int] = Field(24, description="Number of frequency bins")
    filegrid: Optional[str] = Field(
        "hgrid_WWM.gr3",
        description="Name of the grid file. Should be 'hgridi_WWM.gr3' if IGRIDTYPE is 3 (SCHISM)",
    )
    igridtype: Optional[int] = Field(
        3,
        description="Grid type used. 1: XFN system.dat, 2: WWM-PERIODIC, 3: SCHISM, 4: old WWM type",
    )
    lslop: Optional[bool] = Field(
        False, description="Flag to enable bottom slope limiter"
    )
    slmax: Optional[float] = Field(
        0.2, description="Maximum allowed bottom slope when LSLOP is True"
    )
    lvar1d: Optional[bool] = Field(
        False,
        description="Flag to use variable dx in 1D mode. Not used with SCHISM (IGRIDTYPE = 3)",
    )
    loptsig: Optional[bool] = Field(
        False,
        description="Flag to use optimal distributions of frequencies in spectral space (fi+1 = fi * 1.1)",
    )

    @field_validator("lcird")
    def validate_lcird(cls, v):
        return v

    @field_validator("lstag")
    def validate_lstag(cls, v, info):
        if v and info.data.get("igridtype") != 1:
            raise ValueError("LSTAG can only be True for regular grids (IGRIDTYPE = 1)")
        return v

    @field_validator("mindir")
    def validate_mindir(cls, v):
        if not 0 <= v < 360:
            raise ValueError("MINDIR must be between 0 and 360 degrees")
        return v

    @field_validator("maxdir")
    def validate_maxdir(cls, v):
        if not 0 <= v <= 360:
            raise ValueError("MAXDIR must be between 0 and 360 degrees")
        return v

    @field_validator("mdc")
    def validate_mdc(cls, v):
        if v <= 0:
            raise ValueError("MDC must be a positive integer")
        return v

    @field_validator("frlow")
    def validate_frlow(cls, v):
        if v <= 0:
            raise ValueError("FRLOW must be a positive number")
        return v

    @field_validator("frhigh")
    def validate_frhigh(cls, v, info):
        if v <= info.data.get("frlow", 0):
            raise ValueError("FRHIGH must be greater than FRLOW")
        return v

    @field_validator("msc")
    def validate_msc(cls, v):
        if v <= 0:
            raise ValueError("MSC must be a positive integer")
        return v

    @field_validator("filegrid")
    def validate_filegrid(cls, v):
        if not v.strip():
            raise ValueError("FILEGRID must not be empty")
        return v

    @field_validator("igridtype")
    def validate_igridtype(cls, v):
        if v not in [1, 2, 3, 4]:
            raise ValueError("IGRIDTYPE must be 1, 2, 3, or 4")
        return v

    @field_validator("lslop")
    def validate_lslop(cls, v):
        return v

    @field_validator("slmax")
    def validate_slmax(cls, v):
        if v <= 0:
            raise ValueError("SLMAX must be a positive number")
        return v

    @field_validator("lvar1d")
    def validate_lvar1d(cls, v, info):
        if v and info.data.get("igridtype") == 3:
            raise ValueError("LVAR1D cannot be True when IGRIDTYPE is 3 (SCHISM)")
        return v

    @field_validator("loptsig")
    def validate_loptsig(cls, v):
        return v


class Init(NamelistBaseModel):
    lhotr: Optional[bool] = Field(
        False,
        description="Flag to indicate whether to use a hotstart file for initial conditions. If True, the model will read initial conditions from a file specified in the HOTFILE section.",
    )
    linid: Optional[bool] = Field(
        True,
        description="Flag to control the initial condition setup. If False, default initial conditions are used. If True, it allows for using external sources (e.g., WW3) as initial conditions.",
    )
    initstyle: Optional[int] = Field(
        2,
        description="Specifies the method for initializing wave conditions. 1 for Parametric Jonswap, 2 for reading from Global NETCDF files. Option 2 only works if IBOUNDFORMAT is set to 3.",
    )

    # validator to ensure thatn only one of lhotr and linid is True
    @model_validator(mode="after")
    def check_init(self):
        if self.lhotr and self.linid:
            raise ValueError("Only one of LHOTR and LINID can be True")
        return self

    @field_validator("initstyle")
    @classmethod
    def validate_initstyle(cls, v):
        if v not in [1, 2]:
            raise ValueError("initstyle must be either 1 or 2")
        return v


class Bouc(NamelistBaseModel):
    lbcse: Optional[bool] = Field(
        True, description="Determines if wave boundary data is time-dependent."
    )
    lbinter: Optional[bool] = Field(
        True,
        description="Controls time interpolation when LBCSE is true. Not available for quasi-steady mode within subtime steps.",
    )
    lbcwa: Optional[bool] = Field(False, description="Enables parametric wave spectra.")
    lbcsp: Optional[bool] = Field(
        True,
        description="Enables specification of non-parametric wave spectra, defined in FILEWAVE.",
    )
    linhom: Optional[bool] = Field(
        True, description="Enables non-uniform wave boundary conditions in space."
    )
    lbsp1d: Optional[bool] = Field(
        False,
        description="Specifies 1D (frequency space only) format for FILEWAVE when LBCSP is true and LINHOM is false.",
    )
    lbsp2d: Optional[bool] = Field(
        True,
        description="Specifies 2D format for FILEWAVE when LBCSP is true and LINHOM is false.",
    )
    begtc: Optional[str] = Field(
        "20200101.000000",
        description="Start time of the wave boundary file (FILEWAVE) in 'YYYYMMDD.HHMMSS' format.",
    )
    deltc: Optional[int] = Field(1, description="Time step in FILEWAVE.")
    unitc: Optional[str] = Field(
        "HR", description="Time unit for DELTC. Can be HR, MIN, or SEC."
    )
    endtc: Optional[str] = Field(
        "20200201.000000",
        description="End time of the wave boundary file in 'YYYYMMDD.HHMMSS' format.",
    )
    filebound: Optional[str] = Field(
        "wwmbnd.gr3",
        description="Boundary file defining boundary conditions and Neumann nodes.",
    )
    iboundformat: Optional[int] = Field(
        6,
        description="Format of the boundary file. 1: WWM, 3: WW3 (2D spectra in netcdf format only - LBCWA=T), 6: WW3 2D spectra in netcdf format with LBCSP=T.",
    )
    filewave: Optional[str | Path] = Field(
        "ww3.acs.SCHISM.nc",
        description="Boundary file defining boundary input from WW3. This will be generated by rompy if wave data is specified in the data input",
    )
    lindsprdeg: Optional[bool] = Field(
        True,
        description="For 1D wave spectra, defines whether directional spreading input is in degrees (true) or exponent (false).",
    )
    lparmdir: Optional[bool] = Field(
        False,
        description="If true, directional spreading is read from WBDS in exponential format. Only valid for 1D spectra.",
    )
    wbhs: Optional[float] = Field(
        2.0,
        description="Significant wave height at the boundary for parametric spectra.",
    )
    wbss: Optional[int] = Field(
        2,
        description="Spectral shape parameter. 1/-1: Pierson-Moskowitz, 2/-2: JONSWAP, 3/-3: all in one BIN, 4: Gauss. Sign determines if WBTP is peak (+) or mean period (-).",
    )
    wbtp: Optional[float] = Field(
        8.0,
        description="Peak or mean wave period at the boundary (seconds), depending on the sign of WBSS.",
    )
    wbdm: Optional[float] = Field(
        90.0, description="Average wave direction at the boundary (degrees)."
    )
    wbdsms: Optional[int] = Field(
        1, description="Directional spreading value format. 1: degrees, 2: exponent."
    )
    wbds: Optional[float] = Field(
        10.0, description="Directional spreading at the boundary (degrees or exponent)."
    )
    wbgauss: Optional[float] = Field(
        0.1, description="Factor for Gaussian distribution if WBSS=4."
    )
    wbpken: Optional[float] = Field(
        3.3, description="Peak enhancement factor for JONSWAP spectra if WBSS=2."
    )

    @field_validator("lbcse")
    @classmethod
    def validate_lbcse(cls, v):
        return v

    @field_validator("lbinter")
    @classmethod
    def validate_lbinter(cls, v, info):
        if v and not info.data.get("lbcse"):
            raise ValueError("LBINTER can only be True if LBCSE is True")
        return v

    @field_validator("lbcwa")
    @classmethod
    def validate_lbcwa(cls, v):
        return v

    @field_validator("lbcsp")
    @classmethod
    def validate_lbcsp(cls, v):
        return v

    @field_validator("linhom")
    @classmethod
    def validate_linhom(cls, v):
        return v

    # Not clear from sample inputs wheither these are strict requirements, or
    # are ignored if not met. Leaving out for now
    # @field_validator("lbsp1d")
    # @classmethod
    # def validate_lbsp1d(cls, v, info):
    #     if v and (not info.data.get("lbcsp") or info.data.get("linhom")):
    #         raise ValueError(
    #             "LBSP1D can only be True if LBCSP is True and LINHOM is False"
    #         )
    #     return v
    #
    # @field_validator("lbsp2d")
    # @classmethod
    # def validate_lbsp2d(cls, v, info):
    #     if v and (not info.data.get("lbcsp") or info.data.get("linhom")):
    #         raise ValueError(
    #             "LBSP2D can only be True if LBCSP is True and LINHOM is False"
    #         )
    #     return v

    @field_validator("begtc")
    @classmethod
    def validate_begtc(cls, v):
        if not re.match(r"\d{8}\.\d{6}", v):
            raise ValueError("BEGTC must be in YYYYMMDD.HHMMSS format")
        return v

    @field_validator("deltc")
    @classmethod
    def validate_deltc(cls, v):
        if v <= 0:
            raise ValueError("DELTC must be positive")
        return v

    @field_validator("unitc")
    @classmethod
    def validate_unitc(cls, v):
        if v not in ["HR", "MIN", "SEC"]:
            raise ValueError("UNITC must be HR, MIN, or SEC")
        return v

    @field_validator("endtc")
    @classmethod
    def validate_endtc(cls, v):
        if not re.match(r"\d{8}\.\d{6}", v):
            raise ValueError("ENDTC must be in YYYYMMDD.HHMMSS format")
        return v

    @field_validator("filebound")
    @classmethod
    def validate_filebound(cls, v):
        if not v.endswith(".gr3"):
            raise ValueError("FILEBOUND must have a .gr3 extension")
        return v

    @field_validator("iboundformat")
    @classmethod
    def validate_iboundformat(cls, v):
        if v not in [1, 3, 6]:
            raise ValueError("IBOUNDFORMAT must be 1, 3, or 6")
        return v

    @field_validator("filewave")
    @classmethod
    def validate_filewave(cls, v):
        if not str(v).endswith(".nc"):
            raise ValueError("FILEWAVE must have a .nc extension")
        return v

    @field_validator("lindsprdeg")
    @classmethod
    def validate_lindsprdeg(cls, v):
        return v

    @field_validator("lparmdir")
    @classmethod
    def validate_lparmdir(cls, v):
        return v

    @field_validator("wbhs")
    @classmethod
    def validate_wbhs(cls, v):
        if v <= 0:
            raise ValueError("WBHS must be positive")
        return v

    @field_validator("wbss")
    @classmethod
    def validate_wbss(cls, v):
        if v not in [-3, -2, -1, 1, 2, 3, 4]:
            raise ValueError("WBSS must be -3, -2, -1, 1, 2, 3, or 4")
        return v

    @field_validator("wbtp")
    @classmethod
    def validate_wbtp(cls, v):
        if v <= 0:
            raise ValueError("WBTP must be positive")
        return v

    @field_validator("wbdm")
    @classmethod
    def validate_wbdm(cls, v):
        if not 0 <= v <= 360:
            raise ValueError("WBDM must be between 0 and 360")
        return v

    @field_validator("wbdsms")
    @classmethod
    def validate_wbdsms(cls, v):
        if v not in [1, 2]:
            raise ValueError("WBDSMS must be 1 or 2")
        return v

    @field_validator("wbds")
    @classmethod
    def validate_wbds(cls, v):
        if v < 0:
            raise ValueError("WBDS must be non-negative")
        return v

    @field_validator("wbgauss")
    @classmethod
    def validate_wbgauss(cls, v, info):
        if info.data.get("wbss") == 4 and v <= 0:
            raise ValueError("WBGAUSS must be positive when WBSS is 4")
        return v

    @field_validator("wbpken")
    @classmethod
    def validate_wbpken(cls, v, info):
        if abs(info.data.get("wbss", 0)) == 2 and v <= 1:
            raise ValueError("WBPKEN must be greater than 1 when WBSS is 2 or -2")
        return v


class Engs(NamelistBaseModel):
    mesnl: Optional[int] = Field(
        1,
        description="Controls the nonlinear wave-wave interactions (NL4) using the Discrete Interaction Approximation. 1 enables the interactions, 0 disables them.",
    )
    mesin: Optional[int] = Field(
        1,
        description="Specifies the wind input formulation. Options include: 0 (no wind), 1 (Ardhuin et al.), 2 (ECMWF physics), 3 (Makin Stam), 4 (Babanin et al.), 5 (Cycle 3).",
    )
    ifric: Optional[int] = Field(
        1,
        description="Determines the formulation for the atmospheric boundary layer. Should be 1 when MESIN=1, and 4 when MESIN=3.",
    )
    mesbf: Optional[int] = Field(
        2,
        description="Selects the bottom friction formulation. 1 for JONSWAP (Default), 2 for Madsen et al. (1989), 3 for SHOWEX.",
    )
    fricc: Optional[float] = Field(
        0.006,
        description="Bottom friction coefficient or roughness, depending on MESBF. For MESBF=1: JONSWAP coefficient [0.038,0.067]. For MESBF=2: physical bottom roughness. For MESBF=3: D50 (negative value reads from SHOWEX_D50.gr3).",
    )
    mesbr: Optional[int] = Field(
        1, description="Enables (1) or disables (0) shallow water wave breaking."
    )
    ibreak: Optional[int] = Field(
        1,
        description="Selects the wave breaking formulation. Options range from 1 to 6, each representing a different model or approach.",
    )
    icrit: Optional[int] = Field(
        1,
        description="Specifies the wave breaking criterion. Options 1-6 represent different methods for determining the breaking point.",
    )
    brcr: Optional[float] = Field(
        0.73,
        description="Breaking criterion parameter. Its meaning depends on IBREAK and ICRIT settings.",
    )
    a_brcr: Optional[float] = Field(
        0.76,
        description="Coefficient used in ICRIT=4,5 for calculating the breaking criterion.",
    )
    b_brcr: Optional[float] = Field(
        0.29,
        description="Coefficient used in ICRIT=4,5 for calculating the breaking criterion.",
    )
    min_brcr: Optional[float] = Field(
        0.25, description="Minimum value for the breaking criterion when ICRIT=4,5."
    )
    max_brcr: Optional[float] = Field(
        0.8, description="Maximum value for the breaking criterion when ICRIT=4,5."
    )
    a_biph: Optional[float] = Field(
        0.2, description="Biphase coefficient, used when IBREAK=3."
    )
    br_coef_method: Optional[int] = Field(
        1,
        description="Method for determining the breaking coefficient. 1 for constant, 2 for adaptive.",
    )
    b_alp: Optional[float] = Field(
        0.5,
        description="Breaking coefficient. If BR_COEF_METHOD = 2, B_ALP should be around 40.",
    )
    zprof_break: Optional[int] = Field(
        2,
        description="Specifies the vertical distribution function of the wave breaking source term in 3D runs. Options 1-6 represent different distribution functions.",
    )
    bc_break: Optional[int] = Field(
        1,
        description="Controls the application of depth-limited breaking at boundaries. 1 to enable, 0 to disable.",
    )
    iroller: Optional[int] = Field(
        0,
        description="Enables (1) or disables (0) the wave roller model. Currently not in use.",
    )
    alprol: Optional[float] = Field(
        0.85,
        description="Alpha coefficient for the wave roller model, determining the energy transfer to the roller. Range: 0 to 1.",
    )
    meveg: Optional[int] = Field(
        0,
        description="Enables (1) or disables (0) vegetation effects. If enabled, isav must be 1 in param.nml.",
    )
    lmaxetot: Optional[bool] = Field(
        True,
        description="Controls the use of wave breaking limiter to limit shallow water wave height. True to enable, False to disable.",
    )
    mesds: Optional[int] = Field(
        1,
        description="Specifies the formulation for the whitecapping source function. Should have the same value as MESIN.",
    )
    mestr: Optional[int] = Field(
        1,
        description="Selects the formulation for triad 3 wave interactions. 0 (off), 1 (Lumped Triad Approx.), 2 (corrected LTA by Salmon et al. (2016)).",
    )
    trico: Optional[float] = Field(
        0.1,
        description="Proportionality constant (α_EB) for triad interactions. Default is 0.1.",
    )
    trira: Optional[float] = Field(
        2.5,
        description="Ratio of maximum frequency considered in triads over mean frequency. Suggested value is 2.5.",
    )
    triurs: Optional[float] = Field(
        0.1,
        description="Critical Ursell number for triad computations. Triads are not computed if Ursell number < TRIURS.",
    )

    @field_validator("mesnl")
    @classmethod
    def validate_mesnl(cls, v):
        if v not in [0, 1]:
            raise ValueError("MESNL must be either 0 or 1")
        return v

    @field_validator("mesin")
    @classmethod
    def validate_mesin(cls, v):
        if v not in range(6):
            raise ValueError("MESIN must be between 0 and 5")
        return v

    @field_validator("ifric")
    @classmethod
    def validate_ifric(cls, v, info):
        if info.data.get("mesin") == 1 and v != 1:
            raise ValueError("IFRIC should be 1 when MESIN is 1")
        elif info.data.get("mesin") == 3 and v != 4:
            raise ValueError("IFRIC should be 4 when MESIN is 3")
        return v

    @field_validator("mesbf")
    @classmethod
    def validate_mesbf(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("MESBF must be 1, 2, or 3")
        return v

    @field_validator("fricc")
    @classmethod
    def validate_fricc(cls, v, info):
        if info.data.get("mesbf") == 1 and not 0.038 <= v <= 0.067:
            raise ValueError("FRICC must be between 0.038 and 0.067 when MESBF is 1")
        return v

    @field_validator("mesbr")
    @classmethod
    def validate_mesbr(cls, v):
        if v not in [0, 1]:
            raise ValueError("MESBR must be either 0 or 1")
        return v

    @field_validator("ibreak")
    @classmethod
    def validate_ibreak(cls, v):
        if v not in range(1, 7):
            raise ValueError("IBREAK must be between 1 and 6")
        return v

    @field_validator("icrit")
    @classmethod
    def validate_icrit(cls, v):
        if v not in range(1, 7):
            raise ValueError("ICRIT must be between 1 and 6")
        return v

    @field_validator("brcr")
    @classmethod
    def validate_brcr(cls, v, info):
        if info.data.get("ibreak") in [1, 5] and v != 0.73:
            print("Warning: Default BRCR for IBREAK 1 or 5 is 0.73")
        elif info.data.get("ibreak") in [2, 3] and v != 0.42:
            print("Warning: Default BRCR for IBREAK 2 or 3 is 0.42")
        elif info.data.get("ibreak") == 4 and v != -1.3963:
            print("Warning: Default BRCR for IBREAK 4 is -1.3963")
        return v

    @field_validator("br_coef_method")
    @classmethod
    def validate_br_coef_method(cls, v):
        if v not in [1, 2]:
            raise ValueError("BR_COEF_METHOD must be either 1 or 2")
        return v

    @field_validator("b_alp")
    @classmethod
    def validate_b_alp(cls, v, info):
        if info.data.get("br_coef_method") == 2 and v != 40:
            print("Warning: B_ALP should be around 40 when BR_COEF_METHOD is 2")
        return v

    @field_validator("zprof_break")
    @classmethod
    def validate_zprof_break(cls, v):
        if v not in range(1, 7):
            raise ValueError("ZPROF_BREAK must be between 1 and 6")
        return v

    @field_validator("bc_break")
    @classmethod
    def validate_bc_break(cls, v):
        if v not in [0, 1]:
            raise ValueError("BC_BREAK must be either 0 or 1")
        return v

    @field_validator("iroller")
    @classmethod
    def validate_iroller(cls, v):
        if v not in [0, 1]:
            raise ValueError("IROLLER must be either 0 or 1")
        return v

    @field_validator("alprol")
    @classmethod
    def validate_alprol(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("ALPROL must be between 0 and 1")
        return v

    @field_validator("meveg")
    @classmethod
    def validate_meveg(cls, v):
        if v not in [0, 1]:
            raise ValueError("MEVEG must be either 0 or 1")
        return v

    @field_validator("mesds")
    @classmethod
    def validate_mesds(cls, v, info):
        if v != info.data.get("mesin"):
            raise ValueError("MESDS should have the same value as MESIN")
        return v

    @field_validator("mestr")
    @classmethod
    def validate_mestr(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("MESTR must be 0, 1, or 2")
        return v


class Nums(NamelistBaseModel):
    icomp: Optional[int] = Field(
        3,
        description="Controls the integration scheme for splitting and advection. 0: All explicit. 1: Implicit geographical advection. 2: Implicit advection and semi-implicit source terms. 3: Fully implicit, no splitting.",
    )
    amethod: Optional[int] = Field(
        7,
        description="Controls the advection method in geographical space. Values 0-7 represent different schemes, including explicit, implicit, and PETSc-based methods.",
    )
    smethod: Optional[int] = Field(
        6,
        description="Controls the integration method for source terms. 0: No source terms. 1-6: Various splitting and integration schemes.",
    )
    dmethod: Optional[int] = Field(
        2,
        description="Controls the numerical method in directional space. 0: No advection. 1-4: Various schemes including Crank-Nicholson, Ultimate Quickest, RK5-WENO, and Explicit FVM Upwind.",
    )
    rtheta: Optional[float] = Field(
        0.5,
        description="Weighing factor for DMETHOD = 1. Only useful for Crank Nicholson integration with CFL <= 2.",
    )
    litersplit: Optional[bool] = Field(
        False,
        description="Splitting method. True: double Strang split. False: simple split (more efficient).",
    )
    lfilterth: Optional[bool] = Field(
        False,
        description="Use a CFL filter to limit the advection velocity in directional space. Similar to WW3, but mostly unused as WWMII is always stable.",
    )
    maxcflth: Optional[float] = Field(
        1.0,
        description="Maximum CFL number in Theta space. Used only if LFILTERTH=True.",
    )
    fmethod: Optional[int] = Field(
        1,
        description="Controls the numerical method in frequency space. 0: No advection. 1: Ultimate Quickest as in WW3 (best).",
    )
    lfiltersig: Optional[bool] = Field(
        False,
        description="Limit the advection velocity in frequency space. Usually False.",
    )
    maxcflsig: Optional[float] = Field(
        1.0,
        description="Maximum CFL number in frequency space. Used only if LFILTERSIG=True.",
    )
    llimt: Optional[bool] = Field(
        True, description="Switch on/off Action limiter. Must mostly be turned on."
    )
    melim: Optional[int] = Field(
        1,
        description="Formulation for the action limiter. 1: WAM group (1988). 2: Hersbach Janssen (1999). 3: For Cycle 4 formulation.",
    )
    limfak: Optional[float] = Field(
        0.1,
        description="Proportionality coefficient for the action limiter. MAX_DAC_DT = Limfak * Limiter.",
    )
    ldifr: Optional[bool] = Field(
        False,
        description="Use phase decoupled diffraction approximation. Usually True; if crash, use False.",
    )
    idiffr: Optional[int] = Field(
        1,
        description="Extended WAE accounting for higher order effects. 1: Holthuijsen et al. 2: Liau et al. 3: Toledo et al.",
    )
    lconv: Optional[bool] = Field(
        False,
        description="Estimate convergence criteria and write to disk (quasi-steady - qstea.out).",
    )
    lcfl: Optional[bool] = Field(
        False, description="Write out CFL numbers. Use False to save time."
    )
    nqsiter: Optional[int] = Field(
        1,
        description="Number of quasi-steady (Q-S) sub-divisions within each WWM time step.",
    )
    qsconv1: Optional[float] = Field(
        0.98,
        description="Fraction of grid points that must fulfill absolute wave height criteria EPSH1.",
    )
    qsconv2: Optional[float] = Field(
        0.98,
        description="Fraction of grid points that must fulfill relative wave height criteria EPSH2.",
    )
    qsconv3: Optional[float] = Field(
        0.98,
        description="Fraction of grid points that must fulfill sum. rel. wave action criteria EPSH3.",
    )
    qsconv4: Optional[float] = Field(
        0.98,
        description="Fraction of grid points that must fulfill rel. avg. wave steepness criteria EPSH4.",
    )
    qsconv5: Optional[float] = Field(
        0.98,
        description="Fraction of grid points that must fulfill avg. rel. wave period criteria EPSH5.",
    )
    lexpimp: Optional[bool] = Field(
        False,
        description="Use implicit schemes for frequencies lower than FREQEXP. Used only if ICOMP=0.",
    )
    freqexp: Optional[float] = Field(
        0.1,
        description="Minimum frequency for explicit schemes. Only used if LEXPIMP=True and ICOMP=0.",
    )
    epsh1: Optional[float] = Field(
        0.01, description="Convergence criteria for relative wave height."
    )
    epsh2: Optional[float] = Field(
        0.01, description="Convergence criteria for absolute wave height."
    )
    epsh3: Optional[float] = Field(
        0.01, description="Convergence criteria for the relative sum of wave action."
    )
    epsh4: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the relative average wave steepness.",
    )
    epsh5: Optional[float] = Field(
        0.01, description="Convergence criteria for the relative average wave period."
    )
    lvector: Optional[bool] = Field(
        False,
        description="Use optimized propagation routines for large high performance computers. Try False first.",
    )
    ivector: Optional[int] = Field(
        2,
        description="Used if LVECTOR=True. Different flavors of communications and propagation styles.",
    )
    ladvtest: Optional[bool] = Field(
        False, description="For testing the advection schemes."
    )
    lchkconv: Optional[bool] = Field(
        False,
        description="Needs to be set to True for quasi-steady mode to compute and check the QSCONVi criteria.",
    )
    dtmin_dyn: Optional[float] = Field(
        1.0,
        description="Minimum time step (seconds) for dynamic integration. Controls the smallest time step for triads in SMETHOD.",
    )
    ndyniter: Optional[int] = Field(
        100,
        description="Maximum iterations for dynamic scheme before limiter is applied in the last step.",
    )
    dtmin_sin: Optional[float] = Field(
        1.0,
        description="Minimum time step for the full fractional step method, where each source term is integrated with its own fractional step.",
    )
    dtmin_snl4: Optional[float] = Field(
        1.0,
        description="Minimum time step for SNL4 source term in fractional step method.",
    )
    dtmin_sds: Optional[float] = Field(
        1.0,
        description="Minimum time step for SDS source term in fractional step method.",
    )
    dtmin_snl3: Optional[float] = Field(
        1.0,
        description="Minimum time step for SNL3 source term in fractional step method.",
    )
    dtmin_sbr: Optional[float] = Field(
        0.1,
        description="Minimum time step for SBR source term in fractional step method.",
    )
    dtmin_sbf: Optional[float] = Field(
        1.0,
        description="Minimum time step for SBF source term in fractional step method.",
    )
    ndyniter_sin: Optional[int] = Field(
        10,
        description="Maximum iterations for SIN source term in fractional step approach.",
    )
    ndyniter_snl4: Optional[int] = Field(
        10,
        description="Maximum iterations for SNL4 source term in fractional step approach.",
    )
    ndyniter_sds: Optional[int] = Field(
        10,
        description="Maximum iterations for SDS source term in fractional step approach.",
    )
    ndyniter_sbr: Optional[int] = Field(
        10,
        description="Maximum iterations for SBR source term in fractional step approach.",
    )
    ndyniter_snl3: Optional[int] = Field(
        10,
        description="Maximum iterations for SNL3 source term in fractional step approach.",
    )
    ndyniter_sbf: Optional[int] = Field(
        10,
        description="Maximum iterations for SBF source term in fractional step approach.",
    )
    lsoubound: Optional[bool] = Field(
        False,
        description="Do source terms on boundary. Useful for harbor studies and flume experiments.",
    )
    wae_solverthr: Optional[float] = Field(
        1e-06,
        description="Threshold for the Block-Jacobi or Block-Gauss-Seider solver.",
    )
    maxiter: Optional[int] = Field(
        1000, description="Maximum number of iterations for solver."
    )
    pmin: Optional[float] = Field(
        1.0, description="Maximum percentage of non-converged grid points allowed."
    )
    lnaninfchk: Optional[bool] = Field(
        False, description="Check for NaN and INF. Usually turned off for efficiency."
    )
    lzeta_setup: Optional[bool] = Field(
        False, description="Compute wave setup (simple momentum equation)."
    )
    zeta_meth: Optional[int] = Field(
        0, description="Method for wave setup calculation."
    )
    lsourceswam: Optional[bool] = Field(
        False, description="Use ECMWF WAM formulation for deep water physics."
    )
    block_gauss_seidel: Optional[bool] = Field(
        True,
        description="Use Gauss-Seidel method on each computer block. Faster and uses less memory, but iterations depend on number of processors.",
    )
    lnonl: Optional[bool] = Field(
        False,
        description="Solve the nonlinear system using simpler algorithm (Patankar).",
    )
    aspar_local_level: Optional[int] = Field(
        0,
        description="ASPAR locality level. Controls memory allocation and optimization strategies.",
    )
    l_solver_norm: Optional[bool] = Field(
        False,
        description="Compute solver norm ||A*x-b|| as termination check of Jacobi-Gauss-Seidel solver. Increases cost if True.",
    )
    laccel: Optional[bool] = Field(False, description="Enable acceleration for solver.")

    @field_validator("icomp")
    @classmethod
    def check_icomp(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("ICOMP must be 0, 1, 2, or 3")
        return v

    @field_validator("amethod")
    @classmethod
    def check_amethod(cls, v):
        if v not in range(8):
            raise ValueError("AMETHOD must be between 0 and 7")
        return v

    @field_validator("smethod")
    @classmethod
    def check_smethod(cls, v):
        if v not in range(7):
            raise ValueError("SMETHOD must be between 0 and 6")
        return v

    @field_validator("dmethod")
    @classmethod
    def check_dmethod(cls, v):
        if v not in range(5):
            raise ValueError("DMETHOD must be between 0 and 4")
        return v

    @field_validator("rtheta")
    @classmethod
    def check_rtheta(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("RTHETA must be between 0 and 1")
        return v

    @field_validator("maxcflth")
    @classmethod
    def check_maxcflth(cls, v):
        if v <= 0:
            raise ValueError("MAXCFLTH must be positive")
        return v

    @field_validator("fmethod")
    @classmethod
    def check_fmethod(cls, v):
        if v not in [0, 1]:
            raise ValueError("FMETHOD must be 0 or 1")
        return v

    @field_validator("maxcflsig")
    @classmethod
    def check_maxcflsig(cls, v):
        if v <= 0:
            raise ValueError("MAXCFLSIG must be positive")
        return v

    @field_validator("melim")
    @classmethod
    def check_melim(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("MELIM must be 1, 2, or 3")
        return v

    @field_validator("limfak")
    @classmethod
    def check_limfak(cls, v):
        if v <= 0:
            raise ValueError("LIMFAK must be positive")
        return v

    @field_validator("idiffr")
    @classmethod
    def check_idiffr(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("IDIFFR must be 1, 2, or 3")
        return v

    @field_validator("nqsiter")
    @classmethod
    def check_nqsiter(cls, v):
        if v < 1:
            raise ValueError("NQSITER must be at least 1")
        return v

    @field_validator("qsconv1")
    @classmethod
    def check_qsconv1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV1 must be between 0 and 1")
        return v

    @field_validator("qsconv2")
    @classmethod
    def check_qsconv2(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV2 must be between 0 and 1")
        return v

    @field_validator("qsconv3")
    @classmethod
    def check_qsconv3(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV3 must be between 0 and 1")
        return v

    @field_validator("qsconv4")
    @classmethod
    def check_qsconv4(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV4 must be between 0 and 1")
        return v

    @field_validator("qsconv5")
    @classmethod
    def check_qsconv5(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("QSCONV5 must be between 0 and 1")
        return v

    @field_validator("freqexp")
    @classmethod
    def check_freqexp(cls, v):
        if v <= 0:
            raise ValueError("FREQEXP must be positive")
        return v

    @field_validator("epsh1")
    @classmethod
    def check_epsh1(cls, v):
        if v <= 0:
            raise ValueError("EPSH1 must be positive")
        return v

    @field_validator("epsh2")
    @classmethod
    def check_epsh2(cls, v):
        if v <= 0:
            raise ValueError("EPSH2 must be positive")
        return v

    @field_validator("epsh3")
    @classmethod
    def check_epsh3(cls, v):
        if v <= 0:
            raise ValueError("EPSH3 must be positive")
        return v

    @field_validator("epsh4")
    @classmethod
    def check_epsh4(cls, v):
        if v <= 0:
            raise ValueError("EPSH4 must be positive")
        return v

    @field_validator("epsh5")
    @classmethod
    def check_epsh5(cls, v):
        if v <= 0:
            raise ValueError("EPSH5 must be positive")
        return v

    @field_validator("ivector")
    @classmethod
    def check_ivector(cls, v):
        if v not in range(1, 7):
            raise ValueError("IVECTOR must be between 1 and 6")
        return v

    @field_validator("dtmin_dyn")
    @classmethod
    def check_dtmin_dyn(cls, v):
        if v <= 0:
            raise ValueError("DTMIN_DYN must be positive")
        return v

    @field_validator("ndyniter")
    @classmethod
    def check_ndyniter(cls, v):
        if v < 1:
            raise ValueError("NDYNITER must be at least 1")
        return v

    @field_validator("dtmin_sin")
    @classmethod
    def check_dtmin_sin(cls, v):
        if v <= 0:
            raise ValueError("DTMIN_SIN must be positive")
        return v

    @field_validator("dtmin_snl4")
    @classmethod
    def check_dtmin_snl4(cls, v):
        if v <= 0:
            raise ValueError("DTMIN_SNL4 must be positive")
        return v

    @field_validator("dtmin_sds")
    @classmethod
    def check_dtmin_sds(cls, v):
        if v <= 0:
            raise ValueError("DTMIN_SDS must be positive")
        return v

    @field_validator("dtmin_snl3")
    @classmethod
    def check_dtmin_snl3(cls, v):
        if v <= 0:
            raise ValueError("DTMIN_SNL3 must be positive")
        return v

    @field_validator("dtmin_sbr")
    @classmethod
    def check_dtmin_sbr(cls, v):
        if v <= 0:
            raise ValueError("DTMIN_SBR must be positive")
        return v

    @field_validator("dtmin_sbf")
    @classmethod
    def check_dtmin_sbf(cls, v):
        if v <= 0:
            raise ValueError("DTMIN_SBF must be positive")
        return v

    @field_validator("ndyniter_sin")
    @classmethod
    def check_ndyniter_sin(cls, v):
        if v < 1:
            raise ValueError("NDYNITER_SIN must be at least 1")
        return v

    @field_validator("ndyniter_snl4")
    @classmethod
    def check_ndyniter_snl4(cls, v):
        if v < 1:
            raise ValueError("NDYNITER_SNL4 must be at least 1")
        return v

    @field_validator("ndyniter_sds")
    @classmethod
    def check_ndyniter_sds(cls, v):
        if v < 1:
            raise ValueError("NDYNITER_SDS must be at least 1")
        return v

    @field_validator("ndyniter_sbr")
    @classmethod
    def check_ndyniter_sbr(cls, v):
        if v < 1:
            raise ValueError("NDYNITER_SBR must be at least 1")
        return v

    @field_validator("ndyniter_snl3")
    @classmethod
    def check_ndyniter_snl3(cls, v):
        if v < 1:
            raise ValueError("NDYNITER_SNL3 must be at least 1")
        return v

    @field_validator("ndyniter_sbf")
    @classmethod
    def check_ndyniter_sbf(cls, v):
        if v < 1:
            raise ValueError("NDYNITER_SBF must be at least 1")
        return v

    @field_validator("wae_solverthr")
    @classmethod
    def check_wae_solverthr(cls, v):
        if v <= 0:
            raise ValueError("WAE_SOLVERTHR must be positive")
        return v

    @field_validator("maxiter")
    @classmethod
    def check_maxiter(cls, v):
        if v < 1:
            raise ValueError("MAXITER must be at least 1")
        return v

    @field_validator("pmin")
    @classmethod
    def check_pmin(cls, v):
        if not 0 <= v <= 100:
            raise ValueError("PMIN must be between 0 and 100")
        return v

    @field_validator("zeta_meth")
    @classmethod
    def check_zeta_meth(cls, v):
        if v < 0:
            raise ValueError("ZETA_METH must be non-negative")
        return v

    @field_validator("aspar_local_level")
    @classmethod
    def check_aspar_local_level(cls, v):
        if v not in range(6):
            raise ValueError("ASPAR_LOCAL_LEVEL must be between 0 and 5")
        return v


class History(NamelistBaseModel):
    begtc: Optional[str] = Field(
        "20200101.000000",
        description="Start output time in 'yyyymmdd.hhmmss' format. Must fit within the simulation time, otherwise no output. Defaults to PROC%BEGTC if not specified.",
    )
    deltc: Optional[int] = Field(
        3600,
        description="Time step for output in seconds. If smaller than simulation time step, the latter is used. Used for better 1D and 2D spectra analysis.",
    )
    unitc: Optional[str] = Field(
        "SEC",
        description="Unit of time for DELTC. Currently only supports 'SEC' for seconds.",
    )
    endtc: Optional[str] = Field(
        "20200201.000000",
        description="Stop time for output in 'yyyymmdd.hhmmss' format. Defaults to PROC%ENDC if not specified.",
    )
    definetc: Optional[int] = Field(
        -1,
        description="Time scoop (in seconds) for history files. If negative or unset, only one file is generated. For example, 86400 creates daily output files.",
    )
    outstyle: Optional[str] = Field(
        "NO",
        description="Output option. 'NO' for no output, 'NC' for netCDF, 'XFN' for XFN (default), 'SHP' for DARKO SHP output.",
    )
    multipleout: Optional[int] = Field(
        0,
        description="Output file configuration. 0 for single netCDF file using MPI_reduce (default), 1 for separate netCDF files for each process.",
    )
    use_single_out: Optional[bool] = Field(
        True,
        description="Use single precision in the output of model variables. True by default.",
    )
    paramwrite: Optional[bool] = Field(
        True,
        description="Write the physical parametrization and chosen numerical method in the netCDF file. True by default.",
    )
    gridwrite: Optional[bool] = Field(
        True, description="Write the grid in the netCDF history file. True by default."
    )
    printmma: Optional[bool] = Field(
        False,
        description="Print minimum, maximum and average value of statistics during runtime. Requires MPI_REDUCE. False by default.",
    )
    fileout: Optional[str] = Field(
        "wwm_hist.nc", description="Name of the output file."
    )
    hs: Optional[bool] = Field(True, description="Output significant wave height.")
    tm01: Optional[bool] = Field(True, description="Output mean period.")
    tm02: Optional[bool] = Field(True, description="Output zero-crossing mean period.")
    klm: Optional[bool] = Field(False, description="Output mean wave number.")
    wlm: Optional[bool] = Field(False, description="Output mean wave length.")
    etotc: Optional[bool] = Field(False, description="Output variable ETOTC.")
    etots: Optional[bool] = Field(False, description="Output variable ETOTS.")
    dm: Optional[bool] = Field(True, description="Output mean wave direction.")
    dspr: Optional[bool] = Field(True, description="Output directional spreading.")
    tppd: Optional[bool] = Field(
        False,
        description="Output direction of the peak (check source code for details).",
    )
    tpp: Optional[bool] = Field(True, description="Output peak period.")
    cpp: Optional[bool] = Field(False, description="Output peak phase velocity.")
    wnpp: Optional[bool] = Field(False, description="Output peak wave number.")
    cgpp: Optional[bool] = Field(False, description="Output peak group speed.")
    kpp: Optional[bool] = Field(False, description="Output peak wave number.")
    lpp: Optional[bool] = Field(False, description="Output peak wave length.")
    peakd: Optional[bool] = Field(True, description="Output peak direction.")
    peakdspr: Optional[bool] = Field(
        True, description="Output peak directional spreading."
    )
    dpeak: Optional[bool] = Field(False, description="Output peak direction.")
    ubot: Optional[bool] = Field(False, description="Output bottom excursion velocity.")
    orbital: Optional[bool] = Field(
        False, description="Output bottom orbital velocity."
    )
    botexper: Optional[bool] = Field(False, description="Output bottom excursion.")
    tmbot: Optional[bool] = Field(False, description="Output bottom period.")
    ursell: Optional[bool] = Field(False, description="Output Ursell number.")
    ufric: Optional[bool] = Field(False, description="Output air friction velocity.")
    z0: Optional[bool] = Field(False, description="Output air roughness length.")
    alpha_ch: Optional[bool] = Field(
        False, description="Output Charnock coefficient for air."
    )
    windx: Optional[bool] = Field(False, description="Output wind in X direction.")
    windy: Optional[bool] = Field(False, description="Output wind in Y direction.")
    cd: Optional[bool] = Field(False, description="Output drag coefficient.")
    currtx: Optional[bool] = Field(False, description="Output current in X direction.")
    currty: Optional[bool] = Field(False, description="Output current in Y direction.")
    watlev: Optional[bool] = Field(False, description="Output water level.")
    watlevold: Optional[bool] = Field(
        False, description="Output water level at previous time step."
    )
    depdt: Optional[bool] = Field(
        False, description="Output change of water level in time."
    )
    dep: Optional[bool] = Field(False, description="Output depth.")
    tauw: Optional[bool] = Field(
        True, description="Output surface stress from the wave."
    )
    tauhf: Optional[bool] = Field(
        False, description="Output high frequency surface stress."
    )
    tautot: Optional[bool] = Field(True, description="Output total surface stress.")
    stokessurfx: Optional[bool] = Field(
        True, description="Output surface Stokes drift in X direction."
    )
    stokessurfy: Optional[bool] = Field(
        True, description="Output surface Stokes drift in Y direction."
    )
    stokesbarox: Optional[bool] = Field(
        False, description="Output barotropic Stokes drift in X direction."
    )
    stokesbaroy: Optional[bool] = Field(
        False, description="Output barotropic Stokes drift in Y direction."
    )
    rsxx: Optional[bool] = Field(False, description="Output RSXX potential of LH.")
    rsxy: Optional[bool] = Field(False, description="Output RSXY potential of LH.")
    rsyy: Optional[bool] = Field(False, description="Output RSYY potential of LH.")
    cfl1: Optional[bool] = Field(False, description="Output CFL number 1.")
    cfl2: Optional[bool] = Field(False, description="Output CFL number 2.")
    cfl3: Optional[bool] = Field(False, description="Output CFL number 3.")

    @field_validator("begtc")
    @classmethod
    def validate_begtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use yyyymmdd.hhmmss")

    @field_validator("deltc")
    @classmethod
    def validate_deltc(cls, v):
        if v <= 0:
            raise ValueError("DELTC must be positive")
        return v

    @field_validator("unitc")
    @classmethod
    def validate_unitc(cls, v):
        if v.upper() != "SEC":
            raise ValueError("UNITC must be SEC")
        return v.upper()

    @field_validator("endtc")
    @classmethod
    def validate_endtc(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use yyyymmdd.hhmmss")

    @field_validator("definetc")
    @classmethod
    def validate_definetc(cls, v):
        if not isinstance(v, int):
            raise ValueError("DEFINETC must be an integer")
        return v

    @field_validator("outstyle")
    @classmethod
    def validate_outstyle(cls, v):
        valid_options = ["NO", "NC", "XFN", "SHP"]
        if v.upper() not in valid_options:
            raise ValueError(f"OUTSTYLE must be one of {valid_options}")
        return v.upper()

    @field_validator("multipleout")
    @classmethod
    def validate_multipleout(cls, v):
        if v not in [0, 1]:
            raise ValueError("MULTIPLEOUT must be 0 or 1")
        return v

    @field_validator("use_single_out")
    @classmethod
    def validate_use_single_out(cls, v):
        return v

    @field_validator("paramwrite")
    @classmethod
    def validate_paramwrite(cls, v):
        return v

    @field_validator("gridwrite")
    @classmethod
    def validate_gridwrite(cls, v):
        return v

    @field_validator("printmma")
    @classmethod
    def validate_printmma(cls, v):
        return v

    @field_validator("fileout")
    @classmethod
    def validate_fileout(cls, v):
        if not v.endswith(".nc"):
            raise ValueError("FILEOUT must have .nc extension")
        return v

    @field_validator("hs")
    @classmethod
    def validate_hs(cls, v):
        return v

    @field_validator("tm01")
    @classmethod
    def validate_tm01(cls, v):
        return v

    @field_validator("tm02")
    @classmethod
    def validate_tm02(cls, v):
        return v

    @field_validator("klm")
    @classmethod
    def validate_klm(cls, v):
        return v

    @field_validator("wlm")
    @classmethod
    def validate_wlm(cls, v):
        return v

    @field_validator("etotc")
    @classmethod
    def validate_etotc(cls, v):
        return v

    @field_validator("etots")
    @classmethod
    def validate_etots(cls, v):
        return v

    @field_validator("dm")
    @classmethod
    def validate_dm(cls, v):
        return v

    @field_validator("dspr")
    @classmethod
    def validate_dspr(cls, v):
        return v

    @field_validator("tppd")
    @classmethod
    def validate_tppd(cls, v):
        return v

    @field_validator("tpp")
    @classmethod
    def validate_tpp(cls, v):
        return v

    @field_validator("cpp")
    @classmethod
    def validate_cpp(cls, v):
        return v

    @field_validator("wnpp")
    @classmethod
    def validate_wnpp(cls, v):
        return v

    @field_validator("cgpp")
    @classmethod
    def validate_cgpp(cls, v):
        return v

    @field_validator("kpp")
    @classmethod
    def validate_kpp(cls, v):
        return v

    @field_validator("lpp")
    @classmethod
    def validate_lpp(cls, v):
        return v

    @field_validator("peakd")
    @classmethod
    def validate_peakd(cls, v):
        return v

    @field_validator("peakdspr")
    @classmethod
    def validate_peakdspr(cls, v):
        return v

    @field_validator("dpeak")
    @classmethod
    def validate_dpeak(cls, v):
        return v

    @field_validator("ubot")
    @classmethod
    def validate_ubot(cls, v):
        return v

    @field_validator("orbital")
    @classmethod
    def validate_orbital(cls, v):
        return v

    @field_validator("botexper")
    @classmethod
    def validate_botexper(cls, v):
        return v

    @field_validator("tmbot")
    @classmethod
    def validate_tmbot(cls, v):
        return v

    @field_validator("ursell")
    @classmethod
    def validate_ursell(cls, v):
        return v

    @field_validator("ufric")
    @classmethod
    def validate_ufric(cls, v):
        return v

    @field_validator("z0")
    @classmethod
    def validate_z0(cls, v):
        return v

    @field_validator("alpha_ch")
    @classmethod
    def validate_alpha_ch(cls, v):
        return v

    @field_validator("windx")
    @classmethod
    def validate_windx(cls, v):
        return v

    @field_validator("windy")
    @classmethod
    def validate_windy(cls, v):
        return v

    @field_validator("cd")
    @classmethod
    def validate_cd(cls, v):
        return v

    @field_validator("currtx")
    @classmethod
    def validate_currtx(cls, v):
        return v

    @field_validator("currty")
    @classmethod
    def validate_currty(cls, v):
        return v

    @field_validator("watlev")
    @classmethod
    def validate_watlev(cls, v):
        return v

    @field_validator("watlevold")
    @classmethod
    def validate_watlevold(cls, v):
        return v

    @field_validator("depdt")
    @classmethod
    def validate_depdt(cls, v):
        return v

    @field_validator("dep")
    @classmethod
    def validate_dep(cls, v):
        return v

    @field_validator("tauw")
    @classmethod
    def validate_tauw(cls, v):
        return v

    @field_validator("tauhf")
    @classmethod
    def validate_tauhf(cls, v):
        return v

    @field_validator("tautot")
    @classmethod
    def validate_tautot(cls, v):
        return v

    @field_validator("stokessurfx")
    @classmethod
    def validate_stokessurfx(cls, v):
        return v

    @field_validator("stokessurfy")
    @classmethod
    def validate_stokessurfy(cls, v):
        return v

    @field_validator("stokesbarox")
    @classmethod
    def validate_stokesbarox(cls, v):
        return v

    @field_validator("stokesbaroy")
    @classmethod
    def validate_stokesbaroy(cls, v):
        return v

    @field_validator("rsxx")
    @classmethod
    def validate_rsxx(cls, v):
        return v

    @field_validator("rsxy")
    @classmethod
    def validate_rsxy(cls, v):
        return v

    @field_validator("rsyy")
    @classmethod
    def validate_rsyy(cls, v):
        return v

    @field_validator("cfl1")
    @classmethod
    def validate_cfl1(cls, v):
        return v

    @field_validator("cfl2")
    @classmethod
    def validate_cfl2(cls, v):
        return v

    @field_validator("cfl3")
    @classmethod
    def validate_cfl3(cls, v):
        return v


class Station(NamelistBaseModel):
    begtc: Optional[str] = Field(
        "20200101.000000",
        description="Start simulation time in 'yyyymmdd.hhmmss' format. Must fit the simulation time, otherwise no output is generated. Defaults to PROC%BEGTC if not specified.",
    )
    deltc: Optional[int] = Field(
        3600,
        description="Time step for output in seconds. If smaller than simulation time step, the latter is used. Used for better 1D and 2D spectra analysis.",
    )
    unitc: Optional[str] = Field(
        "SEC", description="Time unit for DELTC. Only 'SEC' is currently supported."
    )
    endtc: Optional[str] = Field(
        "20200201.000000",
        description="Stop time for simulation in 'yyyymmdd.hhmmss' format. Defaults to PROC%ENDC if not specified.",
    )
    definetc: Optional[int] = Field(
        -1,
        description="Time for definition of station files in seconds. If negative or unset, only one file is generated. Otherwise, it defines the interval for creating output files (e.g., 86400 for daily files).",
    )
    outstyle: Optional[str] = Field(
        "NC",
        description="Output option. 'NO' for no output, 'STE' for classic station output (default), 'NC' for netCDF output.",
    )
    multipleout: Optional[int] = Field(
        0,
        description="Output file configuration. 0 for a single netCDF file using MPI_reduce (default), 1 for separate netCDF files for each process.",
    )
    use_single_out: Optional[bool] = Field(
        True,
        description="Use single precision in the output of model variables. True by default.",
    )
    paramwrite: Optional[bool] = Field(
        True,
        description="Write the physical parametrization and chosen numerical method in the netCDF file. True by default.",
    )
    fileout: Optional[str] = Field(
        "wwm_sta.nc", description="Output file name (not used)."
    )
    loutiter: Optional[bool] = Field(
        False, description="Boolean flag for output iteration."
    )
    iouts: Optional[int] = Field(12, description="Number of output stations.")
    nouts: Optional[list[str]] = Field(
        [
            "BatemansBay",
            "Brisbane",
            "ByronBay",
            "Cairns",
            "CoffsHbr",
            "CrowdyHead",
            "Eden",
            "HayPt",
            "Mackay",
            "PortKembla",
            "Tsv",
            "Mandurah",
        ],
        description="Names of output stations.",
    )
    xouts: Optional[list] = Field(
        [
            150.31972,
            153.63166,
            153.745,
            145.715167,
            153.27722,
            152.85333,
            150.15833,
            149.31025,
            149.5467,
            151.01667,
            147.059333,
            115.572227,
        ],
        description="Longitude coordinates of output stations.",
    )
    youts: Optional[list] = Field(
        [
            -35.75528,
            -27.48716,
            -28.67167,
            -16.7305,
            -30.34361,
            -31.82694,
            -37.175,
            -21.2715,
            -21.037333,
            -34.46694,
            -19.159167,
            -32.452787,
        ],
        description="Latitude coordinates of output stations.",
    )
    cutoff: Optional[float] = Field(
        0.0,
        description="Cutoff frequency (Hz) for each station, consistent with buoys.",
    )
    lsp1d: Optional[bool] = Field(
        True, description="Enable 1D spectral station output."
    )
    lsp2d: Optional[bool] = Field(
        True, description="Enable 2D spectral station output."
    )
    lsigmax: Optional[bool] = Field(
        True,
        description="Adjust the cut-off frequency for the output (e.g., consistent with buoy cut-off frequency).",
    )
    ac: Optional[bool] = Field(True, description="Output spectrum.")
    wk: Optional[bool] = Field(False, description="Output variable WK.")
    acout_1d: Optional[bool] = Field(False, description="Output variable ACOUT_1D.")
    acout_2d: Optional[bool] = Field(False, description="Output variable ACOUT_2D.")
    hs: Optional[bool] = Field(True, description="Output significant wave height.")
    tm01: Optional[bool] = Field(True, description="Output mean period.")
    tm02: Optional[bool] = Field(True, description="Output zero-crossing mean period.")
    klm: Optional[bool] = Field(False, description="Output mean wave number.")
    wlm: Optional[bool] = Field(False, description="Output mean wave length.")
    etotc: Optional[bool] = Field(False, description="Output variable ETOTC.")
    etots: Optional[bool] = Field(False, description="Output variable ETOTS.")
    dm: Optional[bool] = Field(True, description="Output mean wave direction.")
    dspr: Optional[bool] = Field(True, description="Output directional spreading.")
    tppd: Optional[bool] = Field(True, description="Output discrete peak period.")
    tpp: Optional[bool] = Field(True, description="Output peak period.")
    cpp: Optional[bool] = Field(False, description="Output variable CPP.")
    wnpp: Optional[bool] = Field(False, description="Output peak wave number.")
    cgpp: Optional[bool] = Field(False, description="Output peak group speed.")
    kpp: Optional[bool] = Field(False, description="Output peak wave number.")
    lpp: Optional[bool] = Field(False, description="Output peak wavelength.")
    peakd: Optional[bool] = Field(True, description="Output peak direction.")
    peakdspr: Optional[bool] = Field(
        True, description="Output peak directional spreading."
    )
    dpeak: Optional[bool] = Field(False, description="Output variable DPEAK.")
    ubot: Optional[bool] = Field(False, description="Output variable UBOT.")
    orbital: Optional[bool] = Field(False, description="Output orbital velocity.")
    botexper: Optional[bool] = Field(
        False, description="Output bottom excursion period."
    )
    tmbot: Optional[bool] = Field(False, description="Output variable TMBOT.")
    ursell: Optional[bool] = Field(False, description="Output Ursell number.")
    ufric: Optional[bool] = Field(False, description="Output air friction velocity.")
    z0: Optional[bool] = Field(False, description="Output air roughness length.")
    alpha_ch: Optional[bool] = Field(
        False, description="Output Charnock coefficient for air."
    )
    windx: Optional[bool] = Field(True, description="Output wind in X direction.")
    windy: Optional[bool] = Field(True, description="Output wind in Y direction.")
    cd: Optional[bool] = Field(False, description="Output drag coefficient.")
    currtx: Optional[bool] = Field(False, description="Output current in X direction.")
    currty: Optional[bool] = Field(False, description="Output current in Y direction.")
    watlev: Optional[bool] = Field(False, description="Output water level.")
    watlevold: Optional[bool] = Field(
        False, description="Output water level at previous time step."
    )
    depdt: Optional[bool] = Field(
        False, description="Output change of water level in time."
    )
    dep: Optional[bool] = Field(True, description="Output depth.")
    tauw: Optional[bool] = Field(
        False, description="Output surface stress from the wave."
    )
    tauhf: Optional[bool] = Field(
        False, description="Output high frequency surface stress."
    )
    tautot: Optional[bool] = Field(False, description="Output total surface stress.")
    stokessurfx: Optional[bool] = Field(
        True, description="Output surface Stokes drift in X direction."
    )
    stokessurfy: Optional[bool] = Field(
        True, description="Output surface Stokes drift in Y direction."
    )
    stokesbarox: Optional[bool] = Field(
        False, description="Output barotropic Stokes drift in X direction."
    )
    stokesbaroy: Optional[bool] = Field(
        False, description="Output barotropic Stokes drift in Y direction."
    )
    rsxx: Optional[bool] = Field(False, description="Output RSXX potential of LH.")
    rsxy: Optional[bool] = Field(False, description="Output RSXY potential of LH.")
    rsyy: Optional[bool] = Field(False, description="Output RSYY potential of LH.")
    cfl1: Optional[bool] = Field(False, description="Output CFL number 1.")
    cfl2: Optional[bool] = Field(False, description="Output CFL number 2.")
    cfl3: Optional[bool] = Field(False, description="Output CFL number 3.")

    @field_validator("begtc")
    def validate_begtc(cls, v):
        import re

        if not re.match(r"^\d{8}\.\d{6}$", v):
            raise ValueError("BEGTC must be in format yyyymmdd.hhmmss")
        return v

    @field_validator("deltc")
    def validate_deltc(cls, v):
        if v <= 0:
            raise ValueError("DELTC must be a positive integer")
        return v

    @field_validator("unitc")
    def validate_unitc(cls, v):
        if v != "SEC":
            raise ValueError("UNITC must be SEC")
        return v

    @field_validator("endtc")
    def validate_endtc(cls, v):
        import re

        if not re.match(r"^\d{8}\.\d{6}$", v):
            raise ValueError("ENDTC must be in format yyyymmdd.hhmmss")
        return v

    @field_validator("definetc")
    def validate_definetc(cls, v):
        if v != -1 and v < 0:
            raise ValueError("DEFINETC must be -1 or a non-negative integer")
        return v

    @field_validator("outstyle")
    def validate_outstyle(cls, v):
        if v not in ["NO", "STE", "NC"]:
            raise ValueError("OUTSTYLE must be NO, STE, or NC")
        return v

    @field_validator("multipleout")
    def validate_multipleout(cls, v):
        if v not in [0, 1]:
            raise ValueError("MULTIPLEOUT must be 0 or 1")
        return v

    @field_validator("iouts")
    def validate_iouts(cls, v):
        if v <= 0:
            raise ValueError("IOUTS must be a positive integer")
        return v

    @field_validator("nouts")
    def validate_nouts(cls, v, info):
        if len(v) != info.data.get("iouts"):
            raise ValueError("Number of NOUTS must match IOUTS")
        return v

    @field_validator("xouts")
    def validate_xouts(cls, v, info):
        if len(v) != info.data.get("iouts"):
            raise ValueError("Number of XOUTS must match IOUTS")
        return v

    @field_validator("youts")
    def validate_youts(cls, v, info):
        if len(v) != info.data.get("iouts"):
            raise ValueError("Number of YOUTS must match IOUTS")
        return v

    @field_validator("cutoff")
    def validate_cutoff(cls, v):
        if v < 0:
            raise ValueError("CUTOFF must be non-negative")
        return v


class Hotfile(NamelistBaseModel):
    lhotf: Optional[bool] = Field(True, description="Write hotfile")
    filehot_out: Optional[str] = Field("wwm_hot_out.nc", description="name of output")
    begtc: Optional[str] = Field(
        "20200101.000000",
        description="Starting time of hotfile writing. With ihot!=0 in SCHISM,",
    )
    deltc: Optional[float] = Field(2678400.0, description="time between hotfile writes")
    unitc: Optional[str] = Field("SEC", description="unit used above")
    endtc: Optional[str] = Field(
        "20200201.000000",
        description="Ending time of hotfile writing (adjust with BEGTC)",
    )
    lcyclehot: Optional[bool] = Field(False, description="Applies only to netcdf")
    hotstyle_out: Optional[int] = Field(
        2, description="1: binary hotfile of data as output"
    )
    multipleout: Optional[int] = Field(
        0, description="0: hotfile in a single file (binary or netcdf)"
    )
    filehot_in: Optional[str] = Field(
        "wwm_hot_in.nc",
        description="(Full) hot file name for input (which can be copied from FILEHOT_OUT above)",
    )
    hotstyle_in: Optional[int] = Field(
        2, description="1: binary hotfile of data as input"
    )
    ihotpos_in: Optional[int] = Field(
        2, description="Position in hotfile (only for netcdf)"
    )
    multiplein: Optional[int] = Field(
        0, description="0: read hotfile from one single file"
    )


class Petscoptions(NamelistBaseModel):
    ksptype: Optional[str] = Field(
        "LGMRES",
        description="Controls the linear solver type used by PETSc. Options include GMRES, LGMRES (augmented GMRES), DGMRES (deflated GMRES), PGMRES (pipelined GMRES), and KSPBCGSL (variant of Enhanced BiCGStab(L)).",
    )
    rtol: Optional[float] = Field(
        1e-20,
        description="Relative convergence tolerance, representing the relative decrease in the residual norm for the iterative solver.",
    )
    abstol: Optional[float] = Field(
        1e-20,
        description="Absolute convergence tolerance, representing the absolute size of the residual norm for the iterative solver.",
    )
    dtol: Optional[float] = Field(
        10000.0, description="Divergence tolerance for the iterative solver."
    )
    maxits: Optional[int] = Field(
        1000,
        description="Maximum number of iterations allowed for the iterative solver.",
    )
    initialguessnonzero: Optional[bool] = Field(
        False,
        description="Boolean flag indicating whether the initial guess for the iterative solver is nonzero.",
    )
    gmrespreallocate: Optional[bool] = Field(
        True,
        description="Boolean flag indicating whether GMRES and FGMRES should preallocate all needed work vectors at initial setup.",
    )
    pctype: Optional[str] = Field(
        "SOR",
        description="Controls the preconditioner type used by PETSc. Options include SOR (successive over relaxation), ASM (additive Schwarz method), HYPRE (LLNL package), SPAI (Sparse Approximate Inverse), and NONE (no preconditioning).",
    )

    @field_validator("ksptype")
    @classmethod
    def validate_ksptype(cls, v):
        valid_types = ["GMRES", "LGMRES", "DGMRES", "PGMRES", "KSPBCGSL"]
        if v.upper() not in valid_types:
            raise ValueError(f"KSPTYPE must be one of {valid_types}")
        return v.upper()

    @field_validator("rtol")
    @classmethod
    def validate_rtol(cls, v):
        if v <= 0:
            raise ValueError("RTOL must be positive")
        return v

    @field_validator("abstol")
    @classmethod
    def validate_abstol(cls, v):
        if v <= 0:
            raise ValueError("ABSTOL must be positive")
        return v

    @field_validator("dtol")
    @classmethod
    def validate_dtol(cls, v):
        if v <= 0:
            raise ValueError("DTOL must be positive")
        return v

    @field_validator("maxits")
    @classmethod
    def validate_maxits(cls, v):
        if v <= 0 or not isinstance(v, int):
            raise ValueError("MAXITS must be a positive integer")
        return v

    @field_validator("initialguessnonzero")
    @classmethod
    def validate_initialguessnonzero(cls, v):
        return v

    @field_validator("gmrespreallocate")
    @classmethod
    def validate_gmrespreallocate(cls, v):
        return v

    @field_validator("pctype")
    @classmethod
    def validate_pctype(cls, v):
        valid_types = ["SOR", "ASM", "HYPRE", "SPAI", "NONE"]
        if v.upper() not in valid_types:
            raise ValueError(f"PCTYPE must be one of {valid_types}")
        return v.upper()


# class Nesting(NamelistBaseModel):
#     listbegtc: Optional[str] = Field(
#         "", description="Start date/time for the nested grid simulation"
#     )
#     listdeltc: Optional[str] = Field(
#         "ZERO", description="Time step for the nested grid simulation"
#     )
#     listunitc: Optional[str] = Field(
#         "", description="Time unit for the nested grid simulation"
#     )
#     listendtc: Optional[str] = Field(
#         "", description="End date/time for the nested grid simulation"
#     )
#     listigridtype: Optional[int] = Field(
#         0,
#         description="Grid type for the nested simulation (0 for regular lat-lon grid)",
#     )
#     listfilegrid: Optional[str] = Field(
#         "", description="File containing grid definition for the nested simulation"
#     )
#     listfilebound: Optional[str] = Field(
#         "", description="File containing boundary conditions for the nested simulation"
#     )
#     listprefix: Optional[str] = Field(
#         "", description="Prefix for output files from the nested simulation"
#     )
#
# @field_validator("listbegtc")
# @classmethod
# def validate_listbegtc(cls, v):
#     if not v:
#         raise ValueError("ListBEGTC must not be empty")
#     return v
#
# @field_validator("listdeltc")
# @classmethod
# def validate_listdeltc(cls, v):
#     if v.upper() != "ZERO":
#         raise ValueError("ListDELTC must be ZERO")
#     return v
#
# @field_validator("listunitc")
# @classmethod
# def validate_listunitc(cls, v):
#     if not v:
#         raise ValueError("ListUNITC must not be empty")
#     return v
#
# @field_validator("listendtc")
# @classmethod
# def validate_listendtc(cls, v):
#     if not v:
#         raise ValueError("ListENDTC must not be empty")
#     return v
#
# @field_validator("listigridtype")
# @classmethod
# def validate_listigridtype(cls, v):
#     if v != 0:
#         raise ValueError("ListIGRIDTYPE must be 0")
#     return v
#
# @field_validator("listfilegrid")
# @classmethod
# def validate_listfilegrid(cls, v):
#     if not v:
#         raise ValueError("ListFILEGRID must not be empty")
#     return v
#
# @field_validator("listfilebound")
# @classmethod
# def validate_listfilebound(cls, v):
#     if not v:
#         raise ValueError("ListFILEBOUND must not be empty")
#     return v
#
# @field_validator("listprefix")
# @classmethod
# def validate_listprefix(cls, v):
#     if not v:
#         raise ValueError("ListPrefix must not be empty")
#     return v


class Wind(NamelistBaseModel):
    "Empty class to fill empty wind namelist"


class Curr(NamelistBaseModel):
    "Empty class to fill empty curr namelist"


class Walv(NamelistBaseModel):
    "Empty class to fill empty wavl namelist"


class Nesting(NamelistBaseModel):
    "Empty class to fill empty nesting namelist"


class Wwminput(NamelistBaseModel):
    proc: Optional[Proc] = Field(default_factory=Proc)
    coupl: Optional[Coupl] = Field(default_factory=Coupl)
    grid: Optional[Grid] = Field(default_factory=Grid)
    init: Optional[Init] = Field(default_factory=Init)
    bouc: Optional[Bouc] = Field(default_factory=Bouc)
    wind: Optional[Wind] = Field(default_factory=Wind)
    curr: Optional[Curr] = Field(default_factory=Curr)
    walv: Optional[Walv] = Field(default_factory=Walv)
    engs: Optional[Engs] = Field(default_factory=Engs)
    nums: Optional[Nums] = Field(default_factory=Nums)
    history: Optional[History] = Field(default_factory=History)
    station: Optional[Station] = Field(default_factory=Station)
    hotfile: Optional[Hotfile] = Field(default_factory=Hotfile)
    petscoptions: Optional[Petscoptions] = Field(default_factory=Petscoptions)
    nesting: Optional[Nesting] = Field(default_factory=Nesting)

    def render(self) -> str:
        # Needs an additional backslash due to to uknown issue with schism
        # All sameple namelists have this feature
        return super().render() + "\n" + "/"
