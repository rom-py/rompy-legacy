# This file was auto generated from a schism namelist file on 2025-01-24.

from typing import Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Hotfile(NamelistBaseModel):
    lhotf: Optional[str] = Field("T", description="Write hotfile")
    begtc: Optional[str] = Field(
        "19980901.000000", description="Starting time of hotfile writing"
    )
    deltc: Optional[int] = Field(3600, description="time between hotfile writes")
    unitc: Optional[str] = Field("SEC", description="unit used above")
    endtc: Optional[str] = Field(
        "19980901.060000", description="Ending time of hotfile writing"
    )
    lcyclehot: Optional[str] = Field("T", description="Applies only to netcdf")
    hotstyle_out: Optional[int] = Field(
        2, description="1: binary hotfile of data as output"
    )
    multipleout: Optional[int] = Field(
        0, description="0: hotfile in a single file (binary or netcdf)"
    )
    filehot_out: Optional[str] = Field(
        "hotfile_out_WWM.nc", description="name of hot outputs"
    )


class Proc(NamelistBaseModel):
    procname: Optional[str] = Field("limon", description="Project Name")
    dimmode: Optional[int] = Field(
        2,
        description="Mode of run (ex: 1 = 1D, 2 = 2D) always 2D when coupled to SCHISM",
    )
    lstea: Optional[str] = Field("F", description="steady mode; under development")
    lqstea: Optional[str] = Field(
        "F",
        description="Quasi-Steady Mode; In this case WWM-II is doing subiterations defined as DELTC/NQSITER unless QSCONVI is not reached",
    )
    lsphe: Optional[str] = Field("F", description="Spherical coordinates (lon/lat)")
    lnautin: Optional[str] = Field(
        "T",
        description="Nautical convention for all inputs given in degrees (suggestion: T)",
    )
    lmono_in: Optional[str] = Field("F", description="")
    lmono_out: Optional[str] = Field("F", description="")
    lnautout: Optional[str] = Field(
        "T", description="Nautical output of all quantities in degrees"
    )
    begtc: Optional[str] = Field(
        "19980901.000000",
        description="Time for start the simulation, ex:yyyymmdd. hhmmss",
    )
    deltc: Optional[int] = Field(
        5, description="Time step (MUST match dt*nstep_wwm in SCHISM!)"
    )
    unitc: Optional[str] = Field("SEC", description="Unity of time step")
    endtc: Optional[str] = Field(
        "19980901.060000",
        description="Time for stop the simulation, ex:yyyymmdd. hhmmss",
    )
    dmin: Optional[float] = Field(
        0.01, description="Minimum water depth. This must be the same as h0 in SCHISM"
    )


class Coupl(NamelistBaseModel):
    lcpl: Optional[str] = Field(
        "T",
        description="Couple with current model ... main switch - keep it on for SCHISM-WWM",
    )
    lroms: Optional[str] = Field("F", description="ROMS (set as F)")
    ltimor: Optional[str] = Field("F", description="TIMOR (set as F)")
    lshyfem: Optional[str] = Field("F", description="SHYFEM (set as F)")
    radflag: Optional[str] = Field("LON", description="")
    letot: Optional[str] = Field(
        "F",
        description="Option to compute the wave induced radiation stress. If .T. the radiation stress is based on the integrated wave spectrum",
    )
    nlvt: Optional[int] = Field(
        10, description="Number of vertical Layers; not used with SCHISM"
    )
    dtcoup: Optional[float] = Field(
        600.0, description="Couple time step - not used when coupled to SCHISM"
    )


class Grid(NamelistBaseModel):
    lcird: Optional[str] = Field("T", description="Full circle in directional space")
    lstag: Optional[str] = Field(
        "F",
        description="Stagger directional bins with a half Dtheta; may use T only for regular grid to avoid char. line aligning with grid line",
    )
    mindir: Optional[float] = Field(
        0.0,
        description="Minimum direction for simulation (unit: degrees; nautical convention; 0: from N; 90: from E); not used if LCIRD = .T.",
    )
    maxdir: Optional[float] = Field(
        360.0,
        description="Maximum direction for simulation (unit: degrees); may be < MINDIR; not used if LCIRD = .T.",
    )
    mdc: Optional[int] = Field(36, description="Number of directional bins")
    frlow: Optional[float] = Field(
        0.04,
        description="Low frequency limit of the discrete wave period (Hz; 1/period)",
    )
    frhigh: Optional[float] = Field(
        1.0, description="High frequency limit of the discrete wave period."
    )
    msc: Optional[int] = Field(36, description="Number of frequency bins")
    igridtype: Optional[int] = Field(
        3,
        description="Gridtype used. 1 ~ XFN, 2 ~ WWM-PERIODIC, 3 ~ SCHISM, 4 ~ OLD WWM GRID",
    )
    filegrid: Optional[str] = Field(
        "hgrid_WWM.gr3",
        description="Name of the grid file. hgridi_WWM.gr3 if IGRIDTYPE = 3 (SCHISM)",
    )
    lslop: Optional[str] = Field("F", description="Bottom Slope limiter (default=F)")
    slmax: Optional[float] = Field(0.2, description="Max Slope;")
    lvar1d: Optional[str] = Field(
        "F", description="For 1d-mode if variable dx is used; not used with SCHISM"
    )


class Init(NamelistBaseModel):
    lhotr: Optional[bool] = Field(
        False,
        description="Full path and filename for the input hot file, which can be copied from FILEHOT_OUT. Used for restarting simulations.",
    )
    hotstyle_in: Optional[int] = Field(
        2,
        description="Specifies the format of the input hot file. 1 for binary format, 2 for NetCDF format (default).",
    )
    ihotpos_in: Optional[int] = Field(
        1,
        description="Position in the hot file for reading data (only applicable for NetCDF files). If LCYCLEHOT=T, this can be 1 or 2 (out of the 2 time records). '1' is the most recent time.",
    )
    multiplein: Optional[int] = Field(
        0,
        description="Specifies whether to read the hot file from a single file (0) or multiple files (1). Multiple files might require the same number of CPUs.",
    )

    @field_validator("filehot_in")
    @classmethod
    def validate_filehot_in(cls, v: str) -> str:
        if not v.endswith(".nc"):
            raise ValueError("FILEHOT_IN must be a NetCDF file with .nc extension")
        return v

    @field_validator("hotstyle_in")
    @classmethod
    def validate_hotstyle_in(cls, v: int) -> int:
        if v not in [1, 2]:
            raise ValueError("HOTSTYLE_IN must be either 1 (binary) or 2 (NetCDF)")
        return v

    @field_validator("ihotpos_in")
    @classmethod
    def validate_ihotpos_in(cls, v: int) -> int:
        if v not in [1, 2]:
            raise ValueError("IHOTPOS_IN must be either 1 or 2")
        return v

    @field_validator("multiplein")
    @classmethod
    def validate_multiplein(cls, v: int) -> int:
        if v not in [0, 1]:
            raise ValueError(
                "MULTIPLEIN must be either 0 (single file) or 1 (multiple files)"
            )
        return v


class Bouc(NamelistBaseModel):
    lbcse: Optional[str] = Field(
        "F", description="The wave boundary data is time dependent"
    )
    lbinter: Optional[str] = Field(
        "F",
        description="Do interpolation in time if LBCSE=T (not available for quasi-steady mode within the subtime steps)",
    )
    lbcwa: Optional[str] = Field("T", description="Parametric Wave Spectra")
    linhom: Optional[str] = Field("F", description="Non-uniform wave b.c. in space")
    lbcsp: Optional[str] = Field(
        "F",
        description="Specify (non-parametric) wave spectra, specified in 'FILEWAVE' below",
    )
    lindsprdeg: Optional[str] = Field(
        "F",
        description="If 1-d wave spectra are read this flag defines whether the input for the directional spreading is in degrees (true) or exponent (false)",
    )
    lparmdir: Optional[str] = Field(
        "F",
        description="If LPARMDIR is true than directional spreading is read from WBDS and must be in exponential format at this time, only valid for 1d Spectra",
    )
    filewave: Optional[str] = Field(
        "wwmbnd.gr3", description="Boundary file including discrete wave spectra"
    )
    lbsp1d: Optional[str] = Field(
        "F",
        description="1D (freq. space only) format for FILEWAVE if LBCSP=T and LINHOM=F",
    )
    lbsp2d: Optional[str] = Field(
        "F", description="not functional (freq. + directional space)"
    )
    begtc: Optional[str] = Field(
        "19980901.000000",
        description="Begin time of the wave boundary file (FILEWAVE)",
    )
    deltc: Optional[int] = Field(1, description="Time step in FILEWAVE")
    unitc: Optional[str] = Field("HR", description="Unit can be HR, MIN, SEC")
    endtc: Optional[str] = Field("19981002.000000", description="End time")
    filebound: Optional[str] = Field(
        "wwmbnd.gr3", description="Boundary file defining boundary and Neumann nodes."
    )
    iboundformat: Optional[int] = Field(
        1, description="1 ~ WWM, 3 ~ WW3 (2D spectra in netcdf format only - LBCWA=T)."
    )
    wbhs: Optional[float] = Field(
        4.0, description="Hs at the boundary for parametric spectra"
    )
    wbss: Optional[float] = Field(
        2.0,
        description="1 or -1: Pierson-Moskowitz, 2 or -2: JONSWAP, 3 or -3: all in one BIN,",
    )
    wbtp: Optional[float] = Field(
        8.0,
        description="Tp at the boundary (sec); mean or peak depending on the sign of WBSS",
    )
    wbdm: Optional[float] = Field(
        90.0, description="Avg. Wave Direction at the boundary"
    )
    wbdsms: Optional[float] = Field(
        1.0, description="Directional spreading value in degrees (1) or as exponent (2)"
    )
    wbds: Optional[float] = Field(
        20.0, description="Directional spreading at the boundary (degrees/exponent)"
    )
    wbgauss: Optional[float] = Field(
        0.1, description="factor for gaussian distribution if WBSS=1"
    )
    wbpken: Optional[float] = Field(
        3.3, description="Peak enhancement factor for Jonswap Spectra if WBSS=2"
    )
    ncdf_hs_name: Optional[str] = Field(
        "hs",
        description="NETCDF var. name for the significant wave height (normally it is just 'hs')",
    )
    ncdf_dir_name: Optional[str] = Field(
        "dir",
        description="NETCDF var. name for the mean wave direction (normally it is just 'dir')",
    )
    ncdf_spr_name: Optional[str] = Field(
        "spr",
        description="NETCDF var. name for the mean directional spreading (normally it is just 'spr')",
    )
    ncdf_fp_name: Optional[str] = Field(
        "fp",
        description="NETCDF var. name for the peak freq. (normally it is just 'fp')",
    )
    ncdf_f02_name: Optional[str] = Field(
        "t02",
        description="NETCDF var. name for the zero down crossing freq. (normally it is just 't02')",
    )


class Engs(NamelistBaseModel):
    mesnl: Optional[int] = Field(
        1,
        description="Nonlinear Interaction NL4 , 1 ~ on, 0 ~ off (Discrete Interaction approx.)",
    )
    mesin: Optional[int] = Field(
        1, description="Wind input: Ardhuin et al. (1) (use LSOURCESWAM = F);"
    )
    ifric: Optional[int] = Field(
        1,
        description="Formulation for atmospheric boundary layer, (IFRIC = 1 for MESIN = 1, IFRIC = 4 for MESIN=3);",
    )
    mesbf: Optional[int] = Field(
        1,
        description="Bottom friction: 1 - JONSWAP (Default); 2 - Madsen et al. (1989); 3 - SHOWEX",
    )
    fricc: Optional[float] = Field(
        0.067,
        description="if MESBF=1: JONSWAP bottom friction coefficient [0.038,0.067]. If MESBF=2: physical bottom roughness (ignored if given in rough.gr3). If MESBF=3: D50 (if negative read from SHOWEX_D50.gr3)",
    )
    mesbr: Optional[int] = Field(
        1, description="Shallow water wave breaking; 0: off; 1: on"
    )
    ibreak: Optional[int] = Field(
        1, description="Wave breaking formulation: 1 - Battjes and Janssen (1978)"
    )
    icrit: Optional[int] = Field(
        1,
        description="Wave breaking criterion: 1   - Constant breaker index (gamma) or gamma_TG defined with BRCR",
    )
    brcr: Optional[float] = Field(
        0.78,
        description="either gamma, default is 0.73 for IBREAK=1,5 or gamma_TG, default is 0.42 for IBREAK=2,3 or biphase_ref, default is -4pi/9 = -1.3963 for IBREAK=4",
    )
    a_brcr: Optional[float] = Field(0.76, description="cf ICRIT = 4, 5")
    b_brcr: Optional[float] = Field(0.29, description="cf ICRIT = 4, 5")
    min_brcr: Optional[float] = Field(0.25, description="cf ICRIT = 4, 5")
    max_brcr: Optional[float] = Field(0.8, description="cf ICRIT = 4, 5")
    a_biph: Optional[float] = Field(
        0.2, description="Biphase coefficient, default 0.2 (intended for IBREAK=3)"
    )
    br_coef_method: Optional[int] = Field(
        1, description="Method for the breaking coefficient: 1 - constant, 2 - adaptive"
    )
    b_alp: Optional[float] = Field(
        0.5, description="breaking coefficient. If BR_COEF_METHOD = 2, B_ALP ~ 40"
    )
    zprof_break: Optional[int] = Field(
        2,
        description="Vertical distribution function of wave breaking source term, only used in 3D run",
    )
    bc_break: Optional[int] = Field(
        1, description="Apply depth-limited breaking at the boundaries: 1 - On; 0 - Off"
    )
    iroller: Optional[int] = Field(
        0,
        description="Wave roller model (e.g., see Uchiyama et al., 2010): 1 - On; 0 - Off; not used at the moment",
    )
    alprol: Optional[float] = Field(
        0.85,
        description="Alpha coefficient for the wave roller model (between 0 and 1): 1 - full conversion; 0 - no energy transferred to the roller",
    )
    meveg: Optional[int] = Field(
        0, description="Vegetation on/off. If on, isav must = 1 in param.nml"
    )
    lmaxetot: Optional[str] = Field(
        "T",
        description="Limit shallow water wave height by wave breaking limiter (default=T)",
    )
    mesds: Optional[int] = Field(
        1,
        description="Formulation for the whitecapping source function; same value as MESIN",
    )
    mestr: Optional[int] = Field(
        1,
        description="Formulation for the triad 3 wave interactions (MESTR = 0 (off), MESTR = 1 (Lumped Triad Approx. (LTA)), MESTR = 2 (corrected version of LTA by Salmon et al. (2016)))",
    )
    trico: Optional[float] = Field(
        0.1, description="proportionality const. (\alpha_EB); default is 0.1"
    )
    trira: Optional[float] = Field(
        2.5,
        description="ratio of max. freq. considered in triads over mean freq.; 2.5 is suggested",
    )
    triurs: Optional[float] = Field(
        0.1,
        description="critical Ursell number; if Ursell # < TRIURS; triads are not computed",
    )


class Sin4(NamelistBaseModel):
    zwnd: Optional[list] = Field([10.0, ""], description="")
    alpha0: Optional[list] = Field(["9.499999694526196E-003", ""], description="")
    z0max: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    betamax: Optional[list] = Field([1.54, ""], description="")
    sinthp: Optional[list] = Field([2.0, ""], description="")
    zalp: Optional[list] = Field(["6.000000052154064E-003", ""], description="")
    tauwshelter: Optional[list] = Field([0.300000011920929, ""], description="")
    swellfpar: Optional[list] = Field([1.0, ""], description="")
    swellf: Optional[list] = Field([0.660000026226044, ""], description="")
    swellf2: Optional[list] = Field(["-1.799999922513962E-002", ""], description="")
    swellf3: Optional[list] = Field(["2.199999988079071E-002", ""], description="")
    swellf4: Optional[list] = Field([150000.0, ""], description="")
    swellf5: Optional[list] = Field([1.20000004768372, ""], description="")
    swellf6: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    swellf7: Optional[list] = Field([360000.0, ""], description="")
    z0rat: Optional[list] = Field(["3.999999910593033E-002", ""], description="")
    sinbr: Optional[float] = Field(0.0, description="")


class Sds4(NamelistBaseModel):
    sdsc1: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    fxpm3: Optional[list] = Field([4.0, ""], description="")
    fxfm3: Optional[list] = Field([2.5, ""], description="")
    fxfmage: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsc2: Optional[list] = Field(["-2.200000017182902E-005", ""], description="")
    sdscum: Optional[list] = Field([-0.403439998626709, ""], description="")
    sdsstrain: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsc4: Optional[list] = Field([1.0, ""], description="")
    sdsc5: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsc6: Optional[list] = Field([0.300000011920929, ""], description="")
    sdsbr: Optional[list] = Field(["8.999999845400453E-004", ""], description="")
    sdsbr2: Optional[list] = Field([0.800000011920929, ""], description="")
    sdsp: Optional[list] = Field([2.0, ""], description="")
    sdsiso: Optional[list] = Field([2.0, ""], description="")
    sdsbck: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsabk: Optional[list] = Field([1.5, ""], description="")
    sdspbk: Optional[list] = Field([4.0, ""], description="")
    sdsbint: Optional[list] = Field([0.300000011920929, ""], description="")
    sdshck: Optional[list] = Field([1.5, ""], description="")
    sdsdth: Optional[list] = Field([80.0, ""], description="")
    sdscos: Optional[list] = Field([2.0, ""], description="")
    sdsbrf1: Optional[list] = Field([0.5, ""], description="")
    sdsbrfdf: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsbm0: Optional[list] = Field([1.0, ""], description="")
    sdsbm1: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsbm2: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsbm3: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdsbm4: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdshfgen: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    sdslfgen: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    whitecapwidth: Optional[list] = Field([0.300000011920929, ""], description="")
    fxincut: Optional[list] = Field(["0.000000000000000E+000", ""], description="")
    fxdscut: Optional[float] = Field(0.0, description="")


class Nums(NamelistBaseModel):
    icomp: Optional[int] = Field(3, description="")
    amethod: Optional[int] = Field(7, description="")
    aspar_local_level: Optional[list] = Field([0, ""], description="")
    smethod: Optional[int] = Field(1, description="")
    dmethod: Optional[int] = Field(2, description="")
    rtheta: Optional[float] = Field(
        0.5,
        description="Weighing factor for DMETHOD = 1, not really useful since Crank Nicholson integration can only be monotone for CFL .le. 2",
    )
    litersplit: Optional[str] = Field(
        "F",
        description="T: double Strang split; F: simple split (more efficient). Default: F",
    )
    lfilterth: Optional[str] = Field("F", description="")
    maxcflth: Optional[float] = Field(
        1.0, description="Max Cfl in Theta space; used only if LFILTERTH=T"
    )
    fmethod: Optional[int] = Field(1, description="")
    lfiltersig: Optional[str] = Field(
        "F", description="Limit the advection velocity in freq. space (usually F)"
    )
    maxcflsig: Optional[float] = Field(
        1.0, description="Max Cfl in freq. space; used only if LFILTERSIG=T"
    )
    llimt: Optional[str] = Field(
        "T",
        description="Switch on/off Action limiter, Action limiter must mostly be turned on.",
    )
    lsigbound: Optional[str] = Field(
        "F", description="Theta space on wet land/island boundary"
    )
    lthbound: Optional[str] = Field(
        "F", description="Sigma space on wet land/island boundary"
    )
    lsoubound: Optional[str] = Field(
        "F", description="Source Terms on wet land/island boundary. Use T if SMETHOD=6"
    )
    melim: Optional[int] = Field(1, description="Formulation for the action limiter")
    limfak: Optional[float] = Field(
        0.1,
        description="Proportionality coefficient for the action limiter MAX_DAC_DT = Limfak * Limiter; see notes above for value",
    )
    ldifr: Optional[str] = Field(
        "F",
        description="Use phase decoupled diffraction approximation according to Holthuijsen et al. (2003) (usually T; if crash, use F)",
    )
    idiffr: Optional[int] = Field(
        1,
        description="Extended WAE accounting for higher order effects WAE becomes nonlinear; 1: Holthuijsen et al. ; 2: Liau et al. ; 3: Toledo et al. (in preparation)",
    )
    lconv: Optional[str] = Field(
        "F",
        description="Estimate convergence criterian and write disk (quasi-steady - qstea.out)",
    )
    lcfl: Optional[str] = Field(
        "F", description="Write out CFL numbers; use F to save time"
    )
    nqsiter: Optional[int] = Field(
        10,
        description="# of quasi-steady (Q-S) sub-divisions within each WWM time step (trial and errors)",
    )
    qsconv1: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill abs. wave height criteria EPSH1",
    )
    qsconv2: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill rel. wave height criteria EPSH2",
    )
    qsconv3: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill sum. rel. wave action criteria EPSH3",
    )
    qsconv4: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave period criteria EPSH4",
    )
    qsconv5: Optional[float] = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave steepness criteria EPSH5",
    )
    lexpimp: Optional[str] = Field(
        "F",
        description="Use implicit schemes for freq. lower than given below by FREQEXP; used only if ICOMP=0",
    )
    freqexp: Optional[float] = Field(
        0.1,
        description="Minimum frequency for explicit schemes; only used if LEXPIMP=T and ICOMP=0",
    )
    epsh1: Optional[float] = Field(
        0.01,
        description="Convergence criteria for rel. wave height ! EPSH1 < CONVK1 = REAL(ABS(HSOLD(IP)-HS2)/HS2)",
    )
    epsh2: Optional[float] = Field(
        0.01,
        description="Convergence criteria for abs. wave height ! EPSH2 < CONVK2 = REAL(ABS(HS2-HSOLD(IP)))",
    )
    epsh3: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. sum of wave action ! EPSH3 < CONVK3 = REAL(ABS(SUMACOLD(IP)-SUMAC)/SUMAC)",
    )
    epsh4: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. avg. wave steepness criteria ! EPSH4 < CONVK4 = REAL(ABS(KHS2-KHSOLD(IP))/KHSOLD(IP))",
    )
    epsh5: Optional[float] = Field(
        0.01,
        description="Convergence criteria for the rel. avg. waveperiod ! EPSH5 < REAL(ABS(TM02-TM02OLD(IP))/TM02OLD(IP))",
    )
    lvector: Optional[str] = Field(
        "F",
        description="Use optmized propagation routines for large high performance computers e.g. at least more than 128 CPU. Try LVECTOR=F first.",
    )
    ivector: Optional[int] = Field(
        2, description="USed if LVECTOR=T; Different flavours of communications"
    )
    ladvtest: Optional[str] = Field(
        "F",
        description="for testing the advection schemes, testcase will be added soon",
    )
    lchkconv: Optional[str] = Field(
        "T",
        description="needs to set to .true. for quasi-steady mode. in order to compute the QSCONVi criteria and check them",
    )
    nb_block: Optional[list] = Field([3, ""], description="")
    wae_solverthr: Optional[list] = Field(["1.E-6", ""], description="")
    maxiter: Optional[list] = Field([1000, ""], description="")
    lsourceswam: Optional[list] = Field(
        ["F", ""], description="Use ECMWF WAM formualtion for deep water physics."
    )
    lnaninfchk: Optional[list] = Field(["F", ""], description="")
    lzeta_setup: Optional[list] = Field(["F", ""], description="")
    zeta_meth: Optional[list] = Field([0, ""], description="")
    pmin: Optional[list] = Field([5.0, ""], description="")
    block_gauss_seidel: Optional[list] = Field(["T", ""], description="")
    lnonl: Optional[list] = Field(["F", ""], description="")
    l_solver_norm: Optional[list] = Field(["F", ""], description="")


class History(NamelistBaseModel):
    begtc: Optional[str] = Field(
        "19980901.000000", description="Start output time, yyyymmdd. hhmmss;"
    )
    deltc: Optional[int] = Field(
        1,
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )
    unitc: Optional[str] = Field("SEC", description="Unit")
    endtc: Optional[str] = Field(
        "19980910.000000", description="Stop time output, yyyymmdd. hhmmss"
    )
    definetc: Optional[int] = Field(
        86400, description="Time for definition of history files"
    )
    outstyle: Optional[str] = Field(
        "NO", description="output option - use 'NO' for no output"
    )
    multipleout: Optional[int] = Field(0, description="0: output in a single netcdf")
    use_single_out: Optional[str] = Field(
        "T",
        description="T: Use single precision in the output of model variables (default)",
    )
    paramwrite: Optional[str] = Field(
        "T", description="T/F: Write the physical parametrization and"
    )
    gridwrite: Optional[str] = Field(
        "T", description="T/F: Write the grid in the netcdf history file (default T)"
    )
    printmma: Optional[str] = Field(
        "F", description="T/F: Print minimum, maximum and average"
    )
    fileout: Optional[str] = Field("history.dat", description="")
    lwxfn: Optional[str] = Field("T", description="")
    hs: Optional[str] = Field("F", description="significant wave height")
    tm01: Optional[str] = Field("F", description="mean period")
    tm02: Optional[str] = Field("F", description="zero-crossing mean period")
    klm: Optional[str] = Field("F", description="mean wave number")
    wlm: Optional[str] = Field("F", description="mean wave length")
    etotc: Optional[str] = Field("F", description="Variable ETOTC")
    etots: Optional[str] = Field("F", description="Variable ETOTS")
    dm: Optional[str] = Field("T", description="mean wave direction")
    dspr: Optional[str] = Field("F", description="directional spreading")
    tppd: Optional[str] = Field("F", description="")
    tpp: Optional[str] = Field("F", description="")
    cpp: Optional[str] = Field("F", description="")
    wnpp: Optional[str] = Field("F", description="peak wave number")
    cgpp: Optional[str] = Field("F", description="peak group speed")
    kpp: Optional[str] = Field("F", description="peak wave number")
    lpp: Optional[str] = Field("F", description="peak")
    peakd: Optional[str] = Field("F", description="peak direction")
    peakdspr: Optional[str] = Field("F", description="peak directional spreading")
    dpeak: Optional[str] = Field("F", description="")
    ubot: Optional[str] = Field("F", description="")
    orbital: Optional[str] = Field("F", description="")
    botexper: Optional[str] = Field("F", description="")
    tmbot: Optional[str] = Field("F", description="")
    ursell: Optional[str] = Field("F", description="Ursell number")
    ufric: Optional[str] = Field("F", description="air friction velocity")
    z0: Optional[str] = Field("F", description="air roughness length")
    alpha_ch: Optional[str] = Field("F", description="Charnoch coefficient for air")
    windx: Optional[str] = Field("F", description="Wind in X direction")
    windy: Optional[str] = Field("F", description="Wind in Y direction")
    cd: Optional[str] = Field("F", description="Drag coefficient")
    currtx: Optional[str] = Field("F", description="current in X direction")
    currty: Optional[str] = Field("F", description="current in Y direction")
    watlev: Optional[str] = Field("F", description="water level")
    watlevold: Optional[str] = Field(
        "F", description="water level at previous time step"
    )
    dep: Optional[str] = Field("F", description="depth")
    tauw: Optional[str] = Field("F", description="surface stress from the wave")
    tauhf: Optional[str] = Field("F", description="high frequency surface stress")
    tautot: Optional[str] = Field("F", description="total surface stress")
    stokessurfx: Optional[str] = Field(
        "F", description="Surface Stokes drift in X direction"
    )
    stokessurfy: Optional[str] = Field(
        "F", description="Surface Stokes drift in X direction"
    )
    stokesbarox: Optional[str] = Field(
        "F", description="Barotropic Stokes drift in X direction"
    )
    stokesbaroy: Optional[str] = Field(
        "F", description="Barotropic Stokes drift in Y direction"
    )


class Station(NamelistBaseModel):
    begtc: Optional[str] = Field(
        "19980901.000000",
        description="Start simulation time, yyyymmdd. hhmmss; must fit the simulation time otherwise no output",
    )
    deltc: Optional[list] = Field(
        [1, ""],
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )
    unitc: Optional[str] = Field("SEC", description="Unit")
    endtc: Optional[str] = Field(
        "20081101.000000", description="Stop time simulation, yyyymmdd. hhmmss"
    )
    outstyle: Optional[str] = Field(
        "STE",
        description="output option - use 'NO' to maximize efficiency during parallel run when using MPI",
    )
    fileout: Optional[str] = Field("station.dat", description="")
    loutiter: Optional[str] = Field("F", description="")
    llouts: Optional[str] = Field("F", description="station output flag")
    ilouts: Optional[int] = Field(1, description="Number of output stations")
    nlouts: Optional[list] = Field(["P-1", ""], description="Name of output locations")
    iouts: Optional[int] = Field(1, description="")
    nouts: Optional[list] = Field(["P-1", ""], description="")
    xouts: Optional[list] = Field(
        [1950.0, ""], description="X-Coordinate of output locations"
    )
    youts: Optional[list] = Field(
        [304.0, ""], description="Y-Coordinate of output locations"
    )
    cutoff: Optional[str] = Field(
        "8*0.44",
        description="cutoff freq (Hz) for each station - consistent with buoys",
    )
    lsp1d: Optional[str] = Field("F", description="1D spectral station output")
    lsp2d: Optional[str] = Field("F", description="2D spectral station output")
    lsigmax: Optional[str] = Field(
        "T",
        description="Adjust the cut-freq. for the output (e.g. consistent with buoy cut-off freq.)",
    )


class Petscoptions(NamelistBaseModel):
    ksptype: Optional[str] = Field("bcgs", description="")
    rtol: Optional[float] = Field(
        1e-20,
        description="the relative convergence tolerance (relative decrease in the residual norm)",
    )
    abstol: Optional[float] = Field(
        1e-20,
        description="the absolute convergence tolerance (absolute size of the residual norm)",
    )
    dtol: Optional[float] = Field(10000.0, description="the divergence tolerance")
    maxits: Optional[int] = Field(
        1000, description="maximum number of iterations to use"
    )
    initialguessnonzero: Optional[str] = Field(
        "T",
        description="Tells the iterative solver that the initial guess is nonzero; otherwise KSP assumes the initial guess is to be zero",
    )
    gmrespreallocate: Optional[str] = Field(
        "T",
        description="Causes GMRES and FGMRES to preallocate all its needed work vectors at initial setup rather than the default, which is to allocate them in chunks when needed.",
    )
    pctype: Optional[str] = Field("sor", description="")


class Wwminput(NamelistBaseModel):
    hotfile: Optional[Hotfile] = Field(default_factory=Hotfile)
    proc: Optional[Proc] = Field(default_factory=Proc)
    coupl: Optional[Coupl] = Field(default_factory=Coupl)
    grid: Optional[Grid] = Field(default_factory=Grid)
    init: Optional[Init] = Field(default_factory=Init)
    bouc: Optional[Bouc] = Field(default_factory=Bouc)
    engs: Optional[Engs] = Field(default_factory=Engs)
    sin4: Optional[Sin4] = Field(default_factory=Sin4)
    sds4: Optional[Sds4] = Field(default_factory=Sds4)
    nums: Optional[Nums] = Field(default_factory=Nums)
    history: Optional[History] = Field(default_factory=History)
    station: Optional[Station] = Field(default_factory=Station)
    petscoptions: Optional[Petscoptions] = Field(default_factory=Petscoptions)
