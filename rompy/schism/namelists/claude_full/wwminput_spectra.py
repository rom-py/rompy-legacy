# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import Literal, Optional

from pydantic import Field

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Proc(NamelistBaseModel):
    """Project and time settings for the simulation."""

    procname: str = Field("limon", description="Project Name")
    dimmode: int = Field(
        2,
        description="Mode of run (ex: 1 = 1D, 2 = 2D) always 2D when coupled to SCHISM",
    )
    lstea: bool = Field(False, description="steady mode; under development")
    lqstea: bool = Field(
        False,
        description="Quasi-Steady Mode; In this case WWM-II is doing subiterations defined as DELTC/NQSITER unless QSCONVI is not reached",
    )
    lsphe: bool = Field(False, description="Spherical coordinates (lon/lat)")
    lnautin: bool = Field(
        True, description="Nautical convention for all inputs given in degrees"
    )
    lmono_in: bool = Field(False)
    lmono_out: bool = Field(False)
    lnautout: bool = Field(
        True, description="Nautical output of all quantities in degrees"
    )
    begtc: str = Field(
        "19980901.000000",
        description="Time for start the simulation, ex:yyyymmdd. hhmmss",
    )
    deltc: int = Field(5, description="Time step (MUST match dt*nstep_wwm in SCHISM!)")
    unitc: Literal["SEC"] = Field("SEC", description="Unity of time step")
    endtc: str = Field(
        "19980901.060000",
        description="Time for stop the simulation, ex:yyyymmdd. hhmmss",
    )
    dmin: float = Field(
        0.01, description="Minimum water depth. This must be the same as h0 in SCHISM"
    )


class Coupl(NamelistBaseModel):
    """Coupling settings with other models."""

    lcpl: bool = Field(
        True,
        description="Couple with current model ... main switch - keep it on for SCHISM-WWM",
    )
    lroms: bool = Field(False, description="ROMS (set as F)")
    ltimor: bool = Field(False, description="TIMOR (set as F)")
    lshyfem: bool = Field(False, description="SHYFEM (set as F)")
    radflag: Literal["LON"] = Field("LON")
    letot: bool = Field(
        False,
        description="Option to compute the wave induced radiation stress. If .T. the radiation stress is based on the integrated wave spectrum",
    )
    nlvt: int = Field(10, description="Number of vertical Layers; not used with SCHISM")
    dtcoup: float = Field(
        600.0, description="Couple time step - not used when coupled to SCHISM"
    )


class Grid(NamelistBaseModel):
    """Grid and discretization settings."""

    lcird: bool = Field(True, description="Full circle in directional space")
    lstag: bool = Field(
        False,
        description="Stagger directional bins with a half Dtheta; may use T only for regular grid to avoid char. line aligning with grid line",
    )
    mindir: float = Field(
        0.0,
        description="Minimum direction for simulation (unit: degrees; nautical convention; 0: from N; 90: from E); not used if LCIRD = .T.",
    )
    maxdir: float = Field(
        360.0,
        description="Maximum direction for simulation (unit: degrees); may be < MINDIR; not used if LCIRD = .T.",
    )
    mdc: int = Field(36, description="Number of directional bins")
    frlow: float = Field(
        0.04,
        description="Low frequency limit of the discrete wave period (Hz; 1/period)",
    )
    frhigh: float = Field(
        1.0, description="High frequency limit of the discrete wave period."
    )
    msc: int = Field(36, description="Number of frequency bins")
    igridtype: int = Field(
        3,
        description="Gridtype used. 1 ~ XFN, 2 ~ WWM-PERIODIC, 3 ~ SCHISM, 4 ~ OLD WWM GRID",
    )
    filegrid: str = Field(
        "hgrid_WWM.gr3",
        description="Name of the grid file. hgridi_WWM.gr3 if IGRIDTYPE = 3 (SCHISM)",
    )
    lslop: bool = Field(False, description="Bottom Slope limiter (default=F)")
    slmax: float = Field(0.2, description="Max Slope;")
    lvar1d: bool = Field(
        False, description="For 1d-mode if variable dx is used; not used with SCHISM"
    )


class Init(NamelistBaseModel):
    """Initialization settings."""

    lhotr: bool = Field(
        False, description="Use hotstart file. (see &HOTFILE section for input file)"
    )
    linid: bool = Field(
        False,
        description="Initial condition; F for default; use T if using WW3 as i.c. etc",
    )
    initstyle: int = Field(
        1,
        description="1 - Parametric Jonswap, 2 - Read from Global NETCDF files, work only if IBOUNDFORMAT=3",
    )


class Bouc(NamelistBaseModel):
    """Boundary condition settings."""

    lbcse: bool = Field(False, description="The wave boundary data is time dependent")
    lbinter: bool = Field(
        False,
        description="Do interpolation in time if LBCSE=T (not available for quasi-steady mode within the subtime steps)",
    )
    lbcwa: bool = Field(True, description="Parametric Wave Spectra")
    linhom: bool = Field(False, description="Non-uniform wave b.c. in space")
    lbcsp: bool = Field(
        False,
        description="Specify (non-parametric) wave spectra, specified in 'FILEWAVE' below",
    )
    lindsprdeg: bool = Field(
        False,
        description="If 1-d wave spectra are read this flag defines whether the input for the directional spreading is in degrees (true) or exponent (false)",
    )
    lparmdir: bool = Field(
        False,
        description="If LPARMDIR is true than directional spreading is read from WBDS and must be in exponential format at this time, only valid for 1d Spectra",
    )
    filewave: str = Field(
        "wwmbnd.gr3", description="Boundary file including discrete wave spectra"
    )
    lbsp1d: bool = Field(
        False,
        description="1D (freq. space only) format for FILEWAVE if LBCSP=T and LINHOM=F",
    )
    lbsp2d: bool = Field(
        False, description="not functional (freq. + directional space)"
    )
    begtc: str = Field(
        "19980901.000000", description="Begin time of the wave boundary file (FILEWAVE)"
    )
    deltc: int = Field(1, description="Time step in FILEWAVE")
    unitc: Literal["HR", "MIN", "SEC"] = Field(
        "HR", description="Unit can be HR, MIN, SEC"
    )
    endtc: str = Field("19981002.000000", description="End time")
    filebound: str = Field(
        "wwmbnd.gr3", description="Boundary file defining boundary and Neumann nodes."
    )
    iboundformat: int = Field(
        1, description="1 ~ WWM, 3 ~ WW3 (2D spectra in netcdf format only - LBCWA=T)"
    )
    wbhs: float = Field(4.0, description="Hs at the boundary for parametric spectra")
    wbss: float = Field(
        2.0,
        description="1 or -1: Pierson-Moskowitz, 2 or -2: JONSWAP, 3 or -3: all in one BIN, 4: Gauss. The sign decides whether WBTP below is peak (+) or mean period (-)",
    )
    wbtp: float = Field(
        8.0,
        description="Tp at the boundary (sec); mean or peak depending on the sign of WBSS",
    )
    wbdm: float = Field(90.0, description="Avg. Wave Direction at the boundary")
    wbdsms: float = Field(
        1.0, description="Directional spreading value in degrees (1) or as exponent (2)"
    )
    wbds: float = Field(
        20.0, description="Directional spreading at the boundary (degrees/exponent)"
    )
    wbgauss: float = Field(
        0.1, description="factor for gaussian distribution if WBSS=1"
    )
    wbpken: float = Field(
        3.3, description="Peak enhancement factor for Jonswap Spectra if WBSS=2"
    )
    ncdf_hs_name: str = Field(
        "hs",
        description="NETCDF var. name for the significant wave height (normally it is just 'hs')",
    )
    ncdf_dir_name: str = Field(
        "dir",
        description="NETCDF var. name for the mean wave direction (normally it is just 'dir')",
    )
    ncdf_spr_name: str = Field(
        "spr",
        description="NETCDF var. name for the mean directional spreading (normally it is just 'spr')",
    )
    ncdf_fp_name: str = Field(
        "fp",
        description="NETCDF var. name for the peak freq. (normally it is just 'fp')",
    )
    ncdf_f02_name: str = Field(
        "t02",
        description="NETCDF var. name for the zero down crossing freq. (normally it is just 't02')",
    )


class Wind(NamelistBaseModel):
    """Wind settings."""

    pass


class Curr(NamelistBaseModel):
    """Current settings."""

    pass


class Walv(NamelistBaseModel):
    """Water level settings."""

    pass


class Engs(NamelistBaseModel):
    """Source term settings."""

    mesnl: int = Field(
        1,
        description="Nonlinear Interaction NL4 , 1 ~ on, 0 ~ off (Discrete Interaction approx.)",
    )
    mesin: int = Field(
        1,
        description="Wind input: Ardhuin et al. (1) (use LSOURCESWAM = F); for ECMWF physics (2); Makin & Stam (3); Babanin et al. (4); Cycle 3 (5); no wind (0). Try MESIN=1, LSOURCESWAM=F or MESIN=2 and LSOURCESWAM=T",
    )
    ifric: int = Field(
        1,
        description="Formulation for atmospheric boundary layer, (IFRIC = 1 for MESIN = 1, IFRIC = 4 for MESIN=3);",
    )
    mesbf: int = Field(
        1,
        description="Bottom friction: 1 - JONSWAP (Default); 2 - Madsen et al. (1989); 3 - SHOWEX",
    )
    fricc: float = Field(
        0.067,
        description="if MESBF=1: JONSWAP bottom friction coefficient [0.038,0.067]. If MESBF=2: physical bottom roughness (ignored if given in rough.gr3). If MESBF=3: D50 (if negative read from SHOWEX_D50.gr3)",
    )
    mesbr: int = Field(1, description="Shallow water wave breaking; 0: off; 1: on")
    ibreak: int = Field(
        1, description="Wave breaking formulation: 1 - Battjes and Janssen (1978)"
    )
    icrit: int = Field(
        1,
        description="Wave breaking criterion: 1   - Constant breaker index (gamma) or gamma_TG defined with BRCR",
    )
    brcr: float = Field(
        0.78,
        description="either gamma, default is 0.73 for IBREAK=1,5 or gamma_TG, default is 0.42 for IBREAK=2,3 or biphase_ref, default is -4pi/9 = -1.3963 for IBREAK=4",
    )
    a_brcr: float = Field(0.76, description="cf ICRIT = 4, 5")
    b_brcr: float = Field(0.29, description="cf ICRIT = 4, 5")
    min_brcr: float = Field(0.25, description="cf ICRIT = 4, 5")
    max_brcr: float = Field(0.8, description="cf ICRIT = 4, 5")
    a_biph: float = Field(
        0.2, description="Biphase coefficient, default 0.2 (intended for IBREAK=3)"
    )
    br_coef_method: int = Field(
        1, description="Method for the breaking coefficient: 1 - constant, 2 - adaptive"
    )
    b_alp: float = Field(
        0.5, description="breaking coefficient. If BR_COEF_METHOD = 2, B_ALP ~ 40"
    )
    zprof_break: int = Field(
        2,
        description="Vertical distribution function of wave breaking source term, only used in 3D run",
    )
    bc_break: int = Field(
        1, description="Apply depth-limited breaking at the boundaries: 1 - On; 0 - Off"
    )
    iroller: int = Field(
        0,
        description="Wave roller model (e.g., see Uchiyama et al., 2010): 1 - On; 0 - Off; not used at the moment",
    )
    alprol: float = Field(
        0.85,
        description="Alpha coefficient for the wave roller model (between 0 and 1): 1 - full conversion; 0 - no energy transferred to the roller",
    )
    meveg: int = Field(
        0, description="Vegetation on/off. If on, isav must = 1 in param.nml"
    )
    lmaxetot: bool = Field(
        True,
        description="Limit shallow water wave height by wave breaking limiter (default=T)",
    )
    mesds: int = Field(
        1,
        description="Formulation for the whitecapping source function; same value as MESIN",
    )
    mestr: int = Field(
        1,
        description="Formulation for the triad 3 wave interactions (MESTR = 0 (off), MESTR = 1 (Lumped Triad Approx. (LTA)), MESTR = 2 (corrected version of LTA by Salmon et al. (2016)))",
    )
    trico: float = Field(
        0.1, description="proportionality const. (\\alpha_EB); default is 0.1"
    )
    trira: float = Field(
        2.5,
        description="ratio of max. freq. considered in triads over mean freq.; 2.5 is suggested",
    )
    triurs: float = Field(
        0.1,
        description="critical Ursell number; if Ursell # < TRIURS; triads are not computed",
    )


class Sin4(NamelistBaseModel):
    """Ardhuin et al. (2009, 2010, 2011) coefficients."""

    zwnd: float = Field(10.0)
    alpha0: float = Field(9.499999694526196e-3)
    z0max: float = Field(0.0)
    betamax: float = Field(1.54)
    sinthp: float = Field(2.0)
    zalp: float = Field(6.000000052154064e-3)
    tauwshelter: float = Field(0.300000011920929)
    swellfpar: float = Field(1.0)
    swellf: float = Field(0.660000026226044)
    swellf2: float = Field(-1.799999922513962e-2)
    swellf3: float = Field(2.199999988079071e-2)
    swellf4: float = Field(150000.0)
    swellf5: float = Field(1.20000004768372)
    swellf6: float = Field(0.0)
    swellf7: float = Field(360000.0)
    z0rat: float = Field(3.999999910593033e-2)
    sinbr: float = Field(0.0)


class Sds4(NamelistBaseModel):
    """Additional coefficients."""

    sdsc1: float = Field(0.0)
    fxpm3: float = Field(4.0)
    fxfm3: float = Field(2.5)
    fxfmage: float = Field(0.0)
    sdsc2: float = Field(-2.200000017182902e-5)
    sdscum: float = Field(-0.403439998626709)
    sdsstrain: float = Field(0.0)
    sdsc4: float = Field(1.0)
    sdsc5: float = Field(0.0)
    sdsc6: float = Field(0.300000011920929)
    sdsbr: float = Field(8.999999845400453e-4)
    sdsbr2: float = Field(0.800000011920929)
    sdsp: float = Field(2.0)
    sdsiso: float = Field(2.0)
    sdsbck: float = Field(0.0)
    sdsabk: float = Field(1.5)
    sdspbk: float = Field(4.0)
    sdsbint: float = Field(0.300000011920929)
    sdshck: float = Field(1.5)
    sdsdth: float = Field(80.0)
    sdscos: float = Field(2.0)
    sdsbrf1: float = Field(0.5)
    sdsbrfdf: float = Field(0.0)
    sdsbm0: float = Field(1.0)
    sdsbm1: float = Field(0.0)
    sdsbm2: float = Field(0.0)
    sdsbm3: float = Field(0.0)
    sdsbm4: float = Field(0.0)
    sdshfgen: float = Field(0.0)
    sdslfgen: float = Field(0.0)
    whitecapwidth: float = Field(0.300000011920929)
    fxincut: float = Field(0.0)
    fxdscut: float = Field(0.0)


class Nums(NamelistBaseModel):
    """Numerical scheme settings."""

    icomp: int = Field(
        3,
        description="Controls splitting and implicit/explicit schemes for spectral advection",
    )
    amethod: int = Field(
        7, description="Controls different Methods in geographical space"
    )
    aspar_local_level: int = Field(
        0,
        description="locality level 0 - a lot of memory 10 - no memory the rest r hybrid levels",
    )
    smethod: int = Field(1, description="Controls how source terms are integrated")
    dmethod: int = Field(
        2, description="Controls numerical method in directional space"
    )
    rtheta: float = Field(0.5, description="Weighing factor for DMETHOD = 1")
    litersplit: bool = Field(
        False, description="T: double Strang split; F: simple split (more efficient)"
    )
    lfilterth: bool = Field(
        False,
        description="Use a CFL filter to limit the advection vel. In directional space",
    )
    maxcflth: float = Field(
        1.0, description="Max Cfl in Theta space; used only if LFILTERTH=T"
    )
    fmethod: int = Field(1, description="Controls numerical method used in freq. space")
    lfiltersig: bool = Field(
        False, description="Limit the advection velocity in freq. space"
    )
    maxcflsig: float = Field(
        1.0, description="Max Cfl in freq. space; used only if LFILTERSIG=T"
    )
    llimt: bool = Field(True, description="Switch on/off Action limiter")
    lsigbound: bool = Field(
        False, description="Theta space on wet land/island boundary"
    )
    lthbound: bool = Field(False, description="Sigma space on wet land/island boundary")
    lsoubound: bool = Field(
        False, description="Source Terms on wet land/island boundary"
    )
    melim: int = Field(1, description="Formulation for the action limiter")
    limfak: float = Field(
        0.1, description="Proportionality coefficient for the action limiter"
    )
    ldifr: bool = Field(
        False, description="Use phase decoupled diffraction approximation"
    )
    idiffr: int = Field(
        1, description="Extended WAE accounting for higher order effects"
    )
    lconv: bool = Field(
        False, description="Estimate convergence criterian and write disk"
    )
    lcfl: bool = Field(False, description="Write out CFL numbers")
    nqsiter: int = Field(
        10,
        description="# of quasi-steady (Q-S) sub-divisions within each WWM time step",
    )
    qsconv1: float = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill abs. wave height criteria EPSH1",
    )
    qsconv2: float = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill rel. wave height criteria EPSH2",
    )
    qsconv3: float = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill sum. rel. wave action criteria EPSH3",
    )
    qsconv4: float = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave period criteria EPSH4",
    )
    qsconv5: float = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave steepness criteria EPSH5",
    )
    lexpimp: bool = Field(
        False,
        description="Use implicit schemes for freq. lower than given below by FREQEXP",
    )
    freqexp: float = Field(0.1, description="Minimum frequency for explicit schemes")
    epsh1: float = Field(0.01, description="Convergence criteria for rel. wave height")
    epsh2: float = Field(0.01, description="Convergence criteria for abs. wave height")
    epsh3: float = Field(
        0.01, description="Convergence criteria for the rel. sum of wave action"
    )
    epsh4: float = Field(
        0.01,
        description="Convergence criteria for the rel. avg. wave steepness criteria",
    )
    epsh5: float = Field(
        0.01, description="Convergence criteria for the rel. avg. waveperiod"
    )
    lvector: bool = Field(
        False,
        description="Use optmized propagation routines for large high performance computers",
    )
    ivector: int = Field(2, description="Different flavours of communications")
    ladvtest: bool = Field(False, description="for testing the advection schemes")
    lchkconv: bool = Field(
        True,
        description="needs to set to .true. for quasi-steady mode. in order to compute the QSCONVi criteria and check them",
    )
    nb_block: int = Field(3)
    wae_solverthr: float = Field(1e-6)
    maxiter: int = Field(1000)
    lsourceswam: bool = Field(
        False, description="Use ECMWF WAM formualtion for deep water physics"
    )
    lnaninfchk: bool = Field(False)
    lzeta_setup: bool = Field(False)
    zeta_meth: int = Field(0)
    pmin: float = Field(5.0)
    block_gauss_seidel: bool = Field(True)
    lnonl: bool = Field(False)
    l_solver_norm: bool = Field(False)


class History(NamelistBaseModel):
    """Output settings for statistical variables over the whole domain."""

    begtc: str = Field(
        "19980901.000000", description="Start output time, yyyymmdd. hhmmss"
    )
    deltc: int = Field(1, description="Time step for output")
    unitc: Literal["SEC"] = Field("SEC", description="Unit")
    endtc: str = Field(
        "19980910.000000", description="Stop time output, yyyymmdd. hhmmss"
    )
    definetc: int = Field(86400, description="Time for definition of history files")
    outstyle: Literal["NO", "NC", "XFN", "SHP"] = Field(
        "NO", description="output option"
    )
    multipleout: int = Field(
        0,
        description="0: output in a single netcdf file, 1: output in separate netcdf files, 2: output using the parallel netcdf library",
    )
    use_single_out: bool = Field(
        True, description="T: Use single precision in the output of model variables"
    )
    paramwrite: bool = Field(
        True,
        description="T/F: Write the physical parametrization and chosen numerical discretization in the netcdf history file",
    )
    gridwrite: bool = Field(
        True, description="T/F: Write the grid in the netcdf history file"
    )
    printmma: bool = Field(
        False,
        description="T/F: Print minimum, maximum and average value of statistics during runtime",
    )
    fileout: str = Field("history.dat")
    lwxfn: bool = Field(True)
    hs: bool = Field(False, description="significant wave height")
    tm01: bool = Field(False, description="mean period")
    tm02: bool = Field(False, description="zero-crossing mean period")
    klm: bool = Field(False, description="mean wave number")
    wlm: bool = Field(False, description="mean wave length")
    etotc: bool = Field(False, description="Variable ETOTC")
    etots: bool = Field(False, description="Variable ETOTS")
    dm: bool = Field(True, description="mean wave direction")
    dspr: bool = Field(False, description="directional spreading")
    tppd: bool = Field(False)
    tpp: bool = Field(False)
    cpp: bool = Field(False)
    wnpp: bool = Field(False, description="peak wave number")
    cgpp: bool = Field(False, description="peak group speed")
    kpp: bool = Field(False, description="peak wave number")
    lpp: bool = Field(False, description="peak")
    peakd: bool = Field(False, description="peak direction")
    peakdspr: bool = Field(False, description="peak directional spreading")
    dpeak: bool = Field(False)
    ubot: bool = Field(False)
    orbital: bool = Field(False)
    botexper: bool = Field(False)
    tmbot: bool = Field(False)
    ursell: bool = Field(False, description="Ursell number")
    ufric: bool = Field(False, description="air friction velocity")
    z0: bool = Field(False, description="air roughness length")
    alpha_ch: bool = Field(False, description="Charnoch coefficient for air")
    windx: bool = Field(False, description="Wind in X direction")
    windy: bool = Field(False, description="Wind in Y direction")
    cd: bool = Field(False, description="Drag coefficient")
    currtx: bool = Field(False, description="current in X direction")
    currty: bool = Field(False, description="current in Y direction")
    watlev: bool = Field(False, description="water level")
    watlevold: bool = Field(False, description="water level at previous time step")
    dep: bool = Field(False, description="depth")
    tauw: bool = Field(False, description="surface stress from the wave")
    tauhf: bool = Field(False, description="high frequency surface stress")
    tautot: bool = Field(False, description="total surface stress")
    stokessurfx: bool = Field(False, description="Surface Stokes drift in X direction")
    stokessurfy: bool = Field(False, description="Surface Stokes drift in X direction")
    stokesbarox: bool = Field(
        False, description="Barotropic Stokes drift in X direction"
    )
    stokesbaroy: bool = Field(
        False, description="Barotropic Stokes drift in Y direction"
    )


class Station(NamelistBaseModel):
    """Station output settings."""

    begtc: str = Field(
        "19980901.000000", description="Start simulation time, yyyymmdd. hhmmss"
    )
    deltc: int = Field(1, description="Time step for output")
    unitc: Literal["SEC"] = Field("SEC", description="Unit")
    endtc: str = Field(
        "20081101.000000", description="Stop time simulation, yyyymmdd. hhmmss"
    )
    outstyle: Literal["STE", "NO"] = Field("STE", description="output option")
    fileout: str = Field("station.dat")
    loutiter: bool = Field(False)
    llouts: bool = Field(False, description="station output flag")
    ilouts: int = Field(1, description="Number of output stations")
    nlouts: str = Field("P-1", description="Name of output locations")
    iouts: int = Field(1)
    nouts: str = Field("P-1")
    xouts: float = Field(1950.0, description="X-Coordinate of output locations")
    youts: float = Field(304.0, description="Y-Coordinate of output locations")
    cutoff: float = Field(
        8 * 0.44,
        description="cutoff freq (Hz) for each station - consistent with buoys",
    )
    lsp1d: bool = Field(False, description="1D spectral station output")
    lsp2d: bool = Field(False, description="2D spectral station output")
    lsigmax: bool = Field(True, description="Adjust the cut-freq. for the output")


class Hotfile(NamelistBaseModel):
    """Hotfile settings for writing and reading."""

    lhotf: bool = Field(True, description="Write hotfile")
    begtc: str = Field(
        "19980901.000000", description="Starting time of hotfile writing"
    )
    deltc: int = Field(3600, description="time between hotfile writes")
    unitc: Literal["SEC"] = Field("SEC", description="unit used above")
    endtc: str = Field("19980901.060000", description="Ending time of hotfile writing")
    lcyclehot: bool = Field(
        True,
        description="If T then hotfile contains 2 last records (1st record is most recent)",
    )
    hotstyle_out: int = Field(
        2,
        description="1: binary hotfile of data as output, 2: netcdf hotfile of data as output",
    )
    multipleout: int = Field(
        0, description="0: hotfile in a single file, 1: hotfiles in separate files"
    )
    filehot_out: str = Field("hotfile_out_WWM.nc", description="name of hot outputs")
    hotstyle_in: int = Field(
        2,
        description="1: binary hotfile of data as input, 2: netcdf hotfile of data as input",
    )
    ihotpos_in: int = Field(
        1, description="Position in hotfile (only for netcdf) for reading"
    )
    multiplein: int = Field(
        0,
        description="0: read hotfile from one single file, 1: read hotfile from multiple files",
    )
    filehot_in: str = Field(
        "hotfile_in_WWM.nc", description="Hot file name for input if LHOTR=T in &INIT"
    )


class Nesting(NamelistBaseModel):
    """Nesting settings."""

    pass


class PetscOptions(NamelistBaseModel):
    """PETSc solver options."""

    ksptype: Literal["bcgs", "GMRES", "LGMRES", "DGMRES", "PGMRES", "KSPBCGSL"] = Field(
        "bcgs", description="Controls which solver is used"
    )
    rtol: float = Field(1e-20, description="the relative convergence tolerance")
    abstol: float = Field(1e-20, description="the absolute convergence tolerance")
    dtol: float = Field(10000.0, description="the divergence tolerance")
    maxits: int = Field(1000, description="maximum number of iterations to use")
    initialguessnonzero: bool = Field(
        True, description="Tells the iterative solver that the initial guess is nonzero"
    )
    gmrespreallocate: bool = Field(
        True,
        description="Causes GMRES and FGMRES to preallocate all its needed work vectors at initial setup",
    )
    pctype: Literal["sor", "ASM", "HYPRE", "SPAI", "NONE"] = Field(
        "sor", description="Controls which preconditioner is used"
    )


class Wwminput_Spectra(NamelistBaseModel):
    """Main input for WWM (based on limon)."""

    proc: Optional[Proc] = Field(default=None)
    coupl: Optional[Coupl] = Field(default=None)
    grid: Optional[Grid] = Field(default=None)
    init: Optional[Init] = Field(default=None)
    bouc: Optional[Bouc] = Field(default=None)
    wind: Optional[Wind] = Field(default=None)
    curr: Optional[Curr] = Field(default=None)
    walv: Optional[Walv] = Field(default=None)
