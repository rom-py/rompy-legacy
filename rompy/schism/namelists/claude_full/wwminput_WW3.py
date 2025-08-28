# This file was auto generated from a schism namelist file on 2024-09-06.

from typing import Literal, Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Proc(NamelistBaseModel):
    """Main processing parameters"""

    procname: str = Field("Isabel", description="Project Name")
    dimmode: Literal[2] = Field(
        2,
        description="Mode of run (ex: 1 = 1D, 2 = 2D) always 2D when coupled to SCHISM",
    )
    lstea: bool = Field(False, description="steady mode; under development")
    lqstea: bool = Field(
        False,
        description="Quasi-Steady Mode; In this case WWM-II is doing subiterations defined as DELTC/NQSITER unless QSCONVI is not reached",
    )
    lsphe: bool = Field(True, description="Spherical coordinates (lon/lat)")
    lnautin: bool = Field(
        True,
        description="Nautical convention for all inputs given in degrees (suggestion: T)",
    )
    lnautout: bool = Field(True, description="Output in Nautical convention")
    lmono_in: bool = Field(
        False,
        description="For prescribing monochromatic wave height Hmono as a boundary conditions; incident wave is defined as monochromatic wave height, which is Hmono = sqrt(2) * Hs",
    )
    lmono_out: bool = Field(False, description="Output wave heights in terms of Lmono")
    begtc: str = Field(
        "20030908.000000",
        description="Time for start the simulation, ex:yyyymmdd. hhmmss",
    )
    deltc: int = Field(
        600, description="Time step (MUST match dt*nstep_wwm in SCHISM!)"
    )
    unitc: Literal["SEC"] = Field("SEC", description="Unity of time step")
    endtc: str = Field(
        "20031008.000000",
        description="Time for stop the simulation, ex:yyyymmdd. hhmmss",
    )
    dmin: float = Field(
        0.01, description="Minimum water depth. This must be the same as h0 in SCHISM"
    )


class Coupl(NamelistBaseModel):
    """Coupling parameters"""

    lcpl: bool = Field(
        True,
        description="Couple with current model ... main switch - keep it on for SCHISM-WWM",
    )
    radflag: Literal["LON", "VOR"] = Field(
        "LON", description="LON: Longuet-Higgin; VOR: vortex formulation"
    )
    letot: bool = Field(
        False, description="Option to compute the wave induced radiation stress"
    )
    nlvt: int = Field(10, description="Number of vertical Layers; not used with SCHISM")
    dtcoup: float = Field(
        600.0, description="Couple time step - not used when coupled to SCHISM"
    )


class Grid(NamelistBaseModel):
    """Grid parameters"""

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
    filegrid: str = Field(
        "hgrid_WWM.gr3",
        description="Name of the grid file. hgridi_WWM.gr3 if IGRIDTYPE = 3 (SCHISM)",
    )
    igridtype: Literal[3] = Field(3, description="Gridtype used")
    lslop: bool = Field(False, description="Bottom Slope limiter (default=F)")
    slmax: float = Field(0.2, description="Max Slope")
    lvar1d: bool = Field(
        False, description="For 1d-mode if variable dx is used; not used with SCHISM"
    )
    loptsig: bool = Field(
        False,
        description="Use optimal distributions of freq. in spectral space ... fi+1 = fi * 1.1. Take care what you high freq. limit is!",
    )


class Init(NamelistBaseModel):
    """Initialization parameters"""

    lhotr: bool = Field(
        False, description="Use hotstart file (see &HOTFILE section for input file)"
    )
    linid: bool = Field(
        True,
        description="Initial condition; F for default; use T if using WW3 as i.c. etc",
    )
    initstyle: Literal[1, 2] = Field(
        2,
        description="1 - Parametric Jonswap, 2 - Read from Global NETCDF files, work only if IBOUNDFORMAT=3",
    )


class Bouc(NamelistBaseModel):
    """Boundary condition parameters"""

    lbcse: bool = Field(True, description="The wave boundary data is time dependent")
    lbinter: bool = Field(
        True,
        description="Do interpolation in time if LBCSE=T (not available for quasi-steady mode within the subtime steps)",
    )
    lbcwa: bool = Field(True, description="Parametric Wave Spectra")
    lbcsp: bool = Field(
        False,
        description="Specify (non-parametric) wave spectra, specified in 'FILEWAVE' below",
    )
    linhom: bool = Field(True, description="Non-uniform wave b.c. in space")
    lbsp1d: bool = Field(
        False,
        description="1D (freq. space only) format for FILEWAVE if LBCSP=T and LINHOM=F",
    )
    lbsp2d: bool = Field(
        False, description="2D format for FILEWAVE if LBCSP=T and LINHOM=F"
    )
    begtc: str = Field(
        "20030908.000000", description="Begin time of the wave boundary file (FILEWAVE)"
    )
    deltc: int = Field(1, description="Time step in FILEWAVE")
    unitc: Literal["HR", "MIN", "SEC"] = Field(
        "HR", description="Unit can be HR, MIN, SEC"
    )
    endtc: str = Field("20031008.000000", description="End time")
    filebound: str = Field(
        "wwmbnd.gr3",
        description="Boundary file defining boundary conditions and Neumann nodes",
    )
    iboundformat: Literal[1, 3, 6] = Field(
        3,
        description="1 ~ WWM, 3 ~ WW3 (2D spectra in netcdf format only - LBCWA=T), 6 ~ WW3 2D spectra in netcdf format with LBCSP=T (prescribed spectra)",
    )
    filewave: str = Field(
        "bndfiles.dat", description="Boundary file defining boundary input from WW3"
    )
    lindsprdeg: bool = Field(
        False,
        description="If 1-d wave spectra are read this flag defines whether the input for the directional spreading is in degrees (true) or exponent (false)",
    )
    lparmdir: bool = Field(
        False,
        description="If LPARMDIR is true then directional spreading is read from WBDS and must be in exponential format at this time, only valid for 1d Spectra",
    )
    wbhs: float = Field(2.0, description="Hs at the boundary for parametric spectra")
    wbss: Literal[1, -1, 2, -2, 3, -3, 4] = Field(
        2, description="Spectral shape parameter"
    )
    wbtp: float = Field(
        8.0,
        description="Tp at the boundary (sec); mean or peak depending on the sign of WBSS",
    )
    wbdm: float = Field(90.0, description="Avg. Wave Direction at the boundary")
    wbdsms: Literal[1, 2] = Field(
        1, description="Directional spreading value in degrees (1) or as exponent (2)"
    )
    wbds: float = Field(
        10.0, description="Directional spreading at the boundary (degrees/exponent)"
    )
    wbgauss: float = Field(
        0.1, description="factor for gaussian distribution if WBSS=1"
    )
    wbpken: float = Field(
        3.3, description="Peak enhancement factor for Jonswap Spectra if WBSS=2"
    )

    @field_validator("wbss")
    def validate_wbss(cls, v):
        if v not in [1, -1, 2, -2, 3, -3, 4]:
            raise ValueError("WBSS must be 1, -1, 2, -2, 3, -3, or 4")
        return v


class Engs(NamelistBaseModel):
    """Source terms parameters"""

    mesnl: Literal[0, 1] = Field(
        1,
        description="Nonlinear Interaction NL4 , 1 ~ on, 0 ~ off (Discrete Interaction approx.)",
    )
    mesin: Literal[0, 1, 2, 3, 4, 5] = Field(1, description="Wind input formulation")
    ifric: Literal[1, 4] = Field(
        1, description="Formulation for atmospheric boundary layer"
    )
    mesbf: Literal[1, 2, 3] = Field(1, description="Bottom friction formulation")
    fricc: float = Field(0.067, description="Bottom friction coefficient or roughness")
    mesbr: Literal[0, 1] = Field(
        1, description="Shallow water wave breaking; 0: off; 1: on"
    )
    ibreak: Literal[1, 2, 3, 4, 5, 6] = Field(
        1, description="Wave breaking formulation"
    )
    icrit: Literal[1, 2, 3, 4, 5, 6] = Field(1, description="Wave breaking criterion")
    brcr: float = Field(0.78, description="Wave breaking parameter")
    a_brcr: float = Field(0.76, description="Wave breaking parameter")
    b_brcr: float = Field(0.29, description="Wave breaking parameter")
    min_brcr: float = Field(0.25, description="Wave breaking parameter")
    max_brcr: float = Field(0.8, description="Wave breaking parameter")
    a_biph: float = Field(0.2, description="Biphase coefficient")
    br_coef_method: Literal[1, 2] = Field(
        1, description="Method for the breaking coefficient"
    )
    b_alp: float = Field(0.5, description="breaking coefficient")
    zprof_break: Literal[1, 2, 3, 4, 5, 6] = Field(
        2, description="Vertical distribution function of wave breaking source term"
    )
    bc_break: Literal[0, 1] = Field(
        1, description="Apply depth-limited breaking at the boundaries"
    )
    iroller: Literal[0, 1] = Field(0, description="Wave roller model")
    alprol: float = Field(
        0.85, description="Alpha coefficient for the wave roller model"
    )
    meveg: Literal[0, 1] = Field(0, description="Vegetation on/off")
    lmaxetot: bool = Field(
        True, description="Limit shallow water wave height by wave breaking limiter"
    )
    mesds: Literal[1, 2] = Field(
        1, description="Formulation for the whitecapping source function"
    )
    mestr: Literal[0, 1, 2] = Field(
        1, description="Formulation for the triad 3 wave interactions"
    )
    trico: float = Field(0.1, description="proportionality const. (\\alpha_EB)")
    trira: float = Field(
        2.5, description="ratio of max. freq. considered in triads over mean freq."
    )
    triurs: float = Field(0.1, description="critical Ursell number")


class Nums(NamelistBaseModel):
    """Numerical parameters"""

    icomp: Literal[0, 1, 2, 3] = Field(
        3,
        description="Controls splitting method and implicit/explicit schemes for spectral advection",
    )
    amethod: Literal[0, 1, 2, 3, 4, 5, 6, 7] = Field(
        7, description="Controls different Methods in geographical space"
    )
    smethod: Literal[0, 1, 2, 3, 4, 6] = Field(
        1, description="Controls the way the source terms are integrated"
    )
    dmethod: Literal[0, 1, 2, 3, 4] = Field(
        2, description="Controls the numerical method in directional space"
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
    fmethod: Literal[0, 1] = Field(
        1, description="Controls the numerical method used in freq. space"
    )
    lfiltersig: bool = Field(
        False, description="Limit the advection velocity in freq. space"
    )
    maxcflsig: float = Field(
        1.0, description="Max Cfl in freq. space; used only if LFILTERSIG=T"
    )
    llimt: bool = Field(True, description="Switch on/off Action limiter")
    melim: Literal[1, 2, 3] = Field(1, description="Formulation for the action limiter")
    limfak: float = Field(
        0.1, description="Proportionality coefficient for the action limiter"
    )
    ldifr: bool = Field(
        False, description="Use phase decoupled diffraction approximation"
    )
    idiffr: Literal[1, 2, 3] = Field(
        1, description="Extended WAE accounting for higher order effects"
    )
    lconv: bool = Field(
        False, description="Estimate convergence criterian and write disk"
    )
    lcfl: bool = Field(False, description="Write out CFL numbers")
    nqsiter: int = Field(
        1, description="# of quasi-steady (Q-S) sub-divisions within each WWM time step"
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
        description="Number of grid points [%/100] that have to fulfill rel. avg. wave steepness criteria EPSH4",
    )
    qsconv5: float = Field(
        0.98,
        description="Number of grid points [%/100] that have to fulfill avg. rel. wave period criteria EPSH5",
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
    ivector: Literal[1, 2, 3, 4, 5, 6] = Field(
        2, description="Different flavours of communications"
    )
    ladvtest: bool = Field(False, description="for testing the advection schemes")
    lchkconv: bool = Field(
        False, description="needs to set to .true. for quasi-steady mode"
    )
    dtmin_dyn: float = Field(
        1.0, description="min. time step (sec?) for dynamic integration"
    )
    ndyniter: int = Field(100, description="max. iteration for dyn. scheme")
    dtmin_sin: float = Field(
        1.0, description="min. time steps for the full fractional step method"
    )
    dtmin_snl4: float = Field(1.0, description="min. time steps for SNL4")
    dtmin_sds: float = Field(1.0, description="min. time steps for SDS")
    dtmin_snl3: float = Field(1.0, description="min. time steps for SNL3")
    dtmin_sbr: float = Field(0.10, description="min. time steps for SBR")
    dtmin_sbf: float = Field(1.0, description="min. time steps for SBF")
    ndyniter_sin: int = Field(10, description="max. iterations for SIN")
    ndyniter_snl4: int = Field(10, description="max. iterations for SNL4")
    ndyniter_sds: int = Field(10, description="max. iterations for SDS")
    ndyniter_sbr: int = Field(10, description="max. iterations for SBR")
    ndyniter_snl3: int = Field(10, description="max. iterations for SNL3")
    ndyniter_sbf: int = Field(10, description="max. iterations for SBF")
    lsoubound: bool = Field(False, description="Do source terms on boundary")
    wae_solverthr: float = Field(
        1e-6, description="Threshold for the Block-Jacobi or Block-Gauss-Seider solver"
    )
    maxiter: int = Field(1000, description="Max. number of iterations")
    pmin: float = Field(1.0, description="Max. percentage of non-converged grid points")
    lnaninfchk: bool = Field(False, description="Check for NaN and INF")
    lzeta_setup: bool = Field(False, description="Compute wave setup")
    zeta_meth: Literal[0] = Field(0, description="Method for wave setup")
    lsourceswam: bool = Field(
        False, description="Use ECMWF WAM formualtion for deep water physics"
    )
    block_gauss_seidel: bool = Field(
        True, description="Use the Gauss Seidel on each computer block"
    )
    lnonl: bool = Field(
        False,
        description="Solve the nonlinear system using simpler algorithm (Patankar)",
    )
    aspar_local_level: Literal[0, 1, 2, 3, 4, 5] = Field(
        0, description="Aspar locality level"
    )
    l_solver_norm: bool = Field(
        False,
        description="Compute solver norm ||A*x-b|| as termination check of jacobi-Gauss-Seidel solver",
    )
    laccel: bool = Field(False, description="Acceleration flag")


class History(NamelistBaseModel):
    """History output parameters"""

    begtc: str = Field(
        "20030908.000000", description="Start output time, yyyymmdd. hhmmss"
    )
    deltc: int = Field(1, description="Time step for output")
    unitc: Literal["SEC"] = Field("SEC", description="Unit")
    endtc: str = Field(
        "20031008.000000", description="Stop time output, yyyymmdd. hhmmss"
    )
    definetc: int = Field(86400, description="Time scoop (sec) for history files")
    outstyle: Literal["NO", "NC", "XFN", "SHP"] = Field(
        "NO", description="output option"
    )
    multipleout: Literal[0, 1] = Field(0, description="Output file format")
    use_single_out: bool = Field(
        True, description="Use single precision in the output of model variables"
    )
    paramwrite: bool = Field(
        True,
        description="Write the physical parametrization and chosen numerical method in the netcdf file",
    )
    gridwrite: bool = Field(
        True, description="Write the grid in the netcdf history file"
    )
    printmma: bool = Field(
        False,
        description="Print minimum, maximum and average value of statistics during runtime",
    )
    fileout: str = Field("wwm_hist.dat", description="Output file name")
    hs: bool = Field(False, description="Output significant wave height")
    tm01: bool = Field(False, description="Output mean period")
    tm02: bool = Field(False, description="Output zero-crossing mean period")
    klm: bool = Field(False, description="Output mean wave number")
    wlm: bool = Field(False, description="Output mean wave length")
    etotc: bool = Field(False, description="Output variable ETOTC")
    etots: bool = Field(False, description="Output variable ETOTS")
    dm: bool = Field(False, description="Output mean wave direction")
    dspr: bool = Field(False, description="Output directional spreading")
    tppd: bool = Field(False, description="Output direction of the peak")
    tpp: bool = Field(False, description="Output peak period")
    cpp: bool = Field(False, description="Output peak phase vel.")
    wnpp: bool = Field(False, description="Output peak wave number")
    cgpp: bool = Field(False, description="Output peak group speed")
    kpp: bool = Field(False, description="Output peak wave number")
    lpp: bool = Field(False, description="Output peak wave length")
    peakd: bool = Field(False, description="Output peak direction")
    peakdspr: bool = Field(False, description="Output peak directional spreading")
    dpeak: bool = Field(False, description="Output peak direction")
    ubot: bool = Field(False, description="Output bottom exc. vel.")
    orbital: bool = Field(False, description="Output bottom orbital vel.")
    botexper: bool = Field(False, description="Output bottom exc.")
    tmbot: bool = Field(False, description="Output bottom period")
    ursell: bool = Field(False, description="Output Ursell number")
    ufric: bool = Field(False, description="Output air friction velocity")
    z0: bool = Field(False, description="Output air roughness length")
    alpha_ch: bool = Field(False, description="Output Charnoch coefficient for air")
    windx: bool = Field(False, description="Output Wind in X direction")
    windy: bool = Field(False, description="Output Wind in Y direction")
    cd: bool = Field(False, description="Output Drag coefficient")
    currtx: bool = Field(False, description="Output current in X direction")
    currty: bool = Field(False, description="Output current in Y direction")
    watlev: bool = Field(False, description="Output water level")
    watlevold: bool = Field(
        False, description="Output water level at previous time step"
    )
    depdt: bool = Field(False, description="Output change of water level in time")
    dep: bool = Field(False, description="Output depth")
    tauw: bool = Field(False, description="Output surface stress from the wave")
    tauhf: bool = Field(False, description="Output high frequency surface stress")
    tautot: bool = Field(False, description="Output total surface stress")
    stokessurfx: bool = Field(
        False, description="Output Surface Stokes drift in X direction"
    )
    stokessurfy: bool = Field(
        False, description="Output Surface Stokes drift in Y direction"
    )
    stokesbarox: bool = Field(
        False, description="Output Barotropic Stokes drift in X direction"
    )
    stokesbaroy: bool = Field(
        False, description="Output Barotropic Stokes drift in Y direction"
    )
    rsxx: bool = Field(False, description="Output RSXX potential of LH")
    rsxy: bool = Field(False, description="Output RSXY potential of LH")
    rsyy: bool = Field(False, description="Output RSYY potential of LH")
    cfl1: bool = Field(False, description="Output CFL number 1")
    cfl2: bool = Field(False, description="Output CFL number 2")
    cfl3: bool = Field(False, description="Output CFL number 3")


class Station(NamelistBaseModel):
    """Station output parameters"""

    begtc: str = Field(
        "20030908.000000", description="Start simulation time, yyyymmdd. hhmmss"
    )
    deltc: int = Field(600, description="Time step for output")
    unitc: Literal["SEC"] = Field("SEC", description="Unit")
    endtc: str = Field(
        "20031008.000000", description="Stop time simulation, yyyymmdd. hhmmss"
    )
    definetc: int = Field(86400, description="Time for definition of station files")
    outstyle: Literal["NO", "STE", "NC"] = Field("NO", description="output option")
    multipleout: Literal[0, 1] = Field(0, description="Output file format")
    use_single_out: bool = Field(
        True, description="Use single precision in the output of model variables"
    )
    paramwrite: bool = Field(
        True,
        description="Write the physical parametrization and chosen numerical method in the netcdf file",
    )
    fileout: str = Field("wwm_sta.dat", description="Output file name")
    loutiter: bool = Field(False, description="Output iteration flag")
    iouts: int = Field(15, description="Number of output stations")
    nouts: list[str] = Field(
        [
            "P-1",
            "P-2",
            "P-3",
            "P-4",
            "P-5",
            "P-6",
            "P-7",
            "P-8",
            "P-9",
            "P-10",
            "P-11",
            "P-12",
            "P-13",
            "P-14",
            "P-15",
        ],
        description="Names of output stations",
    )
    xouts: list[float] = Field(
        [
            -76.046,
            -76.778,
            -75.81,
            -75.72,
            -74.842,
            -74.703,
            -75.33,
            -72.631,
            -74.835,
            -69.248,
            -72.6,
        ],
        description="X coordinates of output stations",
    )
    youts: list[float] = Field(
        [
            39.152,
            38.556,
            38.033,
            37.551,
            36.974,
            37.204,
            37.023,
            36.915,
            36.611,
            38.461,
            35.75,
            34.561,
            31.862,
            40.503,
            39.584,
        ],
        description="Y coordinates of output stations",
    )
    cutoff: list[float] = Field(
        [0.44] * 15, description="cutoff freq (Hz) for each station"
    )
    lsp1d: bool = Field(True, description="1D spectral station output")
    lsp2d: bool = Field(False, description="2D spectral station output")
    lsigmax: bool = Field(True, description="Adjust the cut-freq. for the output")
    ac: bool = Field(False, description="Output spectrum")
    wk: bool = Field(False, description="Output variable WK")
    acout_1d: bool = Field(False, description="Output variable ACOUT_1D")
    acout_2d: bool = Field(False, description="Output variable ACOUT_2D")
    hs: bool = Field(False, description="Output significant wave height")
    tm01: bool = Field(False, description="Output mean period")
    tm02: bool = Field(False, description="Output zero-crossing mean period")
    klm: bool = Field(False, description="Output mean wave number")
    wlm: bool = Field(False, description="Output mean wave length")
    etotc: bool = Field(False, description="Output Variable ETOTC")
    etots: bool = Field(False, description="Output Variable ETOTS")
    dm: bool = Field(False, description="Output mean wave direction")
    dspr: bool = Field(False, description="Output directional spreading")
    tppd: bool = Field(False, description="Output Discrete Peak Period")
    tpp: bool = Field(False, description="Output Peak Period")
    cpp: bool = Field(False, description="Output CPP")
    wnpp: bool = Field(False, description="Output peak wave number")
    cgpp: bool = Field(False, description="Output peak group speed")
    kpp: bool = Field(False, description="Output peak wave number")
    lpp: bool = Field(False, description="Output peak")
    peakd: bool = Field(False, description="Output peak direction")
    peakdspr: bool = Field(False, description="Output peak directional spreading")
    dpeak: bool = Field(False, description="Output DPEAK")
    ubot: bool = Field(False, description="Output UBOT")
    orbital: bool = Field(False, description="Output ORBITAL")
    botexper: bool = Field(False, description="Output BOTEXPER")
    tmbot: bool = Field(False, description="Output TMBOT")
    ursell: bool = Field(False, description="Output Ursell number")
    ufric: bool = Field(False, description="Output air friction velocity")
    z0: bool = Field(False, description="Output air roughness length")
    alpha_ch: bool = Field(False, description="Output Charnoch coefficient for air")
    windx: bool = Field(False, description="Output Wind in X direction")
    windy: bool = Field(False, description="Output Wind in Y direction")
    cd: bool = Field(False, description="Output Drag coefficient")
    currtx: bool = Field(False, description="Output current in X direction")
    currty: bool = Field(False, description="Output current in Y direction")
    watlev: bool = Field(False, description="Output water level")
    watlevold: bool = Field(
        False, description="Output water level at previous time step"
    )
    depdt: bool = Field(False, description="Output change of water level in time")
    dep: bool = Field(False, description="Output depth")
    tauw: bool = Field(False, description="Output surface stress from the wave")
    tauhf: bool = Field(False, description="Output high frequency surface stress")
    tautot: bool = Field(False, description="Output total surface stress")
    stokessurfx: bool = Field(
        False, description="Output Surface Stokes drift in X direction"
    )
    stokessurfy: bool = Field(
        False, description="Output Surface Stokes drift in Y direction"
    )
    stokesbarox: bool = Field(
        False, description="Output Barotropic Stokes drift in X direction"
    )
    stokesbaroy: bool = Field(
        False, description="Output Barotropic Stokes drift in Y direction"
    )
    rsxx: bool = Field(False, description="Output RSXX potential of LH")
    rsxy: bool = Field(False, description="Output RSXY potential of LH")
    rsyy: bool = Field(False, description="Output RSYY potential of LH")
    cfl1: bool = Field(False, description="Output CFL number 1")
    cfl2: bool = Field(False, description="Output CFL number 2")
    cfl3: bool = Field(False, description="Output CFL number 3")


class Hotfile(NamelistBaseModel):
    """Configuration for WWM hotfile input and output."""

    lhotf: bool = Field(True, description="Write hotfile")
    filehot_out: str = Field("hotfile_out_WWM.nc", description="Name of output hotfile")
    begtc: str = Field(
        "20030908.000000", description="Starting time of hotfile writing"
    )
    deltc: float = Field(86400.0, description="Time between hotfile writes")
    unitc: Literal["SEC"] = Field("SEC", description="Unit used for time")
    endtc: str = Field("20031008.000000", description="Ending time of hotfile writing")
    lcyclehot: bool = Field(True, description="Cyclic hotfile writing for netcdf")
    hotstyle_out: Literal[1, 2] = Field(
        2, description="Output hotfile style (1: binary, 2: netcdf)"
    )
    multipleout: Literal[0, 1] = Field(0, description="Single or multiple output files")
    filehot_in: str = Field("hotfile_in_WWM.nc", description="Input hotfile name")
    hotstyle_in: Literal[1, 2] = Field(
        2, description="Input hotfile style (1: binary, 2: netcdf)"
    )
    ihotpos_in: Literal[1, 2] = Field(
        1, description="Position in input hotfile for reading"
    )
    multiplein: Literal[0, 1] = Field(0, description="Single or multiple input files")

    @field_validator("begtc", "endtc")
    def validate_datetime(cls, v):
        try:
            datetime.strptime(v, "%Y%m%d.%H%M%S")
        except ValueError:
            raise ValueError("Incorrect date format, should be YYYYMMDD.HHMMSS")
        return v

    @model_validator(mode="after")
    def check_time_range(self):
        start = datetime.strptime(self.begtc, "%Y%m%d.%H%M%S")
        end = datetime.strptime(self.endtc, "%Y%m%d.%H%M%S")
        if start >= end:
            raise ValueError("BEGTC must be earlier than ENDTC")
        return self


class PetscOptions(NamelistBaseModel):
    """Configuration for PETSc linear solver options."""

    ksptype: Literal["LGMRES", "GMRES", "DGMRES", "PGMRES", "KSPBCGSL"] = Field(
        "LGMRES", description="PETSc solver type"
    )
    rtol: float = Field(1e-20, description="Relative convergence tolerance")
    abstol: float = Field(1e-20, description="Absolute convergence tolerance")
    dtol: float = Field(10000.0, description="Divergence tolerance")
    maxits: int = Field(1000, description="Maximum number of iterations")
    initialguessnonzero: bool = Field(False, description="Initial guess is nonzero")
    gmrespreallocate: bool = Field(True, description="Preallocate GMRES work vectors")
    pctype: Literal["SOR", "ASM", "HYPRE", "SPAI", "NONE"] = Field(
        "SOR", description="PETSc preconditioner type"
    )

    @field_validator("rtol", "abstol")
    def validate_tolerance(cls, v):
        if v <= 0:
            raise ValueError("Tolerance must be positive")
        return v

    @field_validator("maxits")
    def validate_maxits(cls, v):
        if v <= 0:
            raise ValueError("Maximum iterations must be positive")
        return v


class Nesting(NamelistBaseModel):
    """Configuration for WWM nesting options."""

    list_begtc: str = Field("", description="List of beginning times for nesting")
    list_deltc: float = Field(0.0, description="List of time steps for nesting")
    list_unitc: str = Field("", description="List of time units for nesting")
    list_endtc: str = Field("", description="List of ending times for nesting")
    list_igridtype: int = Field(0, description="List of grid types for nesting")
    list_filegrid: str = Field("", description="List of grid files for nesting")
    list_filebound: str = Field("", description="List of boundary files for nesting")
    list_prefix: str = Field("", description="List of prefixes for nesting")


class Wwminput_WW3(NamelistBaseModel):
    """Main configuration for WWM (Wind Wave Model) based on Hurricane Isabel for Chesapeake Bay test."""

    hotfile: Optional[Hotfile] = Field(default=None)
    petscoptions: Optional[PetscOptions] = Field(default=None)
    nesting: Optional[Nesting] = Field(default=None)
