# This file was auto generated from a SCHISM namelist file on 2025-01-24.

from datetime import datetime
from typing import List, Optional

from pydantic import Field, field_validator, model_validator

from rompy.schism.namelists.basemodel import NamelistBaseModel


class Core(NamelistBaseModel):
    ipre: Optional[int] = Field(
        0,
        description="Pre-processor flag (1: on; 0: off). Useful for checking grid errors etc. Use 1 core only for compute (plus necessary scribe cores) when enabled. Under scribe I/O, the code (scribe part) will hang but outputs will be available. Job should be manually terminated.",
    )
    ibc: Optional[int] = Field(
        0,
        description="Baroclinic option flag. If set to 0 (baroclinic model), ibtp is not used.",
    )
    ibtp: Optional[int] = Field(
        1, description="Barotropic option flag. Only used when ibc is not 0."
    )
    rnday: Optional[float] = Field(30, description="Total run time in days.")
    dt: Optional[float] = Field(100.0, description="Time step in seconds.")
    msc2: Optional[int] = Field(
        24,
        description="Number of spectral frequencies for WWM grid. Must be the same as msc in .nml for consistency between SCHISM and WWM.",
    )
    mdc2: Optional[int] = Field(
        24,
        description="Number of directional bins for WWM grid. Must be the same as mdc in .nml for consistency between SCHISM and WWM.",
    )
    ntracer_gen: Optional[int] = Field(
        2, description="Number of tracers in user-defined module (USE_GEN)."
    )
    ntracer_age: Optional[int] = Field(
        4,
        description="Number of tracers for age calculation (USE_AGE). Must be equal to 2*N where N is the number of age tracers.",
    )
    sed_class: Optional[int] = Field(
        5, description="Number of sediment classes for SED3D module (USE_SED)."
    )
    eco_class: Optional[int] = Field(
        27,
        description="Number of ecological variables for EcoSim module (USE_ECO). Must be between 25 and 60, inclusive.",
    )
    nspool: Optional[int] = Field(
        36, description="Output step spool for global output controls."
    )
    ihfskip: Optional[int] = Field(
        864,
        description="Stack spool for global output controls. Every ihfskip steps will be put into 1_*, 2_*, etc.",
    )
    nbins_veg_vert: Optional[int] = Field(
        2,
        description="Number of vertical bins for vegetation model. Only used if iveg=1.",
    )

    @field_validator("ipre")
    @classmethod
    def validate_ipre(cls, v):
        if v not in [0, 1]:
            raise ValueError("ipre must be either 0 or 1")
        return v

    @field_validator("ibc")
    @classmethod
    def validate_ibc(cls, v):
        if v not in [0, 1]:
            raise ValueError("ibc must be either 0 or 1")
        return v

    @field_validator("ibtp")
    @classmethod
    def validate_ibtp(cls, v):
        if v not in [0, 1]:
            raise ValueError("ibtp must be either 0 or 1")
        return v

    @field_validator("rnday")
    @classmethod
    def validate_rnday(cls, v):
        if v <= 0:
            raise ValueError("rnday must be positive")
        return v

    @field_validator("dt")
    @classmethod
    def validate_dt(cls, v):
        if v <= 0:
            raise ValueError("dt must be positive")
        return v

    @field_validator("msc2")
    @classmethod
    def validate_msc2(cls, v):
        if v <= 0:
            raise ValueError("msc2 must be positive")
        return v

    @field_validator("mdc2")
    @classmethod
    def validate_mdc2(cls, v):
        if v <= 0:
            raise ValueError("mdc2 must be positive")
        return v

    @field_validator("ntracer_gen")
    @classmethod
    def validate_ntracer_gen(cls, v):
        if v < 0:
            raise ValueError("ntracer_gen must be non-negative")
        return v

    @field_validator("ntracer_age")
    @classmethod
    def validate_ntracer_age(cls, v):
        if v % 2 != 0 or v < 0:
            raise ValueError("ntracer_age must be a non-negative even number")
        return v

    @field_validator("sed_class")
    @classmethod
    def validate_sed_class(cls, v):
        if v < 0:
            raise ValueError("sed_class must be non-negative")
        return v

    @field_validator("eco_class")
    @classmethod
    def validate_eco_class(cls, v):
        if v < 25 or v > 60:
            raise ValueError("eco_class must be between 25 and 60")
        return v

    @field_validator("nspool")
    @classmethod
    def validate_nspool(cls, v):
        if v <= 0:
            raise ValueError("nspool must be positive")
        return v

    @field_validator("ihfskip")
    @classmethod
    def validate_ihfskip(cls, v):
        if v <= 0:
            raise ValueError("ihfskip must be positive")
        return v

    @model_validator(mode="after")
    def validate_ibc_ibtp(self):
        if self.ibc == 0 and self.ibtp != 1:
            raise ValueError("When ibc is 0, ibtp must be 1")
        return self


class Opt(NamelistBaseModel):
    ipre2: Optional[int] = Field(
        0,
        description="Pre-processing flag for diagnostic outputs. If non-zero, the code will output drag coefficients (Cdp) and stop.",
    )
    itransport_only: Optional[int] = Field(
        0,
        description="Option to solve only tracer transport. 0: off, 1 or 2: on. When 2, additional variables are needed.",
    )
    iloadtide: Optional[int] = Field(
        0,
        description="Option to add self-attracting and loading tide (SAL) into tidal potential. 0: off, 1: needs loadtide_[FREQ].gr3 inputs, 2 or 3: simple scaling for gravity approach.",
    )
    loadtide_coef: Optional[float] = Field(
        0.1,
        description="Coefficient for SAL scaling. Used only if iloadtide is 2 or 3.",
    )
    start_year: Optional[int] = Field(
        2000, description="Starting year for the simulation."
    )
    start_month: Optional[int] = Field(
        1, description="Starting month for the simulation."
    )
    start_day: Optional[int] = Field(1, description="int")
    start_hour: Optional[int] = Field(0, description="double")
    utc_start: Optional[int] = Field(0, description="double")
    ics: Optional[int] = Field(2, description="Coordinate option")
    ihot: Optional[int] = Field(0, description="")
    ieos_type: Optional[int] = Field(0, description="")
    ieos_pres: Optional[int] = Field(
        0, description="used only if ieos_type=0. 0: without pressure effects"
    )
    eos_a: Optional[float] = Field(
        -0.1, description="needed if ieos_type=1; should be <=0"
    )
    eos_b: Optional[float] = Field(1001.0, description="needed if ieos_type=1")
    dramp: Optional[float] = Field(
        1.0, description="ramp-up period in days for b.c. etc (no ramp-up if <=0)"
    )
    drampbc: Optional[float] = Field(
        0.0, description="ramp-up period in days for baroclinic force"
    )
    iupwind_mom: Optional[int] = Field(0, description="")
    indvel: Optional[int] = Field(0, description="")
    ihorcon: Optional[int] = Field(0, description="")
    hvis_coef0: Optional[float] = Field(
        0.025,
        description="const. diffusion # if ihorcon/=0; <=0.025 for ihorcon=2, <=0.125 for ihorcon=1",
    )
    ishapiro: Optional[int] = Field(1, description="options")
    niter_shap: Optional[int] = Field(
        1, description="needed if ishapiro/=0: # of iterations with Shapiro filter"
    )
    shapiro0: Optional[float] = Field(0.5, description="needed only if ishapiro=1")
    thetai: Optional[float] = Field(0.6, description="")
    icou_elfe_wwm: Optional[int] = Field(0, description="")
    nstep_wwm: Optional[int] = Field(
        1, description="call WWM every this many time steps"
    )
    iwbl: Optional[int] = Field(
        0, description="wave boundary layer formulation (used only if USE_WMM and"
    )
    hmin_radstress: Optional[float] = Field(
        1.0,
        description="min. total water depth used only in radiation stress calculation [m]",
    )
    drampwafo: Optional[float] = Field(
        0.0,
        description="ramp-up period in days for the wave forces (no ramp-up if <=0)",
    )
    turbinj: Optional[float] = Field(
        0.15,
        description="% of depth-induced wave breaking energy injected in turbulence",
    )
    turbinjds: Optional[float] = Field(
        1.0,
        description="% of wave energy dissipated through whitecapping injected in turbulence",
    )
    alphaw: Optional[float] = Field(
        0.5,
        description="for itur=4 : scaling parameter for the surface roughness z0s = alphaw*Hm0.",
    )
    fwvor_advxy_stokes: Optional[int] = Field(
        1, description="--> Stokes drift advection (xy), Coriolis"
    )
    fwvor_advz_stokes: Optional[int] = Field(
        1, description="--> Stokes drift advection (z) , Coriolis"
    )
    fwvor_gradpress: Optional[int] = Field(1, description="--> Pressure term")
    fwvor_breaking: Optional[int] = Field(1, description="--> Wave breaking")
    fwvor_streaming: Optional[int] = Field(
        1, description="--> Wave streaming (works with iwbl /= 0)"
    )
    fwvor_wveg: Optional[int] = Field(
        0, description="--> Wave dissipation by vegetation acceleration term"
    )
    fwvor_wveg_nl: Optional[int] = Field(
        0,
        description="--> Non linear intrawave vegetation force (see Dean and Bender, 2006 or van Rooijen et al., 2016 for details)",
    )
    cur_wwm: Optional[int] = Field(0, description="Coupling current in WWM")
    wafo_obcramp: Optional[int] = Field(
        0, description="Ramp on wave forces at open boundary (1: on / 0: off)"
    )
    imm: Optional[int] = Field(0, description="")
    ibdef: Optional[int] = Field(
        10, description="needed if imm=1; # of steps used in deformation"
    )
    slam0: Optional[int] = Field(-124, description="lon - not really used")
    sfea0: Optional[int] = Field(45, description="lat")
    iunder_deep: Optional[int] = Field(0, description="")
    h1_bcc: Optional[float] = Field(50.0, description="[m]")
    h2_bcc: Optional[float] = Field(100.0, description="[m]; >h1_bcc")
    hw_depth: Optional[float] = Field(1000000.0, description="threshold depth in [m]")
    hw_ratio: Optional[float] = Field(0.5, description="ratio")
    ihydraulics: Optional[int] = Field(0, description="")
    if_source: Optional[int] = Field(0, description="")
    dramp_ss: Optional[int] = Field(
        2,
        description="needed if if_source/=0; ramp-up period in days for source/sinks (no ramp-up if <=0)",
    )
    # Removed in SCHISM 5.12
    # meth_sink: Optional[int] = Field(1, description="options to treat sinks @ dry elem")
    lev_tr_source__1: Optional[int] = Field(-9, description="T")
    lev_tr_source__2: Optional[int] = Field(-9, description="S")
    lev_tr_source__3: Optional[int] = Field(-9, description="GEN")
    lev_tr_source__4: Optional[int] = Field(
        -9, description="AGE: set -9999. in msource's AGE section"
    )
    lev_tr_source__5: Optional[int] = Field(-9, description="SED3D")
    lev_tr_source__6: Optional[int] = Field(-9, description="EcoSim")
    lev_tr_source__7: Optional[int] = Field(-9, description="ICM")
    lev_tr_source__8: Optional[int] = Field(-9, description="CoSINE")
    lev_tr_source__9: Optional[int] = Field(-9, description="Feco")
    lev_tr_source__10: Optional[int] = Field(-9, description="TIMOR")
    lev_tr_source__11: Optional[int] = Field(-9, description="FABM")
    lev_tr_source__12: Optional[int] = Field(-9, description="DVD")
    level_age: Optional[list] = Field(
        [9, -999], description="default: -999 (all levels)"
    )
    ihdif: Optional[int] = Field(0, description="")
    nchi: Optional[int] = Field(0, description="")
    dzb_min: Optional[float] = Field(
        0.5, description="needed if nchi=1; min. bottom boundary layer thickness [m]."
    )
    hmin_man: Optional[float] = Field(
        1.0, description="needed if nchi=-1: min. depth in Manning's formulation [m]"
    )
    ncor: Optional[int] = Field(1, description="should usually be 1 if ics=2")
    rlatitude: Optional[int] = Field(46, description="if ncor=-1")
    coricoef: Optional[int] = Field(0, description="if ncor=0")
    ic_elev: Optional[int] = Field(0, description="")
    nramp_elev: Optional[int] = Field(0, description="")
    inv_atm_bnd: Optional[int] = Field(0, description="0: off; 1: on")
    prmsl_ref: Optional[float] = Field(
        101325.0, description="reference atmos. pressure on bnd [Pa]"
    )
    flag_ic__1: Optional[int] = Field(0, description="T")
    flag_ic__2: Optional[int] = Field(0, description="S")
    flag_ic__3: Optional[int] = Field(1, description="GEN (user defined module)")
    flag_ic__5: Optional[int] = Field(1, description="SED3D")
    flag_ic__6: Optional[int] = Field(1, description="EcoSim")
    flag_ic__7: Optional[int] = Field(1, description="ICM")
    flag_ic__8: Optional[int] = Field(1, description="CoSINE")
    flag_ic__9: Optional[int] = Field(1, description="FIB")
    flag_ic__10: Optional[int] = Field(1, description="TIMOR")
    flag_ic__11: Optional[int] = Field(1, description="FABM")
    flag_ic__12: Optional[int] = Field(0, description="DVD (must=0)")
    gen_wsett: Optional[int] = Field(0, description="1.e-4")
    ibcc_mean: Optional[int] = Field(0, description="")
    rmaxvel: Optional[float] = Field(5.0, description="")
    velmin_btrack: Optional[float] = Field(0.0001, description="")
    btrack_nudge: Optional[float] = Field(0.009013, description="")
    ihhat: Optional[int] = Field(1, description="")
    inunfl: Optional[int] = Field(0, description="")
    h0: Optional[float] = Field(
        0.01, description="min. water depth for wetting/drying [m]"
    )
    shorewafo: Optional[int] = Field(0, description="Matters only if USE_WWM")
    moitn0: Optional[int] = Field(
        50, description="output spool for solver info; used only with JCG"
    )
    mxitn0: Optional[int] = Field(1500, description="max. iteration allowed")
    rtol0: Optional[float] = Field(1e-12, description="error tolerance")
    nadv: Optional[int] = Field(1, description="")
    dtb_max: Optional[float] = Field(30.0, description="in sec")
    dtb_min: Optional[float] = Field(10.0, description="")
    inter_mom: Optional[int] = Field(0, description="")
    kr_co: Optional[int] = Field(1, description="not used if inter_mom=0")
    itr_met: Optional[int] = Field(3, description="")
    h_tvd: Optional[float] = Field(5.0, description="cut-off depth (m)")
    eps1_tvd_imp: Optional[float] = Field(
        0.0001,
        description="suggested value is 1.e-4, but for large suspended load, need to use a smaller value (e.g. 1.e-9)",
    )
    eps2_tvd_imp: Optional[float] = Field(1e-14, description="")
    ielm_transport: Optional[int] = Field(0, description="1: turn on")
    max_subcyc: Optional[int] = Field(
        10,
        description="used only if ielm_transport/=0. Max # of subcycling per time step in transport allowed",
    )
    ip_weno: Optional[int] = Field(
        2,
        description="order of accuracy: 0- upwind; 1- linear polynomial, 2nd order; 2- quadratic polynomial, 3rd order",
    )
    courant_weno: Optional[float] = Field(
        0.5, description="Courant number for weno transport"
    )
    nquad: Optional[int] = Field(
        2, description="number of quad points on each side, nquad= 1 or 2"
    )
    ntd_weno: Optional[int] = Field(
        1,
        description="order of temporal discretization: (1) Euler (default); (3): 3rd-order Runge-Kutta (only for benchmarking)",
    )
    epsilon1: Optional[float] = Field(
        1e-15,
        description="coefficient for 2nd order weno smoother (larger values are more prone to numerical dispersion)",
    )
    epsilon2: Optional[float] = Field(
        1e-10,
        description="1st coefficient for 3rd order weno smoother (larger values are more prone to numerical dispersion",
    )
    i_prtnftl_weno: Optional[int] = Field(
        0,
        description="option for writing nonfatal errors on invalid temp. or salinity for density: (0) off; (1) on.",
    )
    epsilon3: Optional[float] = Field(
        1e-25,
        description="2nd coefficient for 3rd order weno smoother (inactive at the moment)",
    )
    ielad_weno: Optional[int] = Field(
        0,
        description="ielad, if ielad=1, use ELAD method to suppress dispersion (inactive at the moment)",
    )
    small_elad: Optional[float] = Field(
        0.0001, description="small (inactive at the moment)"
    )
    nws: Optional[int] = Field(
        0,
        description="""
        Atmos. option. Use nws=2 and USE_ATMOS for coupling with atmospheric model.
        If nws=0, no atmos. forcing is applied. If nws=1, atmos.
        variables are read in from wind.th. If nws=2, atmos. variables are
        read in from sflux_ files.
        If nws=4, ascii format is used for wind and atmos. pressure at each node (see source code).
        If nws=-1 (requires USE_PAHM), use Holland parametric wind model (barotropic only with wind and atmos. pressure).
        In this case, the Holland model is called every step so wtiminc is not used. An extra 
        input file is needed: hurricane-track.dat, in addition to a few parameters below.

        Stress calculation:
        If nws=2, ihconsv=1 and iwind_form=0, the stress is calculated from heat exchange
        routine; in this case USE_ATMOS cannot be on.
        Otherwise if iwind_form=-1, the stress is calculated from Pond & Pichard formulation;
        if iwind_form=1, Hwang (2018) formulation (Cd tapers off at high wind).
        If WWM is enabled and icou_elfe_wwm>0 and iwind_form=-2 or -3, stress is overwritten by WWM:
        If iwind_form=-2, stress=rho_air*ufric^2; scaled by rho_water
        If iwind_form=-3, the stress is calculated according to the param. of Donelan et al. (1993) based on the wave age.
        In all cases, if USE_ICE the stress in ice-covered portion is calculated by ICE routine.
        """,
    )
    wtiminc: Optional[float] = Field(
        150.0, description="time step for atmos. forcing. Default: same as dt"
    )
    drampwind: Optional[float] = Field(
        1.0, description="ramp-up period in days for wind (no ramp-up if <=0)"
    )
    iwindoff: Optional[int] = Field(
        0, description="needed only if nws/=0; '1': needs windfactor.gr3"
    )
    iwind_form: Optional[int] = Field(1, description="needed if nws/=0")
    model_type_pahm: Optional[int] = Field(
        10,
        description="only used if nws=-1: hurricane model type (1: Holland; 10: GAHM)",
    )
    ihconsv: Optional[int] = Field(0, description="heat exchange option")
    isconsv: Optional[int] = Field(0, description="evaporation/precipitation model")
    i_hmin_airsea_ex: Optional[int] = Field(2, description="no effect if ihconsv=0")
    hmin_airsea_ex: Optional[float] = Field(
        0.2, description="[m], no effect if ihconsv=0"
    )
    i_hmin_salt_ex: Optional[int] = Field(2, description="no effect if isconsv=0")
    hmin_salt_ex: Optional[float] = Field(
        0.2, description="[m], no effect if isconsv=0"
    )
    iprecip_off_bnd: Optional[int] = Field(
        0, description="if /=0, precip will be turned off near land bnd"
    )
    itur: Optional[int] = Field(3, description="Default: 0")
    dfv0: Optional[float] = Field(0.01, description="needed if itur=0")
    dfh0: Optional[float] = Field(0.0001, description="needed if itur=0")
    mid: Optional[str] = Field("KL", description="needed if itur=3,5. Use KE if itur=5")
    stab: Optional[str] = Field(
        "KC",
        description="needed if itur=3 or 5. Use 'GA' if turb_met='MY'; otherwise use 'KC'.",
    )
    xlsc0: Optional[float] = Field(
        0.1,
        description="needed if itur=3 or 5. Scale for surface & bottom mixing length (>0)",
    )
    inu_elev: Optional[int] = Field(0, description="")
    inu_uv: Optional[int] = Field(0, description="")
    inu_tr__1: Optional[int] = Field(0, description="T")
    inu_tr__2: Optional[int] = Field(0, description="S")
    inu_tr__3: Optional[int] = Field(0, description="GEN")
    inu_tr__4: Optional[int] = Field(0, description="Age")
    inu_tr__5: Optional[int] = Field(0, description="SED3D")
    inu_tr__6: Optional[int] = Field(0, description="EcoSim")
    inu_tr__7: Optional[int] = Field(0, description="ICM")
    inu_tr__8: Optional[int] = Field(0, description="CoSINE")
    inu_tr__9: Optional[int] = Field(0, description="FIB")
    inu_tr__10: Optional[int] = Field(0, description="TIMOR")
    inu_tr__11: Optional[int] = Field(0, description="FABM")
    inu_tr__12: Optional[int] = Field(0, description="DVD (must=0)")
    nu_sum_mult: Optional[int] = Field(
        1, description="1: final relax is sum of horizontal"
    )

    @field_validator("ipre2")
    @classmethod
    def validate_ipre2(cls, v):
        if v not in [0, 1]:
            raise ValueError("ipre2 must be 0 or 1")
        return v

    @field_validator("itransport_only")
    @classmethod
    def validate_itransport_only(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("itransport_only must be 0, 1, or 2")
        return v

    @field_validator("iloadtide")
    @classmethod
    def validate_iloadtide(cls, v):
        if v not in [0, 1, 2, 3]:
            raise ValueError("iloadtide must be 0, 1, 2, or 3")
        return v

    @field_validator("loadtide_coef")
    @classmethod
    def validate_loadtide_coef(cls, v):
        if v < 0 or v > 1:
            raise ValueError("loadtide_coef must be between 0 and 1")
        return v

    @field_validator("start_year")
    @classmethod
    def validate_start_year(cls, v):
        if v < 1900 or v > 2100:
            raise ValueError("start_year must be between 1900 and 2100")
        return v

    @field_validator("start_month")
    @classmethod
    def validate_start_month(cls, v):
        if v < 1 or v > 12:
            raise ValueError("start_month must be between 1 and 12")
        return v

    @model_validator(mode="after")
    def check_loadtide_coef(self):
        if self.iloadtide in [2, 3] and self.loadtide_coef == 0:
            raise ValueError("loadtide_coef must be set when iloadtide is 2 or 3")
        return self

class Vegetation(NamelistBaseModel):
    # isav: Optional[int] = Field(
    #     0,
    #     description="Flag for vegetation model. 0: off, 1: on. Requires additional input files if enabled.",
    # )
    iveg: Optional[int] = Field(
        0,
        description="""
        !----------------------------------------------------------------------
        ! Vegetation model
        ! If iveg/=0, need 4 extra inputs: (1) veg_D.gr3 (depth is stem diameter in meters);
        ! (2) veg_N.gr3 (depth is # of stems per m^2);
        ! (3) veg_h.gr3 (height of canopy in meters);
        ! (4) veg_cd.gr3 (drag coefficient).
        ! With iveg=1, the vertical scaling is given by veg_vert_scale_[cd,N,D](1:nbins_veg_vert+1) below.
        ! veg_vert_z(:) specify the distance from bed for each bin (ascending order starting from 0).
        ! iveg=2: flex vegeation using Ganthy (2011) fomulation

        ! If one of these depths=0 at a node, the code will set all to 0. 
        ! If USE_MARSH is on and isav=1, all .gr3 must have constant depths!
        !----------------------------------------------------------------------
        """,
    )
    veg_vert_z: Optional[List[float]] = Field(
        [0.,0.5,1.],
        description="Depths for vertical scaling of vegetation model (in meters).",
    )
    veg_vert_scale_cd: Optional[List[float]] = Field(
        [1.0, 1.0, 1.0],
        description="Vertical scaling for drag coefficient. Only used if iveg=1.",
    )
    veg_vert_scale_n: Optional[List[float]] = Field(
        [1.0, 1.0, 1.0],
        description="Vertical scaling for number of stems per m^2. Only used if iveg=1.",
    )
    veg_vert_scale_d: Optional[List[float]] = Field(
        [1.0, 1.0, 1.0],
        description="Vertical scaling for stem diameter. Only used if iveg=1.",
    )
    veg_lai: Optional[float] = Field(
        1.0,
        description="Leaf Area Index [-]; used if iveg=2. Ganthy suggests 0-10?",
    )
    veg_cw: Optional[float] = Field(
        1.5,
        description="calibration coefficient in diameter of bent leaf [-]; used if iveg=2. Ganthy suggests 0-25?",
    )

    @field_validator("iveg")
    @classmethod
    def validate_iveg(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("iveg must be 0, 1, or 2")
        return v
    
    @field_validator("veg_vert_z")
    @classmethod
    def validate_veg_vert_z(cls, v):
        # Check if the list is in ascending order and contains floats
        if not all(isinstance(i, float) for i in v):
            raise ValueError("All elements in veg_vert_z must be floats")
        if not all(v[i] < v[i + 1] for i in range(len(v) - 1)):
            raise ValueError("veg_vert_z must be in ascending order")
        return v
    @field_validator("veg_vert_scale_cd")
    @classmethod
    def validate_veg_vert_scale_cd(cls, v, values):
        if not all(isinstance(i, float) for i in v):
            raise ValueError("All elements in veg_vert_scale_cd must be floats")
        return v
    @field_validator("veg_vert_scale_n")
    @classmethod
    def validate_veg_vert_scale_n(cls, v, values):
        if not all(isinstance(i, float) for i in v):
            raise ValueError("All elements in veg_vert_scale_n must be floats")
        return v
    @field_validator("veg_vert_scale_d")
    @classmethod
    def validate_veg_vert_scale_d(cls, v, values):
        if not all(isinstance(i, float) for i in v):
            raise ValueError("All elements in veg_vert_scale_d must be floats")
        return v
    @field_validator("veg_lai")
    @classmethod
    def validate_veg_lai(cls, v):
        if v < 0:
            raise ValueError("veg_lai must be non-negative")
        return v
    @field_validator("veg_cw")
    @classmethod
    def validate_veg_cw(cls, v):
        if v < 0:
            raise ValueError("veg_cw must be non-negative")
        return v


class Vertical(NamelistBaseModel):
    vnh1: Optional[int] = Field(
        400,
        description="Vertical nudging depth 1 in meters. Used in vertical relaxation scheme.",
    )
    vnf1: Optional[float] = Field(
        0.0,
        description="Vertical relaxation factor for depth 1. Must be between 0 and 1.",
    )
    vnh2: Optional[int] = Field(
        500,
        description="Vertical nudging depth 2 in meters. Must be greater than vnh1.",
    )
    vnf2: Optional[float] = Field(
        0.0,
        description="Vertical relaxation factor for depth 2. Must be between 0 and 1.",
    )
    step_nu_tr: Optional[float] = Field(
        86400.0,
        description="Time step in seconds for all *_nu.nc files when inu_[MOD]=2.",
    )
    h_bcc1: Optional[float] = Field(
        100.0,
        description="Cut-off depth for cubic spline interpolation near bottom when computing horizontal gradients.",
    )
    s1_mxnbt: Optional[float] = Field(
        0.5, description="Dimensioning parameter for inter-subdomain backtracking."
    )
    s2_mxnbt: Optional[float] = Field(
        3.5,
        description="Another dimensioning parameter for inter-subdomain backtracking.",
    )
    iharind: Optional[int] = Field(
        0,
        description="Flag for harmonic analysis of elevation. 0 for off, non-zero for on.",
    )
    iflux: Optional[int] = Field(
        0,
        description="Conservation check option. 0: off, 1: basic output, 2: more elaborate outputs.",
    )
    izonal5: Optional[int] = Field(
        0,
        description="Flag for Williamson test #5 (zonal flow over an isolated mount). 0 for off, non-zero for on.",
    )
    ibtrack_test: Optional[int] = Field(
        0,
        description="Flag for rotating Gausshill test with stratified T,S. 0: off, 1: on.",
    )
    irouse_test: Optional[int] = Field(
        0,
        description="Flag for Rouse profile test. 0: off, 1: on. Requires USE_TIMOR to be on if enabled.",
    )
    flag_fib: Optional[int] = Field(
        1,
        description="Flag to choose FIB model for bacteria decay. 1: Constant decay rate, 2: Canteras et al., 1995, 3: Servais et al., 2007.",
    )
    slr_rate: Optional[float] = Field(
        120.0,
        description="Sea-level rise rate in mm/year for marsh model. Only used if USE_MARSH is on.",
    )
    nstep_ice: Optional[int] = Field(
        1, description="Number of SCHISM steps between calls to the ICE module."
    )
    rearth_pole: Optional[float] = Field(
        6378206.4, description="Earth's radius at the pole in meters."
    )
    rearth_eq: Optional[float] = Field(
        6378206.4, description="Earth's radius at the equator in meters."
    )
    shw: Optional[float] = Field(
        4184.0, description="Specific heat of water (C_p) in J/kg/K."
    )
    rho0: Optional[float] = Field(
        1000.0,
        description="Reference water density for Boussinesq approximation in kg/m^3.",
    )
    vclose_surf_frac: Optional[float] = Field(
        1.0,
        description="Fraction of vertical flux closure adjustment applied at surface for T,S.",
    )
    iadjust_mass_consv0__1: Optional[int] = Field(0, description="T")
    iadjust_mass_consv0__2: Optional[int] = Field(0, description="S")
    iadjust_mass_consv0__3: Optional[int] = Field(0, description="GEN")
    iadjust_mass_consv0__4: Optional[int] = Field(0, description="AGE")
    iadjust_mass_consv0__5: Optional[int] = Field(
        0, description="SED3D (code won't allow non-0 for this module)"
    )
    iadjust_mass_consv0__6: Optional[int] = Field(0, description="EcoSim")
    iadjust_mass_consv0__7: Optional[int] = Field(0, description="ICM")
    iadjust_mass_consv0__8: Optional[int] = Field(0, description="CoSiNE")
    iadjust_mass_consv0__9: Optional[int] = Field(0, description="Feco")
    iadjust_mass_consv0__10: Optional[int] = Field(0, description="TIMOR")
    iadjust_mass_consv0__11: Optional[int] = Field(0, description="FABM")
    iadjust_mass_consv0__12: Optional[int] = Field(0, description="DVD (must=0)")
    h_massconsv: Optional[float] = Field(
        2.0, description="Threshold depth for ICM mass conservation in meters."
    )
    rinflation_icm: Optional[float] = Field(
        0.001,
        description="Maximum ratio between H^{n+1} and H^n allowed for ICM mass conservation.",
    )

    @field_validator("vnh1")
    @classmethod
    def check_vnh1(cls, v):
        if v <= 0:
            raise ValueError("vnh1 must be positive")
        return v

    @field_validator("vnf1")
    @classmethod
    def check_vnf1(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("vnf1 must be between 0 and 1")
        return v

    @field_validator("vnh2")
    @classmethod
    def check_vnh2(cls, v):
        if v <= 0:
            raise ValueError("vnh2 must be positive")
        return v

    @field_validator("vnf2")
    @classmethod
    def check_vnf2(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("vnf2 must be between 0 and 1")
        return v

    @field_validator("step_nu_tr")
    @classmethod
    def check_step_nu_tr(cls, v):
        if v <= 0:
            raise ValueError("step_nu_tr must be positive")
        return v

    @field_validator("h_bcc1")
    @classmethod
    def check_h_bcc1(cls, v):
        if v <= 0:
            raise ValueError("h_bcc1 must be positive")
        return v

    @field_validator("s1_mxnbt")
    @classmethod
    def check_s1_mxnbt(cls, v):
        if v <= 0:
            raise ValueError("s1_mxnbt must be positive")
        return v

    @field_validator("s2_mxnbt")
    @classmethod
    def check_s2_mxnbt(cls, v):
        if v <= 0:
            raise ValueError("s2_mxnbt must be positive")
        return v

    @field_validator("iharind")
    @classmethod
    def check_iharind(cls, v):
        if not isinstance(v, int):
            raise ValueError("iharind must be an integer")
        return v

    @field_validator("iflux")
    @classmethod
    def check_iflux(cls, v):
        if v not in [0, 1, 2]:
            raise ValueError("iflux must be 0, 1, or 2")
        return v

    @field_validator("izonal5")
    @classmethod
    def check_izonal5(cls, v):
        if not isinstance(v, int):
            raise ValueError("izonal5 must be an integer")
        return v

    @field_validator("ibtrack_test")
    @classmethod
    def check_ibtrack_test(cls, v):
        if v not in [0, 1]:
            raise ValueError("ibtrack_test must be 0 or 1")
        return v

    @field_validator("irouse_test")
    @classmethod
    def check_irouse_test(cls, v):
        if v not in [0, 1]:
            raise ValueError("irouse_test must be 0 or 1")
        return v

    @field_validator("flag_fib")
    @classmethod
    def check_flag_fib(cls, v):
        if v not in [1, 2, 3]:
            raise ValueError("flag_fib must be 1, 2, or 3")
        return v

    @field_validator("slr_rate")
    @classmethod
    def check_slr_rate(cls, v):
        if v < 0:
            raise ValueError("slr_rate must be non-negative")
        return v

    @field_validator("nstep_ice")
    @classmethod
    def check_nstep_ice(cls, v):
        if v <= 0:
            raise ValueError("nstep_ice must be positive")
        return v

    @field_validator("rearth_pole")
    @classmethod
    def check_rearth_pole(cls, v):
        if v <= 0:
            raise ValueError("rearth_pole must be positive")
        return v

    @field_validator("rearth_eq")
    @classmethod
    def check_rearth_eq(cls, v):
        if v <= 0:
            raise ValueError("rearth_eq must be positive")
        return v

    @field_validator("shw")
    @classmethod
    def check_shw(cls, v):
        # if float(v.replace("d", "")) <= 0:
        if v <= 0:
            raise ValueError("shw must be positive")
        return v

    @field_validator("rho0")
    @classmethod
    def check_rho0(cls, v):
        if v <= 900:
        # if float(v.replace("d", "")) <= 0:
            raise ValueError("rho0 must be positive")
        return v

    @field_validator("vclose_surf_frac")
    @classmethod
    def check_vclose_surf_frac(cls, v):
        if not 0 <= v <= 1:
            raise ValueError("vclose_surf_frac must be between 0 and 1")
        return v

    @field_validator("h_massconsv")
    @classmethod
    def check_h_massconsv(cls, v):
        if v <= 0:
            raise ValueError("h_massconsv must be positive")
        return v

    @field_validator("rinflation_icm")
    @classmethod
    def check_rinflation_icm(cls, v):
        if v <= 0:
            raise ValueError("rinflation_icm must be positive")
        return v

    @model_validator(mode="after")
    def check_vnh2_greater_than_vnh1(self):
        if self.vnh2 <= self.vnh1:
            raise ValueError("vnh2 must be greater than vnh1")
        return self


class Schout(NamelistBaseModel):
    nc_out: Optional[int] = Field(
        1,
        description="Main switch to control netcdf output. If 0, SCHISM won't output nc files at all.",
    )
    iof_ugrid: Optional[int] = Field(
        0,
        description="UGRID option for 3D outputs. If non-zero, 3D outputs will have UGRID metadata.",
    )
    nhot: Optional[int] = Field(
        0,
        description="Option for hotstart outputs. If 1, outputs hotstart every 'nhot_write' steps.",
    )
    nhot_write: Optional[int] = Field(
        8640,
        description="Number of steps between hotstart outputs. Must be a multiple of ihfskip if nhot=1.",
    )
    iout_sta: Optional[int] = Field(
        0,
        description="Station output option. If non-zero, requires output skip (nspool_sta) and a station.in file.",
    )
    nspool_sta: Optional[int] = Field(
        10,
        description="Number of steps between station outputs. Required if iout_sta is non-zero.",
    )
    iof_hydro__1: Optional[int] = Field(
        1, description="0: off; 1: on - elev. [m]  {elevation}  2D"
    )
    iof_hydro__2: Optional[int] = Field(
        0, description="air pressure [Pa]  {airPressure}  2D"
    )
    iof_hydro__3: Optional[int] = Field(
        0, description="air temperature [C] {airTemperature}  2D"
    )
    iof_hydro__4: Optional[int] = Field(
        0, description="Specific humidity [-] {specificHumidity}  2D"
    )
    iof_hydro__5: Optional[int] = Field(
        0,
        description="Net downward solar (shortwave) radiation after albedo [W/m/m] {solarRadiation}  2D",
    )
    iof_hydro__6: Optional[int] = Field(
        0, description="sensible flux (positive upward) [W/m/m]  {sensibleHeat}  2D"
    )
    iof_hydro__7: Optional[int] = Field(
        0, description="latent heat flux (positive upward) [W/m/m] {latentHeat}  2D"
    )
    iof_hydro__8: Optional[int] = Field(
        0,
        description="upward longwave radiation (positive upward) [W/m/m] {upwardLongwave}  2D",
    )
    iof_hydro__9: Optional[int] = Field(
        0,
        description="downward longwave radiation (positive downward) [W/m/m] {downwardLongwave}  2D",
    )
    iof_hydro__10: Optional[int] = Field(
        0, description="total flux=-flsu-fllu-(radu-radd) [W/m/m] {totalHeat}  2D"
    )
    iof_hydro__11: Optional[int] = Field(
        0, description="evaporation rate [kg/m/m/s] {evaporationRate}  2D"
    )
    iof_hydro__12: Optional[int] = Field(
        0, description="precipitation rate [kg/m/m/s] {precipitationRate}  2D"
    )
    iof_hydro__13: Optional[int] = Field(
        0,
        description="Bottom stress vector [kg/m/s^2(Pa)] {bottomStressX,Y}  2D vector",
    )
    iof_hydro__14: Optional[int] = Field(
        0, description="wind velocity vector [m/s] {windSpeedX,Y}  2D vector"
    )
    iof_hydro__15: Optional[int] = Field(
        0, description="wind stress vector [m^2/s/s] {windStressX,Y}  2D vector"
    )
    iof_hydro__16: Optional[int] = Field(
        1, description="depth-averaged vel vector [m/s] {depthAverageVelX,Y}  2D vector"
    )
    iof_hydro__17: Optional[int] = Field(
        0, description="vertical velocity [m/s] {verticalVelocity}  3D"
    )
    iof_hydro__18: Optional[int] = Field(
        0, description="water temperature [C] {temperature}  3D"
    )
    iof_hydro__19: Optional[int] = Field(
        0, description="water salinity [PSU] {salinity}  3D"
    )
    iof_hydro__20: Optional[int] = Field(
        0, description="water density [kg/m^3] {waterDensity}  3D"
    )
    iof_hydro__21: Optional[int] = Field(
        0, description="vertical eddy diffusivity [m^2/s] {diffusivity}  3D"
    )
    iof_hydro__22: Optional[int] = Field(
        0, description="vertical eddy viscosity [m^2/s] {viscosity}  3D"
    )
    iof_hydro__23: Optional[int] = Field(
        0, description="turbulent kinetic energy {turbulentKineticEner}   3D"
    )
    iof_hydro__24: Optional[int] = Field(
        0, description="turbulent mixing length [m] {mixingLength}  3D"
    )
    iof_hydro__26: Optional[int] = Field(
        1, description="horizontal vel vector [m/s] {horizontalVelX,Y} 3D vector"
    )
    iof_hydro__27: Optional[int] = Field(
        0,
        description="horizontal vel vector defined @side [m/s] {horizontalSideVelX,Y} 3D vector",
    )
    iof_hydro__28: Optional[int] = Field(
        0, description="vertical vel. @elem [m/s] {verticalVelAtElement} 3D"
    )
    iof_hydro__29: Optional[int] = Field(
        0, description="T @prism centers [C] {temperatureAtElement} 3D"
    )
    iof_hydro__30: Optional[int] = Field(
        0, description="S @prism centers [PSU] {salinityAtElement} 3D"
    )
    iof_hydro__31: Optional[int] = Field(
        0,
        description="Barotropic pressure gradient force vector (m.s-2) @side centers  {pressure_gradient} 2D vector",
    )
    iof_wwm__1: Optional[int] = Field(
        0, description="sig. height (m) {sigWaveHeight}   2D"
    )
    iof_wwm__2: Optional[int] = Field(
        0, description="Mean average period (sec) - TM01 {meanWavePeriod}  2D"
    )
    iof_wwm__3: Optional[int] = Field(
        0,
        description="Zero down crossing period for comparison with buoy (s) - TM02 {zeroDowncrossPeriod}  2D",
    )
    iof_wwm__4: Optional[int] = Field(
        0, description="Average period of wave runup/overtopping - TM10 {TM10}  2D"
    )
    iof_wwm__5: Optional[int] = Field(
        0, description="Mean wave number (1/m) {meanWaveNumber}  2D"
    )
    iof_wwm__6: Optional[int] = Field(
        0, description="Mean wave length (m) {meanWaveLength}  2D"
    )
    iof_wwm__7: Optional[int] = Field(
        0,
        description="Mean average energy transport direction (degr) - MWD in NDBC? {meanWaveDirection}  2D",
    )
    iof_wwm__8: Optional[int] = Field(
        0, description="Mean directional spreading (degr) {meanDirSpreading}  2D"
    )
    iof_wwm__9: Optional[int] = Field(
        0, description="Discrete peak period (sec) - Tp {peakPeriod}  2D"
    )
    iof_wwm__10: Optional[int] = Field(
        0,
        description="Continuous peak period based on higher order moments (sec) {continuousPeakPeriod}  2D",
    )
    iof_wwm__11: Optional[int] = Field(
        0, description="Peak phase vel. (m/s) {peakPhaseVel}  2D"
    )
    iof_wwm__12: Optional[int] = Field(
        0, description="Peak n-factor {peakNFactor}   2D"
    )
    iof_wwm__13: Optional[int] = Field(
        0, description="Peak group vel. (m/s) {peakGroupVel}   2D"
    )
    iof_wwm__14: Optional[int] = Field(
        0, description="Peak wave number {peakWaveNumber}  2D"
    )
    iof_wwm__15: Optional[int] = Field(
        0, description="Peak wave length {peakWaveLength}  2D"
    )
    iof_wwm__16: Optional[int] = Field(
        0, description="Peak (dominant) direction (degr) {dominantDirection}  2D"
    )
    iof_wwm__17: Optional[int] = Field(
        0, description="Peak directional spreading {peakSpreading}  2D"
    )
    iof_wwm__18: Optional[int] = Field(
        0, description="Discrete peak direction (radian?) {discretePeakDirectio}  2D"
    )
    iof_wwm__19: Optional[int] = Field(
        0, description="Orbital vel. (m/s) {orbitalVelocity}  2D"
    )
    iof_wwm__20: Optional[int] = Field(
        0, description="RMS Orbital vel. (m/s) {rmsOrbitalVelocity}  2D"
    )
    iof_wwm__21: Optional[int] = Field(
        0, description="Bottom excursion period (sec?) {bottomExcursionPerio}  2D"
    )
    iof_wwm__22: Optional[int] = Field(
        0, description="Bottom wave period (sec) {bottomWavePeriod}  2D"
    )
    iof_wwm__23: Optional[int] = Field(
        0, description="Uresell number based on peak period {UresellNumber}  2D"
    )
    iof_wwm__24: Optional[int] = Field(
        0, description="Friction velocity (m/s?) {frictionalVelocity}  2D"
    )
    iof_wwm__25: Optional[int] = Field(
        0, description="Charnock coefficient {CharnockCoeff}  2D"
    )
    iof_wwm__26: Optional[int] = Field(
        0, description="Rougness length {rougnessLength}  2D"
    )
    iof_wwm__27: Optional[int] = Field(
        0, description="Roller energy dissipation rate (W/m²) @nodes {Drol} 2D"
    )
    iof_wwm__28: Optional[int] = Field(
        0,
        description="Total wave energy dissipation rate by depth-induced breaking (W/m²) @nodes {wave_sbrtot}  2D",
    )
    iof_wwm__29: Optional[int] = Field(
        0,
        description="Total wave energy dissipation rate by bottom friction (W/m²) @nodes {wave_sbftot} 2D",
    )
    iof_wwm__30: Optional[int] = Field(
        0,
        description="Total wave energy dissipation rate by whitecapping (W/m²) @nodes {wave_sdstot} 2D",
    )
    iof_wwm__31: Optional[int] = Field(
        0,
        description="Total wave energy dissipation rate by vegetation (W/m²) @nodes {wave_svegtot} 2D",
    )
    iof_wwm__32: Optional[int] = Field(
        0,
        description="Total wave energy input rate from atmospheric forcing (W/m²) @nodes {wave_sintot} 2D",
    )
    iof_wwm__33: Optional[int] = Field(
        0, description="WWM_energy vector {waveEnergyDirX,Y}  2D vector"
    )
    iof_wwm__34: Optional[int] = Field(
        0,
        description="Vertical Stokes velocity (m.s-1) @sides and whole levels {stokes_wvel}  3D",
    )
    iof_wwm__35: Optional[int] = Field(
        0,
        description="Wave force vector (m.s-2) computed by wwm @side centers and whole levels {waveForceX,Y}   3D vector",
    )
    iof_wwm__36: Optional[int] = Field(
        0,
        description="Horizontal Stokes velocity (m.s-1) @nodes and whole levels {stokes_hvel} 3D vector",
    )
    iof_wwm__37: Optional[int] = Field(
        0,
        description="Roller contribution to horizontal Stokes velocity (m.s-1) @nodes and whole levels {roller_stokes_hvel} 3D vector",
    )
    # iof_gen__1: Optional[int] = Field(0, description="1st tracer {GEN_1}  3D")
    # iof_gen__2: Optional[int] = Field(0, description="2nd tracer {GEN_2}  3D")
    # iof_age__1: Optional[int] = Field(0, description="{AGE_1}  3D")
    # iof_age__2: Optional[int] = Field(0, description="{AGE_2}  3D")
    # iof_sed__1: Optional[int] = Field(
    #     0, description="total bed thickness @elem (m) {sedBedThickness}  2D"
    # )
    # iof_sed__2: Optional[int] = Field(
    #     0, description="total bed age over all layers @elem (sec) {sedBedAge}  2D"
    # )
    # iof_sed__3: Optional[int] = Field(
    #     0,
    #     description="Sediment transport roughness length @elem (m) (sedTransportRough) {z0st}  2D",
    # )
    # iof_sed__4: Optional[int] = Field(
    #     0,
    #     description="current-ripples roughness length @elem (m) (sedRoughCurrentRippl) {z0cr}  2D",
    # )
    # iof_sed__5: Optional[int] = Field(
    #     0,
    #     description="sand-waves roughness length (m) @elem (z0sw_elem) {sedRoughSandWave}  2D",
    # )
    # iof_sed__6: Optional[int] = Field(
    #     0,
    #     description="wave-ripples roughness length @elem (m) (z0wr_elem) {sedRoughWaveRipple}  2D",
    # )
    # iof_sed__7: Optional[int] = Field(
    #     0,
    #     description="bottom depth _change_ from init. condition (m) {sedDepthChange}  2D",
    # )
    # iof_sed__8: Optional[int] = Field(
    #     0, description="Bed median grain size in the active layer (mm) {sedD50}  2D"
    # )
    # iof_sed__9: Optional[int] = Field(
    #     0, description="Bottom shear stress (Pa) {sedBedStress}  2D"
    # )
    # iof_sed__10: Optional[int] = Field(
    #     0, description="Bottom roughness lenghth (mm) {sedBedRoughness}  2D"
    # )
    # iof_sed__11: Optional[int] = Field(
    #     0, description="sediment porosity in the top layer (-) {sedPorocity}  2D"
    # )
    # iof_sed__12: Optional[int] = Field(
    #     0,
    #     description="erosion flux for suspended transport (kg/m/m/s) {sedErosionalFlux}  2D",
    # )
    # iof_sed__13: Optional[int] = Field(
    #     0,
    #     description="deposition flux for suspended transport (kg/m/m/s) {sedDepositionalFlux}  2D",
    # )
    # iof_sed__14: Optional[int] = Field(
    #     0,
    #     description="Bedload transport rate vector due to wave acceleration (kg/m/s) {sedBedloadTransportX,Y}  2D vector",
    # )
    # iof_sed__15: Optional[int] = Field(
    #     0,
    #     description="Bedload transport rate _vector_ (kg.m-1.s-1) for 1st tracer (one output per class) {sedBedload[X,Y]_1}  2D vector",
    # )
    # iof_sed__16: Optional[int] = Field(
    #     0,
    #     description="Bedload transport of 2nd class {sedBedFraction_[X,Y]_2}  2D vector",
    # )
    # iof_sed__17: Optional[int] = Field(
    #     0,
    #     description="Bed fraction 1st tracer (one output per class) [-] {sedBedFraction_1}   2D",
    # )
    # iof_sed__18: Optional[int] = Field(
    #     0, description="Bed fraction of 2nd class {sedBedFraction_2}   2D"
    # )
    # iof_sed__19: Optional[int] = Field(
    #     0,
    #     description="conc. of 1st class (one output need by each class) [g/L] {sedConcentration_1}   3D",
    # )
    # iof_sed__20: Optional[int] = Field(
    #     0, description="conc. of 2nd class {sedConcentration_2}   3D"
    # )
    # iof_sed__21: Optional[int] = Field(
    #     0, description="total suspended concentration (g/L) {totalSuspendedLoad}  3D"
    # )
    # # iof_eco__1: Optional[list] = Field([0, "{ECO_1}", "3D"], description="") # TODO: Ask Vanessa
    # iof_eco__1: Optional[int] = Field(0, description="")
    # iof_icm_core__1: Optional[int] = Field(1, description="PB1")
    # iof_icm_core__2: Optional[int] = Field(1, description="PB2")
    # iof_icm_core__3: Optional[int] = Field(1, description="PB3")
    # iof_icm_core__4: Optional[int] = Field(1, description="RPOC")
    # iof_icm_core__5: Optional[int] = Field(1, description="LPOC")
    # iof_icm_core__6: Optional[int] = Field(1, description="DOC")
    # iof_icm_core__7: Optional[int] = Field(1, description="RPON")
    # iof_icm_core__8: Optional[int] = Field(1, description="LPON")
    # iof_icm_core__9: Optional[int] = Field(1, description="DON")
    # iof_icm_core__10: Optional[int] = Field(1, description="NH4")
    # iof_icm_core__11: Optional[int] = Field(1, description="NO3")
    # iof_icm_core__12: Optional[int] = Field(1, description="RPOP")
    # iof_icm_core__13: Optional[int] = Field(1, description="LPOP")
    # iof_icm_core__14: Optional[int] = Field(1, description="DOP")
    # iof_icm_core__15: Optional[int] = Field(1, description="PO4")
    # iof_icm_core__16: Optional[int] = Field(1, description="COD")
    # iof_icm_core__17: Optional[int] = Field(1, description="DOX")
    # iof_icm_silica__1: Optional[int] = Field(1, description="SU")
    # iof_icm_silica__2: Optional[int] = Field(1, description="SA")
    # iof_icm_zb__1: Optional[int] = Field(1, description="ZB1")
    # iof_icm_zb__2: Optional[int] = Field(1, description="ZB2")
    # iof_icm_ph__1: Optional[int] = Field(1, description="TIC")
    # iof_icm_ph__2: Optional[int] = Field(1, description="ALK")
    # iof_icm_ph__3: Optional[int] = Field(1, description="CA")
    # iof_icm_ph__4: Optional[int] = Field(1, description="CACO3")
    # iof_icm_cbp__1: Optional[int] = Field(1, description="SRPOC")
    # iof_icm_cbp__2: Optional[int] = Field(1, description="SRPON")
    # iof_icm_cbp__3: Optional[int] = Field(1, description="SRPOP")
    # iof_icm_cbp__4: Optional[int] = Field(1, description="PIP")
    # iof_icm_sav__1: Optional[int] = Field(
    #     1, description="stleaf: total leaf biomass @elem [gC/m^2]"
    # )
    # iof_icm_sav__2: Optional[int] = Field(
    #     1, description="ststem: total stem biomass @elem [gC/m^2]"
    # )
    # iof_icm_sav__3: Optional[int] = Field(
    #     1, description="stroot: total root biomass @elem [gC/m^2]"
    # )
    # iof_icm_sav__4: Optional[int] = Field(
    #     1, description="sht:    canopy height @elem [m]"
    # )
    # iof_icm_veg__1: Optional[int] = Field(
    #     1, description="vtleaf1: leaf biomass group 1 [gC/m^2]"
    # )
    # iof_icm_veg__2: Optional[int] = Field(
    #     1, description="vtleaf2: leaf biomass group 2 [gC/m^2]"
    # )
    # iof_icm_veg__3: Optional[int] = Field(
    #     1, description="vtleaf3: leaf biomass group 3 [gC/m^2]"
    # )
    # iof_icm_veg__4: Optional[int] = Field(
    #     1, description="vtstem1: stem biomass group 1 [gC/m^2]"
    # )
    # iof_icm_veg__5: Optional[int] = Field(
    #     1, description="vtstem2: stem biomass group 2 [gC/m^2]"
    # )
    # iof_icm_veg__6: Optional[int] = Field(
    #     1, description="vtstem3: stem biomass group 3 [gC/m^2]"
    # )
    # iof_icm_veg__7: Optional[int] = Field(
    #     1, description="vtroot1: root biomass group 1 [gC/m^2]"
    # )
    # iof_icm_veg__8: Optional[int] = Field(
    #     1, description="vtroot2: root biomass group 2 [gC/m^2]"
    # )
    # iof_icm_veg__9: Optional[int] = Field(
    #     1, description="vtroot3: root biomass group 3 [gC/m^2]"
    # )
    # iof_icm_veg__10: Optional[int] = Field(
    #     1, description="vht1:    canopy height group 1 [m]"
    # )
    # iof_icm_veg__11: Optional[int] = Field(
    #     1, description="vht2:    canopy height group 2 [m]"
    # )
    # iof_icm_veg__12: Optional[int] = Field(
    #     1, description="vht3:    canopy height group 3 [m]"
    # )
    # iof_icm_sed__1: Optional[int] = Field(1, description="bPOC1 (g.m-3)")
    # iof_icm_sed__2: Optional[int] = Field(1, description="bPOC2 (g.m-3)")
    # iof_icm_sed__3: Optional[int] = Field(1, description="bPOC3 (g.m-3)")
    # iof_icm_sed__4: Optional[int] = Field(1, description="bPON1 (g.m-3)")
    # iof_icm_sed__5: Optional[int] = Field(1, description="bPON2 (g.m-3)")
    # iof_icm_sed__6: Optional[int] = Field(1, description="bPON3 (g.m-3)")
    # iof_icm_sed__7: Optional[int] = Field(1, description="bPOP1 (g.m-3)")
    # iof_icm_sed__8: Optional[int] = Field(1, description="bPOP2 (g.m-3)")
    # iof_icm_sed__9: Optional[int] = Field(1, description="bPOP3 (g.m-3)")
    # iof_icm_sed__10: Optional[int] = Field(1, description="bNH4  (g.m-3)")
    # iof_icm_sed__11: Optional[int] = Field(1, description="bNO3  (g.m-3)")
    # iof_icm_sed__12: Optional[int] = Field(1, description="bPO4  (g.m-3)")
    # iof_icm_sed__13: Optional[int] = Field(1, description="bH2S  (g.m-3)")
    # iof_icm_sed__14: Optional[int] = Field(1, description="bCH4  (g.m-3)")
    # iof_icm_sed__15: Optional[int] = Field(1, description="bPOS  (g.m-3)")
    # iof_icm_sed__16: Optional[int] = Field(1, description="bSA   (g.m-3)")
    # iof_icm_sed__17: Optional[int] = Field(
    #     1, description="bstc: surface transfer coeff. (m/day)"
    # )
    # iof_icm_sed__18: Optional[int] = Field(
    #     1, description="bSTR: benthic stress      (day)"
    # )
    # iof_icm_sed__19: Optional[int] = Field(
    #     1, description="bThp: consective days of hypoxia (day)"
    # )
    # iof_icm_sed__20: Optional[int] = Field(
    #     1, description="bTox: consective days of oxic condition after hypoxia (day)"
    # )
    # iof_icm_sed__21: Optional[int] = Field(1, description="SOD   (g.m-2.day-1)")
    # iof_icm_sed__22: Optional[int] = Field(1, description="JNH4  (g.m-2.day-1)")
    # iof_icm_sed__23: Optional[int] = Field(1, description="JNO3  (g.m-2.day-1)")
    # iof_icm_sed__24: Optional[int] = Field(1, description="JPO4  (g.m-2.day-1)")
    # iof_icm_sed__25: Optional[int] = Field(1, description="JSA   (g.m-2.day-1)")
    # iof_icm_sed__26: Optional[int] = Field(1, description="JCOD  (g.m-2.day-1)")
    # iof_icm_ba__1: Optional[int] = Field(1, description="BA    (g[C].m-2)")
    # iof_icm_dbg__1: Optional[int] = Field(1, description="2D ICM debug variables")
    # iof_icm_dbg__2: Optional[int] = Field(1, description="3D ICM debug variables")
    # iof_cos__1: Optional[int] = Field(0, description="NO3 (uM)")
    # iof_cos__2: Optional[int] = Field(0, description="SiO4(uM)")
    # iof_cos__3: Optional[int] = Field(0, description="NH4 (uM)")
    # iof_cos__4: Optional[int] = Field(0, description="S1  (uM)")
    # iof_cos__5: Optional[int] = Field(0, description="S2  (uM)")
    # iof_cos__6: Optional[int] = Field(0, description="Z1  (uM)")
    # iof_cos__7: Optional[int] = Field(0, description="Z2  (uM)")
    # iof_cos__8: Optional[int] = Field(0, description="DN  (uM)")
    # iof_cos__9: Optional[int] = Field(0, description="DSi (uM)")
    # iof_cos__10: Optional[int] = Field(0, description="PO4 (uM)")
    # iof_cos__11: Optional[int] = Field(0, description="DOX (uM)")
    # iof_cos__12: Optional[int] = Field(0, description="CO2 (uM)")
    # iof_cos__13: Optional[int] = Field(0, description="ALK (uM)")
    # iof_fib__1: Optional[int] = Field(0, description="{FIB_1}  3D")
    # iof_sed2d__1: Optional[int] = Field(
    #     0,
    #     description="bottom depth _change_ from init. condition (m) {SED2D_depth_change}",
    # )
    # iof_sed2d__2: Optional[int] = Field(
    #     0, description="drag coefficient used in transport formulae SED2D_Cd{}"
    # )
    # iof_sed2d__3: Optional[int] = Field(
    #     0, description="Courant number (b.qtot.dt / h.dx) {SED2D_cflsed}"
    # )
    # iof_sed2d__4: Optional[int] = Field(0, description="Top layer d50 (m) {SED2D_d50}")
    # iof_sed2d__5: Optional[int] = Field(
    #     0, description="total transport rate vector (kg/m/s) {SED2D_total_transport}"
    # )
    # iof_sed2d__6: Optional[int] = Field(
    #     0, description="suspended tranport rate vector (kg/m/s) {SED2D_susp_load}"
    # )
    # iof_sed2d__7: Optional[int] = Field(
    #     0, description="bedload transport rate vector (kg/m/s) {SED2D_bed_load}"
    # )
    # iof_sed2d__8: Optional[int] = Field(
    #     0,
    #     description="time averaged total transport rate vector (kg/m/s) {SED2D_average_transport}",
    # )
    # iof_sed2d__9: Optional[int] = Field(
    #     0, description="bottom slope vector (m/m); negative uphill {SED2D_bottom_slope}"
    # )
    # iof_sed2d__10: Optional[int] = Field(
    #     0, description="Total roughness length @elem (m) (z0eq) {z0eq}"
    # )
    # iof_sed2d__11: Optional[int] = Field(
    #     0, description="current-ripples roughness length @elem (m) (z0cr) {z0cr}"
    # )
    # iof_sed2d__12: Optional[int] = Field(
    #     0, description="sand-waves roughness length @elem (m) (z0sw) {z0sw}"
    # )
    # iof_sed2d__13: Optional[int] = Field(
    #     0, description="wave-ripples roughness length @elem (m) (z0wr) {z0wr}"
    # )
    # iof_ice__1: Optional[int] = Field(
    #     0, description="divergence @ elem ('Delta') [1/sec] {iceStrainRate}  2D"
    # )
    # iof_ice__2: Optional[int] = Field(
    #     0, description="ice advective velcoity vector [m/s] {iceVelocityX,Y}  2D vector"
    # )
    # iof_ice__3: Optional[int] = Field(
    #     0,
    #     description="net heat flux to ocean (>0 warm up SST) [W/m/m] {iceNetHeatFlux}  2D",
    # )
    # iof_ice__4: Optional[int] = Field(
    #     0,
    #     description="net fresh water flux to ocean (>0 freshens up SSS) [kg/s/m/m] {iceFreshwaterFlux}  2D",
    # )
    # iof_ice__5: Optional[int] = Field(
    #     0,
    #     description="ice temperature [C] at air-ice interface {iceTopTemperature}  2D",
    # )
    # iof_ice__6: Optional[int] = Field(
    #     0, description="ice volume [m] {iceTracer_1}   2D"
    # )
    # iof_ice__7: Optional[int] = Field(
    #     0, description="ice concentration [-] {iceTracer_2}  2D"
    # )
    # iof_ice__8: Optional[int] = Field(
    #     0, description="snow volume [m] {iceTracer_3}  2D"
    # )
    # iof_ana__1: Optional[int] = Field(
    #     0,
    #     description="min time step at each element over all subcycles in horizontal transport solver [s]   {minTransportTimeStep}  2D",
    # )
    # iof_ana__2: Optional[int] = Field(
    #     0,
    #     description="x-component of \nabla air_pres /\rho_0 [m/s/s] {airPressureGradientX}  2D",
    # )
    # iof_ana__3: Optional[int] = Field(
    #     0,
    #     description="y-component of \nabla air_pres /\rho_0 [m/s/s] {airPressureGradientY}  2D",
    # )
    # iof_ana__4: Optional[int] = Field(
    #     0,
    #     description="\alpha*g*\nabla \Psi [m/s/s] (gradient of tidal potential) {tidePotentialGradX}  2D",
    # )
    # iof_ana__5: Optional[int] = Field(
    #     0, description="\alpha*g*\nabla \Psi [m/s/s] {tidePotentialGradY}  2D"
    # )
    # iof_ana__6: Optional[int] = Field(
    #     0,
    #     description="\nabla \cdot (\mu \nabla u) [m/s/s] (horizontal momentum mixing) {horzontalViscosityX}  3D side",
    # )
    # iof_ana__7: Optional[int] = Field(
    #     0,
    #     description="\nabla \cdot (\mu \nabla v) [m/s/s] {horzontalViscosityY}   3D side",
    # )
    # iof_ana__8: Optional[int] = Field(
    #     0,
    #     description="-g/rho0* \int_z^\eta dr_dx dz  [m/s/s] (b-clinic gradient) {baroclinicForceX}  3D side",
    # )
    # iof_ana__9: Optional[int] = Field(
    #     0,
    #     description="-g/rho0* \int_z^\eta dr_dy dz  [m/s/s] {baroclinicForceY}  3D side",
    # )
    # iof_ana__10: Optional[int] = Field(
    #     0,
    #     description="d (\nu du/dz)/dz [m/s/s] - no vegetation effects (vertical momentum mixing) {verticalViscosityX}  3D side",
    # )
    # iof_ana__11: Optional[int] = Field(
    #     0,
    #     description="d (\nu dv/dz)/dz [m/s/s] - no vegetation effects {verticalViscosityY}  3D side",
    # )
    # iof_ana__12: Optional[int] = Field(
    #     0,
    #     description="(u \cdot \nabla) u [m/s/s] (momentum advection) {mommentumAdvectionX}  3D side",
    # )
    # iof_ana__13: Optional[int] = Field(
    #     0, description="(u \cdot \nabla) u [m/s/s] {mommentumAdvectionY}  3D side"
    # )
    # iof_ana__14: Optional[int] = Field(
    #     0, description="gradient Richardson number [-] {gradientRichardson}   3D"
    # )

    @field_validator("nc_out")
    @classmethod
    def validate_nc_out(cls, v):
        if v not in [0, 1]:
            raise ValueError("nc_out must be 0 or 1")
        return v

    @field_validator("iof_ugrid")
    @classmethod
    def validate_iof_ugrid(cls, v):
        if v < 0:
            raise ValueError("iof_ugrid must be non-negative")
        return v

    @field_validator("nhot")
    @classmethod
    def validate_nhot(cls, v):
        if v not in [0, 1]:
            raise ValueError("nhot must be 0 or 1")
        return v

    @field_validator("nhot_write")
    @classmethod
    def validate_nhot_write(cls, v):
        if v <= 0:
            raise ValueError("nhot_write must be positive")
        return v

    @field_validator("iout_sta")
    @classmethod
    def validate_iout_sta(cls, v):
        if v < 0:
            raise ValueError("iout_sta must be non-negative")
        return v

    @field_validator("nspool_sta")
    @classmethod
    def validate_nspool_sta(cls, v):
        if v <= 0:
            raise ValueError("nspool_sta must be positive")
        return v

    @model_validator(mode="after")
    def validate_nhot_write_multiple(self):
        if self.nhot == 1:
            if self.nhot_write % self.ihfskip != 0:
                raise ValueError("nhot_write must be a multiple of ihfskip when nhot=1")
            if self.ihfskip % self.dt != 0:
                raise ValueError("ihfskip must be a multiple of dt")
        return self

    @model_validator(mode="after")
    def validate_nspool_sta_requirement(self):
        if self.iout_sta != 0 and self.nspool_sta <= 0:
            raise ValueError("nspool_sta must be positive when iout_sta is non-zero")
        return self


class Param(NamelistBaseModel):
    core: Optional[Core] = Field(default_factory=Core)
    opt: Optional[Opt] = Field(default_factory=Opt)
    vertical: Optional[Vertical] = Field(default_factory=Vertical)
    vegetation: Optional[Vegetation] = Field(default_factory=Vegetation)
    schout: Optional[Schout] = Field(default_factory=Schout)
