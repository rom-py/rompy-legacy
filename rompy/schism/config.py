import logging
from pathlib import Path
from typing import Any, Literal, Optional, Union

from pydantic import ConfigDict, Field, model_serializer, model_validator

from rompy.core import (BaseConfig, DataBlob, RompyBaseModel, Spectrum,
                        TimeRange)

# Import plotting functions
from .config_plotting import plot_sflux_spatial, plot_sflux_timeseries
from .config_plotting_boundary import (plot_boundary_points,
                                       plot_boundary_profile,
                                       plot_boundary_timeseries)
from .config_plotting_tides import (plot_tidal_boundaries, plot_tidal_dataset,
                                    plot_tidal_rose, plot_tidal_stations)
from .data import SCHISMData
from .grid import SCHISMGrid
from .interface import TimeInterface
from .namelists import NML
from .namelists.param import Param

logger = logging.getLogger(__name__)

HERE = Path(__file__).parent

CSIRO_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "schismcsiro")
SCHISM_TEMPLATE = str(Path(__file__).parent.parent / "templates" / "schism")


class Inputs(RompyBaseModel):
    filewave: DataBlob | None = Field(
        None,
        description="TODO",
    )

    def get(self, staging_dir: Path):
        ret = {}
        for source in self:
            ret[source[0]] = source[1].get(staging_dir).name
        return ret

    def __str__(self):
        ret = ""
        for forcing in self:
            if forcing[1]:
                ret += f"\t{forcing[0]}: {forcing[1].source}\n"
        return ret


class SchismCSIROConfig(BaseConfig):
    model_type: Literal["schismcsiro"] = Field(
        "schismcsiro", description="The model type for SCHISM."
    )
    grid: SCHISMGrid = Field(description="The model grid")
    data: SCHISMData = Field(description="Model inputs")
    project: str = Field("WAXA", description="TODO")
    utc_start: int = Field(0, description="TODO")
    time_step: float = Field(120.0, description="TODO")
    msc2: int = Field(
        description="same as msc in .nml ... for consitency check between SCHISM and WWM",
        default=36,
    )
    mdc2: int = Field(description="same as mdc in .nml", default=36)
    ihfskip: float = Field(
        description="stack spool; every ihfskip steps will be put into 1_*, 2_*, etc... use 22320.0 for 31 days; 5040 for 7 days; 21600 for 30 days; 20880 for 29 days",
        default=720,
    )
    icou_elfe_wwm: int = Field(
        1,
        description="If WWM is used, set coupling/decoupling flag. Not used if USE_WWM is distabled in Makefile. 0: decoupled so 2 models will run independently; 1: full coupled (elevation, vel, and wind are all passed to WWM); 2: elevation and currents in wwm, no wave force in SCHISM; 3: no elevation and no currents in wwm, wave force in SCHISM; 4: elevation but no currents in wwm, wave force in SCHISM; 5: elevation but no currents in wwm, no wave force in SCHISM; 6: no elevation but currents in wwm, wave force in SCHISM; 7: no elevation but currents in wwm, no wave force in SCHISM; Note that all these parameters must be present in this file (even though not used).",
    )
    nstep_wwm: int = Field(
        3,
        description="call WWM every this many time steps. If /=1, consider using quasi-steady mode in WWM",
    )
    deltc: int = Field(360, description="TODO")
    h1_bcc: float = Field(
        50.0,
        description="Baroclinicity calculation in off/nearshore with iunder_deep=ibc=0. The 'below-bottom' gradient is zeroed out if h>=h2_bcc (i.e. like Z) or uses const extrap (i.e. like terrain-following) if h<=h1_bcc(<h2_bcc) (and linear transition in between based on local depth)",
    )
    h2_bcc: float = Field(
        100.0,
        description="Baroclinicity calculation in off/nearshore with iunder_deep=ibc=0. The 'below-bottom' gradient is zeroed out if h>=h2_bcc (i.e. like Z) or uses const extrap (i.e. like terrain-following) if h<=h1_bcc(<h2_bcc) (and linear transition in between based on local depth)",
    )
    h_bcc1: float = Field(
        default=100.0,
        description="Cut-off depth for cubic spline interpolation near bottom when computing horizontal gradients",
    )
    thetai: float = Field(0.8, description="Implicitness factor (0.5<thetai<=1).")
    iwbl: int = Field(
        0,
        description="wave boundary layer formulation (used only if USE_WMM and icou_elfe_wwm/=0 and nchi=1. If icou_elfe_wwm=0, set iwbl=0): 1-modified Grant-Madsen formulation; 2-Soulsby (1997)",
    )
    slam0: float = Field(
        120.0,
        description="Reference latitude for beta-plane approximation when ncor=1 (not used if ics=2)",
    )
    sfea0: float = Field(
        -29.0,
        description="Reference latitude for beta-plane approximation when ncor=1 (not used if ics=2)",
    )
    nchi: int = Field(-1, description="bottom friction")
    dzb_decayYN: str = Field("!", description="TODO")
    rlatitude: float = Field(-29, description="if ncor=-1")
    ic_elev: int = Field(
        0, description="elevation initial condition flag for cold start only"
    )
    inv_atm_bnd: int = Field(1, description="TODO")
    ibtrack_openbndYN: str = Field("!", description="TODO")
    iwindoffYN: str = Field("!", description="TODO")
    iwind_form: int = Field(
        1,
        description="Needed if nws/=0   !usually use -1, trialling -2 to see if makes a difference",
    )
    sav_cdYN: str = Field("!", description="Save current direction (T/F)")
    iout_sta: int = Field(0, description="Station output option")
    lindsprdeg: str = Field(
        "F", description="Linear interpolation of directional spread (T/F)"
    )
    wbdm: int = Field(90, description="Wave boundary data mode (1-4)")
    extrapYN: str = Field("!", description="Extrapolation of wave boundary data (T/F)")
    extrap: str = Field("T", description="Extrapolation of wave boundary data (T/F)")
    windYN: str = Field("!", description="Wind data (T/F)")
    filewind: str = Field("wind.dat", description="Name of the wind data file")
    currYN: str = Field("!", description="Current data (T/F)")
    walvYN: str = Field("!", description="Wave-induced current data (T/F)")
    mesin: int = Field(1, description="Input message level (0-2)")
    mesbf: int = Field(2, description="Output message level (0-2)")
    fricc: float = Field(0.11, description="Bottom friction factor")
    ibreak: int = Field(1, description="Wave breaking parameterization (1-3)")
    brcrYN: str = Field("", description="Wave breaking parameterization (T/F)")
    melim: int = Field(1, description="Wave energy limit (1-3)")
    limfak: float = Field(0.1, description="Wave energy limit factor")
    lsourceswam: str = Field("F", description="Source term for SWAN (T/F)")
    deltc_out: int = Field(3600, description="Output time step (s)")
    definetc: int = Field(-1, description="Definition of output time step (1-3)")
    outstyle: str = Field("NC", description="Output style (NC/WW3)")
    wwm1: int = Field(1, description="sig. height (m) {sigWaveHeight}   2D")
    wwm2: int = Field(
        1, description="Mean average period (sec) - TM01 {meanWavePeriod}  2D"
    )
    wwm3: int = Field(
        0,
        description="Zero down crossing period for comparison with buoy (s) - TM02 {zeroDowncrossPeriod}  2D",
    )
    wwm4: int = Field(
        0, description="Average period of wave runup/overtopping - TM10 {TM10}  2D"
    )
    wwm5: int = Field(0, description="Mean wave number (1/m) {meanWaveNumber}  2D")
    wwm6: int = Field(0, description="Mean wave length (m) {meanWaveLength}  2D")
    wwm7: int = Field(
        0,
        description="Mean average energy transport direction (degr) - MWD in NDBC? {meanWaveDirection}  2D",
    )
    wwm8: int = Field(
        1, description="Mean directional spreading (degr) {meanDirSpreading}  2D"
    )
    wwm9: int = Field(1, description="Discrete peak period (sec) - Tp {peakPeriod}  2D")
    wwm10: int = Field(
        0,
        description="Continuous peak period based on higher order moments (sec) {continuousPeakPeriod}  2D",
    )
    wwm11: int = Field(0, description="Peak phase vel. (m/s) {peakPhaseVel}  2D")
    wwm12: int = Field(0, description="Peak n-factor {peakNFactor}   2D")
    wwm13: int = Field(0, description="Peak group vel. (m/s) {peakGroupVel}   2D")
    wwm14: int = Field(0, description="Peak wave number {peakWaveNumber}  2D")
    wwm15: int = Field(0, description="Peak wave length {peakWaveLength}  2D")
    wwm16: int = Field(
        1, description="Peak (dominant) direction (degr) {dominantDirection}  2D"
    )
    wwm17: int = Field(1, description="Peak directional spreading {peakSpreading}  2D")
    wwm18: int = Field(
        1, description="Discrete peak direction (radian?) {discretePeakDirectio}  2D"
    )
    wwm19: int = Field(0, description="Orbital vel. (m/s) {orbitalVelocity}  2D")
    wwm20: int = Field(0, description="RMS Orbital vel. (m/s) {rmsOrbitalVelocity}  2D")
    wwm21: int = Field(
        0, description="Bottom excursion period (sec?) {bottomExcursionPerio}  2D"
    )
    wwm22: int = Field(0, description="Bottom wave period (sec) {bottomWavePeriod}  2D")
    wwm23: int = Field(
        0, description="Uresell number based on peak period {UresellNumber}  2D"
    )
    wwm24: int = Field(
        0, description="Friction velocity (m/s?) {frictionalVelocity}  2D"
    )
    wwm25: int = Field(0, description="Charnock coefficient {CharnockCoeff}  2D")
    wwm26: int = Field(0, description="Rougness length {rougnessLength}  2D")
    wwm27: int = Field(
        0, description="Roller energy dissipation rate (W/m²) @nodes {Drol} 2D"
    )
    wwm28: int = Field(
        0,
        description="Total wave energy dissipation rate by depth-induced breaking (W/m²) @nodes {wave_sbrtot}  2D",
    )
    wwm29: int = Field(
        0,
        description="Total wave energy dissipation rate by bottom friction (W/m²) @nodes {wave_sbftot} 2D",
    )
    wwm30: int = Field(
        0,
        description="Total wave energy dissipation rate by whitecapping (W/m²) @nodes {wave_sdstot} 2D",
    )
    wwm31: int = Field(
        0,
        description="Total wave energy dissipation rate by vegetation (W/m²) @nodes {wave_svegtot} 2D",
    )
    wwm32: int = Field(
        0,
        description="Total wave energy input rate from atmospheric forcing (W/m²) @nodes {wave_sintot} 2D",
    )
    wwm33: int = Field(0, description="WWM_energy vector {waveEnergyDirX,Y}  2D vector")
    wwm34: int = Field(
        0,
        description="Vertical Stokes velocity (m.s-1) @sides and whole levels {stokes_wvel}  3D",
    )
    wwm35: int = Field(
        0,
        description="Wave force vector (m.s-2) computed by wwm @side centers and whole levels {waveForceX,Y}   3D vector",
    )
    wwm36: int = Field(
        0,
        description="Horizontal Stokes velocity (m.s-1) @nodes and whole levels {stokes_hvel} 3D vector",
    )
    wwm37: int = Field(
        0,
        description="Roller contribution to horizontal Stokes velocity (m.s-1) @nodes and whole levels {roller_stokes_hvel} 3D vector",
    )
    wwm31YN: str = Field(
        description="Total wave energy dissipation rate by vegetation (W/m2) @nodes {wave_svegtot} 2D",
        default="!",
    )
    wwm32YN: str = Field(
        description="Total wave energy input rate from atmospheric forcing (W/m2) @nodes {wave_sintot} 2D",
        default="!",
    )
    wwm33YN: str = Field(
        description="WWM_energy vector {waveEnergyDirX,Y}  2D vector", default="!"
    )
    wwm34YN: str = Field(
        description="Vertical Stokes velocity (m.s-1) @sides and whole levels {stokes_wvel}  3D",
        default="!",
    )
    wwm35YN: str = Field(
        description="Wave force vector (m.s-2) computed by wwm @side centers and whole levels {waveForceX,Y}   3D vector",
        default="!",
    )
    wwm36YN: str = Field(
        description="Horizontal Stokes velocity (m.s-1) @nodes and whole levels {stokes_hvel} 3D vector",
        default="!",
    )
    wwm37YN: str = Field(
        description="Roller contribution to horizontal Stokes velocity (m.s-1) @nodes and whole levels {roller_stokes_hvel} 3D vector ",
        default="!",
    )
    HS: str = Field("T", description="significant wave height")
    TM01: str = Field("T", description="mean period")
    TM02: str = Field("F", description="zero-crossing mean period")
    DM: str = Field("T", description="mean wave direction")
    DSPR: str = Field("T", description="directional spreading")
    TPP: str = Field("T", description="peak period")
    TPPD: str = Field("T", description="direaction of the peak ... check source code")
    CPP: str = Field("F", description="peak phase velocity")
    WNPP: str = Field("F", description="peak wave number")
    CGPP: str = Field("F", description="peak group speed")
    KPP: str = Field("F", description="peak wave number")
    LPP: str = Field("F", description="peak wave length")
    PEAKD: str = Field("T", description="peak direction")
    DPEAK: str = Field("T", description="peak direction")
    PEAKDSPR: str = Field("T", description="peak directional spreading")
    UBOT: str = Field("F", description="bottom exc. velocity")
    ORBITAL: str = Field("F", description="bottom orbital velocity")
    iouts: str = Field(default=15)
    nouts: str = Field(
        "'AWAC_in','AWAC_mid','AWAC_off','SPOT_1002','SPOT_1011','SPOT_1018','SPOT_1026'",
        description="TODO",
    )
    xouts: str = Field(
        "115.6208687,115.5941886,115.58077,115.5942931,115.5830497,115.5807825,115.5960683",
        description="TODO",
    )
    youts: str = Field(
        "-32.611605,-32.611605,-32.613682,-32.6253914,-32.6135870,-32.6294226,-32.6096741",
        description="TODO",
    )
    lsp2d: str = Field("T", description="TODO")
    ac: str = Field("T", description="TODO")
    template: Optional[str] = Field(
        description="The path to the model template",
        default=CSIRO_TEMPLATE,
    )

    drampwafo: float = Field(1.0, description="TODO")
    nadv: int = Field(1, description="TODO")
    drampwind: float = Field(1.0, description="TODO")
    dramp: float = Field(1.0, description="TODO")
    wwminput_history_DEP: str = Field("F", description="TODO")
    wwminput_history_TAUW: str = Field("F", description="TODO")
    wwminput_history_TAUHF: str = Field("T", description="TODO")
    wwminput_history_TAUTOT: str = Field("T", description="TODO")
    wwminput_history_STOKESSURFX: str = Field("T", description="TODO")
    wwminput_history_STOKESSURFY: str = Field("T", description="TODO")
    wwminput_history_STOKESBAROX: str = Field("T", description="TODO")
    wwminput_history_STOKESBAROY: str = Field("T", description="TODO")
    wwminput_station_DEP: str = Field("T", description="TODO")
    wwminput_station_TAUW: str = Field("T", description="TODO")
    wwminput_station_TAUHF: str = Field("T", description="TODO")
    wwminput_station_TAUTOT: str = Field("T", description="TODO")
    wwminput_station_STOKESSURFX: str = Field("T", description="TODO")
    wwminput_station_STOKESSURFY: str = Field("T", description="TODO")
    wwminput_station_STOKESBAROX: str = Field("T", description="TODO")
    wwminput_station_STOKESBAROY: str = Field("T", description="TODO")
    wwminput_station_OUTSTYLE: str = Field(
        "NO",
        description="'NO' no output 'STE' classic station output (default) 'NC' for netcdf output",
    )
    wwminput_LHOTF: str = Field("F", description="wwminput Write hotfile")
    param_nhot: int = Field(
        0,
        description="use 1 to write out hotstart: output *_hotstart every 'hotout_write' steps",
    )
    param_nhot_writeYN: str = Field(
        "!", description="enables or disables (!) the nhot write field"
    )
    param_nhot_write: float = Field(
        22320.0,
        description="if enabled when nhot and nhotwriteYN enabled then must be a multiple of ihfskip if nhot=1",
    )
    wwminput_LHOTR: str = Field(
        "F", description="Use hotstart file (see &HOTFILE section)"
    )
    wwminput_LINID: str = Field(
        "T",
        description=" Initial condition; F for default; use T if using WW3 as i.c. etc",
    )
    param_iof_hydro1: int = Field(
        1, description="output 0: off; 1: on - elev. [m]  {elev} 2D - Default 1"
    )
    param_iof_hydro2: int = Field(
        0,
        description="output 0: off; 1: on - air pressure [Pa]  {air_pressure} 2D - Default 0",
    )
    param_iof_hydro3: int = Field(
        0,
        description="output 0: off; 1: on - air temperature [C] {air_temperature} 2D - Default 0",
    )
    param_iof_hydro4: int = Field(
        0,
        description="output 0: off; 1: on - Specific humidity [-] {specific_humidity}  2D - Default 0",
    )
    param_iof_hydro5: int = Field(
        0,
        description="output 0: off; 1: on - solar (shortwave) radiation [W/m/m] {solar_radiation}  2D - Default 0",
    )
    param_iof_hydro6: int = Field(
        0,
        description="output 0: off; 1: on - sensible flux (positive upward) [W/m/m]  {sensible_flux}  2D - Default 0",
    )
    param_iof_hydro7: int = Field(
        0,
        description="output 0: off; 1: on - latent heat flux (positive upward) [W/m/m] {latent_heat}   2D - Default 0",
    )
    param_iof_hydro8: int = Field(
        0,
        description="output 0: off; 1: on - upward longwave radiation (positive upward) [W/m/m] {upward_longwave} 2D - Default 0",
    )
    param_iof_hydro9: int = Field(
        0,
        description="output 0: off; 1: on - downward longwave radiation (positive downward) [W/m/m] {downward_longwave}  2D - Default 0",
    )
    param_iof_hydro10: int = Field(
        0,
        description="output 0: off; 1: on - total flux=-flsu-fllu-(radu-radd) [W/m/m] {total_heat_flux}  2D - Default 0",
    )
    param_iof_hydro11: int = Field(
        0,
        description="output 0: off; 1: on - evaporation rate [kg/m/m/s] {evaporation}  2D - Default 0",
    )
    param_iof_hydro12: int = Field(
        0,
        description="output 0: off; 1: on - precipitation rate [kg/m/m/s] {precipitation}  2D - Default 0",
    )
    param_iof_hydro13: int = Field(
        0,
        description="output 0: off; 1: on - Bottom stress vector [kg/m/s^2(Pa)] {bottom_stress} 2D vector - Default 0",
    )
    param_iof_hydro14: int = Field(
        0,
        description="output 0: off; 1: on - wind velocity vector [m/s] {wind_speed}  2D vector - Default 0",
    )
    param_iof_hydro15: int = Field(
        0,
        description="output 0: off; 1: on - wind stress vector [m^2/s/s] {wind_stress}  2D vector - Default 0",
    )
    param_iof_hydro16: int = Field(
        1,
        description="output 0: off; 1: on - depth-averaged vel vector [m/s] {dahv}  2D vector - Default 1",
    )
    param_iof_hydro17: int = Field(
        0,
        description="output 0: off; 1: on - vertical velocity [m/s] {vertical_velocity}  3D - Default 0",
    )
    param_iof_hydro18: int = Field(
        0,
        description="output 0: off; 1: on - water temperature [C] {temp}  3D - Default 0",
    )
    param_iof_hydro19: int = Field(
        0,
        description="output 0: off; 1: on - water salinity [PSU] {salt}   3D - Default 0",
    )
    param_iof_hydro20: int = Field(
        0,
        description="output 0: off; 1: on - water density [kg/m^3] {water_density}   3D - Default 0",
    )
    param_iof_hydro21: int = Field(
        0,
        description="output 0: off; 1: on - eddy diffusivity [m^2/s] {diffusivity}   3D - Default 0",
    )
    param_iof_hydro22: int = Field(
        0,
        description="output 0: off; 1: on - eddy viscosity [m^2/s] {viscosity}      3D - Default 0",
    )
    param_iof_hydro23: int = Field(
        0,
        description="output 0: off; 1: on - turbulent kinetic energy {TKE}   3D - Default 0",
    )
    param_iof_hydro24: int = Field(
        0,
        description="output 0: off; 1: on - turbulent mixing length [m] {mixing_length}   3D - Default 0",
    )
    param_iof_hydro25: int = Field(
        0,
        description="output 0: off; 1: on - horizontal vel vector [m/s] {hvel}   3D vector - Default 0",
    )
    param_iof_hydro26: int = Field(
        0,
        description="output 0: off; 1: on - horizontal vel vector defined @side [m/s] {hvel_side}   3D vector  - Default 0",
    )
    param_iof_hydro27: int = Field(
        0,
        description="output 0: off; 1: on - vertical vel. @elem [m/s] {wvel_elem}   3D vector  - Default 0",
    )
    param_iof_hydro28: int = Field(
        0,
        description="output 0: off; 1: on - T @prism centers [C] {temp_elem}   3D - Default 0",
    )
    param_iof_hydro29: int = Field(
        0,
        description="output 0: off; 1: on - S @prism centers [PSU] {salt_elem}   3D - Default 0",
    )
    param_iof_hydro30: int = Field(
        0,
        description="output 0: off; 1: on - Barotropic pressure gradient force vector (m.s-2) @side centers  {pressure_gradient}  2D vector  - Default 0",
    )
    wwminput_history_OUTSTYLE: str = Field(
        "NC", description="'output option - use 'NO' for no output"
    )
    param_nspool_sta: int = Field(
        30,
        description="needed if iout_sta/=0; mod(nhot_write,nspool_sta) must=0 defaults to 30",
    )
    ihot: int = Field(
        0,
        description="hotstart 0: off; 1: on - whether to expect hotstarts - Default - 0-",
    )
    wwminput_hotfile_DELTC: int = Field(
        0,
        description="hotfile time in UNITC (typically seconds) when hotfile should be written, defaults to 3600",
    )
    wwminput_station_DELTC: int = Field(
        3600,
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )
    wwminput_history_DELTC: int = Field(
        3600,
        description="Time step for output; if smaller than simulation time step, the latter is used (output every step for better 1D 2D spectra analysis)",
    )

    # validator example - ensure the following
    # Bottom friction.
    #           nchi=0: drag coefficients specified in drag.gr3; nchi=-1: Manning's
    #           formulation (even for 3D prisms) with n specified in manning.gr3.
    #           nchi=1: bottom roughness (in meters) specified in rough.gr3 (and in this case, negative
    #           or 0 depths in rough.gr3 indicate time-independent Cd, not roughness!).
    #           Cd is calculated using the log law, when dzb>=dzb_min; when dzb<dzb_min,
    #           Cd=Cdmax*exp[dzb_decay*(1-dzb/dzb_min)], where Cdmax=Cd(dzb=dzb_min),
    #           and dzb_decay (<=0) is a decay const specified below. We recommend dzb_decay=0
    #           and may remove this constant in the future.
    #           If iwbl/=0, nchi must =1.
    #   nchi = -1
    #   dzb_min = 0.5 !needed if nchi=1; min. bottom boundary layer thickness [m].
    #   dzb_decay = 0. !needed if nchi=1; a decay const. [-]. should =0
    #   hmin_man = 1.0 !needed if nchi=-1: min. depth in Manning's formulation [m]

    @model_validator(mode="after")
    def validate_bottom_friction(cls, v):
        if v.nchi == 0:
            if v.grid.drag is None:
                raise ValueError("drag.gr3 must be specified when nchi=0")
        elif v.nchi == -1:
            if v.grid.manning is None:
                raise ValueError("manning.gr3 must be specified when nchi=-1")
        elif v.nchi == 1:
            if v.grid.rough is None:
                raise ValueError("rough.gr3 must be specified when nchi=1")
        else:
            raise ValueError("nchi must be 0, -1, or 1")
        return v

    def __call__(self, runtime) -> str:
        # Copy grid files
        ret = self.model_dump()
        ret["_rnday"] = runtime.period.duration.total_seconds() / 86400
        ret["grid"] = self.grid.get(runtime.staging_dir)
        # TODO Still need to link up these maybe?
        ret.update(
            self.data.get(
                destdir=runtime.staging_dir, grid=self.grid, time=runtime.period
            )
        )
        return ret


class SCHISMConfig(BaseConfig):
    model_type: Literal["schism"] = Field(
        "schism", description="The model type for SCHISM."
    )
    grid: SCHISMGrid = Field(description="The model grid")
    data: Optional[SCHISMData] = Field(None, description="Model inputs")
    nml: Optional[NML] = Field(
        default_factory=lambda: NML(param=Param()), description="The namelist"
    )
    template: Optional[str] = Field(
        description="The path to the model template",
        default=SCHISM_TEMPLATE,
    )

    # add a validator that checks that nml.param.ihot is 1 if data.hotstart is not none
    @model_validator(mode="after")
    def check_hotstart(self):
        if (
            self.data is not None
            and hasattr(self.data, "hotstart")
            and self.data.hotstart is not None
        ):
            self.nml.param.opt.ihot = 1
        return self

    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to handle proper serialization of nested components."""
        from rompy.schism.grid import GR3Generator

        result = {}

        # Explicitly handle required fields
        result["model_type"] = self.model_type

        # Handle grid separately to process GR3Generator objects
        if self.grid is not None:
            grid_dict = {}
            for field_name in self.grid.model_fields:
                value = getattr(self.grid, field_name, None)

                # Special handling for GR3Generator objects
                if value is not None and isinstance(value, GR3Generator):
                    # For GR3Generator objects, extract just the value field
                    grid_dict[field_name] = value.value
                elif value is not None and not field_name.startswith("_"):
                    grid_dict[field_name] = value

            result["grid"] = grid_dict

        # Add optional fields that are not None
        if self.data is not None:
            result["data"] = self.data

        if self.nml is not None:
            result["nml"] = self.nml

        if self.template is not None:
            result["template"] = self.template

        return result

    # Enable arbitrary types and validation from instances in Pydantic v2
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True)

    # Add data visualization methods
    # Atmospheric (sflux) plotting
    plot_sflux_spatial = plot_sflux_spatial
    plot_sflux_timeseries = plot_sflux_timeseries

    # Boundary data plotting
    plot_boundary_points = plot_boundary_points
    plot_boundary_timeseries = plot_boundary_timeseries
    plot_boundary_profile = plot_boundary_profile

    # Tidal data plotting
    plot_tidal_boundaries = plot_tidal_boundaries
    plot_tidal_stations = plot_tidal_stations
    plot_tidal_rose = plot_tidal_rose
    plot_tidal_dataset = plot_tidal_dataset

    def __call__(self, runtime) -> str:
        logger = logging.getLogger(__name__)

        logger.info(f"Generating grid files using {type(self.grid).__name__}")
        self.grid.get(runtime.staging_dir)

        if self.data is not None:
            self.nml.update_data_sources(
                self.data.get(
                    destdir=runtime.staging_dir, grid=self.grid, time=runtime.period
                )
            )
        self.nml.update_times(period=runtime.period)

        self.nml.write_nml(runtime.staging_dir)

        return str(runtime.staging_dir)


class SchismCSIROMigrationConfig(SchismCSIROConfig):
    model_type: Literal["schismcsiromigration"] = Field(
        "schismcsiromigration", description="The model type for SCHISM."
    )
    template: Optional[str] = Field(
        description="The path to the model template",
        default=SCHISM_TEMPLATE,
    )

    def __call__(self, runtime) -> str:
        # Create translation dictionary
        config_dict = {
            "param": {
                "opt": {
                    "ihot": self.ihot,
                    "project": self.project,
                    "utc_start": self.utc_start,
                    "time_step": self.time_step,
                    "msc2": self.msc2,
                    "mdc2": self.mdc2,
                    "ihfskip": self.ihfskip,
                    "icou_elfe_wwm": self.icou_elfe_wwm,
                    "nstep_wwm": self.nstep_wwm,
                    "deltc": self.deltc,
                    "h1_bcc": self.h1_bcc,
                    "h2_bcc": self.h2_bcc,
                    "h_bcc1": self.h_bcc1,
                    "thetai": self.thetai,
                    "iwbl": self.iwbl,
                    "slam0": self.slam0,
                    "sfea0": self.sfea0,
                    "nchi": self.nchi,
                    "dzb_decayYN": self.dzb_decayYN,
                    "rlatitude": self.rlatitude,
                    "ic_elev": self.ic_elev,
                    "inv_atm_bnd": self.inv_atm_bnd,
                    "ibtrack_openbndYN": self.ibtrack_openbndYN,
                    "iwindoffYN": self.iwindoffYN,
                    "iwind_form": self.iwind_form,
                    "param_nhot": self.param_nhot,
                    "param_nhot_writeYN": self.param_nhot_writeYN,
                    "param_nhot_write": self.param_nhot_write,
                },
                "schout": {
                    "param_nspool_sta": self.param_nspool_sta,
                    "param_iof_hydro1": self.param_iof_hydro1,
                    "param_iof_hydro2": self.param_iof_hydro2,
                    "param_iof_hydro3": self.param_iof_hydro3,
                    "param_iof_hydro4": self.param_iof_hydro4,
                    "param_iof_hydro5": self.param_iof_hydro5,
                    "param_iof_hydro6": self.param_iof_hydro6,
                    "param_iof_hydro7": self.param_iof_hydro7,
                    "param_iof_hydro8": self.param_iof_hydro8,
                    "param_iof_hydro9": self.param_iof_hydro9,
                    "param_iof_hydro10": self.param_iof_hydro10,
                    "param_iof_hydro11": self.param_iof_hydro11,
                    "param_iof_hydro12": self.param_iof_hydro12,
                    "param_iof_hydro13": self.param_iof_hydro13,
                    "param_iof_hydro14": self.param_iof_hydro14,
                    "param_iof_hydro15": self.param_iof_hydro15,
                    "param_iof_hydro16": self.param_iof_hydro16,
                    "param_iof_hydro17": self.param_iof_hydro17,
                    "param_iof_hydro18": self.param_iof_hydro18,
                    "param_iof_hydro19": self.param_iof_hydro19,
                    "param_iof_hydro20": self.param_iof_hydro20,
                    "param_iof_hydro21": self.param_iof_hydro21,
                    "param_iof_hydro22": self.param_iof_hydro22,
                    "param_iof_hydro23": self.param_iof_hydro23,
                    "param_iof_hydro24": self.param_iof_hydro24,
                    "param_iof_hydro25": self.param_iof_hydro25,
                    "param_iof_hydro26": self.param_iof_hydro26,
                    "param_iof_hydro27": self.param_iof_hydro27,
                    "param_iof_hydro28": self.param_iof_hydro28,
                    "param_iof_hydro29": self.param_iof_hydro29,
                    "param_iof_hydro30": self.param_iof_hydro30,
                },
            },
            "wwminput_WW3": {
                "proc": {
                    "PROCNAME": self.project,
                },
                "history": {
                    "HS": self.HS,
                    "TM01": self.TM01,
                    "TM02": self.TM02,
                    "DM": self.DM,
                    "DSPR": self.DSPR,
                    "TPP": self.TPP,
                    "TPPD": self.TPPD,
                    "CPP": self.CPP,
                    "WNPP": self.WNPP,
                    "CGPP": self.CGPP,
                    "KPP": self.KPP,
                    "LPP": self.LPP,
                    "PEAKD": self.PEAKD,
                    "DPEAK": self.DPEAK,
                    "PEAKDSPR": self.PEAKDSPR,
                    "UBOT": self.UBOT,
                    "ORBITAL": self.ORBITAL,
                    "DEP": self.wwminput_history_DEP,
                    "TAUW": self.wwminput_history_TAUW,
                    "TAUHF": self.wwminput_history_TAUHF,
                    "TAUTOT": self.wwminput_history_TAUTOT,
                    "STOKESSURFX": self.wwminput_history_STOKESSURFX,
                    "STOKESSURFY": self.wwminput_history_STOKESSURFY,
                    "STOKESBAROX": self.wwminput_history_STOKESBAROX,
                    "STOKESBAROY": self.wwminput_history_STOKESBAROY,
                    "DELTC": self.wwminput_history_DELTC,
                    "OUTSTYLE": self.wwminput_history_OUTSTYLE,
                },
                "station": {
                    "iouts": self.iouts,
                    "nouts": self.nouts,
                    "xouts": self.xouts,
                    "youts": self.youts,
                    "lsp2d": self.lsp2d,
                    "ac": self.ac,
                    "drampwafo": self.drampwafo,
                    "nadv": self.nadv,
                    "drampwind": self.drampwind,
                    "dramp": self.dramp,
                    "DEP": self.wwminput_station_DEP,
                    "TAUW": self.wwminput_station_TAUW,
                    "TAUHF": self.wwminput_station_TAUHF,
                    "TAUTOT": self.wwminput_station_TAUTOT,
                    "STOKESSURFX": self.wwminput_station_STOKESSURFX,
                    "STOKESSURFY": self.wwminput_station_STOKESSURFY,
                    "STOKESBAROX": self.wwminput_station_STOKESBAROX,
                    "STOKESBAROY": self.wwminput_station_STOKESBAROY,
                    "OUTSTYLE": self.wwminput_station_OUTSTYLE,
                    "DELTC": self.wwminput_station_DELTC,
                },
                "hotfile": {
                    "LHOTF": self.wwminput_LHOTF,
                    "hotfile_DELTC": self.wwminput_hotfile_DELTC,
                },
            },
        }
        schism_config = SCHISMConfig(
            grid=self.grid, data=self.data, nml=NML(**config_dict)
        )
        schism_config.__call__(runtime)
