from pydantic import Field

from rompy.schism.namelists.basemodel import NamelistBaseModel


class MARCO(NamelistBaseModel):
    nsub: int = Field(1)
    iKe: int = Field(0)
    Ke0: float = Field(0.26)
    KeC: float = Field(0.017)
    KeS: float = Field(0.07)
    KeSalt: float = Field(-0.02)
    tss2c: float = Field(6.0)


class CORE(NamelistBaseModel):
    GPM: list = Field([2.5, 2.8, 3.5])
    TGP: list = Field([15.0, 22.0, 27.0])
    KTGP: list = Field([0.005, 0.004, 0.003, 0.008, 0.006, 0.004])
    MTR: list = Field([0.0, 0.0, 0.0])
    MTB: list = Field([0.01, 0.02, 0.03])
    TMT: list = Field([20.0, 20.0, 20.0])
    KTMT: list = Field([0.0322, 0.0322, 0.0322])


class example(NamelistBaseModel):
    """

    This file was auto generated from a schism namelist file on 2023-11-15.
    The full contents of the namelist file are shown below providing
    associated documentation for the objects:

    &MARCO
    !-----------------------------------------------------------------------
    !number of subcycles in ICM kinetics
    !-----------------------------------------------------------------------
    nsub=1

    !-----------------------------------------------------------------------
    !options of formulations for computing light attenuation coefficients
    !  iKe=0: Ke=Ke0+KeC*Chl+KeS*(tss2c*POC);  TSS=tss2c*POC
    !  iKe=1: Ke=Ke0+KeC*Chl+KeS*TSS;        from SED3D or saved total_sed_conc
    !  iKe=2: Ke=Ke0+KeC*Chl+KeSalt*Salt;    CDOM effect related to Salinity
    !-----------------------------------------------------------------------
    iKe = 0

    Ke0    = 0.26            !backgroud light extinction coefficient (1/m)
    KeC    = 0.017           !Light attenu. due to chlorophyll
    KeS    = 0.07            !Light attenu. due to TSS
    KeSalt = -0.02           !Light attenu. due to CDOM (related to salinity)
    tss2c  = 6.0             !TSS to carbon ratio



    &CORE
    !-----------------------------------------------------------------------
    !ICM parameters for water column; PB: phytoplankton
    !CBP module parameters are included: (SRPOC,SRPON,SRPOP,PIP)
    !-----------------------------------------------------------------------
    !phytoplankton growth
    GPM  = 2.5    2.8    3.5        !PB growth rates (day-1)
    TGP  = 15.0   22.0   27.0        !optimal temp. for PB growth (oC)
    KTGP = 0.005  0.004  0.003    0.008  0.006  0.004  !temp. dependence for PB growth; dim(PB=1:3,1:2) (oC-2)

    !phytoplankton photorespiration & metabolism
    MTR  = 0.0    0.0    0.0          !PB photorespiration coefficient (0<MTR<1)
    MTB  = 0.01   0.02   0.03         !PB metabolism rates (day-1)
    TMT  = 20.0   20.0   20.0         !reference temp. for PB metabolism (oC)
    KTMT = 0.0322 0.0322 0.0322       !temp. dependence for PB metabolism (oC-1)

    """

    MARCO: MARCO = MARCO()
    CORE: CORE = CORE()
