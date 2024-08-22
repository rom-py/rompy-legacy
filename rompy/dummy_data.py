from pydantic import BaseModel

from rompy.registry import Registry


@Registry.register_datatype("swan", "boundary")
class SwanBoundary(BaseModel):
    boundary_field: str


@Registry.register_datatype("swan", "wind")
class SwanWind(BaseModel):
    wind_field: float


@Registry.register_datatype("schism", "boundary")
class SchismBoundary(BaseModel):
    boundary_field: str


@Registry.register_datatype("schism", "wind")
class SchismWind(BaseModel):
    wind_field: float


@Registry.register_datatype("schism", "ocean")
class SchismOcean(BaseModel):
    ocean_field: dict
