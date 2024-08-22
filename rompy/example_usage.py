# Register datatypes (this would typically be done in your main application setup)
from dummy_config import SchismConfig, SwanConfig, XbeachConfig
from dummy_data import SchismBoundary, SchismOcean, SchismWind, SwanBoundary, SwanWind

from rompy.registry import Registry

# Usage example
kwargs = {
    "config": {
        "config_type": "swan",
        "data": [
            {"data_type": "boundary", "boundary_field": "some_boundary"},
            {"data_type": "wind", "wind_field": 5.5},
        ],
    }
}
model_run = Registry.create_model_run()(**kwargs)
print(model_run.config)

kwargs = {
    "config": {
        "config_type": "schism",
        "data": [
            {"data_type": "boundary", "boundary_field": "some_boundary"},
            {"data_type": "wind", "wind_field": 6.7},
            {"data_type": "ocean", "ocean_field": {"temperature": 20, "salinity": 35}},
        ],
    }
}
model_run = Registry.create_model_run()(**kwargs)
print(model_run.config)

# Try with an unregistered data type
try:
    kwargs = {
        "config": {
            "config_type": "swan",
            "data": [
                {"data_type": "unregistered", "some_field": "some_value"},
            ],
        }
    }
    model_run = Registry.create_model_run()(**kwargs)
except ValueError as e:
    print(f"Validation error: {e}")

# You can still validate available config types and datatypes:
print(Registry.get_config_types().keys())
print(Registry.get_datatypes_for_config("swan"))
print(Registry.get_datatypes_for_config("schism"))
