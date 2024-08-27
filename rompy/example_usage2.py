import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from dummy_config import SchismConfig, SwanConfig, XbeachConfig
from dummy_model import ModelRun

from rompy.registry import Registry

# Usage example
kwargs = {
    "output_dir": "./simulations2",
    "config": {
        "config_type": "swan",
        "data": [
            {"data_type": "boundary", "boundary_field": "some_boundary"},
            {"data_type": "wind", "wind_field": 5.5},
        ],
    },
}

logger.debug("Creating ModelRun instance before imports")
model_run = ModelRun(**kwargs)
logger.debug(f"ModelRun instance created: {model_run}")
logger.debug(f"Config: {model_run.config}")
logger.debug(f"Config data: {model_run.config.data}")
for item in model_run.config.data:
    logger.debug(f"Data item: {item}, type: {type(item)}")
print(model_run.config)

from dummy_data import (SchismBoundary, SchismOcean, SchismWind, SwanBoundary,
                        SwanWind)

logger.debug("Creating ModelRun instance after imports")
model_run = ModelRun(**kwargs)
logger.debug(f"ModelRun instance created: {model_run}")
logger.debug(f"Config: {model_run.config}")
logger.debug(f"Config data: {model_run.config.data}")
for item in model_run.config.data:
    logger.debug(f"Data item: {item}, type: {type(item)}")
print(model_run.config)
