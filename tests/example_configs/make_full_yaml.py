from rompy.swan import SwanModel


# Import test utilities
from test_utils.logging import get_test_logger

# Initialize logger
logger = get_test_logger(__name__)

# Create a SwanModel object
swan = SwanModel()

# dumpy to yaml
with open("full_example.yaml", "w") as f:
    f.write(swan.yaml())
