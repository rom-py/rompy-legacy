from rompy.swan import SwanModel

# Create a SwanModel object
swan = SwanModel()

# dumpy to yaml
with open("full_example.yaml", "w") as f:
    f.write(swan.yaml())
