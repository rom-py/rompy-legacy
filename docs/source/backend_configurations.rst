=============================
Backend Configurations
=============================

ROMPY provides a type-safe backend configuration system using Pydantic models. This system enables precise control over model execution environments while providing validation, IDE support, and clear error messages.

Overview
--------

Backend configurations are Pydantic models that define how and where models should be executed. These configurations provide:

* **Type Safety**: Full validation at configuration time
* **IDE Support**: Autocompletion and inline documentation
* **Clear Validation**: Descriptive error messages for invalid configurations
* **Serialization**: Easy saving/loading of configurations as YAML/JSON
* **Schema Generation**: Self-documenting configuration structure

Configuration Types
-------------------

All backend configurations inherit from ``BaseBackendConfig`` and provide specific parameters for their execution environment.

BaseBackendConfig
^^^^^^^^^^^^^^^^^

The base class for all backend configurations provides common fields:

.. code-block:: python

    from rompy.backends import BaseBackendConfig

    class BaseBackendConfig(BaseModel):
        timeout: int = Field(3600, ge=60, le=86400)  # 1 minute to 24 hours
        env_vars: Dict[str, str] = Field(default_factory=dict)
        working_dir: Optional[Path] = None

**Common Fields:**

* ``timeout``: Maximum execution time in seconds (60-86400)
* ``env_vars``: Environment variables to set during execution
* ``working_dir``: Working directory for execution (optional)

LocalConfig
^^^^^^^^^^^

Configuration for local system execution:

.. code-block:: python

    from rompy.backends import LocalConfig

    # Basic local execution
    config = LocalConfig(
        timeout=3600,
        command="python run_model.py"
    )

    # Advanced local execution
    config = LocalConfig(
        timeout=7200,
        env_vars={
            "OMP_NUM_THREADS": "4",
            "PYTHONPATH": "/custom/path"
        },
        command="python run_model.py --verbose",
        shell=True,
        capture_output=True
    )

**LocalConfig Fields:**

* ``command``: Shell command to execute (optional)
* ``shell``: Whether to execute through shell (default: True)
* ``capture_output``: Whether to capture stdout/stderr (default: True)

DockerConfig
^^^^^^^^^^^^

Configuration for Docker container execution:

.. code-block:: python

    from rompy.backends import DockerConfig

    # Using pre-built image
    config = DockerConfig(
        image="swan:latest",
        cpu=4,
        memory="2g",
        timeout=7200,
        env_vars={"SWAN_THREADS": "4"},
        volumes=["/data:/app/data:ro"]
    )

    # Building from Dockerfile
    config = DockerConfig(
        dockerfile="./docker/Dockerfile",
        cpu=2,
        build_args={"BASE_IMAGE": "ubuntu:20.04"},
        executable="/usr/local/bin/swan"
    )

**DockerConfig Fields:**

* ``image``: Docker image to use (mutually exclusive with dockerfile)
* ``dockerfile``: Path to Dockerfile to build (mutually exclusive with image)
* ``cpu``: Number of CPU cores to allocate (1-128)
* ``memory``: Memory limit (e.g., "2g", "512m")
* ``executable``: Path to executable inside container
* ``mpiexec``: MPI execution command for parallel runs
* ``volumes``: Volume mounts in "host:container" format
* ``build_args``: Arguments for Docker build process
* ``user``: User to run as inside container (default: "root")
* ``remove_container``: Whether to remove container after execution (default: True)

Using Backend Configurations
-----------------------------

With ModelRun
^^^^^^^^^^^^^

Backend configurations are used directly with the ``ModelRun.run()`` method:

.. code-block:: python

    from rompy.model import ModelRun
    from rompy.backends import LocalConfig, DockerConfig

    # Load your model configuration
    model_run = ModelRun.from_file("model_config.yml")

    # Execute locally
    local_config = LocalConfig(
        timeout=3600,
        command="python run_simulation.py"
    )
    success = model_run.run(backend=local_config)

    # Execute in Docker
    docker_config = DockerConfig(
        image="rompy/swan:latest",
        cpu=4,
        memory="4g",
        volumes=["/data:/app/data"]
    )
    success = model_run.run(backend=docker_config)

Configuration Files
^^^^^^^^^^^^^^^^^^^

Backend configurations can be saved and loaded as YAML or JSON files:

**local_backend.yml:**

.. code-block:: yaml

    type: local
    timeout: 7200
    env_vars:
      OMP_NUM_THREADS: "4"
      MODEL_DEBUG: "true"
    command: "python run_model.py --verbose"
    shell: true
    capture_output: true

**docker_backend.yml:**

.. code-block:: yaml

    type: docker
    image: "swan:latest"
    cpu: 4
    memory: "2g"
    timeout: 10800
    env_vars:
      SWAN_THREADS: "4"
    volumes:
      - "/data/input:/app/input:ro"
      - "/data/output:/app/output:rw"
    executable: "/usr/local/bin/swan"

**Loading configurations:**

.. code-block:: python

    import yaml
    from rompy.backends import LocalConfig, DockerConfig

    # Load local configuration
    with open("local_backend.yml") as f:
        config_data = yaml.safe_load(f)
        backend_type = config_data.pop("type")
        local_config = LocalConfig(**config_data)

    # Load Docker configuration
    with open("docker_backend.yml") as f:
        config_data = yaml.safe_load(f)
        backend_type = config_data.pop("type")
        docker_config = DockerConfig(**config_data)

Command Line Interface
----------------------

The CLI supports backend configurations through configuration files:

.. code-block:: bash

    # Run with backend configuration file
    rompy run model_config.yml --backend-config local_backend.yml

    # Run with Docker configuration
    rompy run model_config.yml --backend-config docker_backend.yml

**CLI Backend Management:**

.. code-block:: bash

    # Create configuration template
    rompy backends create --backend-type local --output local_config.yml
    rompy backends create --backend-type docker --with-examples --output docker_config.yml

    # Validate configuration
    rompy backends validate local_config.yml --backend-type local

    # Generate JSON schema
    rompy backends schema --backend-type docker --format json

    # List available backends
    rompy backends list

Validation and Error Handling
------------------------------

Type Safety
^^^^^^^^^^^

Pydantic provides comprehensive validation with clear error messages:

.. code-block:: python

    from rompy.backends import LocalConfig, DockerConfig
    from pydantic import ValidationError

    try:
        # Invalid timeout (too short)
        config = LocalConfig(timeout=30)
    except ValidationError as e:
        print(e)
        # ValidationError: timeout - Input should be greater than or equal to 60

    try:
        # Missing required image/dockerfile
        config = DockerConfig()
    except ValidationError as e:
        print(e)
        # ValidationError: Either 'image' or 'dockerfile' must be provided

Field Validation
^^^^^^^^^^^^^^^^

Each configuration class validates its fields according to the execution environment requirements:

**LocalConfig validation:**

* Working directory must exist if specified
* Environment variables must be string key-value pairs
* Timeout must be between 60 and 86400 seconds

**DockerConfig validation:**

* Either ``image`` or ``dockerfile`` must be provided (not both)
* CPU count must be between 1 and 128
* Memory format must match pattern (e.g., "2g", "512m")
* Volume mounts must use "host:container" format with existing host paths
* Docker image names must follow valid naming conventions

Schema Generation
-----------------

Generate configuration schemas for documentation or validation:

.. code-block:: python

    from rompy.backends import LocalConfig, DockerConfig
    import json

    # Generate JSON schema
    local_schema = LocalConfig.model_json_schema()
    docker_schema = DockerConfig.model_json_schema()

    # Pretty print schema
    print(json.dumps(local_schema, indent=2))

**Example schema output:**

.. code-block:: json

    {
      "type": "object",
      "properties": {
        "timeout": {
          "type": "integer",
          "minimum": 60,
          "maximum": 86400,
          "description": "Maximum execution time in seconds"
        },
        "env_vars": {
          "type": "object",
          "additionalProperties": {"type": "string"},
          "description": "Environment variables to set during execution"
        }
      }
    }

Best Practices
--------------

Configuration Organization
^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Environment-Specific Configs**: Create separate configuration files for different environments:

   .. code-block:: text

       configs/
       ├── backends/
       │   ├── local_dev.yml      # Development environment
       │   ├── local_hpc.yml      # HPC cluster local execution
       │   ├── docker_dev.yml     # Development Docker setup
       │   └── docker_prod.yml    # Production Docker setup

2. **Template Configurations**: Use the CLI to generate templates:

   .. code-block:: bash

       rompy backends create --backend-type docker --with-examples > docker_template.yml

3. **Validation Before Execution**: Always validate configurations:

   .. code-block:: bash

       rompy backends validate my_config.yml --backend-type local

Security Considerations
^^^^^^^^^^^^^^^^^^^^^^^

1. **Environment Variables**: Avoid storing sensitive data in configuration files:

   .. code-block:: python

       # Good: Reference environment variables
       config = LocalConfig(
           env_vars={"API_KEY": os.environ["API_KEY"]}
       )

       # Avoid: Hardcode sensitive data
       config = LocalConfig(
           env_vars={"API_KEY": "secret-key-123"}
       )

2. **Docker Security**: Use appropriate user permissions and volume mounts:

   .. code-block:: python

       config = DockerConfig(
           image="myapp:latest",
           user="appuser",  # Don't run as root
           volumes=["/data:/app/data:ro"]  # Read-only when possible
       )

Performance Optimization
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Resource Allocation**: Set appropriate limits for Docker containers:

   .. code-block:: python

       config = DockerConfig(
           image="heavy-computation:latest",
           cpu=8,
           memory="16g"
       )

2. **Timeout Management**: Set realistic timeouts based on model complexity:

   .. code-block:: python

       # Short timeout for quick models
       quick_config = LocalConfig(timeout=600)  # 10 minutes

       # Longer timeout for complex simulations
       complex_config = DockerConfig(
           image="long-simulation:latest",
           timeout=21600  # 6 hours
       )

Creating Custom Backend Configurations
---------------------------------------

You can extend the system with custom backend configurations:

.. code-block:: python

    from rompy.backends import BaseBackendConfig
    from pydantic import Field
    from typing import Optional

    class HPCConfig(BaseBackendConfig):
        """Configuration for HPC cluster execution."""

        queue: str = Field(..., description="SLURM queue name")
        nodes: int = Field(1, ge=1, le=100, description="Number of nodes")
        partition: str = Field("compute", description="Cluster partition")
        account: Optional[str] = Field(None, description="Account for billing")

        def get_backend_class(self):
            from mypackage.backends import HPCRunBackend
            return HPCRunBackend

    # Register with entry points in pyproject.toml
    # [project.entry-points."rompy.run"]
    # hpc = "mypackage.backends:HPCRunBackend"

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**"Either 'image' or 'dockerfile' must be provided"**

   Ensure DockerConfig has exactly one of image or dockerfile:

   .. code-block:: python

       # Correct
       config = DockerConfig(image="myapp:latest")
       # OR
       config = DockerConfig(dockerfile="./Dockerfile")

       # Incorrect - missing both
       config = DockerConfig()
       # Incorrect - both provided
       config = DockerConfig(image="myapp:latest", dockerfile="./Dockerfile")

**"Working directory does not exist"**

   Ensure the working directory exists before creating the configuration:

   .. code-block:: python

       from pathlib import Path

       work_dir = Path("/path/to/workdir")
       work_dir.mkdir(parents=True, exist_ok=True)

       config = LocalConfig(working_dir=work_dir)

**"Volume mount validation failed"**

   Ensure host paths exist and use correct format:

   .. code-block:: python

       # Correct format with existing host path
       config = DockerConfig(
           image="myapp:latest",
           volumes=["/existing/host/path:/container/path"]
       )

Debugging Configuration Issues
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Use schema validation**:

   .. code-block:: python

       from rompy.backends import DockerConfig

       try:
           config = DockerConfig(**config_data)
       except ValidationError as e:
           print("Validation errors:")
           for error in e.errors():
               print(f"  {error['loc']}: {error['msg']}")

2. **Check configuration serialization**:

   .. code-block:: python

       config = LocalConfig(timeout=3600)
       print(config.model_dump_json(indent=2))

3. **Validate against schema**:

   .. code-block:: bash

       rompy backends validate my_config.yml --backend-type docker

The Pydantic backend configuration system provides a robust, type-safe foundation for model execution while maintaining flexibility and extensibility for future backend types.
