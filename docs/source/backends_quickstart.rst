========================
Backend Quickstart Guide
========================

This guide helps you quickly get started with ROMPY's backend system for executing models. Learn the essential configurations and commands to run your models locally or in Docker containers.

.. note::
   For comprehensive backend documentation, see :doc:`backends` and :doc:`backend_configurations`.

Quick Start
-----------

ROMPY uses backend configurations to control how and where models are executed. Here's what you need to know:

**1. Create a configuration file**

.. code-block:: yaml

    # my_backend.yml
    backend_type: local
    timeout: 3600
    command: "python run_model.py"
    env_vars:
      OMP_NUM_THREADS: "4"

**2. Validate your configuration**

.. code-block:: bash

    rompy backends validate my_backend.yml

**3. Run your model**

.. code-block:: python

    from rompy.model import ModelRun
    from rompy.backends import LocalConfig

    # Load and run
    model = ModelRun.from_file("model_config.yml")
    config = LocalConfig.from_file("my_backend.yml")
    success = model.run(backend=config)

Local Execution
---------------

Execute models directly on your local system.

Basic Local Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # local_basic.yml
    backend_type: local
    timeout: 3600
    command: "python run_simulation.py"

.. code-block:: python

    from rompy.backends import LocalConfig

    # Create configuration
    config = LocalConfig(
        timeout=3600,
        command="python run_simulation.py"
    )

    # Run model
    success = model.run(backend=config)

Advanced Local Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # local_advanced.yml
    backend_type: local
    timeout: 7200  # 2 hours
    command: "python run_simulation.py --verbose"
    shell: true
    capture_output: true
    env_vars:
      OMP_NUM_THREADS: "8"
      PYTHONPATH: "/custom/path"
      MODEL_DEBUG: "true"
      DATA_DIR: "/data"

.. code-block:: python

    from rompy.backends import LocalConfig

    # Advanced configuration
    config = LocalConfig(
        timeout=7200,
        command="python run_simulation.py --verbose",
        env_vars={
            "OMP_NUM_THREADS": "8",
            "PYTHONPATH": "/custom/path",
            "MODEL_DEBUG": "true",
            "DATA_DIR": "/data"
        },
        shell=True,
        capture_output=True
    )

Local Configuration Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Development/Testing:**

.. code-block:: yaml

    backend_type: local
    timeout: 1800  # 30 minutes
    command: "python -m pytest tests/ && python run_model.py"
    env_vars:
      ENV: "development"
      LOG_LEVEL: "DEBUG"

**Production:**

.. code-block:: yaml

    backend_type: local
    timeout: 14400  # 4 hours
    env_vars:
      ENV: "production"
      LOG_LEVEL: "INFO"
      OMP_NUM_THREADS: "16"
    # No command specified - uses model's built-in run method

Docker Execution
----------------

Execute models inside Docker containers for reproducible, isolated environments.

Basic Docker Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # docker_basic.yml
    backend_type: docker
    image: "swan:latest"
    timeout: 3600
    cpu: 4
    memory: "2g"

.. code-block:: python

    from rompy.backends import DockerConfig

    # Create configuration
    config = DockerConfig(
        image="swan:latest",
        timeout=3600,
        cpu=4,
        memory="2g"
    )

    # Run model
    success = model.run(backend=config)

Advanced Docker Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # docker_advanced.yml
    backend_type: docker
    image: "swan:latest"
    timeout: 10800  # 3 hours
    cpu: 8
    memory: "4g"
    mpiexec: "mpirun -np 8"
    user: "modeluser"
    volumes:
      - "/data/input:/app/input:ro"
      - "/data/output:/app/output:rw"
      - "/tmp:/tmp:rw"
    env_vars:
      MODEL_THREADS: "8"
      DATA_DIR: "/app/input"
      RESULTS_DIR: "/app/output"

Building from Dockerfile
^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: yaml

    # docker_build.yml
    backend_type: docker
    dockerfile: "./docker/Dockerfile"
    build_context: "./docker"
    build_args:
      MODEL_VERSION: "2.1.0"
      BASE_IMAGE: "ubuntu:20.04"
    timeout: 7200
    cpu: 4
    memory: "2g"
    volumes:
      - "/data:/app/data:rw"

.. code-block:: python

    from rompy.backends import DockerConfig

    # Build and run configuration
    config = DockerConfig(
        dockerfile="./docker/Dockerfile",
        build_context="./docker",
        build_args={
            "MODEL_VERSION": "2.1.0",
            "BASE_IMAGE": "ubuntu:20.04"
        },
        timeout=7200,
        cpu=4,
        memory="2g",
        volumes=["/data:/app/data:rw"]
    )

Docker Configuration Examples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**High-Performance Computing:**

.. code-block:: yaml

    backend_type: docker
    image: "hpc-swan:latest"
    timeout: 86400  # 24 hours
    cpu: 32
    memory: "64g"
    mpiexec: "mpirun -np 32"
    user: "hpcuser"
    volumes:
      - "/ocean_data:/data:ro"
      - "/scratch:/scratch:rw"
      - "/results:/results:rw"
    env_vars:
      OMP_NUM_THREADS: "1"
      MODEL_PRECISION: "double"

**Development Environment:**

.. code-block:: yaml

    backend_type: docker
    dockerfile: "./Dockerfile.dev"
    build_context: "."
    timeout: 3600
    cpu: 2
    memory: "2g"
    remove_container: false  # Keep for debugging
    volumes:
      - "./src:/app/src:rw"
      - "./tests:/app/tests:rw"
    env_vars:
      ENV: "development"
      LOG_LEVEL: "DEBUG"

CLI Usage
---------

The CLI provides commands for managing and validating backend configurations.

### Validation Commands

.. code-block:: bash

    # Validate configuration file
    rompy backends validate local_backend.yml

    # Validate with explicit backend type
    rompy backends validate --backend-type local local_backend.yml

    # Validate Docker configuration
    rompy backends validate --backend-type docker docker_backend.yml

Schema and Documentation
^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    # Show available backends
    rompy backends list

    # Show schema for local backend
    rompy backends schema --backend-type local

    # Show schema for Docker backend
    rompy backends schema --backend-type docker

    # Create configuration template
    rompy backends create --backend-type local --output my_local_config.yml

Running Models
^^^^^^^^^^^^^^

.. code-block:: bash

    # Run model with backend configuration
    rompy run model_config.yml --backend-config local_backend.yml

    # Run pipeline with configuration
    rompy pipeline --config pipeline_config.yml

Common Patterns
---------------

Environment-Specific Configurations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Development:**

.. code-block:: yaml

    backend_type: local
    timeout: 1800
    command: "python run_model.py --debug"
    env_vars:
      ENV: "development"
      LOG_LEVEL: "DEBUG"
      PYTHONUNBUFFERED: "1"

**Testing:**

.. code-block:: yaml

    backend_type: local
    timeout: 900  # 15 minutes
    command: "python -m pytest tests/ && python run_model.py --test-mode"
    env_vars:
      ENV: "testing"
      SKIP_HEAVY_TESTS: "true"

**Production:**

.. code-block:: yaml

    backend_type: docker
    image: "mymodel:production"
    timeout: 14400  # 4 hours
    cpu: 16
    memory: "32g"
    mpiexec: "mpirun -np 16"
    volumes:
      - "/data/production:/app/data:ro"
      - "/results/production:/app/results:rw"
    env_vars:
      ENV: "production"
      LOG_LEVEL: "INFO"

Resource Allocation
^^^^^^^^^^^^^^^^^^^

**Small Models:**

.. code-block:: yaml

    backend_type: local
    timeout: 3600
    env_vars:
      OMP_NUM_THREADS: "4"

**Medium Models:**

.. code-block:: yaml

    backend_type: docker
    image: "swan:latest"
    timeout: 7200
    cpu: 8
    memory: "8g"
    env_vars:
      MODEL_THREADS: "8"

**Large Models:**

.. code-block:: yaml

    backend_type: docker
    image: "swan:hpc"
    timeout: 86400
    cpu: 32
    memory: "64g"
    mpiexec: "mpirun -np 32"
    env_vars:
      OMP_NUM_THREADS: "1"
      MODEL_PRECISION: "double"

Best Practices
--------------

Configuration Management
^^^^^^^^^^^^^^^^^^^^^^^^^

1. **Use version control** for configuration files
2. **Validate configurations** before running
3. **Use environment variables** for sensitive or environment-specific values
4. **Document your configurations** with comments

.. code-block:: yaml

    # Production SWAN model configuration
    backend_type: docker
    image: "swan:2.1.0"  # Pin specific version
    timeout: 14400  # 4 hours for typical runs
    cpu: 16  # Match your server capabilities
    memory: "32g"  # Ensure sufficient memory
    volumes:
      - "/data/bathymetry:/app/bathy:ro"  # Read-only bathymetry
      - "/data/forcing:/app/forcing:ro"   # Read-only forcing data
      - "/results:/app/results:rw"        # Write results here
    env_vars:
      MODEL_THREADS: "16"  # Match CPU count
      OUTPUT_FREQUENCY: "3600"  # Hourly output

Resource Planning
^^^^^^^^^^^^^^^^^

1. **Start small** and scale up based on actual needs
2. **Monitor resource usage** to optimize configurations
3. **Use appropriate timeouts** based on model complexity
4. **Consider memory requirements** for large datasets

Error Handling
^^^^^^^^^^^^^^

1. **Validate configurations** before production use
2. **Use appropriate timeouts** to prevent hanging
3. **Check log outputs** for troubleshooting
4. **Test configurations** with smaller datasets first

Troubleshooting
---------------

Common Issues
^^^^^^^^^^^^^

**Configuration Validation Errors:**

.. code-block:: bash

    # Check syntax
    rompy backends validate my_config.yml

    # Check schema
    rompy backends schema --backend-type local

**Docker Issues:**

.. code-block:: bash

    # Check if image exists
    docker images | grep swan

    # Test Docker configuration
    rompy backends validate --backend-type docker docker_config.yml

**Timeout Issues:**

.. code-block:: yaml

    # Increase timeout for long-running models
    backend_type: local
    timeout: 21600  # 6 hours

**Memory Issues:**

.. code-block:: yaml

    # Increase memory for Docker
    backend_type: docker
    image: "swan:latest"
    memory: "8g"  # or higher

Getting Help
^^^^^^^^^^^^

.. code-block:: bash

    # General help
    rompy backends --help

    # Command-specific help
    rompy backends validate --help
    rompy backends schema --help

    # Show configuration examples
    rompy backends create --backend-type local --with-examples

Complete Tutorial
-----------------

A complete hands-on tutorial is available that demonstrates all the concepts in this quickstart:

.. code-block:: bash

    # Run the interactive tutorial
    cd rompy/examples
    python quickstart_backend_example.py

The tutorial covers:

1. **Basic Local Execution** - Simple LocalConfig usage
2. **Advanced Local Configuration** - Environment variables and settings
3. **Basic Docker Execution** - Running models in containers
4. **Advanced Docker Configuration** - Resource allocation and volumes
5. **Configuration Validation** - Error handling and validation
6. **Dictionary-based Configuration** - Loading from YAML/JSON
7. **Complete Workflow** - Model execution with postprocessing

Each example builds on the previous one to show increasingly sophisticated usage patterns.

Testing and Validation
-----------------------

To validate that your backend system is working correctly, you can run the test suite:

.. code-block:: bash

    # Run backend system tests
    cd rompy/tests
    python test_backend_examples.py

This validates:
- Backend configuration objects
- Model integration
- Example files
- Configuration files
- Validation scripts

Next Steps
----------

- Read the comprehensive :doc:`backends` documentation
- Explore :doc:`backend_configurations` for detailed configuration options
- Try the :doc:`demo` for hands-on examples
- Check out the :doc:`cli` documentation for more command options
- Run the complete example in ``examples/quickstart_backend_example.py``

This quickstart covers the essential patterns for backend configuration. For advanced use cases, custom backends, or integration with HPC systems, refer to the full backend documentation.
