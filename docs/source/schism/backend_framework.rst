===============================
SCHISM Backend Framework
===============================

Overview
========

The SCHISM backend framework provides a unified approach to executing SCHISM simulations using the ROMPY backend system. This framework integrates SCHISM model execution with Docker containers, providing consistent, reproducible, and scalable model runs.

Key Features
============

* **Docker Integration** - Automatic Docker image building and container management
* **Backend Framework** - Uses the unified ROMPY backend system
* **Boundary Conditions Testing** - Comprehensive testing suite for boundary condition examples
* **Python-based Execution** - Python scripts replace bash scripts for better maintainability
* **Automatic Resource Management** - Proper cleanup and resource allocation

Architecture
============

The backend framework consists of several key components:

Backend Configuration
---------------------

The framework uses ``DockerConfig`` from the ROMPY backend system to configure Docker execution:

.. code-block:: python

    from rompy.backends import DockerConfig
    from pathlib import Path

    # Docker configuration with automatic image building
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),
        build_context=Path("docker/schism"),
        timeout=3600,
        cpu=8,
        memory="4g",
        executable="bash -c 'cd /tmp/schism && mpirun --allow-run-as-root -n 8 schism_v5.13.0 4'",
        volumes=[f"{schism_dir}:/tmp/schism:rw"],
        env_vars={
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"
        },
        remove_container=True,
        user="root"
    )

Model Execution
---------------

The framework integrates with ``ModelRun`` for complete model execution:

.. code-block:: python

    from rompy.model import ModelRun
    from rompy.core.time import TimeRange
    from datetime import datetime

    # Create model run
    model_run = ModelRun(
        run_id="schism_example",
        period=TimeRange(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 2),
            interval="1H"
        ),
        output_dir="/path/to/output",
        delete_existing=True
    )

    # Generate configuration files
    model_run.generate()

    # Execute with backend
    success = model_run.run(backend=docker_config)

Boundary Conditions Testing
============================

The framework includes a comprehensive testing system for SCHISM boundary conditions examples.

Test Runner
-----------

The ``run_boundary_conditions_examples.py`` script provides automated testing of boundary condition configurations:

.. code-block:: bash

    # Run all examples
    python run_boundary_conditions_examples.py

    # Run specific categories
    python run_boundary_conditions_examples.py --tidal
    python run_boundary_conditions_examples.py --hybrid
    python run_boundary_conditions_examples.py --river
    python run_boundary_conditions_examples.py --nested

    # Run single example
    python run_boundary_conditions_examples.py --single basic_tidal

    # Dry run (validation only)
    python run_boundary_conditions_examples.py --dry-run

Example Categories
------------------

The testing framework supports several categories of boundary condition examples:

**Tidal Examples:**

* ``basic_tidal`` - Pure tidal forcing with M2, S2, N2 constituents
* ``extended_tidal`` - Enhanced tidal setup with refined parameters
* ``tidal_with_potential`` - Tidal forcing with earth tidal potential
* ``tide_wave`` - Tidal forcing with wave interaction (WWM)
* ``tidal_with_mdt`` - Tidal forcing with Mean Dynamic Topography correction
* ``tidal_with_mdt_const`` - Tidal forcing with constant MDT

**Hybrid Examples:**

* ``hybrid_elevation`` - Combined tidal and external elevation data
* ``full_hybrid`` - Complete hybrid setup with all boundary types

**River Examples:**

* ``simple_river`` - Single river inflow with tidal ocean boundary
* ``multi_river`` - Multiple river boundaries with different properties

**Nested Examples:**

* ``nested_with_tides`` - Nested boundary conditions with relaxation

Docker Image Management
=======================

The framework automatically manages Docker image building and container lifecycle.

Automatic Image Building
-------------------------

Docker images are built automatically from the SCHISM Dockerfile:

.. code-block:: python

    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),
        build_context=project_root / "docker" / "schism",
        # ... other parameters
    )

The framework will:

1. Locate the Dockerfile in the build context
2. Build the Docker image with appropriate tags
3. Use the built image for model execution
4. Clean up containers after execution

Resource Management
-------------------

The framework provides proper resource allocation and cleanup:

.. code-block:: python

    docker_config = DockerConfig(
        cpu=8,              # Number of CPU cores
        memory="4g",        # Memory limit
        timeout=3600,       # Execution timeout (seconds)
        remove_container=True,  # Clean up after execution
        volumes=[           # Volume mounts
            f"{schism_dir}:/tmp/schism:rw"
        ]
    )

Testing Framework
=================

The backend framework includes comprehensive testing capabilities.

Test Suite Components
---------------------

The test suite validates multiple aspects of the framework:

**Initialization Testing:**

.. code-block:: python

    def test_runner_initialization():
        """Test that the runner initializes correctly."""
        runner = SchismExampleRunner()
        assert runner.project_root.exists()
        assert runner.examples_dir.exists()
        assert len(runner.examples) > 0

**Configuration Validation:**

.. code-block:: python

    def test_configuration_validation():
        """Test that example configurations are valid."""
        runner = SchismExampleRunner()
        for name, config in runner.examples.items():
            config_file = config["file"]
            assert config_file.exists()
            # Validate YAML structure
            with open(config_file, 'r') as f:
                yaml_data = yaml.safe_load(f)
            assert "run_id" in yaml_data
            assert "period" in yaml_data
            assert "config" in yaml_data

**Docker Configuration Testing:**

.. code-block:: python

    def test_docker_config_creation():
        """Test Docker configuration creation."""
        runner = SchismExampleRunner()
        test_path = Path("/tmp/test_schism")
        test_path.mkdir(parents=True, exist_ok=True)

        config = runner._create_docker_config(test_path, "")
        assert config.dockerfile == Path("Dockerfile")
        assert config.build_context.exists()
        assert config.cpu == 8
        assert config.memory == "4g"

Running Tests
-------------

Execute the test suite to validate the framework:

.. code-block:: bash

    # Run comprehensive test suite
    python test_backend_examples.py

    # Expected output:
    # ============================================================
    # SCHISM Backend Framework Test Suite
    # ============================================================
    # Testing SchismExampleRunner initialization...             ✓
    # Testing example discovery...                              ✓
    # Testing configuration validation...                       ✓
    # Testing dry run functionality...                          ✓
    # Testing Docker configuration creation...                  ✓
    # Testing prerequisites...                                  ✓
    # ============================================================
    # TEST SUMMARY
    # ============================================================
    # Passed: 6
    # Failed: 0
    # Total:  6
    #
    # 🎉 All tests passed!

Configuration Examples
======================

YAML Configuration
------------------

Configure backend execution in YAML configuration files:

.. code-block:: yaml

    # Basic SCHISM configuration with backend
    run_id: schism_example
    period:
      start: 2023-01-01T00:00:00
      end: 2023-01-02T00:00:00
      interval: 1H
    output_dir: schism_output

    config:
      model_type: schism
      grid:
        grid_type: schism
        hgrid:
          model_type: data_blob
          source: tests/schism/test_data/hgrid.gr3

      data:
        data_type: schism
        boundary_conditions:
          data_type: boundary_conditions
          setup_type: tidal
          tidal_data:
            tidal_database: tests/schism/test_data/tides
            tidal_model: 'OCEANUM-atlas'
            constituents: [M2, S2, N2]
          boundaries:
            0:
              elev_type: 3
              vel_type: 3
              temp_type: 0
              salt_type: 0

Python Configuration
--------------------

Configure backend execution programmatically:

.. code-block:: python

    from rompy.model import ModelRun
    from rompy.backends import DockerConfig
    from rompy.core.time import TimeRange
    from datetime import datetime
    from pathlib import Path

    # Create model configuration
    model_run = ModelRun(
        run_id="schism_backend_example",
        period=TimeRange(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 2),
            interval="1H"
        ),
        output_dir="schism_output",
        delete_existing=True
    )

    # Configure Docker backend
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),
        build_context=Path("docker/schism"),
        timeout=3600,
        cpu=8,
        memory="4g",
        executable="bash -c 'cd /tmp/schism && mpirun --allow-run-as-root -n 8 schism_v5.13.0 4'",
        volumes=[f"{output_dir}:/tmp/schism:rw"],
        env_vars={
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"
        }
    )

    # Generate configuration
    model_run.generate()

    # Execute simulation
    success = model_run.run(backend=docker_config)

    if success:
        print("SCHISM simulation completed successfully")
        # Process results
        model_run.postprocess()
    else:
        print("SCHISM simulation failed")

Advanced Usage
==============

Custom Docker Images
--------------------

Use custom Docker images for specific SCHISM versions or configurations:

.. code-block:: python

    # Build custom image with specific SCHISM version
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile.custom"),
        build_context=Path("docker/schism"),
        build_args={
            "SCHISM_VERSION": "v5.11.1",
            "ENABLE_WWM": "ON"
        }
    )

Parallel Execution
------------------

Configure parallel execution for multiple examples:

.. code-block:: python

    import concurrent.futures
    from rompy.backends import DockerConfig

    def run_example(example_name):
        """Run a single example."""
        runner = SchismExampleRunner()
        return runner._run_example(example_name)

    # Run multiple examples in parallel
    examples = ["basic_tidal", "extended_tidal", "hybrid_elevation"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_example, name) for name in examples]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]

Performance Optimization
------------------------

Optimize Docker configuration for better performance:

.. code-block:: python

    # High-performance configuration
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),
        build_context=Path("docker/schism"),
        cpu=16,                 # More CPU cores
        memory="8g",            # More memory
        timeout=7200,           # Longer timeout
        volumes=[
            f"{schism_dir}:/tmp/schism:rw",
            "/tmp:/tmp:rw"      # Additional temp space
        ],
        env_vars={
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "OMP_NUM_THREADS": "16"
        }
    )

Error Handling
==============

The framework provides comprehensive error handling and logging.

Common Error Scenarios
----------------------

**Docker Build Failures:**

.. code-block:: python

    try:
        success = model_run.run(backend=docker_config)
    except Exception as e:
        if "dockerfile" in str(e).lower():
            print(f"Docker build failed: {e}")
            print("Check Dockerfile and build context paths")
        else:
            raise

**Resource Constraints:**

.. code-block:: python

    # Handle resource constraint errors
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),
        build_context=Path("docker/schism"),
        cpu=min(8, os.cpu_count()),  # Don't exceed available CPUs
        memory="4g",
        timeout=3600
    )

**Volume Mount Issues:**

.. code-block:: python

    # Ensure directories exist before mounting
    output_dir = Path("schism_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    docker_config = DockerConfig(
        volumes=[f"{output_dir.absolute()}:/tmp/schism:rw"]
    )

Logging and Monitoring
----------------------

Enable detailed logging for debugging:

.. code-block:: python

    import logging

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # Run with detailed logging
    logger = logging.getLogger(__name__)
    logger.info("Starting SCHISM backend execution")

    try:
        success = model_run.run(backend=docker_config)
        logger.info(f"Execution completed: {'success' if success else 'failed'}")
    except Exception as e:
        logger.error(f"Execution failed: {e}")
        raise

Migration Guide
===============

Migrating from Bash Scripts
----------------------------

If you're migrating from bash-based SCHISM execution:

**Old Approach (Bash):**

.. code-block:: bash

    #!/bin/bash
    docker build -t schism -f Dockerfile .
    docker run -v $PWD:/tmp/schism schism bash -c "cd /tmp/schism && mpirun --allow-run-as-root -n 8 schism_v5.13.0 4"

**New Approach (Python + Backend Framework):**

.. code-block:: python

    from rompy.backends import DockerConfig
    from rompy.model import ModelRun
    from pathlib import Path

    # Configure backend
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),
        build_context=Path("."),
        executable="bash -c 'cd /tmp/schism && mpirun --allow-run-as-root -n 8 schism_v5.13.0 4'",
        volumes=[f"{Path.cwd()}:/tmp/schism:rw"]
    )

    # Execute
    model_run = ModelRun(run_id="example", ...)
    success = model_run.run(backend=docker_config)

Benefits of Migration
---------------------

1. **Consistent Backend System** - Uses the same backend framework as other ROMPY models
2. **Automatic Image Building** - No manual Docker commands required
3. **Better Error Handling** - Comprehensive exception handling and logging
4. **Type Safety** - Python type hints and Pydantic validation
5. **Testing Framework** - Built-in testing and validation capabilities
6. **Maintainability** - Python code is easier to maintain than bash scripts

Best Practices
===============

1. **Use Dockerfile Parameters**

   Always use ``dockerfile`` and ``build_context`` instead of pre-built images for reproducibility.

2. **Resource Management**

   Set appropriate CPU, memory, and timeout limits based on your model requirements.

3. **Volume Mounts**

   Use absolute paths for volume mounts to avoid path resolution issues.

4. **Environment Variables**

   Set required environment variables for MPI execution in Docker containers.

5. **Testing**

   Always test configurations with dry runs before full execution.

6. **Cleanup**

   Enable container cleanup with ``remove_container=True`` to avoid accumulating containers.

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Docker Build Failures:**
   - Check that Dockerfile exists in build context
   - Verify build context path is correct
   - Ensure Docker daemon is running

**Container Execution Failures:**
   - Check volume mount paths exist
   - Verify executable command syntax
   - Review environment variable settings

**Resource Exhaustion:**
   - Reduce CPU/memory allocation
   - Increase timeout values
   - Monitor system resources during execution

**Permission Issues:**
   - Use ``user: root`` for MPI execution
   - Set appropriate volume mount permissions
   - Configure MPI environment variables

See Also
========

* :doc:`boundary_conditions` - SCHISM boundary conditions documentation
* :doc:`hotstart` - Hotstart configuration documentation
* :doc:`../backends` - ROMPY backend framework documentation
* :class:`rompy.backends.DockerConfig` - Docker backend configuration
* :class:`rompy.model.ModelRun` - Model execution framework
