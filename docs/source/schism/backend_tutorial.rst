===============================
SCHISM Backend Framework Tutorial
===============================

This tutorial guides you through using the SCHISM backend framework to execute SCHISM simulations with Docker containers, automatic image building, and comprehensive testing.

Getting Started
===============

Prerequisites
-------------

Before starting, ensure you have:

* Docker installed and running
* ROMPY installed with backend framework support
* SCHISM boundary conditions examples (included with ROMPY)

Quick Start
-----------

The easiest way to get started is with the boundary conditions examples:

.. code-block:: bash

    # Navigate to the SCHISM examples directory
    cd rompy/notebooks/schism

    # Run a single tidal example (dry run)
    python run_boundary_conditions_examples.py --single basic_tidal --dry-run

    # Run all tidal examples
    python run_boundary_conditions_examples.py --tidal

    # Run a complete test suite
    python test_backend_examples.py

Basic Tutorial
==============

Step 1: Understanding the Framework
-----------------------------------

The SCHISM backend framework consists of three main components:

1. **ModelRun** - ROMPY's model execution framework
2. **DockerConfig** - Backend configuration for Docker execution
3. **Boundary Conditions** - SCHISM-specific configuration system

Here's how they work together:

.. code-block:: python

    from rompy.model import ModelRun
    from rompy.backends import DockerConfig
    from rompy.core.time import TimeRange
    from datetime import datetime
    from pathlib import Path

    # 1. Create a model run
    model_run = ModelRun(
        run_id="my_schism_example",
        period=TimeRange(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 2),
            interval="1H"
        ),
        output_dir="schism_output",
        delete_existing=True
    )

    # 2. Configure Docker backend
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile"),
        build_context=Path("docker/schism"),
        timeout=3600,
        cpu=8,
        memory="4g"
    )

    # 3. Generate configuration files
    model_run.generate()

    # 4. Execute simulation
    success = model_run.run(backend=docker_config)

Step 2: Creating Your First SCHISM Configuration
------------------------------------------------

Create a YAML configuration file for a simple tidal simulation:

.. code-block:: yaml

    # my_tidal_example.yaml
    run_id: my_first_schism_run
    period:
      start: 2023-01-01T00:00:00
      end: 2023-01-01T12:00:00
      interval: 1H
    output_dir: my_schism_output
    delete_existing: true

    config:
      model_type: schism
      grid:
        grid_type: schism
        hgrid:
          model_type: data_blob
          source: tests/schism/test_data/hgrid.gr3
        drag: 2.5e-3

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
              elev_type: 3  # HARMONIC
              vel_type: 3   # HARMONIC
              temp_type: 0  # NONE
              salt_type: 0  # NONE

      nml:
        param:
          core:
            dt: 150.0
            ibc: 1  # Barotropic
            ibtp: 0  # No tracer transport
            nspool: 24
            ihfskip: 1152
          schout:
            iof_hydro__1: 1  # elevation
            iof_hydro__26: 1  # velocity vector

Step 3: Running Your Configuration
----------------------------------

Now create a Python script to execute your configuration:

.. code-block:: python

    # run_my_example.py
    import yaml
    from rompy.model import ModelRun
    from rompy.backends import DockerConfig
    from pathlib import Path

    def main():
        # Load configuration
        with open("my_tidal_example.yaml", "r") as f:
            config = yaml.safe_load(f)

        # Create model run
        model_run = ModelRun(**config)

        # Configure Docker backend
        docker_config = DockerConfig(
            dockerfile=Path("Dockerfile"),
            build_context=Path("docker/schism"),
            timeout=3600,
            cpu=8,
            memory="4g",
            executable="bash -c 'cd /tmp/schism && mpirun --allow-run-as-root -n 8 schism_v5.13.0 4'",
            volumes=[f"{Path.cwd() / 'my_schism_output'}:/tmp/schism:rw"],
            env_vars={
                "OMPI_ALLOW_RUN_AS_ROOT": "1",
                "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1"
            },
            remove_container=True
        )

        try:
            # Generate configuration files
            print("Generating SCHISM configuration...")
            model_run.generate()

            # Execute simulation
            print("Running SCHISM simulation...")
            success = model_run.run(backend=docker_config)

            if success:
                print("✅ Simulation completed successfully!")

                # Check for output files
                output_dir = Path("my_schism_output")
                outputs = list(output_dir.glob("**/outputs/*.nc"))
                print(f"Generated {len(outputs)} output files")

            else:
                print("❌ Simulation failed")

        except Exception as e:
            print(f"❌ Error: {e}")

    if __name__ == "__main__":
        main()

Step 4: Execute Your Example
----------------------------

Run your example:

.. code-block:: bash

    python run_my_example.py

Expected output:

.. code-block:: text

    Generating SCHISM configuration...
    Running SCHISM simulation...
    ✅ Simulation completed successfully!
    Generated 1 output files

Intermediate Tutorial
=====================

Step 5: Adding Wave Coupling
-----------------------------

Extend your configuration to include wave coupling:

.. code-block:: yaml

    # Add to your existing configuration
    config:
      data:
        wave:
          buffer: 0.0
          coords:
            t: time
            x: lon
            y: lat
            z: depth
          id: wavedata
          source:
            catalog_uri: tests/data/catalog.yaml
            dataset_id: ausspec
            model_type: intake

      nml:
        param:
          opt:
            ihot: 0
            nstep_wwm: 4
          schout:
            iof_wwm__1: 1   # significant wave height
            iof_wwm__9: 1   # peak wave period
            iof_wwm__18: 1  # peak wave direction
        wwminput:
          proc:
            deltc: 600

Step 6: Hybrid Boundary Conditions
-----------------------------------

Configure hybrid boundaries that combine tidal and ocean data:

.. code-block:: yaml

    config:
      data:
        boundary_conditions:
          data_type: boundary_conditions
          setup_type: hybrid
          tidal_data:
            tidal_database: tests/schism/test_data/tides
            tidal_model: 'OCEANUM-atlas'
            constituents: [M2, S2, N2]
          boundaries:
            0:
              elev_type: 5  # HARMONICEXTERNAL
              vel_type: 5   # HARMONICEXTERNAL
              temp_type: 4  # EXTERNAL
              salt_type: 4  # EXTERNAL
              elev_source:
                data_type: boundary
                source:
                  model_type: file
                  uri: tests/schism/test_data/hycom.nc
                variables: [surf_el]
                coords:
                  t: time
                  x: xlon
                  y: ylat
              vel_source:
                data_type: boundary
                source:
                  model_type: file
                  uri: tests/schism/test_data/hycom.nc
                variables: [water_u, water_v]
                coords:
                  t: time
                  x: xlon
                  y: ylat
                  z: depth

Step 7: Adding Hotstart Generation
-----------------------------------

Generate initial conditions from your ocean data:

.. code-block:: yaml

    config:
      data:
        boundary_conditions:
          hotstart_config:
            enabled: true
            temp_var: temperature
            salt_var: salinity
            output_filename: hotstart.nc

Advanced Tutorial
=================

Step 8: Custom Docker Configuration
------------------------------------

Create advanced Docker configurations for specific needs:

.. code-block:: python

    # High-performance configuration
    docker_config = DockerConfig(
        dockerfile=Path("Dockerfile.optimized"),
        build_context=Path("docker/schism"),
        build_args={
            "SCHISM_VERSION": "v5.13.0",
            "ENABLE_OPTIMIZATION": "ON"
        },
        cpu=16,
        memory="16g",
        timeout=7200,
        volumes=[
            f"{output_dir}:/tmp/schism:rw",
            "/tmp:/tmp_host:rw"
        ],
        env_vars={
            "OMPI_ALLOW_RUN_AS_ROOT": "1",
            "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
            "OMP_NUM_THREADS": "16"
        }
    )

Step 9: Multiple Boundary Types
--------------------------------

Configure complex scenarios with multiple boundary types:

.. code-block:: yaml

    config:
      data:
        boundary_conditions:
          data_type: boundary_conditions
          setup_type: mixed
          tidal_data:
            tidal_database: tests/schism/test_data/tides
            tidal_model: 'OCEANUM-atlas'
            constituents: [M2, S2, N2]
          boundaries:
            0:  # Ocean boundary (tidal + external)
              elev_type: 5
              vel_type: 5
              temp_type: 4
              salt_type: 4
              # ... data sources
            1:  # River boundary (constant flow)
              elev_type: 0
              vel_type: 2
              temp_type: 2
              salt_type: 2
              const_flow: -100.0
              const_temp: 15.0
              const_salt: 0.1
            2:  # Nested boundary (relaxation)
              elev_type: 5
              vel_type: 7
              temp_type: 4
              salt_type: 4
              inflow_relax: 0.8
              outflow_relax: 0.2

Step 10: Automated Testing
---------------------------

Create automated testing for your configurations:

.. code-block:: python

    # test_my_configurations.py
    import pytest
    import yaml
    from pathlib import Path
    from rompy.model import ModelRun
    from rompy.backends import DockerConfig

    class TestSchismConfigurations:

        def test_basic_tidal_config(self):
            """Test basic tidal configuration."""
            config_file = Path("my_tidal_example.yaml")
            assert config_file.exists()

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            model_run = ModelRun(**config)
            assert model_run.run_id == "my_first_schism_run"

        def test_docker_config_creation(self):
            """Test Docker configuration creation."""
            docker_config = DockerConfig(
                dockerfile=Path("Dockerfile"),
                build_context=Path("docker/schism"),
                cpu=8,
                memory="4g"
            )

            assert docker_config.dockerfile == Path("Dockerfile")
            assert docker_config.cpu == 8
            assert docker_config.memory == "4g"

        def test_dry_run_execution(self):
            """Test configuration generation without execution."""
            with open("my_tidal_example.yaml", "r") as f:
                config = yaml.safe_load(f)

            model_run = ModelRun(**config)

            # Test configuration generation
            try:
                model_run.generate()
                assert True, "Configuration generation succeeded"
            except Exception as e:
                pytest.fail(f"Configuration generation failed: {e}")

    if __name__ == "__main__":
        pytest.main([__file__, "-v"])

Production Usage
================

Step 11: Batch Processing
--------------------------

Run multiple configurations in batch:

.. code-block:: python

    # batch_runner.py
    from concurrent.futures import ThreadPoolExecutor
    from pathlib import Path
    import yaml
    from rompy.model import ModelRun
    from rompy.backends import DockerConfig

    def run_configuration(config_file):
        """Run a single configuration file."""
        try:
            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            model_run = ModelRun(**config)

            docker_config = DockerConfig(
                dockerfile=Path("Dockerfile"),
                build_context=Path("docker/schism"),
                timeout=3600,
                cpu=8,
                memory="4g"
            )

            model_run.generate()
            success = model_run.run(backend=docker_config)

            return config_file.name, success

        except Exception as e:
            return config_file.name, False

    def main():
        # Find all configuration files
        config_files = list(Path(".").glob("*_example.yaml"))

        # Run configurations in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            results = list(executor.map(run_configuration, config_files))

        # Report results
        for config_name, success in results:
            status = "✅" if success else "❌"
            print(f"{status} {config_name}")

    if __name__ == "__main__":
        main()

Step 12: Monitoring and Logging
--------------------------------

Add comprehensive monitoring to your runs:

.. code-block:: python

    # monitored_runner.py
    import logging
    import time
    from pathlib import Path
    from rompy.model import ModelRun
    from rompy.backends import DockerConfig

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('schism_runs.log'),
            logging.StreamHandler()
        ]
    )

    logger = logging.getLogger(__name__)

    def run_with_monitoring(config_file):
        """Run configuration with detailed monitoring."""
        start_time = time.time()

        try:
            logger.info(f"Starting run: {config_file}")

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            model_run = ModelRun(**config)

            logger.info(f"Run ID: {model_run.run_id}")
            logger.info(f"Period: {model_run.period.start} to {model_run.period.end}")

            docker_config = DockerConfig(
                dockerfile=Path("Dockerfile"),
                build_context=Path("docker/schism"),
                timeout=3600,
                cpu=8,
                memory="4g"
            )

            # Generate configuration
            logger.info("Generating configuration files...")
            model_run.generate()

            # Execute simulation
            logger.info("Starting SCHISM simulation...")
            success = model_run.run(backend=docker_config)

            execution_time = time.time() - start_time

            if success:
                logger.info(f"✅ Run completed successfully in {execution_time:.1f}s")

                # Check output files
                output_dir = Path(config["output_dir"])
                outputs = list(output_dir.glob("**/outputs/*.nc"))
                logger.info(f"Generated {len(outputs)} output files")

                return True
            else:
                logger.error(f"❌ Run failed after {execution_time:.1f}s")
                return False

        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"❌ Run crashed after {execution_time:.1f}s: {e}")
            return False

Troubleshooting
===============

Common Issues and Solutions
---------------------------

**Docker Build Failures:**

.. code-block:: bash

    # Check Docker daemon
    docker version

    # Verify Dockerfile exists
    ls -la docker/schism/Dockerfile

    # Test manual build
    cd docker/schism
    docker build -t test-schism .

**Configuration Errors:**

.. code-block:: python

    # Validate YAML syntax
    import yaml
    with open("my_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print("✅ YAML is valid")

    # Test ModelRun creation
    from rompy.model import ModelRun
    model_run = ModelRun(**config)
    print("✅ ModelRun created successfully")

**Resource Issues:**

.. code-block:: python

    # Check system resources
    import psutil
    print(f"Available CPUs: {psutil.cpu_count()}")
    print(f"Available Memory: {psutil.virtual_memory().available / 1024**3:.1f} GB")

    # Adjust Docker configuration accordingly
    docker_config = DockerConfig(
        cpu=min(8, psutil.cpu_count()),
        memory=f"{min(4, psutil.virtual_memory().available // 1024**3)}g"
    )

**File Path Issues:**

.. code-block:: python

    # Use absolute paths for volume mounts
    from pathlib import Path
    output_dir = Path("my_output").absolute()

    docker_config = DockerConfig(
        volumes=[f"{output_dir}:/tmp/schism:rw"]
    )

Best Practices Summary
======================

1. **Start Simple** - Begin with basic tidal configurations
2. **Test Configurations** - Use dry runs to validate before execution
3. **Monitor Resources** - Set appropriate CPU and memory limits
4. **Use Absolute Paths** - Avoid path resolution issues
5. **Enable Logging** - Track execution progress and errors
6. **Clean Up** - Remove containers after execution
7. **Version Control** - Track configuration changes
8. **Document Runs** - Record parameters and results

Next Steps
==========

After completing this tutorial, you should be able to:

- Create and run basic SCHISM configurations
- Use the Docker backend framework effectively
- Configure complex boundary conditions
- Implement automated testing
- Monitor and troubleshoot runs

For more advanced features, see:

- :doc:`boundary_conditions` - Comprehensive boundary conditions guide
- :doc:`hotstart` - Initial conditions configuration
- :doc:`backend_framework` - Complete backend framework reference
- :doc:`../backends` - ROMPY backend system documentation

Happy modeling with SCHISM and ROMPY! 🌊
