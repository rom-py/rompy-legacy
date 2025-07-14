=================
Backend Reference
=================

This document provides comprehensive technical reference for ROMPY's backend system, focusing on concepts, usage patterns, and advanced configuration techniques.

.. note::
   For getting started with backends, see :doc:`backends`. For complete API documentation, see :doc:`api`.

Backend Configuration System
=============================

The backend system uses Pydantic models to provide type-safe, validated execution parameters. All configurations inherit from :class:`rompy.backends.config.BaseBackendConfig`.

Configuration Hierarchy
------------------------

.. code-block:: text

    BaseBackendConfig
    ├── LocalConfig          # Local system execution
    ├── DockerConfig         # Docker container execution
    └── CustomConfig         # User-defined configurations

Configuration Loading
----------------------

Configurations can be loaded from files or created programmatically:

.. code-block:: python

    import yaml
    from rompy.backends import LocalConfig, DockerConfig

    # From YAML file
    with open("config.yml") as f:
        config_data = yaml.safe_load(f)
        config = LocalConfig(**config_data)

    # Programmatically
    config = DockerConfig(
        image="swan:latest",
        cpu=4,
        memory="2g"
    )

For complete configuration class documentation, see:

* :class:`rompy.backends.config.BaseBackendConfig`
* :class:`rompy.backends.config.LocalConfig`
* :class:`rompy.backends.config.DockerConfig`

Configuration File Formats
===========================

Backend configurations support YAML and JSON formats with a common structure.

YAML Format
-----------

.. code-block:: yaml

    # Local execution example
    type: local
    timeout: 3600
    command: "python run_model.py"
    env_vars:
      OMP_NUM_THREADS: "4"
      MODEL_DEBUG: "true"

    ---
    # Docker execution example
    type: docker
    image: "swan:latest"
    cpu: 8
    memory: "4g"
    timeout: 10800
    volumes:
      - "/data/input:/app/input:ro"
      - "/data/output:/app/output:rw"
    env_vars:
      MODEL_THREADS: "8"

JSON Format
-----------

.. code-block:: json

    {
      "type": "local",
      "timeout": 3600,
      "command": "python run_model.py",
      "env_vars": {
        "OMP_NUM_THREADS": "4"
      }
    }

Configuration Validation
=========================

Pydantic provides comprehensive validation with descriptive error messages.

Validation Rules
----------------

**Common Validation (BaseBackendConfig):**

* ``timeout``: Must be between 60 and 86400 seconds
* ``env_vars``: Must be string key-value pairs
* ``working_dir``: Must exist if specified

**LocalConfig Validation:**

* ``command``: Must be non-empty string if provided
* ``shell``: Must be boolean
* ``capture_output``: Must be boolean

**DockerConfig Validation:**

* Either ``image`` or ``dockerfile`` must be provided (not both)
* ``cpu``: Must be between 1 and 128
* ``memory``: Must match pattern (e.g., "2g", "512m")
* ``volumes``: Must use "host:container[:mode]" format with existing host paths

Error Handling
--------------

.. code-block:: python

    from rompy.backends import DockerConfig
    from pydantic import ValidationError

    try:
        config = DockerConfig(cpu=200)  # Invalid - exceeds maximum
    except ValidationError as e:
        for error in e.errors():
            print(f"Field {error['loc']}: {error['msg']}")

Schema Generation
=================

Generate configuration schemas for validation and documentation:

.. code-block:: python

    from rompy.backends import LocalConfig
    import json

    # Generate JSON schema
    schema = LocalConfig.model_json_schema()

    # Save for external validation
    with open("local_schema.json", "w") as f:
        json.dump(schema, f, indent=2)

Using Schemas
-------------

.. code-block:: python

    import jsonschema

    # Validate configuration data against schema
    config_data = {"timeout": 3600, "command": "python run.py"}
    schema = LocalConfig.model_json_schema()

    try:
        jsonschema.validate(config_data, schema)
        print("Configuration is valid")
    except jsonschema.ValidationError as e:
        print(f"Validation error: {e.message}")

Advanced Configuration Patterns
===============================

Dynamic Configuration
----------------------

Create configurations based on runtime conditions:

.. code-block:: python

    import psutil
    from rompy.backends import LocalConfig, DockerConfig

    def create_optimal_config():
        """Create configuration based on system resources."""
        cpu_count = psutil.cpu_count()
        memory_gb = psutil.virtual_memory().total // (1024**3)

        if memory_gb > 16 and cpu_count > 8:
            return DockerConfig(
                image="swan:hpc",
                cpu=cpu_count,
                memory=f"{memory_gb}g",
                mpiexec=f"mpirun -np {cpu_count}"
            )
        else:
            return LocalConfig(
                timeout=7200,
                env_vars={"OMP_NUM_THREADS": str(min(cpu_count, 4))}
            )

Environment-Based Configuration
-------------------------------

Load different configurations based on environment:

.. code-block:: python

    import os
    from rompy.backends import LocalConfig, DockerConfig

    def load_config_for_environment():
        """Load configuration based on ROMPY_ENV environment variable."""
        env = os.getenv("ROMPY_ENV", "development")

        configs = {
            "production": DockerConfig(
                image="swan:production",
                cpu=16,
                memory="32g",
                timeout=21600
            ),
            "staging": DockerConfig(
                image="swan:staging",
                cpu=8,
                memory="16g",
                timeout=10800
            ),
            "development": LocalConfig(
                timeout=3600,
                env_vars={"LOG_LEVEL": "DEBUG"}
            )
        }

        return configs.get(env, configs["development"])

Configuration Templates
-----------------------

Create reusable configuration templates:

.. code-block:: python

    from rompy.backends import DockerConfig

    # Base template
    BASE_SWAN_CONFIG = {
        "image": "swan:latest",
        "user": "modeluser",
        "timeout": 7200,
        "env_vars": {
            "MODEL_DEBUG": "false",
            "LOG_LEVEL": "INFO"
        }
    }

    # Specialized configurations
    def create_hpc_config(**overrides):
        """Create HPC-optimized configuration."""
        config_data = {
            **BASE_SWAN_CONFIG,
            "cpu": 32,
            "memory": "64g",
            "mpiexec": "mpirun -np 32",
            **overrides
        }
        return DockerConfig(**config_data)

    def create_dev_config(**overrides):
        """Create development configuration."""
        config_data = {
            **BASE_SWAN_CONFIG,
            "cpu": 2,
            "memory": "2g",
            "remove_container": False,  # Keep for debugging
            "env_vars": {
                **BASE_SWAN_CONFIG["env_vars"],
                "MODEL_DEBUG": "true",
                "LOG_LEVEL": "DEBUG"
            },
            **overrides
        }
        return DockerConfig(**config_data)

Creating Custom Backends
=========================

The backend system supports custom implementations through inheritance and entry points.

Custom Configuration Classes
-----------------------------

Create custom configuration classes by inheriting from :class:`rompy.backends.config.BaseBackendConfig`:

.. code-block:: python

    from rompy.backends.config import BaseBackendConfig
    from pydantic import Field, validator
    from typing import Optional

    class SlurmConfig(BaseBackendConfig):
        """Configuration for SLURM cluster execution."""

        queue: str = Field(..., description="SLURM queue name")
        nodes: int = Field(1, ge=1, le=100, description="Number of nodes")
        partition: str = Field("compute", description="Cluster partition")
        time_limit: str = Field("1:00:00", description="Time limit (HH:MM:SS)")
        account: Optional[str] = Field(None, description="Account for billing")

        @validator('time_limit')
        def validate_time_limit(cls, v):
            import re
            if not re.match(r'^\d{1,2}:\d{2}:\d{2}$', v):
                raise ValueError("Time limit must be in format HH:MM:SS")
            return v

        def get_backend_class(self):
            from mypackage.backends import SlurmRunBackend
            return SlurmRunBackend

Custom Backend Implementation
-----------------------------

Implement backend classes that work with your custom configurations:

.. code-block:: python

    import logging
    from pathlib import Path

    class SlurmRunBackend:
        """Execute models on SLURM clusters."""

        def __init__(self):
            self.logger = logging.getLogger(__name__)

        def run(self, model_run, config: SlurmConfig) -> bool:
            """Submit model run to SLURM queue."""
            try:
                # Generate model input files
                model_run.generate()

                # Create and submit SLURM job
                job_script = self._create_job_script(model_run, config)
                job_id = self._submit_job(job_script)

                if job_id:
                    return self._wait_for_completion(job_id, config)
                return False

            except Exception as e:
                self.logger.error(f"SLURM execution failed: {e}")
                return False

        def _create_job_script(self, model_run, config):
            """Create SLURM job script."""
            # Implementation details...
            pass

        def _submit_job(self, job_script):
            """Submit job to SLURM."""
            # Implementation details...
            pass

        def _wait_for_completion(self, job_id, config):
            """Wait for job completion."""
            # Implementation details...
            pass

Entry Points Registration
-------------------------

Register custom backends in your package's ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."rompy.run"]
    slurm = "mypackage.backends:SlurmRunBackend"

    [project.entry-points."rompy.config"]
    slurm = "mypackage.config:SlurmConfig"

Backend Discovery
-----------------

The system automatically discovers registered backends:

.. code-block:: python

    from rompy.backends import get_available_backends

    # Get all available backends
    backends = get_available_backends()
    print("Available backends:", list(backends.keys()))

    # Use custom backend
    from mypackage.config import SlurmConfig

    config = SlurmConfig(
        queue="gpu",
        nodes=2,
        partition="compute",
        time_limit="2:00:00"
    )

    success = model_run.run(backend=config)

For complete backend discovery implementation, see :mod:`rompy.backends`.

Postprocessor System
====================

Postprocessors handle model outputs after execution. The system supports built-in and custom postprocessors.

Built-in Postprocessors
-----------------------

Available postprocessors include:

* **noop**: No-operation processor (default)
* **archive**: Archive outputs to compressed files
* **analyze**: Analyze model results
* **visualize**: Generate visualization outputs

For complete postprocessor documentation, see :mod:`rompy.backends.postprocessors`.

Usage Patterns
--------------

.. code-block:: python

    # Basic postprocessing
    results = model_run.postprocess(processor="archive")

    # Custom postprocessing with options
    results = model_run.postprocess(
        processor="analyze",
        output_format="netcdf",
        compress=True,
        analysis_type="spectral"
    )

Custom Postprocessors
---------------------

Create custom postprocessors by implementing the processor interface:

.. code-block:: python

    from typing import Dict, Any

    class CustomPostprocessor:
        """Custom postprocessor example."""

        def process(self, model_run, **kwargs) -> Dict[str, Any]:
            """Process model outputs."""
            try:
                # Custom processing logic here
                output_dir = Path(model_run.output_dir) / model_run.run_id

                # Process files in output_dir
                processed_files = self._process_outputs(output_dir, **kwargs)

                return {
                    "success": True,
                    "processed_files": processed_files,
                    "message": "Custom processing completed"
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }

        def _process_outputs(self, output_dir, **kwargs):
            """Implementation-specific processing."""
            # Custom processing logic
            pass

Best Practices
==============

Configuration Management
-------------------------

1. **Use Version Control**: Store configuration files in version control
2. **Environment Variables**: Use environment variables for sensitive data
3. **Validation**: Always validate configurations before production use
4. **Documentation**: Document custom configurations thoroughly
5. **Testing**: Test configurations with different scenarios

.. code-block:: python

    # Good: Use environment variables for sensitive data
    config = LocalConfig(
        env_vars={"API_KEY": os.environ.get("API_KEY")}
    )

    # Avoid: Hardcoding sensitive data
    config = LocalConfig(
        env_vars={"API_KEY": "secret-key-123"}
    )

Security Considerations
-----------------------

1. **Container Security**: Use non-root users in containers
2. **Volume Mounts**: Use read-only mounts when possible
3. **Resource Limits**: Set appropriate CPU/memory limits
4. **Environment Variables**: Never store secrets in configuration files

.. code-block:: python

    # Secure Docker configuration
    config = DockerConfig(
        image="swan:latest",
        user="appuser",  # Non-root user
        volumes=["/data:/app/data:ro"],  # Read-only mount
        cpu=4,  # Resource limit
        memory="4g"  # Memory limit
    )

Performance Optimization
------------------------

1. **Resource Allocation**: Match resources to model requirements
2. **Parallel Execution**: Use MPI for large models
3. **Image Optimization**: Use optimized Docker images
4. **Configuration Caching**: Cache validated configurations
5. **Monitoring**: Track resource usage patterns

.. code-block:: python

    # Performance-optimized configuration
    config = DockerConfig(
        image="swan:hpc-optimized",
        cpu=16,
        memory="32g",
        mpiexec="mpirun -np 16",
        env_vars={
            "OMP_NUM_THREADS": "1",  # Avoid thread oversubscription
            "MODEL_PRECISION": "double"
        }
    )

Error Handling
--------------

1. **Graceful Degradation**: Handle errors gracefully
2. **Informative Messages**: Provide clear error messages
3. **Logging**: Log important events and errors
4. **Retry Logic**: Implement retry mechanisms for transient failures
5. **Cleanup**: Ensure proper cleanup on failure

.. code-block:: python

    def safe_model_execution(model_run, config):
        """Safely execute model with error handling."""
        try:
            # Validate configuration
            if not config.validate():
                raise ValueError("Invalid configuration")

            # Execute model
            success = model_run.run(backend=config)

            if not success:
                logger.error("Model execution failed")
                return False

            return True

        except Exception as e:
            logger.error(f"Execution error: {e}")
            # Cleanup logic here
            return False

Testing
=======

Backend configurations and implementations should be thoroughly tested.

Configuration Testing
---------------------

.. code-block:: python

    import pytest
    from rompy.backends import LocalConfig
    from pydantic import ValidationError

    def test_local_config_validation():
        """Test LocalConfig validation."""
        # Valid configuration
        config = LocalConfig(timeout=3600, command="python test.py")
        assert config.timeout == 3600

        # Invalid configuration
        with pytest.raises(ValidationError):
            LocalConfig(timeout=30)  # Too short

Backend Testing
---------------

.. code-block:: python

    def test_backend_execution():
        """Test backend execution."""
        config = LocalConfig(timeout=600, command="echo 'test'")

        # Mock model run
        mock_model = create_mock_model()

        # Test execution
        backend = config.get_backend_class()()
        success = backend.run(mock_model, config)

        assert success is True

Integration Testing
-------------------

.. code-block:: python

    def test_full_workflow():
        """Test complete workflow with backend."""
        model_run = ModelRun.from_file("test_model.yml")
        config = LocalConfig(timeout=1800)

        # Test full workflow
        success = model_run.run(backend=config)
        results = model_run.postprocess(processor="archive")

        assert success is True
        assert results["success"] is True

For comprehensive testing examples, see the test suite in ``tests/backends/``.

Troubleshooting
===============

Common Issues
-------------

**Configuration Validation Errors**
    Use ``rompy backends validate`` to check configuration syntax and validate against schema.

**Docker Issues**
    Verify Docker installation, image availability, and volume mount permissions.

**Timeout Issues**
    Adjust timeout values based on model complexity and system performance.

**Memory Issues**
    Monitor memory usage and adjust allocation in Docker configurations.

**Permission Issues**
    Check file permissions for volume mounts and working directories.

Debug Mode
----------

Enable debug logging for detailed troubleshooting:

.. code-block:: python

    import logging

    # Enable debug logging
    logging.basicConfig(level=logging.DEBUG)

    # Create debug configuration
    config = LocalConfig(
        timeout=3600,
        env_vars={"LOG_LEVEL": "DEBUG", "MODEL_DEBUG": "true"}
    )

Getting Help
------------

1. **Check Documentation**: Review :doc:`backends` and :doc:`api`
2. **Validate Configuration**: Use ``rompy backends validate``
3. **Check Logs**: Review execution logs for error details
4. **Test Incrementally**: Start with simple configurations
5. **Community Support**: Check GitHub issues and discussions

For additional help, see the troubleshooting section in :doc:`backends` or file an issue on GitHub.

API Reference
=============

For complete API documentation, see:

* :doc:`api` - Complete API documentation
* :class:`rompy.backends.config.BaseBackendConfig` - Base configuration class
* :class:`rompy.backends.config.LocalConfig` - Local execution configuration
* :class:`rompy.backends.config.DockerConfig` - Docker execution configuration
* :mod:`rompy.run` - Backend implementation classes
* :mod:`rompy.backends.postprocessors` - Postprocessor implementations
* :mod:`rompy.backends` - Backend discovery and registry

This reference covers the key concepts and patterns for working with ROMPY's backend system. For implementation details and complete parameter documentation, refer to the API documentation.
