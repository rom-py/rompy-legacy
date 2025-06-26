=============================
Backend Systems
=============================

ROMPY uses a flexible backend system to execute models, process outputs, and orchestrate complete workflows. The system combines Python's entry point mechanism for backend discovery with Pydantic configuration models for type-safe, validated execution parameters.

Overview
--------

The backend system consists of three main components:

* **Run Backends**: Execute model runs in different environments (local, Docker, cloud, etc.)
* **Postprocessors**: Process model outputs after execution (compression, analysis, visualization, etc.)
* **Pipeline Backends**: Orchestrate complete workflows from generation through postprocessing

**Key Features:**

* **Type-Safe Configuration**: Pydantic models provide validation and IDE support
* **Automatic Discovery**: Backends are discovered through Python entry points
* **Extensible Architecture**: Easy addition of custom backends without core changes
* **Configuration Files**: YAML/JSON support for reproducible execution setups

For detailed information on type-safe backend configurations, see :doc:`backend_configurations`.
Backend configurations use Pydantic models for type safety and validation, while backend implementations are discovered through Python entry points for extensibility.

**Key Features:**

* **Type Safety**: Pydantic configuration objects with full validation
* **IDE Support**: Autocompletion and inline documentation
* **Configuration Files**: YAML/JSON serialization support
* **Schema Generation**: Self-documenting configuration structure
* **Clear Validation**: Descriptive error messages for invalid configurations

Run Backends
------------

Run backends execute model runs using Pydantic configuration objects that provide type safety and validation. Each backend type has a corresponding configuration class that defines all available parameters.

.. note::
   For detailed information on backend configurations, see :doc:`backend_configurations`.

Built-in Run Backends
^^^^^^^^^^^^^^^^^^^^^^

**Local Backend**
    Executes models directly on the local system using ``LocalConfig`` configuration objects.

    .. code-block:: python

        from rompy.backends import LocalConfig

        # Basic local execution
        config = LocalConfig(
            timeout=3600,
            command="python run_model.py"
        )
        success = model_run.run(backend=config)

        # Advanced local execution
        config = LocalConfig(
            timeout=7200,
            env_vars={"OMP_NUM_THREADS": "4"},
            command="python run_model.py --verbose",
            shell=True,
            capture_output=True
        )
        success = model_run.run(backend=config)

**Docker Backend**
    Executes models inside Docker containers using ``DockerConfig`` configuration objects.

    .. code-block:: python

        from rompy.backends import DockerConfig

        # Run with pre-built image
        config = DockerConfig(
            image="swan:latest",
            cpu=4,
            memory="2g",
            env_vars={"SWAN_THREADS": "4"},
            volumes=["/data:/app/data:ro"]
        )
        success = model_run.run(backend=config)

        # Build and run from Dockerfile
        config = DockerConfig(
            dockerfile="./docker/Dockerfile",
            cpu=2,
            build_args={"BASE_IMAGE": "ubuntu:20.04"},
            executable="/usr/local/bin/swan"
        )
        success = model_run.run(backend=config)

Using Run Backends
^^^^^^^^^^^^^^^^^^

The ``run()`` method accepts Pydantic configuration objects that provide type safety and validation:

.. code-block:: python

    from rompy.model import ModelRun
    from rompy.backends import LocalConfig, DockerConfig

    # Load your model configuration
    model_run = ModelRun.from_file("model_config.yml")

    # Execute locally with type-safe configuration
    local_config = LocalConfig(
        timeout=3600,
        env_vars={"OMP_NUM_THREADS": "4"},
        command="python run_simulation.py"
    )
    success = model_run.run(backend=local_config)

    # Execute in Docker with validated parameters
    docker_config = DockerConfig(
        image="rompy/swan:latest",
        cpu=4,
        memory="4g",
        volumes=["/data:/app/data:ro"],
        env_vars={"MODEL_THREADS": "4"}
    )
    success = model_run.run(backend=docker_config)

Backend Configuration Files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Configurations can be saved and loaded from YAML or JSON files:

.. code-block:: yaml

    # local_backend.yml
    type: local
    timeout: 7200
    env_vars:
      OMP_NUM_THREADS: "4"
      MODEL_DEBUG: "true"
    command: "python run_model.py --verbose"

.. code-block:: yaml

    # docker_backend.yml
    type: docker
    image: "swan:latest"
    cpu: 4
    memory: "2g"
    timeout: 10800
    volumes:
      - "/data/input:/app/input:ro"
      - "/data/output:/app/output:rw"

.. code-block:: python

    import yaml
    from rompy.backends import LocalConfig, DockerConfig

    # Load configuration from file
    with open("local_backend.yml") as f:
        config_data = yaml.safe_load(f)
        backend_type = config_data.pop("type")
        config = LocalConfig(**config_data)

    success = model_run.run(backend=config)

CLI Support
^^^^^^^^^^^

The command line interface supports backend configuration files:

.. code-block:: bash

    # Run with backend configuration file
    rompy run model_config.yml --backend-config local_backend.yml

    # Create configuration templates
    rompy backends create --backend-type local --output local_config.yml
    rompy backends create --backend-type docker --with-examples --output docker_config.yml

    # Validate configurations
    rompy backends validate local_config.yml --backend-type local

    # Generate JSON schema
    rompy backends schema --backend-type docker --format json

Postprocessors
--------------

Postprocessors handle model outputs after execution, enabling tasks like data analysis, visualization, archiving, or integration with external services.

Built-in Postprocessors
^^^^^^^^^^^^^^^^^^^^^^^^

**No-op Processor** (``noop``)
    A placeholder processor that performs no operations but returns success. Useful for testing or when no postprocessing is needed.

    .. code-block:: python

        # Use no-op processor (default)
        results = model_run.postprocess()

        # Explicitly specify no-op processor
        results = model_run.postprocess(processor="noop")

.. note::
    Future versions will include Pydantic configuration support for postprocessors, similar to run backends.

Using Postprocessors
^^^^^^^^^^^^^^^^^^^^

The ``postprocess()`` method accepts a ``processor`` parameter and processor-specific keyword arguments:

.. code-block:: python

    # Basic postprocessing
    results = model_run.postprocess(processor="noop")
    print(results)  # {'success': True, 'message': 'No postprocessing requested'}

Pipeline Backends
-----------------

Pipeline backends orchestrate complete workflows, executing the full sequence of model generation, execution, and postprocessing.

Built-in Pipeline Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^

**Local Pipeline** (``local``)
    Executes the complete pipeline locally using the existing ModelRun methods.

    .. code-block:: python

        # Run complete pipeline locally
        results = model_run.pipeline()

        # Run pipeline with specific backends
        results = model_run.pipeline(
            pipeline_backend="local",
            run_backend="docker",
            processor="noop",
            run_kwargs={"cpu": 4},
            process_kwargs={"compress": True}
        )

.. note::
    Future versions will include Pydantic configuration support for pipeline backends.

Using Pipeline Backends
^^^^^^^^^^^^^^^^^^^^^^^^

The ``pipeline()`` method orchestrates the complete workflow:

.. code-block:: python

    # Complete workflow with default settings
    results = model_run.pipeline()

    # Complete workflow with backend configurations
    from rompy.backends import DockerConfig

    docker_config = DockerConfig(
        image="rompy/swan:latest",
        cpu=4,
        memory="4g"
    )

    # Note: Currently uses run_kwargs, future versions will use backend configs directly
    results = model_run.pipeline(
        pipeline_backend="local",
        run_backend="docker",
        processor="custom_analyzer",
        run_kwargs={
            "image": "rompy/swan:latest",
            "cpu": 4
        },
        process_kwargs={
            "output_format": "netcdf",
            "compress": True
        }
    )

    print(results)
    # {
    #     'success': True,
    #     'run_success': True,
    #     'postprocess_results': {...}
    # }

Creating Custom Backends
-------------------------

The system supports custom backends with Pydantic configuration classes and entry point registration.

Custom Backend Configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Create a custom configuration class extending ``BaseBackendConfig``:

.. code-block:: python

    # my_backends/config.py
    from rompy.backends import BaseBackendConfig
    from pydantic import Field
    from typing import Optional

    class SlurmConfig(BaseBackendConfig):
        """Configuration for SLURM cluster execution."""

        queue: str = Field(..., description="SLURM queue name")
        nodes: int = Field(1, ge=1, le=100, description="Number of nodes")
        partition: str = Field("compute", description="Cluster partition")
        account: Optional[str] = Field(None, description="Account for billing")
        time_limit: str = Field("1:00:00", description="Time limit (HH:MM:SS)")

        def get_backend_class(self):
            from my_backends.run import SlurmRunBackend
            return SlurmRunBackend

Custom Run Backend
^^^^^^^^^^^^^^^^^^

Implement a backend class that accepts your configuration:

.. code-block:: python

    # my_backends/run.py
    import logging

    logger = logging.getLogger(__name__)

    class SlurmRunBackend:
        """Execute models on SLURM clusters."""

        def run(self, model_run, config: 'SlurmConfig'):
            """Submit model run to SLURM queue."""
            # Generate model input files
            model_run.generate()

            # Create SLURM job script using validated config
            job_script = self._create_slurm_script(model_run, config)

            # Submit job
            return self._submit_job(job_script)

        def _create_slurm_script(self, model_run, config):
            # Use config.queue, config.nodes, config.partition, etc.
            # All parameters are validated and type-safe
            pass

        def _submit_job(self, job_script):
            # Implementation details...
            pass

Custom Postprocessor
^^^^^^^^^^^^^^^^^^^^

Create a custom postprocessor by implementing a class with a ``process()`` method:

.. code-block:: python

    # my_backends/postprocess.py
    import zipfile
    from pathlib import Path
    from typing import Dict, Any

    class ArchivePostprocessor:
        """Archive model outputs to compressed files."""

        def process(self, model_run, archive_format="zip", **kwargs) -> Dict[str, Any]:
            """Archive model outputs."""
            output_dir = Path(model_run.output_dir) / model_run.run_id

            if archive_format == "zip":
                return self._create_zip_archive(output_dir, **kwargs)
            else:
                return {"success": False, "error": f"Unsupported format: {archive_format}"}

        def _create_zip_archive(self, output_dir, **kwargs):
            # Implementation details...
            archive_path = output_dir.parent / f"{output_dir.name}.zip"

            with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for file_path in output_dir.rglob('*'):
                    if file_path.is_file():
                        zipf.write(file_path, file_path.relative_to(output_dir))

            return {
                "success": True,
                "archive_path": str(archive_path),
                "message": f"Archived to {archive_path}"
            }

Registering Custom Backends
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Register your custom backend using entry points in your package's ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."rompy.run"]
    slurm = "my_backends.run:SlurmRunBackend"

After installation, your backend will be automatically available:

.. code-block:: python

    from my_backends.config import SlurmConfig

    # Use custom backend with type-safe configuration
    slurm_config = SlurmConfig(
        queue="gpu",
        nodes=2,
        partition="compute",
        time_limit="2:00:00",
        timeout=7200
    )
    success = model_run.run(backend=slurm_config)

**Configuration File Support:**

.. code-block:: yaml

    # slurm_backend.yml
    type: slurm
    queue: "gpu"
    nodes: 2
    partition: "compute"
    time_limit: "2:00:00"
    timeout: 7200

.. code-block:: bash

    # Use with CLI
    rompy run model.yml --backend-config slurm_backend.yml

Error Handling and Validation
------------------------------

The Pydantic configuration system provides comprehensive validation with clear error messages:

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

    try:
        # Invalid backend type for ModelRun
        success = model_run.run(backend="invalid_string")
    except TypeError as e:
        print(e)
        # TypeError: Backend must be a BackendConfig instance

Discovering Available Backends
-------------------------------

You can discover available backends and configurations:

.. code-block:: python

    from rompy.model import RUN_BACKENDS, POSTPROCESSORS, PIPELINE_BACKENDS
    from rompy.backends import LocalConfig, DockerConfig

    print("Available run backends:", list(RUN_BACKENDS.keys()))
    print("Available postprocessors:", list(POSTPROCESSORS.keys()))
    print("Available pipeline backends:", list(PIPELINE_BACKENDS.keys()))

    # Available configuration types
    print("Available backend configurations:")
    print("  LocalConfig →", LocalConfig().get_backend_class().__name__)
    print("  DockerConfig →", DockerConfig(image="test").get_backend_class().__name__)

Use the CLI to list and inspect backends:

.. code-block:: bash

    # List all available backends
    rompy backends list

    # Generate configuration schema
    rompy backends schema --backend-type local --format json

Best Practices
--------------

When developing with backend configurations:

1. **Use Configuration Files**: Store backend configurations in YAML/JSON files for different environments
2. **Validate Early**: Use Pydantic validation to catch configuration errors before execution
3. **Leverage IDE Support**: Take advantage of autocompletion and type checking
4. **Document Configurations**: Use Pydantic field descriptions for self-documenting configs
5. **Test Configurations**: Create unit tests for both valid and invalid configuration scenarios
6. **Environment-Specific Configs**: Maintain separate configurations for development, testing, and production

When developing custom backends:

1. **Create Pydantic Configs**: Always provide a corresponding configuration class
2. **Follow the Interface**: Implement ``run(model_run, config: YourConfig)`` method signature
3. **Use Type Hints**: Leverage full type safety throughout your implementation
4. **Handle Errors Gracefully**: Return appropriate success/failure indicators
5. **Use Logging**: Log important events and errors for debugging
6. **Test Thoroughly**: Create tests for both backend implementation and configuration validation

Backend Development Guidelines
------------------------------

**Configuration Classes**:
- Must inherit from ``BaseBackendConfig``
- Should use Pydantic fields with appropriate validation
- Must implement ``get_backend_class()`` method
- Include clear field descriptions for documentation
- Use proper type hints for all fields

**Run Backends**:
- Must implement ``run(model_run, config) -> bool`` where config is a BaseBackendConfig subclass
- Should call ``model_run.generate()`` to create input files
- Return ``True`` for success, ``False`` for failure
- Handle exceptions gracefully and log errors
- Use the config object for all execution parameters

**Postprocessors**:
- Must implement ``process(model_run, **kwargs) -> Dict[str, Any]``
- Should return a dictionary with at least a ``success`` key
- Include meaningful error messages in the return dictionary
- Process files in the model's output directory
- Future versions will use Pydantic configuration objects

**Pipeline Backends**:
- Must implement ``execute(model_run, **kwargs) -> Dict[str, Any]``
- Should coordinate the full workflow (generate, run, postprocess)
- Return comprehensive results including individual stage outcomes
- Handle partial failures gracefully
- Future versions will use Pydantic configuration objects

**Configuration Classes**:
- Must inherit from ``BaseBackendConfig``
- Must implement ``get_backend_class()`` method
- Should use Pydantic Field validators for parameter validation
- Should provide clear descriptions for all fields

The backend system's entry point architecture combined with Pydantic configurations makes ROMPY highly extensible while providing type safety and validation.
The combination of Pydantic configuration validation and entry point discovery makes ROMPY highly extensible while maintaining type safety and clear interfaces.

For comprehensive information on backend configurations, see :doc:`backend_configurations`.
