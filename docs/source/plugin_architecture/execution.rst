Execution and Output Processing Plugin Architecture
====================================================

Rompy features a flexible plugin-based architecture that allows for extensible model execution and output processing. The system uses Python entry points to automatically discover and load backends, making it easy to extend with custom implementations.

Overview
--------

The plugin architecture is built around three main categories:

1. **Run Backends** (``rompy.run``): Handle model execution in different environments
2. **Postprocessors** (``rompy.postprocess``): Handle model output analysis and transformation
3. **Pipeline Backends** (``rompy.pipeline``): Orchestrate complete model workflows

Each category uses Python entry points for automatic discovery and loading, allowing third-party packages to easily extend rompy's capabilities.

Run Backends
------------

Run backends are responsible for executing models in different environments. They all implement a common interface with a ``run()`` method.

Built-in Run Backends
~~~~~~~~~~~~~~~~~~~~~

Local Backend
^^^^^^^^^^^^^

The ``local`` backend executes models directly on the local system:

.. code-block:: python

    # Basic local execution
    success = model.run(backend="local")

    # With custom command
    success = model.run(
        backend="local",
        command="./my_model_executable",
        env_vars={"OMP_NUM_THREADS": "4"},
        timeout=3600
    )

Docker Backend
^^^^^^^^^^^^^^

The ``docker`` backend executes models inside Docker containers:

.. code-block:: python

    # Using pre-built image
    success = model.run(
        backend="docker",
        image="rompy/schism:latest",
        executable="/usr/local/bin/schism",
        cpu=4,
        volumes=["./data:/data:ro"],
        env_vars={"MODEL_CONFIG": "production"}
    )

    # Building from Dockerfile
    success = model.run(
        backend="docker",
        dockerfile="./docker/Dockerfile",
        build_args={"MODEL_VERSION": "1.0.0"},
        executable="/usr/local/bin/model",
        mpiexec="mpiexec",
        cpu=8
    )

Custom Run Backends
~~~~~~~~~~~~~~~~~~~

You can create custom run backends by implementing the run interface:

.. code-block:: python

    class CustomRunBackend:
        """Custom run backend example."""

        def run(self, model_run, **kwargs):
            """Execute the model run.

            Args:
                model_run: The ModelRun instance
                **kwargs: Backend-specific parameters

            Returns:
                bool: True if successful, False otherwise
            """
            try:
                # Generate model inputs
                model_run.generate()

                # Custom execution logic here
                return self._execute_custom_logic(model_run, **kwargs)

            except Exception as e:
                logger.exception(f"Custom backend failed: {e}")
                return False

Register custom backends via entry points in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."rompy.run"]
    custom = "mypackage.backends:CustomRunBackend"

Postprocessors
--------------

Postprocessors handle analysis and transformation of model outputs. They implement a ``process()`` method that returns a dictionary with results.

Built-in Postprocessors
~~~~~~~~~~~~~~~~~~~~~~~

No-op Processor
^^^^^^^^^^^^^^^

The ``noop`` processor provides basic validation without processing:

.. code-block:: python

    # Basic validation
    results = model.postprocess(processor="noop")

    # With custom validation
    results = model.postprocess(
        processor="noop",
        validate_outputs=True,
        output_dir="./custom_output"
    )

Custom Postprocessors
~~~~~~~~~~~~~~~~~~~~~

Create custom postprocessors by implementing the process interface:

.. code-block:: python

    class AnalysisPostprocessor:
        """Custom postprocessor for model analysis."""

        def process(self, model_run, **kwargs):
            """Process model outputs.

            Args:
                model_run: The ModelRun instance
                **kwargs: Processor-specific parameters

            Returns:
                dict: Processing results
            """
            try:
                output_dir = Path(model_run.output_dir) / model_run.run_id

                # Custom analysis logic
                metrics = self._calculate_metrics(output_dir)
                plots = self._generate_plots(output_dir)

                return {
                    "success": True,
                    "metrics": metrics,
                    "plots": plots,
                    "message": "Analysis completed successfully"
                }

            except Exception as e:
                return {
                    "success": False,
                    "error": str(e),
                    "message": f"Analysis failed: {e}"
                }

Register via entry points:

.. code-block:: toml

    [project.entry-points."rompy.postprocess"]
    analysis = "mypackage.processors:AnalysisPostprocessor"

Pipeline Backends
-----------------

Pipeline backends orchestrate the complete model workflow from input generation through execution to output processing.

Built-in Pipeline Backends
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Local Pipeline
^^^^^^^^^^^^^^

The ``local`` pipeline executes all stages locally:

.. code-block:: python

    # Basic pipeline
    results = model.pipeline(pipeline_backend="local")

    # With custom backends
    results = model.pipeline(
        pipeline_backend="local",
        run_backend="docker",
        processor="analysis",
        run_kwargs={"image": "rompy/model:latest", "cpu": 4},
        process_kwargs={"create_plots": True},
        cleanup_on_failure=True
    )

Custom Pipeline Backends
~~~~~~~~~~~~~~~~~~~~~~~~~

Create custom pipeline backends for distributed or cloud execution:

.. code-block:: python

    class CloudPipelineBackend:
        """Pipeline backend for cloud execution."""

        def execute(self, model_run, **kwargs):
            """Execute the complete pipeline.

            Args:
                model_run: The ModelRun instance
                **kwargs: Pipeline-specific parameters

            Returns:
                dict: Pipeline execution results
            """
            results = {
                "success": False,
                "run_id": model_run.run_id,
                "stages_completed": []
            }

            try:
                # Stage 1: Generate inputs
                model_run.generate()
                results["stages_completed"].append("generate")

                # Stage 2: Submit to cloud
                job_id = self._submit_cloud_job(model_run, **kwargs)
                results["job_id"] = job_id
                results["stages_completed"].append("submit")

                # Stage 3: Wait for completion
                self._wait_for_completion(job_id)
                results["stages_completed"].append("execute")

                # Stage 4: Download and process results
                outputs = self._download_results(job_id)
                processed = self._process_outputs(outputs, **kwargs)
                results["outputs"] = processed
                results["stages_completed"].append("postprocess")

                results["success"] = True
                return results

            except Exception as e:
                results["error"] = str(e)
                return results

Best Practices
--------------

Error Handling
~~~~~~~~~~~~~~

- Always wrap main logic in try-catch blocks
- Return appropriate boolean/dict responses
- Log errors with sufficient detail for debugging
- Clean up resources on failure when possible

Parameter Validation
~~~~~~~~~~~~~~~~~~~~

- Validate required parameters early
- Provide clear error messages for invalid inputs
- Use type hints for better IDE support
- Document all parameters in docstrings

Logging
~~~~~~~

- Use structured logging with appropriate levels
- Include run_id and context in log messages
- Log progress for long-running operations
- Avoid logging sensitive information

Resource Management
~~~~~~~~~~~~~~~~~~~

- Clean up temporary files and directories
- Handle timeouts gracefully
- Implement proper cancellation mechanisms
- Monitor resource usage for long-running processes

Testing
~~~~~~~

- Write unit tests for all backend methods
- Mock external dependencies (Docker, cloud APIs)
- Test error conditions and edge cases
- Include integration tests where appropriate

Examples
--------

Complete examples demonstrating the plugin architecture can be found in the ``examples/backends/`` directory:

- ``01_basic_local_run.py``: Simple local execution
- ``02_docker_run.py``: Docker container execution
- ``03_custom_postprocessor.py``: Custom output processing
- ``04_complete_workflow.py``: End-to-end custom workflow

For interactive examples, see the ``notebooks/backend_examples.ipynb`` notebook.

API Reference
-------------

.. automodule:: rompy.run
   :members:
   :undoc-members:

.. automodule:: rompy.postprocess
   :members:
   :undoc-members:

.. automodule:: rompy.pipeline
   :members:
   :undoc-members:
