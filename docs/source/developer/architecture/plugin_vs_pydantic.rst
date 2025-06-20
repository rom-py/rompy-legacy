===============================================
Plugin Architecture vs Pydantic Union Patterns
===============================================

One of the most important architectural decisions in rompy is the use of two different extension patterns for different types of functionality. Understanding when and why to use each pattern is crucial for effective rompy development.

Overview
========

rompy uses two distinct approaches for extensibility:

1. **Pydantic Union Pattern** for model configurations (``CONFIG_TYPES``)
2. **Plugin Architecture Pattern** for execution backends (run, postprocess, pipeline)

This document explains the rationale behind this dual approach and provides guidance on when to use each pattern.

The Two Patterns
================

Pydantic Union Pattern (CONFIG_TYPES)
--------------------------------------

Model configurations use Pydantic's discriminated union pattern:

.. code-block:: python

    from typing import Union
    from pydantic import Field

    class ModelRun(RompyBaseModel):
        config: Union[CONFIG_TYPES] = Field(
            default_factory=BaseConfig,
            description="The configuration object",
            discriminator="model_type",
        )

This approach requires all configuration types to be known at import time and uses a ``model_type`` discriminator field to determine which configuration class to instantiate.

Plugin Architecture Pattern (Backends)
---------------------------------------

Execution backends use runtime plugin loading via entry points:

.. code-block:: python

    def run(self, backend: str = "local", **kwargs) -> bool:
        if backend not in RUN_BACKENDS:
            available = list(RUN_BACKENDS.keys())
            raise ValueError(f"Unknown run backend: {backend}...")

        backend_class = RUN_BACKENDS[backend]
        backend_instance = backend_class()
        return backend_instance.run(self, **kwargs)

This approach dynamically discovers available backends at runtime and instantiates them on demand.

Comparative Analysis
====================

Pydantic Union Approach
------------------------

**✅ Strengths**

*Strong Type Safety*
    Full Pydantic validation happens at model instantiation time, catching configuration errors early in the workflow.

    .. code-block:: python

        # Validation happens here - invalid configs rejected immediately
        model = ModelRun(config={"model_type": "swan", "grid": {...}})

*IDE Support & Developer Experience*
    Excellent autocomplete, type checking, and refactoring support in modern IDEs.

*Serialization & Reproducibility*
    Configuration is part of the model state and fully serializable, enabling reproducible science.

    .. code-block:: yaml

        # Complete model configuration saved as YAML
        config:
          model_type: swan
          grid:
            x0: 115.68
            y0: -32.76
            # ... full configuration preserved

*Schema Documentation*
    Clear, declarative schema with automatic documentation generation and validation rules.

*Immutability*
    Once instantiated, configurations are immutable, preventing accidental modification during execution.

**❌ Limitations**

*Static Discovery*
    All configuration types must be imported and available at startup time.

    .. code-block:: python

        # All configs loaded even if unused
        CONFIG_TYPES = load_entry_points("rompy.config")

*Tight Coupling*
    Adding new configuration types requires changes to the core model and package reinstallation.

*Memory Overhead*
    All configuration classes are loaded into memory regardless of usage.

*Installation Dependencies*
    New configuration types must be installed as package dependencies.

Plugin Architecture Approach
-----------------------------

**✅ Strengths**

*True Plugin Architecture*
    Backends can be added without any changes to core rompy code.

    .. code-block:: python

        # Third-party package can add backends via entry points
        [project.entry-points."rompy.run"]
        hpc_backend = "my_package.backends:HPCBackend"

*Lazy Loading*
    Only instantiate backends when actually needed, reducing memory footprint and startup time.

*Runtime Discovery*
    Dynamically discover what backends are available in the current environment.

    .. code-block:: python

        # Different backends available in different environments
        available_backends = list(RUN_BACKENDS.keys())
        # ['local'] on developer machine
        # ['local', 'slurm', 'kubernetes'] on HPC cluster

*Loose Coupling*
    Backends are completely independent and can have their own dependencies.

*Environment Flexibility*
    Same model configuration can use different backends based on deployment environment.

*Optional Dependencies*
    Graceful handling when optional backends aren't available (e.g., Docker not installed).

**❌ Limitations**

*Reduced Type Safety*
    Backend selection via strings means errors are only caught at execution time.

    .. code-block:: python

        # Error only discovered when run() is called
        model.run(backend="typo_backend")  # ValueError at runtime

*Late Validation*
    Backend availability and parameter validation happens during execution, not configuration.

*Discovery Challenges*
    Harder to know what backends are available during development and configuration.

*Complex Error Handling*
    More sophisticated error handling needed for missing backends and runtime failures.

*Non-Serializable State*
    Backend choice is not part of the serializable model configuration.

Why Different Patterns for Different Concerns?
===============================================

The architectural decision reflects the **fundamental difference in purpose** between these two types of extensibility:

Configuration vs Execution Separation
--------------------------------------

**Configuration Represents Domain Logic**

Model configurations encode scientific and mathematical knowledge:

- **What** physics to simulate (wave propagation, hydrodynamics)
- **Where** to simulate it (grid definition, boundaries)
- **When** to simulate it (time periods, forcing data)

This domain knowledge needs:

- **Strong validation** (incorrect physics parameters = invalid science)
- **Reproducibility** (same config = same results)
- **Documentation** (clear schema for scientific parameters)
- **Immutability** (configurations shouldn't change during execution)

**Execution Represents Infrastructure Concerns**

Execution backends handle operational and deployment concerns:

- **How** to run the model (local process, container, HPC queue)
- **Where** to run it (laptop, cluster, cloud)
- **With what resources** (CPU cores, memory, time limits)

This infrastructure knowledge needs:

- **Environment flexibility** (different options in different deployments)
- **Optional dependencies** (some backends may not be available)
- **Runtime adaptation** (choose backend based on current conditions)
- **Operational parameters** (that vary per execution, not per model)

Practical Examples
==================

Configuration Example (Pydantic Union)
---------------------------------------

Scientific parameters are validated and preserved:

.. code-block:: yaml

    # This represents scientific intent - must be validated and preserved
    config:
      model_type: swan
      grid:
        x0: 115.68      # Geographic coordinate - must be valid
        y0: -32.76      # Geographic coordinate - must be valid
        dx: 0.001       # Grid resolution - affects numerical accuracy
        dy: 0.001       # Grid resolution - affects numerical accuracy
      physics:
        friction: MAD   # Physics model choice - affects results
        friction_coeff: 0.1  # Physics parameter - must be scientifically valid

Any error in these parameters would produce scientifically invalid results, so they must be validated immediately.

Execution Example (Plugin Architecture)
----------------------------------------

Operational parameters vary by environment:

.. code-block:: python

    # Development environment
    model.run(
        backend="local",
        timeout=600,
        env_vars={"OMP_NUM_THREADS": "2"}
    )

    # Production HPC environment
    model.run(
        backend="slurm",
        partition="compute",
        nodes=4,
        time_limit="24:00:00",
        env_vars={"OMP_NUM_THREADS": "16"}
    )

    # Cloud deployment
    model.run(
        backend="kubernetes",
        image="rompy/swan:v1.2.3",
        resources={"cpu": "8", "memory": "32Gi"},
        node_selector={"instance-type": "compute-optimized"}
    )

The same scientific configuration runs in all environments, but with different operational parameters.

Design Patterns in Practice
============================

When to Use Pydantic Union Pattern
-----------------------------------

Use the Pydantic union pattern when extending rompy with:

**✅ Model Configuration Types**
    New model types (SCHISM, XBeach, FVCOM) that define scientific computation.

**✅ Grid Definitions**
    New grid types that define spatial discretization approaches.

**✅ Forcing Data Specifications**
    New ways to specify input data (wind, waves, boundaries) with validation requirements.

**✅ Physics Parameterizations**
    New physics options that require parameter validation and documentation.

Example - Adding a new model type:

.. code-block:: python

    class XBeachConfig(BaseConfig):
        """XBeach model configuration."""
        model_type: Literal["xbeach"] = "xbeach"

        # Validated scientific parameters
        grid: XBeachGrid
        physics: XBeachPhysics
        outputs: XBeachOutputs

        # Strong validation rules
        @validator('physics')
        def validate_physics_consistency(cls, v, values):
            # Ensure physics parameters are scientifically consistent
            return v

When to Use Plugin Architecture Pattern
----------------------------------------

Use the plugin architecture pattern when extending rompy with:

**✅ Execution Environments**
    New ways to run models (HPC schedulers, cloud platforms, containers).

**✅ Output Processing**
    New analysis, visualization, or data transformation capabilities.

**✅ Workflow Orchestration**
    New ways to coordinate multi-stage model workflows.

**✅ Integration Points**
    Connections to external systems (databases, monitoring, notifications).

Example - Adding a new execution backend:

.. code-block:: python

    class SlurmBackend:
        """Execute models via SLURM job scheduler."""

        def run(self, model_run, partition="compute", nodes=1, **kwargs):
            """Submit model to SLURM queue."""
            # Generate SLURM job script
            job_script = self._create_slurm_script(
                model_run, partition, nodes, **kwargs
            )

            # Submit job and monitor execution
            job_id = self._submit_job(job_script)
            return self._wait_for_completion(job_id)

Register via entry points:

.. code-block:: toml

    [project.entry-points."rompy.run"]
    slurm = "rompy_hpc.backends:SlurmBackend"

Best Practices
==============

For Configuration Extensions (Pydantic)
----------------------------------------

**Comprehensive Validation**
    Implement validators that check scientific and mathematical consistency.

    .. code-block:: python

        @validator('grid_resolution')
        def validate_resolution(cls, v):
            if v <= 0:
                raise ValueError("Grid resolution must be positive")
            if v > 0.1:
                warnings.warn("Very coarse resolution may affect accuracy")
            return v

**Clear Documentation**
    Provide detailed docstrings explaining scientific meaning and valid ranges.

**Immutable Design**
    Avoid mutable state that could change during model execution.

**Schema Versioning**
    Plan for configuration schema evolution and backward compatibility.

For Backend Extensions (Plugin)
--------------------------------

**Robust Error Handling**
    Handle missing dependencies and environment issues gracefully.

    .. code-block:: python

        def run(self, model_run, **kwargs):
            try:
                return self._execute_backend(model_run, **kwargs)
            except ImportError as e:
                raise RuntimeError(f"Backend dependencies not available: {e}")
            except Exception as e:
                logger.exception(f"Backend execution failed: {e}")
                return False

**Environment Detection**
    Check if the backend can run in the current environment.

**Parameter Validation**
    Validate backend-specific parameters at execution time.

**Resource Cleanup**
    Ensure proper cleanup of resources on success and failure.

Conclusion
==========

The dual extension pattern in rompy reflects a sophisticated understanding of different types of extensibility requirements:

- **Domain extensions** (configurations) need type safety, validation, and reproducibility
- **Infrastructure extensions** (backends) need flexibility, optional loading, and environment adaptation

This architectural decision enables rompy to maintain scientific rigor while supporting diverse computational environments. When extending rompy, carefully consider whether your extension represents domain knowledge (use Pydantic) or infrastructure concerns (use plugins).

The pattern demonstrates that **different types of extensibility have different requirements**, and a well-designed system should use the most appropriate mechanism for each type of extension rather than forcing everything into a single pattern.

Further Reading
===============

- :doc:`../extending/custom_backends` - Practical guide to creating new backends
- :doc:`../extending/custom_models` - Guide to adding new model configurations
- :doc:`../api_design/entry_points` - Technical details on the entry point system
- :doc:`configuration_patterns` - Deep dive into configuration design patterns
