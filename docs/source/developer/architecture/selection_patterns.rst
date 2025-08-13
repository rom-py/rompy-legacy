============================
Component Selection Patterns
============================

One of the most important architectural decisions in rompy is the use of two different selection patterns for different types of functionality. Both patterns use entry points for discovery, but differ in when and how selection occurs. Understanding when and why to use each pattern is crucial for effective rompy development.

Overview
========

rompy uses two distinct approaches for component selection:

1. **Pydantic Discriminated Union Pattern** for model configurations (``CONFIG_TYPES``)
2. **Runtime String Selection Pattern** for execution backends (run, postprocess, pipeline)

Both patterns use Python entry points for plugin discovery, but serve fundamentally different purposes. This document explains the rationale behind this dual approach and provides guidance on when to use each pattern.

The Two Patterns
================

Both patterns use Python entry points for plugin discovery, but differ in when and how selection occurs.

Pydantic Discriminated Union Pattern (CONFIG_TYPES)
----------------------------------------------------

Model configurations use entry points to build a discriminated union:

.. code-block:: python

    from typing import Union
    from pydantic import Field
    from rompy.utils import load_entry_points

    # Load config types from entry points at import time
    CONFIG_TYPES = load_entry_points("rompy.config")

    class ModelRun(RompyBaseModel):
        config: Union[CONFIG_TYPES] = Field(
            default_factory=BaseConfig,
            description="The configuration object",
            discriminator="model_type",  # Selection via discriminator field
        )

Selection happens at **model instantiation time** via the ``model_type`` discriminator field in the configuration data.

Runtime String Selection Pattern (Backends)
--------------------------------------------

Execution backends use entry points for runtime selection:

.. code-block:: python

    from rompy.utils import load_entry_points

    # Load backends from entry points at import time
    def _load_backends():
        run_backends = {}
        for backend in load_entry_points("rompy.run"):
            name = backend.__name__.lower().replace('runbackend', '')
            run_backends[name] = backend
        return run_backends

    RUN_BACKENDS = _load_backends()

    def run(self, backend: str = "local", **kwargs) -> bool:
        # Selection happens at execution time via string parameter
        backend_class = RUN_BACKENDS[backend]
        backend_instance = backend_class()
        return backend_instance.run(self, **kwargs)

Selection happens at **execution time** via string parameters passed to methods.

Comparative Analysis
====================

Pydantic Discriminated Union Approach
--------------------------------------

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

*Plugin Support*
    Uses entry points for discovery, allowing third-party configuration types.

    .. code-block:: python

        # Third-party configs discovered via entry points
        CONFIG_TYPES = load_entry_points("rompy.config")

**❌ Limitations**

*Selection Timing*
    Configuration type must be known at model instantiation time.

*State Coupling*
    Configuration choice becomes part of persistent model state.

*Validation Completeness*
    All possible configurations must be validated upfront, even if unused.

Runtime String Selection Approach
----------------------------------

**✅ Strengths**

*Execution-Time Flexibility*
    Backend choice can be made based on runtime conditions and environment.

    .. code-block:: python

        # Different backends for different environments
        backend = "docker" if has_docker() else "local"
        model.run(backend=backend)

*Operational Independence*
    Backend choice is independent of scientific configuration.

*Environment Adaptation*
    Same model configuration can use different backends based on deployment environment.

    .. code-block:: python

        # Same config, different execution strategies
        model.run(backend="local")     # Development
        model.run(backend="slurm")     # HPC cluster
        model.run(backend="k8s")       # Cloud deployment

*Plugin Support*
    Uses entry points for discovery, allowing third-party backends.

    .. code-block:: python

        # Third-party backends discovered via entry points
        RUN_BACKENDS = dict(load_entry_points("rompy.run"))

*Lazy Instantiation*
    Only instantiate backends when actually needed.

*Optional Dependencies*
    Graceful handling when optional backends aren't available.

**❌ Limitations**

*Reduced Type Safety*
    Backend selection via strings means errors are only caught at execution time.

    .. code-block:: python

        # Error only discovered when run() is called
        model.run(backend="typo_backend")  # ValueError at runtime

*Late Validation*
    Backend availability and parameter validation happens during execution, not configuration.

*Non-Serializable Choice*
    Backend choice is not part of the serializable model configuration.

*Discovery Complexity*
    Harder to know what backends are available during development.

Why Different Patterns for Different Concerns?
===============================================

The architectural decision reflects the **fundamental difference in purpose** between these two types of selection, despite both using entry points:

State vs Behavior Separation
-----------------------------

**Configuration Represents Persistent Domain State**

Model configurations encode scientific and mathematical knowledge that must be preserved:

- **What** physics to simulate (wave propagation, hydrodynamics)
- **Where** to simulate it (grid definition, boundaries)
- **When** to simulate it (time periods, forcing data)

This domain state needs:

- **Strong validation** (incorrect physics parameters = invalid science)
- **Reproducibility** (same config = same results)
- **Serialization** (configurations must be saveable and shareable)
- **Immutability** (configurations shouldn't change during execution)
- **Early validation** (catch errors before expensive computation starts)

**Execution Represents Runtime Behavior**

Execution backends handle operational and deployment behavior:

- **How** to run the model (local process, container, HPC queue)
- **Where** to run it (laptop, cluster, cloud)
- **With what resources** (CPU cores, memory, time limits)

This runtime behavior needs:

- **Environment flexibility** (different options in different deployments)
- **Late binding** (choose backend based on current conditions)
- **Optional availability** (some backends may not be installed)
- **Operational parameters** (that vary per execution, not per model)
- **Ephemeral choice** (backend selection shouldn't be saved with scientific config)

Practical Examples
==================

Configuration Example (Discriminated Union)
--------------------------------------------

Scientific parameters are validated, serialized, and preserved:

.. code-block:: yaml

    # This represents scientific intent - must be validated and preserved
    config:
      model_type: swan  # ← Discriminator field for Pydantic union selection
      grid:
        x0: 115.68      # Geographic coordinate - must be valid
        y0: -32.76      # Geographic coordinate - must be valid
        dx: 0.001       # Grid resolution - affects numerical accuracy
        dy: 0.001       # Grid resolution - affects numerical accuracy
      physics:
        friction: MAD   # Physics model choice - affects results
        friction_coeff: 0.1  # Physics parameter - must be scientifically valid

The ``model_type`` field triggers Pydantic's discriminated union to select the correct configuration class. Any error in these parameters would produce scientifically invalid results, so they must be validated at instantiation time.

Execution Example (Runtime String Selection)
---------------------------------------------

Operational parameters vary by environment and are not serialized:

.. code-block:: python

    # Same config object, different execution environments
    config_data = load_yaml("scientific_config.yaml")  # Contains model_type discriminator
    model = ModelRun(**config_data)                     # Pydantic selects config class

    # Development environment - runtime string selection
    model.run(
        backend="local",        # ← String parameter for runtime selection
        timeout=600,
        env_vars={"OMP_NUM_THREADS": "2"}
    )

    # Production HPC environment - same config, different backend
    model.run(
        backend="slurm",        # ← Different string, same config
        partition="compute",
        nodes=4,
        time_limit="24:00:00",
        env_vars={"OMP_NUM_THREADS": "16"}
    )

    # Cloud deployment - same config, cloud backend
    model.run(
        backend="kubernetes",   # ← Runtime choice, not saved
        image="rompy/swan:v1.2.3",
        resources={"cpu": "8", "memory": "32Gi"}
    )

The same scientific configuration (with its ``model_type`` discriminator) runs in all environments, but with different runtime backend selections that are not part of the serializable state.

Design Patterns in Practice
============================

When to Use Discriminated Union Pattern
----------------------------------------

Use the discriminated union pattern when extending rompy with components that need to be:

**✅ Part of Serializable State**
    Components that must be saved, shared, and reproduced exactly.

**✅ Validated at Instantiation**
    Components where early validation prevents expensive failures later.

**✅ Scientifically Critical**
    Components where incorrect parameters lead to invalid scientific results.

**✅ Model Configuration Types**
    New model types (SCHISM, XBeach, FVCOM) that define scientific computation.

**✅ Grid Definitions**
    New grid types that define spatial discretization approaches.

**✅ Physics Parameterizations**
    New physics options that require parameter validation and documentation.

Example - Adding a new model type with entry point registration:

.. code-block:: python

    class XBeachConfig(BaseConfig):
        """XBeach model configuration."""
        model_type: Literal["xbeach"] = "xbeach"  # Discriminator field

        # Validated scientific parameters
        grid: XBeachGrid
        physics: XBeachPhysics
        outputs: XBeachOutputs

        # Strong validation rules
        @validator('physics')
        def validate_physics_consistency(cls, v, values):
            # Ensure physics parameters are scientifically consistent
            return v

.. code-block:: toml

    # Register via entry points for discovery
    [project.entry-points."rompy.config"]
    xbeach = "mypackage.config:XBeachConfig"

When to Use Runtime String Selection Pattern
---------------------------------------------

Use the runtime string selection pattern when extending rompy with components that are:

**✅ Environment-Specific**
    Components that vary based on where the code is running.

**✅ Operationally Focused**
    Components that handle execution, processing, or infrastructure concerns.

**✅ Optional Dependencies**
    Components that may not be available in all environments.

**✅ Execution Environments**
    New ways to run models (HPC schedulers, cloud platforms, containers).

**✅ Output Processing**
    New analysis, visualization, or data transformation capabilities.

**✅ Workflow Orchestration**
    New ways to coordinate multi-stage model workflows.

Example - Adding a new execution backend with entry point registration:

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

.. code-block:: toml

    # Register via entry points for discovery
    [project.entry-points."rompy.run"]
    slurm = "rompy_hpc.backends:SlurmBackend"

Best Practices
==============

For Discriminated Union Extensions (Configuration)
---------------------------------------------------

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

**Entry Point Registration**
    Register new configuration types via entry points for automatic discovery.

    .. code-block:: toml

        [project.entry-points."rompy.config"]
        mymodel = "mypackage.config:MyModelConfig"

For Runtime String Selection Extensions (Backends)
---------------------------------------------------

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

**Entry Point Registration**
    Register new backends via entry points for automatic discovery.

    .. code-block:: toml

        [project.entry-points."rompy.run"]
        mybackend = "mypackage.backends:MyBackend"

Conclusion
==========

The dual selection pattern in rompy reflects a sophisticated understanding of different types of component selection requirements:

- **State-based selection** (configurations) needs early validation, serialization, and reproducibility
- **Behavior-based selection** (backends) needs late binding, environment adaptation, and optional availability

Both patterns use entry points for plugin discovery, but differ fundamentally in **when selection occurs** and **what gets serialized**:

- **Configurations**: Selected at instantiation time via discriminator fields, become part of persistent state
- **Backends**: Selected at execution time via string parameters, remain ephemeral operational choices

This architectural decision enables rompy to maintain scientific rigor while supporting diverse computational environments. When extending rompy, carefully consider whether your extension represents:

- **Persistent domain state** → Use discriminated unions with entry point discovery
- **Runtime behavior choice** → Use string selection with entry point discovery

The pattern demonstrates that **the same plugin discovery mechanism can serve different selection patterns**, and a well-designed system should choose the selection timing and state management approach that best fits the component's purpose.

Further Reading
===============

- :doc:`../extending/custom_backends` - Practical guide to creating new backends
- :doc:`../extending/custom_models` - Guide to adding new model configurations
- :doc:`../api_design/entry_points` - Technical details on the entry point system
- :doc:`configuration_patterns` - Deep dive into configuration design patterns
