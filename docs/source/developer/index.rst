====================
Developer Guide
====================

Welcome to the rompy developer documentation. This section provides detailed technical information about rompy's architecture, design patterns, and extension mechanisms.

rompy is designed with extensibility and maintainability in mind, featuring a modular architecture that separates concerns between configuration, execution, and processing. Understanding these architectural decisions will help you contribute effectively to the project or extend it for your specific needs.

.. toctree::
   :maxdepth: 2
   :caption: Architecture & Design

   architecture/selection_patterns
   architecture/configuration_patterns
   architecture/extensibility_guide

.. toctree::
   :maxdepth: 2
   :caption: Contributing

   contributing/development_setup
   contributing/coding_standards
   contributing/testing_guidelines

.. toctree::
   :maxdepth: 2
   :caption: Extending rompy

   extending/custom_backends
   extending/custom_models
   extending/custom_processors

.. toctree::
   :maxdepth: 2
   :caption: API Design

   api_design/entry_points
   api_design/base_classes
   api_design/error_handling

Overview
========

rompy's architecture is built around several key principles:

**Separation of Concerns**
   Configuration (what to compute) is separated from execution (how to compute) and processing (what to do with results).

**Plugin Architecture**
   Core functionality can be extended through entry points without modifying the base code.

**Type Safety**
   Pydantic models provide strong validation for configuration while maintaining flexibility for runtime concerns.

**Reproducibility**
   Model configurations are fully serializable and version-controlled, ensuring reproducible science.

**Environment Agnostic**
   The same model configuration can be executed in different environments (local, HPC, cloud, containers).

Key Architectural Decisions
===========================

The rompy codebase makes several important architectural decisions that affect how you should approach development:

1. **Dual Extension Patterns**: Different extension mechanisms for different concerns (Pydantic unions for configs, entry points for backends)

2. **Configuration Immutability**: Model configurations are treated as immutable scientific artifacts

3. **Late Binding**: Execution backends are resolved at runtime to support environment-specific deployments

4. **Composable Workflows**: Pipeline architecture allows mixing and matching of different execution and processing strategies

Quick Start for Developers
===========================

If you're new to rompy development, start here:

1. **Understanding the Architecture**: Read :doc:`architecture/selection_patterns` to understand the core design patterns

2. **Setting Up Development**: Follow :doc:`contributing/development_setup` to get your environment ready

3. **Extending rompy**: Check :doc:`extending/custom_backends` for adding new functionality

4. **Contributing**: Review :doc:`contributing/coding_standards` before submitting code

Design Philosophy
=================

rompy is designed around the principle that **configuration should be declarative and execution should be imperative**. This means:

- **What to compute** (model physics, grids, forcing) is declared in strongly-typed, validated configuration objects
- **How to compute** (local vs cloud, serial vs parallel) is handled by pluggable execution backends
- **What to do with results** (analysis, visualization, archiving) is handled by composable processing pipelines

This separation allows the same scientific model configuration to be executed in vastly different computational environments while maintaining reproducibility and traceability.

Getting Help
============

- **Architecture Questions**: Check the architecture guides in this developer section
- **Implementation Help**: See the extending guides for practical examples
- **Code Issues**: Review the contributing guidelines for debugging and testing approaches
- **Community**: Join discussions in the project's issue tracker or mailing list

The developer documentation is continuously evolving. If you find gaps or have suggestions for improvement, please contribute back to help other developers.
