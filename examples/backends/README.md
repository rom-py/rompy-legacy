# Backend Examples

This directory contains examples demonstrating how to use ROMPY's backend configuration system to execute models in different environments.

## Overview

ROMPY uses Pydantic-based backend configurations to provide type-safe, validated execution parameters for different environments. This system enables precise control over model execution while maintaining flexibility and extensibility.

## Available Examples

### 1. Basic Local Run (`01_basic_local_run.py`)
Demonstrates the simplest use case:
- Local execution with `LocalConfig`
- Basic timeout and command configuration
- No-op postprocessing

### 2. Docker Run (`02_docker_run.py`)
Shows Docker container execution:
- Using pre-built Docker images
- Volume mounting for data access
- Environment variable configuration
- Resource limits (CPU, memory)

### 3. Custom Postprocessor (`03_custom_postprocessor.py`)
Illustrates custom postprocessing:
- Creating custom postprocessor classes
- Processing model outputs after execution
- Error handling and result reporting

### 4. Complete Workflow (`04_complete_workflow.py`)
Demonstrates a full workflow:
- Model execution with local backend
- Custom postprocessing with file analysis
- Comprehensive logging and error handling

## Backend Configuration Types

### LocalConfig
For execution on the local system:
```python
from rompy.backends import LocalConfig

config = LocalConfig(
    timeout=3600,  # 1 hour
    command="python run_model.py",
    env_vars={"OMP_NUM_THREADS": "4"},
    shell=True,
    capture_output=True
)
```

### DockerConfig
For execution in Docker containers:
```python
from rompy.backends import DockerConfig

config = DockerConfig(
    image="python:3.9-slim",
    cpu=2,
    memory="2g",
    timeout=7200,
    volumes=["/data:/app/data:rw"],
    env_vars={"MODEL_CONFIG": "production"}
)
```

## Running the Examples

Each example can be run directly:

```bash
# Basic local execution
python 01_basic_local_run.py

# Docker execution (requires Docker)
python 02_docker_run.py

# Custom postprocessing
python 03_custom_postprocessor.py

# Complete workflow
python 04_complete_workflow.py
```

## Key Features

- **Type Safety**: All configurations are validated using Pydantic
- **IDE Support**: Full autocompletion and inline documentation
- **Flexibility**: Easy to extend with custom backends and postprocessors
- **Error Handling**: Clear validation errors and execution feedback
- **Serialization**: Configurations can be saved/loaded as YAML/JSON

## Configuration Validation

Backend configurations provide comprehensive validation:
- Timeout values must be between 60 and 86400 seconds
- Working directories must exist if specified
- Docker image names must follow valid conventions
- Volume mounts must reference existing host paths

## Best Practices

1. **Set appropriate timeouts** based on your model complexity
2. **Use environment variables** for sensitive configuration
3. **Validate configurations** before execution
4. **Handle errors gracefully** in your postprocessors
5. **Use resource limits** appropriately in Docker configurations

## Output Structure

All examples create output in the `./output` directory with the following structure:
```
output/
├── <run_id>/
│   ├── INPUT              # Generated model input file
│   ├── datasets/          # Placeholder for input datasets
│   ├── outputs/           # Placeholder for model outputs
│   └── <additional files> # Any files created during execution
```

## Extending the Examples

You can extend these examples by:
- Creating custom backend configurations
- Implementing custom postprocessors
- Adding new execution environments
- Integrating with workflow orchestration systems

For more detailed information, see the [Backend Configurations documentation](../../docs/source/backend_configurations.rst).