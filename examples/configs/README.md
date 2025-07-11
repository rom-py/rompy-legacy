# ROMPY Backend Configuration Examples

This directory contains example configuration files for ROMPY backend systems. These YAML files demonstrate how to configure local and Docker backends for various use cases.

## Configuration Files

### Backend Configurations

- **`local_backend.yml`** - Single-document local backend configuration
- **`docker_backend.yml`** - Single-document Docker backend configuration
- **`local_backend_examples.yml`** - Multi-document local backend examples
- **`docker_backend_examples.yml`** - Multi-document Docker backend examples
- **`pipeline_config.yml`** - Complete pipeline configuration examples
- **`validate_configs.py`** - Validation script for configuration files

## Configuration Format

ROMPY uses a unified configuration format that works for all use cases:

```yaml
# Backend type
backend_type: local  # or 'docker'

# Configuration parameters
timeout: 3600
command: "python run_model.py"
env_vars:
  OMP_NUM_THREADS: "4"
  MODEL_CONFIG: "production"
```

This format works for:
- CLI validation
- Pipeline usage
- Programmatic usage

## Using Configuration Files

### CLI Commands

```bash
# Validate configuration files
rompy backends validate local_backend.yml
rompy backends validate docker_backend.yml

# List available backends
rompy backends list

# Show backend schema
rompy backends schema --backend-type local
rompy backends schema --backend-type docker

# Create new configuration template
rompy backends create --backend-type local --output my_config.yml
```

### Pipeline Usage

```bash
# Run with backend configuration
rompy run --config model_config.yml --backend-config local_backend.yml

# Run complete pipeline
rompy pipeline --config pipeline_config.yml
```

## Configuration Options

### Local Backend Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `timeout` | int | 3600 | Maximum execution time (seconds) |
| `command` | string | null | Custom shell command to run |
| `shell` | bool | true | Execute through shell |
| `capture_output` | bool | true | Capture stdout/stderr |
| `working_dir` | string | null | Working directory |
| `env_vars` | dict | {} | Environment variables |

### Docker Backend Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `image` | string | null | Docker image to use |
| `dockerfile` | string | null | Path to Dockerfile |
| `timeout` | int | 3600 | Maximum execution time (seconds) |
| `cpu` | int | 1 | CPU cores to allocate |
| `memory` | string | null | Memory limit (e.g., "2g") |
| `mpiexec` | string | "" | MPI command |
| `volumes` | list | [] | Volume mounts |
| `env_vars` | dict | {} | Environment variables |
| `user` | string | "root" | Container user |
| `remove_container` | bool | true | Remove after execution |

## Example Configurations

### Local Backend

```yaml
backend_type: local
timeout: 7200
command: "python run_model.py"
env_vars:
  OMP_NUM_THREADS: "4"
  MODEL_CONFIG: "production"
```

### Docker Backend

```yaml
backend_type: docker
image: "python:3.9-slim"
timeout: 7200
cpu: 4
memory: "2g"
volumes:
  - "/data:/app/data:rw"
env_vars:
  MODEL_THREADS: "4"
```

### Pipeline Configuration

```yaml
pipeline_backend: local

model_run:
  run_id: "example_run"
  output_dir: "./output"
  period:
    start: "2023-01-01T00:00:00"
    end: "2023-01-02T00:00:00"
    interval: "1H"
  config:
    model_type: "swan"

run_backend:
  backend_type: local
  timeout: 3600
  env_vars:
    OMP_NUM_THREADS: "4"

postprocessing:
  processor: "noop"
```

## Common Use Cases

### Development and Testing

```yaml
backend_type: local
timeout: 1800  # 30 minutes
env_vars:
  ENV: "development"
  LOG_LEVEL: "DEBUG"
```

### Production Runs

```yaml
backend_type: docker
image: "my-model:latest"
timeout: 14400  # 4 hours
cpu: 8
memory: "4g"
env_vars:
  MODEL_CONFIG: "production"
```

### High-Performance Computing

```yaml
backend_type: docker
image: "hpc-model:latest"
timeout: 86400  # 24 hours
cpu: 32
memory: "64g"
mpiexec: "mpirun -np 32"
```

## Environment Variables

Configuration files support environment variable substitution:

```yaml
backend_type: local
timeout: ${ROMPY_TIMEOUT:-3600}
env_vars:
  MODEL_CONFIG: ${MODEL_CONFIG:-production}
  DATA_DIR: ${DATA_DIR:-/data}
```

## Validation

Always validate configuration files before use:

```bash
# Validate specific configuration
rompy backends validate local_backend.yml

# Validate all configurations in directory
python validate_configs.py
```

## Best Practices

1. **Set appropriate timeouts** based on expected run duration
2. **Use environment variables** for configuration that changes between environments
3. **Validate configurations** before deployment
4. **Use non-root users** in Docker containers when possible
5. **Mount only necessary directories** to containers
6. **Enable cleanup options** in production to save disk space

## Troubleshooting

### Common Issues

1. **Configuration validation errors**: Check YAML syntax and required fields
2. **Docker build failures**: Ensure Dockerfile paths are correct
3. **Volume mount errors**: Verify host paths exist and have correct permissions
4. **Timeout issues**: Increase timeout values for long-running models
5. **Resource limits**: Ensure sufficient system resources are available

### Getting Help

- Use `rompy backends --help` for backend management commands
- Use `rompy backends schema --backend-type <type>` to see all available options
- Check individual configuration files for comprehensive examples
- Run `python validate_configs.py` to check all configurations

## Creating Custom Configurations

1. **Start with a template**: Copy and modify existing configuration files
2. **Validate early**: Use `rompy backends validate` frequently during development
3. **Test with short runs**: Use minimal configurations first
4. **Scale gradually**: Increase resources and complexity incrementally

### Quick Start Template

```yaml
backend_type: local  # or 'docker'
timeout: 3600
command: "your_command_here"
env_vars:
  YOUR_VAR: "your_value"
```

## File Structure

```
rompy/examples/configs/
├── local_backend.yml              # Simple local configuration
├── docker_backend.yml             # Simple Docker configuration
├── local_backend_examples.yml     # Comprehensive local examples
├── docker_backend_examples.yml    # Comprehensive Docker examples
├── pipeline_config.yml            # Pipeline workflow examples
├── validate_configs.py            # Validation script
└── README.md                      # This documentation
```

This clean, focused configuration system provides everything needed to configure ROMPY backends for any use case, from simple local execution to complex distributed workflows.