"""
ROMPY Command Line Interface

This module provides the command-line interface for ROMPY.
"""

import json
import sys
import os
import warnings
from importlib.metadata import entry_points
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import click
import yaml

from rompy.backends import LocalConfig, DockerConfig, BackendConfig
from rompy.model import ModelRun, RUN_BACKENDS, POSTPROCESSORS, PIPELINE_BACKENDS
from rompy.core.logging import get_logger, LoggingConfig, LogLevel, LogFormat

# Initialize the logger
logger = get_logger(__name__)

# Get installed entry points
installed = entry_points(group="rompy.config").names


def configure_logging(
    verbosity: int = 0,
    log_dir: Optional[str] = None,
    simple_logs: bool = False,
    ascii_only: bool = False,
    show_warnings: bool = False,
) -> None:
    """Configure logging based on verbosity level and other options.

    This function configures the logging system using the LoggingConfig class.

    Args:
        verbosity: 0=WARNING, 1=INFO, 2=DEBUG
        log_dir: Directory to save log files
        simple_logs: Use simple log format without timestamps and module names
        ascii_only: Use ASCII-only characters in output
        show_warnings: Whether to show Python warnings
    """
    # Get the singleton instance of LoggingConfig
    logging_config = LoggingConfig()

    # Map verbosity to log level
    log_level = LogLevel.WARNING
    if verbosity >= 1:
        log_level = LogLevel.INFO
    if verbosity >= 2:
        log_level = LogLevel.DEBUG

    # Determine log format
    log_format = LogFormat.SIMPLE if simple_logs else LogFormat.VERBOSE

    # Prepare update parameters
    update_params = {"level": log_level, "format": log_format, "use_ascii": ascii_only}

    # Set log directory and file if provided
    if log_dir:
        from pathlib import Path

        log_file = f"rompy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        update_params["log_dir"] = Path(log_dir)
        update_params["log_file"] = log_file

    # Apply the configuration (update() will call configure_logging() if needed)
    logging_config.update(**update_params)

    # Configure warnings
    if not show_warnings:
        warnings.filterwarnings("ignore")
    else:
        # Show deprecation warnings
        warnings.filterwarnings("default", category=DeprecationWarning)

    # Log configuration
    logger.debug("Logging configured with level: %s", log_level.value)
    if log_dir:
        logger.info("Log directory: %s", log_dir)


# Common CLI options
common_options = [
    click.option(
        "-v",
        "--verbose",
        count=True,
        help="Increase verbosity (can be used multiple times)",
    ),
    click.option("--log-dir", envvar="ROMPY_LOG_DIR", help="Directory to save log files"),
    click.option(
        "--show-warnings/--hide-warnings", default=False, help="Show Python warnings"
    ),
    click.option(
        "--ascii-only/--unicode",
        default=False,
        help="Use ASCII-only characters in output",
        envvar="ROMPY_ASCII_ONLY",
    ),
    click.option(
        "--simple-logs/--detailed-logs",
        default=False,
        help="Use simple log format without timestamps and module names",
        envvar="ROMPY_SIMPLE_LOGS",
    ),
]

def add_common_options(f):
    """Decorator to add common CLI options to commands."""
    for option in reversed(common_options):
        f = option(f)
    return f

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from file or string."""
    try:
        with open(config_path, "r") as f:
            content = f.read()
    except (FileNotFoundError, IsADirectoryError, OSError):
        # Not a file, treat as raw string
        content = config_path

    logger.info(f"Loading config from: {config_path}")

    # Try JSON first
    try:
        config = json.loads(content)
        logger.info("Parsed config as JSON")
        return config
    except json.JSONDecodeError:
        pass

    # If JSON failed, try YAML
    try:
        config = yaml.safe_load(content)
        logger.info("Parsed config as YAML")
        return config
    except yaml.YAMLError as e:
        logger.error(f"Failed to parse config as JSON or YAML: {e}")
        raise click.UsageError("Config file is not valid JSON or YAML")

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--version", is_flag=True, help="Show version information and exit")
@click.pass_context
def cli(ctx, version):
    """ROMPY (Regional Ocean Modeling PYthon) - Ocean Model Configuration and Execution Tool.

    ROMPY provides tools for generating, running, and processing ocean, wave, and
    hydrodynamic model configurations with support for multiple execution backends.
    """
    # Import here to avoid circular imports
    import rompy

    if version:
        click.echo(f"ROMPY Version: {rompy.__version__}")
        ctx.exit()

    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)

@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--backend-config", type=click.Path(exists=True), required=True, help="YAML/JSON file with backend configuration")
@click.option("--dry-run", is_flag=True, help="Generate inputs only, don't run")
@add_common_options
def run(config, backend_config, dry_run, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Run a model configuration using Pydantic backend configuration.

    Examples:
        # Run with local backend configuration
        rompy run config.yml --backend-config local_backend.yml

        # Run with Docker backend configuration
        rompy run config.yml --backend-config docker_backend.yml
    """
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Load model configuration
        config_data = load_config(config)
        model_run = ModelRun(**config_data)

        logger.info(f"Running model: {model_run.config.model_type}")
        logger.info(f"Run ID: {model_run.run_id}")

        # Load backend configuration
        backend_cfg = _load_backend_config(backend_config)

        # Generate inputs
        start_time = datetime.now()
        staging_dir = model_run.generate()
        logger.info(f"Inputs generated in: {staging_dir}")

        if dry_run:
            logger.info("Dry run mode - skipping model execution")
            return

        # Execute model
        success = model_run.run(backend=backend_cfg)

        elapsed = datetime.now() - start_time
        if success:
            logger.info(f"✅ Model completed successfully in {elapsed.total_seconds():.2f}s")
        else:
            logger.error(f"❌ Model execution failed after {elapsed.total_seconds():.2f}s")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running model: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)


def _load_backend_config(backend_config_file):
    """Load backend configuration from config file.

    Args:
        backend_config_file: Path to backend config file

    Returns:
        BackendConfig instance
    """
    logger.info(f"Loading backend configuration from: {backend_config_file}")
    config_data = load_config(backend_config_file)

    # Determine backend type from config
    if "type" not in config_data:
        raise click.UsageError("Backend configuration must include a 'type' field")

    backend_type = config_data.pop("type")

    # Create appropriate config
    if backend_type == "local":
        return LocalConfig(**config_data)
    elif backend_type == "docker":
        return DockerConfig(**config_data)
    else:
        raise click.UsageError(f"Unknown backend type in config: {backend_type}")

@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--run-backend", default="local", help="Execution backend for run stage")
@click.option("--processor", default="noop", help="Postprocessor to use")
@click.option("--cleanup-on-failure/--no-cleanup", default=False, help="Clean up on failure")
@click.option("--validate-stages/--no-validate", default=True, help="Validate each stage")
@add_common_options
def pipeline(config, run_backend, processor, cleanup_on_failure, validate_stages, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Run complete model pipeline (generate -> run -> postprocess)."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Load configuration
        config_data = load_config(config)
        model_run = ModelRun(**config_data)

        logger.info(f"Running pipeline for: {model_run.config.model_type}")
        logger.info(f"Run ID: {model_run.run_id}")
        logger.info(f"Pipeline: generate → run({run_backend}) → postprocess({processor})")

        start_time = datetime.now()

        # Execute pipeline
        results = model_run.pipeline(
            pipeline_backend="local",
            run_backend=run_backend,
            processor=processor,
            cleanup_on_failure=cleanup_on_failure,
            validate_stages=validate_stages
        )

        elapsed = datetime.now() - start_time

        # Report results
        success = results.get("success", False)
        stages = results.get("stages_completed", [])

        logger.info(f"Pipeline completed in {elapsed.total_seconds():.2f}s")
        logger.info(f"Stages completed: {', '.join(stages)}")

        if success:
            logger.info("✅ Pipeline completed successfully")
        else:
            logger.error(f"❌ Pipeline failed: {results.get('message', 'Unknown error')}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)

@cli.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--output-dir", help="Override output directory")
@add_common_options
def generate(config, output_dir, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Generate model input files only."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Load configuration
        config_data = load_config(config)
        if output_dir:
            config_data["output_dir"] = output_dir

        model_run = ModelRun(**config_data)

        logger.info(f"Generating inputs for: {model_run.config.model_type}")
        logger.info(f"Run ID: {model_run.run_id}")

        start_time = datetime.now()
        staging_dir = model_run.generate()
        elapsed = datetime.now() - start_time

        logger.info(f"✅ Inputs generated in {elapsed.total_seconds():.2f}s")
        logger.info(f"📁 Staging directory: {staging_dir}")

        # List generated files
        if Path(staging_dir).exists():
            files = list(Path(staging_dir).glob("*"))
            logger.info(f"Generated {len(files)} files")

    except Exception as e:
        logger.error(f"Error generating inputs: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)

@cli.command()
@click.argument("config", type=click.Path(exists=True))
@add_common_options
def validate(config, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Validate model configuration."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Load and validate configuration
        config_data = load_config(config)
        model_run = ModelRun(**config_data)

        logger.info(f"✅ Configuration is valid")
        logger.info(f"Model type: {model_run.config.model_type}")
        logger.info(f"Run ID: {model_run.run_id}")
        logger.info(f"Period: {model_run.period}")
        logger.info(f"Output directory: {model_run.output_dir}")

    except Exception as e:
        logger.error(f"❌ Configuration validation failed: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)

@cli.group()
def backends():
    """Manage execution backends."""
    pass

@backends.command("list")
@add_common_options
def list_backends(verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """List available backends."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    logger.info("Available Backends:")

    logger.info("\n🏃 Run Backends:")
    for name, backend_class in RUN_BACKENDS.items():
        logger.info(f"  - {name}: {backend_class.__name__}")
        if hasattr(backend_class, '__doc__') and backend_class.__doc__:
            doc = backend_class.__doc__.strip().split('\n')[0]
            logger.info(f"    {doc}")

    logger.info("\n🔄 Postprocessors:")
    for name, proc_class in POSTPROCESSORS.items():
        logger.info(f"  - {name}: {proc_class.__name__}")
        if hasattr(proc_class, '__doc__') and proc_class.__doc__:
            doc = proc_class.__doc__.strip().split('\n')[0]
            logger.info(f"    {doc}")

    logger.info("\n🔗 Pipeline Backends:")
    for name, pipeline_class in PIPELINE_BACKENDS.items():
        logger.info(f"  - {name}: {pipeline_class.__name__}")
        if hasattr(pipeline_class, '__doc__') and pipeline_class.__doc__:
            doc = pipeline_class.__doc__.strip().split('\n')[0]
            logger.info(f"    {doc}")

    # Show Pydantic backend configurations
    logger.info("\n⚙️  Backend Configurations:")
    logger.info(f"  - LocalConfig → LocalRunBackend")
    logger.info(f"  - DockerConfig → DockerRunBackend")

@backends.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.option("--backend-type", type=click.Choice(["local", "docker"]), help="Backend type to validate as")
@add_common_options
def validate_backend_config(config_file, backend_type, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Validate a backend configuration file."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Load configuration
        config_data = load_config(config_file)

        # Determine backend type
        if backend_type:
            config_type = backend_type
        elif "type" in config_data:
            config_type = config_data.pop("type")
        else:
            raise click.UsageError("Backend type must be specified via --backend-type or 'type' field in config")

        # Validate configuration
        if config_type == "local":
            config = LocalConfig(**config_data)
            logger.info("✅ Local backend configuration is valid")
        elif config_type == "docker":
            config = DockerConfig(**config_data)
            logger.info("✅ Docker backend configuration is valid")
        else:
            raise click.UsageError(f"Unknown backend type: {config_type}")

        # Show configuration details
        logger.info(f"Backend type: {config_type}")
        logger.info(f"Timeout: {config.timeout}s")
        if config.env_vars:
            logger.info(f"Environment variables: {list(config.env_vars.keys())}")
        if config.working_dir:
            logger.info(f"Working directory: {config.working_dir}")

        # Type-specific details
        if isinstance(config, LocalConfig):
            if config.command:
                logger.info(f"Command: {config.command}")
        elif isinstance(config, DockerConfig):
            if config.image:
                logger.info(f"Image: {config.image}")
            if config.dockerfile:
                logger.info(f"Dockerfile: {config.dockerfile}")
            logger.info(f"CPU: {config.cpu}")
            if config.memory:
                logger.info(f"Memory: {config.memory}")
            if config.volumes:
                logger.info(f"Volumes: {len(config.volumes)} mounts")

    except Exception as e:
        logger.error(f"❌ Backend configuration validation failed: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)

@backends.command("schema")
@click.option("--backend-type", type=click.Choice(["local", "docker"]), required=True, help="Backend type to show schema for")
@click.option("--format", "output_format", type=click.Choice(["json", "yaml"]), default="json", help="Output format")
@click.option("--examples", is_flag=True, help="Include examples in output")
@add_common_options
def show_backend_schema(backend_type, output_format, examples, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Show JSON schema for backend configurations."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Get the appropriate configuration class
        if backend_type == "local":
            config_class = LocalConfig
        elif backend_type == "docker":
            config_class = DockerConfig
        else:
            raise click.UsageError(f"Unknown backend type: {backend_type}")

        # Generate schema
        schema = config_class.schema()

        if not examples:
            # Remove examples from schema
            schema.pop("examples", None)
            for prop in schema.get("properties", {}).values():
                prop.pop("examples", None)

        # Output schema
        if output_format == "json":
            import json
            output = json.dumps(schema, indent=2)
        else:  # yaml
            import yaml
            output = yaml.dump(schema, default_flow_style=False)

        click.echo(output)

    except Exception as e:
        logger.error(f"Error generating schema: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)

@backends.command("create")
@click.option("--backend-type", type=click.Choice(["local", "docker"]), required=True, help="Backend type to create")
@click.option("--output", type=click.Path(), help="Output file (default: stdout)")
@click.option("--format", "output_format", type=click.Choice(["json", "yaml"]), default="yaml", help="Output format")
@click.option("--with-examples", is_flag=True, help="Include example values")
@add_common_options
def create_backend_config(backend_type, output, output_format, with_examples, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Create a template backend configuration file."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Create template configuration
        if backend_type == "local":
            if with_examples:
                config_data = {
                    "type": "local",
                    "timeout": 7200,
                    "env_vars": {"OMP_NUM_THREADS": "4"},
                    "command": "python run_model.py",
                    "shell": True,
                    "capture_output": True
                }
            else:
                config_data = {
                    "type": "local",
                    "timeout": 3600,
                    "env_vars": {},
                    "command": None,
                    "shell": True,
                    "capture_output": True
                }
        elif backend_type == "docker":
            if with_examples:
                config_data = {
                    "type": "docker",
                    "image": "swan:latest",
                    "timeout": 7200,
                    "cpu": 4,
                    "memory": "2g",
                    "env_vars": {"DOCKER_ENV": "value"},
                    "volumes": ["/data:/app/data"],
                    "executable": "/usr/local/bin/run.sh"
                }
            else:
                config_data = {
                    "type": "docker",
                    "image": "your-image:latest",
                    "timeout": 3600,
                    "cpu": 1,
                    "env_vars": {},
                    "volumes": [],
                    "executable": "/usr/local/bin/run.sh"
                }

        # Format output
        if output_format == "json":
            import json
            content = json.dumps(config_data, indent=2)
        else:  # yaml
            import yaml
            content = yaml.dump(config_data, default_flow_style=False)

        # Write to file or stdout
        if output:
            with open(output, 'w') as f:
                f.write(content)
            logger.info(f"✅ Backend configuration template created: {output}")
        else:
            click.echo(content)

    except Exception as e:
        logger.error(f"Error creating backend configuration: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)

@cli.command()
@click.option("--model-type", help="Show schema for specific model type")
@add_common_options
def schema(model_type, verbose, log_dir, show_warnings, ascii_only, simple_logs):
    """Show configuration schema information."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    logger.info("Available Model Types:")
    for model in installed:
        marker = "→" if model == model_type else " "
        logger.info(f"  {marker} {model}")

    if model_type:
        logger.info(f"\nConfiguration schema for '{model_type}' would be shown here")
        logger.info("(Schema introspection not yet implemented)")

# Legacy command for backward compatibility
@cli.command(hidden=True)
@click.argument(
    "model", type=click.Choice(installed), envvar="ROMPY_MODEL", required=False
)
@click.argument("config", envvar="ROMPY_CONFIG", required=False)
@click.option("zip", "--zip/--no-zip", default=False, envvar="ROMPY_ZIP")
@add_common_options
def legacy_main(
    model,
    config,
    zip,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
):
    """Legacy command for backward compatibility (DEPRECATED).

    Use 'rompy run' instead for new functionality.
    """
    # Configure logging
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    # Import here to avoid circular imports
    import rompy

    # If no model or config is provided, show help and available models
    if not model or not config:
        logger.info(f"ROMPY Version: {rompy.__version__}")
        logger.info(f"Available models: {', '.join(installed)}")
        logger.info("Run 'rompy --help' for usage information")
        ctx = click.get_current_context()
        click.echo(ctx.get_help())
        ctx.exit()

    try:
        # Load configuration
        config_data = load_config(config)

        # Log version and execution information
        logger.info(f"ROMPY Version: {rompy.__version__}")
        logger.info(f"Running model: {model}")
        logger.info(f"Configuration: {config}")

        # Create and run the model
        start_time = datetime.now()
        logger.info("Running model...")
        model_run = ModelRun(**config_data)
        model_run()

        if zip:
            logger.info("Zipping model outputs...")
            zip_file = model_run.zip()
            logger.info(f"Model archive created: {zip_file}")

        # Log completion time
        elapsed = datetime.now() - start_time
        logger.info(f"Model run completed in {elapsed.total_seconds():.2f} seconds")

        if log_dir:
            logger.info(f"Log directory: {log_dir}")
    except TypeError as e:
        if "unsupported format string" in str(e) and "timedelta" in str(e):
            logger.error(f"Error with time format: {str(e)}")
            logger.error(
                "This is likely due to formatting issues with time duration values"
            )
            if verbose > 0:
                logger.error("", exc_info=True)
        else:
            logger.error(f"Type error in model: {str(e)}", exc_info=verbose > 0)
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running model: {str(e)}", exc_info=verbose > 0)
        sys.exit(1)

# Set the main command as the default for backward compatibility
main = cli

if __name__ == "__main__":
    cli()
