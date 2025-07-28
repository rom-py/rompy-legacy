"""
ROMPY Command Line Interface

This module provides the command-line interface for ROMPY.
"""

import importlib
import importlib.metadata
import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import click
import yaml

import rompy
from rompy.backends import DockerConfig, LocalConfig
from rompy.core.logging import LogFormat, LoggingConfig, LogLevel, get_logger
from rompy.model import (PIPELINE_BACKENDS, POSTPROCESSORS, RUN_BACKENDS,
                         ModelRun)

# Initialize the logger
logger = get_logger(__name__)

# Get installed entry points
installed = importlib.metadata.entry_points(group="rompy.config").names


def configure_logging(
    verbosity: int = 0,
    log_dir: Optional[str] = None,
    simple_logs: bool = False,
    ascii_only: bool = False,
    show_warnings: bool = False,
) -> None:
    """Configure logging based on verbosity level and other options.

    This function configures the logging system using the LoggingConfig class.
    It sets up the log level, format, and output destinations based on the
    provided parameters.

    Args:
        verbosity: Level of verbosity (0 = warning, 1 = info, 2+ = debug)
        log_dir: Directory to save log files (optional)
        simple_logs: Use simple log format without timestamps and module names
        ascii_only: Use ASCII-only characters in output
        show_warnings: Show Python warnings
    """
    # Determine log level based on verbosity
    if verbosity == 0:
        log_level = LogLevel.WARNING
    elif verbosity == 1:
        log_level = LogLevel.INFO
    else:
        log_level = LogLevel.DEBUG

    # Determine log format
    log_format = LogFormat.SIMPLE if simple_logs else LogFormat.VERBOSE

    # Configure logging
    logging_config = LoggingConfig(
        level=log_level,
        format=log_format,
        log_dir=Path(log_dir) if log_dir else None,
        use_ascii=ascii_only,
    )

    # Apply configuration
    logging_config.configure_logging()

    # Handle warnings
    if show_warnings:
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
    click.option(
        "--log-dir", envvar="ROMPY_LOG_DIR", help="Directory to save log files"
    ),
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
    click.option(
        "--config-from-env",
        is_flag=True,
        help="Load configuration from ROMPY_CONFIG environment variable instead of file",
    ),
]


def add_common_options(f):
    """Decorator to add common CLI options to commands."""
    for option in reversed(common_options):
        f = option(f)
    return f


def load_config(
    config_path: str, from_env: bool = False, env_var: str = "ROMPY_CONFIG"
) -> Dict[str, Any]:
    """Load configuration from file, string, or environment variable.

    Args:
        config_path: Path to config file or raw config string
        from_env: If True, load from environment variable instead of config_path
        env_var: Environment variable name to load from when from_env=True

    Returns:
        Dict containing parsed configuration

    Raises:
        click.UsageError: If config cannot be loaded or parsed
    """
    if from_env:
        content = os.environ.get(env_var)
        if content is None:
            raise click.UsageError(f"Environment variable {env_var} is not set")
        logger.info(f"Loading config from environment variable: {env_var}")
    else:
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


def print_version(ctx, param, value):
    """Callback to print version and exit."""
    if not value or ctx.resilient_parsing:
        return

    # Import here to avoid circular imports
    import rompy

    click.echo(f"ROMPY Version: {rompy.__version__}")
    click.echo(f"Available models: {', '.join(installed)}")
    ctx.exit(0)


@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--version",
    is_flag=True,
    expose_value=False,
    is_eager=True,
    callback=print_version,
    help="Show version information and exit",
)
@click.pass_context
def cli(ctx):
    """ROMPY (Regional Ocean Modeling PYthon) - Ocean Model Configuration and Execution Tool.

    ROMPY provides tools for generating, running, and processing ocean, wave, and
    hydrodynamic model configurations with support for multiple execution backends.
    """
    # Ensure that ctx.obj exists and is a dict
    ctx.ensure_object(dict)


@cli.command()
@click.argument("config", type=click.Path(exists=True), required=False)
@click.option(
    "--backend-config",
    type=click.Path(exists=True),
    required=True,
    help="YAML/JSON file with backend configuration",
)
@click.option("--dry-run", is_flag=True, help="Generate inputs only, don't run")
@add_common_options
def run(
    config,
    backend_config,
    dry_run,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    config_from_env,
):
    """Run a model configuration using Pydantic backend configuration.

    Examples:
        # Run with local backend configuration
        rompy run config.yml --backend-config unified_local_single.yml

        # Run with Docker backend configuration
        rompy run config.yml --backend-config unified_docker_single.yml

        # Run with config from environment variable
        rompy run --config-from-env --backend-config unified_local_single.yml
    """
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    # Validate config source
    if config_from_env and config:
        raise click.UsageError("Cannot specify both config file and --config-from-env")
    if not config_from_env and not config:
        raise click.UsageError("Must specify either config file or --config-from-env")

    try:
        # Load model configuration
        config_data = load_config(config, from_env=config_from_env)
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

        # Execute model with workspace directory to avoid double generation
        success = model_run.run(backend=backend_cfg, workspace_dir=staging_dir)

        elapsed = datetime.now() - start_time
        if success:
            logger.info(
                f"✅ Model completed successfully in {elapsed.total_seconds():.2f}s"
            )
        else:
            logger.error(
                f"❌ Model execution failed after {elapsed.total_seconds():.2f}s"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running model: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)


def _get_backend_config_registry():
    """
    Build a registry of backend config classes from entry points and built-ins.
    Returns: dict mapping backend type name to config class
    """
    registry = {
        "local": LocalConfig,
        "docker": DockerConfig,
    }
    # Try to load from entry points (rompy.config and rompy.backend_config)
    try:
        eps = importlib.metadata.entry_points()
        # Support both 'rompy.config' and 'rompy.backend_config' for flexibility
        for group in ["rompy.config", "rompy.backend_config"]:
            if hasattr(eps, "select"):  # Python 3.10+
                entries = eps.select(group=group)
            elif hasattr(eps, "get"):  # Python 3.8-3.9
                entries = eps.get(group, [])
            else:
                entries = []
            for ep in entries:
                try:
                    cls = ep.load()
                    registry[ep.name] = cls
                except Exception as e:
                    logger.warning(
                        f"Failed to load backend config entry point {ep.name}: {e}"
                    )
    except Exception as e:
        logger.warning(f"Could not load backend config entry points: {e}")
    return registry


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
    registry = _get_backend_config_registry()
    if backend_type in registry:
        return registry[backend_type](**config_data)
    else:
        available = ", ".join(registry.keys())
        raise click.UsageError(
            f"Unknown backend type in config: {backend_type}. Available types: {available}"
        )


@cli.command()
@click.argument("config", type=click.Path(exists=True), required=False)
@click.option("--run-backend", default="local", help="Execution backend for run stage")
@click.option("--processor", default="noop", help="Postprocessor to use")
@click.option(
    "--cleanup-on-failure/--no-cleanup", default=False, help="Clean up on failure"
)
@click.option(
    "--validate-stages/--no-validate", default=True, help="Validate each stage"
)
@add_common_options
def pipeline(
    config,
    run_backend,
    processor,
    cleanup_on_failure,
    validate_stages,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    config_from_env,
):
    """Run full model pipeline: generate → run → postprocess."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    # Validate config source
    if config_from_env and config:
        raise click.UsageError("Cannot specify both config file and --config-from-env")
    if not config_from_env and not config:
        raise click.UsageError("Must specify either config file or --config-from-env")

    try:
        # Load configuration
        config_data = load_config(config, from_env=config_from_env)
        model_run = ModelRun(**config_data)

        logger.info(f"Running pipeline for: {model_run.config.model_type}")
        logger.info(f"Run ID: {model_run.run_id}")
        logger.info(
            f"Pipeline: generate → run({run_backend}) → postprocess({processor})"
        )

        start_time = datetime.now()

        # Execute pipeline
        results = model_run.pipeline(
            pipeline_backend="local",
            run_backend=run_backend,
            processor=processor,
            cleanup_on_failure=cleanup_on_failure,
            validate_stages=validate_stages,
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
            logger.error(
                f"❌ Pipeline failed: {results.get('message', 'Unknown error')}"
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error running pipeline: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)


@cli.command()
@click.argument("config", type=click.Path(exists=True), required=False)
@click.option("--output-dir", help="Override output directory")
@add_common_options
def generate(
    config,
    output_dir,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    config_from_env,
):
    """Generate model input files only."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    # Validate config source
    if config_from_env and config:
        raise click.UsageError("Cannot specify both config file and --config-from-env")
    if not config_from_env and not config:
        raise click.UsageError("Must specify either config file or --config-from-env")

    try:
        # Load configuration
        config_data = load_config(config, from_env=config_from_env)
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
@click.argument("config", type=click.Path(exists=True), required=False)
@click.option("--processor", default="noop", help="Postprocessor to use (default: noop)")
@click.option("--output-dir", help="Override output directory for postprocessing")
@click.option("--validate-outputs/--no-validate", default=True, help="Validate outputs exist (default: True)")
@add_common_options
def postprocess(
    config,
    processor,
    output_dir,
    validate_outputs,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    config_from_env,
):
    """Run postprocessing on model outputs using the specified postprocessor."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    # Validate config source
    if config_from_env and config:
        raise click.UsageError("Cannot specify both config file and --config-from-env")
    if not config_from_env and not config:
        raise click.UsageError("Must specify either config file or --config-from-env")

    try:
        # Load configuration
        config_data = load_config(config, from_env=config_from_env)
        model_run = ModelRun(**config_data)

        logger.info(f"Running postprocessing for: {model_run.config.model_type}")
        logger.info(f"Run ID: {model_run.run_id}")
        logger.info(f"Postprocessor: {processor}")

        # Run postprocessing
        start_time = datetime.now()
        results = model_run.postprocess(
            processor=processor,
            output_dir=output_dir,
            validate_outputs=validate_outputs,
        )
        elapsed = datetime.now() - start_time

        logger.info(f"✅ Postprocessing completed in {elapsed.total_seconds():.2f}s")
        logger.info(f"Results: {results}")

    except Exception as e:
        logger.error(f"❌ Postprocessing failed: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)

@cli.command()
@click.argument("config", type=click.Path(exists=True), required=False)
@add_common_options
def validate(
    config, verbose, log_dir, show_warnings, ascii_only, simple_logs, config_from_env
):
    """Validate model configuration."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    # Validate config source
    if config_from_env and config:
        raise click.UsageError("Cannot specify both config file and --config-from-env")
    if not config_from_env and not config:
        raise click.UsageError("Must specify either config file or --config-from-env")

    try:
        # Load and validate configuration
        config_data = load_config(config, from_env=config_from_env)
        model_run = ModelRun(**config_data)

        logger.info("✅ Configuration is valid")
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
def list_backends(
    verbose, log_dir, show_warnings, ascii_only, simple_logs, config_from_env
):
    """List available backends."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    logger.info("Available Backends:")

    logger.info("\n🏃 Run Backends:")
    for name, backend_class in RUN_BACKENDS.items():
        logger.info(f"  - {name}: {backend_class.__name__}")
        if hasattr(backend_class, "__doc__") and backend_class.__doc__:
            doc = backend_class.__doc__.strip().split("\n")[0]
            logger.info(f"    {doc}")

    logger.info("\n🔄 Postprocessors:")
    for name, proc_class in POSTPROCESSORS.items():
        logger.info(f"  - {name}: {proc_class.__name__}")
        if hasattr(proc_class, "__doc__") and proc_class.__doc__:
            doc = proc_class.__doc__.strip().split("\n")[0]
            logger.info(f"    {doc}")

    logger.info("\n🔗 Pipeline Backends:")
    for name, pipeline_class in PIPELINE_BACKENDS.items():
        logger.info(f"  - {name}: {pipeline_class.__name__}")
        if hasattr(pipeline_class, "__doc__") and pipeline_class.__doc__:
            doc = pipeline_class.__doc__.strip().split("\n")[0]
            logger.info(f"    {doc}")

    # Show Pydantic backend configurations
    logger.info("\n⚙️  Backend Configurations:")
    logger.info("  - LocalConfig → LocalRunBackend")
    logger.info("  - DockerConfig → DockerRunBackend")


@backends.command("validate")
@click.argument("config_file", type=click.Path(exists=True))
@click.option(
    "--backend-type",
    type=click.Choice(["local", "docker"]),
    help="Backend type to validate as",
)
@add_common_options
def validate_backend_config(
    config_file,
    backend_type,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    config_from_env,
):
    """Validate a backend configuration file."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Load configuration
        config_data = load_config(config_file)

        # Determine backend type and extract config parameters
        if backend_type:
            config_type = backend_type
        elif "backend_type" in config_data:
            config_type = config_data.pop("backend_type")
        elif "type" in config_data:
            config_type = config_data.pop("type")
        else:
            raise click.UsageError(
                "Backend type must be specified via --backend-type or 'type' field in config"
            )

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
@click.option(
    "--backend-type",
    type=click.Choice(["local", "docker"]),
    required=True,
    help="Backend type to show schema for",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"]),
    default="json",
    help="Output format",
)
@click.option("--examples", is_flag=True, help="Include examples in output")
@add_common_options
def show_backend_schema(
    backend_type,
    output_format,
    examples,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    config_from_env,
):
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
        schema = config_class.model_json_schema()

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
@click.option(
    "--backend-type",
    type=click.Choice(["local", "docker"]),
    required=True,
    help="Backend type to create",
)
@click.option("--output", type=click.Path(), help="Output file (default: stdout)")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"]),
    default="yaml",
    help="Output format",
)
@click.option("--with-examples", is_flag=True, help="Include example values")
@add_common_options
def create_backend_config(
    backend_type,
    output,
    output_format,
    with_examples,
    verbose,
    log_dir,
    show_warnings,
    ascii_only,
    simple_logs,
    config_from_env,
):
    """Create a template backend configuration file."""
    configure_logging(verbose, log_dir, simple_logs, ascii_only, show_warnings)

    try:
        # Create template configuration
        config_data = {}
        if backend_type == "local":
            if with_examples:
                config_data = {
                    "type": "local",
                    "timeout": 7200,
                    "env_vars": {"OMP_NUM_THREADS": "4"},
                    "command": "python run_model.py",
                    "shell": True,
                    "capture_output": True,
                }
            else:
                config_data = {
                    "type": "local",
                    "timeout": 3600,
                    "env_vars": {},
                    "command": None,
                    "shell": True,
                    "capture_output": True,
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
                    "executable": "/usr/local/bin/run.sh",
                }
            else:
                config_data = {
                    "type": "docker",
                    "image": "your-image:latest",
                    "timeout": 3600,
                    "cpu": 1,
                    "env_vars": {},
                    "volumes": [],
                    "executable": "/usr/local/bin/run.sh",
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
            with open(output, "w") as f:
                f.write(content)
            logger.info(f"✅ Backend configuration template created: {output}")
        else:
            click.echo(content)

    except Exception as e:
        logger.error(f"Error creating backend configuration: {e}")
        if verbose > 0:
            logger.exception("Full traceback:")
        sys.exit(1)


@cli.command(name="schema")
@click.argument("model_type", default="ModelRun", type=str, required=False)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, writable=True),
    help="Output file to save the schema (default: print to stdout)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml"], case_sensitive=False),
    default="json",
    help="Output format (default: json)",
)
@add_common_options
def schema(
    model_type: str,
    output: Optional[str] = None,
    output_format: str = "json",
    verbose: int = 0,
    log_dir: Optional[str] = None,
    show_warnings: bool = False,
    ascii_only: bool = False,
    simple_logs: bool = False,
    config_from_env: bool = False,
) -> None:
    """Show JSON schema for a rompy model.

    Examples:
        # Show schema for ModelRun (default)
        rompy schema

        # Show schema for a specific model type
        rompy schema "rompy.model.ModelRun"
        rompy schema "rompy.swan.SWAN"

        # Save schema to a file
        rompy schema --output=model_schema.json
        rompy schema --output=model_schema.yaml --format=yaml
    """
    # Configure logging
    configure_logging(
        verbosity=verbose,
        log_dir=log_dir,
        simple_logs=simple_logs,
        ascii_only=ascii_only,
        show_warnings=show_warnings,
    )

    # Get logger for this module
    logger = get_logger(__name__)

    try:
        logger.debug(f"Showing schema for model: {model_type}")

        # Import the model class
        try:
            if "." in model_type:
                # Full module path provided (e.g., "rompy.model.ModelRun")
                module_path, class_name = model_type.rsplit(".", 1)
                logger.debug(f"Importing {class_name} from {module_path}")
                module = importlib.import_module(module_path)
                model_class = getattr(module, class_name)
            else:
                # Try to import from rompy.model first
                try:
                    logger.debug(f"Trying to import {model_type} from rompy.model")
                    model_class = getattr(rompy.model, model_type)
                except AttributeError:
                    logger.debug(
                        f"{model_type} not found in rompy.model, checking entry points"
                    )
                    # Try to find the model in entry points with different approaches
                    try:
                        # Python 3.10+ style
                        model_eps = importlib.metadata.entry_points()
                        if hasattr(model_eps, "select"):
                            # Python 3.10+
                            model_entries = model_eps.select(group="rompy.model")
                        elif hasattr(model_eps, "get"):
                            # Python 3.8-3.9
                            model_entries = model_eps.get("rompy.model", [])
                        else:
                            # Fallback for older Python versions
                            model_entries = []
                            if hasattr(model_eps, "items"):
                                for group, entries in model_eps.items():
                                    if group == "rompy.model":
                                        model_entries = entries
                                        break
                    except Exception as e:
                        logger.debug(f"Error getting entry points: {e}")
                        model_entries = []

                    # Try to find the model in entry points
                    found = False
                    for entry_point in model_entries:
                        if entry_point.name.lower() == model_type.lower():
                            logger.debug(
                                f"Found {model_type} in entry points, loading..."
                            )
                            model_class = entry_point.load()
                            found = True
                            break

                    if not found:
                        # Try direct import as a last resort
                        try:
                            model_class = importlib.import_module(
                                f"rompy.{model_type.lower()}"
                            )
                            found = True
                        except ImportError:
                            raise ImportError(
                                f"No model found with name '{model_type}'. "
                                "Please provide the full module path (e.g., 'rompy.swan.SWAN')"
                            )

            logger.debug(
                f"Successfully imported model class: {model_class.__module__}.{model_class.__name__}"
            )

        except (ImportError, AttributeError) as e:
            # Initialize error message
            error_msg = [f"Could not import model class '{model_type}'"]

            # Get available models from rompy.model
            try:
                from rompy.model import __all__ as model_classes

                error_msg.append(
                    f"Available models in rompy.model: {', '.join(model_classes)}"
                )
            except Exception as e:
                logger.debug(f"Could not get models from rompy.model: {e}")

            # Try to get available models from entry points
            try:
                model_eps = importlib.metadata.entry_points()
                if hasattr(model_eps, "select"):
                    # Python 3.10+
                    model_entries = model_eps.select(group="rompy.model")
                elif hasattr(model_eps, "get"):
                    # Python 3.8-3.9
                    model_entries = model_eps.get("rompy.model", [])
                else:
                    # Fallback for older Python versions
                    model_entries = []
                    if hasattr(model_eps, "items"):
                        for group, entries in model_eps.items():
                            if group == "rompy.model":
                                model_entries = entries
                                break

                available_models = [ep.name for ep in model_entries]
                if available_models:
                    error_msg.append(
                        f"Available models from entry points: {', '.join(available_models)}"
                    )
                else:
                    error_msg.append("No models found in entry points")

            except Exception as ep_error:
                error_msg.append(f"Error checking entry points: {ep_error}")

            # Log all error messages
            for msg in error_msg:
                logger.error(msg)

            # Add more detailed error info if verbose
            if verbose > 0:
                logger.error(f"Error details: {str(e)}")
                import traceback

                logger.error(traceback.format_exc())

            # Suggest using full module path
            logger.error("\nTry using the full module path, e.g., 'rompy.swan.SWAN'")
            logger.error(
                "For a list of available models, run: python -m rompy.cli list-models"
            )

            sys.exit(1)

        # Generate the schema
        schema = model_class.model_json_schema()

        # Output the schema
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_format == "json":
                with open(output_path, "w") as f:
                    json.dump(schema, f, indent=2)
            else:  # yaml
                with open(output_path, "w") as f:
                    yaml.dump(schema, f, default_flow_style=False)
            logger.info(f"Schema written to {output_path}")
        else:
            if output_format == "json":
                print(json.dumps(schema, indent=2))
            else:  # yaml
                print(yaml.dump(schema, default_flow_style=False))

    except Exception as e:
        logger.error(f"Error generating schema: {e}")
        if verbose > 0:
            import traceback

            logger.error(traceback.format_exc())
        sys.exit(1)


@cli.command(name="legacy", hidden=True)
@click.argument(
    "model", type=click.Choice(installed), envvar="ROMPY_MODEL", required=False
)
@click.argument("config", envvar="ROMPY_CONFIG", required=False)
@click.option("zip", "--zip/--no-zip", default=False, envvar="ROMPY_ZIP")
@click.option(
    "-v",
    "--verbose",
    count=True,
    help="Increase verbosity (can be used multiple times)",
)
@click.option("--log-dir", envvar="ROMPY_LOG_DIR", help="Directory to save log files")
@click.option(
    "--show-warnings/--hide-warnings", default=False, help="Show Python warnings"
)
@click.option(
    "--ascii-only/--unicode",
    default=False,
    help="Use ASCII-only characters in output",
    envvar="ROMPY_ASCII_ONLY",
)
@click.option(
    "--simple-logs/--detailed-logs",
    default=False,
    help="Use simple log format without timestamps and module names",
    envvar="ROMPY_SIMPLE_LOGS",
)
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


def main():
    """Entry point for the rompy CLI.

    This function is used by the console script entry point.
    """
    cli()


if __name__ == "__main__":
    cli()
