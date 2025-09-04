#!/usr/bin/env python3
"""
Validation script for ROMPY CLI configuration files.

This script validates the YAML configuration files in this directory
to ensure they follow the correct schema and can be loaded properly.
"""

import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def validate_backend_config(config_data: Dict[str, Any]) -> List[str]:
    """Validate backend configuration structure.

    Args:
        config_data: Configuration data from YAML

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if "backend_type" not in config_data:
        errors.append("Missing required field: backend_type")
        return errors

    backend_type = config_data["backend_type"]
    if backend_type not in ["local", "docker"]:
        errors.append(
            f"Invalid backend_type: {backend_type}. Must be 'local' or 'docker'"
        )
        return errors

    # Use direct format: backend_type: local, timeout: 3600, ...
    config = config_data.copy()
    config.pop("backend_type", None)  # Remove backend_type from parameters

    # Validate common fields
    if "timeout" in config:
        if (
            not isinstance(config["timeout"], int)
            or config["timeout"] < 60
            or config["timeout"] > 86400
        ):
            errors.append("timeout must be an integer between 60 and 86400")

    if "env_vars" in config:
        if not isinstance(config["env_vars"], dict):
            errors.append("env_vars must be a dictionary")

    # Validate backend-specific fields
    if backend_type == "local":
        if "command" in config and not isinstance(config["command"], (str, type(None))):
            errors.append("command must be a string or null")
        if "shell" in config and not isinstance(config["shell"], bool):
            errors.append("shell must be a boolean")
        if "capture_output" in config and not isinstance(
            config["capture_output"], bool
        ):
            errors.append("capture_output must be a boolean")

    elif backend_type == "docker":
        # Check that either image or dockerfile is provided
        has_image = "image" in config and config["image"]
        has_dockerfile = "dockerfile" in config and config["dockerfile"]

        if not has_image and not has_dockerfile:
            errors.append("Either 'image' or 'dockerfile' must be provided")

        if has_image and has_dockerfile:
            errors.append("Cannot specify both 'image' and 'dockerfile'")

        if "cpu" in config:
            if (
                not isinstance(config["cpu"], int)
                or config["cpu"] < 1
                or config["cpu"] > 128
            ):
                errors.append("cpu must be an integer between 1 and 128")

        if "memory" in config and config["memory"]:
            import re

            if not re.match(r"^\d+[kmgKMG]?$", config["memory"]):
                errors.append("memory must be in format like '2g', '512m', or '1024'")

        if "volumes" in config:
            if not isinstance(config["volumes"], list):
                errors.append("volumes must be a list")
            else:
                for volume in config["volumes"]:
                    if ":" not in volume:
                        errors.append(
                            f"Volume mount must contain ':' separator: {volume}"
                        )

    return errors


def validate_pipeline_config(config_data: Dict[str, Any]) -> List[str]:
    """Validate pipeline configuration structure.

    Args:
        config_data: Configuration data from YAML

    Returns:
        List of validation errors (empty if valid)
    """
    errors = []

    # Check required fields
    if "pipeline_backend" not in config_data:
        errors.append("Missing required field: pipeline_backend")

    if "model_run" not in config_data:
        errors.append("Missing required field: model_run")
    else:
        model_run = config_data["model_run"]
        if not isinstance(model_run, dict):
            errors.append("model_run must be a dictionary")
        else:
            # Validate model_run fields
            if "run_id" not in model_run:
                errors.append("model_run missing required field: run_id")
            if "output_dir" not in model_run:
                errors.append("model_run missing required field: output_dir")
            if "period" not in model_run:
                errors.append("model_run missing required field: period")
            else:
                period = model_run["period"]
                if not isinstance(period, dict):
                    errors.append("period must be a dictionary")
                else:
                    if "start" not in period:
                        errors.append("period missing required field: start")
                    if "end" not in period:
                        errors.append("period missing required field: end")

    if "run_backend" not in config_data:
        errors.append("Missing required field: run_backend")
    else:
        run_backend = config_data["run_backend"]
        if isinstance(run_backend, dict):
            # Validate as backend config
            backend_errors = validate_backend_config(run_backend)
            errors.extend([f"run_backend.{error}" for error in backend_errors])

    return errors


def validate_yaml_file(file_path: Path) -> bool:
    """Validate a single YAML configuration file.

    Args:
        file_path: Path to the YAML file

    Returns:
        True if valid, False otherwise
    """
    logger.info(f"Validating {file_path.name}...")

    try:
        with open(file_path, "r") as f:
            # Load all documents from the YAML file
            documents = list(yaml.safe_load_all(f))

        if not documents:
            logger.error(f"  ❌ No documents found in {file_path.name}")
            return False

        all_valid = True

        for i, doc in enumerate(documents):
            if doc is None:
                continue

            doc_name = f"document {i+1}" if len(documents) > 1 else "document"

            # Determine configuration type and validate
            if "pipeline_backend" in doc:
                # Pipeline configuration
                errors = validate_pipeline_config(doc)
                config_type = "pipeline"
            elif "backend_type" in doc:
                # Backend configuration
                errors = validate_backend_config(doc)
                config_type = "backend"
            else:
                logger.warning(f"  ⚠️  Unknown configuration type in {doc_name}")
                continue

            if errors:
                logger.error(f"  ❌ {doc_name} ({config_type}) has validation errors:")
                for error in errors:
                    logger.error(f"     - {error}")
                all_valid = False
            else:
                logger.info(f"  ✅ {doc_name} ({config_type}) is valid")

        return all_valid

    except yaml.YAMLError as e:
        logger.error(f"  ❌ YAML parsing error in {file_path.name}: {e}")
        return False
    except Exception as e:
        logger.error(f"  ❌ Error validating {file_path.name}: {e}")
        return False


def main():
    """Main validation function."""
    # Get the directory containing this script
    config_dir = Path(__file__).parent

    # Find all YAML files (excluding README and Python files)
    yaml_files = list(config_dir.glob("*.yml")) + list(config_dir.glob("*.yaml"))

    if not yaml_files:
        logger.warning("No YAML configuration files found")
        return True

    logger.info(f"Found {len(yaml_files)} configuration files to validate")
    logger.info("=" * 60)

    all_valid = True

    for yaml_file in sorted(yaml_files):
        valid = validate_yaml_file(yaml_file)
        if not valid:
            all_valid = False
        logger.info("")  # Empty line between files

    # Summary
    logger.info("=" * 60)
    if all_valid:
        logger.info("✅ All configuration files are valid!")
        return True
    else:
        logger.error("❌ Some configuration files have validation errors")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
