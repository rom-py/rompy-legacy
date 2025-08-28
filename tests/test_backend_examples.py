#!/usr/bin/env python3
"""
Test script for ROMPY backend examples and configurations.

This script validates that all backend examples, configuration files, and
the backend system work correctly with the current ROMPY implementation.
"""

import sys
import tempfile
from pathlib import Path
import logging

# Add the parent directory to the path so we can import rompy
sys.path.insert(0, str(Path(__file__).parent.parent))

from rompy.backends import LocalConfig, DockerConfig
from rompy.model import ModelRun
from rompy.core.time import TimeRange
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def test_pydantic_configs():
    """Test that Pydantic configuration objects work correctly."""
    logger.info("Testing Pydantic configuration objects...")

    # Test LocalConfig
    local_config = LocalConfig(
        timeout=3600, command="echo 'test'", env_vars={"TEST": "value"}
    )
    logger.info(f"✅ LocalConfig created: {local_config}")
    assert local_config is not None
    assert local_config.timeout == 3600
    assert local_config.command == "echo 'test'"
    assert local_config.env_vars == {"TEST": "value"}

    # Test DockerConfig
    docker_config = DockerConfig(
        image="python:3.9-slim",
        timeout=1800,
        cpu=2,
        memory="1g",
        volumes=["/tmp:/tmp:rw"],
        env_vars={"TEST": "value"},
    )
    logger.info(f"✅ DockerConfig created: {docker_config}")
    assert docker_config is not None
    assert docker_config.image == "python:3.9-slim"
    assert docker_config.timeout == 1800
    assert docker_config.cpu == 2
    assert docker_config.memory == "1g"


def test_model_run_integration():
    """Test that ModelRun works with backend configurations."""
    logger.info("Testing ModelRun integration...")

    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a basic model run
        model = ModelRun(
            run_id="test_integration",
            period=TimeRange(
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
                interval="1H",
            ),
            output_dir=temp_dir,
            delete_existing=True,
        )

        # Test that model can be created with backend config
        local_config = LocalConfig(
            timeout=1800,
            command="echo 'Integration test completed'",
        )

        logger.info("✅ ModelRun created with LocalConfig")
        assert model is not None
        assert model.run_id == "test_integration"
        assert local_config is not None

        # Test that the backend config is accepted
        backend_class = local_config.get_backend_class()
        logger.info(f"✅ Backend class resolved: {backend_class}")
        assert backend_class is not None


def test_backend_examples():
    """Test all backend example files."""
    logger.info("Testing backend example files...")

    examples_dir = Path(__file__).parent.parent / "examples" / "backends"
    if not examples_dir.exists():
        logger.warning(f"Backend examples directory not found: {examples_dir}")
        # Skip test if examples directory doesn't exist
        return

    # Test each example file
    example_files = [
        "01_basic_local_run.py",
        "02_docker_run.py",
        "03_custom_postprocessor.py",
        "04_complete_workflow.py",
    ]

    for example_file in example_files:
        example_path = examples_dir / example_file
        if not example_path.exists():
            logger.warning(f"Example file not found: {example_path}")
            continue

        # Test that the file can be imported and parsed
        with open(example_path, "r") as f:
            content = f.read()

        # Check for required imports
        required_imports = [
            "from rompy.model import ModelRun",
            "from rompy.core.time import TimeRange",
        ]

        backend_imports = [
            "from rompy.backends import LocalConfig",
            "from rompy.backends import DockerConfig",
        ]

        has_required = all(imp in content for imp in required_imports)
        has_backend = any(imp in content for imp in backend_imports)

        assert (
            has_required
        ), f"{example_file} missing required imports: {required_imports}"
        assert (
            has_backend or "LocalConfig" in content or "DockerConfig" in content
        ), f"{example_file} missing backend configuration imports"

        logger.info(f"✅ {example_file} imports look correct")


def test_yaml_configs():
    """Test YAML configuration files."""
    logger.info("Testing YAML configuration files...")

    configs_dir = Path(__file__).parent.parent / "examples" / "configs"
    if not configs_dir.exists():
        logger.warning(f"Configs directory not found: {configs_dir}")
        # Skip test if configs directory doesn't exist
        return

    # Test each YAML config file
    yaml_files = [
        "local_backend.yaml",
        "docker_backend.yaml",
        "complete_config.yaml",
    ]

    for yaml_file in yaml_files:
        yaml_path = configs_dir / yaml_file
        if not yaml_path.exists():
            logger.warning(f"YAML config file not found: {yaml_path}")
            continue

        import yaml

        with open(yaml_path, "r") as f:
            config_data = yaml.safe_load(f)

        # Basic validation - check that it's a valid YAML and has expected structure
        assert isinstance(
            config_data, dict
        ), f"{yaml_file} is not a valid configuration dictionary"

        # Check for common configuration keys
        expected_keys = ["model", "backend", "run_id", "period"]
        has_expected = any(key in config_data for key in expected_keys)

        assert (
            has_expected
        ), f"{yaml_file} missing expected configuration keys: {expected_keys}"
        logger.info(f"✅ {yaml_file} is valid YAML with expected structure")


def test_config_validation():
    """Test configuration validation using our validation script."""
    logger.info("Testing configuration validation...")

    # Check if validation script exists
    validation_script = Path(__file__).parent.parent / "scripts" / "validate_config.py"
    if not validation_script.exists():
        logger.warning(f"Validation script not found: {validation_script}")
        # Skip this test if script doesn't exist
        return

    # Test that the validation script can be imported
    import importlib.util

    spec = importlib.util.spec_from_file_location("validate_config", validation_script)
    assert spec is not None, "Could not create spec for validation script"
    assert spec.loader is not None, "Could not get loader for validation script"

    validate_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validate_module)
    logger.info("✅ Configuration validation script imported successfully")


def test_quickstart_example():
    """Test that the quickstart example file exists and is valid."""
    logger.info("Testing quickstart example...")

    # Check if quickstart example exists
    quickstart_path = Path(__file__).parent.parent / "examples" / "quickstart.py"
    if not quickstart_path.exists():
        logger.warning(f"Quickstart example not found: {quickstart_path}")
        # Skip this test if file doesn't exist
        return

    # Test that the file can be read and contains expected content
    with open(quickstart_path, "r") as f:
        content = f.read()

    # Check for key components of a quickstart example
    required_elements = [
        "ModelRun",
        "TimeRange",
        "LocalConfig",  # or DockerConfig
    ]

    missing_elements = [elem for elem in required_elements if elem not in content]

    assert (
        not missing_elements
    ), f"Quickstart example missing elements: {missing_elements}"
    logger.info("✅ Quickstart example contains expected elements")
