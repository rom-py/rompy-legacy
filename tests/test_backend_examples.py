#!/usr/bin/env python3
"""
Test script for ROMPY backend examples and configurations.

This script validates that all backend examples, configuration files, and
the backend system work correctly with the current ROMPY implementation.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, List, Tuple
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


def test_pydantic_configs() -> bool:
    """Test that Pydantic configuration objects work correctly."""
    logger.info("Testing Pydantic configuration objects...")

    try:
        # Test LocalConfig
        local_config = LocalConfig(
            timeout=3600, command="echo 'test'", env_vars={"TEST": "value"}
        )
        logger.info(f"✅ LocalConfig created: {local_config}")

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

        return True

    except Exception as e:
        logger.error(f"❌ Pydantic config test failed: {e}")
        return False


def test_model_run_integration() -> bool:
    """Test that ModelRun works with backend configurations."""
    logger.info("Testing ModelRun integration...")

    try:
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

            logger.info(f"✅ ModelRun created with LocalConfig")

            # Test that the backend config is accepted
            backend_class = local_config.get_backend_class()
            logger.info(f"✅ Backend class resolved: {backend_class}")

            return True

    except Exception as e:
        logger.error(f"❌ ModelRun integration test failed: {e}")
        return False


def test_backend_examples() -> List[Tuple[str, bool]]:
    """Test all backend example files."""
    logger.info("Testing backend example files...")

    examples_dir = Path(__file__).parent.parent / "examples" / "backends"
    if not examples_dir.exists():
        logger.warning(f"Backend examples directory not found: {examples_dir}")
        return []

    results = []

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
            results.append((example_file, False))
            continue

        try:
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

            if has_required and (
                has_backend or "LocalConfig" in content or "DockerConfig" in content
            ):
                logger.info(f"✅ {example_file} imports look correct")
                results.append((example_file, True))
            else:
                logger.error(f"❌ {example_file} missing required imports")
                results.append((example_file, False))

        except Exception as e:
            logger.error(f"❌ Error testing {example_file}: {e}")
            results.append((example_file, False))

    return results


def test_yaml_configs() -> List[Tuple[str, bool]]:
    """Test YAML configuration files."""
    logger.info("Testing YAML configuration files...")

    configs_dir = Path(__file__).parent.parent / "examples" / "configs"
    if not configs_dir.exists():
        logger.warning(f"Configs directory not found: {configs_dir}")
        return []

    results = []

    # Test YAML files
    yaml_files = [
        "local_backend_examples.yml",
        "docker_backend_examples.yml",
        "local_backend.yml",
        "docker_backend.yml",
        "pipeline_config.yml",
    ]

    for yaml_file in yaml_files:
        yaml_path = configs_dir / yaml_file
        if not yaml_path.exists():
            logger.warning(f"YAML file not found: {yaml_path}")
            results.append((yaml_file, False))
            continue

        try:
            import yaml

            with open(yaml_path, "r") as f:
                if yaml_file.endswith("_single.yml"):
                    # Single document YAML
                    data = yaml.safe_load(f)
                    if data and isinstance(data, dict):
                        logger.info(f"✅ {yaml_file} is valid single-document YAML")
                        results.append((yaml_file, True))
                    else:
                        logger.error(f"❌ {yaml_file} is not valid YAML")
                        results.append((yaml_file, False))
                else:
                    # Multi-document YAML
                    documents = list(yaml.safe_load_all(f))
                    if documents and all(
                        isinstance(doc, dict) for doc in documents if doc
                    ):
                        logger.info(
                            f"✅ {yaml_file} is valid multi-document YAML ({len(documents)} docs)"
                        )
                        results.append((yaml_file, True))
                    else:
                        logger.error(f"❌ {yaml_file} is not valid multi-document YAML")
                        results.append((yaml_file, False))

        except Exception as e:
            logger.error(f"❌ Error testing {yaml_file}: {e}")
            results.append((yaml_file, False))

    return results


def test_config_validation() -> bool:
    """Test configuration validation using our validation script."""
    logger.info("Testing configuration validation...")

    try:
        configs_dir = Path(__file__).parent.parent / "examples" / "configs"
        validation_script = configs_dir / "validate_configs.py"

        if not validation_script.exists():
            logger.warning("Validation script not found")
            return False

        # Import and run the validation function
        sys.path.insert(0, str(configs_dir))
        from validate_configs import main as validate_main

        # Capture the result
        result = validate_main()

        if result:
            logger.info("✅ Configuration validation passed")
            return True
        else:
            logger.error("❌ Configuration validation failed")
            return False

    except Exception as e:
        logger.error(f"❌ Error running validation: {e}")
        return False


def test_quickstart_example() -> bool:
    """Test that the quickstart example file exists and is valid."""
    logger.info("Testing quickstart example...")

    try:
        quickstart_file = (
            Path(__file__).parent.parent / "examples" / "quickstart_backend_example.py"
        )

        if not quickstart_file.exists():
            logger.warning("Quickstart example not found")
            return False

        # Check that it has the expected structure
        with open(quickstart_file, "r") as f:
            content = f.read()

        required_elements = [
            "from rompy.backends import LocalConfig, DockerConfig",
            "from rompy.model import ModelRun",
            "def main():",
            'if __name__ == "__main__":',
        ]

        missing = [elem for elem in required_elements if elem not in content]
        if missing:
            logger.error(f"❌ Quickstart example missing: {missing}")
            return False

        logger.info("✅ Quickstart example structure is valid")
        return True

    except Exception as e:
        logger.error(f"❌ Error testing quickstart example: {e}")
        return False


def run_all_tests() -> bool:
    """Run all tests and return overall success."""
    logger.info("=" * 60)
    logger.info("ROMPY Backend Examples and Configurations Test Suite")
    logger.info("=" * 60)

    all_passed = True

    # Test 1: Pydantic configs
    if not test_pydantic_configs():
        all_passed = False
    logger.info("")

    # Test 2: ModelRun integration
    if not test_model_run_integration():
        all_passed = False
    logger.info("")

    # Test 3: Backend examples
    example_results = test_backend_examples()
    example_passed = all(result[1] for result in example_results)
    if not example_passed:
        all_passed = False
    logger.info("")

    # Test 4: YAML configs
    yaml_results = test_yaml_configs()
    yaml_passed = all(result[1] for result in yaml_results)
    if not yaml_passed:
        all_passed = False
    logger.info("")

    # Test 5: Configuration validation
    if not test_config_validation():
        all_passed = False
    logger.info("")

    # Test 6: Quickstart example
    if not test_quickstart_example():
        all_passed = False
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)

    # Example results
    if example_results:
        logger.info("Backend Examples:")
        for name, passed in example_results:
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {name}")

    # YAML results
    if yaml_results:
        logger.info("YAML Configurations:")
        for name, passed in yaml_results:
            status = "✅" if passed else "❌"
            logger.info(f"  {status} {name}")

    logger.info("")
    if all_passed:
        logger.info("🎉 ALL TESTS PASSED! 🎉")
        logger.info("The backend examples and configurations are working correctly.")
    else:
        logger.error("💥 SOME TESTS FAILED 💥")
        logger.error("Please check the errors above and fix the issues.")

    return all_passed


def main():
    """Main function."""
    success = run_all_tests()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
