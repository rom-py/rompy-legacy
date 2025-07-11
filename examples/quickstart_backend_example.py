#!/usr/bin/env python3
"""
ROMPY Backend Quickstart Example

This example demonstrates the essential patterns for using ROMPY backends
to execute models locally and in Docker containers.

Run this example:
    python quickstart_backend_example.py
"""

import logging
import tempfile
from pathlib import Path
from datetime import datetime

# ROMPY imports
from rompy.model import ModelRun
from rompy.core.time import TimeRange
from rompy.backends import LocalConfig, DockerConfig

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def example_local_basic():
    """Example 1: Basic local execution"""
    logger.info("=" * 60)
    logger.info("Example 1: Basic Local Execution")
    logger.info("=" * 60)

    # Create a temporary directory for this example
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a basic model run
        model = ModelRun(
            run_id="local_basic_example",
            period=TimeRange(
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
                interval="1H",
            ),
            output_dir=temp_dir,
            delete_existing=True,
        )

        # Basic local configuration
        config = LocalConfig(
            timeout=1800,  # 30 minutes
            command="echo 'Running basic local model' && ls -la",
        )

        logger.info(f"Configuration: {config}")
        logger.info("Running model...")

        try:
            success = model.run(backend=config)
            logger.info(f"Model run {'succeeded' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Model run failed: {e}")

def example_local_advanced():
    """Example 2: Advanced local execution with environment variables"""
    logger.info("=" * 60)
    logger.info("Example 2: Advanced Local Execution")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        model = ModelRun(
            run_id="local_advanced_example",
            period=TimeRange(
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
                interval="1H",
            ),
            output_dir=temp_dir,
            delete_existing=True,
        )

        # Advanced local configuration
        config = LocalConfig(
            timeout=3600,  # 1 hour
            command="echo 'Advanced local execution' && echo 'Environment:' && env | grep MODEL",
            env_vars={
                "MODEL_THREADS": "4",
                "MODEL_DEBUG": "true",
                "DATA_DIR": "/tmp/data",
                "RESULTS_DIR": "/tmp/results"
            },
            shell=True,
            capture_output=True
        )

        logger.info(f"Configuration: {config}")
        logger.info("Running model with environment variables...")

        try:
            success = model.run(backend=config)
            logger.info(f"Model run {'succeeded' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Model run failed: {e}")

def example_docker_basic():
    """Example 3: Basic Docker execution"""
    logger.info("=" * 60)
    logger.info("Example 3: Basic Docker Execution")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        model = ModelRun(
            run_id="docker_basic_example",
            period=TimeRange(
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
                interval="1H",
            ),
            output_dir=temp_dir,
            delete_existing=True,
        )

        # Basic Docker configuration
        config = DockerConfig(
            image="python:3.9-slim",
            timeout=1800,  # 30 minutes
            cpu=2,
            memory="1g",
            executable="python",
            volumes=[f"{temp_dir}:/app/work:rw"],
            env_vars={
                "PYTHONUNBUFFERED": "1",
                "WORKDIR": "/app/work"
            }
        )

        logger.info(f"Configuration: {config}")
        logger.info("Running model in Docker container...")

        try:
            success = model.run(backend=config)
            logger.info(f"Model run {'succeeded' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Model run failed: {e}")

def example_docker_advanced():
    """Example 4: Advanced Docker execution with volume mounts"""
    logger.info("=" * 60)
    logger.info("Example 4: Advanced Docker Execution")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        model = ModelRun(
            run_id="docker_advanced_example",
            period=TimeRange(
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
                interval="1H",
            ),
            output_dir=temp_dir,
            delete_existing=True,
        )

        # Advanced Docker configuration
        config = DockerConfig(
            image="python:3.9-slim",
            timeout=3600,  # 1 hour
            cpu=4,
            memory="2g",
            executable="python",
            mpiexec="",  # No MPI for this example
            volumes=[
                f"{temp_dir}:/app/work:rw",
                "/tmp:/tmp:rw"
            ],
            env_vars={
                "PYTHONUNBUFFERED": "1",
                "MODEL_THREADS": "4",
                "WORKDIR": "/app/work",
                "TMPDIR": "/tmp"
            },
            user="root",
            remove_container=True
        )

        logger.info(f"Configuration: {config}")
        logger.info("Running model in Docker with advanced configuration...")

        try:
            success = model.run(backend=config)
            logger.info(f"Model run {'succeeded' if success else 'failed'}")
        except Exception as e:
            logger.error(f"Model run failed: {e}")

def example_configuration_validation():
    """Example 5: Configuration validation"""
    logger.info("=" * 60)
    logger.info("Example 5: Configuration Validation")
    logger.info("=" * 60)

    # Valid configuration
    try:
        config = LocalConfig(
            timeout=3600,
            command="echo 'Valid configuration'",
            env_vars={"TEST": "value"}
        )
        logger.info("✅ Valid LocalConfig created successfully")
    except Exception as e:
        logger.error(f"❌ LocalConfig validation failed: {e}")

    # Invalid configuration (timeout too high)
    try:
        config = LocalConfig(
            timeout=100000,  # Too high (max is 86400)
            command="echo 'Invalid configuration'"
        )
        logger.info("❌ This should not succeed")
    except Exception as e:
        logger.info(f"✅ Validation correctly caught error: {e}")

    # Valid Docker configuration
    try:
        config = DockerConfig(
            image="python:3.9-slim",
            timeout=3600,
            cpu=2,
            memory="1g"
        )
        logger.info("✅ Valid DockerConfig created successfully")
    except Exception as e:
        logger.error(f"❌ DockerConfig validation failed: {e}")

    # Invalid Docker configuration (no image or dockerfile)
    try:
        config = DockerConfig(
            timeout=3600,
            cpu=2,
            memory="1g"
            # Missing required image or dockerfile
        )
        logger.info("❌ This should not succeed")
    except Exception as e:
        logger.info(f"✅ Validation correctly caught error: {e}")

def example_config_from_dict():
    """Example 6: Creating configurations from dictionaries (like YAML)"""
    logger.info("=" * 60)
    logger.info("Example 6: Configuration from Dictionary")
    logger.info("=" * 60)

    # Simulate loading from YAML file
    local_config_data = {
        "timeout": 3600,
        "command": "python run_model.py",
        "env_vars": {
            "OMP_NUM_THREADS": "4",
            "MODEL_DEBUG": "true"
        },
        "shell": True,
        "capture_output": True
    }

    docker_config_data = {
        "image": "python:3.9-slim",
        "timeout": 7200,
        "cpu": 4,
        "memory": "2g",
        "volumes": ["/tmp:/tmp:rw"],
        "env_vars": {
            "PYTHONUNBUFFERED": "1",
            "MODEL_THREADS": "4"
        }
    }

    try:
        # Create configurations from dictionaries
        local_config = LocalConfig(**local_config_data)
        docker_config = DockerConfig(**docker_config_data)

        logger.info("✅ Configurations created from dictionaries:")
        logger.info(f"  Local: {local_config}")
        logger.info(f"  Docker: {docker_config}")

    except Exception as e:
        logger.error(f"❌ Configuration creation failed: {e}")

def example_postprocessing():
    """Example 7: Basic postprocessing"""
    logger.info("=" * 60)
    logger.info("Example 7: Model Run with Postprocessing")
    logger.info("=" * 60)

    with tempfile.TemporaryDirectory() as temp_dir:
        model = ModelRun(
            run_id="postprocessing_example",
            period=TimeRange(
                start=datetime(2023, 1, 1),
                end=datetime(2023, 1, 2),
                interval="1H",
            ),
            output_dir=temp_dir,
            delete_existing=True,
        )

        # Run the model
        config = LocalConfig(
            timeout=1800,
            command="echo 'Model run complete' && echo 'Output data' > output.txt",
        )

        logger.info("Running model...")
        try:
            success = model.run(backend=config)
            logger.info(f"Model run {'succeeded' if success else 'failed'}")

            if success:
                # Run postprocessing
                logger.info("Running postprocessing...")
                results = model.postprocess(processor="noop")
                logger.info(f"Postprocessing results: {results}")

        except Exception as e:
            logger.error(f"Model run failed: {e}")

def main():
    """Run all examples"""
    logger.info("ROMPY Backend Quickstart Examples")
    logger.info("=================================")

    # Run examples
    examples = [
        example_local_basic,
        example_local_advanced,
        example_docker_basic,
        example_docker_advanced,
        example_configuration_validation,
        example_config_from_dict,
        example_postprocessing
    ]

    for example in examples:
        try:
            example()
            logger.info("")
        except Exception as e:
            logger.error(f"Example {example.__name__} failed: {e}")
            logger.info("")

    logger.info("=" * 60)
    logger.info("All examples completed!")
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("1. Try creating your own configuration files")
    logger.info("2. Validate them with: rompy backends validate config.yml")
    logger.info("3. Use them with your own models")
    logger.info("4. Check the documentation for advanced features")

if __name__ == "__main__":
    main()
