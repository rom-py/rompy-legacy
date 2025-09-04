#!/usr/bin/env python3
"""
ROMPY Backend Quickstart Tutorial

This tutorial demonstrates how to use ROMPY backends to execute models
locally and in Docker containers. Each example builds on the previous one
to show increasingly sophisticated configurations.

Run this tutorial:
    python quickstart_backend_example.py

This is a hands-on tutorial for learning ROMPY backends.
"""

import logging
import tempfile
from datetime import datetime

from rompy.backends import DockerConfig, LocalConfig
from rompy.core.time import TimeRange
# ROMPY imports
from rompy.model import ModelRun

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def example_local_basic():
    """Example 1: Basic local execution

    This example shows the simplest way to run a model locally using ROMPY backends.
    We create a LocalConfig with basic settings and run a simple command.
    """
    logger.info("=" * 60)
    logger.info("Example 1: Basic Local Execution")
    logger.info("=" * 60)
    logger.info("This example demonstrates the simplest local backend configuration.")
    logger.info("")

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
            command="echo 'Running basic local model' && echo 'Model completed successfully'",
        )

        logger.info(f"LocalConfig created: {config}")
        logger.info("Running model with basic local configuration...")

        try:
            success = model.run(backend=config)
            logger.info(f"✅ Model run {'succeeded' if success else 'failed'}")
            logger.info("Key concepts: LocalConfig, timeout, command execution")
        except Exception as e:
            logger.error(f"❌ Model run failed: {e}")


def example_local_advanced():
    """Example 2: Advanced local execution with environment variables

    This example shows how to use environment variables and advanced LocalConfig options.
    Environment variables are useful for configuring model behavior without changing code.
    """
    logger.info("=" * 60)
    logger.info("Example 2: Advanced Local Execution")
    logger.info("=" * 60)
    logger.info(
        "This example shows environment variables and advanced local configuration."
    )
    logger.info("")

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

        # Advanced local configuration with environment variables
        config = LocalConfig(
            timeout=3600,  # 1 hour
            command="echo 'Advanced local execution' && echo 'Environment variables:' && env | grep MODEL | sort",
            env_vars={
                "MODEL_THREADS": "4",
                "MODEL_DEBUG": "true",
                "MODEL_PRECISION": "double",
                "DATA_DIR": "/tmp/data",
                "RESULTS_DIR": "/tmp/results",
            },
            shell=True,
            capture_output=True,
        )

        logger.info(f"LocalConfig with environment variables: {config}")
        logger.info("Running model with custom environment...")

        try:
            success = model.run(backend=config)
            logger.info(f"✅ Model run {'succeeded' if success else 'failed'}")
            logger.info("Key concepts: env_vars, shell=True, capture_output")
        except Exception as e:
            logger.error(f"❌ Model run failed: {e}")


def example_docker_basic():
    """Example 3: Basic Docker execution

    This example shows how to run models in Docker containers for isolated execution.
    Docker provides reproducible environments and better resource control.
    """
    logger.info("=" * 60)
    logger.info("Example 3: Basic Docker Execution")
    logger.info("=" * 60)
    logger.info("This example demonstrates running models in Docker containers.")
    logger.info("")

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
            executable="python -c \"print('Docker model execution complete'); import os; print(f'Working in: {os.getcwd()}')\"",
            volumes=[f"{temp_dir}:/app/work:rw"],
            env_vars={"PYTHONUNBUFFERED": "1", "WORKDIR": "/app/work"},
        )

        logger.info(f"DockerConfig created: {config}")
        logger.info("Running model in Docker container...")

        try:
            success = model.run(backend=config)
            logger.info(f"✅ Model run {'succeeded' if success else 'failed'}")
            logger.info("Key concepts: DockerConfig, image, cpu/memory limits, volumes")
        except Exception as e:
            logger.error(f"❌ Model run failed: {e}")


def example_docker_advanced():
    """Example 4: Advanced Docker execution with volume mounts

    This example shows advanced Docker configuration with multiple volumes,
    resource allocation, and container management settings.
    """
    logger.info("=" * 60)
    logger.info("Example 4: Advanced Docker Execution")
    logger.info("=" * 60)
    logger.info(
        "This example shows advanced Docker configuration with multiple volumes."
    )
    logger.info("")

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
            executable='python -c "print(\'Advanced Docker execution\'); import os; print(f\'Threads: {os.environ.get(\\"MODEL_THREADS\\", \\"unknown\\")}\')"',
            mpiexec="",  # No MPI for this example
            volumes=[f"{temp_dir}:/app/work:rw", "/tmp:/tmp:rw"],
            env_vars={
                "PYTHONUNBUFFERED": "1",
                "MODEL_THREADS": "4",
                "MODEL_PRECISION": "double",
                "WORKDIR": "/app/work",
                "TMPDIR": "/tmp",
            },
            user="root",
            remove_container=True,
        )

        logger.info(f"DockerConfig with advanced settings: {config}")
        logger.info("Running model with advanced Docker configuration...")

        try:
            success = model.run(backend=config)
            logger.info(f"✅ Model run {'succeeded' if success else 'failed'}")
            logger.info(
                "Key concepts: multiple volumes, resource allocation, container management"
            )
        except Exception as e:
            logger.error(f"❌ Model run failed: {e}")


def example_configuration_validation():
    """Example 5: Configuration validation

    This example demonstrates ROMPY's built-in configuration validation.
    The Pydantic models catch configuration errors before runtime.
    """
    logger.info("=" * 60)
    logger.info("Example 5: Configuration Validation")
    logger.info("=" * 60)
    logger.info("This example shows how ROMPY validates configurations automatically.")
    logger.info("")

    # Valid local configuration
    try:
        LocalConfig(
            timeout=3600,
            command="echo 'Valid configuration'",
            env_vars={"TEST": "value"},
        )
        logger.info("✅ Valid LocalConfig created successfully")
    except Exception as e:
        logger.error(f"❌ LocalConfig validation failed: {e}")

    # Invalid configuration (timeout too high)
    logger.info("Testing invalid configuration (timeout too high)...")
    try:
        LocalConfig(
            timeout=100000,  # Too high (max is 86400)
            command="echo 'Invalid configuration'",
        )
        logger.info("❌ This should not succeed")
    except Exception as e:
        logger.info(f"✅ Validation correctly caught error: {e}")

    # Valid Docker configuration
    try:
        DockerConfig(image="python:3.9-slim", timeout=3600, cpu=2, memory="1g")
        logger.info("✅ Valid DockerConfig created successfully")
    except Exception as e:
        logger.error(f"❌ DockerConfig validation failed: {e}")

    # Invalid Docker configuration (no image or dockerfile)
    logger.info("Testing invalid Docker configuration (missing image)...")
    try:
        DockerConfig(
            timeout=3600,
            cpu=2,
            memory="1g",
            # Missing required image or dockerfile
        )
        logger.info("❌ This should not succeed")
    except Exception as e:
        logger.info(f"✅ Validation correctly caught error: {e}")

    logger.info(
        "Key concepts: Pydantic validation, error handling, configuration safety"
    )


def example_config_from_dict():
    """Example 6: Creating configurations from dictionaries (like YAML)

    This example shows how to create configurations from dictionaries,
    which is how YAML/JSON configuration files are loaded in practice.
    """
    logger.info("=" * 60)
    logger.info("Example 6: Configuration from Dictionary")
    logger.info("=" * 60)
    logger.info(
        "This example shows creating configurations from dictionaries (like YAML files)."
    )
    logger.info("")

    # Simulate loading from YAML file
    local_config_data = {
        "timeout": 3600,
        "command": "python run_model.py",
        "env_vars": {"OMP_NUM_THREADS": "4", "MODEL_DEBUG": "true"},
        "shell": True,
        "capture_output": True,
    }

    docker_config_data = {
        "image": "python:3.9-slim",
        "timeout": 7200,
        "cpu": 4,
        "memory": "2g",
        "volumes": ["/tmp:/tmp:rw"],
        "env_vars": {"PYTHONUNBUFFERED": "1", "MODEL_THREADS": "4"},
    }

    try:
        # Create configurations from dictionaries
        local_config = LocalConfig(**local_config_data)
        docker_config = DockerConfig(**docker_config_data)

        logger.info("✅ Configurations created from dictionaries:")
        logger.info(f"  Local config timeout: {local_config.timeout}s")
        logger.info(f"  Docker config CPU: {docker_config.cpu} cores")
        logger.info(f"  Docker config memory: {docker_config.memory}")
        logger.info("Key concepts: dictionary unpacking, YAML/JSON compatibility")

    except Exception as e:
        logger.error(f"❌ Configuration creation failed: {e}")


def example_postprocessing():
    """Example 7: Complete workflow with postprocessing

    This example shows a complete workflow: model run followed by postprocessing.
    This demonstrates the full ROMPY pipeline from execution to results processing.
    """
    logger.info("=" * 60)
    logger.info("Example 7: Complete Workflow with Postprocessing")
    logger.info("=" * 60)
    logger.info(
        "This example demonstrates a complete workflow: model run + postprocessing."
    )
    logger.info("")

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
            command="echo 'Model simulation complete' && echo 'Creating output file...' && echo 'Sample output data' > output.txt",
        )

        logger.info("Step 1: Running model...")
        try:
            success = model.run(backend=config)
            logger.info(f"✅ Model run {'succeeded' if success else 'failed'}")

            if success:
                # Run postprocessing
                logger.info("Step 2: Running postprocessing...")
                results = model.postprocess(processor="noop")
                logger.info(f"✅ Postprocessing completed: {results}")
                logger.info(
                    "Key concepts: complete workflow, model.run() + model.postprocess()"
                )

        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")


def main():
    """Run all tutorial examples"""
    logger.info("🚀 ROMPY Backend Tutorial")
    logger.info("========================")
    logger.info(
        "This tutorial covers essential ROMPY backend concepts through practical examples."
    )
    logger.info(
        "Each example builds on the previous one to show increasingly sophisticated usage."
    )
    logger.info("")

    # Run examples
    examples = [
        example_local_basic,
        example_local_advanced,
        example_docker_basic,
        example_docker_advanced,
        example_configuration_validation,
        example_config_from_dict,
        example_postprocessing,
    ]

    completed_examples = 0
    for i, example in enumerate(examples, 1):
        try:
            logger.info(f"Running example {i}/{len(examples)}...")
            example()
            completed_examples += 1
            logger.info("")
        except Exception as e:
            logger.error(f"❌ Example {example.__name__} failed: {e}")
            logger.info("")

    logger.info("=" * 60)
    logger.info(
        f"🎉 Tutorial completed! ({completed_examples}/{len(examples)} examples ran successfully)"
    )
    logger.info("=" * 60)
    logger.info("What you learned:")
    logger.info("• Basic local execution with LocalConfig")
    logger.info("• Environment variables and advanced local settings")
    logger.info("• Docker containers with DockerConfig")
    logger.info("• Resource allocation and volume mounting")
    logger.info("• Configuration validation and error handling")
    logger.info("• Loading configurations from dictionaries/YAML")
    logger.info("• Complete workflows with postprocessing")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Explore the configuration files in examples/configs/")
    logger.info("2. Try: rompy backends validate examples/configs/local_backend.yml")
    logger.info("3. Read the documentation for advanced features")
    logger.info("4. Create your own configuration files for your models")


if __name__ == "__main__":
    main()
