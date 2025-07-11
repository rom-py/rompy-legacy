"""
Example 2: Docker Run with Custom Configuration

This example demonstrates how to:
1. Configure a model run for Docker execution
2. Use a pre-built Docker image
3. Mount custom volumes and set environment variables
"""
import logging
from datetime import datetime
from pathlib import Path

from rompy.model import ModelRun
from rompy.core.time import TimeRange
from rompy.backends import DockerConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run a model in a Docker container with custom configuration."""
    # Create a model run with a specific configuration
    model = ModelRun(
        run_id="test_docker_run",
        period=TimeRange(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 2),
            interval="1H",
        ),
        output_dir="./output",
        delete_existing=True,
    )

    # Create a data directory for volume mounting
    data_dir = Path("./data")
    data_dir.mkdir(exist_ok=True)

    # Run the model in a Docker container
    logger.info("Running model in Docker...")
    docker_config = DockerConfig(
        image="python:3.9-slim",  # Example image
        # Or build from a Dockerfile:
        # dockerfile="./Dockerfile",
        # build_args={"MODEL_VERSION": "1.0.0"},
        executable="echo",  # Just print the directory contents
        mpiexec="",  # No MPI for this example
        cpu=2,  # Number of CPU cores
        volumes=[  # Mount the data directory
            f"{data_dir.absolute()}:/data:rw",
        ],
        env_vars={
            "MODEL_CONFIG": "production",
            "DATA_DIR": "/data",
        },
    )
    success = model.run(backend=docker_config)

    logger.info(f"Model run {'succeeded' if success else 'failed'}")

if __name__ == "__main__":
    main()
