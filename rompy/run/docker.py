"""
Docker backend for running models.

This module provides a Docker-based execution backend for rompy models.
"""
import logging
import os
import pathlib
import subprocess
import time
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DockerRunBackend:
    """Execute models inside Docker containers.

    This backend builds Docker images if needed and runs models
    inside containers with appropriate volume mounts.
    """

    def run(self, model_run,
            image: Optional[str] = None,
            dockerfile: Optional[str] = None,
            executable: str = "/usr/local/bin/run.sh",
            mpiexec: str = "",
            cpu: int = 1,
            build_args: Optional[Dict[str, str]] = None,
            volumes: Optional[List[str]] = None,
            env_vars: Optional[Dict[str, str]] = None,
            **kwargs) -> bool:
        """Run the model inside a Docker container.

        Args:
            model_run: The ModelRun instance to execute
            image: Docker image to use (if not building from Dockerfile)
            dockerfile: Path to Dockerfile to build (if not using pre-built image)
            executable: Path to the executable inside the container
            mpiexec: MPI execution command (if using MPI)
            cpu: Number of CPU cores to use
            build_args: Arguments to pass to docker build
            volumes: Additional volumes to mount
            env_vars: Environment variables to pass to the container
            **kwargs: Additional parameters

        Returns:
            True if execution was successful, False otherwise
        """
        # Generate model input files
        model_run.generate()

        try:
            # Build or use a Docker image
            image_name = self._prepare_image(image, dockerfile, build_args)
            if not image_name:
                return False

            # Set up the run command
            run_command = self._get_run_command(executable, mpiexec, cpu)

            # Set up volume mounts
            volume_mounts = self._prepare_volumes(model_run, volumes)

            # Run the Docker container
            success = self._run_container(
                image_name=image_name,
                run_command=run_command,
                volume_mounts=volume_mounts,
                env_vars=env_vars or {}
            )

            return success
        except Exception as e:
            logger.exception(f"Failed to run model in Docker: {e}")
            return False

    def _prepare_image(self,
                      image: Optional[str],
                      dockerfile: Optional[str],
                      build_args: Optional[Dict[str, str]] = None) -> Optional[str]:
        """Prepare the Docker image to use.

        This will either use a pre-built image or build one from a Dockerfile.

        Args:
            image: Docker image to use
            dockerfile: Path to Dockerfile to build
            build_args: Arguments to pass to docker build

        Returns:
            Image name to use, or None if preparation failed
        """
        # If image is provided, use it directly
        if image:
            logger.info(f"Using provided Docker image: {image}")
            return image

        # If Dockerfile is provided, build the image
        if dockerfile:
            # Create a unique image name
            image_name = f"rompy-{int(time.time())}"

            # Build arguments
            build_args_list = []
            if build_args:
                for key, value in build_args.items():
                    build_args_list.extend(["--build-arg", f"{key}={value}"])

            # Build the Docker image
            logger.info(f"Building Docker image {image_name} from {dockerfile}")
            build_cmd = [
                "docker", "build",
                "-t", image_name,
                "-f", dockerfile,
                *build_args_list,
                pathlib.Path(dockerfile).parent.parent
            ]

            try:
                result = subprocess.run(
                    build_cmd,
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                logger.debug(f"Docker build output: {result.stdout}")
                return image_name
            except subprocess.CalledProcessError as e:
                logger.error(f"Docker build failed: {e.stderr}")
                return None

        # If neither is provided, use a default image
        logger.warning("No image or Dockerfile provided, using default rompy image")
        return "rompy/rompy:latest"

    def _get_run_command(self,
                         executable: str,
                         mpiexec: str,
                         cpu: int) -> str:
        """Create the run command to execute inside the container.

        Args:
            executable: Path to the executable
            mpiexec: MPI execution command
            cpu: Number of CPU cores

        Returns:
            Command string to execute
        """
        # Add diagnostic commands to list directory contents and verify input file
        diagnostics = (
            "cd /app/run_id && "
            "echo 'Directory contents:' && "
            "ls -la && "
            "echo 'Checking for INPUT file:' && "
            "if [ -f INPUT ]; then echo 'INPUT file found:'; cat INPUT; else echo 'No INPUT file found'; fi && "
            "echo 'Executing model...'"
        )

        if mpiexec:
            # Add --allow-run-as-root flag for MPI inside Docker containers
            return f"{diagnostics} && {mpiexec} --allow-run-as-root -n {cpu} {executable}"
        return f"{diagnostics} && {executable}"

    def _prepare_volumes(self,
                        model_run,
                        additional_volumes: Optional[List[str]] = None) -> List[str]:
        """Prepare volume mounts for the Docker container.

        Args:
            model_run: The ModelRun instance
            additional_volumes: Additional volumes to mount

        Returns:
            List of volume mount specifications
        """
        # The run directory is always mounted
        # Add :Z for SELinux contexts and proper permissions
        run_dir = model_run.output_dir / model_run.run_id
        volumes = [f"{run_dir.absolute()}:/app/run_id:Z"]

        # Add any additional volumes
        if additional_volumes:
            volumes.extend(additional_volumes)

        return volumes

    def _run_container(self,
                      image_name: str,
                      run_command: str,
                      volume_mounts: List[str],
                      env_vars: Dict[str, str]) -> bool:
        """Run the Docker container with the given configuration.

        Args:
            image_name: Docker image to use
            run_command: Command to run inside the container
            volume_mounts: Volume mounts to set up
            env_vars: Environment variables to pass to the container

        Returns:
            True if execution was successful, False otherwise
        """
        # Set up the Docker command
        docker_cmd = [
            "docker", "run",
            "--rm",  # Remove container after run
            "--user", "root",  # Run as root to avoid permission issues
        ]

        # Add environment variables
        for key, value in env_vars.items():
            docker_cmd.extend(["-e", f"{key}={value}"])

        # Add volume mounts
        for volume in volume_mounts:
            docker_cmd.extend(["-v", volume])

        # Add the image name and command
        docker_cmd.append(image_name)

        # Add bash and -c as separate arguments
        docker_cmd.append("bash")
        docker_cmd.append("-c")

        # Add the run command as a separate argument
        docker_cmd.append(run_command)

        try:
            logger.info(f"Executing: {' '.join(docker_cmd)}")
            # Don't use check=True, so we can see the output even if it fails
            result = subprocess.run(
                docker_cmd,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Always log the output regardless of success/failure
            if result.stdout:
                logger.info(f"Docker stdout: \n{result.stdout}")
            if result.stderr:
                logger.warning(f"Docker stderr: \n{result.stderr}")

            # Check return code manually
            if result.returncode == 0:
                logger.info(f"Model run completed successfully with exit code 0")
                return True
            else:
                logger.error(f"Model run failed with exit code {result.returncode}")
                logger.error(f"Command: {' '.join(docker_cmd)}")
                return False

        except Exception as e:
            logger.error(f"Docker run error: {str(e)}")
            logger.error(f"Command: {' '.join(docker_cmd)}")
            return False
