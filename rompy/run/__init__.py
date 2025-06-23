"""
Local execution backend for model runs.

This module provides the local run backend implementation.
"""
import logging
import os
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

if TYPE_CHECKING:
    from rompy.backends import LocalConfig

logger = logging.getLogger(__name__)


class LocalRunBackend:
    """Execute models locally using the system's Python interpreter.

    This is the simplest backend that just runs the model directly
    on the local system.
    """

    def run(self, model_run, config: 'LocalConfig', workspace_dir: Optional[str] = None) -> bool:
        """Run the model locally.

        Args:
            model_run: The ModelRun instance to execute
            config: LocalConfig instance with execution parameters
            workspace_dir: Path to the generated workspace directory (if None, will generate)

        Returns:
            True if execution was successful, False otherwise

        Raises:
            ValueError: If model_run is invalid
            TimeoutError: If execution exceeds timeout
        """
        # Validate input parameters
        if not model_run:
            raise ValueError("model_run cannot be None")

        if not hasattr(model_run, 'run_id'):
            raise ValueError("model_run must have a run_id attribute")

        # Use config parameters
        exec_command = config.command
        exec_working_dir = config.working_dir
        exec_env_vars = config.env_vars
        exec_timeout = config.timeout

        logger.debug(f"Using LocalConfig: timeout={exec_timeout}, env_vars={list(exec_env_vars.keys())}")

        logger.info(f"Starting local execution for run_id: {model_run.run_id}")

        try:
            # Use provided workspace or generate if not provided (for backwards compatibility)
            if workspace_dir is None:
                logger.warning("No workspace_dir provided, generating files (this may cause double generation in pipeline)")
                staging_dir = model_run.generate()
                logger.info(f"Model inputs generated in: {staging_dir}")
            else:
                logger.info(f"Using provided workspace directory: {workspace_dir}")
                staging_dir = workspace_dir

            # Set working directory
            if exec_working_dir:
                work_dir = Path(exec_working_dir)
            else:
                work_dir = Path(staging_dir) if staging_dir else Path(model_run.output_dir) / model_run.run_id

            if not work_dir.exists():
                logger.error(f"Working directory does not exist: {work_dir}")
                return False

            # Prepare environment
            env = os.environ.copy()
            if exec_env_vars:
                env.update(exec_env_vars)
                logger.debug(f"Added environment variables: {list(exec_env_vars.keys())}")

            # Execute command or config.run()
            if exec_command:
                success = self._execute_command(exec_command, work_dir, env, exec_timeout)
            else:
                success = self._execute_config_run(model_run, work_dir, env)

            if success:
                logger.info(f"Local execution completed successfully for run_id: {model_run.run_id}")
            else:
                logger.error(f"Local execution failed for run_id: {model_run.run_id}")

            return success

        except TimeoutError:
            logger.error(f"Model execution timed out after {exec_timeout} seconds")
            raise
        except Exception as e:
            logger.exception(f"Failed to run model locally: {e}")
            return False

    def _execute_command(self, command: str, work_dir: Path, env: Dict[str, str],
                        timeout: Optional[int]) -> bool:
        """Execute a shell command.

        Args:
            command: Command to execute
            work_dir: Working directory
            env: Environment variables
            timeout: Execution timeout

        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Executing command: {command}")
        logger.debug(f"Working directory: {work_dir}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=work_dir,
                env=env,
                timeout=timeout,
                capture_output=True,
                text=True,
                check=False
            )

            # Log output
            if result.stdout:
                logger.info(f"Command stdout:\n{result.stdout}")
            if result.stderr:
                if result.returncode == 0:
                    logger.warning(f"Command stderr:\n{result.stderr}")
                else:
                    logger.error(f"Command stderr:\n{result.stderr}")

            if result.returncode == 0:
                logger.debug(f"Command completed successfully")
                return True
            else:
                logger.error(f"Command failed with return code: {result.returncode}")
                return False

        except subprocess.TimeoutExpired:
            logger.error(f"Command timed out after {timeout} seconds")
            raise TimeoutError(f"Command execution timed out after {timeout} seconds")
        except Exception as e:
            logger.exception(f"Error executing command: {e}")
            return False

    def _execute_config_run(self, model_run, work_dir: Path, env: Dict[str, str]) -> bool:
        """Execute the model using config.run() method.

        Args:
            model_run: The ModelRun instance
            work_dir: Working directory
            env: Environment variables

        Returns:
            True if successful, False otherwise
        """
        # Check if config has a run method
        if not hasattr(model_run.config, "run") or not callable(model_run.config.run):
            logger.warning("Model config does not have a run method. Nothing to execute.")
            return True

        logger.info("Executing model using config.run() method")

        try:
            # Set working directory in environment for config.run()
            original_cwd = os.getcwd()
            os.chdir(work_dir)

            # Update environment
            original_env = {}
            for key, value in env.items():
                if key in os.environ:
                    original_env[key] = os.environ[key]
                os.environ[key] = value

            try:
                # Execute the config run method
                result = model_run.config.run(model_run)

                if isinstance(result, bool):
                    return result
                else:
                    logger.warning(f"config.run() returned non-boolean value: {result}")
                    return True

            finally:
                # Restore original environment and directory
                os.chdir(original_cwd)
                for key, value in env.items():
                    if key in original_env:
                        os.environ[key] = original_env[key]
                    else:
                        os.environ.pop(key, None)

        except Exception as e:
            logger.exception(f"Error in config.run(): {e}")
            return False
