"""
Backend configuration classes for ROMPY.

This module provides Pydantic-based configuration classes for different
execution backends. These configurations handle transient execution parameters
while maintaining type safety and validation.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field, field_validator, ConfigDict

if TYPE_CHECKING:
    from rompy.run import LocalRunBackend
    from rompy.run.docker import DockerRunBackend


class BaseBackendConfig(BaseModel, ABC):
    """Base class for all backend configurations.

    This class defines common configuration parameters that apply to all
    backend types, such as timeouts and environment variables.
    """

    timeout: int = Field(
        3600,
        ge=60,
        le=86400,
        description="Maximum execution time in seconds (1 minute to 24 hours)"
    )

    env_vars: Dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables to set during execution"
    )

    working_dir: Optional[Path] = Field(
        None,
        description="Working directory for execution (defaults to model output directory)"
    )

    model_config = ConfigDict(
        validate_assignment=True,
        extra="forbid",  # Don't allow extra fields
        use_enum_values=True
    )

    @field_validator('working_dir')
    @classmethod
    def validate_working_dir(cls, v):
        """Validate working directory exists if specified."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Working directory does not exist: {path}")
            if not path.is_dir():
                raise ValueError(f"Working directory is not a directory: {path}")
        return v

    @field_validator('env_vars')
    @classmethod
    def validate_env_vars(cls, v):
        """Validate environment variables."""
        if not isinstance(v, dict):
            raise ValueError("env_vars must be a dictionary")

        for key, value in v.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise ValueError("Environment variable keys and values must be strings")
            if not key:
                raise ValueError("Environment variable keys cannot be empty")

        return v

    @abstractmethod
    def get_backend_class(self):
        """Return the backend class that should handle this configuration.

        Returns:
            The backend class to use for execution
        """
        pass


class LocalConfig(BaseBackendConfig):
    """Configuration for local execution backend.

    This configuration is used when running models directly on the local system
    using the system's Python interpreter or shell commands.
    """

    command: Optional[str] = Field(
        None,
        description="Optional shell command to run instead of config.run()"
    )

    shell: bool = Field(
        True,
        description="Whether to execute commands through the shell"
    )

    capture_output: bool = Field(
        True,
        description="Whether to capture stdout and stderr"
    )

    def get_backend_class(self):
        """Return the LocalRunBackend class."""
        from rompy.run import LocalRunBackend
        return LocalRunBackend

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "timeout": 7200,
                    "env_vars": {"OMP_NUM_THREADS": "4"},
                    "command": "python run_model.py"
                },
                {
                    "timeout": 3600,
                    "working_dir": "/path/to/model/dir"
                }
            ]
        }
    )


class DockerConfig(BaseBackendConfig):
    """Configuration for Docker execution backend.

    This configuration is used when running models inside Docker containers
    with appropriate resource limits and volume mounts.
    """

    image: Optional[str] = Field(
        None,
        description="Docker image to use (required if not building from dockerfile)",
        pattern=r'^[a-zA-Z0-9._:/-]+$'
    )

    dockerfile: Optional[Path] = Field(
        None,
        description="Path to Dockerfile to build (alternative to image)"
    )

    executable: str = Field(
        "/usr/local/bin/run.sh",
        description="Path to executable inside the container"
    )

    cpu: int = Field(
        1,
        ge=1,
        le=128,
        description="Number of CPU cores to allocate"
    )

    memory: Optional[str] = Field(
        None,
        description="Memory limit (e.g., '2g', '512m')",
        pattern=r'^\d+[kmgKMG]?$'
    )

    mpiexec: str = Field(
        "",
        description="MPI execution command (if using MPI)"
    )

    build_args: Dict[str, str] = Field(
        default_factory=dict,
        description="Arguments to pass to docker build"
    )

    volumes: List[str] = Field(
        default_factory=list,
        description="Additional volumes to mount (format: 'host_path:container_path')"
    )

    remove_container: bool = Field(
        True,
        description="Whether to remove container after execution"
    )

    user: str = Field(
        "root",
        description="User to run as inside the container"
    )

    @field_validator('image', 'dockerfile', mode='before')
    @classmethod
    def validate_image_or_dockerfile(cls, v, info):
        """Validate that either image or dockerfile is provided, but not both."""
        # Skip validation during individual field validation
        # This will be handled by model_validator
        return v

    @field_validator('dockerfile')
    @classmethod
    def validate_dockerfile_exists(cls, v):
        """Validate dockerfile exists if specified."""
        if v is not None:
            path = Path(v)
            if not path.exists():
                raise ValueError(f"Dockerfile does not exist: {path}")
            if not path.is_file():
                raise ValueError(f"Dockerfile is not a file: {path}")
        return v

    @field_validator('volumes')
    @classmethod
    def validate_volumes(cls, v):
        """Validate volume mount specifications."""
        for volume in v:
            if ':' not in volume:
                raise ValueError(f"Volume mount must contain ':' separator: {volume}")

            parts = volume.split(':')
            if len(parts) < 2:
                raise ValueError(f"Volume mount must have host:container format: {volume}")

            host_path = Path(parts[0])
            if not host_path.exists():
                raise ValueError(f"Host path does not exist: {host_path}")

        return v

    @field_validator('memory')
    @classmethod
    def validate_memory_format(cls, v):
        """Validate memory format."""
        if v is not None:
            import re
            if not re.match(r'^\d+[kmg]?$', v.lower()):
                raise ValueError(
                    "Memory must be in format like '2g', '512m', or '1024' (bytes)"
                )
        return v

    def get_backend_class(self):
        """Return the DockerRunBackend class."""
        from rompy.run.docker import DockerRunBackend
        return DockerRunBackend

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        """Add model validator for image/dockerfile validation."""
        super().__pydantic_init_subclass__(**kwargs)

    def model_post_init(self, __context) -> None:
        """Validate that either image or dockerfile is provided, but not both."""
        if not self.image and not self.dockerfile:
            raise ValueError("Either 'image' or 'dockerfile' must be provided")

        if self.image and self.dockerfile:
            raise ValueError("Cannot specify both 'image' and 'dockerfile'")

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "image": "swan:latest",
                    "cpu": 4,
                    "memory": "2g",
                    "timeout": 7200,
                    "volumes": ["/data:/app/data"]
                },
                {
                    "dockerfile": "./docker/Dockerfile",
                    "cpu": 2,
                    "build_args": {"VERSION": "1.0"},
                    "mpiexec": "mpirun"
                }
            ]
        }
    )


# Type alias for all backend configurations
BackendConfig = Union[LocalConfig, DockerConfig]
