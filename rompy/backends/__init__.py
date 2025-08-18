"""
Backend configuration system for ROMPY.

This module provides Pydantic-based configuration classes for different
execution backends, enabling type-safe and validated backend configurations.
"""

from .config import (
    BackendConfig,
    BaseBackendConfig,
    DockerConfig,
    LocalConfig,
)

__all__ = [
    "BackendConfig",
    "BaseBackendConfig",
    "DockerConfig",
    "LocalConfig",
]
