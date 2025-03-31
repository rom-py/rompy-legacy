"""
Base classes for model execution backends.

This module provides the interfaces and base classes for running models
using different backends (local, Docker, etc.).
"""
import abc
import importlib
import logging
from typing import Dict, List, Optional, Type, Union

from rompy.utils import load_entry_points

logger = logging.getLogger(__name__)

# Load run backends from entry points (delayed to avoid circular imports)
_RUN_BACKENDS = None

# Forward declaration of the function to avoid circular imports
def get_run_backends():
    """Load and cache run backends from entry points."""
    return _initialize_backends()


class RunBackend(abc.ABC):
    """Abstract base class for model run backends.
    
    This class defines the interface that all run backends must implement.
    Run backends are responsible for executing model runs in different
    environments (e.g., locally, in Docker, in the cloud).
    """
    
    @abc.abstractmethod
    def run(self, model_run, **kwargs) -> bool:
        """Run the model using this backend.
        
        Args:
            model_run: The ModelRun instance to execute
            **kwargs: Additional backend-specific parameters
            
        Returns:
            True if execution was successful, False otherwise
        """
        pass
    
    @classmethod
    def get_backend(cls, backend_name: str) -> Optional[Type["RunBackend"]]:
        """Get a run backend by name.
        
        Args:
            backend_name: Name of the backend to retrieve
            
        Returns:
            The requested backend class if available, None otherwise
        """
        backends = get_run_backends()
        if backend_name in backends:
            return backends[backend_name]
        return None
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all available run backends.
        
        Returns:
            List of backend names
        """
        return list(get_run_backends().keys())


# Basic local execution backend
class LocalRunBackend(RunBackend):
    """Execute models locally using the system's Python interpreter.
    
    This is the simplest backend that just runs the model directly
    on the local system.
    """
    
    def run(self, model_run, **kwargs) -> bool:
        """Run the model locally.
        
        Args:
            model_run: The ModelRun instance to execute
            **kwargs: Additional parameters (unused)
            
        Returns:
            True if execution was successful, False otherwise
        """
        try:
            # Generate model input files
            model_run.generate()
            
            # Run any model-specific run code
            if hasattr(model_run.config, "run") and callable(model_run.config.run):
                return model_run.config.run(model_run)
            
            logger.warning("Model config does not have a run method. Nothing to execute.")
            return True
        except Exception as e:
            logger.exception(f"Failed to run model: {e}")
            return False


# Function to initialize the backends dictionary
def _initialize_backends():
    """Initialize the backends dictionary.
    
    Returns:
        Dictionary mapping backend names to backend classes
    """
    global _RUN_BACKENDS
    if _RUN_BACKENDS is None:
        # Create an empty dictionary
        backend_dict = {}
        
        # Include the local backend
        backend_dict["local"] = LocalRunBackend
        
        # Add backends from entry points
        backend_list = load_entry_points("rompy.run")
        for backend_class in backend_list:
            # Use the class name without 'RunBackend' as the key
            backend_name = backend_class.__name__.replace('RunBackend', '').lower()
            backend_dict[backend_name] = backend_class
            logger.info(f"Registered run backend: {backend_name}")
        
        _RUN_BACKENDS = backend_dict
    
    return _RUN_BACKENDS
