"""
Base classes for model postprocessing.

This module provides the interfaces and base classes for postprocessing model outputs
using different backends and methods.
"""
import abc
import importlib
import logging
from typing import Dict, List, Optional, Type, Union, Any

from rompy.utils import load_entry_points

logger = logging.getLogger(__name__)

# Lazy load postprocessors to avoid circular imports
_POSTPROCESSORS = None

# Forward declaration of the function to avoid circular imports
def get_postprocessors():
    """Load and cache postprocessors from entry points."""
    return _initialize_postprocessors()


class Postprocessor(abc.ABC):
    """Abstract base class for model output postprocessors.
    
    This class defines the interface that all postprocessors must implement.
    Postprocessors are responsible for processing and possibly transforming
    model outputs (e.g., registering with data services, creating visualizations).
    """
    
    @abc.abstractmethod
    def process(self, model_run, **kwargs) -> Dict[str, Any]:
        """Process the output of a model run.
        
        Args:
            model_run: The ModelRun instance whose outputs to process
            **kwargs: Additional processor-specific parameters
            
        Returns:
            Dictionary of results from the processing
        """
        pass
    
    @classmethod
    def get_processor(cls, processor_name: str) -> Optional[Type["Postprocessor"]]:
        """Get a postprocessor by name.
        
        Args:
            processor_name: Name of the postprocessor to retrieve
            
        Returns:
            The requested postprocessor class if available, None otherwise
        """
        processors = get_postprocessors()
        if processor_name in processors:
            return processors[processor_name]
        return None
    
    @classmethod
    def list_processors(cls) -> List[str]:
        """List all available postprocessors.
        
        Returns:
            List of postprocessor names
        """
        return list(get_postprocessors().keys())


# Basic postprocessor that does nothing
class NoopPostprocessor(Postprocessor):
    """A postprocessor that does nothing.
    
    This is a placeholder implementation that simply returns a success message.
    It's useful as a base class or for testing.
    """
    
    def process(self, model_run, **kwargs) -> Dict[str, Any]:
        """Process the output of a model run (does nothing).
        
        Args:
            model_run: The ModelRun instance whose outputs to process
            **kwargs: Additional parameters (unused)
            
        Returns:
            Dictionary with a success flag
        """
        logger.info(f"No postprocessing performed for {model_run.run_id}")
        return {"success": True, "message": "No postprocessing requested"}


# Function to initialize the postprocessors dictionary
def _initialize_postprocessors():
    """Initialize the postprocessors dictionary.
    
    Returns:
        Dictionary mapping processor names to processor classes
    """
    global _POSTPROCESSORS
    if _POSTPROCESSORS is None:
        # Create an empty dictionary
        processor_dict = {}
        
        # Include the noop processor
        processor_dict["noop"] = NoopPostprocessor
        
        # Add processors from entry points
        processor_list = load_entry_points("rompy.postprocess")
        for processor_class in processor_list:
            # Use the class name without 'Postprocessor' as the key
            processor_name = processor_class.__name__.replace('Postprocessor', '').lower()
            processor_dict[processor_name] = processor_class
            logger.info(f"Registered postprocessor: {processor_name}")
        
        _POSTPROCESSORS = processor_dict
    
    return _POSTPROCESSORS
