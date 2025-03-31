"""
Base classes for model pipeline execution.

This module provides the interfaces and base classes for executing complete
model pipelines (generate, run, postprocess) using different backends.
"""
import abc
import logging
from typing import Dict, List, Optional, Type, Union, Any

from rompy.utils import load_entry_points

logger = logging.getLogger(__name__)

# Lazy load pipeline backends to avoid circular imports
_PIPELINE_BACKENDS = None

# Forward declaration of the function to avoid circular imports
def get_pipeline_backends():
    """Load and cache pipeline backends from entry points."""
    return _initialize_backends()


class PipelineBackend(abc.ABC):
    """Abstract base class for model pipeline backends.
    
    This class defines the interface for backends that handle the complete
    pipeline: generate, run, and postprocess.
    """
    
    @abc.abstractmethod
    def execute(self, model_run, **kwargs) -> Dict[str, Any]:
        """Execute the complete pipeline for a model run.
        
        Args:
            model_run: The ModelRun instance to execute
            **kwargs: Additional pipeline-specific parameters
            
        Returns:
            Dictionary of results from the pipeline execution
        """
        pass
    
    @classmethod
    def get_backend(cls, backend_name: str) -> Optional[Type["PipelineBackend"]]:
        """Get a pipeline backend by name.
        
        Args:
            backend_name: Name of the backend to retrieve
            
        Returns:
            The requested backend class if available, None otherwise
        """
        backends = get_pipeline_backends()
        if backend_name in backends:
            return backends[backend_name]
        return None
    
    @classmethod
    def list_backends(cls) -> List[str]:
        """List all available pipeline backends.
        
        Returns:
            List of backend names
        """
        return list(get_pipeline_backends().keys())


class LocalPipelineBackend(PipelineBackend):
    """Local pipeline backend that executes the full workflow locally.
    
    This backend uses the existing generate(), run() and postprocess() methods
    to execute the complete pipeline locally.
    """
    
    def execute(self, model_run, 
                run_backend: str = "local",
                processor: str = "noop",
                run_kwargs: Dict = None,
                process_kwargs: Dict = None) -> Dict[str, Any]:
        """Execute the model pipeline locally.
        
        Args:
            model_run: The ModelRun instance to execute
            run_backend: Backend to use for the run stage
            processor: Processor to use for the postprocess stage
            run_kwargs: Additional parameters for the run stage
            process_kwargs: Additional parameters for the postprocess stage
            
        Returns:
            Combined results from the pipeline execution
        """
        # Generate input files
        logger.info(f"Generating input files for {model_run.run_id}")
        model_run.generate()
        
        # Run the model
        run_kwargs = run_kwargs or {}
        logger.info(f"Running model using {run_backend} backend")
        success = model_run.run(backend=run_backend, **run_kwargs)
        
        if not success:
            logger.error("Model run failed")
            return {"success": False, "stage": "run", "message": "Model run failed"}
        
        # Postprocess outputs
        process_kwargs = process_kwargs or {}
        logger.info(f"Postprocessing with {processor}")
        results = model_run.postprocess(processor=processor, **process_kwargs)
        
        # Combine results
        return {
            "success": True,
            "run_success": success,
            "postprocess_results": results
        }


# Function to initialize the backends dictionary
def _initialize_backends():
    """Initialize the backends dictionary.
    
    Returns:
        Dictionary mapping backend names to backend classes
    """
    global _PIPELINE_BACKENDS
    if _PIPELINE_BACKENDS is None:
        # Create an empty dictionary
        backend_dict = {}
        
        # Include the local backend
        backend_dict["local"] = LocalPipelineBackend
        
        # Add backends from entry points
        backend_list = load_entry_points("rompy.pipeline")
        for backend_class in backend_list:
            # Use the class name without 'PipelineBackend' as the key
            backend_name = backend_class.__name__.replace('PipelineBackend', '').lower()
            backend_dict[backend_name] = backend_class
            logger.info(f"Registered pipeline backend: {backend_name}")
        
        _PIPELINE_BACKENDS = backend_dict
    
    return _PIPELINE_BACKENDS
