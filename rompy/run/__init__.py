"""
Local execution backend for model runs.

This module provides the local run backend implementation.
"""
import logging

logger = logging.getLogger(__name__)


# Basic local execution backend
class LocalRunBackend:
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



