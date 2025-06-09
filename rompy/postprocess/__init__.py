"""
No-op postprocessor for model outputs.

This module provides a basic postprocessor that does nothing.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


# Basic postprocessor that does nothing
class NoopPostprocessor:
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



