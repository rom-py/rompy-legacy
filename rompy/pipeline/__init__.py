"""
Local pipeline backend for model execution.

This module provides the local pipeline backend implementation.
"""
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class LocalPipelineBackend:
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



