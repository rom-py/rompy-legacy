"""
Local pipeline backend for model execution.

This module provides the local pipeline backend implementation.
"""
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union

from rompy.backends import LocalConfig, DockerConfig

logger = logging.getLogger(__name__)

class LocalPipelineBackend:
    """Local pipeline backend that executes the full workflow locally.

    This backend uses the existing generate(), run() and postprocess() methods
    to execute the complete pipeline locally.
    """

    def execute(self, model_run,
                run_backend: str = "local",
                processor: str = "noop",
                run_kwargs: Optional[Dict[str, Any]] = None,
                process_kwargs: Optional[Dict[str, Any]] = None,
                cleanup_on_failure: bool = False,
                validate_stages: bool = True,
                **kwargs) -> Dict[str, Any]:
        """Execute the model pipeline locally.

        Args:
            model_run: The ModelRun instance to execute
            run_backend: Backend to use for the run stage ("local" or "docker")
            processor: Processor to use for the postprocess stage
            run_kwargs: Additional parameters for the run stage
            process_kwargs: Additional parameters for the postprocess stage
            cleanup_on_failure: Whether to cleanup outputs on pipeline failure
            validate_stages: Whether to validate each stage before proceeding
            **kwargs: Additional parameters (unused)

        Returns:
            Combined results from the pipeline execution

        Raises:
            ValueError: If model_run is invalid or parameters are invalid
        """
        # Validate input parameters
        if not model_run:
            raise ValueError("model_run cannot be None")

        if not hasattr(model_run, 'run_id'):
            raise ValueError("model_run must have a run_id attribute")

        if not isinstance(run_backend, str) or not run_backend.strip():
            raise ValueError("run_backend must be a non-empty string")

        if not isinstance(processor, str) or not processor.strip():
            raise ValueError("processor must be a non-empty string")

        # Initialize parameters
        run_kwargs = run_kwargs or {}
        process_kwargs = process_kwargs or {}

        logger.info(f"Starting pipeline execution for run_id: {model_run.run_id}")
        logger.info(f"Pipeline configuration: run_backend='{run_backend}', processor='{processor}'")

        pipeline_results = {
            "success": False,
            "run_id": model_run.run_id,
            "stages_completed": [],
            "run_backend": run_backend,
            "processor": processor
        }

        try:
            # Stage 1: Generate input files
            logger.info(f"Stage 1: Generating input files for {model_run.run_id}")

            try:
                staging_dir = model_run.generate()
                pipeline_results["staging_dir"] = str(staging_dir) if staging_dir else None
                pipeline_results["stages_completed"].append("generate")
                logger.info(f"Input files generated successfully in: {staging_dir}")
            except Exception as e:
                logger.exception(f"Failed to generate input files: {e}")
                return {
                    **pipeline_results,
                    "stage": "generate",
                    "message": f"Input file generation failed: {str(e)}",
                    "error": str(e)
                }

            # Validate generation stage
            if validate_stages:
                output_dir = Path(model_run.output_dir) / model_run.run_id
                if not output_dir.exists():
                    logger.error(f"Output directory was not created: {output_dir}")
                    return {
                        **pipeline_results,
                        "stage": "generate",
                        "message": f"Output directory not found after generation: {output_dir}"
                    }

            # Stage 2: Run the model
            logger.info(f"Stage 2: Running model using {run_backend} backend")

            try:
                # Create appropriate backend configuration
                backend_config = self._create_backend_config(run_backend, run_kwargs)

                # Pass the generated workspace directory to avoid duplicate generation
                run_success = model_run.run(backend=backend_config, workspace_dir=staging_dir)
                pipeline_results["run_success"] = run_success

                if not run_success:
                    logger.error("Model run failed")
                    if cleanup_on_failure:
                        self._cleanup_outputs(model_run)
                    return {
                        **pipeline_results,
                        "stage": "run",
                        "message": "Model run failed"
                    }

                pipeline_results["stages_completed"].append("run")
                logger.info("Model run completed successfully")

            except Exception as e:
                logger.exception(f"Error during model run: {e}")
                if cleanup_on_failure:
                    self._cleanup_outputs(model_run)
                return {
                    **pipeline_results,
                    "stage": "run",
                    "message": f"Model run error: {str(e)}",
                    "error": str(e)
                }

            # Stage 3: Postprocess outputs
            logger.info(f"Stage 3: Postprocessing with {processor}")

            try:
                postprocess_results = model_run.postprocess(processor=processor, **process_kwargs)
                pipeline_results["postprocess_results"] = postprocess_results
                pipeline_results["stages_completed"].append("postprocess")

                # Check if postprocessing was successful
                if isinstance(postprocess_results, dict) and not postprocess_results.get("success", True):
                    logger.warning("Postprocessing reported failure but pipeline will continue")

                logger.info("Postprocessing completed")

            except Exception as e:
                logger.exception(f"Error during postprocessing: {e}")
                return {
                    **pipeline_results,
                    "stage": "postprocess",
                    "message": f"Postprocessing error: {str(e)}",
                    "error": str(e)
                }

            # Pipeline completed successfully
            pipeline_results["success"] = True
            pipeline_results["message"] = "Pipeline completed successfully"

            logger.info(f"Pipeline execution completed successfully for run_id: {model_run.run_id}")
            return pipeline_results

        except Exception as e:
            logger.exception(f"Unexpected error in pipeline execution: {e}")
            if cleanup_on_failure:
                self._cleanup_outputs(model_run)
            return {
                **pipeline_results,
                "stage": "pipeline",
                "message": f"Pipeline error: {str(e)}",
                "error": str(e)
            }

    def _cleanup_outputs(self, model_run) -> None:
        """Clean up output files on pipeline failure.

        Args:
            model_run: The ModelRun instance
        """
        try:
            output_dir = Path(model_run.output_dir) / model_run.run_id
            if output_dir.exists():
                logger.info(f"Cleaning up output directory: {output_dir}")
                import shutil
                shutil.rmtree(output_dir)
                logger.info("Cleanup completed")
        except Exception as e:
            logger.warning(f"Failed to cleanup output directory: {e}")

    def _create_backend_config(self, run_backend: str, run_kwargs: Dict[str, Any]):
        """Create appropriate backend configuration from string name and kwargs.

        Args:
            run_backend: Backend name ("local" or "docker")
            run_kwargs: Additional configuration parameters

        Returns:
            Backend configuration object

        Raises:
            ValueError: If backend name is not supported
        """
        if run_backend == "local":
            # Filter kwargs to only include valid LocalConfig fields
            valid_fields = set(LocalConfig.model_fields.keys())
            filtered_kwargs = {k: v for k, v in run_kwargs.items() if k in valid_fields}
            if filtered_kwargs != run_kwargs:
                invalid_fields = set(run_kwargs.keys()) - valid_fields
                logger.warning(f"Ignoring invalid LocalConfig fields: {invalid_fields}")
            return LocalConfig(**filtered_kwargs)
        elif run_backend == "docker":
            # Filter kwargs to only include valid DockerConfig fields
            valid_fields = set(DockerConfig.model_fields.keys())
            filtered_kwargs = {k: v for k, v in run_kwargs.items() if k in valid_fields}
            if filtered_kwargs != run_kwargs:
                invalid_fields = set(run_kwargs.keys()) - valid_fields
                logger.warning(f"Ignoring invalid DockerConfig fields: {invalid_fields}")
            return DockerConfig(**filtered_kwargs)
        else:
            raise ValueError(f"Unsupported backend: {run_backend}. Supported: local, docker")
