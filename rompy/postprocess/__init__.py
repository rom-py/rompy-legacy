"""
No-op postprocessor for model outputs.

This module provides a basic postprocessor that does nothing.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class NoopPostprocessor:
    """A postprocessor that does nothing.

    This is a placeholder implementation that simply returns a success message.
    It's useful as a base class or for testing.
    """

    def process(
        self,
        model_run,
        validate_outputs: bool = True,
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Process the output of a model run (does nothing).

        Args:
            model_run: The ModelRun instance whose outputs to process
            validate_outputs: Whether to validate that output directory exists
            output_dir: Override output directory to check (defaults to model_run output)
            **kwargs: Additional parameters (unused)

        Returns:
            Dictionary with processing results

        Raises:
            ValueError: If model_run is invalid
        """
        # Validate input parameters
        if not model_run:
            raise ValueError("model_run cannot be None")

        if not hasattr(model_run, "run_id"):
            raise ValueError("model_run must have a run_id attribute")

        logger.info(f"Starting no-op postprocessing for run_id: {model_run.run_id}")

        try:
            # Determine output directory
            if output_dir:
                check_dir = Path(output_dir)
            else:
                check_dir = Path(model_run.output_dir) / model_run.run_id

            # Validate outputs if requested
            if validate_outputs:
                if not check_dir.exists():
                    logger.warning(f"Output directory does not exist: {check_dir}")
                    return {
                        "success": False,
                        "message": f"Output directory not found: {check_dir}",
                        "run_id": model_run.run_id,
                        "output_dir": str(check_dir),
                    }
                else:
                    # Count files in output directory
                    file_count = sum(1 for f in check_dir.rglob("*") if f.is_file())
                    logger.info(f"Found {file_count} output files in {check_dir}")

            logger.info(
                f"No-op postprocessing completed for run_id: {model_run.run_id}"
            )

            return {
                "success": True,
                "message": "No postprocessing requested - validation only",
                "run_id": model_run.run_id,
                "output_dir": str(check_dir),
                "validated": validate_outputs,
            }

        except Exception as e:
            logger.exception(f"Error in no-op postprocessor: {e}")
            return {
                "success": False,
                "message": f"Error in postprocessor: {str(e)}",
                "run_id": getattr(model_run, "run_id", "unknown"),
                "error": str(e),
            }
