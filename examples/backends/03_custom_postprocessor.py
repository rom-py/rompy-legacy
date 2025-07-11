"""
Example 3: Custom Postprocessor

This example demonstrates how to:
1. Create a custom postprocessor class
2. Register it for use with the model run
3. Use it to process model outputs
"""

import os
import zipfile
import logging
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

from rompy.model import ModelRun
from rompy.core.time import TimeRange
from rompy.backends import LocalConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ZipOutputsPostprocessor:
    """Custom postprocessor that zips the model outputs.

    This class implements the postprocessor interface by providing a process() method
    that takes a model_run instance and returns a dictionary with results.
    """

    def process(
        self, model_run, output_zip: str = "outputs.zip", **kwargs
    ) -> Dict[str, Any]:
        """Zip the model outputs.

        Args:
            model_run: The ModelRun instance
            output_zip: Name of the output zip file
            **kwargs: Additional parameters

        Returns:
            Dictionary with results
        """
        output_dir = Path(model_run.output_dir) / model_run.run_id
        zip_path = output_dir.parent / output_zip

        try:
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
                for root, _, files in os.walk(output_dir):
                    for file in files:
                        file_path = Path(root) / file
                        arcname = file_path.relative_to(output_dir)
                        zipf.write(file_path, arcname)

            return {
                "success": True,
                "message": f"Outputs zipped to {zip_path}",
                "zip_path": str(zip_path),
                "file_count": len(files),
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to zip outputs: {str(e)}",
                "error": str(e),
            }


def main():
    """Run a model and process outputs with custom postprocessor."""
    # Create a basic model run
    model = ModelRun(
        run_id="test_custom_postprocessor",
        period=TimeRange(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 2),
            interval="1H",
        ),
        output_dir="./output",
        delete_existing=True,
    )

    # Run the model locally
    logger.info("Running model locally...")
    local_config = LocalConfig(
        timeout=3600,  # 1 hour timeout
        command="echo 'Model execution completed' && echo 'Creating test output' > output.txt",
    )
    success = model.run(backend=local_config)

    if not success:
        logger.error("Model run failed")
        return

    # Create and use the custom postprocessor
    logger.info("Running custom postprocessor...")
    postprocessor = ZipOutputsPostprocessor()
    results = postprocessor.process(model, output_zip="model_outputs.zip")

    if results["success"]:
        logger.info(f"Successfully created zip archive: {results['zip_path']}")
        logger.info(f"Zipped {results.get('file_count', 'unknown')} files")
    else:
        logger.error(
            f"Postprocessing failed: {results.get('message', 'Unknown error')}"
        )


if __name__ == "__main__":
    main()
