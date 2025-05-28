"""
Example 1: Basic Local Run with No-Op Postprocessing

This example demonstrates how to:
1. Create a basic model run configuration
2. Run the model locally
3. Perform no-op postprocessing
"""
import logging
from datetime import datetime

from rompy.model import ModelRun
from rompy.core.time import TimeRange

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Run a basic model locally with no-op postprocessing."""
    # Create a basic model run
    model = ModelRun(
        run_id="test_local_run",
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
    success = model.run(backend="local")
    logger.info(f"Model run {'succeeded' if success else 'failed'}")

    # Postprocess the results (no-op by default)
    logger.info("Running postprocessing...")
    results = model.postprocess(processor="noop")
    logger.info(f"Postprocessing results: {results}")

if __name__ == "__main__":
    main()
