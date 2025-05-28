"""
Example 4: Complete Workflow with Custom Backend and Postprocessor

This example demonstrates how to:
1. Create a custom run backend
2. Create a custom postprocessor
3. Use them together in a complete workflow
"""
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from rompy.model import ModelRun
from rompy.core.time import TimeRange
from rompy.postprocess import Postprocessor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 1. Define a custom run backend
class ShellCommandBackend:
    """Custom backend that runs shell commands."""
    
    def run(self, model_run, command: str, **kwargs) -> bool:
        """Run a shell command in the model's output directory.
        
        Args:
            model_run: The ModelRun instance
            command: Shell command to run
            **kwargs: Additional parameters
            
        Returns:
            True if execution was successful, False otherwise
        """
        try:
            # Generate model input files
            model_run.generate()
            
            # Run the command in the model's output directory
            output_dir = Path(model_run.output_dir) / model_run.run_id
            logger.info(f"Running command in {output_dir}: {command}")
            
            result = subprocess.run(
                command,
                shell=True,
                cwd=output_dir,
                check=True,
                capture_output=True,
                text=True,
            )
            
            logger.info(f"Command output:\n{result.stdout}")
            if result.stderr:
                logger.warning(f"Command stderr:\n{result.stderr}")
                    
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with code {e.returncode}: {e.stderr}")
            return False
        except Exception as e:
            logger.exception(f"Error in custom backend: {e}")
            return False

# 2. Define a custom postprocessor
class FileInfoPostprocessor(Postprocessor):
    """Postprocessor that collects information about output files."""
    
    def process(self, model_run, **kwargs) -> Dict[str, Any]:
        """Collect information about output files.
        
        Args:
            model_run: The ModelRun instance
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with file information
        """
        output_dir = Path(model_run.output_dir) / model_run.run_id
        
        if not output_dir.exists():
            return {
                "success": False,
                "message": f"Output directory not found: {output_dir}",
            }
        
        try:
            file_info = {}
            total_size = 0
            
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    file_info[str(file_path.relative_to(output_dir))] = {
                        "size_bytes": file_size,
                        "size_mb": file_size / (1024 * 1024),
                        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    }
                    total_size += file_size
            
            return {
                "success": True,
                "message": f"Collected info for {len(file_info)} files",
                "output_dir": str(output_dir),
                "total_size_bytes": total_size,
                "total_size_mb": total_size / (1024 * 1024),
                "files": file_info,
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to collect file info: {str(e)}",
                "error": str(e),
            }

def main():
    """Run a complete workflow with custom backend and postprocessor."""
    # Create a model run
    model = ModelRun(
        run_id="complete_workflow",
        period=TimeRange(
            start=datetime(2023, 1, 1),
            end=datetime(2023, 1, 2),
            interval="1H",
        ),
        output_dir="./output",
        delete_existing=True,
    )
    
    # 1. Run with custom backend
    logger.info("Running model with custom backend...")
    backend = ShellCommandBackend()
    success = backend.run(
        model,
        command="echo 'Running custom command' && \
                echo 'This is a test file' > output.txt && \
                ls -la > file_list.txt",
    )
    
    if not success:
        logger.error("Model run failed")
        return
    
    # 2. Process with custom postprocessor
    logger.info("Running custom postprocessor...")
    postprocessor = FileInfoPostprocessor()
    results = postprocessor.process(model)
    
    if results["success"]:
        logger.info(f"Successfully processed {len(results['files'])} files")
        logger.info(f"Total output size: {results['total_size_mb']:.2f} MB")
        logger.info("Files created:")
        for file_path, info in results["files"].items():
            logger.info(f"  - {file_path} ({info['size_mb']:.2f} MB)")
    else:
        logger.error(f"Postprocessing failed: {results.get('message', 'Unknown error')}")

if __name__ == "__main__":
    main()
