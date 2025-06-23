"""
Enhanced unit tests for the improved backend system with validation and error handling.

Tests cover the enhanced LocalRunBackend, NoopPostprocessor, and LocalPipelineBackend
with their new validation, error handling, and logging capabilities.
"""
import os
import pytest
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock, call

from rompy.backends import LocalConfig
from rompy.core.config import BaseConfig
from rompy.core.time import TimeRange
from rompy.model import ModelRun
from rompy.run import LocalRunBackend
from rompy.postprocess import NoopPostprocessor
from rompy.pipeline import LocalPipelineBackend


@pytest.fixture
def model_run(tmp_path):
    """Create a basic ModelRun instance for testing."""
    return ModelRun(
        run_id="test_run",
        period=TimeRange(
            start=datetime(2020, 2, 21, 4),
            end=datetime(2020, 2, 24, 4),
            interval="15M",
        ),
        output_dir=str(tmp_path),
        config=BaseConfig(arg1="foo", arg2="bar"),
    )


@pytest.fixture
def model_run_with_run_method(tmp_path):
    """Create a ModelRun with a config that has a run method."""
    config = BaseConfig(arg1="foo", arg2="bar")
    config.run = MagicMock(return_value=True)

    return ModelRun(
        run_id="test_run_with_method",
        period=TimeRange(
            start=datetime(2020, 2, 21, 4),
            end=datetime(2020, 2, 24, 4),
            interval="15M",
        ),
        output_dir=str(tmp_path),
        config=config,
    )


class TestEnhancedLocalRunBackend:
    """Test the enhanced LocalRunBackend with validation and error handling."""

    def test_run_validation_none_model_run(self):
        """Test validation when model_run is None."""
        backend = LocalRunBackend()
        config = LocalConfig()

        with pytest.raises(ValueError, match="model_run cannot be None"):
            backend.run(None, config)

    def test_run_validation_invalid_model_run(self):
        """Test that run raises ValueError for invalid model_run."""
        backend = LocalRunBackend()
        invalid_model = object()  # Object without run_id attribute
        config = LocalConfig()

        with pytest.raises(ValueError, match="model_run must have a run_id attribute"):
            backend.run(invalid_model, config)

    def test_run_with_command_success(self, model_run, tmp_path):
        """Test successful execution with custom command."""
        backend = LocalRunBackend()

        # Create output directory
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        config = LocalConfig(
            command="echo 'test output' > test_file.txt",
            working_dir=output_dir
        )

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            result = backend.run(model_run, config)

        assert result is True
        assert (output_dir / "test_file.txt").exists()
        assert "test output" in (output_dir / "test_file.txt").read_text()

    def test_run_with_command_failure(self, model_run, tmp_path):
        """Test execution failure with custom command."""
        backend = LocalRunBackend()

        # Create output directory
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        config = LocalConfig(
            command="exit 1",  # Command that will fail
            working_dir=output_dir
        )

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            result = backend.run(model_run, config)

        assert result is False

    def test_run_with_command_timeout(self, model_run, tmp_path):
        """Test execution timeout with custom command."""
        backend = LocalRunBackend()

        # Create output directory
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        config = LocalConfig(
            command="sleep 10",  # Long running command
            timeout=60,  # Minimum allowed timeout
            working_dir=output_dir
        )

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            # Mock subprocess.run to raise TimeoutExpired
            with patch('subprocess.run') as mock_run:
                mock_run.side_effect = subprocess.TimeoutExpired("sleep 10", 60)
                with pytest.raises(TimeoutError, match="Command execution timed out"):
                    backend.run(model_run, config)

    def test_run_with_env_vars(self, model_run, tmp_path):
        """Test execution with environment variables."""
        backend = LocalRunBackend()

        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        config = LocalConfig(
            command="echo $TEST_VAR > env_test.txt",
            env_vars={"TEST_VAR": "test_value"},
            working_dir=output_dir
        )

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            result = backend.run(model_run, config)

        assert result is True
        env_file = output_dir / "env_test.txt"
        assert env_file.exists()
        assert "test_value" in env_file.read_text()

    def test_run_with_config_run_method(self, model_run_with_run_method, tmp_path):
        """Test execution using config.run() method."""
        backend = LocalRunBackend()

        output_dir = tmp_path / model_run_with_run_method.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        config = LocalConfig(working_dir=output_dir)

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            result = backend.run(model_run_with_run_method, config)

        assert result is True
        model_run_with_run_method.config.run.assert_called_once_with(model_run_with_run_method)

    def test_run_with_config_run_method_failure(self, model_run_with_run_method, tmp_path):
        """Test execution failure using config.run() method."""
        backend = LocalRunBackend()

        output_dir = tmp_path / model_run_with_run_method.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Make config.run() raise an exception
        model_run_with_run_method.config.run.side_effect = Exception("Config run failed")

        config = LocalConfig(working_dir=output_dir)

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            result = backend.run(model_run_with_run_method, config)

        assert result is False

    def test_run_with_nonexistent_working_dir(self, model_run, tmp_path):
        """Test execution with nonexistent working directory."""
        backend = LocalRunBackend()

        nonexistent_dir = tmp_path / "nonexistent"

        # LocalConfig validation should catch this, but let's test runtime behavior
        with patch('rompy.model.ModelRun.generate', return_value=str(tmp_path)):
            with pytest.raises(ValueError, match="Working directory does not exist"):
                LocalConfig(
                    command="echo test",
                    working_dir=nonexistent_dir
                )

    def test_execute_config_run_no_method(self, model_run, tmp_path):
        """Test _execute_config_run when config has no run method."""
        backend = LocalRunBackend()
        output_dir = tmp_path / "test"
        output_dir.mkdir()

        result = backend._execute_config_run(model_run, output_dir, {})

        # Should return True but log a warning
        assert result is True

    def test_execute_config_run_non_boolean_return(self, model_run_with_run_method, tmp_path):
        """Test _execute_config_run when config.run() returns non-boolean."""
        backend = LocalRunBackend()
        output_dir = tmp_path / "test"
        output_dir.mkdir()

        # Make config.run() return a string instead of boolean
        model_run_with_run_method.config.run.return_value = "some_string"

        result = backend._execute_config_run(model_run_with_run_method, output_dir, {})

        # Should still return True but log a warning
        assert result is True


class TestEnhancedNoopPostprocessor:
    """Test the enhanced NoopPostprocessor with validation and error handling."""

    def test_process_validation_none_model_run(self):
        """Test that process raises ValueError for None model_run."""
        processor = NoopPostprocessor()

        with pytest.raises(ValueError, match="model_run cannot be None"):
            processor.process(None)

    def test_process_validation_invalid_model_run(self):
        """Test that process raises ValueError for invalid model_run."""
        processor = NoopPostprocessor()
        invalid_model = object()  # Object without run_id attribute

        with pytest.raises(ValueError, match="model_run must have a run_id attribute"):
            processor.process(invalid_model)

    def test_process_with_validation_success(self, model_run, tmp_path):
        """Test successful processing with output validation."""
        processor = NoopPostprocessor()

        # Create output directory with some files
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "output1.txt").write_text("test1")
        (output_dir / "output2.txt").write_text("test2")

        result = processor.process(model_run, validate_outputs=True)

        assert result["success"] is True
        assert result["run_id"] == model_run.run_id
        assert result["validated"] is True
        assert "validation only" in result["message"]

    def test_process_with_validation_missing_dir(self, model_run, tmp_path):
        """Test processing with validation when output directory is missing."""
        processor = NoopPostprocessor()

        # Don't create output directory
        result = processor.process(model_run, validate_outputs=True)

        assert result["success"] is False
        assert "not found" in result["message"]
        assert result["run_id"] == model_run.run_id

    def test_process_without_validation(self, model_run, tmp_path):
        """Test processing without output validation."""
        processor = NoopPostprocessor()

        result = processor.process(model_run, validate_outputs=False)

        assert result["success"] is True
        assert result["run_id"] == model_run.run_id
        assert result["validated"] is False

    def test_process_with_custom_output_dir(self, model_run, tmp_path):
        """Test processing with custom output directory."""
        processor = NoopPostprocessor()

        custom_dir = tmp_path / "custom_output"
        custom_dir.mkdir(parents=True, exist_ok=True)
        (custom_dir / "custom_file.txt").write_text("custom content")

        result = processor.process(
            model_run,
            validate_outputs=True,
            output_dir=str(custom_dir)
        )

        assert result["success"] is True
        assert result["output_dir"] == str(custom_dir)

    def test_process_exception_handling(self, model_run):
        """Test exception handling in process method."""
        processor = NoopPostprocessor()

        # Mock Path to raise an exception
        with patch('rompy.postprocess.Path') as mock_path:
            mock_path.side_effect = Exception("File system error")

            result = processor.process(model_run)

            assert result["success"] is False
            assert "error" in result
            assert "File system error" in result["message"]


class TestEnhancedLocalPipelineBackend:
    """Test the enhanced LocalPipelineBackend with validation and error handling."""

    def test_execute_validation_none_model_run(self):
        """Test that execute raises ValueError for None model_run."""
        backend = LocalPipelineBackend()

        with pytest.raises(ValueError, match="model_run cannot be None"):
            backend.execute(None)

    def test_execute_validation_invalid_model_run(self):
        """Test that execute raises ValueError for invalid model_run."""
        backend = LocalPipelineBackend()
        invalid_model = object()  # Object without run_id attribute

        with pytest.raises(ValueError, match="model_run must have a run_id attribute"):
            backend.execute(invalid_model)

    def test_execute_validation_invalid_run_backend(self, model_run):
        """Test that execute raises ValueError for invalid run_backend."""
        backend = LocalPipelineBackend()

        with pytest.raises(ValueError, match="run_backend must be a non-empty string"):
            backend.execute(model_run, run_backend="")

    def test_execute_validation_invalid_processor(self, model_run):
        """Test that execute raises ValueError for invalid processor."""
        backend = LocalPipelineBackend()

        with pytest.raises(ValueError, match="processor must be a non-empty string"):
            backend.execute(model_run, processor="")

    def test_execute_generate_failure(self, model_run):
        """Test pipeline failure during generate stage."""
        backend = LocalPipelineBackend()

        with patch('rompy.model.ModelRun.generate', side_effect=Exception("Generate failed")):
            result = backend.execute(model_run)

        assert result["success"] is False
        assert result["stage"] == "generate"
        assert "Generate failed" in result["message"]
        assert "generate" not in result["stages_completed"]

    def test_execute_run_failure(self, model_run, tmp_path):
        """Test pipeline failure during run stage."""
        backend = LocalPipelineBackend()

        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            with patch('rompy.model.ModelRun.run', return_value=False):
                result = backend.execute(model_run)

        assert result["success"] is False
        assert result["stage"] == "run"
        assert "generate" in result["stages_completed"]
        assert "run" not in result["stages_completed"]

    def test_execute_run_exception(self, model_run, tmp_path):
        """Test pipeline failure during run stage with exception."""
        backend = LocalPipelineBackend()

        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            with patch('rompy.model.ModelRun.run', side_effect=Exception("Run failed")):
                result = backend.execute(model_run)

        assert result["success"] is False
        assert result["stage"] == "run"
        assert "Run failed" in result["message"]

    def test_execute_postprocess_failure(self, model_run, tmp_path):
        """Test pipeline failure during postprocess stage."""
        backend = LocalPipelineBackend()

        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            with patch('rompy.model.ModelRun.run', return_value=True):
                with patch('rompy.model.ModelRun.postprocess', side_effect=Exception("Postprocess failed")):
                    result = backend.execute(model_run)

        assert result["success"] is False
        assert result["stage"] == "postprocess"
        assert "generate" in result["stages_completed"]
        assert "run" in result["stages_completed"]
        assert "postprocess" not in result["stages_completed"]

    def test_execute_success_complete(self, model_run, tmp_path):
        """Test successful complete pipeline execution."""
        backend = LocalPipelineBackend()

        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        mock_postprocess_result = {"success": True, "message": "Postprocessing done"}

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            with patch('rompy.model.ModelRun.run', return_value=True):
                with patch('rompy.model.ModelRun.postprocess', return_value=mock_postprocess_result):
                    result = backend.execute(
                        model_run,
                        run_backend="local",
                        processor="noop",
                        run_kwargs={"param1": "value1"},
                        process_kwargs={"param2": "value2"}
                    )

        assert result["success"] is True
        assert result["run_success"] is True
        assert result["postprocess_results"] == mock_postprocess_result
        assert "generate" in result["stages_completed"]
        assert "run" in result["stages_completed"]
        assert "postprocess" in result["stages_completed"]
        assert result["message"] == "Pipeline completed successfully"

    def test_execute_with_validation_failure(self, model_run, tmp_path):
        """Test pipeline with stage validation failure."""
        backend = LocalPipelineBackend()

        # Create generate result but not the actual directory
        with patch('rompy.model.ModelRun.generate', return_value=str(tmp_path / "nonexistent")):
            result = backend.execute(model_run, validate_stages=True)

        assert result["success"] is False
        assert result["stage"] == "generate"
        assert "not found after generation" in result["message"]

    def test_execute_with_cleanup_on_failure(self, model_run, tmp_path):
        """Test pipeline with cleanup on failure."""
        backend = LocalPipelineBackend()

        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        test_file = output_dir / "test_file.txt"
        test_file.write_text("test content")

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            with patch('rompy.model.ModelRun.run', return_value=False):
                result = backend.execute(model_run, cleanup_on_failure=True)

        assert result["success"] is False
        # Directory should be cleaned up
        assert not output_dir.exists()

    def test_cleanup_outputs_success(self, model_run, tmp_path):
        """Test successful cleanup of output directory."""
        backend = LocalPipelineBackend()

        # Create output directory with files
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "file1.txt").write_text("content1")
        (output_dir / "file2.txt").write_text("content2")

        model_run.output_dir = tmp_path

        backend._cleanup_outputs(model_run)

        assert not output_dir.exists()

    def test_cleanup_outputs_failure(self, model_run, tmp_path):
        """Test cleanup failure handling."""
        backend = LocalPipelineBackend()

        model_run.output_dir = tmp_path

        # Mock shutil.rmtree to raise an exception
        with patch('shutil.rmtree', side_effect=Exception("Permission denied")):
            # Should not raise exception, just log warning
            backend._cleanup_outputs(model_run)

    def test_execute_postprocess_warning_on_failure(self, model_run, tmp_path):
        """Test pipeline continues when postprocessing reports failure but doesn't raise."""
        backend = LocalPipelineBackend()

        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Postprocessor returns success=False but doesn't raise exception
        mock_postprocess_result = {"success": False, "message": "Postprocessing had issues"}

        with patch('rompy.model.ModelRun.generate', return_value=str(output_dir)):
            with patch('rompy.model.ModelRun.run', return_value=True):
                with patch('rompy.model.ModelRun.postprocess', return_value=mock_postprocess_result):
                    result = backend.execute(model_run)

        # Pipeline should still succeed
        assert result["success"] is True
        assert result["postprocess_results"] == mock_postprocess_result
        assert "postprocess" in result["stages_completed"]
