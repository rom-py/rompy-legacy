"""
Unit tests for the entry point-based backend system.

Tests verify that the ModelRun methods (run, postprocess, pipeline) work correctly
with the new entry point system instead of the old class-based backend lookup.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime
from pathlib import Path

from rompy.core.config import BaseConfig
from rompy.core.time import TimeRange
from rompy.model import ModelRun, RUN_BACKENDS, POSTPROCESSORS, PIPELINE_BACKENDS


@pytest.fixture
def model_run(tmpdir):
    """Create a basic ModelRun instance for testing."""
    return ModelRun(
        run_id="test_run",
        period=TimeRange(
            start=datetime(2020, 2, 21, 4),
            end=datetime(2020, 2, 24, 4),
            interval="15M",
        ),
        output_dir=str(tmpdir),
        config=BaseConfig(arg1="foo", arg2="bar"),
    )


class TestRunBackends:
    """Test the run backend functionality using entry points."""
    
    def test_run_backends_loaded(self):
        """Test that run backends are loaded from entry points."""
        # Should have at least the local backend
        assert "local" in RUN_BACKENDS
        assert RUN_BACKENDS["local"].__name__ == "LocalRunBackend"
    
    def test_run_with_valid_backend(self, model_run):
        """Test running with a valid backend."""
        # Mock the backend instance and its run method
        mock_backend_class = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend_class.return_value = mock_backend_instance
        mock_backend_instance.run.return_value = True
        
        with patch.dict(RUN_BACKENDS, {"test_backend": mock_backend_class}):
            result = model_run.run(backend="test_backend", param1="value1")
            
            # Verify the backend was instantiated and called correctly
            mock_backend_class.assert_called_once()
            mock_backend_instance.run.assert_called_once_with(model_run, param1="value1")
            assert result is True
    
    def test_run_with_invalid_backend(self, model_run):
        """Test running with an invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown run backend: nonexistent"):
            model_run.run(backend="nonexistent")
    
    def test_run_local_backend(self, model_run):
        """Test running with the local backend."""
        # Mock the config's run method
        mock_run_method = MagicMock(return_value=True)
        model_run.config.run = mock_run_method
        
        with patch('rompy.model.ModelRun.generate') as mock_generate:
            result = model_run.run(backend="local")
            
            # Verify generate was called and config.run was called
            mock_generate.assert_called_once()
            mock_run_method.assert_called_once_with(model_run)
            assert result is True
    
    def test_run_error_handling(self, model_run):
        """Test error handling in run backends."""
        # Mock a backend that raises an exception
        mock_backend_class = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend_class.return_value = mock_backend_instance
        mock_backend_instance.run.side_effect = Exception("Test error")
        
        with patch.dict(RUN_BACKENDS, {"failing_backend": mock_backend_class}):
            # The exception should be handled by the backend itself
            mock_backend_instance.run.side_effect = Exception("Test error")
            with pytest.raises(Exception, match="Test error"):
                model_run.run(backend="failing_backend")


class TestPostprocessors:
    """Test the postprocessor functionality using entry points."""
    
    def test_postprocessors_loaded(self):
        """Test that postprocessors are loaded from entry points."""
        # Should have at least the noop processor
        assert "noop" in POSTPROCESSORS
        assert POSTPROCESSORS["noop"].__name__ == "NoopPostprocessor"
    
    def test_postprocess_with_valid_processor(self, model_run):
        """Test postprocessing with a valid processor."""
        mock_processor_class = MagicMock()
        mock_processor_instance = MagicMock()
        mock_processor_class.return_value = mock_processor_instance
        mock_processor_instance.process.return_value = {"success": True}
        
        with patch.dict(POSTPROCESSORS, {"test_processor": mock_processor_class}):
            result = model_run.postprocess(processor="test_processor", param1="value1")
            
            # Verify the processor was instantiated and called correctly
            mock_processor_class.assert_called_once()
            mock_processor_instance.process.assert_called_once_with(model_run, param1="value1")
            assert result == {"success": True}
    
    def test_postprocess_with_invalid_processor(self, model_run):
        """Test postprocessing with an invalid processor raises ValueError."""
        with pytest.raises(ValueError, match="Unknown postprocessor: nonexistent"):
            model_run.postprocess(processor="nonexistent")
    
    def test_postprocess_noop_processor(self, model_run):
        """Test postprocessing with the noop processor."""
        result = model_run.postprocess(processor="noop")
        
        # Noop processor should return success message
        assert result["success"] is True
        assert "message" in result


class TestPipelineBackends:
    """Test the pipeline backend functionality using entry points."""
    
    def test_pipeline_backends_loaded(self):
        """Test that pipeline backends are loaded from entry points."""
        # Should have at least the local pipeline backend
        assert "local" in PIPELINE_BACKENDS
        assert PIPELINE_BACKENDS["local"].__name__ == "LocalPipelineBackend"
    
    def test_pipeline_with_valid_backend(self, model_run):
        """Test pipeline execution with a valid backend."""
        mock_backend_class = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend_class.return_value = mock_backend_instance
        mock_backend_instance.execute.return_value = {"success": True}
        
        with patch.dict(PIPELINE_BACKENDS, {"test_pipeline": mock_backend_class}):
            result = model_run.pipeline(pipeline_backend="test_pipeline", param1="value1")
            
            # Verify the backend was instantiated and called correctly
            mock_backend_class.assert_called_once()
            mock_backend_instance.execute.assert_called_once_with(model_run, param1="value1")
            assert result == {"success": True}
    
    def test_pipeline_with_invalid_backend(self, model_run):
        """Test pipeline execution with an invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown pipeline backend: nonexistent"):
            model_run.pipeline(pipeline_backend="nonexistent")
    
    def test_pipeline_local_backend(self, model_run):
        """Test pipeline execution with the local backend."""
        with patch('rompy.model.ModelRun.generate') as mock_generate, \
             patch('rompy.model.ModelRun.run') as mock_run, \
             patch('rompy.model.ModelRun.postprocess') as mock_postprocess:
            
            # Set up mock return values
            mock_run.return_value = True
            mock_postprocess.return_value = {"processed": True}
            
            result = model_run.pipeline(
                pipeline_backend="local", 
                run_backend="local", 
                processor="noop"
            )
            
            # Verify all pipeline steps were called
            mock_generate.assert_called_once()
            mock_run.assert_called_once_with(backend="local")
            mock_postprocess.assert_called_once_with(processor="noop")
            
            # Verify the results
            assert result["success"] is True
            assert result["run_success"] is True
            assert result["postprocess_results"] == {"processed": True}


class TestBackendLoading:
    """Test the backend loading functionality."""
    
    @patch('rompy.model.load_entry_points')
    def test_load_backends_with_failures(self, mock_load_entry_points):
        """Test that backend loading handles failures gracefully."""
        from rompy.model import _load_backends
        
        # Mock load_entry_points to raise an exception
        mock_load_entry_points.side_effect = Exception("Entry point loading failed")
        
        # Should not raise an exception, just log warnings
        run_backends, postprocessors, pipeline_backends = _load_backends()
        
        # Should return empty dictionaries
        assert run_backends == {}
        assert postprocessors == {}
        assert pipeline_backends == {}
    
    def test_backend_name_normalization(self):
        """Test that backend names are normalized correctly."""
        # Create mock backend classes with typical naming
        class TestRunBackend:
            pass
        
        class CustomPostprocessor:
            pass
        
        class MyPipelineBackend:
            pass
        
        # Test name normalization
        with patch('rompy.model.load_entry_points') as mock_load:
            mock_load.side_effect = [
                [TestRunBackend],  # run backends
                [CustomPostprocessor],  # postprocessors  
                [MyPipelineBackend]  # pipeline backends
            ]
            
            from rompy.model import _load_backends
            run_backends, postprocessors, pipeline_backends = _load_backends()
            
            # Check that names are normalized correctly
            assert "test" in run_backends
            assert "custom" in postprocessors  
            assert "my" in pipeline_backends


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_run_with_empty_kwargs(self, model_run):
        """Test running with empty kwargs."""
        mock_backend_class = MagicMock()
        mock_backend_instance = MagicMock()
        mock_backend_class.return_value = mock_backend_instance
        mock_backend_instance.run.return_value = True
        
        with patch.dict(RUN_BACKENDS, {"test": mock_backend_class}):
            result = model_run.run(backend="test")
            
            # Should pass empty kwargs
            mock_backend_instance.run.assert_called_once_with(model_run)
            assert result is True
    
    def test_postprocess_with_none_return(self, model_run):
        """Test postprocessing when processor returns None."""
        mock_processor_class = MagicMock()
        mock_processor_instance = MagicMock()
        mock_processor_class.return_value = mock_processor_instance
        mock_processor_instance.process.return_value = None
        
        with patch.dict(POSTPROCESSORS, {"test": mock_processor_class}):
            result = model_run.postprocess(processor="test")
            assert result is None
    
    def test_pipeline_failure_propagation(self, model_run):
        """Test that pipeline failures are properly propagated."""
        with patch('rompy.model.ModelRun.generate') as mock_generate, \
             patch('rompy.model.ModelRun.run') as mock_run:
            
            # Make run fail
            mock_run.return_value = False
            
            result = model_run.pipeline(pipeline_backend="local")
            
            # Should indicate failure
            assert result["success"] is False
            assert result["stage"] == "run"
            assert "failed" in result["message"].lower()