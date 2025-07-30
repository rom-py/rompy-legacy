"""
Integration tests for ModelRun with the new Pydantic backend system.

These tests verify that the ModelRun.run() method works correctly with
BackendConfig instances instead of the old string-based backend system.
"""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rompy.backends import DockerConfig, LocalConfig
from rompy.core.config import BaseConfig
from rompy.core.time import TimeRange
from rompy.model import ModelRun


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


class TestModelRunPydanticIntegration:
    """Test ModelRun.run() with the new Pydantic backend system."""

    def test_run_with_local_config(self, model_run, tmp_path):
        """Test ModelRun.run() with LocalConfig instance."""
        # Create output directory
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create LocalConfig
        config = LocalConfig(
            command="echo 'test output' > test_file.txt",
            working_dir=output_dir,
            timeout=3600,
        )

        with patch("rompy.model.ModelRun.generate", return_value=str(output_dir)):
            result = model_run.run(backend=config)

        assert result is True
        assert (output_dir / "test_file.txt").exists()
        assert "test output" in (output_dir / "test_file.txt").read_text()

    def test_run_with_local_config_using_config_run_method(
        self, model_run_with_run_method, tmp_path
    ):
        """Test ModelRun.run() with LocalConfig that uses config.run() method."""
        # Create output directory
        output_dir = tmp_path / model_run_with_run_method.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create LocalConfig without command (will use config.run())
        config = LocalConfig(working_dir=output_dir)

        with patch("rompy.model.ModelRun.generate", return_value=str(output_dir)):
            result = model_run_with_run_method.run(backend=config)

        assert result is True
        # Verify config.run() was called
        model_run_with_run_method.config.run.assert_called_once_with(
            model_run_with_run_method
        )

    def test_run_with_docker_config(self, model_run, tmp_path):
        """Test ModelRun.run() with DockerConfig instance."""
        # Mock DockerRunBackend since we don't want to actually run Docker
        with patch("rompy.run.docker.DockerRunBackend") as mock_docker_backend_class:
            mock_backend_instance = MagicMock()
            mock_backend_instance.run.return_value = True
            mock_docker_backend_class.return_value = mock_backend_instance

            # Create DockerConfig
            config = DockerConfig(
                image="test-image:latest", cpu=2, memory="1g", timeout=7200
            )

            result = model_run.run(backend=config)

            assert result is True
            # Verify DockerRunBackend was instantiated and called
            mock_docker_backend_class.assert_called_once()
            mock_backend_instance.run.assert_called_once_with(
                model_run, config=config, workspace_dir=None
            )

    def test_run_with_invalid_backend_type(self, model_run):
        """Test ModelRun.run() raises TypeError for invalid backend types."""
        # Invalid types should raise TypeError
        with pytest.raises(TypeError, match="Backend must be a BackendConfig instance"):
            model_run.run(backend="invalid_string")

        with pytest.raises(TypeError, match="Backend must be a BackendConfig instance"):
            model_run.run(backend={"invalid": "dict"})

        with pytest.raises(TypeError, match="Backend must be a BackendConfig instance"):
            model_run.run(backend=123)

    def test_run_with_local_config_env_vars(self, model_run, tmp_path):
        """Test ModelRun.run() with LocalConfig and environment variables."""
        # Create output directory
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create LocalConfig with environment variables
        config = LocalConfig(
            command="echo $TEST_VAR > env_test.txt",
            working_dir=output_dir,
            env_vars={"TEST_VAR": "hello_world"},
        )

        with patch("rompy.model.ModelRun.generate", return_value=str(output_dir)):
            result = model_run.run(backend=config)

        assert result is True
        env_file = output_dir / "env_test.txt"
        assert env_file.exists()
        assert "hello_world" in env_file.read_text()

    def test_run_backend_failure_propagation(self, model_run, tmp_path):
        """Test that backend failures are properly propagated."""
        # Create output directory
        output_dir = tmp_path / model_run.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create LocalConfig with failing command
        config = LocalConfig(
            command="exit 1", working_dir=output_dir  # Command that will fail
        )

        with patch("rompy.model.ModelRun.generate", return_value=str(output_dir)):
            result = model_run.run(backend=config)

        assert result is False

    def test_run_backend_exception_handling(self, model_run):
        """Test that backend exceptions are handled gracefully."""
        # Mock LocalRunBackend to raise an exception
        with patch("rompy.run.LocalRunBackend") as mock_backend_class:
            mock_backend_instance = MagicMock()
            mock_backend_instance.run.side_effect = Exception("Backend error")
            mock_backend_class.return_value = mock_backend_instance

            config = LocalConfig()

            # Exception should be raised by the backend
            with pytest.raises(Exception, match="Backend error"):
                model_run.run(backend=config)

    def test_local_config_validation_in_modelrun_context(self, model_run, tmp_path):
        """Test LocalConfig validation works in ModelRun context."""
        # Test timeout validation
        with pytest.raises(
            ValueError, match="Input should be greater than or equal to 60"
        ):
            LocalConfig(timeout=30)  # Below minimum

        # Test working directory validation
        nonexistent_dir = tmp_path / "nonexistent"
        with pytest.raises(ValueError, match="Working directory does not exist"):
            LocalConfig(working_dir=nonexistent_dir)

        # Test valid config creation
        valid_dir = tmp_path / "valid"
        valid_dir.mkdir()
        config = LocalConfig(working_dir=valid_dir, timeout=3600)
        assert config.working_dir == valid_dir
        assert config.timeout == 3600

    def test_docker_config_validation_in_modelrun_context(self, model_run, tmp_path):
        """Test DockerConfig validation works in ModelRun context."""
        # Test image validation
        config = DockerConfig(image="valid-image:tag")
        assert config.image == "valid-image:tag"

        # Test dockerfile validation
        dockerfile = Path("Dockerfile")
        dockerfile_full_path = tmp_path / "Dockerfile"
        dockerfile_full_path.write_text("FROM python:3.9")
        config = DockerConfig(dockerfile=dockerfile, build_context=tmp_path)
        assert config.dockerfile == dockerfile

        # Test image or dockerfile requirement
        with pytest.raises(
            ValueError, match="Either 'image' or 'dockerfile' must be provided"
        ):
            DockerConfig()

        # Test mutual exclusion of image and dockerfile
        with pytest.raises(
            ValueError, match="Cannot specify both 'image' and 'dockerfile'"
        ):
            DockerConfig(image="test:latest", dockerfile=dockerfile)

    def test_backend_config_type_safety(self, model_run):
        """Test that the backend config system provides proper type safety."""
        # LocalConfig should work
        local_config = LocalConfig(command="echo test")
        assert isinstance(local_config, LocalConfig)
        assert local_config.get_backend_class().__name__ == "LocalRunBackend"

        # DockerConfig should work
        docker_config = DockerConfig(image="test:latest")
        assert isinstance(docker_config, DockerConfig)
        assert docker_config.get_backend_class().__name__ == "DockerRunBackend"

        # Both should be BackendConfig instances
        from rompy.backends import BackendConfig

        assert isinstance(local_config, BackendConfig)
        assert isinstance(docker_config, BackendConfig)
