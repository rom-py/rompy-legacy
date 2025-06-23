"""
Unit tests for the Pydantic backend configuration system.

Tests verify that the new backend configuration classes work correctly,
provide proper validation, and integrate seamlessly with existing backends.
"""

import os
import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from rompy.backends import (
    BackendConfig,
    BaseBackendConfig,
    DockerConfig,
    LocalConfig,
)


class TestBaseBackendConfig:
    """Test the base backend configuration class."""

    def test_cannot_instantiate_abstract_base(self):
        """Test that BaseBackendConfig cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseBackendConfig()

    def test_timeout_validation(self):
        """Test timeout field validation."""
        # Valid timeout values should work in subclasses
        config = LocalConfig(timeout=3600)
        assert config.timeout == 3600

        # Test minimum value
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 60"):
            LocalConfig(timeout=30)

        # Test maximum value
        with pytest.raises(ValidationError, match="Input should be less than or equal to 86400"):
            LocalConfig(timeout=100000)

    def test_env_vars_validation(self):
        """Test environment variables validation."""
        # Valid env vars
        config = LocalConfig(env_vars={"KEY1": "value1", "KEY2": "value2"})
        assert config.env_vars == {"KEY1": "value1", "KEY2": "value2"}

        # Empty dict should be allowed
        config = LocalConfig(env_vars={})
        assert config.env_vars == {}

        # Invalid: non-string keys - Pydantic v2 converts dict keys to strings automatically in some cases
        # So we need to test with a more complex invalid case
        with pytest.raises(ValidationError):
            LocalConfig(env_vars="not_a_dict")

        # Invalid: non-string values - Pydantic v2 might coerce some types
        with pytest.raises(ValidationError):
            LocalConfig(env_vars={"key": ["not", "a", "string"]})

        # Invalid: empty key - custom validator should catch this
        with pytest.raises(ValidationError, match="Environment variable keys cannot be empty"):
            LocalConfig(env_vars={"": "value"})

    def test_working_dir_validation(self):
        """Test working directory validation."""
        with TemporaryDirectory() as tmp_dir:
            # Valid directory
            config = LocalConfig(working_dir=Path(tmp_dir))
            assert config.working_dir == Path(tmp_dir)

            # Non-existent directory should fail
            with pytest.raises(ValidationError, match="Working directory does not exist"):
                LocalConfig(working_dir=Path("/non/existent/path"))

        # Test with file instead of directory
        with TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "test_file.txt"
            file_path.write_text("test")

            with pytest.raises(ValidationError, match="Working directory is not a directory"):
                LocalConfig(working_dir=file_path)

    def test_extra_fields_forbidden(self):
        """Test that extra fields are not allowed."""
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            LocalConfig(invalid_field="value")


class TestLocalConfig:
    """Test the LocalConfig class."""

    def test_default_values(self):
        """Test default values for LocalConfig."""
        config = LocalConfig()

        assert config.timeout == 3600
        assert config.env_vars == {}
        assert config.working_dir is None
        assert config.command is None
        assert config.shell is True
        assert config.capture_output is True

    def test_custom_values(self):
        """Test setting custom values."""
        with TemporaryDirectory() as tmp_dir:
            config = LocalConfig(
                timeout=7200,
                env_vars={"OMP_NUM_THREADS": "4"},
                working_dir=Path(tmp_dir),
                command="python run_model.py",
                shell=False,
                capture_output=False
            )

            assert config.timeout == 7200
            assert config.env_vars == {"OMP_NUM_THREADS": "4"}
            assert config.working_dir == Path(tmp_dir)
            assert config.command == "python run_model.py"
            assert config.shell is False
            assert config.capture_output is False

    def test_get_backend_class(self):
        """Test that get_backend_class returns the correct class."""
        config = LocalConfig()
        backend_class = config.get_backend_class()

        # Should return LocalRunBackend class
        assert backend_class.__name__ == "LocalRunBackend"

    def test_config_examples(self):
        """Test that the schema examples are valid."""
        schema = LocalConfig.model_json_schema()
        examples = schema.get("examples", [])

        for example in examples:
            # Should be able to create config from example
            if "working_dir" in example:
                # Skip working_dir validation for examples
                example = example.copy()
                del example["working_dir"]

            config = LocalConfig(**example)
            assert isinstance(config, LocalConfig)


class TestDockerConfig:
    """Test the DockerConfig class."""

    def test_default_values(self):
        """Test default values for DockerConfig."""
        config = DockerConfig(image="test:latest")

        assert config.timeout == 3600
        assert config.env_vars == {}
        assert config.working_dir is None
        assert config.image == "test:latest"
        assert config.dockerfile is None
        assert config.executable == "/usr/local/bin/run.sh"
        assert config.cpu == 1
        assert config.memory is None
        assert config.mpiexec == ""
        assert config.build_args == {}
        assert config.volumes == []
        assert config.remove_container is True
        assert config.user == "root"

    def test_image_validation(self):
        """Test Docker image validation."""
        # Valid image names (adjusted for actual regex pattern)
        valid_images = [
            "ubuntu:20.04",
            "registry.example.com/repo/image:tag",
            "localhost:5000/image:latest",
            "simple-name",
            "repo/image"
        ]

        for image in valid_images:
            config = DockerConfig(image=image)
            assert config.image == image

    def test_dockerfile_validation(self):
        """Test Dockerfile validation."""
        with TemporaryDirectory() as tmp_dir:
            # Create a valid Dockerfile
            dockerfile_path = Path(tmp_dir) / "Dockerfile"
            dockerfile_path.write_text("FROM ubuntu:20.04\n")

            config = DockerConfig(dockerfile=dockerfile_path)
            assert config.dockerfile == dockerfile_path

            # Non-existent dockerfile should fail
            with pytest.raises(ValidationError, match="Dockerfile does not exist"):
                DockerConfig(dockerfile=Path("/non/existent/Dockerfile"))

    def test_image_or_dockerfile_validation(self):
        """Test that either image or dockerfile must be provided."""
        # Should fail if neither provided
        with pytest.raises(ValidationError, match="Either 'image' or 'dockerfile' must be provided"):
            DockerConfig()

        # Should fail if both provided
        with TemporaryDirectory() as tmp_dir:
            dockerfile_path = Path(tmp_dir) / "Dockerfile"
            dockerfile_path.write_text("FROM ubuntu:20.04\n")

            with pytest.raises(ValidationError, match="Cannot specify both 'image' and 'dockerfile'"):
                DockerConfig(image="test:latest", dockerfile=dockerfile_path)

    def test_cpu_validation(self):
        """Test CPU validation."""
        # Valid CPU values
        config = DockerConfig(image="test:latest", cpu=4)
        assert config.cpu == 4

        # Invalid: too low
        with pytest.raises(ValidationError, match="Input should be greater than or equal to 1"):
            DockerConfig(image="test:latest", cpu=0)

        # Invalid: too high
        with pytest.raises(ValidationError, match="Input should be less than or equal to 128"):
            DockerConfig(image="test:latest", cpu=256)

    def test_memory_validation(self):
        """Test memory format validation."""
        # Valid memory formats (both cases supported now)
        valid_memories = ["1g", "512m", "1024", "2G", "256M"]

        for memory in valid_memories:
            config = DockerConfig(image="test:latest", memory=memory)
            assert config.memory == memory

        # Invalid memory formats
        invalid_memories = ["1x", "invalid", "1.5g", ""]

        for memory in invalid_memories:
            with pytest.raises(ValidationError, match="String should match pattern"):
                DockerConfig(image="test:latest", memory=memory)

    def test_volumes_validation(self):
        """Test volume mount validation."""
        with TemporaryDirectory() as tmp_dir:
            # Valid volume mount
            config = DockerConfig(
                image="test:latest",
                volumes=[f"{tmp_dir}:/app/data"]
            )
            assert config.volumes == [f"{tmp_dir}:/app/data"]

            # Invalid: missing colon
            with pytest.raises(ValidationError, match="Volume mount must contain ':' separator"):
                DockerConfig(image="test:latest", volumes=["invalid_mount"])

            # Invalid: non-existent host path
            with pytest.raises(ValidationError, match="Host path does not exist"):
                DockerConfig(image="test:latest", volumes=["/non/existent:/app/data"])

    def test_get_backend_class(self):
        """Test that get_backend_class returns the correct class."""
        config = DockerConfig(image="test:latest")
        backend_class = config.get_backend_class()

        # Should return DockerRunBackend class
        assert backend_class.__name__ == "DockerRunBackend"

    def test_config_examples(self):
        """Test that the schema examples are valid."""
        schema = DockerConfig.model_json_schema()
        examples = schema.get("examples", [])

        for example in examples:
            # Skip validation for volumes in examples
            if "volumes" in example:
                example = example.copy()
                del example["volumes"]

            # Skip dockerfile validation for examples
            if "dockerfile" in example:
                example = example.copy()
                del example["dockerfile"]
                example["image"] = "test:latest"  # Add image instead

            config = DockerConfig(**example)
            assert isinstance(config, DockerConfig)


class TestConfigToBackendMapping:
    """Test that configurations correctly map to backend classes."""

    def test_local_config_backend_mapping(self):
        """Test LocalConfig maps to correct backend class."""
        config = LocalConfig()
        backend_class = config.get_backend_class()
        assert backend_class.__name__ == "LocalRunBackend"

    def test_docker_config_backend_mapping(self):
        """Test DockerConfig maps to correct backend class."""
        config = DockerConfig(image="test:latest")
        backend_class = config.get_backend_class()
        assert backend_class.__name__ == "DockerRunBackend"


class TestBackendIntegration:
    """Test integration between configurations and backends."""

    @pytest.fixture
    def mock_model_run(self):
        """Create a mock ModelRun instance."""
        model_run = MagicMock()
        model_run.run_id = "test_run_123"
        model_run.output_dir = Path("/tmp/test_output")

        # Create a temporary directory for staging
        import tempfile
        temp_dir = tempfile.mkdtemp()
        model_run.generate.return_value = temp_dir
        model_run.config.run.return_value = True
        return model_run

    def test_local_config_integration(self, mock_model_run):
        """Test LocalConfig integration with LocalRunBackend."""
        import tempfile

        config = LocalConfig(
            timeout=1800,
            env_vars={"TEST_VAR": "test_value"},
            command="echo 'test command'"
        )

        # Get backend class from config
        backend_class = config.get_backend_class()
        backend = backend_class()

        # Create a temporary directory that exists
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update mock to return existing directory
            mock_model_run.generate.return_value = temp_dir

            # Mock subprocess for command execution
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "test output"
                mock_run.return_value.stderr = ""

                # Run with config
                result = backend.run(mock_model_run, config=config)

                assert result is True
                mock_run.assert_called_once()

    def test_docker_config_integration(self, mock_model_run):
        """Test DockerConfig integration with DockerRunBackend."""
        import tempfile

        config = DockerConfig(
            image="test:latest",
            cpu=2,
            memory="1g",
            env_vars={"DOCKER_VAR": "docker_value"}
        )

        # Get backend class from config
        backend_class = config.get_backend_class()
        backend = backend_class()

        # Create a temporary directory that exists
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update mock to return existing directory
            mock_model_run.generate.return_value = temp_dir

            # Mock docker subprocess call
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "docker output"
                mock_run.return_value.stderr = ""

                # Run with config
                result = backend.run(mock_model_run, config=config)

                assert result is True
                mock_run.assert_called_once()

    def test_pydantic_config_integration(self, mock_model_run):
        """Test that backends work with Pydantic config objects only."""
        from rompy.run import LocalRunBackend
        import tempfile

        config = LocalConfig(
            command="echo 'pydantic config test'",
            timeout=1800,
            env_vars={"CONFIG_VAR": "config_value"}
        )

        backend = LocalRunBackend()

        # Create a temporary directory that exists
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update mock to return existing directory
            mock_model_run.generate.return_value = temp_dir

            # Mock subprocess for command execution
            with patch('subprocess.run') as mock_run:
                mock_run.return_value.returncode = 0
                mock_run.return_value.stdout = "test output"
                mock_run.return_value.stderr = ""

                # Run with Pydantic config
                result = backend.run(mock_model_run, config=config)

                assert result is True
                mock_run.assert_called_once()


class TestBackendConfigSerialization:
    """Test serialization and deserialization of backend configurations."""

    def test_local_config_dict_round_trip(self):
        """Test LocalConfig to/from dict conversion."""
        original_config = LocalConfig(
            timeout=7200,
            env_vars={"KEY": "value"},
            command="python script.py"
        )

        # Convert to dict
        config_dict = original_config.model_dump()

        # Create new config from dict
        new_config = LocalConfig(**config_dict)

        assert new_config.timeout == original_config.timeout
        assert new_config.env_vars == original_config.env_vars
        assert new_config.command == original_config.command

    def test_docker_config_json_round_trip(self):
        """Test DockerConfig to/from JSON conversion."""
        original_config = DockerConfig(
            image="test:latest",
            cpu=4,
            memory="2g",
            env_vars={"DOCKER_ENV": "value"}
        )

        # Convert to JSON
        config_json = original_config.model_dump_json()

        # Create new config from JSON
        new_config = DockerConfig.model_validate_json(config_json)

        assert new_config.image == original_config.image
        assert new_config.cpu == original_config.cpu
        assert new_config.memory == original_config.memory
        assert new_config.env_vars == original_config.env_vars

    def test_config_schema_generation(self):
        """Test that configuration schemas can be generated."""
        local_schema = LocalConfig.model_json_schema()
        docker_schema = DockerConfig.model_json_schema()

        # Should be valid JSON schema dictionaries
        assert isinstance(local_schema, dict)
        assert isinstance(docker_schema, dict)
        assert "properties" in local_schema
        assert "properties" in docker_schema

        # Should have required properties
        assert "timeout" in local_schema["properties"]
        assert "image" in docker_schema["properties"]
