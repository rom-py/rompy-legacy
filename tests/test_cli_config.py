"""Tests for CLI configuration loading functionality."""

import json
import os
import tempfile
from unittest.mock import patch

import pytest
import yaml
from click.testing import CliRunner

from rompy.cli import cli, load_config


class TestLoadConfig:
    """Test the load_config function."""

    def test_load_config_from_file_json(self):
        """Test loading configuration from JSON file."""
        config_data = {"model_type": "swan", "run_id": "test_run"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(config_data, f)
            config_path = f.name

        try:
            result = load_config(config_path)
            assert result == config_data
        finally:
            os.unlink(config_path)

    def test_load_config_from_file_yaml(self):
        """Test loading configuration from YAML file."""
        config_data = {"model_type": "swan", "run_id": "test_run"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yml", delete=False) as f:
            yaml.dump(config_data, f)
            config_path = f.name

        try:
            result = load_config(config_path)
            assert result == config_data
        finally:
            os.unlink(config_path)

    def test_load_config_from_string_json(self):
        """Test loading configuration from JSON string."""
        config_data = {"model_type": "swan", "run_id": "test_run"}
        config_string = json.dumps(config_data)

        result = load_config(config_string)
        assert result == config_data

    def test_load_config_from_string_yaml(self):
        """Test loading configuration from YAML string."""
        config_data = {"model_type": "swan", "run_id": "test_run"}
        config_string = yaml.dump(config_data)

        result = load_config(config_string)
        assert result == config_data

    def test_load_config_from_env_default(self):
        """Test loading configuration from default environment variable."""
        config_data = {"model_type": "swan", "run_id": "test_run"}
        config_string = json.dumps(config_data)

        with patch.dict(os.environ, {"ROMPY_CONFIG": config_string}):
            result = load_config("", from_env=True)
            assert result == config_data

    def test_load_config_from_env_custom(self):
        """Test loading configuration from custom environment variable."""
        config_data = {"model_type": "swan", "run_id": "test_run"}
        config_string = json.dumps(config_data)

        with patch.dict(os.environ, {"CUSTOM_CONFIG": config_string}):
            result = load_config("", from_env=True, env_var="CUSTOM_CONFIG")
            assert result == config_data

    def test_load_config_from_env_missing_var(self):
        """Test error when environment variable is not set."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(Exception) as exc_info:
                load_config("", from_env=True)
            assert "ROMPY_CONFIG" in str(exc_info.value)

    def test_load_config_from_env_yaml(self):
        """Test loading YAML configuration from environment variable."""
        config_data = {"model_type": "swan", "run_id": "test_run"}
        config_string = yaml.dump(config_data)

        with patch.dict(os.environ, {"ROMPY_CONFIG": config_string}):
            result = load_config("", from_env=True)
            assert result == config_data

    def test_load_config_invalid_format(self):
        """Test error handling for invalid configuration format."""
        invalid_config = "invalid: yaml: content: ["

        with pytest.raises(Exception):
            load_config(invalid_config)


class TestCLIConfigFromEnv:
    """Test CLI commands with environment variable configuration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_data = {
            "model_type": "swan",
            "run_id": "test_run",
            "output_dir": "/tmp/test_output",
            "period": {"start": "2020-01-01", "end": "2020-01-02"},
        }
        self.config_json = json.dumps(self.config_data)

    def test_validate_with_config_from_env(self):
        """Test validate command with config from environment variable."""
        with patch.dict(os.environ, {"ROMPY_CONFIG": self.config_json}):
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(cli, ["validate", "--config-from-env"])
                assert result.exit_code == 0

    def test_validate_config_source_conflict(self):
        """Test error when both config file and --config-from-env are specified."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name

        try:
            result = self.runner.invoke(
                cli, ["validate", config_path, "--config-from-env"]
            )
            assert result.exit_code != 0
            assert "Cannot specify both" in result.output
        finally:
            os.unlink(config_path)

    def test_validate_no_config_source(self):
        """Test error when neither config file nor --config-from-env is specified."""
        result = self.runner.invoke(cli, ["validate"])
        assert result.exit_code != 0
        assert "Must specify either" in result.output

    def test_generate_with_config_from_env(self):
        """Test generate command with config from environment variable."""
        with patch.dict(os.environ, {"ROMPY_CONFIG": self.config_json}):
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(cli, ["generate", "--config-from-env"])
                assert result.exit_code == 0

    def test_generate_config_source_conflict(self):
        """Test error when both config file and --config-from-env are specified for generate."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name

        try:
            result = self.runner.invoke(
                cli, ["generate", config_path, "--config-from-env"]
            )
            assert result.exit_code != 0
            assert "Cannot specify both" in result.output
        finally:
            os.unlink(config_path)

    def test_run_with_config_from_env(self):
        """Test run command with config from environment variable."""
        backend_config = {"backend_type": "local", "timeout": 3600}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(backend_config, f)
            backend_path = f.name

        try:
            with patch.dict(os.environ, {"ROMPY_CONFIG": self.config_json}):
                with patch("rompy.cli.ModelRun") as mock_model_run:
                    with patch("rompy.cli._load_backend_config"):
                        result = self.runner.invoke(
                            cli,
                            [
                                "run",
                                "--config-from-env",
                                "--backend-config",
                                backend_path,
                                "--dry-run",
                            ],
                        )
                        assert result.exit_code == 0
        finally:
            os.unlink(backend_path)

    def test_pipeline_with_config_from_env(self):
        """Test pipeline command with config from environment variable."""
        with patch.dict(os.environ, {"ROMPY_CONFIG": self.config_json}):
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(
                    cli,
                    [
                        "pipeline",
                        "--config-from-env",
                        "--run-backend",
                        "local",
                        "--processor",
                        "noop",
                    ],
                )
                assert result.exit_code == 0

    def test_postprocess_with_config_from_env(self):
        """Test postprocess command with config from environment variable."""
        with patch.dict(os.environ, {"ROMPY_CONFIG": self.config_json}):
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(
                    cli, ["postprocess", "--config-from-env", "--processor", "noop"]
                )
                assert result.exit_code == 0
                assert mock_model_run.called

    def test_config_from_env_missing_variable(self):
        """Test error when ROMPY_CONFIG environment variable is missing."""
        with patch.dict(os.environ, {}, clear=True):
            result = self.runner.invoke(cli, ["validate", "--config-from-env"])
            assert result.exit_code != 0
            assert "ROMPY_CONFIG" in result.output

    def test_config_from_env_yaml_format(self):
        """Test loading YAML configuration from environment variable."""
        config_yaml = yaml.dump(self.config_data)

        with patch.dict(os.environ, {"ROMPY_CONFIG": config_yaml}):
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(cli, ["validate", "--config-from-env"])
                assert result.exit_code == 0

    def test_config_from_env_invalid_format(self):
        """Test error handling for invalid configuration in environment variable."""
        invalid_config = "invalid: yaml: content: ["

        with patch.dict(os.environ, {"ROMPY_CONFIG": invalid_config}):
            result = self.runner.invoke(cli, ["validate", "--config-from-env"])
            assert result.exit_code != 0


class TestCLIBackwardCompatibility:
    """Test that existing file-based configuration still works."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()
        self.config_data = {
            "model_type": "swan",
            "run_id": "test_run",
            "output_dir": "/tmp/test_output",
            "period": {"start": "2020-01-01", "end": "2020-01-02"},
        }

    def test_postprocess_error_on_missing_config(self):
        """Test error when neither config file nor --config-from-env is specified for postprocess."""
        result = self.runner.invoke(cli, ["postprocess"])
        assert result.exit_code != 0
        assert "Must specify either" in result.output

    def test_postprocess_error_on_both_config_and_env(self):
        """Test error when both config file and --config-from-env are specified for postprocess."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name
        try:
            result = self.runner.invoke(
                cli, ["postprocess", config_path, "--config-from-env"]
            )
            assert result.exit_code != 0
            assert "Cannot specify both" in result.output
        finally:
            os.unlink(config_path)

    def test_validate_with_config_file(self):
        """Test that validate command still works with config files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name

        try:
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(cli, ["validate", config_path])
                assert result.exit_code == 0
        finally:
            os.unlink(config_path)

    def test_generate_with_config_file(self):
        """Test that generate command still works with config files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name

        try:
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(cli, ["generate", config_path])
                assert result.exit_code == 0
        finally:
            os.unlink(config_path)

    def test_run_with_config_file(self):
        """Test that run command still works with config files."""
        backend_config = {"backend_type": "local", "timeout": 3600}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f2:
            json.dump(backend_config, f2)
            backend_path = f2.name

        try:
            with patch("rompy.cli.ModelRun") as mock_model_run:
                with patch("rompy.cli._load_backend_config"):
                    result = self.runner.invoke(
                        cli,
                        [
                            "run",
                            config_path,
                            "--backend-config",
                            backend_path,
                            "--dry-run",
                        ],
                    )
                    assert result.exit_code == 0
        finally:
            os.unlink(config_path)
            os.unlink(backend_path)

    def test_postprocess_with_config_file(self):
        """Test that postprocess command works with config files."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(self.config_data, f)
            config_path = f.name
        try:
            with patch("rompy.cli.ModelRun") as mock_model_run:
                result = self.runner.invoke(
                    cli, ["postprocess", config_path, "--processor", "noop"]
                )
                assert result.exit_code == 0
                assert mock_model_run.called
        finally:
            os.unlink(config_path)


class TestCLIHelpOutput:
    """Test that help output includes information about environment variable option."""

    def setup_method(self):
        """Set up test fixtures."""
        self.runner = CliRunner()

    def test_validate_help_includes_config_from_env(self):
        """Test that validate command help mentions --config-from-env option."""
        result = self.runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0
        assert "--config-from-env" in result.output
        assert "ROMPY_CONFIG" in result.output

    def test_generate_help_includes_config_from_env(self):
        """Test that generate command help mentions --config-from-env option."""
        result = self.runner.invoke(cli, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--config-from-env" in result.output

    def test_run_help_includes_config_from_env(self):
        """Test that run command help mentions --config-from-env option."""
        result = self.runner.invoke(cli, ["run", "--help"])
        assert result.exit_code == 0
        assert "--config-from-env" in result.output

    def test_pipeline_help_includes_config_from_env(self):
        """Test that pipeline command help mentions --config-from-env option."""
        result = self.runner.invoke(cli, ["pipeline", "--help"])
        assert result.exit_code == 0
        assert "--config-from-env" in result.output

    def test_postprocess_help_includes_config_from_env(self):
        """Test that postprocess command help mentions --config-from-env option."""
        result = self.runner.invoke(cli, ["postprocess", "--help"])
        assert result.exit_code == 0
        assert "--config-from-env" in result.output
