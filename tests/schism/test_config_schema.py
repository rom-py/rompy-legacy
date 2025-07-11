"""
Test for SCHISM configuration schema serialization with numpy type handling.

This test ensures that the SCHISMConfig class can properly generate JSON schemas
despite having numpy boolean types in its model definitions.
"""

import json
import tempfile
from pathlib import Path

import pytest

from rompy.core.data import DataBlob
from rompy.schism.config import SCHISMConfig
from rompy.schism.grid import SCHISMGrid
from rompy.schism.namelists import NML
from rompy.schism.namelists.param import Param


class TestSCHISMConfigSchema:
    """Test cases for SCHISM configuration schema generation."""

    @pytest.fixture
    def temp_grid_file(self):
        """Create a temporary grid file for testing."""
        with tempfile.NamedTemporaryFile(suffix=".gr3", delete=False) as f:
            f.write(b"test grid content")
            temp_file = f.name

        yield temp_file

        # Cleanup
        import os

        if os.path.exists(temp_file):
            os.unlink(temp_file)

    @pytest.fixture
    def minimal_schism_config(self, temp_grid_file):
        """Create a minimal SCHISM configuration for testing."""
        data_blob = DataBlob(source=temp_grid_file)
        grid = SCHISMGrid(hgrid=data_blob, rough=0.02)
        nml = NML(param=Param())

        config = SCHISMConfig(grid=grid, nml=nml)
        return config

    def test_schism_config_schema_generation(self, minimal_schism_config):
        """Test that SCHISMConfig can generate a valid JSON schema."""
        # This should not raise an exception
        schema = minimal_schism_config.model_json_schema()

        # Verify the schema is a dictionary
        assert isinstance(schema, dict)

        # Verify basic schema structure
        assert "type" in schema or "$defs" in schema

        # Verify schema can be serialized to JSON
        json_str = json.dumps(schema)
        assert isinstance(json_str, str)

        # Verify it can be parsed back
        parsed_schema = json.loads(json_str)
        assert isinstance(parsed_schema, dict)

    def test_schism_config_schema_has_required_fields(self, minimal_schism_config):
        """Test that the generated schema includes required fields."""
        schema = minimal_schism_config.model_json_schema()

        # Check for model_type and grid in schema
        if "properties" in schema:
            properties = schema["properties"]
            assert "model_type" in properties
            assert "grid" in properties

        # Check for required fields - grid is the main required field
        if "required" in schema:
            required = schema["required"]
            assert "grid" in required
            # model_type may or may not be required depending on schema generation

    def test_schism_config_schema_handles_numpy_types(self, minimal_schism_config):
        """Test that numpy types are properly handled in schema generation."""
        # This test ensures that numpy boolean types don't cause schema generation to fail
        schema = minimal_schism_config.model_json_schema()

        # If we get here without an exception, numpy types were handled properly
        assert schema is not None

        # Verify no numpy types leak into the schema
        schema_str = json.dumps(schema)
        assert "numpy.bool" not in schema_str
        assert "numpy.integer" not in schema_str
        assert "numpy.floating" not in schema_str

    def test_schism_config_schema_no_fallback_needed(self, minimal_schism_config):
        """Test that schema generation works without fallback after fixing numpy types."""
        # After fixing the numpy.bool issue, normal schema generation should work
        schema = minimal_schism_config.model_json_schema()

        # Verify schema structure - should be a full schema, not a fallback
        assert isinstance(schema, dict)

        # Should have proper pydantic schema structure with definitions
        assert "$defs" in schema or "definitions" in schema or "properties" in schema

        # Should contain detailed field information, not just basic types
        schema_str = str(schema)
        assert "SCHISMGrid" in schema_str or "grid" in schema_str

    def test_schism_config_schema_json_serializable(self, minimal_schism_config):
        """Test that the schema is fully JSON serializable."""
        schema = minimal_schism_config.model_json_schema()

        # Should not raise any exceptions
        json_str = json.dumps(schema, indent=2)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)

        # Should be able to serialize again
        json_str2 = json.dumps(parsed)
        assert isinstance(json_str2, str)

    def test_schism_config_schema_no_numpy_objects(self, minimal_schism_config):
        """Test that no numpy objects remain in the generated schema."""
        import numpy as np

        schema = minimal_schism_config.model_json_schema()

        def check_no_numpy_objects(obj):
            """Recursively check that no numpy objects exist in the schema."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    # Check key is not a numpy type
                    assert not isinstance(
                        key, (np.bool_, np.integer, np.floating, np.ndarray)
                    )
                    # Check value recursively
                    check_no_numpy_objects(value)
            elif isinstance(obj, list):
                for item in obj:
                    check_no_numpy_objects(item)
            else:
                # Check that the object itself is not a numpy type
                assert not isinstance(
                    obj, (np.bool_, np.integer, np.floating, np.ndarray)
                )

        check_no_numpy_objects(schema)

    def test_schism_config_class_schema_generation(self):
        """Test that the class-level schema generation works."""
        # This should not raise an exception
        schema = SCHISMConfig.model_json_schema()

        # Verify the schema is a dictionary
        assert isinstance(schema, dict)

        # Verify it's JSON serializable
        json_str = json.dumps(schema)
        assert isinstance(json_str, str)
