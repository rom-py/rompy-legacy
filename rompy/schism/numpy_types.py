"""
Type handling for numpy types in Pydantic models.

This module provides utilities for handling numpy types in Pydantic models
without compromising type safety.
"""

import numpy as np
from typing import Any
from pydantic import field_validator

# Type aliases for type annotations
NumpyBool = bool  # We'll use validators instead of Union types
NumpyInt = int
NumpyFloat = float


# Function to convert numpy types to Python native types
def to_python_type(value: Any) -> Any:
    """Convert numpy types to Python native types."""
    if isinstance(value, np.bool_):
        return bool(value)
    elif isinstance(value, np.integer):
        return int(value)
    elif isinstance(value, np.floating):
        return float(value)
    elif isinstance(value, np.ndarray) and value.size == 1:
        # Handle single-element arrays
        return to_python_type(value.item())
    return value


# Reusable validators for Pydantic models
def numpy_bool_validator(v: Any) -> bool:
    """Validate and convert numpy boolean to Python boolean."""
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def numpy_int_validator(v: Any) -> int:
    """Validate and convert numpy integer to Python integer."""
    if isinstance(v, np.integer):
        return int(v)
    return v


def numpy_float_validator(v: Any) -> float:
    """Validate and convert numpy float to Python float."""
    if isinstance(v, np.floating):
        return float(v)
    return v


# Create decorator functions for easy use in models
def validate_numpy_types(cls):
    """Class decorator to add numpy type validators to all fields."""
    for field_name in cls.model_fields:
        # Add appropriate validators based on field type
        field_info = cls.model_fields[field_name]
        if field_info.annotation in (bool, NumpyBool):
            cls.model_validators[f"_validate_{field_name}"] = field_validator(
                field_name
            )(numpy_bool_validator)
        elif field_info.annotation in (int, NumpyInt):
            cls.model_validators[f"_validate_{field_name}"] = field_validator(
                field_name
            )(numpy_int_validator)
        elif field_info.annotation in (float, NumpyFloat):
            cls.model_validators[f"_validate_{field_name}"] = field_validator(
                field_name
            )(numpy_float_validator)

    return cls
