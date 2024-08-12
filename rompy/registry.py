from typing import Dict, Type
from pydantic import BaseModel


class TypeRegistry:
    """A registry to store the types of the models."""
    _types: Dict[str, Type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str, model_class: Type[BaseModel]):
        cls._types[model_type] = model_class

    @classmethod
    def get_types(cls):
        return cls._types


def register_type(model_type: str):
    """A decorator to register a new type in the registry."""
    def decorator(cls):
        TypeRegistry.register(model_type, cls)
        return cls
    return decorator