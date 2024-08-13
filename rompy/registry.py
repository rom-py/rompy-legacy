# registry.py
from typing import Dict, Optional, Type

from pydantic import BaseModel, Field, create_model


class TypeRegistry:
    _types: Dict[str, Type[BaseModel]] = {}
    _model_run: Optional[Type[BaseModel]] = None

    @classmethod
    def register(cls, name: str):
        def decorator(model: Type[BaseModel]):
            cls._types[name] = model
            cls._model_run = None  # Reset ModelRun to force recreation
            return model

        return decorator

    @classmethod
    def get_types(cls):
        return cls._types

    @classmethod
    def create_model_run(cls):
        from typing import Union

        if not cls._types:
            raise ValueError("No models registered")

        config_type = Union[tuple(cls._types.values())]
        return create_model(
            "ModelRun", config=(config_type, Field(..., discriminator="model_type"))
        )
