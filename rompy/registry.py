from typing import Dict, List, Optional, Type
from typing import Union as TypingUnion

from pydantic import BaseModel, create_model, validator


class Registry:
    _config_types: Dict[str, Type[BaseModel]] = {}
    _datatypes: Dict[str, Dict[str, Type[BaseModel]]] = {}
    _model_run: Optional[Type[BaseModel]] = None

    @classmethod
    def register_config(cls, name: str):
        def decorator(model: Type[BaseModel]):
            cls._config_types[name] = model
            cls._model_run = None  # Reset ModelRun to force recreation
            return model

        return decorator

    @classmethod
    def register_datatype(cls, config_type: str, data_type: str):
        def decorator(datatype: Type[BaseModel]):
            if config_type not in cls._datatypes:
                cls._datatypes[config_type] = {}
            cls._datatypes[config_type][data_type] = datatype
            cls._model_run = None  # Reset ModelRun to force recreation
            # Recreate the config model with updated datatypes
            if config_type in cls._config_types:
                cls._recreate_config_model(config_type)
            return datatype

        return decorator

    @classmethod
    def _recreate_config_model(cls, config_type: str):
        base_model = cls._config_types[config_type]
        datatypes = cls.get_data_types(config_type)

        class ConfigDataModel(BaseModel):
            data_type: str

            @validator("data_type")
            def validate_data_type(cls, v):
                if v not in Registry._datatypes.get(config_type, {}):
                    raise ValueError(
                        f"Invalid data_type '{v}' for config_type '{config_type}'"
                    )
                return v

            class Config:
                extra = "allow"

        new_model = create_model(
            f"{config_type.capitalize()}Config",
            __base__=base_model,
            data=(List[ConfigDataModel], ...),
        )

        @validator("data", each_item=True)
        def validate_data_item(cls, v):
            data_type = v.data_type
            datatype_class = Registry.get_datatype(config_type, data_type)
            return datatype_class(**v.dict())

        new_model.validate_data_item = classmethod(validate_data_item)

        cls._config_types[config_type] = new_model

    @classmethod
    def get_config_types(cls):
        return cls._config_types

    @classmethod
    def get_data_types(cls, config_type: str):
        return tuple(cls._datatypes.get(config_type, {}).values())

    @classmethod
    def get_datatype(cls, config_type: str, data_type: str) -> Type[BaseModel]:
        return cls._datatypes.get(config_type, {}).get(data_type)

    @classmethod
    def get_datatypes_for_config(cls, config_type: str) -> List[str]:
        return list(cls._datatypes.get(config_type, {}).keys())

    @classmethod
    def create_model_run(cls):
        if not cls._config_types:
            raise ValueError("No config types registered")

        if cls._model_run is None:
            config_types = tuple(cls._config_types.values())
            cls._model_run = create_model(
                "ModelRun",
                config=(TypingUnion[config_types], ...),
            )

        return cls._model_run
