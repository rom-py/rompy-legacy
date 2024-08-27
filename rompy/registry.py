from typing import Dict, List, Optional, Type
from typing import Union as TypingUnion

from pydantic import BaseModel, create_model, root_validator, validator


class BaseConfigModel(BaseModel):
    data: List[BaseModel]

    @root_validator(pre=True)
    def validate_data(cls, values):
        if "data" in values and "config_type" in values:
            data = values["data"]
            config_type = values["config_type"]
            validated_data = []
            for item in data:
                if isinstance(item, dict):
                    data_type = item.get("data_type")
                    if data_type:
                        datatype_class = Registry.get_datatype(config_type, data_type)
                        if datatype_class:
                            validated_data.append(datatype_class(**item))
                        else:
                            raise ValueError(f"No registered data type: {data_type}")
                    else:
                        validated_data.append(item)
                else:
                    validated_data.append(item)
            values["data"] = validated_data
        return values


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
            return datatype

        return decorator

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
