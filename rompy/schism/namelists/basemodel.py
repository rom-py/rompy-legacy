from pathlib import Path
from typing import Any, Dict, Optional, Type, Union, get_type_hints

from pydantic import BaseModel, model_serializer, model_validator

from rompy.core.types import RompyBaseModel


def get_model_field_type(
    model: Type[BaseModel], field: str
) -> Optional[Type[BaseModel]]:
    model_fields = get_type_hints(model)
    annotation = model_fields.get(field)

    if annotation:
        try:
            if hasattr(annotation, "__origin__") and annotation.__origin__ is Union:
                for arg in annotation.__args__:
                    if arg is not type(None) and issubclass(arg, BaseModel):
                        return arg
            elif issubclass(annotation, BaseModel):
                return annotation
        except Exception as e:
            __import__("ipdb").set_trace()

    return None


def recursive_update(model: BaseModel, updates: Dict[str, Any]) -> BaseModel:
    updates_applied = model.model_dump()
    model_fields = get_type_hints(model.__class__)

    for key, value in updates.items():
        if key in model_fields:
            field_type = get_model_field_type(model.__class__, key)
            current_value = getattr(model, key)

            if isinstance(value, dict) and field_type:
                if current_value is None:
                    current_value = field_type()  # Initialize if None

                updated_value = recursive_update(current_value, value)
                updates_applied[key] = updated_value
            else:
                updates_applied[key] = value
        else:
            updates_applied[key] = value  # Extending with new fields, if any

    return model.model_copy(update=updates_applied)


class NamelistBaseModel(RompyBaseModel):
    """Base model for namelist variables"""

    @model_validator(mode="before")
    def __lowercase_property_keys__(cls, values: Any) -> Any:
        def __lower__(value: Any) -> Any:
            if isinstance(value, dict):
                return {k.lower(): __lower__(v) for k, v in value.items()}
            return value

        return __lower__(values)
        
    @model_serializer
    def serialize_model(self, **kwargs):
        """Custom serializer to handle proper serialization of nested components."""
        result = {}
        
        # Include only non-None fields in the serialized output
        for field_name in self.model_fields:
            value = getattr(self, field_name, None)
            if value is not None and not field_name.startswith("_"):
                result[field_name] = value
                
        return result

    def update(self, update: Dict[str, Any]):
        """Update the namelist variable with new values. Reninitializes the instance, ensuring all validations are run"""
        updated_self = recursive_update(self, update)
        updated_instance = self.__init__(**updated_self.model_dump())
        return updated_instance

    def render(self) -> str:
        """Render the namelist variable as a string"""
        # create string of the form "variable = value"
        ret = []
        ret += [f"! SCHISM {self.__module__} namelist rendered from Rompy\n"]
        for section, values in self.model_dump().items():
            if values is not None:
                ret += [f"&{section}"]
                for variable, value in values.items():
                    if value is not None:
                        for ii in sorted(range(40), reverse=True):
                            variable = variable.replace(f"__{ii}", f"({ii})")
                        if isinstance(value, list):
                            value = ", ".join(
                                [self.process_value(item) for item in value]
                            )
                        else:
                            value = self.process_value(value)
                        ret += [f"{variable} = {value}"]
                ret += ["/\n"]
        return "\n".join(ret)

    def process_value(self, value: Any) -> Any:
        """Process the value before rendering"""
        if isinstance(value, bool):
            value = self.boolean_to_string(value)
        elif isinstance(value, str):
            if value not in ["T", "F"]:
                value = f"'{value}'"
        return str(value)

    def boolean_to_string(self, value: bool) -> str:
        return "T" if value else "F"

    def write_nml(self, workdir: Path | str) -> None:
        """Write the namelist to a file

        Args:
            workdir (Path|str): Working directory to write to
        """
        # Ensure workdir is a Path object
        workdir_path = Path(workdir) if isinstance(workdir, str) else workdir
        output = workdir_path / f"{self.__class__.__name__.lower()}.nml"
        with open(output, "w") as f:
            f.write(self.render())
