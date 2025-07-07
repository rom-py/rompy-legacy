from pydantic import BaseModel
from typing import List
from cookiecutter.main import cookiecutter


class NameModel(BaseModel):
    names: List[str]


def create_pydantic_model(namelist_file):
    with open(namelist_file, "r") as file:
        names = [name.strip() for name in file.readlines()]

    model = NameModel(names=names)
    return model


def create_cookiecutter_file(model, template_dir, output_dir):
    cookiecutter(template_dir, extra_context=model.model_dump(), output_dir=output_dir)


if __name__ == "__main__":
    namelist_file = "path/to/namelist.txt"
    template_dir = "path/to/cookiecutter/template"
    output_dir = "path/to/output"

    model = create_pydantic_model(namelist_file)
    create_cookiecutter_file(model, template_dir, output_dir)
