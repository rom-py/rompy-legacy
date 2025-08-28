import re

from pydantic import BaseModel, Field


# Define a function to extract sections from the input file
def extract_sections(filename):
    with open(filename, "r") as file:
        content = file.read()
        sections = re.split(r"&\w+", content)
        return sections[1:]  # Skip the empty section at the beginning


# Define a function to create Pydantic models from a section
def create_pydantic_model(section):
    section_name, section_content = section.split("\n", 1)
    model_name = section_name.strip()
    fields = {}

    for line in section_content.split("\n"):
        if line.strip().startswith("!"):
            continue
        if "=" not in line:
            continue
        components = line.split("=")
        var_name, var_value = components[0], "=".join(components[1:])
        description = line.strip("!").strip()
        fields[var_name] = Field(default=var_value, description=description)

    return type(model_name, (BaseModel,), fields)


# Load the input text file and extract sections
filename = "example.nml"
sections = extract_sections(filename)

# Create Pydantic models for each section
for section in sections:
    model = create_pydantic_model(section)
    __import__("ipdb").set_trace()
    print(model.dict)
