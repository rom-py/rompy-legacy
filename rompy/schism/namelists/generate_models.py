import datetime
import os
import re
from subprocess import run
from typing import Any

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


def extract_sections_from_text(text):
    sections = re.split(r"&(\w+)", text)
    sections_dict = {}
    sections_dict["description"] = sections[0]
    for i in range(1, len(sections), 2):
        section_name = sections[i]
        section_content = sections[i + 1]
        sections_dict[section_name.strip()] = section_content.strip()
    return sections_dict


def extract_variables(section):
    # remove all comments
    # section = re.sub(r"!.*", "", section)

    # Define a regular expression pattern to match variables and their values
    # pattern = r"(\w+)\s*=\s*([^!\n]+)"

    # # Define a regular expression pattern to match variables and their values. Only match the first instance on each line
    # pattern = r"(\w+)\s*=\s*([^!\n]+)(?:\s*!.*|$)"
    #
    # # Use the findall function to extract variables and values
    # matches = re.findall(pattern, section)

    # Create a dictionary to store the variable-value pairs
    variable_dict = {}

    # Iterate through matches and split values into a list if they contain spaces
    for line in section.split("\n"):
        if line.strip() == "":
            continue
        if line[0] == "!":
            continue
        if "!" in line:
            pair, description = line.split("!")[0], "!".join(line.split("!")[1:])
        else:
            pair = line
            description = ""
        if "=" in pair:
            var, value = pair.split("=")[0], "=".join(pair.split("=")[1:])
            for ii in range(50):
                var = var.replace(f"({ii})", f"__{ii}").lower()
            values = []
            if "," in value:
                items = value.split(",")
            else:
                items = value.split()
            for item in items:
                try:
                    if "e" in item or "E" in item:
                        values.append(item.strip())
                    elif "." in item:
                        values.append(float(item))
                    else:
                        values.append(int(item))
                except Exception:
                    values.append(str(item).strip())
            variable_dict[var.strip()] = dict(
                default=values if len(values) > 1 else values[0],
                description=description.strip().replace('"', "'"),
            )

    return variable_dict


def generate_pydantic_models(
    data: dict,
    filename: str,
    master_model_name=None,
    # basemodel="rompy.schism.basemodel.NamelistBaseModel",
    basemodel="rompy.schism.namelists.basemodel.NamelistBaseModel",
    none_option=False,
):
    with open(filename, "w") as file:
        file.write(
            f"# This file was auto generated from a schism namelist file on {datetime.datetime.now().strftime('%Y-%m-%d')}.\n\n"
        )
        file.write("from typing import Optional\n")
        file.write("from pydantic import Field\n")
        basemodellist = basemodel.split(".")
        file.write(
            f"from {'.'.join(basemodellist[0:-1])} import {basemodellist[-1]}\n\n"
        )
        for key, value in data.items():
            model_name = key.capitalize()
            if key == "description":
                continue
            file.write(f"class {model_name}({basemodellist[-1]}):\n")
            for inner_key, inner_value in value.items():
                if inner_key == "description":
                    file.write(f'    """\n    {inner_value}\n    """""\n')
                    continue
                default = inner_value["default"]
                if isinstance(default, list):
                    if default:  # Checks if the list is not empty
                        inner_type = type(default[0])
                        default_value = default
                    else:
                        inner_type = Any
                        default_value = []
                    inner_type = list[inner_type]  # Adjusting to list type
                else:
                    inner_type = type(default)
                    default_value = default

                # Write the field
                if none_option:
                    file.write(
                        '    {}: {} = Field({}, description="{}")\n'.format(
                            inner_key,
                            f"Optional[{inner_type.__name__}]",
                            None,
                            inner_value["description"],
                        )
                    )
                else:
                    file.write(
                        '    {}: {} = Field({}, description="{}")\n'.format(
                            inner_key,
                            inner_type.__name__,
                            repr(default_value),
                            inner_value["description"],
                        )
                    )
            file.write("\n")

        if master_model_name:
            file.write(f"class {master_model_name}({basemodellist[-1]}):\n")
            for key in data.keys():
                if key == "description":
                    indented_text = "\n".join(
                        ["    " + line for line in data[key].split("\n")]
                    )
                    file.write(f'    """\n    {indented_text}\n    """\n')
                else:
                    if none_option:
                        file.write(
                            f"    {key}: Optional[{key.capitalize()}] = Field(default=None)\n"
                        )
                    else:
                        file.write(
                            f"    {key}: {key.capitalize()} = Field(default={key.capitalize()}())\n"
                        )
    run(["isort", filename])
    run(["black", filename])


def nml_to_models(file_in: str, file_out: str):
    # Load the input text file and extract sections
    nml_dict = nml_to_dict(file_in)
    master_model_name = os.path.basename(file_in).split(".")[0].capitalize()
    # if os.path.basename(file_in).split(".")[0] == "param":
    #     none_option = False
    # else:
    #     none_option = True
    none_option = True
    generate_pydantic_models(
        nml_dict, file_out, master_model_name, none_option=none_option
    )


def nml_to_dict(file_in: str):
    # Load the input text file and extract sections
    with open(file_in, "r") as file:
        input_text = file.read()
    sections = extract_sections_from_text(input_text)
    nml_dict = {}
    blurb = "\n"
    blurb += "The full contents of the namelist file are shown below providing\n"
    blurb += "associated documentation for the objects:\n\n"
    for section, text in sections.items():
        if section == "description":
            nml_dict.update({section: text})
        else:
            input_dict = extract_variables(text)
            if input_dict:
                nml_dict.update({section.lower(): input_dict})
    nml_dict["description"] = blurb + input_text
    return nml_dict


# file_in = "example.nml"
# file_name = "generated_models.py"
# nml_to_models(file_in, file_name)


def main():
    with open("__init__.py", "w") as f:
        for file in os.listdir("sample_inputs"):
            components = file.split(".")
            if len(components) < 2:
                continue
            if components[1] == "nml":
                file_in = os.path.join("sample_inputs", file)
                # file_out_old = file.split(".")[0] + "_nml" + ".py"
                if len(components) > 2:
                    file_out = components[0] + "_" + components[2] + ".py"
                else:
                    file_out = file.split(".")[0] + ".py"
                # run(f"git mv {file_out_old} {file_out}".split())
                print(f" Processing {file_in} to {file_out}")
                nml_to_models(file_in, file_out)
                classname = file_out.split(".")[0]
                f.write(
                    f"from .{classname} import {classname.split('_')[0].capitalize()}\n"
                )
        f.write("from .sflux import Sflux_Inputs\n")
        f.write("from .schism import NML")
    run(["isort", "__init__.py"])
    run(["black", "__init__.py"])


if __name__ == "__main__":
    main()
