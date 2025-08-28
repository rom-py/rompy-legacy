import datetime
import json
import os
import re
from subprocess import run
from typing import Any

import anthropic

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze_namelist(self, content: str, filename: str) -> dict:
        prompt = f"""
        Analyze the following namelist file content from {filename} and provide:
        1. Improved descriptions for each variable, noting that relevant information may not all be inline with that particular variable.
        2. Pydantic validators for each variable (use the new @field_validator and @classmethod decorators). Any functionls should have newlines specified with '\n'. 
        3. Any cross-variable validators that might be necessary (use the new @model_validator(model='after') decorator). Any functionls should have newlines specified with '\n'. 
        4. Variable names in the input files may be upper or lower case, but the Pydantic models should use lower case variable names.

        Namelist content:
        {content}

        Please format your response as a JSON object with the following structure:
        {{
            "variable_name": {{
                "description": "Improved description",
                "validators": ["List of validators"],
                "cross_validators": ["List of potential cross-variable validators"]
            }},
            ...
        }}

        Do not provide any extra information or context in the response.
        """

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=8048,
            messages=[{"role": "user", "content": prompt}],
        )

        # Check if content is a list and join it if so
        if isinstance(message.content, list):
            content = "".join(
                [item.text for item in message.content if hasattr(item, "text")]
            )
        else:
            content = message.content

        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Warning: Could not parse Claude's response for {filename}")
            print(f"Response: {content}")
            print(f"Error: {e}")
            return {}


claude_client = ClaudeClient()


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
    variable_dict = {}

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
                var = var.replace(f"({ii})", f"_{ii}").lower()
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
    basemodel="rompy.schism.namelists.basemodel.NamelistBaseModel",
    none_option=True,
    claude_analysis=None,
):
    analysis = claude_analysis or {}

    with open(filename, "w") as file:
        file.write(
            f"# This file was auto generated from a schism namelist file on {datetime.datetime.now().strftime('%Y-%m-%d')}.\n\n"
        )
        file.write("from typing import Optional\n")
        file.write("from pydantic import Field, field_validator, model_validator\n")
        basemodellist = basemodel.split(".")
        file.write(
            f"from {'.'.join(basemodellist[0:-1])} import {basemodellist[-1]}\n\n"
        )
        for key, value in data.items():
            model_name = key.capitalize()
            if key in ("description", "full_content"):
                continue
            file.write(f"class {model_name}({basemodellist[-1]}):\n")
            for inner_key, inner_value in value.items():
                if inner_key == "description":
                    file.write(f'    """\n    {inner_value}\n    """\n')
                    continue

                description = analysis.get(inner_key, {}).get(
                    "description", inner_value["description"]
                )

                default = inner_value["default"]
                if isinstance(default, list):
                    if default:
                        inner_type = type(default[0])
                        default_value = default
                    else:
                        inner_type = Any
                        default_value = []
                    inner_type = list[inner_type]
                else:
                    inner_type = type(default)
                    default_value = default

                file.write(
                    '    {}: {} = Field({}, description="{}")\n'.format(
                        inner_key,
                        f"Optional[{inner_type.__name__}]",
                        repr(default_value),
                        description,
                    )
                )

            for inner_key, inner_value in analysis.items():
                validators = inner_value.get("validators", [])
                for i, validator_desc in enumerate(validators):
                    file.write("\n")
                    for line in validator_desc.split("\n"):
                        file.write("    " + line + "\n")
                    file.write("\n")

            for inner_key, inner_value in analysis.items():
                cross_validators = inner_value.get("cross_validators", [])
                for i, validator_desc in enumerate(cross_validators):
                    file.write("\n")
                    for line in validator_desc.split("\n"):
                        file.write("    " + line + "\n")
                    file.write("\n")

            file.write("\n")

        if master_model_name:
            file.write(f"class {master_model_name}({basemodellist[-1]}):\n")
            for key in data.keys():
                if key == "description":
                    continue
                    # indented_text = "\n".join(
                    #     ["    " + line for line in data[key].split("\n")]
                    # )
                    # file.write(f'    """\n{indented_text}\n    """\n')
                elif key not in ("full_content"):
                    if none_option == "none":
                        file.write(
                            f"    {key}: Optional[{key.capitalize()}] = Field(default=None)\n"
                        )
                    elif none_option == "factory":
                        file.write(
                            f"    {key}: Optional[{key.capitalize()}] = Field(default_factory={key.capitalize()})\n"
                        )
                    else:
                        file.write(
                            f"    {key}: {key.capitalize()} = Field(default={key.capitalize()}())\n"
                        )
    run(["isort", filename])
    run(["black", filename])


def nml_to_models(file_in: str, file_out: str):
    nml_dict = nml_to_dict(file_in)

    if os.path.exists(f"{file_out}.json"):
        with open(f"{file_out}.json", "r") as f:
            print(f"Loading existing analysis {file_out}.json")
            claude_analysis = json.loads(f.read())
    else:
        claude_analysis = claude_client.analyze_namelist(
            nml_dict["full_content"], file_in
        )
        with open(f"{file_out}.json", "w") as f:
            f.write(json.dumps(claude_analysis, indent=4))
    master_model_name = os.path.basename(file_in).split(".")[0].capitalize()
    none_option = "factory"
    generate_pydantic_models(
        nml_dict,
        file_out,
        master_model_name,
        none_option=none_option,
        claude_analysis=claude_analysis,
    )


def nml_to_dict(file_in: str):
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
    nml_dict["full_content"] = input_text
    return nml_dict


def main():
    exclude_files = [
        "wwminput.nml.spectra",
        # "wwminput.nml.WW3",
        "wwminput.nml",
        "mice.nml",
        "ice.nml",
        "icm.nml",
        "example.nml",
        "cosine.nml",
        "param.nml",
        "sediment.nml",
        "icm_reduced.nml",
        "wwminput_reduced.nml.WW3",
    ]
    with open("__init__.py", "w") as f:
        for file in os.listdir("sample_inputs"):
            if file in exclude_files:
                print(f"Skipping {file}")
                continue
            components = file.split(".")
            if len(components) < 2:
                continue
            if components[1] == "nml":
                file_in = os.path.join("sample_inputs", file)
                if len(components) > 2:
                    file_out = components[0] + "_" + components[2] + ".py"
                else:
                    file_out = file.split(".")[0] + ".py"
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
