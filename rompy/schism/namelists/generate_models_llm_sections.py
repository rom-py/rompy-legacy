import datetime
import json
import os
import re
from subprocess import run
from typing import Any

import anthropic

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")
VAR_SPEC_DIR = "variable_specifications"


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze_namelist_section(self, content: str, section_name: str) -> dict:
        prompt = f"""
        Analyze the following namelist section content from {section_name} and provide:
        1. Improved descriptions for each variable, noting that relevant information may not all be inline with that particular variable.
        2. Pydantic validators for each variable (use the new @field_validator and @classmethod decorators).  You should use the info.data argument in Pydantic v2, which provides access to other fields of the model if a field validator needs access to other variables
        3. Any cross-variable validators that might be necessary (use the @model_validator decorator). If model='after', validator should take and receive self and value should be accessed via attributes not keys
        4. All validations should use pydantic v2 syntax. Examples below:
        5. Variable names should be all lower case

        Namelist section content:
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
            print(f"Warning: Could not parse Claude's response for {section_name}")
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
                var = var.replace(f"({ii})", f"__{ii}").lower()
            values = []
            # remove trailing comma
            # For for weird case in wwminput where the (seamingly) single input has a comma after it? TODO
            value = value.strip()
            if len(value) > 1:
                if value[-1] == ",":
                    value = value[:-1]
            if "," in value:
                items = value.split(",")
                # For for weird case in wwminput where the (seamingly) single input has a comma after it? TODO
                if len(items) == 2:
                    if items[1].strip() == "":
                        items = items[0]
                    if items[0].strip() == "":
                        items = items[1]
            else:
                items = value.split()
            for item in items:
                try:
                    if "e" in item or "E" in item:
                        values.append(float(item.strip()))
                    elif "." in item:
                        values.append(float(item))
                    else:
                        values.append(int(item))
                except Exception:
                    values.append(str(item).strip())
            for ii in range(len(values)):
                if isinstance(values[ii], str):
                    values[ii] = values[ii].replace("'", "").replace('"', "")
            for ii in range(len(values)):
                if values[ii] == "T":
                    values[ii] = True
                if values[ii] == "F":
                    values[ii] = False
            variable_dict[var.strip()] = dict(
                default=values if len(values) > 1 else values[0],
                description=description.strip().replace('"', "'"),
            )

    return variable_dict


def generate_pydantic_model(
    section_name: str,
    data: dict,
    filename: str,
    master_model_name: str,
    basemodel="rompy.schism.namelists.basemodel.NamelistBaseModel",
    claude_analysis: dict = None,
):
    analysis = claude_analysis or {}

    with open(filename, "a") as file:
        model_name = section_name.capitalize()
        file.write(f"class {model_name}({basemodel.split('.')[-1]}):\n")

        for variable_name, variable_data in data.items():
            if variable_name == "description":
                file.write(f'    """\n    {variable_data}\n    """\n')
                continue

            description = analysis.get(variable_name, {}).get(
                "description", variable_data["description"]
            )

            default = variable_data["default"]
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
                    variable_name,
                    f"Optional[{inner_type.__name__}]",
                    repr(default_value),
                    description,
                )
            )

        variable_names = list(data.keys())
        for variable_name, variable_data in analysis.items():
            if variable_name in variable_names:
                validators = variable_data.get("validators", [])
                for i, validator_desc in enumerate(validators):
                    file.write("\n")
                    for line in validator_desc.split("\n"):
                        file.write("    " + line + "\n")
                    file.write("\n")

            cross_validators = analysis.get("cross_validators", [])
            for i, validator_desc in enumerate(cross_validators):
                file.write("\n")
                for line in validator_desc.split("\n"):
                    if line == "@model_validator":
                        line += "(mode='before')"
                    file.write("    " + line + "\n")
                if "return" not in line:
                    file.write(
                        "        " + "return self" + "\n"
                    )  # addresses tendency for cluude to omit return self
                file.write("\n")

        file.write("\n")


def nml_to_models(file_in: str, file_out: str):
    nml_dict = nml_to_dict(file_in)
    master_model_name = os.path.basename(file_in).split(".")[0].capitalize()
    basemodel = "rompy.schism.namelists.basemodel.NamelistBaseModel"

    var_spec_file = os.path.join(VAR_SPEC_DIR, f"{file_out}.json")
    if os.path.exists(var_spec_file):
        with open(var_spec_file, "r") as f:
            print(f"Loading existing analysis {file_out}.json")
            claude_analysis = json.loads(f.read())
    else:
        claude_analysis = {}
        for section_name, section_content in nml_dict.items():
            if section_name not in {"description", "full_content"}:
                print(f"Analyzing {section_name}")
                print(section_content)
                analysis = claude_client.analyze_namelist_section(
                    section_content, section_name
                )
                claude_analysis[section_name] = analysis
        with open(var_spec_file, "w") as f:
            f.write(json.dumps(claude_analysis, indent=4))

    with open(file_out, "w") as file:
        file.write(
            f"# This file was auto generated from a SCHISM namelist file on {datetime.datetime.now().strftime('%Y-%m-%d')}.\n\n"
        )
        file.write("from typing import Optional, List\n")
        file.write("from pydantic import Field, field_validator, model_validator\n")
        file.write("from datetime import datetime\n")
        basemodellist = basemodel.split(".")
        file.write(
            f"from {'.'.join(basemodellist[0:-1])} import {basemodellist[-1]}\n\n"
        )

    for section_name, section_content in nml_dict.items():
        if section_name not in {"description", "full_content"}:
            variables_data = extract_variables(section_content)
            generate_pydantic_model(
                section_name,
                variables_data,
                file_out,
                master_model_name,
                basemodel=basemodel,
                claude_analysis=claude_analysis.get(section_name),
            )

    with open(file_out, "a") as file:
        file.write(f"class {master_model_name}({basemodel.split('.')[-1]}):\n")
        for section_name, section_content in nml_dict.items():
            model_name = section_name.capitalize()
            if model_name not in [master_model_name, "Description", "Full_content"]:
                file.write(
                    f"    {section_name.lower()}: Optional[{model_name}] = Field(default_factory={model_name})\n"
                )

    run(["isort", file_out])
    run(["black", file_out])


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
                nml_dict.update({section.lower(): text})
    nml_dict["description"] = blurb + input_text
    nml_dict["full_content"] = input_text
    return nml_dict


def main():
    exclude_files = [
        "mice.nml",
        "ice.nml",
        "icm.nml",
        "example.nml",
        "cosine.nml",
        "param.nml",
        "sediment.nml",
        # "wwminput.nml",
        "wwminput.nml.spectra",
        "wwminput.nml.WW3",
        "icm_reduced.nml",
        "wwminput_reduced.nml.WW3",
        "wwminput_v511_nessa.nml",
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
                print(f"Processing {file_in} to {file_out}")
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
