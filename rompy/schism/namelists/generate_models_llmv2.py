import datetime
import os
import re
from subprocess import run

import anthropic

TEMPLATE_DIR = os.path.join(os.path.dirname(__file__), "templates")


class ClaudeClient:
    def __init__(self):
        self.client = anthropic.Anthropic()

    def analyze_namelist(self, content: str, filename: str) -> dict:
        # - Use the json_schema_extra kwarg in Field to to provide the default value for each variable as it currently is in the namelist file
        # - Use Field notation to define variables. Scrape as much as you can in terms of descriptions for each variable, noting that relevant information may not all be inline with that particular variable. Add to the description kwarg of Field. All variables should be optional (i.e using Optional[]), with None as the default
        prompt = f"""
        Analyze the following namelist file content from {filename} and provide a set of pydantic v2 models with the following as guidance:
        - Each section starting with & (e.g. &example) should be a sepaate model. The name of the model should be CamelCased (eg. example becomes Example, EXAMPLE becomes Example)
        - Use Field notation to define variables. Scrape as much as you can in terms of descriptions for each variable, noting that relevant information may not all be inline with that particular variable and add to the description kwarg of Field. Set defaults as those in the namelist
        - Create additional submodels if obvious changes to do so exist. i.e. for tracers
        - Construct validators for each variable where possibly using the relevant text (use the new @field_validator decorator). If distinct options are clear, use a Literal type instead of a validator
        - Construct cross-variable validators that might be necessary (use the new @model_validator decorator)
        - Provide a master model that takes all of the other constructed models as inputs. Names of the input variables should be lowercased version of the CamelCased model names. The name for this class should be CamelCased name of the file (whithout extension). Use default_factories of the corresponding models. Determine an overall summary for the docs
        - Add doctrings to each model
        - The baseclass for all models should be rompy.schism.namelists.basemodel.NamelistBaseModel

        Namelist content:
        {content}

        Response should be raw python code, with no plain language explanations
        """

        message = self.client.messages.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=8192,
            messages=[{"role": "user", "content": prompt}],
        )

        # Check if content is a list and join it if so
        if isinstance(message.content, list):
            content = "".join(
                [item.text for item in message.content if hasattr(item, "text")]
            )
        else:
            content = message.content
        # Remove the python and ``` wrapping that Claude adds to the response
        content = re.sub(r"```python", "", content)
        content = re.sub(r"```", "", content)
        if not content:
            print("No analysis returned from Claude")
            __import__("ipdb").set_trace()
        return content


claude_client = ClaudeClient()


def generate_pydantic_models(
    filename: str,
    basemodel="rompy.schism.namelists.basemodel.NamelistBaseModel",
    claude_analysis=None,
):
    analysis = claude_analysis or {}

    with open(filename, "w") as file:
        file.write(
            f"# This file was auto generated from a schism namelist file on {datetime.datetime.now().strftime('%Y-%m-%d')}.\n\n"
        )
        file.write("from typing import Optional\n")
        file.write("from pydantic import Field, validator, model_validator\n")
        basemodellist = basemodel.split(".")
        file.write(
            f"from {'.'.join(basemodellist[0:-1])} import {basemodellist[-1]}\n\n"
        )
        file.write("\n")
        file.write(analysis)

    run(["isort", filename])
    run(["black", filename])


def nml_to_models(file_in: str, file_out: str):
    with open(file_in) as f:
        claude_analysis = claude_client.analyze_namelist(f.read(), file_in)

    generate_pydantic_models(
        file_out,
        claude_analysis=claude_analysis,
    )


def main():
    exclude_files = [
        "wwminput.nml.spectra",
        "wwminput.nml.WW3",
        "mice.nml",
        "ice.nml",
        "icm.nml",
        "example.nml",
        "cosine.nml",
        # "param.nml",
        "sediment.nml",
        "icm_reduced.nml",
        "wwminput_reduced.nml.WW3",
    ]
    with open("__init__.py", "w") as f:
        for file in os.listdir("sample_inputs"):
            if file in exclude_files:
                print(f"Skipping {file}")
                continue
            # for file in os.listdir("sample_inputs")[5:7]:
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
