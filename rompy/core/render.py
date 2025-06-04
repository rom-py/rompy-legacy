import logging
import os
from pathlib import Path

import cookiecutter.config as cc_config
import cookiecutter.generate as cc_generate
import cookiecutter.repository as cc_repository
from cookiecutter.exceptions import NonTemplatedInputDirException
from cookiecutter.find import find_template

logger = logging.getLogger(__name__)


def repository_has_cookiecutter_json(repo_directory):
    """Determine if `repo_directory` contains a `cookiecutter.json` file.

    :param repo_directory: The candidate repository directory.
    :return: True if the `repo_directory` is valid, else False.
    """
    repo_directory_exists = os.path.isdir(repo_directory)

    # repo_config_exists = os.path.isfile(
    #     os.path.join(repo_directory, "cookiecutter.json")
    # )
    repo_config_exists = True
    return repo_directory_exists and repo_config_exists


def find_template(repo_dir, env):
    """Determine which child directory of `repo_dir` is the project template.

    :param repo_dir: Local directory of newly cloned repo.
    :returns project_template: Relative path to project template.
    """
    logger.debug("Searching %s for the project template.", repo_dir)

    for str_path in os.listdir(repo_dir):
        if (
            "runtime" in str_path
            and env.variable_start_string in str_path
            and env.variable_end_string in str_path
        ):
            project_template = Path(repo_dir, str_path)
            break
    else:
        raise NonTemplatedInputDirException

    logger.debug("The project template appears to be %s", project_template)
    return project_template


cc_repository.repository_has_cookiecutter_json = repository_has_cookiecutter_json
cc_generate.find_template = find_template


def render(context, template, checkout=None):
    context["cookiecutter"] = {}
    config_dict = cc_config.get_user_config(
        config_file=None,
        default_config=False,
    )

    repo_dir, cleanup = cc_repository.determine_repo_dir(
        template=template,
        abbreviations=config_dict["abbreviations"],
        clone_to_dir=config_dict["cookiecutters_dir"],
        checkout=checkout,
        no_input=True,
    )
    context["_template"] = repo_dir

    staging_dir = cc_generate.generate_files(
        repo_dir=repo_dir,
        context=context,
        overwrite_if_exists=True,
        output_dir=".",
    )
    return staging_dir
