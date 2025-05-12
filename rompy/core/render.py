import logging
import os
import time as time_module
from pathlib import Path
from datetime import datetime

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


def render(context, template, output_dir, checkout=None):
    """Render the template with the given context.
    
    This function handles the rendering process and provides detailed progress
    information during the rendering.
    
    Args:
        context (dict): The context to use for rendering
        template (str): The template directory or URL
        output_dir (str): The output directory
        checkout (str, optional): The branch, tag or commit to checkout
        
    Returns:
        str: The path to the rendered template
    """
    # Use helper function to avoid circular imports
    from rompy import ROMPY_ASCII_MODE
    USE_ASCII_ONLY = ROMPY_ASCII_MODE()
    
    start_time = time_module.time()
    
    if USE_ASCII_ONLY:
        logger.info("+------------------------------------------------------------------------+")
        logger.info("|                      TEMPLATE RENDERING PROCESS                         |")
        logger.info("+------------------------------------------------------------------------+")
    else:
        logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        logger.info("┃                      TEMPLATE RENDERING PROCESS                   ┃")
        logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        
    logger.info(f"Template source: {template}")
    logger.info(f"Output directory: {output_dir}")
    if checkout:
        logger.info(f"Using template version: {checkout}")
    
    # Initialize context for cookiecutter
    context["cookiecutter"] = {}
    config_dict = cc_config.get_user_config(
        config_file=None,
        default_config=False,
    )

    # Determine the repo directory
    if USE_ASCII_ONLY:
        logger.info("Locating template repository...")
    else:
        logger.info("🔍 Locating template repository...")
        
    repo_dir, cleanup = cc_repository.determine_repo_dir(
        template=template,
        abbreviations=config_dict["abbreviations"],
        clone_to_dir=config_dict["cookiecutters_dir"],
        checkout=checkout,
        no_input=True,
    )
    logger.info(f"Template repository located at: {repo_dir}")
    context["_template"] = repo_dir

    # Generate files from template
    if USE_ASCII_ONLY:
        logger.info("Generating files from template...")
    else:
        logger.info("🔧 Generating files from template...")
    render_start = time_module.time()
    staging_dir = cc_generate.generate_files(
        repo_dir=repo_dir,
        context=context,
        overwrite_if_exists=True,
        output_dir=output_dir,
    )
    
    # Log completion information
    elapsed = time_module.time() - start_time
    render_time = time_module.time() - render_start
    
    # Get number of files created
    file_count = sum([len(files) for _, _, files in os.walk(staging_dir)])
    
    if USE_ASCII_ONLY:
        logger.info("+------------------------------------------------------------------------+")
        logger.info("|                        TEMPLATE RENDERING COMPLETE                      |")
        logger.info("+------------------------------------------------------------------------+")
        logger.info(f"Rendering time:      {render_time:.2f} seconds")
        logger.info(f"Total process time:  {elapsed:.2f} seconds")
        logger.info(f"Files created:       {file_count}")
        logger.info(f"Output location:     {staging_dir}")
        logger.info("+------------------------------------------------------------------------+")
    else:
        logger.info("┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓")
        logger.info("┃                        TEMPLATE RENDERING COMPLETE                 ┃")
        logger.info("┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛")
        logger.info(f"⏱️ Rendering time:      {render_time:.2f} seconds")
        logger.info(f"⏱️ Total process time:  {elapsed:.2f} seconds")
        logger.info(f"📄 Files created:       {file_count}")
        logger.info(f"📂 Output location:     {staging_dir}")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    return staging_dir
