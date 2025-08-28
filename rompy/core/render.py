import os
import time as time_module
from pathlib import Path
from typing import Any, Dict, Optional

import cookiecutter.config as cc_config
import cookiecutter.generate as cc_generate
import cookiecutter.repository as cc_repository
from cookiecutter.exceptions import NonTemplatedInputDirException
from cookiecutter.find import find_template

from rompy.logging import get_logger
from rompy.core.types import RompyBaseModel

logger = get_logger(__name__)


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


class TemplateRenderer(RompyBaseModel):
    """Template renderer class that provides enhanced logging and formatting.

    This class wraps the cookiecutter template rendering process and provides
    detailed formatting through the _format_value method.
    """

    template: str | Path
    output_dir: str | Path
    context: Dict[str, Any]
    checkout: Optional[str] = None

    def _format_value(self, obj) -> Optional[str]:
        """Format specific types of values for display using the new formatting framework.

        This method formats template rendering information with rich details.

        Args:
            obj: The object to format

        Returns:
            A formatted string or None to use default formatting
        """
        # Only format TemplateRenderer objects
        if not isinstance(obj, TemplateRenderer):
            return None

        # Use the new formatting framework
        from rompy.formatting import format_value

        return format_value(obj)

    def __call__(self) -> str:
        """Render the template with the given context.

        Returns:
            str: The path to the rendered template
        """
        return render(self.context, self.template, self.output_dir, self.checkout)


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
    # Use formatting utilities imported at the top of the file

    start_time = time_module.time()

    # Create renderer object for nice formatting
    renderer = TemplateRenderer(
        template=template, output_dir=output_dir, context=context, checkout=checkout
    )

    # Format renderer info
    renderer_info = renderer._format_value(renderer)

    # Log detailed renderer info
    if renderer_info:
        for line in renderer_info.split("\n"):
            logger.info(line)
    else:
        # Fall back to simple logging if formatting failed
        logger.info("Template source: %s", template)
        logger.info("Output directory: %s", output_dir)
        if checkout:
            logger.info("Using template version: %s", checkout)

    # Initialize context for cookiecutter
    context["cookiecutter"] = {}
    config_dict = cc_config.get_user_config(
        config_file=None,
        default_config=False,
    )

    # Determine the repo directory
    logger.bullet_list(["Locating template repository..."])

    repo_dir, cleanup = cc_repository.determine_repo_dir(
        template=template,
        abbreviations=config_dict["abbreviations"],
        clone_to_dir=config_dict["cookiecutters_dir"],
        checkout=checkout,
        no_input=True,
    )
    logger.info("Template repository located at: %s", repo_dir)
    context["_template"] = repo_dir

    # Generate files from template
    logger.bullet_list(["Generating files from template..."])
    render_start = time_module.time()
    staging_dir = cc_generate.generate_files(
        repo_dir=repo_dir,
        context=context,
        overwrite_if_exists=True,
        output_dir=".",
    )

    # Log completion information
    elapsed = time_module.time() - start_time
    render_time = time_module.time() - render_start

    # Get number of files created
    file_count = sum([len(files) for _, _, files in os.walk(staging_dir)])

    # Create render results object for formatting
    class RenderResults(RompyBaseModel):
        """Render results information"""

        staging_dir: str
        render_time: float
        elapsed_time: float
        file_count: int

        def _format_value(self, obj) -> Optional[str]:
            """Format render results for display using the new formatting framework.

            Args:
                obj: The object to format

            Returns:
                A formatted string or None to use default formatting
            """
            # Only format RenderResults objects
            if not isinstance(obj, RenderResults):
                return None

            # Use the new formatting framework
            from rompy.formatting import format_value

            return format_value(obj)

    # Create and format results
    results = RenderResults(
        staging_dir=staging_dir,
        render_time=render_time,
        elapsed_time=elapsed,
        file_count=file_count,
    )

    results_info = results._format_value(results)
    if results_info:
        for line in results_info.split("\n"):
            logger.info(line)
    else:
        # Fallback to bullet list if formatting failed
        logger.bullet_list(
            [
                f"Rendering time:      {render_time:.2f} seconds",
                f"Total process time:  {elapsed:.2f} seconds",
                f"Files created:       {file_count}",
                f"Output location:     {staging_dir}",
            ]
        )

    return staging_dir
