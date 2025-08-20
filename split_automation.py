#!/usr/bin/env python3
"""
Unified ROMPY Split Automation Script

This script performs the complete ROMPY repository split, fixes, verification, and testing in a single workflow.

Usage:
    python split_automation.py [--config CONFIG_FILE] [--dry-run] [--no-test] [--retry-setup]

Author: ROMPY Development Team
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from final_test_fixes import FinalTestFixer

# Import cookiecutter if available
try:
    from cookiecutter.main import cookiecutter
    COOKIECUTTER_AVAILABLE = True
except ImportError:
    COOKIECUTTER_AVAILABLE = False

# Import modern templates if available
try:
    from templates.modern_setup_templates import create_modern_setup_files
    MODERN_TEMPLATES_AVAILABLE = True
except ImportError:
    MODERN_TEMPLATES_AVAILABLE = False

class RepositorySplitter:
    def _commit_all_changes(self, target_dir: str, message: str):
        """
        Commit all staged changes in the target directory with a descriptive message.
        """
        if self.dry_run:
            logger.info(f"DRY RUN: Would commit changes: {message}")
            return
        try:
            # Only commit if there are staged changes
            result = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=target_dir)
            if result.returncode != 0:
                self._run_command(["git", "commit", "-m", message], cwd=target_dir)
                logger.info(f"[GIT] Commit: {message}")
            else:
                logger.info(f"[GIT] No staged changes to commit for: {message}")
        except Exception as e:
            logger.error(f"[GIT] Commit failed: {e}")

    """
    Handles the splitting of a monorepo into multiple repositories.
    """
    def __init__(self, config_path: str, dry_run: bool = False):
        self.config_path = config_path
        self.dry_run = dry_run
        self.config = self._load_config()
        self.source_repo = os.path.abspath(self.config["source_repo"])
        self.target_base_dir = os.path.abspath(self.config["target_base_dir"])

    def _load_config(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, "r") as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    def _run_command(self, cmd: List[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
        cmd_str = " ".join(cmd)
        logger.info(f"Running: {cmd_str} (cwd: {cwd or 'current'})")
        if self.dry_run:
            logger.info("DRY RUN: Command would be executed")
            return subprocess.CompletedProcess(cmd, 0, "", "")
        try:
            result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
            if result.stdout:
                logger.debug(f"STDOUT: {result.stdout}")
            if result.stderr:
                logger.debug(f"STDERR: {result.stderr}")
            return result
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed: {cmd_str}")
            logger.error(f"Exit code: {e.returncode}")
            logger.error(f"STDOUT: {e.stdout}")
            logger.error(f"STDERR: {e.stderr}")
            raise

    def _check_prerequisites(self):
        try:
            self._run_command(["git", "--version"])
            logger.info("Git is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Git is not available or not working")
            sys.exit(1)
        try:
            self._run_command(["git-filter-repo", "--version"])
            logger.info("git-filter-repo is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("git-filter-repo is not available. Install with: pip install git-filter-repo")
            sys.exit(1)

    def _create_target_directory(self, repo_name: str) -> str:
        target_dir = os.path.join(self.target_base_dir, repo_name)
        if not self.dry_run:
            os.makedirs(target_dir, exist_ok=True)
        logger.info(f"Target directory: {target_dir}")
        return target_dir

    def _clone_source_repo(self, target_dir: str):
        logger.info(f"Cloning source repository to {target_dir}")
        if not self.dry_run and os.path.exists(target_dir):
            shutil.rmtree(target_dir)
        self._run_command(["git", "clone", self.source_repo, target_dir])

    def _build_path_filters(self, paths: List[str]) -> List[str]:
        include_paths = []
        exclude_paths = []
        for path in paths:
            if path.startswith("!"):
                exclude_paths.append(path[1:])
            else:
                include_paths.append(path)
        filters = []
        for path in include_paths:
            filters.extend(["--path", path])
        return filters

    def _filter_repository(self, target_dir: str, paths: List[str]):
        logger.info(f"Filtering repository at {target_dir}")
        logger.info(f"Path filters: {paths}")
        cmd = ["git-filter-repo", "--force"]
        path_filters = self._build_path_filters(paths)
        cmd.extend(path_filters)
        self._run_command(cmd, cwd=target_dir)

    def _create_plugin_documentation(self, target_dir: str, package_name: str, plugin_name: str, extends_core_docs: bool):
        docs_dir = os.path.join(target_dir, "docs", "source")
        os.makedirs(docs_dir, exist_ok=True)
        conf_template = self.config.get("templates", {}).get("plugin_docs_conf", "")
        if conf_template:
            conf_content = conf_template.format(package_name=package_name, plugin_name=plugin_name)
            conf_path = os.path.join(docs_dir, "conf.py")
            with open(conf_path, "w") as f:
                f.write(conf_content)
            logger.info(f"Created plugin documentation config for {package_name}")
            index_content = f"""
{package_name}
{'=' * len(package_name)}

{plugin_name.upper()} plugin for the rompy ocean wave modeling framework.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples

Installation
============

.. code-block:: bash

   pip install {package_name}

API Reference
=============

.. automodule:: {package_name.replace('-', '_')}
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
"""
            index_path = os.path.join(docs_dir, "index.rst")
            with open(index_path, "w") as f:
                f.write(index_content)
            logger.info(f"Created plugin documentation index for {package_name}")

    def _update_docs_configuration(self, target_dir: str, package_name: str, is_core: bool, plugin_discovery: bool):
        if is_core and plugin_discovery:
            docs_dir = os.path.join(target_dir, "docs", "source")
            os.makedirs(docs_dir, exist_ok=True)
            conf_base_template = self.config.get("templates", {}).get("core_docs_conf", "")
            if conf_base_template:
                conf_base_path = os.path.join(target_dir, "src", package_name.replace("-", "_"), "docs", "conf_base.py")
                os.makedirs(os.path.dirname(conf_base_path), exist_ok=True)
                with open(conf_base_path, "w") as f:
                    f.write(conf_base_template)
                conf_path = os.path.join(docs_dir, "conf.py")
                with open(conf_path, "w") as f:
                    f.write(conf_base_template)
                logger.info(f"Created core documentation configuration for {package_name}")

    def _create_notebooks_index(self, target_dir: str, ecosystem_packages: List[str]):
        index_content = f"""
# Rompy Ecosystem Examples

This repository contains examples and tutorials for the rompy ecosystem.

## Available Examples

The examples are organized by functionality and required packages:

### Core Examples
Examples using only rompy functionality.

### Plugin Examples
Examples demonstrating specific plugins:

"""
        for package in ecosystem_packages:
            if package != "rompy":
                plugin_name = package.replace("rompy-", "").upper()
                index_content += f"- **{plugin_name}**: Examples using {package}\n"
        index_content += """
## Installation

To run all examples, install the complete ecosystem:

```bash
"""
        for package in ecosystem_packages:
            index_content += f"pip install {package}\n"
        index_content += """```

For specific examples, install only the required packages as noted in each notebook.

## Usage

Each notebook is self-contained and includes:
- Installation requirements
- Setup instructions
- Detailed explanations
- Working examples

Browse the notebooks/ directory to get started!
"""
        index_path = os.path.join(target_dir, "README.md")
        with open(index_path, "w") as f:
            f.write(index_content)
        logger.info("Created notebooks index and README")

    def _perform_post_split_actions(self, target_dir: str, actions: List[Dict[str, Any]]):
        """Perform post-split actions like moving files and updating configs."""
        for action in actions:
            action_type = action.get("action")
            if action_type == "move_files":
                moves = action.get("moves")
                if moves is None:
                    moves = []
                if not isinstance(moves, list):
                    moves = list(moves)
                self._move_files(target_dir, moves)
                self._commit_all_changes(target_dir, "chore(split): move files after split")
            elif action_type == "merge_directory_contents":
                merges = action.get("merges")
                if merges is None:
                    merges = []
                if not isinstance(merges, list):
                    merges = list(merges)
                self._merge_directory_contents(target_dir, merges)
                self._commit_all_changes(target_dir, "chore(split): merge directory contents")
            elif action_type == "create_readme":
                template_name = action.get("template")
                if template_name is None:
                    template_name = ""
                self._create_readme(target_dir, template_name)

            elif action_type == "update_setup":
                package_name = action.get("package_name")
                if package_name is None:
                    package_name = ""
                description = action.get("description")
                if description is None:
                    description = ""
                dependencies = action.get("dependencies")
                if dependencies is None:
                    dependencies = []
                src_layout = action.get("src_layout", False)
                entry_points = action.get("entry_points")
                if entry_points is None:
                    entry_points = {}
                # Ensure types are correct
                if not isinstance(package_name, str):
                    package_name = str(package_name) if package_name is not None else ""
                if package_name is None:
                    package_name = ""
                if not isinstance(description, str):
                    description = str(description) if description is not None else ""
                if description is None:
                    description = ""
                if not isinstance(dependencies, list):
                    dependencies = list(dependencies) if dependencies is not None else []
                if not isinstance(entry_points, dict):
                    entry_points = dict(entry_points) if entry_points is not None else {}
                # Ensure description is not None
                if description is None:
                    description = ""
                self._update_setup_files(
                    target_dir,
                    package_name,
                    description,
                    dependencies,
                    src_layout,
                    entry_points,
                )
                self._commit_all_changes(target_dir, "chore(split): update setup files and config")
            elif action_type == "rename":
                from_path = os.path.join(target_dir, action["from"])
                to_path = os.path.join(target_dir, action["to"])
                import shutil
                if not self.dry_run and os.path.exists(from_path):
                    shutil.move(from_path, to_path)
                    logger.info(f"Renamed {action['from']} to {action['to']}")

            elif action_type == "create_package_structure":
                package_name = action.get("base_package") or ""
                if package_name and not self.dry_run:
                    package_dir = os.path.join(target_dir, package_name)
                    if os.path.exists(package_dir):
                        init_file = os.path.join(package_dir, "__init__.py")
                        if not os.path.exists(init_file):
                            Path(init_file).touch()
                            logger.info(f"Created {init_file}")

            elif action_type == "create_src_layout":
                package_name = action.get("base_package") or ""
                if package_name and not self.dry_run:
                    src_dir = os.path.join(target_dir, "src")
                    os.makedirs(src_dir, exist_ok=True)
                    logger.info(f"Created src directory: {src_dir}")
                    package_dir = os.path.join(src_dir, package_name)
                    if not os.path.exists(package_dir):
                        os.makedirs(package_dir, exist_ok=True)
                        logger.info(f"Created package directory: {package_dir}")
                    is_plugin = (
                        package_name.startswith("rompy_") and package_name != "rompy_core"
                    )
                    plugin_name = package_name.replace("rompy_", "") if is_plugin else ""
                    if plugin_name is None:
                        plugin_name = ""
                    self._create_modern_init_py(
                        package_dir, package_name, is_plugin, plugin_name
                    )
            elif action_type == "create_modern_setup":
                package_name = action.get("package_name") or ""
                package_module = action.get("package_module") or ""
                description = action.get("description")
                if description is None:
                    description = ""
                dependencies = action.get("dependencies")
                if dependencies is None:
                    dependencies = []
                if package_name and package_module and not self.dry_run:
                    self._create_modern_setup_files(
                        target_dir,
                        package_name,
                        package_module,
                        description,
                        dependencies,
                    )
                    self._commit_all_changes(target_dir, "chore(split): create modern setup files")
            elif action_type == "correct_manifest":
                package_name = action.get("package_name")
                description = action.get("description", "")
                dependencies = action.get("dependencies", [])
                src_layout = action.get("src_layout", False)
                entry_points = action.get("entry_points", {})
                self._update_setup_files(
                    target_dir,
                    package_name,
                    description,
                    dependencies,
                    src_layout,
                    entry_points,
                )
                # Also update MANIFEST.in for src layout
                package_module = package_name.replace("-", "_") if package_name else ""
                self._update_manifest_in(target_dir, package_module)
                self._commit_all_changes(target_dir, "chore(split): correct manifest and setup files")

            elif action_type == "create_plugin_docs":
                package_name = action.get("package_name") or ""
                plugin_name = action.get("plugin_name") or ""
                extends_core_docs = action.get("extends_core_docs", False)
                if package_name and plugin_name and not self.dry_run:
                    self._create_plugin_documentation(
                        target_dir, package_name, plugin_name, extends_core_docs
                    )
            elif action_type == "update_docs_config":
                package_name = action.get("package_name") or ""
                is_core = action.get("is_core_package", False)
                plugin_discovery = action.get("plugin_discovery", False)
                if package_name and not self.dry_run:
                    self._update_docs_configuration(
                        target_dir, package_name, is_core, plugin_discovery
                    )
            elif action_type == "create_notebooks_index":
                ecosystem_packages = action.get("ecosystem_packages") or []
                if ecosystem_packages and not self.dry_run:
                    self._create_notebooks_index(target_dir, ecosystem_packages)
            elif action_type == "correct_imports":
                package_type = action.get("package_type") or ""
                target_package = action.get("target_package") or ""
                if package_type and target_package and not self.dry_run:
                    self._correct_imports(target_dir, package_type, target_package)
                self._commit_all_changes(target_dir, "chore(split): correct imports in code and docs")
            elif action_type == "remove_files":
                files_to_remove = action.get("files")
                if files_to_remove is None:
                    files_to_remove = []
                if not isinstance(files_to_remove, list):
                    files_to_remove = list(files_to_remove)
                patterns_to_remove = action.get("patterns")
                if patterns_to_remove is None:
                    patterns_to_remove = []
                if not isinstance(patterns_to_remove, list):
                    patterns_to_remove = list(patterns_to_remove)
                if (files_to_remove or patterns_to_remove) and not self.dry_run:
                    self._remove_files(target_dir, files_to_remove, patterns_to_remove)
            elif action_type == "apply_cookiecutter_template":
                template_repo = action.get("template_repo") or ""
                template_context = action.get("template_context") or {}
                merge_strategy = action.get("merge_strategy", "overlay")
                if template_repo and not self.dry_run:
                    if COOKIECUTTER_AVAILABLE:
                        import shutil
                        import tempfile
                        logger.info(f"Running cookiecutter template: {template_repo} with context: {template_context}")
                        with tempfile.TemporaryDirectory() as temp_dir:
                            # Run cookiecutter to generate template output
                            cookiecutter(
                                template_repo,
                                no_input=True,
                                extra_context=template_context,
                                output_dir=temp_dir
                            )
                            # The output dir will contain a subdir named after project_name
                            project_name = template_context.get("project_name")
                            cookiecutter_output = os.path.join(temp_dir, project_name)
                            if not os.path.exists(cookiecutter_output):
                                # Fallback: try slug
                                project_slug = template_context.get("project_slug", project_name)
                                cookiecutter_output = os.path.join(temp_dir, project_slug)
                            if not os.path.exists(cookiecutter_output):
                                logger.error(f"Cookiecutter output directory not found: {cookiecutter_output}")
                            else:
                                self._merge_cookiecutter_output(target_dir, cookiecutter_output, merge_strategy)
                                logger.info(f"Cookiecutter overlay merge completed for {target_dir}")
                                self._commit_all_changes(target_dir, "chore(split): apply cookiecutter template overlay")
                    else:
                        logger.warning("Cookiecutter is not available. Skipping template application.")

    def _move_files(self, target_dir: str, moves: List[Dict[str, str]]):
        """
        Move files within the repository after filtering, preserving git history.
        Args:
            target_dir: Repository directory
            moves: List of move operations with 'from' and 'to' keys
        """
        for move in moves:
            from_path = os.path.join(target_dir, move["from"])
            to_path = os.path.join(target_dir, move["to"])
            if not self.dry_run:
                if os.path.exists(from_path):
                    os.makedirs(os.path.dirname(to_path), exist_ok=True)
                    if os.path.exists(to_path):
                        if os.path.isdir(to_path):
                            shutil.rmtree(to_path)
                        else:
                            os.remove(to_path)
                    # Check if from_path is tracked by git
                    try:
                        result = subprocess.run([
                            "git", "ls-files", "--error-unmatch", os.path.relpath(from_path, target_dir)
                        ], cwd=target_dir, capture_output=True)
                        tracked = result.returncode == 0
                    except Exception as e:
                        tracked = False
                        logger.warning(f"Failed to check git tracking for {from_path}: {e}")
                    if tracked:
                        # Use git mv to preserve history
                        try:
                            subprocess.run([
                                "git", "mv", os.path.relpath(from_path, target_dir), os.path.relpath(to_path, target_dir)
                            ], cwd=target_dir, check=True)
                            logger.info(f"[GIT] Moved {move['from']} to {move['to']} (git mv)")
                        except Exception as e:
                            logger.error(f"[GIT] git mv failed for {move['from']} to {move['to']}: {e}")
                    else:
                        # Use shutil.move and git add
                        shutil.move(from_path, to_path)
                        logger.info(f"Moved {move['from']} to {move['to']} (shutil.move)")
                        try:
                            subprocess.run([
                                "git", "add", os.path.relpath(to_path, target_dir)
                            ], cwd=target_dir, check=True)
                            logger.info(f"[GIT] Added {move['to']} after move (git add)")
                        except Exception as e:
                            logger.error(f"[GIT] git add failed for {move['to']}: {e}")
                else:
                    logger.warning(f"Source path does not exist, skipping move: {move['from']}")

    def _merge_directory_contents(self, target_dir: str, merges: List[Dict[str, str]]):
        """
        Merge contents of source directory into target directory.
        Args:
            target_dir: Repository directory
            merges: List of merge operations with 'from' and 'to' keys
        """
        for merge in merges:
            from_path = os.path.join(target_dir, merge["from"])
            to_path = os.path.join(target_dir, merge["to"])
            if not self.dry_run:
                if os.path.exists(from_path) and os.path.isdir(from_path):
                    os.makedirs(to_path, exist_ok=True)
                    for item in os.listdir(from_path):
                        src_item = os.path.join(from_path, item)
                        dst_item = os.path.join(to_path, item)
                        if os.path.exists(dst_item):
                            if os.path.isdir(dst_item):
                                shutil.rmtree(dst_item)
                            else:
                                os.remove(dst_item)
                        # Check if src_item is tracked by git
                        try:
                            result = subprocess.run([
                                "git", "ls-files", "--error-unmatch", os.path.relpath(src_item, target_dir)
                            ], cwd=target_dir, capture_output=True)
                            tracked = result.returncode == 0
                        except Exception as e:
                            tracked = False
                            logger.warning(f"Failed to check git tracking for {src_item}: {e}")
                        if tracked:
                            try:
                                subprocess.run([
                                    "git", "mv", os.path.relpath(src_item, target_dir), os.path.relpath(dst_item, target_dir)
                                ], cwd=target_dir, check=True)
                                logger.info(f"[GIT] Moved {os.path.relpath(src_item, target_dir)} to {os.path.relpath(dst_item, target_dir)} (git mv)")
                            except Exception as e:
                                logger.error(f"[GIT] git mv failed for {src_item} to {dst_item}: {e}")
                        else:
                            shutil.move(src_item, dst_item)
                            logger.info(f"Moved {src_item} to {dst_item} (shutil.move)")
                            try:
                                subprocess.run([
                                    "git", "add", os.path.relpath(dst_item, target_dir)
                                ], cwd=target_dir, check=True)
                                logger.info(f"[GIT] Added {os.path.relpath(dst_item, target_dir)} after move (git add)")
                            except Exception as e:
                                logger.error(f"[GIT] git add failed for {dst_item}: {e}")
                    os.rmdir(from_path)
                    logger.info(f"Merged contents of {merge['from']} into {merge['to']}")
                else:
                    logger.warning(f"Source directory does not exist or is not a directory, skipping merge: {merge['from']}")

    def _create_readme(self, target_dir: str, template_name: str):
        """Create README.md from template."""
        if template_name in self.config.get("templates", {}):
            readme_content = self.config["templates"][template_name]
            readme_path = os.path.join(target_dir, "README.md")
            if not self.dry_run:
                with open(readme_path, "w") as f:
                    f.write(readme_content)
                logger.info(f"Created README.md from template {template_name}")
                # Add README.md to git
                try:
                    subprocess.run([
                        "git", "add", os.path.relpath(readme_path, target_dir)
                    ], cwd=target_dir, check=True)
                    logger.info(f"[GIT] Added README.md to git")
                except Exception as e:
                    logger.error(f"[GIT] git add failed for README.md: {e}")

    def _update_setup_files(
        self,
        target_dir: str,
        package_name: str,
        description: str,
        dependencies: List[str] = None,
        src_layout: bool = False,
        entry_points: Dict[str, str] = None,
    ):
        """Update setup.cfg and pyproject.toml for the new package."""
        setup_cfg_path = os.path.join(target_dir, "setup.cfg")
        pyproject_path = os.path.join(target_dir, "pyproject.toml")
        if not self.dry_run:
            if os.path.exists(setup_cfg_path):
                self._update_setup_cfg(
                    setup_cfg_path, package_name, description, src_layout, entry_points
                )
            if os.path.exists(pyproject_path):
                self._update_pyproject_toml(
                    pyproject_path,
                    package_name,
                    description,
                    dependencies or [],
                    src_layout,
                    entry_points,
                )

    def _update_setup_cfg(
        self,
        setup_cfg_path: str,
        package_name: str,
        description: str,
        src_layout: bool = False,
        entry_points: Dict[str, str] = None,
    ):
        """Update setup.cfg file."""
        with open(setup_cfg_path, "r") as f:
            content = f.read()
        lines = content.split("\n")
        new_lines = []
        in_options_section = False
        for line in lines:
            if line.startswith("name ="):
                new_lines.append(f"name = {package_name}")
            elif line.startswith("description ="):
                new_lines.append(f"description = {description}")
            elif line.strip() == "[options]":
                in_options_section = True
                new_lines.append(line)
            elif in_options_section and line.startswith("packages ="):
                if src_layout:
                    new_lines.append("packages = find:")
                else:
                    new_lines.append(line)
            elif in_options_section and line.startswith("package_dir ="):
                if src_layout:
                    continue
                else:
                    new_lines.append(line)
            elif (
                line.strip()
                and line.strip().startswith("[")
                and in_options_section
                and src_layout
            ):
                new_lines.append("")
                new_lines.append("[options.packages.find]")
                new_lines.append("where = src")
                new_lines.append("")
                new_lines.append(line)
                in_options_section = False
            else:
                new_lines.append(line)
        if in_options_section and src_layout:
            new_lines.append("")
            new_lines.append("[options.packages.find]")
            new_lines.append("where = src")
        if entry_points:
            new_lines.append("")
            for group, entry_point in entry_points.items():
                new_lines.append(f"[options.entry_points]")
                new_lines.append(f"{group} =")
                new_lines.append(f"    {entry_point}")
        with open(setup_cfg_path, "w") as f:
            f.write("\n".join(new_lines))
        logger.info(
            f"Updated setup.cfg for {package_name} with {'src layout' if src_layout else 'standard layout'}"
        )

    def _update_pyproject_toml(
        self,
        pyproject_path: str,
        package_name: str,
        description: str,
        dependencies: List[str],
        src_layout: bool = False,
        entry_points: Dict[str, str] = None,
    ):
        """Update pyproject.toml file."""
        try:
            import tomli
            import tomli_w
            with open(pyproject_path, "rb") as f:
                data = tomli.load(f)
            if "project" in data:
                data["project"]["name"] = package_name
                data["project"]["description"] = description
                if dependencies:
                    if "dependencies" not in data["project"]:
                        data["project"]["dependencies"] = []
                    data["project"]["dependencies"].extend(dependencies)
            if src_layout:
                if "tool" not in data:
                    data["tool"] = {}
                if "setuptools" not in data["tool"]:
                    data["tool"]["setuptools"] = {}
                if "packages" not in data["tool"]["setuptools"]:
                    data["tool"]["setuptools"]["packages"] = {}
                data["tool"]["setuptools"]["packages"] = {"find": {"where": ["src"]}}
            if entry_points:
                if "project" not in data:
                    data["project"] = {}
                if "entry-points" not in data["project"]:
                    data["project"]["entry-points"] = {}
                for group, entry_point in entry_points.items():
                    data["project"]["entry-points"][group] = {
                        entry_point.split(" = ")[0]: entry_point.split(" = ")[1]
                    }
            with open(pyproject_path, "wb") as f:
                tomli_w.dump(data, f)
            logger.info(
                f"Updated pyproject.toml for {package_name} with {'src layout' if src_layout else 'standard layout'}"
            )
        except ImportError:
            logger.warning("tomli/tomli_w not available, skipping pyproject.toml update")
        except Exception as e:
            logger.error(f"Failed to update pyproject.toml: {e}")

    def _create_modern_init_py(
        self,
        package_dir: str,
        package_name: str,
        is_plugin: bool = False,
        plugin_name: str = "",
    ):
        """Create a modern __init__.py file with version handling and plugin metadata."""
        init_file = os.path.join(package_dir, "__init__.py")
        if is_plugin and plugin_name:
            init_content = f'''"""
{package_name.replace('_', ' ').title()}

{plugin_name.upper()} plugin for rompy ocean wave modeling framework.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

__plugin_name__ = "{plugin_name}"
__description__ = "{plugin_name.upper()} plugin for rompy ocean wave modeling framework"
__docs_url__ = "https://{package_name}.readthedocs.io/"
__all__ = ["__version__", "__plugin_name__", "__description__", "__docs_url__"]
'''
        else:
            init_content = '''"""
{title}

Core rompy library for ocean wave modeling with plugin system.
"""

try:
    from ._version import __version__
except ImportError:
    __version__ = "0.0.0+unknown"

def discover_plugins():
    try:
        import pkg_resources
        plugins = {{}}
        for entry_point in pkg_resources.iter_entry_points('rompy.plugins'):
            try:
                plugin_module = entry_point.load()
                plugins[entry_point.name] = {{
                    'name': getattr(plugin_module, '__plugin_name__', entry_point.name),
                    'description': getattr(plugin_module, '__description__', ''),
                    'docs_url': getattr(plugin_module, '__docs_url__', ''),
                    'version': getattr(plugin_module, '__version__', 'unknown'),
                    'module': plugin_module,
                }}
            except ImportError:
                continue
        return plugins
    except ImportError:
        return {{}}
__all__ = ["__version__", "discover_plugins"]
''' .format(title=package_name.replace('_', ' ').title())
        with open(init_file, "w") as f:
            f.write(init_content)
        logger.info(f"Created modern __init__.py: {init_file}")

    def _update_manifest_in(self, target_dir: str, package_module: str):
        """Update MANIFEST.in for src layout."""
        import os
        manifest_path = os.path.join(target_dir, "MANIFEST.in")
        # Try to use the modern template if available
        try:
            from templates.modern_setup_templates import (
                MANIFEST_IN_SRC_TEMPLATE, format_template)
            template_vars = {
                'package_module': package_module,
            }
            manifest_content = format_template(MANIFEST_IN_SRC_TEMPLATE, **template_vars)
            with open(manifest_path, "w") as f:
                f.write(manifest_content)
            logger.info(f"Updated MANIFEST.in for src layout using modern template: {manifest_path}")
            return
        except Exception as e:
            logger.warning(f"Could not use modern template for MANIFEST.in: {e}")
        # Fallback: rewrite existing MANIFEST.in
        if os.path.exists(manifest_path):
            with open(manifest_path, "r") as f:
                lines = f.readlines()
            new_lines = []
            for line in lines:
                # Rewrite recursive-include and include paths to src/{package_module}
                if line.startswith("recursive-include "):
                    parts = line.split()
                    if len(parts) > 2 and not parts[1].startswith("src/"):
                        parts[1] = f"src/{package_module}"
                        new_lines.append(" ".join(parts) + "\n")
                    else:
                        new_lines.append(line)
                elif line.startswith("include ") and not line.startswith("include src/"):
                    # For files like include rompy/*.html, rewrite to src/{package_module}/*.html
                    parts = line.split()
                    if len(parts) > 1 and not parts[1].startswith("src/"):
                        if "/" in parts[1]:
                            fname = parts[1].split("/", 1)[-1]
                        else:
                            fname = parts[1]
                        parts[1] = f"src/{package_module}/{fname}"
                        new_lines.append(" ".join(parts) + "\n")
                    else:
                        new_lines.append(line)
                else:
                    new_lines.append(line)
            with open(manifest_path, "w") as f:
                f.writelines(new_lines)
            logger.info(f"Rewrote MANIFEST.in for src layout: {manifest_path}")
        else:
            logger.warning(f"MANIFEST.in not found in {target_dir}, nothing to update.")

    def _create_modern_setup_files(
        self,
        target_dir: str,
        package_name: str,
        package_module: str,
        description: str,
        dependencies: List[str],
    ):
        """Create modern setup files using templates."""
        if MODERN_TEMPLATES_AVAILABLE:
            try:
                from templates.modern_setup_templates import \
                    create_modern_setup_files
                repo_name = package_name
                create_modern_setup_files(
                    target_dir,
                    package_name,
                    package_module,
                    description,
                    repo_name,
                    dependencies,
                )
                logger.info(f"Created modern setup files for {package_name}")
            except Exception as e:
                logger.error(f"Failed to create modern setup files: {e}")
        else:
            logger.warning("Modern templates not available, using basic setup files")
            self._create_basic_modern_files(
                target_dir, package_name, package_module, description, dependencies
            )

    def _create_basic_modern_files(
        self,
        target_dir: str,
        package_name: str,
        package_module: str,
        description: str,
        dependencies: List[str],
    ):
        """Create basic modern setup files without templates."""
        pyproject_content = f"""[build-system]
requires = [\"setuptools>=64\", \"setuptools_scm>=8\"]
build-backend = \"setuptools.build_meta\"

[project]
name = \"{package_name}\"
description = \"{description}\"
readme = \"README.md\"
license = {{file = \"LICENSE\"}}
authors = [
    {{name = \"Rompy Contributors\"}},
]
requires-python = \">=3.8\"
dependencies = [
{chr(10).join([f'    \"{dep}\",' for dep in dependencies])}
]
dynamic = [\"version\"]

[tool.setuptools]
packages = {{find = {{where = [\"src\"]}}}}

[tool.setuptools_scm]
write_to = \"src/{package_module}/_version.py\"
"""
        pyproject_path = os.path.join(target_dir, "pyproject.toml")
        with open(pyproject_path, "w") as f:
            f.write(pyproject_content)
        logger.info(f"Created basic modern pyproject.toml for {package_name}")

    def _correct_imports(self, target_dir: str, package_type: str, target_package: str):
        """
        Correct imports in Python files for the split repository.
        Args:
            target_dir: Target directory containing the split repository
            package_type: Type of package ('core', 'swan', 'schism', 'notebooks')
            target_package: Name of the target package (e.g., 'rompy_core', 'rompy_swan')
        """
        logger.info(f"Correcting imports for {package_type} package: {target_package}")
        corrections = self._get_import_corrections(package_type, target_package)
        file_exts = [".py", ".rst", ".md", ".txt"]
        files_to_process = set()
        for root, dirs, files in os.walk(target_dir):
            dirs[:] = [d for d in dirs if d not in [".git", "notebooks"]]
            for file in files:
                ext = os.path.splitext(file)[1]
                if ext == ".py":
                    file_path = os.path.join(root, file)
                    files_to_process.add(file_path)
                    logger.debug(f"[IMPORT FIX] Found Python file: {file_path}")
                elif ext in [".rst", ".md", ".txt"] or (
                    ext == ".py" and "docs" in root.split(os.sep)
                ):
                    file_path = os.path.join(root, file)
                    files_to_process.add(file_path)
                    logger.debug(f"[IMPORT FIX] Found doc file: {file_path}")
        logger.info(f"Found {len(files_to_process)} files to process for import correction (code + docs)")
        files_modified = 0
        for file_path in files_to_process:
            if self._apply_import_corrections(file_path, corrections):
                logger.debug(f"[IMPORT FIX] Modified imports in: {file_path}")
                files_modified += 1
        logger.info(f"Modified imports in {files_modified} files (code + docs)")

    def _get_import_corrections(self, package_type: str, target_package: str) -> list:
        corrections = []
        if package_type == "core":
            corrections.extend([
                (r"^from rompy\.core", f"from {target_package}.core"),
                (r"^from rompy\.([^.\s]+)", f"from {target_package}.\\1"),
                (r"^import rompy\.core", f"import {target_package}.core"),
                (r"^import rompy\.([^.\s]+)", f"import {target_package}.\\1"),
                (r"^import rompy$", f"import {target_package}"),
                (r"^from rompy import", f"from {target_package} import"),
            ])
        elif package_type == "swan":
            corrections.extend([
                (r"rompy\.swan\.", f"{target_package}."),
                (r"rompy\.swan\b", f"{target_package}"),
                (r"^from rompy\\.([^.\\s]+)", "from rompy.\\1"),
                (r"^import rompy\\.([^.\\s]+)", "import rompy.\\1"),
                (r"^import rompy$", "import rompy"),
                (r"^from rompy import", "from rompy import"),
                (r"rompy\.swan\b", f"{target_package}"),
                (r"parent.parent", "parent"),
                (r'HERE.parent / "templates" / "swancomp"', 'HERE / "templates" / "swancomp"'), 
            ])
        elif package_type == "schism":
            corrections.extend([
                (r"\brompy\.schism\.", f"{target_package}."),
                (r"\brompy\.schism\b", f"{target_package}"),
                (r"``rompy\.schism``", f"``{target_package}``"),
                (r":mod:`rompy\.schism`", f":mod:`{target_package}`"),
                (r":py:mod:`rompy\.schism`", f":py:mod:`{target_package}`"),
                (r"parent.parent", "parent"),
                (r'"test_data"', '"data" / "schism"'),
            ])
        return corrections

    def _apply_import_corrections(self, file_path: str, corrections: list) -> bool:
        import re
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            original_content = content
            modified_content = content
            corrections_sorted = sorted(corrections, key=lambda x: len(x[0]), reverse=True)
            for pattern, replacement in corrections_sorted:
                flags = re.MULTILINE | re.DOTALL
                modified_content = re.sub(pattern, replacement, modified_content, flags=flags)
            if modified_content != original_content:
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(modified_content)
                logger.debug(f"[IMPORT FIX] Actually modified: {file_path}")
                return True
            return False
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            return False

    def _remove_files(
        self, target_dir: str, files_to_remove: list, patterns_to_remove: list
    ):
        import glob
        logger.info(f"Removing unwanted files from {target_dir}")
        removed_count = 0
        for file_path in files_to_remove:
            full_path = os.path.join(target_dir, file_path)
            if os.path.exists(full_path):
                try:
                    if os.path.isdir(full_path):
                        shutil.rmtree(full_path)
                        logger.debug(f"Removed directory: {file_path}")
                    else:
                        os.remove(full_path)
                        logger.debug(f"Removed file: {file_path}")
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {file_path}: {e}")
        for pattern in patterns_to_remove:
            pattern_path = os.path.join(target_dir, pattern)
            matching_files = glob.glob(pattern_path, recursive=True)
            for file_path in matching_files:
                try:
                    if os.path.isdir(file_path):
                        rel_path = os.path.relpath(file_path, target_dir)
                        shutil.rmtree(file_path)
                        logger.debug(f"Removed directory (pattern): {rel_path}")
                    else:
                        rel_path = os.path.relpath(file_path, target_dir)
                        os.remove(file_path)
                        logger.debug(f"Removed file (pattern): {rel_path}")
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {rel_path}: {e}")
        logger.info(f"Removed {removed_count} files/directories")

    def _merge_cookiecutter_output(
        self, target_dir: str, cookiecutter_output: str, merge_strategy: str
    ):
        logger.info(f"Merging cookiecutter output with merge strategy: {merge_strategy}")
        preserve_files = {".git", ".gitignore", "README.md", "LICENSE", "HISTORY.rst"}
        cookiecutter_priority = {
            "pyproject.toml", "setup.cfg", "tox.ini", "requirements_dev.txt", "ruff.toml",
            ".editorconfig", "Makefile", "MANIFEST.in", ".travis.yml", "AUTHORS.rst",
            "CODE_OF_CONDUCT.rst", "CONTRIBUTING.rst"
        }
        exclude_cookiecutter_dirs = {"src", "tests", "docs", "examples", "notebooks"}
        preserve_source_extensions = {
            ".py", ".rst", ".md", ".yml", ".yaml", ".json", ".txt", ".sh"
        }
        for root, dirs, files in os.walk(cookiecutter_output):
            rel_path = os.path.relpath(root, cookiecutter_output)
            if rel_path == ".":
                rel_path = ""
            path_parts = rel_path.split("/") if rel_path else []
            should_exclude_dir = any(
                rel_path.startswith(exclude_dir + "/")
                or rel_path == exclude_dir
                or exclude_dir in path_parts
                for exclude_dir in exclude_cookiecutter_dirs
            )
            if should_exclude_dir:
                logger.debug(f"Skipping cookiecutter directory: {rel_path}")
                continue
            target_subdir = (
                os.path.join(target_dir, rel_path) if rel_path else target_dir
            )
            os.makedirs(target_subdir, exist_ok=True)
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_subdir, file)
                should_copy = False
                if merge_strategy == "replace":
                    should_copy = file not in preserve_files
                elif merge_strategy == "overlay":
                    # Always copy any file from cookiecutter output if it does not exist in split repo,
                    # except for protected files (preserve_files) and .git directory
                    if file in preserve_files or file == ".git":
                        should_copy = False
                        logger.debug(f"Preserving protected file: {file}")
                    elif not os.path.exists(dst_file):
                        should_copy = True
                        logger.debug(f"Adding new file from cookiecutter: {rel_path}/{file}")
                    else:
                        should_copy = False
                        logger.debug(f"Preserving existing file: {rel_path}/{file}")
                elif merge_strategy == "preserve":
                    should_copy = not os.path.exists(dst_file)
                if should_copy:
                    try:
                        shutil.copy2(src_file, dst_file)
                        logger.debug(f"Copied: {file} -> {rel_path}")
                        # Add new file to git
                        try:
                            subprocess.run([
                                "git", "add", os.path.relpath(dst_file, target_dir)
                            ], cwd=target_dir, check=True)
                            logger.info(f"[GIT] Added {os.path.relpath(dst_file, target_dir)} from cookiecutter")
                        except Exception as e:
                            logger.error(f"[GIT] git add failed for {dst_file}: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to copy {file}: {e}")
        logger.info("Cookiecutter template merge completed")

    def _inject_dependencies_from_template_context(
        self, target_dir: str, template_context: dict
    ):
        import ast
        try:
            import tomli
            import tomli_w
        except ImportError:
            logger.warning("tomli/tomli_w not available, skipping pyproject.toml dependency injection")
            return
        pyproject_path = os.path.join(target_dir, "pyproject.toml")
        logger.info(f"[inject_deps] Looking for pyproject.toml at: {pyproject_path}")
        if not os.path.exists(pyproject_path):
            logger.warning(f"[inject_deps] No pyproject.toml found in {target_dir}, skipping dependency injection")
            return
        dependencies = []
        optional_deps = {}
        for key, value in template_context.items():
            if key == "dependencies" and value:
                try:
                    dependencies = (
                        ast.literal_eval(value) if isinstance(value, str) else value
                    )
                    logger.info(f"[inject_deps] Parsed dependencies: {dependencies}")
                except Exception as e:
                    logger.warning(f"[inject_deps] Failed to parse dependencies: {e}")
            elif key.startswith("optional_dependencies_") and value:
                group = key[len("optional_dependencies_") :]
                try:
                    optional_deps[group] = (
                        ast.literal_eval(value) if isinstance(value, str) else value
                    )
                    logger.info(f"[inject_deps] Parsed optional deps for {group}: {optional_deps[group]}")
                except Exception as e:
                    logger.warning(f"[inject_deps] Failed to parse {key}: {e}")
        package_name = (
            template_context.get("package_name")
            or template_context.get("repo_name")
            or os.path.basename(target_dir)
        )
        monorepo_entry_points = None
        if not dependencies and package_name == "rompy":
            monorepo_pyproject = os.path.join(
                os.path.dirname(__file__), "pyproject.toml"
            )
            logger.info(f"[inject_deps] No dependencies in template_context for rompy, extracting from monorepo: {monorepo_pyproject}")
            if os.path.exists(monorepo_pyproject):
                with open(monorepo_pyproject, "rb") as f:
                    monorepo_data = tomli.load(f)
                deps = monorepo_data.get("project", {}).get("dependencies", [])
                opt_deps = monorepo_data.get("project", {}).get("optional-dependencies", {})
                dependencies = deps
                optional_deps = opt_deps
                monorepo_entry_points = monorepo_data.get("project", {}).get("entry-points", {})
                logger.info(f"[inject_deps] Extracted dependencies from monorepo: {dependencies}")
                logger.info(f"[inject_deps] Extracted optional dependencies from monorepo: {optional_deps}")
                logger.info(f"[inject_deps] Extracted entry-points from monorepo: {monorepo_entry_points}")
            else:
                logger.warning(f"[inject_deps] Could not find monorepo pyproject.toml at {monorepo_pyproject}")
        with open(pyproject_path, "rb") as f:
            data = tomli.load(f)
        if "project" not in data:
            data["project"] = {}
        if dependencies:
            data["project"]["dependencies"] = dependencies
        if optional_deps:
            if "optional-dependencies" not in data["project"]:
                data["project"]["optional-dependencies"] = {}
            for group, deps in optional_deps.items():
                data["project"]["optional-dependencies"][group] = deps
        if package_name == "rompy":
            entry_points = data["project"].get("entry-points", {})
            rompy_source = entry_points.get("rompy.source") if entry_points else None
            if (
                not rompy_source
                or (isinstance(rompy_source, dict) and not rompy_source)
            ) and monorepo_entry_points:
                if "entry-points" not in data["project"]:
                    data["project"]["entry-points"] = {}
                if "rompy.source" in monorepo_entry_points:
                    data["project"]["entry-points"]["rompy.source"] = (
                        monorepo_entry_points["rompy.source"]
                    )
                    logger.info(f"[inject_deps] Injected rompy.source entry-points from monorepo into split package.")
            if "rompy.config" in data["project"].get("entry-points", {}):
                data["project"]["entry-points"]["rompy.config"] = {
                    "base": "rompy.core.config:BaseConfig"
                }
                logger.info(f"[inject_deps] Set rompy.config entry-point to only base=rompy.core.config:BaseConfig for core package.")
        with open(pyproject_path, "wb") as f:
            tomli_w.dump(data, f)
        logger.info(f"[inject_deps] Injected dependencies into pyproject.toml in {target_dir}")
        with open(pyproject_path, "r") as f:
            logger.info(f"[inject_deps] Final pyproject.toml contents:\n{f.read()}")

    def _cleanup_empty_directories(self, target_dir: str):
        """Remove empty directories after filtering, but exclude .git directories."""
        if self.dry_run:
            return
        for root, dirs, files in os.walk(target_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
                if ".git" in dir_path:
                    continue
                try:
                    if not os.listdir(dir_path):
                        os.rmdir(dir_path)
                        logger.info(f"Removed empty directory: {dir_path}")
                except OSError:
                    pass


    def split_repository(self, repo_name: str, repo_config: Dict[str, Any]):
        """
        Split a single repository according to its configuration.
        """
        logger.info(f"Starting split for repository: {repo_name}")
        logger.info(f"Description: {repo_config.get('description', 'No description')}")
        target_dir = self._create_target_directory(repo_name)
        try:
            self._clone_source_repo(target_dir)
            paths = repo_config.get("paths", [])
            if paths:
                self._filter_repository(target_dir, paths)
                self._commit_all_changes(target_dir, "chore(split): initial filter with git-filter-repo")
            post_actions = repo_config.get("post_split_actions", [])
            if post_actions:
                self._perform_post_split_actions(target_dir, post_actions)
            self._cleanup_empty_directories(target_dir)
            if not self.dry_run:
                self._run_command(["git", "gc", "--aggressive"], cwd=target_dir)
                self._run_command(["git", "repack", "-ad"], cwd=target_dir)
            logger.info(f"Successfully created repository: {repo_name}")
        except Exception as e:
            logger.error(f"Failed to split repository {repo_name}: {e}")
            if not self.dry_run and self.config.get("cleanup_after_split", False):
                logger.info(f"Cleaning up failed split: {target_dir}")
                shutil.rmtree(target_dir, ignore_errors=True)
            raise

    def run(self):
        """Run the repository splitting process."""
        logger.info("Starting repository splitting process")
        self._check_prerequisites()
        if not self.dry_run:
            os.makedirs(self.target_base_dir, exist_ok=True)
        repositories = self.config.get("repositories", {})
        for repo_name, repo_config in repositories.items():
            try:
                self.split_repository(repo_name, repo_config)
            except Exception as e:
                logger.error(f"Failed to process {repo_name}: {e}")
                if not self.config.get("continue_on_error", False):
                    sys.exit(1)
        logger.info("Repository splitting completed successfully")
        self._print_summary(repositories)

    def _print_summary(self, repositories: Dict[str, Any]):
        print("\n" + "=" * 60)
        print("REPOSITORY SPLITTING SUMMARY")
        print("=" * 60)
        for repo_name, repo_config in repositories.items():
            target_dir = os.path.join(self.target_base_dir, repo_name)
            print(f"\n📁 {repo_name}")
            print(f"   Description: {repo_config.get('description', 'N/A')}")
            print(f"   Location: {target_dir}")
            if not self.dry_run and os.path.exists(target_dir):
                try:
                    result = self._run_command(["git", "rev-list", "--count", "HEAD"], cwd=target_dir, check=False)
                    commit_count = result.stdout.strip() if result.returncode == 0 else "Unknown"
                    print(f"   Commits: {commit_count}")
                    result = self._run_command(["git", "branch", "-a"], cwd=target_dir, check=False)
                    branch_count = len([l for l in result.stdout.split("\n") if l.strip()]) if result.returncode == 0 else 0
                    print(f"   Branches: {branch_count}")
                except Exception:
                    pass
        print(f"\n🎯 Next steps:")
        print(f"   1. Review each split repository in: {self.target_base_dir}")
        print(f"   2. Test that each repository works independently:")
        print(f"      cd {self.target_base_dir}/<repo-name>")
        print(f"      pip install -e .[dev]")
        print(f"      pytest")
        print(f"   3. Create remote repositories and push:")
        print(f"      git remote add origin <remote-url>")
        print(f"      git push -u origin --all")
        print(f"      git push --tags")
        print(f"   4. Set up CI/CD and documentation")
        print("\n💡 Modern src/ layout benefits:")
        print("   - Cleaner package structure")
        print("   - Better import isolation")
        print("   - Standard Python packaging practices")
        print("   - Improved testing and development workflow")
        print("\n")

# --- END: RepositorySplitter and split logic ---



# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Unified ROMPY split automation")
    parser.add_argument("--config", default="repo_split_config_with_cookiecutter.yaml", help="Path to split config file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    parser.add_argument("--no-test", action="store_true", help="Skip comprehensive testing")
    parser.add_argument("--retry-setup", action="store_true", help="Retry setup steps if they fail")
    parser.add_argument("--split-repos-dir", default="../split-repos", help="Directory for split repositories")
    return parser.parse_args()


def fix_rompy_swan_pyproject():
    """Fix rompy-swan pyproject.toml"""
    content = '''[build-system]
requires = ["setuptools", "versioneer[toml]"]
build-backend = "setuptools.build_meta"

[project]
name = "rompy-swan"
description = "SWAN wave model plugin for rompy"
readme = "README.rst"
authors = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
maintainers = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
license = {text = "MIT license"}
requires-python = ">=3.10"
dependencies = [
    "pydantic>2",
]

dynamic = ["version"]

[project.optional-dependencies]
dev = [
    "coverage",
    "mypy",
    "pytest",
    "ruff"
]
test = [
    "pytest",
    "coverage"
]

[project.entry-points."rompy.config"]
swan = "rompy_swan.config:SwanConfig"
swan_components = "rompy_swan.config:SwanConfigComponents"

[project.urls]
bugs = "https://github.com/rom-py/rompy-swan/issues"
changelog = "https://github.com/rom-py/rompy-swan/blob/master/changelog.md"
homepage = "https://github.com/rom-py/rompy-swan"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
"*" = ["*.y*ml", "*.csv", "*.html"]

[tool.setuptools.dynamic]
version = {attr = "rompy_swan.__version__"}

[tool.pytest.ini_options]
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"

[tool.black]
line-length = 88
'''
    return content

def fix_rompy_schism_pyproject():
    """Fix rompy-schism pyproject.toml"""
    content = '''[build-system]
requires = ["setuptools", "versioneer[toml]"]
build-backend = "setuptools.build_meta"

[project]
name = "rompy-schism"
description = "SCHISM model plugin for rompy"
readme = "README.rst"
authors = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
maintainers = [
  {name = "Rompy Contributors", email = "developers@rompy.com"}
]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Natural Language :: English",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Physics",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]
license = {text = "MIT license"}
requires-python = ">=3.10"
dependencies = [
    "pydantic>2",
    "pylibs-ocean",
    "pytmd",
]

dynamic = ["version"]

[project.optional-dependencies]
dev = [
    "coverage",
    "mypy",
    "pytest",
    "ruff"
]
test = [
    "pytest",
    "coverage"
]

[project.entry-points."rompy.config"]
schismcsiro = "rompy_schism.config:SchismCSIROConfig"
schism = "rompy_schism.config:SCHISMConfig"

[project.urls]
bugs = "https://github.com/rom-py/rompy-schism/issues"
changelog = "https://github.com/rom-py/rompy-schism/blob/master/changelog.md"
homepage = "https://github.com/rom-py/rompy-schism"

[tool.setuptools]
package-dir = {"" = "src"}

[tool.setuptools.package-data]
"*" = ["*.y*ml", "*.csv", "*.html"]

[tool.setuptools.dynamic]
version = {attr = "rompy_schism.__version__"}

[tool.pytest.ini_options]
log_cli = true
log_cli_level = "INFO"
log_cli_format = "%(asctime)s [%(levelname)8s] %(message)s (%(filename)s:%(lineno)s)"
log_cli_date_format = "%Y-%m-%d %H:%M:%S"

[tool.black]
line-length = 88
'''
    return content



def get_core_version():
    """Read the version string from rompy/rompy/__init__.py"""
    init_path = Path(__file__).parent / "rompy" / "__init__.py"
    version = "0.0.0"
    if init_path.exists():
        with open(init_path, "r") as f:
            for line in f:
                if line.strip().startswith("__version__"):
                    version = line.split("=")[1].strip().strip('"').strip("'")
                    break
    return version

def inject_version_to_plugin_init(plugin_repo_path, package_module, version_str):
    """Inject __version__ = "..." into plugin __init__.py"""
    init_path = plugin_repo_path / "src" / package_module / "__init__.py"
    if not init_path.exists():
        init_path.parent.mkdir(parents=True, exist_ok=True)
        init_content = f'__version__ = "{version_str}"\n'
    else:
        with open(init_path, "r") as f:
            lines = f.readlines()
        # Remove any existing __version__ line
        lines = [line for line in lines if not line.strip().startswith("__version__")]
        init_content = f'__version__ = "{version_str}"\n' + "".join(lines)
    with open(init_path, "w") as f:
        f.write(init_content)
    logger.info(f"✅ Injected version {version_str} into {init_path}")

def fix_dependencies(split_repos_dir):
    """Fix dependencies in all split repositories."""
    logger.info("\n🔧 Fixing dependencies in split repositories...")
    split_dir = Path(split_repos_dir).resolve()
    plugin_repos = {
        "rompy-swan": fix_rompy_swan_pyproject,
        "rompy-schism": fix_rompy_schism_pyproject,
    }
    version_str = get_core_version()
    # First process plugin repos with their templates
    for repo_name, fix_func in plugin_repos.items():
        repo_path = split_dir / repo_name
        pyproject_path = repo_path / "pyproject.toml"
        if not repo_path.exists():
            logger.warning(f"⚠️  Repository not found: {repo_path}")
            continue
        logger.info(f"🔧 Fixing {repo_name}/pyproject.toml...")
        content = fix_func()
        with open(pyproject_path, 'w') as f:
            f.write(content)
        logger.info(f"✅ Fixed {repo_name}/pyproject.toml with template")
        # Inject version into plugin __init__.py
        package_module = "rompy_swan" if repo_name == "rompy-swan" else "rompy_schism"
        inject_version_to_plugin_init(repo_path, package_module, version_str)
    
    # Now handle rompy specially - use the cookiecutter-generated pyproject.toml and inject deps
    repo_name = "rompy"
    repo_path = split_dir / repo_name
    pyproject_path = repo_path / "pyproject.toml"
    if repo_path.exists():
        logger.info(f"🔧 Processing {repo_name} - keeping cookiecutter pyproject.toml and injecting dependencies")
        
        # Make sure the cookiecutter-generated file has proper src layout
        try:
            import tomli
            import tomli_w
            
            if pyproject_path.exists():
                with open(pyproject_path, "rb") as f:
                    data = tomli.load(f)
                
                # Ensure src layout is properly configured
                if "tool" not in data:
                    data["tool"] = {}
                if "setuptools" not in data["tool"]:
                    data["tool"]["setuptools"] = {}
                if "packages" not in data["tool"]["setuptools"]:
                    data["tool"]["setuptools"]["packages"] = {}
                data["tool"]["setuptools"]["packages"] = {"find": {"where": ["src"]}}
                
                # Inject dependencies and entry points from monorepo
                monorepo_pyproject = Path(__file__).parent / "pyproject.toml"
                if monorepo_pyproject.exists():
                    with open(monorepo_pyproject, "rb") as f:
                        monorepo_data = tomli.load(f)
                    
                    # Get dependencies from monorepo
                    deps = monorepo_data.get("project", {}).get("dependencies", [])
                    opt_deps = monorepo_data.get("project", {}).get("optional-dependencies", {})
                    entry_points = monorepo_data.get("project", {}).get("entry-points", {})
                    
                    # Apply them to the cookiecutter-generated pyproject.toml
                    if "project" not in data:
                        data["project"] = {}
                    
                    data["project"]["dependencies"] = deps
                    
                    if "optional-dependencies" not in data["project"]:
                        data["project"]["optional-dependencies"] = {}
                    for group, group_deps in opt_deps.items():
                        data["project"]["optional-dependencies"][group] = group_deps
                    
                    if "entry-points" not in data["project"]:
                        data["project"]["entry-points"] = {}
                    
                    # Add entry points
                    for group, group_entries in entry_points.items():
                        data["project"]["entry-points"][group] = group_entries
                    
                    # Ensure rompy.config entry point is properly set for the core package
                    if "rompy.config" in data["project"]["entry-points"]:
                        data["project"]["entry-points"]["rompy.config"] = {
                            "base": "rompy.core.config:BaseConfig"
                        }
                    
                    # Make sure setuptools_scm is configured properly
                    if "tool" not in data:
                        data["tool"] = {}
                    if "setuptools_scm" not in data["tool"]:
                        data["tool"]["setuptools_scm"] = {}
                    data["tool"]["setuptools_scm"]["write_to"] = "src/rompy/_version.py"
                    
                    # Write the modified pyproject.toml
                    with open(pyproject_path, "wb") as f:
                        tomli_w.dump(data, f)
                    
                    logger.info(f"✅ Updated {repo_name}/pyproject.toml with dependencies and entry points from monorepo")
                else:
                    logger.warning(f"⚠️ Monorepo pyproject.toml not found: {monorepo_pyproject}")
                    logger.info(f"✅ Only updated {repo_name}/pyproject.toml src layout configuration")
            else:
                logger.warning(f"⚠️ {repo_name}/pyproject.toml not found: {pyproject_path}")
        except ImportError:
            logger.warning("⚠️ tomli/tomli_w not available, could not update pyproject.toml")
        except Exception as e:
            logger.error(f"❌ Failed to update {repo_name}/pyproject.toml: {e}")
    else:
        logger.warning(f"⚠️  Repository not found: {repo_path}")
    
    logger.info("🎉 All pyproject.toml files have been fixed!")

def check_pyproject_entry_points(pyproject_path: Path):
    """Check entry points defined in pyproject.toml."""
    import re
    if not pyproject_path.exists():
        logger.error(f"❌ pyproject.toml not found: {pyproject_path}")
        return {}
    try:
        with open(pyproject_path, 'r') as f:
            content = f.read()
        entry_point_groups = {}
        pattern = r'\[project\.entry-points\."([^"]+)"\](.*?)(?=\[project\.|\Z)'
        for match in re.finditer(pattern, content, re.DOTALL):
            group_name = match.group(1)
            group_content = match.group(2)
            entry_points_list = []
            for line in group_content.strip().split('\n'):
                line = line.strip()
                if '=' in line:
                    entry_name = line.split('=')[0].strip().strip('"\'')
                    entry_points_list.append(entry_name)
            entry_point_groups[group_name] = entry_points_list
        return entry_point_groups
    except Exception as e:
        logger.error(f"❌ Error reading pyproject.toml: {e}")
        return {}

def test_load_entry_points_function(split_repos_dir: Path) -> bool:
    """Test the load_entry_points function by importing it and calling it."""
    import sys
    try:
        sys.path.insert(0, str(split_repos_dir / "rompy" / "src"))
        import rompy.utils
        load_entry_points = rompy.utils.load_entry_points
        sources = load_entry_points("rompy.source")
        logger.info(f"  ✅ load_entry_points('rompy.source') returned {len(sources)} entries")
        sources_ts = load_entry_points("rompy.source", etype="timeseries")
        logger.info(f"  ✅ load_entry_points('rompy.source', etype='timeseries') returned {len(sources_ts)} entries")
        if len(sources_ts) == 0:
            logger.warning("⚠️ SOURCE_TYPES_TS is empty! This will cause errors in data.py")
            logger.warning("   Make sure at least one entry point with ':timeseries' is defined")
            return False
        return True
    except Exception as e:
        logger.error(f"❌ Error testing load_entry_points: {e}")
        return False
    finally:
        if str(split_repos_dir / "rompy" / "src") in sys.path:
            sys.path.remove(str(split_repos_dir / "rompy" / "src"))

def test_source_types_ts_in_data(split_repos_dir: Path) -> bool:
    """Test that SOURCE_TYPES_TS is properly defined and used in data.py."""
    import re
    data_py_path = split_repos_dir / "rompy" / "src" / "rompy" / "core" / "data.py"
    if not data_py_path.exists():
        logger.error(f"❌ data.py not found: {data_py_path}")
        return False
    try:
        with open(data_py_path, 'r') as f:
            content = f.read()
        if "SOURCE_TYPES_TS = load_entry_points" not in content:
            logger.error("❌ SOURCE_TYPES_TS is not defined in data.py")
            return False
        union_pattern = r"source:\s+Union\[([^\]]+)\]"
        union_match = re.search(union_pattern, content)
        if not union_match:
            logger.error("❌ Union type not found in data.py")
            return False
        union_contents = union_match.group(1)
        logger.info(f"📊 Union type in data.py: Union[{union_contents}]")
        if "AnyPath" in union_contents and "SOURCE_TYPES_TS" in union_contents:
            logger.warning("⚠️ Both SOURCE_TYPES_TS and AnyPath are in the Union type")
            logger.warning("   This may cause errors if SOURCE_TYPES_TS is empty")
            return False
        return True
    except Exception as e:
        logger.error(f"❌ Error checking data.py: {e}")
        return False

def verify_entry_points(split_repos_dir):
    """Verify entry points in all split repositories."""
    logger.info("\n🔍 Verifying entry points...")
    split_dir = Path(split_repos_dir).resolve()
    package_names = ['rompy', 'rompy-swan', 'rompy-schism']
    all_verified = True
    for package in package_names:
        if package == 'rompy':
            package_dir = split_dir / "rompy" / "src" / "rompy"
        elif package == 'rompy-swan':
            package_dir = split_dir / "rompy-swan" / "src" / "rompy_swan"
        elif package == 'rompy-schism':
            package_dir = split_dir / "rompy-schism" / "src" / "rompy_schism"
        else:
            logger.warning(f"⚠️ Unknown package: {package}")
            all_verified = False
            continue
        if not package_dir.exists():
            logger.warning(f"⚠️ Package directory not found: {package_dir}")
            all_verified = False
            continue
        pyproject_path = package_dir / "pyproject.toml"
        entry_point_groups = check_pyproject_entry_points(pyproject_path)
        logger.info(f"📊 Entry point groups defined in {package}:")
        for group, entries in entry_point_groups.items():
            logger.info(f"  - {group}: {len(entries)} entries")
            timeseries_entries = [e for e in entries if ":timeseries" in e]
            if timeseries_entries:
                logger.info(f"    ✅ Found {len(timeseries_entries)} timeseries entries: {timeseries_entries}")
            elif package == "rompy" and "rompy_core.source" in group:
                logger.warning(f"    ⚠️ No timeseries entries found in {group}")
                all_verified = False
    # Test the load_entry_points function
    logger.info("\n🔍 Testing load_entry_points function...")
    load_success = test_load_entry_points_function(split_dir)
    if not load_success:
        all_verified = False
    # Test SOURCE_TYPES_TS in data.py
    logger.info("\n🔍 Testing SOURCE_TYPES_TS in data.py...")
    data_success = test_source_types_ts_in_data(split_dir)
    if not data_success:
        all_verified = False
    if all_verified:
        logger.info("✅ All entry points verified successfully!")
    else:
        logger.error("❌ Some entry point verifications failed")
    return all_verified

def main():
    args = parse_args()
    logger.info("🚀 Starting Unified ROMPY Split Automation")
    logger.info("=" * 80)
    # Step 1: Split repository
    splitter = RepositorySplitter(args.config, args.dry_run)
    splitter.run()
    # Always use the config value for split_repos_dir
    split_repos_dir = splitter.target_base_dir
    # Step 2: Fix dependencies
    fix_dependencies(split_repos_dir)
    # Step 3: Fix imports
    # --- BEGIN: ImportCorrector class ---
    fixer = FinalTestFixer(Path(split_repos_dir), args.dry_run)
    if not args.no_test:
        fixer.run_all_fixes()
        # Final commit for all split repos after test/config fixes
        for repo_name in os.listdir(split_repos_dir):
            repo_path = os.path.join(split_repos_dir, repo_name)
            if os.path.isdir(repo_path) and os.path.exists(os.path.join(repo_path, ".git")):
                splitter._commit_all_changes(repo_path, "chore(split): final test/config fixes")
    logger.info("🎉 Unified split automation complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
