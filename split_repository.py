#!/usr/bin/env python3
"""
Repository Splitting Script

This script splits a monorepo into multiple repositories while preserving
git history, branches, and tags. It uses a YAML configuration file to define
the splitting strategy.

Usage:
    python split_repository.py [--config CONFIG_FILE] [--dry-run]

Requirements:
    - git-filter-repo (install with: pip install git-filter-repo)
    - PyYAML (install with: pip install PyYAML)
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

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

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RepositorySplitter:
    """
    Handles the splitting of a monorepo into multiple repositories.
    """

    def __init__(self, config_path: str, dry_run: bool = False):
        """
        Initialize the repository splitter.

        Args:
            config_path: Path to the YAML configuration file
            dry_run: If True, only simulate the operations
        """
        self.config_path = config_path
        self.dry_run = dry_run
        self.config = self._load_config()
        self.source_repo = os.path.abspath(self.config['source_repo'])
        self.target_base_dir = os.path.abspath(self.config['target_base_dir'])

    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    def _run_command(self, cmd: List[str], cwd: str = None, check: bool = True) -> subprocess.CompletedProcess:
        """
        Run a shell command with logging.

        Args:
            cmd: Command and arguments as a list
            cwd: Working directory for the command
            check: Whether to raise an exception on non-zero exit code

        Returns:
            CompletedProcess object
        """
        cmd_str = ' '.join(cmd)
        logger.info(f"Running: {cmd_str} (cwd: {cwd or 'current'})")

        if self.dry_run:
            logger.info("DRY RUN: Command would be executed")
            return subprocess.CompletedProcess(cmd, 0, '', '')

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=check,
                capture_output=True,
                text=True
            )
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
        """Check that required tools are available."""
        try:
            self._run_command(['git', '--version'])
            logger.info("Git is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("Git is not available or not working")
            sys.exit(1)

        try:
            self._run_command(['git-filter-repo', '--version'])
            logger.info("git-filter-repo is available")
        except (subprocess.CalledProcessError, FileNotFoundError):
            logger.error("git-filter-repo is not available. Install with: pip install git-filter-repo")
            sys.exit(1)

    def _create_target_directory(self, repo_name: str) -> str:
        """Create the target directory for a split repository."""
        target_dir = os.path.join(self.target_base_dir, repo_name)

        if not self.dry_run:
            os.makedirs(target_dir, exist_ok=True)

        logger.info(f"Target directory: {target_dir}")
        return target_dir

    def _clone_source_repo(self, target_dir: str):
        """Clone the source repository to the target directory."""
        logger.info(f"Cloning source repository to {target_dir}")

        # Remove existing directory if it exists
        if not self.dry_run and os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        # Clone as a normal repository (not bare)
        self._run_command([
            'git', 'clone', self.source_repo, target_dir
        ])

    def _build_path_filters(self, paths: List[str]) -> List[str]:
        """
        Build git-filter-repo path arguments from path specifications.

        Args:
            paths: List of path patterns (paths starting with ! are excluded)

        Returns:
            List of git-filter-repo arguments
        """
        include_paths = []
        exclude_paths = []

        # Separate includes and excludes
        for path in paths:
            if path.startswith('!'):
                exclude_paths.append(path[1:])  # Remove the !
            else:
                include_paths.append(path)

        filters = []

        # Add all include paths first
        for path in include_paths:
            filters.extend(['--path', path])

        return filters

    def _filter_repository(self, target_dir: str, paths: List[str]):
        """
        Filter the repository to keep only specified paths.

        Args:
            target_dir: Directory containing the cloned repository
            paths: List of path patterns to keep/exclude
        """
        logger.info(f"Filtering repository at {target_dir}")
        logger.info(f"Path filters: {paths}")

        # Build the filter command
        cmd = ['git-filter-repo', '--force']

        # Use the helper method to build path filters
        path_filters = self._build_path_filters(paths)
        cmd.extend(path_filters)



        # Run the filter
        self._run_command(cmd, cwd=target_dir)

    def _move_files(self, target_dir: str, moves: List[Dict[str, str]]):
        """
        Move files within the repository after filtering.

        Args:
            target_dir: Repository directory
            moves: List of move operations with 'from' and 'to' keys
        """
        for move in moves:
            from_path = os.path.join(target_dir, move['from'])
            to_path = os.path.join(target_dir, move['to'])

            if not self.dry_run:
                if os.path.exists(from_path):
                    # Create target directory if it doesn't exist
                    os.makedirs(os.path.dirname(to_path), exist_ok=True)

                    # If destination exists, remove it first
                    if os.path.exists(to_path):
                        if os.path.isdir(to_path):
                            shutil.rmtree(to_path)
                        else:
                            os.remove(to_path)

                    shutil.move(from_path, to_path)
                    logger.info(f"Moved {move['from']} to {move['to']}")
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
            from_path = os.path.join(target_dir, merge['from'])
            to_path = os.path.join(target_dir, merge['to'])

            if not self.dry_run:
                if os.path.exists(from_path) and os.path.isdir(from_path):
                    # Create target directory if it doesn't exist
                    os.makedirs(to_path, exist_ok=True)

                    # Move all contents from source to target
                    for item in os.listdir(from_path):
                        src_item = os.path.join(from_path, item)
                        dst_item = os.path.join(to_path, item)

                        # If destination item exists, remove it first
                        if os.path.exists(dst_item):
                            if os.path.isdir(dst_item):
                                shutil.rmtree(dst_item)
                            else:
                                os.remove(dst_item)

                        shutil.move(src_item, dst_item)

                    # Remove the now-empty source directory
                    os.rmdir(from_path)
                    logger.info(f"Merged contents of {merge['from']} into {merge['to']}")
                else:
                    logger.warning(f"Source directory does not exist or is not a directory, skipping merge: {merge['from']}")

    def _create_readme(self, target_dir: str, template_name: str):
        """Create README.md from template."""
        if template_name in self.config.get('templates', {}):
            readme_content = self.config['templates'][template_name]
            readme_path = os.path.join(target_dir, 'README.md')

            if not self.dry_run:
                with open(readme_path, 'w') as f:
                    f.write(readme_content)
                logger.info(f"Created README.md from template {template_name}")

    def _update_setup_files(self, target_dir: str, package_name: str,
                           description: str, dependencies: List[str] = None, src_layout: bool = False,
                           entry_points: Dict[str, str] = None):
        """Update setup.cfg and pyproject.toml for the new package."""
        setup_cfg_path = os.path.join(target_dir, 'setup.cfg')
        pyproject_path = os.path.join(target_dir, 'pyproject.toml')

        if not self.dry_run:
            # Update setup.cfg if it exists
            if os.path.exists(setup_cfg_path):
                self._update_setup_cfg(setup_cfg_path, package_name, description, src_layout, entry_points)

            # Update pyproject.toml if it exists
            if os.path.exists(pyproject_path):
                self._update_pyproject_toml(pyproject_path, package_name, description, dependencies or [], src_layout, entry_points)

    def _update_setup_cfg(self, setup_cfg_path: str, package_name: str, description: str, src_layout: bool = False,
                          entry_points: Dict[str, str] = None):
        """Update setup.cfg file."""
        # This is a simplified update - you might want to use configparser for more robust handling
        with open(setup_cfg_path, 'r') as f:
            content = f.read()

        # Replace name and description
        lines = content.split('\n')
        new_lines = []
        in_options_section = False

        for line in lines:
            if line.startswith('name ='):
                new_lines.append(f'name = {package_name}')
            elif line.startswith('description ='):
                new_lines.append(f'description = {description}')
            elif line.strip() == '[options]':
                in_options_section = True
                new_lines.append(line)
            elif in_options_section and line.startswith('packages ='):
                if src_layout:
                    new_lines.append('packages = find:')
                else:
                    new_lines.append(line)
            elif in_options_section and line.startswith('package_dir ='):
                if src_layout:
                    # Skip the old package_dir line, we'll add our own
                    continue
                else:
                    new_lines.append(line)
            elif line.strip() and line.strip().startswith('[') and in_options_section and src_layout:
                # We're entering a new section, add package_dir if src layout
                new_lines.append('')
                new_lines.append('[options.packages.find]')
                new_lines.append('where = src')
                new_lines.append('')
                new_lines.append(line)
                in_options_section = False
            else:
                new_lines.append(line)

        # If we're still in options section at end of file and using src layout
        if in_options_section and src_layout:
            new_lines.append('')
            new_lines.append('[options.packages.find]')
            new_lines.append('where = src')

        # Add entry points if specified
        if entry_points:
            new_lines.append('')
            for group, entry_point in entry_points.items():
                new_lines.append(f'[options.entry_points]')
                new_lines.append(f'{group} =')
                new_lines.append(f'    {entry_point}')

        with open(setup_cfg_path, 'w') as f:
            f.write('\n'.join(new_lines))

        logger.info(f"Updated setup.cfg for {package_name} with {'src layout' if src_layout else 'standard layout'}")

    def _update_pyproject_toml(self, pyproject_path: str, package_name: str,
                              description: str, dependencies: List[str], src_layout: bool = False,
                              entry_points: Dict[str, str] = None):
        """Update pyproject.toml file."""
        try:
            import tomli_w
            import tomli

            with open(pyproject_path, 'rb') as f:
                data = tomli.load(f)

            # Update project metadata
            if 'project' in data:
                data['project']['name'] = package_name
                data['project']['description'] = description

                # Add dependencies if specified
                if dependencies:
                    if 'dependencies' not in data['project']:
                        data['project']['dependencies'] = []
                    data['project']['dependencies'].extend(dependencies)

            # Update build system for src layout
            if src_layout:
                if 'tool' not in data:
                    data['tool'] = {}
                if 'setuptools' not in data['tool']:
                    data['tool']['setuptools'] = {}
                if 'packages' not in data['tool']['setuptools']:
                    data['tool']['setuptools']['packages'] = {}

                # Configure setuptools to find packages in src
                data['tool']['setuptools']['packages'] = {'find': {'where': ['src']}}

            # Add entry points if specified
            if entry_points:
                if 'project' not in data:
                    data['project'] = {}
                if 'entry-points' not in data['project']:
                    data['project']['entry-points'] = {}

                for group, entry_point in entry_points.items():
                    data['project']['entry-points'][group] = {entry_point.split(' = ')[0]: entry_point.split(' = ')[1]}

            with open(pyproject_path, 'wb') as f:
                tomli_w.dump(data, f)

            logger.info(f"Updated pyproject.toml for {package_name} with {'src layout' if src_layout else 'standard layout'}")

        except ImportError:
            logger.warning("tomli/tomli_w not available, skipping pyproject.toml update")
        except Exception as e:
            logger.error(f"Failed to update pyproject.toml: {e}")

    def _perform_post_split_actions(self, target_dir: str, actions: List[Dict[str, Any]]):
        """Perform post-split actions like moving files and updating configs."""
        for action in actions:
            action_type = action.get('action')

            if action_type == 'move_files':
                self._move_files(target_dir, action.get('moves', []))

            elif action_type == 'merge_directory_contents':
                self._merge_directory_contents(target_dir, action.get('merges', []))

            elif action_type == 'create_readme':
                self._create_readme(target_dir, action.get('template'))

            elif action_type == 'update_setup':
                self._update_setup_files(
                    target_dir,
                    action.get('package_name'),
                    action.get('description'),
                    action.get('dependencies', []),
                    action.get('src_layout', False),
                    action.get('entry_points', {})
                )

            elif action_type == 'rename':
                from_path = os.path.join(target_dir, action['from'])
                to_path = os.path.join(target_dir, action['to'])
                if not self.dry_run and os.path.exists(from_path):
                    shutil.move(from_path, to_path)
                    logger.info(f"Renamed {action['from']} to {action['to']}")

            elif action_type == 'create_package_structure':
                # Create __init__.py files for proper package structure
                package_name = action.get('base_package')
                if package_name and not self.dry_run:
                    package_dir = os.path.join(target_dir, package_name)
                    if os.path.exists(package_dir):
                        init_file = os.path.join(package_dir, '__init__.py')
                        if not os.path.exists(init_file):
                            Path(init_file).touch()
                            logger.info(f"Created {init_file}")

            elif action_type == 'create_src_layout':
                # Create src directory structure for modern Python packaging
                package_name = action.get('base_package')
                if package_name and not self.dry_run:
                    # Create src directory
                    src_dir = os.path.join(target_dir, 'src')
                    os.makedirs(src_dir, exist_ok=True)
                    logger.info(f"Created src directory: {src_dir}")

                    # Create package directory in src
                    package_dir = os.path.join(src_dir, package_name)
                    if not os.path.exists(package_dir):
                        os.makedirs(package_dir, exist_ok=True)
                        logger.info(f"Created package directory: {package_dir}")

                    # Determine if this is a plugin package
                    is_plugin = package_name.startswith('rompy_') and package_name != 'rompy_core'
                    plugin_name = package_name.replace('rompy_', '') if is_plugin else None

                    # Create modern __init__.py with version handling and plugin metadata
                    self._create_modern_init_py(package_dir, package_name, is_plugin, plugin_name)

            elif action_type == 'create_modern_setup':
                # Create modern setup files with src layout
                package_name = action.get('package_name')
                package_module = action.get('package_module')
                description = action.get('description')
                dependencies = action.get('dependencies', [])

                if package_name and package_module and not self.dry_run:
                    self._create_modern_setup_files(
                        target_dir, package_name, package_module,
                        description, dependencies
                    )

            elif action_type == 'create_plugin_docs':
                # Create plugin-specific documentation
                package_name = action.get('package_name')
                plugin_name = action.get('plugin_name')
                extends_core_docs = action.get('extends_core_docs', False)

                if package_name and plugin_name and not self.dry_run:
                    self._create_plugin_documentation(
                        target_dir, package_name, plugin_name, extends_core_docs
                    )

            elif action_type == 'update_docs_config':
                # Update documentation configuration
                package_name = action.get('package_name')
                is_core = action.get('is_core_package', False)
                plugin_discovery = action.get('plugin_discovery', False)

                if package_name and not self.dry_run:
                    self._update_docs_configuration(
                        target_dir, package_name, is_core, plugin_discovery
                    )

            elif action_type == 'create_notebooks_index':
                # Create notebooks index
                ecosystem_packages = action.get('ecosystem_packages', [])

                if ecosystem_packages and not self.dry_run:
                    self._create_notebooks_index(target_dir, ecosystem_packages)

            elif action_type == 'correct_imports':
                # Correct imports for the split repository
                package_type = action.get('package_type')
                target_package = action.get('target_package')

                if package_type and target_package and not self.dry_run:
                    self._correct_imports(target_dir, package_type, target_package)

            elif action_type == 'remove_files':
                # Remove unwanted files after filtering
                files_to_remove = action.get('files', [])
                patterns_to_remove = action.get('patterns', [])

                if (files_to_remove or patterns_to_remove) and not self.dry_run:
                    self._remove_files(target_dir, files_to_remove, patterns_to_remove)

            elif action_type == 'apply_cookiecutter_template':
                # Apply cookiecutter template for enhanced package structure
                template_repo = action.get('template_repo')
                template_context = action.get('template_context', {})
                merge_strategy = action.get('merge_strategy', 'overlay')

                if template_repo and not self.dry_run:
                    self._apply_cookiecutter_template(
                        target_dir, template_repo, template_context, merge_strategy
                    )

    def _correct_imports(self, target_dir: str, package_type: str, target_package: str):
        """
        Correct imports in Python files for the split repository.

        Args:
            target_dir: Target directory containing the split repository
            package_type: Type of package ('core', 'swan', 'schism', 'notebooks')
            target_package: Name of the target package (e.g., 'rompy_core', 'rompy_swan')
        """
        logger.info(f"Correcting imports for {package_type} package: {target_package}")

        # Define import correction patterns for each package type
        corrections = self._get_import_corrections(package_type, target_package)

        # Find all Python files (excluding notebooks for now)
        python_files = []
        for root, dirs, files in os.walk(target_dir):
            # Skip notebook directories
            if 'notebooks' in root:
                continue
            for file in files:
                if file.endswith('.py'):
                    python_files.append(os.path.join(root, file))

        logger.info(f"Found {len(python_files)} Python files to process")

        # Apply corrections to each file
        files_modified = 0
        for file_path in python_files:
            if self._apply_import_corrections(file_path, corrections):
                files_modified += 1

        logger.info(f"Modified imports in {files_modified} files")

    def _apply_cookiecutter_template(self, target_dir: str, template_repo: str,
                                   template_context: Dict[str, Any], merge_strategy: str):
        """
        Apply a cookiecutter template to enhance the split repository structure.

        Args:
            target_dir: Target directory containing the split repository
            template_repo: Path or URL to cookiecutter template
            template_context: Context variables for template rendering
            merge_strategy: How to handle conflicts ('overlay', 'replace', 'preserve')
        """
        logger.info(f"Applying cookiecutter template: {template_repo}")

        if not COOKIECUTTER_AVAILABLE:
            logger.error("Cookiecutter is not available. Install with: pip install cookiecutter")
            return

        # Create a temporary directory for cookiecutter output
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                # Apply cookiecutter template
                output_dir = cookiecutter(
                    template_repo,
                    extra_context=template_context,
                    output_dir=temp_dir,
                    no_input=True,
                    overwrite_if_exists=True
                )

                logger.info(f"Cookiecutter generated project at: {output_dir}")

                # Merge the generated files with the existing split repository
                self._merge_cookiecutter_output(target_dir, output_dir, merge_strategy)

                # Move dependency injection to after all merges
                self._inject_dependencies_from_template_context(target_dir, template_context)

            except Exception as e:
                logger.error(f"Failed to apply cookiecutter template: {e}")
                raise

    def _inject_dependencies_from_template_context(self, target_dir: str, template_context: dict):
        """
        Inject dependencies from template_context into pyproject.toml in target_dir.
        Handles dependencies, optional_dependencies_test, optional_dependencies_docs, and any other optional dependencies.
        If dependencies are empty and the package is 'rompy', extract them from the original monorepo's pyproject.toml.
        Only update the dependencies/optional-dependencies keys, preserving all other fields (e.g. entry-points).
        If [project.entry-points."rompy.source"] is missing or empty, copy from the monorepo if present.
        For [project.entry-points."rompy.config"] in the core package, ensure only the base entry remains.
        """
        import ast
        try:
            import tomli
            import tomli_w
        except ImportError:
            logger.warning("tomli/tomli_w not available, skipping pyproject.toml dependency injection")
            return

        pyproject_path = os.path.join(target_dir, 'pyproject.toml')
        logger.info(f"[inject_deps] Looking for pyproject.toml at: {pyproject_path}")
        if not os.path.exists(pyproject_path):
            logger.warning(f"[inject_deps] No pyproject.toml found in {target_dir}, skipping dependency injection")
            return

        # Parse dependency fields from template_context
        dependencies = []
        optional_deps = {}
        for key, value in template_context.items():
            if key == 'dependencies' and value:
                try:
                    dependencies = ast.literal_eval(value) if isinstance(value, str) else value
                    logger.info(f"[inject_deps] Parsed dependencies: {dependencies}")
                except Exception as e:
                    logger.warning(f"[inject_deps] Failed to parse dependencies: {e}")
            elif key.startswith('optional_dependencies_') and value:
                group = key[len('optional_dependencies_'):]
                try:
                    optional_deps[group] = ast.literal_eval(value) if isinstance(value, str) else value
                    logger.info(f"[inject_deps] Parsed optional deps for {group}: {optional_deps[group]}")
                except Exception as e:
                    logger.warning(f"[inject_deps] Failed to parse {key}: {e}")

        # If dependencies are empty and this is the rompy core package, extract from monorepo pyproject.toml
        package_name = template_context.get('package_name') or template_context.get('repo_name') or os.path.basename(target_dir)
        monorepo_entry_points = None
        if not dependencies and package_name == 'rompy':
            # Find the original monorepo pyproject.toml
            monorepo_pyproject = os.path.join(os.path.dirname(__file__), 'pyproject.toml')
            logger.info(f"[inject_deps] No dependencies in template_context for rompy, extracting from monorepo: {monorepo_pyproject}")
            if os.path.exists(monorepo_pyproject):
                with open(monorepo_pyproject, 'rb') as f:
                    monorepo_data = tomli.load(f)
                deps = monorepo_data.get('project', {}).get('dependencies', [])
                opt_deps = monorepo_data.get('project', {}).get('optional-dependencies', {})
                dependencies = deps
                optional_deps = opt_deps
                # Also get entry-points from monorepo
                monorepo_entry_points = monorepo_data.get('project', {}).get('entry-points', {})
                logger.info(f"[inject_deps] Extracted dependencies from monorepo: {dependencies}")
                logger.info(f"[inject_deps] Extracted optional dependencies from monorepo: {optional_deps}")
                logger.info(f"[inject_deps] Extracted entry-points from monorepo: {monorepo_entry_points}")
            else:
                logger.warning(f"[inject_deps] Could not find monorepo pyproject.toml at {monorepo_pyproject}")

        # Read pyproject.toml
        with open(pyproject_path, 'rb') as f:
            data = tomli.load(f)

        # Only update dependencies and optional-dependencies keys, preserve all other fields
        if 'project' not in data:
            data['project'] = {}
        if dependencies:
            data['project']['dependencies'] = dependencies
        # Only update optional-dependencies, preserve any other keys (like entry-points)
        if optional_deps:
            if 'optional-dependencies' not in data['project']:
                data['project']['optional-dependencies'] = {}
            for group, deps in optional_deps.items():
                data['project']['optional-dependencies'][group] = deps

        # If this is rompy, ensure entry-points.rompy.source is present and non-empty
        if package_name == 'rompy':
            # Check if entry-points.rompy.source is missing or empty
            entry_points = data['project'].get('entry-points', {})
            rompy_source = entry_points.get('rompy.source') if entry_points else None
            if (not rompy_source or (isinstance(rompy_source, dict) and not rompy_source)) and monorepo_entry_points:
                # Copy from monorepo if present
                if 'entry-points' not in data['project']:
                    data['project']['entry-points'] = {}
                if 'rompy.source' in monorepo_entry_points:
                    data['project']['entry-points']['rompy.source'] = monorepo_entry_points['rompy.source']
                    logger.info(f"[inject_deps] Injected rompy.source entry-points from monorepo into split package.")
            # Ensure only the correct rompy.config entry remains
            if 'rompy.config' in data['project'].get('entry-points', {}):
                data['project']['entry-points']['rompy.config'] = {
                    'base': 'rompy.core.config:BaseConfig'
                }
                logger.info(f"[inject_deps] Set rompy.config entry-point to only base=rompy.core.config:BaseConfig for core package.")

        # Write back
        with open(pyproject_path, 'wb') as f:
            tomli_w.dump(data, f)
        logger.info(f"[inject_deps] Injected dependencies into pyproject.toml in {target_dir}")
        # Log the resulting file for debug
        with open(pyproject_path, 'r') as f:
            logger.info(f"[inject_deps] Final pyproject.toml contents:\n{f.read()}")

    def _merge_cookiecutter_output(self, target_dir: str, cookiecutter_output: str,
                                 merge_strategy: str):
        """
        Merge cookiecutter output with the existing split repository.

        Args:
            target_dir: Target directory of the split repository
            cookiecutter_output: Directory containing cookiecutter output
            merge_strategy: Strategy for handling conflicts
        """
        logger.info(f"Merging cookiecutter output with merge strategy: {merge_strategy}")

        # Files to preserve from original (git history preservation is key)
        preserve_files = {'.git', '.gitignore', 'README.md', 'LICENSE', 'HISTORY.rst'}

        # Files to always take from cookiecutter (modern infrastructure only)
        cookiecutter_priority = {'pyproject.toml', 'setup.cfg', 'tox.ini', 'requirements_dev.txt',
                                'ruff.toml', '.editorconfig', 'Makefile', 'MANIFEST.in', '.travis.yml',
                                'AUTHORS.rst', 'CODE_OF_CONDUCT.rst', 'CONTRIBUTING.rst'}

        # Directories to completely exclude from cookiecutter (we manage these ourselves)
        exclude_cookiecutter_dirs = {'src', 'tests', 'docs', 'examples', 'notebooks'}

        # Always preserve existing Python files and other source code
        preserve_source_extensions = {'.py', '.rst', '.md', '.yml', '.yaml', '.json', '.txt', '.sh'}

        for root, dirs, files in os.walk(cookiecutter_output):
            # Calculate relative path from cookiecutter output
            rel_path = os.path.relpath(root, cookiecutter_output)

            # Skip the top-level directory itself
            if rel_path == '.':
                rel_path = ''

            # Check if this directory should be excluded from cookiecutter
            # Check both with and without trailing slash, and check if any path component matches
            path_parts = rel_path.split('/') if rel_path else []
            should_exclude_dir = any(
                rel_path.startswith(exclude_dir + '/') or
                rel_path == exclude_dir or
                exclude_dir in path_parts
                for exclude_dir in exclude_cookiecutter_dirs
            )

            if should_exclude_dir:
                logger.debug(f"Skipping cookiecutter directory: {rel_path}")
                continue

            # Create corresponding directory in target
            target_subdir = os.path.join(target_dir, rel_path) if rel_path else target_dir
            os.makedirs(target_subdir, exist_ok=True)

            # Copy files based on merge strategy
            for file in files:
                src_file = os.path.join(root, file)
                dst_file = os.path.join(target_subdir, file)

                should_copy = False

                if merge_strategy == 'replace':
                    # Replace everything except preserved files
                    should_copy = file not in preserve_files

                elif merge_strategy == 'overlay':
                    # Check if file has an extension that indicates source code
                    file_ext = os.path.splitext(file)[1]
                    is_source_file = file_ext in preserve_source_extensions

                    # Always take specific infrastructure files from cookiecutter
                    if file in cookiecutter_priority:
                        should_copy = True
                        logger.debug(f"Taking infrastructure file from cookiecutter: {file}")

                    # NEVER overwrite existing source files anywhere, especially .py files
                    elif (is_source_file and os.path.exists(dst_file)) or file_ext == '.py':
                        should_copy = False
                        logger.debug(f"Preserving existing source file: {rel_path}/{file}")

                    # Don't overwrite preserved files
                    elif file in preserve_files:
                        should_copy = False
                        logger.debug(f"Preserving protected file: {file}")

                    # Only add completely new files that don't exist
                    elif not os.path.exists(dst_file):
                        should_copy = True
                        logger.debug(f"Adding new file from cookiecutter: {rel_path}/{file}")

                    # Default: don't overwrite existing files
                    else:
                        should_copy = False
                        logger.debug(f"Preserving existing file: {rel_path}/{file}")

                elif merge_strategy == 'preserve':
                    # Only add files that don't exist
                    should_copy = not os.path.exists(dst_file)

                if should_copy:
                    try:
                        shutil.copy2(src_file, dst_file)
                        logger.debug(f"Copied: {file} -> {rel_path}")
                    except Exception as e:
                        logger.warning(f"Failed to copy {file}: {e}")

        logger.info("Cookiecutter template merge completed")

    def _get_import_corrections(self, package_type: str, target_package: str) -> list:
        """
        Get the import correction patterns for a specific package type.

        Args:
            package_type: Type of package ('core', 'swan', 'schism', 'notebooks')
            target_package: Name of the target package

        Returns:
            List of (pattern, replacement) tuples for import corrections
        """
        corrections = []

        if package_type == 'core':
            # For rompy-core, convert absolute rompy imports to relative imports where appropriate
            # But be careful not to break imports that should remain absolute
            corrections.extend([
                # Convert internal imports to relative
                (r'^from rompy\.core', f'from {target_package}.core'),
                (r'^from rompy\.([^.\s]+)', f'from {target_package}.\\1'),
                (r'^import rompy\.core', f'import {target_package}.core'),
                (r'^import rompy\.([^.\s]+)', f'import {target_package}.\\1'),
                # Convert simple rompy import to target package
                (r'^import rompy$', f'import {target_package}'),
                (r'^from rompy import', f'from {target_package} import'),
            ])

        elif package_type == 'swan':
            # For rompy-swan, convert swan-specific imports and other rompy imports
            corrections.extend([
                # Convert swan-specific imports
                (r'^from rompy\.swan', f'from {target_package}'),
                (r'^import rompy\.swan', f'import {target_package}'),
                # Convert other rompy imports to rompy-core
                (r'^from rompy\.core', 'from rompy'),
                (r'^from rompy\.([^.\s]+)', 'from rompy.\\1'),
                (r'^import rompy\.core', 'import rompy'),
                (r'^import rompy\.([^.\s]+)', 'import rompy.\\1'),
                # Convert simple rompy import to rompy-core (but be careful about context)
                (r'^import rompy$', 'import rompy'),
                (r'^from rompy import', 'from rompy import'),
            ])

        elif package_type == 'schism':
            # For rompy-schism, convert schism-specific imports and other rompy imports
            corrections.extend([
                # Convert schism-specific imports
                (r'^from rompy\.schism', f'from {target_package}'),
                (r'^import rompy\.schism', f'import {target_package}'),
                # Convert other rompy imports to rompy-core
                (r'^from rompy\.core', 'from rompy'),
                (r'^from rompy\.([^.\s]+)', 'from rompy.\\1'),
                (r'^import rompy\.core', 'import rompy'),
                (r'^import rompy\.([^.\s]+)', 'import rompy.\\1'),
                # Convert simple rompy import to rompy-core
                (r'^import rompy$', 'import rompy'),
                (r'^from rompy import', 'from rompy import'),
            ])

        elif package_type == 'notebooks':
            # For notebooks, we'll handle these later as requested
            pass

        return corrections

    def _apply_import_corrections(self, file_path: str, corrections: list) -> bool:
        """
        Apply import corrections to a single Python file.

        Args:
            file_path: Path to the Python file
            corrections: List of (pattern, replacement) tuples

        Returns:
            True if the file was modified, False otherwise
        """
        import re

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            original_content = content
            lines = content.splitlines()
            modified_lines = []

            for line in lines:
                modified_line = line
                for pattern, replacement in corrections:
                    modified_line = re.sub(pattern, replacement, modified_line)
                modified_lines.append(modified_line)

            modified_content = '\n'.join(modified_lines)

            # Only write if content changed
            if modified_content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(modified_content)
                logger.debug(f"Modified imports in: {file_path}")
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            return False

    def _remove_files(self, target_dir: str, files_to_remove: list, patterns_to_remove: list):
        """
        Remove unwanted files and directories after git filtering.

        Args:
            target_dir: Target directory containing the split repository
            files_to_remove: List of specific files/directories to remove
            patterns_to_remove: List of glob patterns to match for removal
        """
        import glob
        import os

        logger.info(f"Removing unwanted files from {target_dir}")

        removed_count = 0

        # Remove specific files/directories
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

        # Remove files matching patterns
        for pattern in patterns_to_remove:
            pattern_path = os.path.join(target_dir, pattern)
            matching_files = glob.glob(pattern_path, recursive=True)
            for file_path in matching_files:
                try:
                    rel_path = os.path.relpath(file_path, target_dir)
                    if os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        logger.debug(f"Removed directory (pattern): {rel_path}")
                    else:
                        os.remove(file_path)
                        logger.debug(f"Removed file (pattern): {rel_path}")
                    removed_count += 1
                except Exception as e:
                    logger.warning(f"Failed to remove {rel_path}: {e}")

        logger.info(f"Removed {removed_count} files/directories")

    def _create_modern_init_py(self, package_dir: str, package_name: str, is_plugin: bool = False, plugin_name: str = None):
        """Create a modern __init__.py file with version handling and plugin metadata."""
        init_file = os.path.join(package_dir, '__init__.py')

        # Create modern __init__.py content
        if is_plugin and plugin_name:
            init_content = f'''"""
{package_name.replace('_', ' ').title()}

{plugin_name.upper()} plugin for rompy ocean wave modeling framework.
"""

try:
    from ._version import __version__
except ImportError:
    # Package is not installed, use a default version
    __version__ = "0.0.0+unknown"

# Plugin metadata for discovery
__plugin_name__ = "{plugin_name}"
__description__ = "{plugin_name.upper()} plugin for rompy ocean wave modeling framework"
__docs_url__ = "https://{package_name}.readthedocs.io/"

# Plugin class (to be implemented)
# class {plugin_name.title()}Plugin:
#     """Plugin class for {plugin_name.upper()} integration."""
#     pass

__all__ = ["__version__", "__plugin_name__", "__description__", "__docs_url__"]
'''
        else:
            init_content = f'''"""
{package_name.replace('_', ' ').title()}

Core rompy library for ocean wave modeling with plugin system.
"""

try:
    from ._version import __version__
except ImportError:
    # Package is not installed, use a default version
    __version__ = "0.0.0+unknown"

def discover_plugins():
    """Discover installed rompy plugins."""
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
'''

        with open(init_file, 'w') as f:
            f.write(init_content)

        logger.info(f"Created modern __init__.py: {init_file}")

    def _create_modern_setup_files(self, target_dir: str, package_name: str,
                                  package_module: str, description: str,
                                  dependencies: List[str]):
        """Create modern setup files using templates."""
        if MODERN_TEMPLATES_AVAILABLE:
            try:
                repo_name = package_name  # Assume repo name matches package name
                create_modern_setup_files(
                    target_dir, package_name, package_module,
                    description, repo_name, dependencies
                )
                logger.info(f"Created modern setup files for {package_name}")
            except Exception as e:
                logger.error(f"Failed to create modern setup files: {e}")
        else:
            logger.warning("Modern templates not available, using basic setup files")
            self._create_basic_modern_files(target_dir, package_name, package_module,
                                           description, dependencies)

    def _create_basic_modern_files(self, target_dir: str, package_name: str,
                                  package_module: str, description: str,
                                  dependencies: List[str]):
        """Create basic modern setup files without templates."""
        # Create a basic modern pyproject.toml
        pyproject_content = f'''[build-system]
requires = ["setuptools>=64", "setuptools_scm>=8"]
build-backend = "setuptools.build_meta"

[project]
name = "{package_name}"
description = "{description}"
readme = "README.md"
license = {{file = "LICENSE"}}
authors = [
    {{name = "Rompy Contributors"}},
]
requires-python = ">=3.8"
dependencies = [
{chr(10).join([f'    "{dep}",' for dep in dependencies])}
]
dynamic = ["version"]

[tool.setuptools]
packages = {{find = {{where = ["src"]}}}}

[tool.setuptools_scm]
write_to = "src/{package_module}/_version.py"
'''

        pyproject_path = os.path.join(target_dir, 'pyproject.toml')
        with open(pyproject_path, 'w') as f:
            f.write(pyproject_content)

        logger.info(f"Created basic modern pyproject.toml for {package_name}")

    def _create_plugin_documentation(self, target_dir: str, package_name: str,
                                   plugin_name: str, extends_core_docs: bool):
        """Create plugin-specific documentation configuration."""
        docs_dir = os.path.join(target_dir, 'docs', 'source')
        os.makedirs(docs_dir, exist_ok=True)

        # Create plugin-specific conf.py
        conf_template = self.config.get('templates', {}).get('plugin_docs_conf', '')
        if conf_template:
            conf_content = conf_template.format(
                package_name=package_name,
                plugin_name=plugin_name
            )

            conf_path = os.path.join(docs_dir, 'conf.py')
            with open(conf_path, 'w') as f:
                f.write(conf_content)

            logger.info(f"Created plugin documentation config for {package_name}")

            # Create plugin-specific index.rst
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

            index_path = os.path.join(docs_dir, 'index.rst')
            with open(index_path, 'w') as f:
                f.write(index_content)

            logger.info(f"Created plugin documentation index for {package_name}")

    def _update_docs_configuration(self, target_dir: str, package_name: str,
                                 is_core: bool, plugin_discovery: bool):
        """Update documentation configuration for core package."""
        if is_core and plugin_discovery:
            docs_dir = os.path.join(target_dir, 'docs', 'source')
            os.makedirs(docs_dir, exist_ok=True)

            # Create base configuration that plugins can extend
            conf_base_template = self.config.get('templates', {}).get('core_docs_conf', '')
            if conf_base_template:
                conf_base_path = os.path.join(target_dir, 'src', package_name.replace('-', '_'), 'docs', 'conf_base.py')
                os.makedirs(os.path.dirname(conf_base_path), exist_ok=True)

                with open(conf_base_path, 'w') as f:
                    f.write(conf_base_template)

                # Also create the main docs conf.py
                conf_path = os.path.join(docs_dir, 'conf.py')
                with open(conf_path, 'w') as f:
                    f.write(conf_base_template)

                logger.info(f"Created core documentation configuration for {package_name}")

    def _create_notebooks_index(self, target_dir: str, ecosystem_packages: List[str]):
        """Create an index for the notebooks repository."""
        # Create a simple index.md for the notebooks
        index_content = f"""
# Rompy Ecosystem Examples

This repository contains examples and tutorials for the rompy ecosystem.

## Available Examples

The examples are organized by functionality and required packages:

### Core Examples
Examples using only rompy-core functionality.

### Plugin Examples
Examples demonstrating specific plugins:

"""

        for package in ecosystem_packages:
            if package != 'rompy-core':
                plugin_name = package.replace('rompy-', '').upper()
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

        index_path = os.path.join(target_dir, 'README.md')
        with open(index_path, 'w') as f:
            f.write(index_content)

        logger.info("Created notebooks index and README")

    def _cleanup_empty_directories(self, target_dir: str):
        """Remove empty directories after filtering, but exclude .git directories."""
        if self.dry_run:
            return

        for root, dirs, files in os.walk(target_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)

                # Skip any directories inside .git to avoid corrupting the git repository
                if '.git' in dir_path:
                    continue

                try:
                    if not os.listdir(dir_path):  # Directory is empty
                        os.rmdir(dir_path)
                        logger.info(f"Removed empty directory: {dir_path}")
                except OSError:
                    pass  # Directory not empty or other error

    def split_repository(self, repo_name: str, repo_config: Dict[str, Any]):
        """
        Split a single repository according to its configuration.

        Args:
            repo_name: Name of the target repository
            repo_config: Configuration for this repository
        """
        logger.info(f"Starting split for repository: {repo_name}")
        logger.info(f"Description: {repo_config.get('description', 'No description')}")

        # Create target directory
        target_dir = self._create_target_directory(repo_name)

        try:
            # Clone source repository
            self._clone_source_repo(target_dir)

            # Filter repository to keep only specified paths
            paths = repo_config.get('paths', [])
            if paths:
                self._filter_repository(target_dir, paths)

            # Perform post-split actions
            post_actions = repo_config.get('post_split_actions', [])
            if post_actions:
                self._perform_post_split_actions(target_dir, post_actions)

            # Clean up empty directories
            self._cleanup_empty_directories(target_dir)

            # Initialize git repository properly
            if not self.dry_run:
                self._run_command(['git', 'gc', '--aggressive'], cwd=target_dir)
                self._run_command(['git', 'repack', '-ad'], cwd=target_dir)

            logger.info(f"Successfully created repository: {repo_name}")

        except Exception as e:
            logger.error(f"Failed to split repository {repo_name}: {e}")
            if not self.dry_run and self.config.get('cleanup_after_split', False):
                logger.info(f"Cleaning up failed split: {target_dir}")
                shutil.rmtree(target_dir, ignore_errors=True)
            raise

    def run(self):
        """Run the repository splitting process."""
        logger.info("Starting repository splitting process")

        # Check prerequisites
        self._check_prerequisites()

        # Create base target directory
        if not self.dry_run:
            os.makedirs(self.target_base_dir, exist_ok=True)

        # Process each repository in the configuration
        repositories = self.config.get('repositories', {})

        for repo_name, repo_config in repositories.items():
            try:
                self.split_repository(repo_name, repo_config)
            except Exception as e:
                logger.error(f"Failed to process {repo_name}: {e}")
                if not self.config.get('continue_on_error', False):
                    sys.exit(1)

        logger.info("Repository splitting completed successfully")

        # Print summary
        self._print_summary(repositories)

    def _print_summary(self, repositories: Dict[str, Any]):
        """Print a summary of the splitting results."""
        print("\n" + "="*60)
        print("REPOSITORY SPLITTING SUMMARY")
        print("="*60)

        for repo_name, repo_config in repositories.items():
            target_dir = os.path.join(self.target_base_dir, repo_name)
            print(f"\n📁 {repo_name}")
            print(f"   Description: {repo_config.get('description', 'N/A')}")
            print(f"   Location: {target_dir}")

            if not self.dry_run and os.path.exists(target_dir):
                # Get some basic stats
                try:
                    result = self._run_command(['git', 'rev-list', '--count', 'HEAD'], cwd=target_dir, check=False)
                    commit_count = result.stdout.strip() if result.returncode == 0 else "Unknown"
                    print(f"   Commits: {commit_count}")

                    result = self._run_command(['git', 'branch', '-a'], cwd=target_dir, check=False)
                    branch_count = len([l for l in result.stdout.split('\n') if l.strip()]) if result.returncode == 0 else 0
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Split a monorepo into multiple repositories while preserving history"
    )
    parser.add_argument(
        '--config', '-c',
        default='repo_split_config.yaml',
        help='Path to the configuration file (default: repo_split_config.yaml)'
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show what would be done without actually doing it'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if args.dry_run:
        logger.info("DRY RUN MODE: No actual changes will be made")

    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        splitter = RepositorySplitter(args.config, args.dry_run)
        splitter.run()
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
