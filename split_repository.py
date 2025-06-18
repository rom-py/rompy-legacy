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
        filters = []

        for path in paths:
            if path.startswith('!'):
                # Exclude path
                exclude_path = path[1:]  # Remove the !
                filters.extend(['--path-glob', exclude_path, '--invert-paths'])
            else:
                # Include path
                filters.extend(['--path-glob', path])

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

        # Process paths to build include/exclude filters
        include_paths = []
        exclude_paths = []

        for path in paths:
            if path.startswith('!'):
                exclude_paths.append(path[1:])  # Remove the !
            else:
                include_paths.append(path)

        # Add include paths
        for path in include_paths:
            cmd.extend(['--path', path])

        # Add exclude paths
        for path in exclude_paths:
            cmd.extend(['--path', path, '--invert-paths'])

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
                           description: str, dependencies: List[str] = None, src_layout: bool = False):
        """Update setup.cfg and pyproject.toml for the new package."""
        setup_cfg_path = os.path.join(target_dir, 'setup.cfg')
        pyproject_path = os.path.join(target_dir, 'pyproject.toml')

        if not self.dry_run:
            # Update setup.cfg if it exists
            if os.path.exists(setup_cfg_path):
                self._update_setup_cfg(setup_cfg_path, package_name, description, src_layout)

            # Update pyproject.toml if it exists
            if os.path.exists(pyproject_path):
                self._update_pyproject_toml(pyproject_path, package_name, description, dependencies or [], src_layout)

    def _update_setup_cfg(self, setup_cfg_path: str, package_name: str, description: str, src_layout: bool = False):
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

        with open(setup_cfg_path, 'w') as f:
            f.write('\n'.join(new_lines))

        logger.info(f"Updated setup.cfg for {package_name} with {'src layout' if src_layout else 'standard layout'}")

    def _update_pyproject_toml(self, pyproject_path: str, package_name: str,
                              description: str, dependencies: List[str], src_layout: bool = False):
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
                    action.get('src_layout', False)
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

                    # Create modern __init__.py with version handling
                    self._create_modern_init_py(package_dir, package_name)

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

    def _create_modern_init_py(self, package_dir: str, package_name: str):
        """Create a modern __init__.py file with version handling."""
        init_file = os.path.join(package_dir, '__init__.py')

        # Create modern __init__.py content
        init_content = f'''"""
{package_name.replace('_', ' ').title()}

Modern Python package for rompy ecosystem.
"""

try:
    from ._version import __version__
except ImportError:
    # Package is not installed, use a default version
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
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

    def _cleanup_empty_directories(self, target_dir: str):
        """Remove empty directories after filtering."""
        if self.dry_run:
            return

        for root, dirs, files in os.walk(target_dir, topdown=False):
            for dir_name in dirs:
                dir_path = os.path.join(root, dir_name)
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
