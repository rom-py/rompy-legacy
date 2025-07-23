#!/usr/bin/env python3
"""
Configuration Validation Script

This script validates the repository splitting configuration file to ensure
all paths exist and the configuration is valid before running the split.

Usage:
    python validate_config.py [--config CONFIG_FILE]
"""

import argparse
import os
import sys
import yaml
from pathlib import Path
from typing import Dict, List, Any, Set
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ConfigValidator:
    """
    Validates repository splitting configuration.
    """

    def __init__(self, config_path: str):
        """
        Initialize the validator.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.source_repo = self.config.get('source_repo', '.')
        self.errors = []
        self.warnings = []

    def _load_config(self) -> Dict[str, Any]:
        """Load the configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            sys.exit(1)

    def _add_error(self, message: str):
        """Add an error message."""
        self.errors.append(message)
        logger.error(message)

    def _add_warning(self, message: str):
        """Add a warning message."""
        self.warnings.append(message)
        logger.warning(message)

    def _check_path_exists(self, path: str) -> bool:
        """Check if a path exists in the source repository."""
        # Remove leading ! for exclude patterns
        clean_path = path[1:] if path.startswith('!') else path

        # Handle glob patterns by checking if the base directory exists
        if '*' in clean_path:
            # For glob patterns, check if the base directory exists
            base_path = clean_path.split('*')[0].rstrip('/')
            if base_path:
                full_path = os.path.join(self.source_repo, base_path)
                return os.path.exists(full_path)
            return True  # Root level glob patterns are assumed valid

        full_path = os.path.join(self.source_repo, clean_path)
        return os.path.exists(full_path)

    def validate_basic_structure(self) -> bool:
        """Validate basic configuration structure."""
        valid = True

        # Check required top-level keys
        required_keys = ['source_repo', 'target_base_dir', 'repositories']
        for key in required_keys:
            if key not in self.config:
                self._add_error(f"Missing required configuration key: {key}")
                valid = False

        # Check if source repository exists
        if not os.path.exists(self.source_repo):
            self._add_error(f"Source repository does not exist: {self.source_repo}")
            valid = False
        elif not os.path.isdir(os.path.join(self.source_repo, '.git')):
            self._add_error(f"Source repository is not a git repository: {self.source_repo}")
            valid = False

        # Check if repositories section is a dictionary
        if 'repositories' in self.config:
            if not isinstance(self.config['repositories'], dict):
                self._add_error("'repositories' must be a dictionary")
                valid = False
            elif not self.config['repositories']:
                self._add_warning("No repositories defined in configuration")

        return valid

    def validate_repository_config(self, repo_name: str, repo_config: Dict[str, Any]) -> bool:
        """
        Validate configuration for a single repository.

        Args:
            repo_name: Name of the repository
            repo_config: Configuration dictionary for the repository

        Returns:
            True if valid, False otherwise
        """
        valid = True

        # Check required fields
        if 'paths' not in repo_config:
            self._add_error(f"Repository '{repo_name}' missing 'paths' configuration")
            valid = False
            return valid

        # Validate paths
        paths = repo_config['paths']
        if not isinstance(paths, list):
            self._add_error(f"Repository '{repo_name}' paths must be a list")
            valid = False
            return valid

        # Check if paths exist
        missing_paths = []
        for path in paths:
            if not self._check_path_exists(path):
                missing_paths.append(path)

        if missing_paths:
            self._add_warning(f"Repository '{repo_name}' has non-existent paths: {missing_paths}")

        # Validate post-split actions
        if 'post_split_actions' in repo_config:
            valid &= self._validate_post_split_actions(repo_name, repo_config['post_split_actions'])

        return valid

    def _validate_post_split_actions(self, repo_name: str, actions: List[Dict[str, Any]]) -> bool:
        """Validate post-split actions for a repository."""
        valid = True

        if not isinstance(actions, list):
            self._add_error(f"Repository '{repo_name}' post_split_actions must be a list")
            return False

        valid_actions = {
            'move_files', 'create_readme', 'update_setup', 'rename',
            'create_package_structure', 'create_src_layout', 'create_modern_setup',
            'merge_directory_contents', 'update_docs_config', 'create_plugin_docs',
            'create_notebooks_index', 'correct_imports', 'remove_files',
            'apply_cookiecutter_template'
        }

        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                self._add_error(f"Repository '{repo_name}' action {i} must be a dictionary")
                valid = False
                continue

            action_type = action.get('action')
            if not action_type:
                self._add_error(f"Repository '{repo_name}' action {i} missing 'action' field")
                valid = False
                continue

            if action_type not in valid_actions:
                self._add_error(f"Repository '{repo_name}' action {i} has invalid action type: {action_type}")
                valid = False
                continue

            # Validate specific action requirements
            if action_type == 'move_files':
                if 'moves' not in action:
                    self._add_error(f"Repository '{repo_name}' move_files action missing 'moves' field")
                    valid = False
                elif not isinstance(action['moves'], list):
                    self._add_error(f"Repository '{repo_name}' move_files 'moves' must be a list")
                    valid = False

            elif action_type == 'create_readme':
                if 'template' not in action:
                    self._add_error(f"Repository '{repo_name}' create_readme action missing 'template' field")
                    valid = False
                elif action['template'] not in self.config.get('templates', {}):
                    self._add_error(f"Repository '{repo_name}' references non-existent template: {action['template']}")
                    valid = False

            elif action_type == 'apply_cookiecutter_template':
                if 'template_repo' not in action:
                    self._add_error(f"Repository '{repo_name}' apply_cookiecutter_template action missing 'template_repo' field")
                    valid = False
                else:
                    template_repo = action['template_repo']
                    # Check if template_repo is a local path and exists
                    if not template_repo.startswith(('http://', 'https://', 'git@', 'git+')):
                        # It's a local path, check if it exists
                        template_path = os.path.abspath(template_repo)
                        if not os.path.exists(template_path):
                            self._add_error(f"Repository '{repo_name}' cookiecutter template path does not exist: {template_path}")
                            valid = False
                        elif not os.path.isdir(template_path):
                            self._add_error(f"Repository '{repo_name}' cookiecutter template path is not a directory: {template_path}")
                            valid = False
                        else:
                            # Check for cookiecutter.json
                            cookiecutter_json = os.path.join(template_path, 'cookiecutter.json')
                            if not os.path.exists(cookiecutter_json):
                                self._add_warning(f"Repository '{repo_name}' cookiecutter template missing cookiecutter.json: {cookiecutter_json}")

                # Validate merge_strategy if provided
                merge_strategy = action.get('merge_strategy', 'overlay')
                valid_strategies = {'overlay', 'replace', 'preserve'}
                if merge_strategy not in valid_strategies:
                    self._add_error(f"Repository '{repo_name}' apply_cookiecutter_template has invalid merge_strategy: {merge_strategy}. Must be one of: {valid_strategies}")
                    valid = False

                # Validate template_context if provided
                if 'template_context' in action and not isinstance(action['template_context'], dict):
                    self._add_error(f"Repository '{repo_name}' apply_cookiecutter_template 'template_context' must be a dictionary")
                    valid = False

            elif action_type == 'update_setup':
                required_fields = ['package_name', 'description']
                for field in required_fields:
                    if field not in action:
                        self._add_error(f"Repository '{repo_name}' update_setup action missing '{field}' field")
                        valid = False

            elif action_type == 'create_src_layout':
                if 'base_package' not in action:
                    self._add_error(f"Repository '{repo_name}' create_src_layout action missing 'base_package' field")
                    valid = False

            elif action_type == 'create_modern_setup':
                required_fields = ['package_name', 'package_module', 'description']
                for field in required_fields:
                    if field not in action:
                        self._add_error(f"Repository '{repo_name}' create_modern_setup action missing '{field}' field")
                        valid = False

            elif action_type == 'merge_directory_contents':
                if 'merges' not in action:
                    self._add_error(f"Repository '{repo_name}' merge_directory_contents action missing 'merges' field")
                    valid = False
                elif not isinstance(action['merges'], list):
                    self._add_error(f"Repository '{repo_name}' merge_directory_contents 'merges' must be a list")
                    valid = False

            elif action_type == 'create_plugin_docs':
                required_fields = ['package_name', 'plugin_name']
                for field in required_fields:
                    if field not in action:
                        self._add_error(f"Repository '{repo_name}' create_plugin_docs action missing '{field}' field")
                        valid = False

            elif action_type == 'update_docs_config':
                if 'package_name' not in action:
                    self._add_error(f"Repository '{repo_name}' update_docs_config action missing 'package_name' field")
                    valid = False

            elif action_type == 'create_notebooks_index':
                if 'ecosystem_packages' not in action:
                    self._add_error(f"Repository '{repo_name}' create_notebooks_index action missing 'ecosystem_packages' field")
                    valid = False
                elif not isinstance(action['ecosystem_packages'], list):
                    self._add_error(f"Repository '{repo_name}' create_notebooks_index 'ecosystem_packages' must be a list")
                    valid = False

            elif action_type == 'correct_imports':
                required_fields = ['package_type', 'target_package']
                for field in required_fields:
                    if field not in action:
                        self._add_error(f"Repository '{repo_name}' correct_imports action missing '{field}' field")
                        valid = False

                # Validate package_type values
                if 'package_type' in action:
                    valid_package_types = {'core', 'swan', 'schism', 'notebooks'}
                    if action['package_type'] not in valid_package_types:
                        self._add_error(f"Repository '{repo_name}' correct_imports has invalid package_type: {action['package_type']}. Must be one of: {valid_package_types}")
                        valid = False

            elif action_type == 'remove_files':
                # Validate that at least one of files or patterns is provided
                if 'files' not in action and 'patterns' not in action:
                    self._add_error(f"Repository '{repo_name}' remove_files action must have either 'files' or 'patterns' field")
                    valid = False

                # Validate that files and patterns are lists if present
                if 'files' in action and not isinstance(action['files'], list):
                    self._add_error(f"Repository '{repo_name}' remove_files 'files' must be a list")
                    valid = False

                if 'patterns' in action and not isinstance(action['patterns'], list):
                    self._add_error(f"Repository '{repo_name}' remove_files 'patterns' must be a list")
                    valid = False

        return valid

    def validate_path_conflicts(self) -> bool:
        """Check for path conflicts between repositories."""
        valid = True
        all_paths = {}  # path -> list of repos that include it

        for repo_name, repo_config in self.config.get('repositories', {}).items():
            paths = repo_config.get('paths', [])
            for path in paths:
                # Skip exclude patterns for conflict checking
                if path.startswith('!'):
                    continue

                if path not in all_paths:
                    all_paths[path] = []
                all_paths[path].append(repo_name)

        # Check for conflicts
        for path, repos in all_paths.items():
            if len(repos) > 1:
                self._add_warning(f"Path '{path}' is included in multiple repositories: {repos}")

        return valid

    def validate_dependencies(self) -> bool:
        """Validate that repository dependencies are valid."""
        valid = True
        repo_names = set(self.config.get('repositories', {}).keys())

        for repo_name, repo_config in self.config.get('repositories', {}).items():
            post_actions = repo_config.get('post_split_actions', [])

            for action in post_actions:
                if action.get('action') == 'update_setup':
                    dependencies = action.get('dependencies', [])
                    for dep in dependencies:
                        # Check if dependency is another repo in this split
                        dep_repo_name = dep.replace('-', '_')  # Convert package name to repo name
                        if dep_repo_name in repo_names and dep_repo_name == repo_name:
                            self._add_error(f"Repository '{repo_name}' cannot depend on itself")
                            valid = False

        return valid

    def check_git_status(self) -> bool:
        """Check if the source repository is in a clean state."""
        try:
            import subprocess

            # Check if there are uncommitted changes
            result = subprocess.run(
                ['git', 'status', '--porcelain'],
                cwd=self.source_repo,
                capture_output=True,
                text=True,
                check=True
            )

            if result.stdout.strip():
                self._add_warning("Source repository has uncommitted changes. Consider committing before splitting.")

            # Check current branch
            result = subprocess.run(
                ['git', 'branch', '--show-current'],
                cwd=self.source_repo,
                capture_output=True,
                text=True,
                check=True
            )

            current_branch = result.stdout.strip()
            if current_branch != 'main' and current_branch != 'master':
                self._add_warning(f"Currently on branch '{current_branch}'. Consider switching to main/master branch.")

            return True

        except (subprocess.CalledProcessError, FileNotFoundError):
            self._add_warning("Could not check git status. Make sure git is available.")
            return True

    def validate(self) -> bool:
        """
        Run all validation checks.

        Returns:
            True if configuration is valid, False otherwise
        """
        logger.info("Starting configuration validation")

        valid = True

        # Basic structure validation
        valid &= self.validate_basic_structure()

        if not valid:
            logger.error("Basic structure validation failed, skipping further checks")
            return False

        # Validate each repository
        for repo_name, repo_config in self.config.get('repositories', {}).items():
            valid &= self.validate_repository_config(repo_name, repo_config)

        # Check for path conflicts
        valid &= self.validate_path_conflicts()

        # Validate dependencies
        valid &= self.validate_dependencies()

        # Check git status
        self.check_git_status()

        return valid

    def print_summary(self):
        """Print validation summary."""
        print("\n" + "="*60)
        print("CONFIGURATION VALIDATION SUMMARY")
        print("="*60)

        if self.errors:
            print(f"\n❌ ERRORS ({len(self.errors)}):")
            for i, error in enumerate(self.errors, 1):
                print(f"   {i}. {error}")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings, 1):
                print(f"   {i}. {warning}")

        if not self.errors and not self.warnings:
            print("\n✅ Configuration is valid with no issues detected!")
        elif not self.errors:
            print(f"\n✅ Configuration is valid with {len(self.warnings)} warnings.")
        else:
            print(f"\n❌ Configuration has {len(self.errors)} errors that must be fixed.")

        # Print repository summary
        repositories = self.config.get('repositories', {})
        if repositories:
            print(f"\n📊 REPOSITORY SUMMARY:")
            for repo_name, repo_config in repositories.items():
                path_count = len(repo_config.get('paths', []))
                action_count = len(repo_config.get('post_split_actions', []))
                print(f"   • {repo_name}: {path_count} paths, {action_count} post-split actions")

        print("\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Validate repository splitting configuration"
    )
    parser.add_argument(
        '--config', '-c',
        default='repo_split_config.yaml',
        help='Path to the configuration file (default: repo_split_config.yaml)'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        sys.exit(1)

    try:
        validator = ConfigValidator(args.config)
        is_valid = validator.validate()
        validator.print_summary()

        if not is_valid:
            sys.exit(1)
        else:
            logger.info("Configuration validation passed!")

    except KeyboardInterrupt:
        logger.info("Validation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
