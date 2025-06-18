# Repository Splitting Guide

This document provides a comprehensive guide for splitting the rompy monorepo into multiple focused repositories while preserving git history, branches, and tags.

## Overview

The rompy repository will be split into the following packages:

1. **rompy-core** - Core functionality and base classes
2. **rompy-swan** - SWAN wave model integration
3. **rompy-schism** - SCHISM model integration
4. **rompy-notebooks** - Example notebooks and tutorials
5. **rompy-docs** - Documentation for the entire ecosystem

## Prerequisites

Before running the splitting process, ensure you have the following installed:

```bash
# Install required Python packages
pip install -r split_requirements.txt

# Or install individually
pip install PyYAML>=6.0 git-filter-repo>=2.38.0 tomli>=2.0.0 tomli-w>=1.0.0
```

### System Requirements

- Python 3.7+
- Git 2.20+
- git-filter-repo (installed via pip)
- Sufficient disk space (the process creates temporary clones)

## Quick Start

1. **Validate the configuration:**
   ```bash
   python validate_config.py --config repo_split_config.yaml
   ```

2. **Run a dry-run to see what would happen:**
   ```bash
   python split_repository.py --config repo_split_config.yaml --dry-run
   ```

3. **Execute the actual split:**
   ```bash
   python split_repository.py --config repo_split_config.yaml
   ```

4. **Review the results:**
   ```bash
   ls -la ../split-repos/
   ```

## Configuration

The splitting process is controlled by `repo_split_config.yaml`. Key sections include:

### Repository Definitions

Each repository is defined with:
- **paths**: List of paths to include (prefix with `!` to exclude)
- **post_split_actions**: Actions to perform after filtering
- **description**: Human-readable description

### Path Patterns

- `rompy/` - Include entire rompy directory
- `!rompy/swan/` - Exclude swan subdirectory
- `tests/` - Include tests directory
- `*.py` - Include all Python files (glob patterns supported)

### Post-Split Actions

Available actions:
- `move_files` - Move files to new locations
- `rename` - Rename directories
- `update_setup` - Update package configuration
- `create_readme` - Generate README from template
- `create_package_structure` - Create proper Python package structure
- `create_src_layout` - Create modern src/ directory structure
- `create_modern_setup` - Generate complete modern packaging setup

## Process Details

### What the Script Does

1. **Clones** the source repository for each target repository
2. **Filters** using git-filter-repo to keep only specified paths
3. **Preserves** complete git history for included files
4. **Maintains** all branches and tags
5. **Restructures** files according to post-split actions
6. **Creates modern src/ layout** following Python packaging best practices
7. **Updates** package configuration files (pyproject.toml, setup.cfg)
8. **Generates** modern development tools (tox.ini, pre-commit, GitHub Actions)

### Directory Structure

After splitting with modern src/ layout, you'll have:
```
../split-repos/
├── rompy-core/
│   ├── src/
│   │   └── rompy_core/      # Moved from rompy/ to src/
│   │       ├── __init__.py  # Modern version handling
│   │       └── ...
│   ├── tests/               # Core tests only
│   ├── docs/                # Core documentation
│   ├── pyproject.toml       # Modern packaging configuration
│   ├── setup.cfg            # Compatibility configuration
│   ├── tox.ini              # Multi-version testing
│   ├── .pre-commit-config.yaml
│   └── .github/workflows/   # CI/CD workflows
├── rompy-swan/
│   ├── src/
│   │   └── rompy_swan/      # Moved from rompy/swan/ to src/
│   │       ├── __init__.py
│   │       └── ...
│   ├── tests/               # Swan tests only
│   ├── docs/                # Swan documentation
│   ├── docker/              # Docker configurations
│   └── pyproject.toml       # Modern packaging
├── rompy-schism/
│   ├── src/
│   │   └── rompy_schism/    # Moved from rompy/schism/ to src/
│   │       ├── __init__.py
│   │       └── ...
│   ├── tests/               # Schism tests only
│   ├── docs/                # Schism documentation
│   └── pyproject.toml       # Modern packaging
├── rompy-notebooks/
│   ├── notebooks/           # All notebooks
│   └── README.md            # Generated README
└── rompy-docs/
    ├── docs/                # Complete documentation
    └── README.md            # Generated README
```

## Customization

### Modifying the Split

To customize the split, edit `repo_split_config.yaml`:

1. **Add/remove paths** in the `paths` sections
2. **Modify post-split actions** to change restructuring
3. **Update package names** and descriptions
4. **Add new repositories** to the `repositories` section

### Adding New Repositories

```yaml
repositories:
  my-new-repo:
    description: "My new repository"
    paths:
      - "path/to/include/"
      - "!path/to/exclude/"
    post_split_actions:
      - action: "update_setup"
        package_name: "my-new-repo"
        description: "Description for my new repo"
```

### Custom Templates

Add custom README templates in the `templates` section:

```yaml
templates:
  my_template: |
    # My Custom README
    
    This is a custom template for my repository.
    
    ## Installation
    
    ```bash
    pip install my-package
    ```
```

## Validation

Always validate your configuration before running the split:

```bash
python validate_config.py --config repo_split_config.yaml --verbose
```

The validator checks:
- ✅ Configuration file syntax
- ✅ Path existence in source repository
- ✅ Post-split action validity
- ✅ Template references
- ✅ Dependency consistency
- ⚠️ Path conflicts between repositories
- ⚠️ Git repository status

## Troubleshooting

### Common Issues

1. **git-filter-repo not found**
   ```bash
   pip install git-filter-repo
   ```

2. **Path doesn't exist warnings**
   - Check that paths in config match actual repository structure
   - Use `find . -name "pattern"` to locate files

3. **Memory issues with large repositories**
   - Close other applications
   - Consider processing repositories one at a time

4. **Permission errors**
   - Ensure write permissions to target directory
   - Check that source repository isn't locked by other processes

### Recovery

If the split fails partway through:
1. Check the error message for specific issues
2. Fix the configuration file
3. Remove partial results: `rm -rf ../split-repos/`
4. Re-run the split

### Getting Help

For detailed debugging:
```bash
python split_repository.py --config repo_split_config.yaml --verbose
```

## Best Practices

### Before Splitting

1. **Backup your repository** (create a complete backup)
2. **Clean git status** (commit or stash changes)
3. **Switch to main branch** (ensure you're on the primary branch)
4. **Validate configuration** (run validation script)
5. **Test with dry-run** (verify the plan)

### After Splitting

1. **Review each repository** independently
2. **Test package installations** and imports using modern src/ layout
3. **Verify git history** is preserved (`git log --oneline`)
4. **Check all branches** are present (`git branch -a`)
5. **Test cross-package dependencies**
6. **Validate modern packaging** works correctly (`pip install -e .`)
7. **Run development tools** (pytest, black, mypy, etc.)

### Creating Remote Repositories

For each split repository:
```bash
cd ../split-repos/rompy-core
git remote add origin https://github.com/your-org/rompy-core.git
git push -u origin --all
git push --tags
```

## Post-Split Development

### Setting Up Development Environment

For each repository with modern src/ layout:
```bash
cd ../split-repos/rompy-core
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with optional dependencies
pip install -e .[dev]

# Set up pre-commit hooks
pre-commit install

# Run tests to verify setup
pytest

# Format code
black src tests
isort src tests
```

### Inter-Package Dependencies

Packages can depend on each other:
- rompy-swan depends on rompy-core
- rompy-schism depends on rompy-core
- rompy-notebooks depends on all packages

### Version Management

Consider using:
- **Semantic versioning** (1.0.0, 1.1.0, etc.)
- **Synchronized releases** (all packages version together)
- **Independent versioning** (each package versions separately)

## Advanced Usage

### Selective Repository Processing

Process only specific repositories:
```python
# Modify the script to process only certain repos
repositories_to_process = ['rompy-core', 'rompy-swan']
for repo_name in repositories_to_process:
    if repo_name in config['repositories']:
        splitter.split_repository(repo_name, config['repositories'][repo_name])
```

### Custom Post-Processing

Add custom post-processing scripts:
```bash
# After splitting, run custom processing
for repo in ../split-repos/*/; do
    echo "Processing $repo"
    cd "$repo"
    # Run custom scripts here
    python setup_custom.py
    cd -
done
```

### Automated Testing

Test each split repository with modern tooling:
```bash
cd ../split-repos/rompy-core

# Run tests with coverage
pytest --cov=src/rompy_core --cov-report=html

# Run all quality checks
tox

# Test multiple Python versions
tox -e py38,py39,py310,py311,py312

# Run linting
tox -e lint

# Check formatting
black --check src tests
isort --check-only src tests
mypy src
```

## Maintenance

### Re-running Splits

The configuration is designed to be reusable. To re-run:
1. Update the configuration file
2. Remove previous results: `rm -rf ../split-repos/`
3. Re-run the split script

### Updating Configurations

When the source repository structure changes:
1. Update paths in `repo_split_config.yaml`
2. Validate the new configuration
3. Re-run the split

## Security Considerations

- **Sensitive data**: Ensure no secrets are in the git history
- **Access controls**: Set appropriate permissions on split repositories
- **Dependency security**: Audit dependencies in each package
- **Modern tooling**: Use pre-commit hooks and automated security scanning
- **Version pinning**: Pin development dependencies for reproducible builds

## Modern Python Packaging Benefits

The split repositories use modern Python packaging practices:

### Src Layout Advantages
- **Import isolation**: Prevents accidental imports from source during testing
- **Cleaner testing**: Tests run against installed package, not source code
- **Better development workflow**: Separates source from other project files
- **Industry standard**: Follows current Python packaging recommendations

### Modern Tooling
- **pyproject.toml**: Modern packaging configuration format
- **setuptools_scm**: Automatic version management from git tags
- **tox**: Multi-environment testing
- **pre-commit**: Automated code quality checks
- **GitHub Actions**: Continuous integration and testing
- **Type hints**: Full mypy type checking support

## Performance Notes

- Large repositories may take significant time to process
- Each repository is processed sequentially
- Temporary disk usage can be 3-5x the source repository size
- Consider running on a machine with adequate resources

## Support

For issues with the splitting process:
1. Check this documentation
2. Validate your configuration
3. Review error messages carefully
4. Consider asking for help with specific error messages

Remember: The goal is to create maintainable, focused repositories while preserving the valuable git history of your work.