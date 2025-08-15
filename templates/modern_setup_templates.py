#!/usr/bin/env python3
"""
Modern Setup Templates for Src Layout

This module contains templates for creating modern Python packages with
the src/ layout following current best practices.
"""

PYPROJECT_TOML_SRC_TEMPLATE = """[build-system]
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
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering",
    "Topic :: Scientific/Engineering :: Hydrology",
]
requires-python = ">=3.8"
dependencies = [
{dependencies}
]
dynamic = ["version"]

[project.urls]
Homepage = "https://github.com/rompy/{repo_name}"
Documentation = "https://{repo_name}.readthedocs.io/"
Repository = "https://github.com/rompy/{repo_name}.git"
Issues = "https://github.com/rompy/{repo_name}/issues"

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "black>=22.0",
    "isort>=5.0",
    "flake8>=5.0",
    "mypy>=1.0",
    "pre-commit>=2.0",
]
docs = [
    "sphinx>=5.0",
    "sphinx-rtd-theme>=1.0",
    "myst-parser>=0.18",
    "sphinx-autodoc-typehints>=1.19",
]
test = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "pytest-xdist>=3.0",
]

[tool.setuptools]
packages = {{find = {{where = ["src"]}}}}

[tool.setuptools_scm]
write_to = "src/{package_module}/_version.py"

[tool.black]
line-length = 88
target-version = ['py38']
include = '\\.pyi?$'
extend-exclude = '''
/(
  # directories
  \\.eggs
  | \\.git
  | \\.hg
  | \\.mypy_cache
  | \\.tox
  | \\.venv
  | _build
  | buck-out
  | build
  | dist
)/
'''

[tool.isort]
profile = "black"
line_length = 88
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
src_paths = ["src", "tests"]

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-v --strict-markers --strict-config --cov=src/{package_module} --cov-report=term-missing"
testpaths = ["tests"]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]

[tool.coverage.run]
source = ["src/{package_module}"]
omit = [
    "*/tests/*",
    "*/test_*",
    "*/_version.py",
]

[tool.coverage.report]
exclude_lines = [
    "pragma: no cover",
    "def __repr__",
    "if self.debug:",
    "if settings.DEBUG",
    "raise AssertionError",
    "raise NotImplementedError",
    "if 0:",
    "if __name__ == .__main__.:",
    "class .*\\\\bProtocol\\\\):",
    "@(abc\\\\.)?abstractmethod",
]

[tool.mypy]
python_version = "3.8"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
disallow_untyped_decorators = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
warn_no_return = true
warn_unreachable = true
strict_equality = true
show_error_codes = true

[[tool.mypy.overrides]]
module = "tests.*"
disallow_untyped_defs = false
"""

SETUP_CFG_SRC_TEMPLATE = """[metadata]
name = {package_name}
description = {description}
long_description = file: README.md
long_description_content_type = text/markdown
url = https://github.com/rompy/{repo_name}
author = Rompy Contributors
license = MIT
license_files = LICENSE
classifiers =
    Development Status :: 4 - Beta
    Intended Audience :: Developers
    Intended Audience :: Science/Research
    License :: OSI Approved :: MIT License
    Operating System :: OS Independent
    Programming Language :: Python :: 3
    Programming Language :: Python :: 3.8
    Programming Language :: Python :: 3.9
    Programming Language :: Python :: 3.10
    Programming Language :: Python :: 3.11
    Programming Language :: Python :: 3.12
    Topic :: Scientific/Engineering
    Topic :: Scientific/Engineering :: Hydrology

[options]
packages = find:
package_dir =
    = src
python_requires = >=3.8
install_requires =
{dependencies}
include_package_data = True
zip_safe = False

[options.packages.find]
where = src

[options.extras_require]
dev =
    pytest>=7.0
    pytest-cov>=4.0
    black>=22.0
    isort>=5.0
    flake8>=5.0
    mypy>=1.0
    pre-commit>=2.0
docs =
    sphinx>=5.0
    sphinx-rtd-theme>=1.0
    myst-parser>=0.18
    sphinx-autodoc-typehints>=1.19
test =
    pytest>=7.0
    pytest-cov>=4.0
    pytest-xdist>=3.0

[bdist_wheel]
universal = 0

[tool:pytest]
addopts = -v --strict-markers --strict-config --cov=src/{package_module} --cov-report=term-missing
testpaths = tests
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
"""

MANIFEST_IN_SRC_TEMPLATE = """include README.md
include LICENSE
include HISTORY.rst
include pyproject.toml
include setup.cfg
recursive-include src *.py
recursive-include tests *.py
recursive-include src/{package_module}/templates *
recursive-include docs *.rst *.py *.bat Makefile
recursive-exclude * __pycache__
recursive-exclude * *.py[co]
prune .git
prune .pytest_cache
prune .mypy_cache
prune build
prune dist
"""

INIT_PY_SRC_TEMPLATE = '''"""
{package_title}

{description}
"""

try:
    from ._version import __version__
except ImportError:
    # Package is not installed, use a default version
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
'''

TOX_INI_SRC_TEMPLATE = """[tox]
envlist = py38,py39,py310,py311,py312,lint,docs
isolated_build = True

[testenv]
deps =
    pytest>=7.0
    pytest-cov>=4.0
    pytest-xdist>=3.0
commands =
    pytest {{posargs}}

[testenv:lint]
deps =
    black>=22.0
    isort>=5.0
    flake8>=5.0
    mypy>=1.0
commands =
    black --check src tests
    isort --check-only src tests
    flake8 src tests
    mypy src

[testenv:format]
deps =
    black>=22.0
    isort>=5.0
commands =
    black src tests
    isort src tests

[testenv:docs]
deps =
    sphinx>=5.0
    sphinx-rtd-theme>=1.0
    myst-parser>=0.18
    sphinx-autodoc-typehints>=1.19
commands =
    sphinx-build -b html docs docs/_build/html

[testenv:type]
deps =
    mypy>=1.0
commands =
    mypy src
"""

GITHUB_WORKFLOW_SRC_TEMPLATE = """name: Tests

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main, develop ]

jobs:
  test:
    runs-on: ${{{{ matrix.os }}}}
    strategy:
      matrix:
        os: [ubuntu-latest, windows-latest, macos-latest]
        python-version: ['3.8', '3.9', '3.10', '3.11', '3.12']

    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0

    - name: Set up Python ${{{{ matrix.python-version }}}}
      uses: actions/setup-python@v4
      with:
        python-version: ${{{{ matrix.python-version }}}}

    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[test]

    - name: Run tests
      run: |
        pytest

    - name: Upload coverage to Codecov
      if: matrix.os == 'ubuntu-latest' && matrix.python-version == '3.11'
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  lint:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[dev]
    - name: Lint with flake8
      run: |
        flake8 src tests
    - name: Check formatting with black
      run: |
        black --check src tests
    - name: Check import sorting with isort
      run: |
        isort --check-only src tests
    - name: Type check with mypy
      run: |
        mypy src
"""

PRE_COMMIT_SRC_TEMPLATE = """repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.4.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
      - id: check-merge-conflict
      - id: debug-statements

  - repo: https://github.com/psf/black
    rev: 23.7.0
    hooks:
      - id: black
        language_version: python3
        args: [--line-length=88]

  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
        args: [--profile=black, --line-length=88]

  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=88, --extend-ignore=E203,W503]

  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.5.1
    hooks:
      - id: mypy
        additional_dependencies: [types-all]
        args: [--strict]
"""

README_SRC_TEMPLATE = """# {package_title}

{description}

## Installation

### From PyPI

```bash
pip install {package_name}
```

### Development Installation

For development, clone the repository and install in editable mode:

```bash
git clone https://github.com/rompy/{repo_name}.git
cd {repo_name}
pip install -e .[dev]
```

## Usage

```python
import {package_module}

# Example usage here
```

## Development

This project uses a modern Python packaging structure with the source code in the `src/` directory.

### Project Structure

```
{repo_name}/
├── src/
│   └── {package_module}/
│       ├── __init__.py
│       └── ...
├── tests/
│   └── test_*.py
├── docs/
│   └── ...
├── pyproject.toml
└── README.md
```

### Running Tests

```bash
# Install test dependencies
pip install -e .[test]

# Run tests
pytest

# Run tests with coverage
pytest --cov=src/{package_module} --cov-report=html
```

### Code Formatting

This project uses Black for code formatting and isort for import sorting:

```bash
# Format code
black src tests
isort src tests

# Check formatting
black --check src tests
isort --check-only src tests
```

### Type Checking

```bash
mypy src
```

### Documentation

```bash
# Install docs dependencies
pip install -e .[docs]

# Build documentation
sphinx-build -b html docs docs/_build/html
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for your changes
5. Ensure all tests pass
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Related Projects

- [rompy-core](https://github.com/rompy/rompy-core) - Core rompy functionality
- [rompy-swan](https://github.com/rompy/rompy-swan) - SWAN model integration
- [rompy-schism](https://github.com/rompy/rompy-schism) - SCHISM model integration
"""


def get_template(template_name: str) -> str:
    """Get a template by name."""
    templates = {
        "pyproject_toml_src": PYPROJECT_TOML_SRC_TEMPLATE,
        "setup_cfg_src": SETUP_CFG_SRC_TEMPLATE,
        "manifest_in_src": MANIFEST_IN_SRC_TEMPLATE,
        "init_py_src": INIT_PY_SRC_TEMPLATE,
        "tox_ini_src": TOX_INI_SRC_TEMPLATE,
        "github_workflow_src": GITHUB_WORKFLOW_SRC_TEMPLATE,
        "pre_commit_src": PRE_COMMIT_SRC_TEMPLATE,
        "readme_src": README_SRC_TEMPLATE,
    }
    return templates.get(template_name, "")


def format_template(template: str, **kwargs) -> str:
    """Format a template with the given keyword arguments."""
    return template.format(**kwargs)


def create_modern_setup_files(
    target_dir: str,
    package_name: str,
    package_module: str,
    description: str,
    repo_name: str,
    dependencies: list = None,
):
    """
    Create modern setup files with src layout in the target directory.

    Args:
        target_dir: Directory to create files in
        package_name: Name of the package (e.g., 'rompy-core')
        package_module: Name of the Python module (e.g., 'rompy_core')
        description: Package description
        repo_name: Repository name
        dependencies: List of dependencies
    """
    import os

    dependencies = dependencies or []
    deps_str = "\n".join([f'    "{dep}",' for dep in dependencies])

    # Format template variables
    template_vars = {
        "package_name": package_name,
        "package_module": package_module,
        "package_title": package_name.replace("-", " ").title(),
        "description": description,
        "repo_name": repo_name,
        "dependencies": deps_str,
    }

    # Create pyproject.toml
    pyproject_content = format_template(PYPROJECT_TOML_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(target_dir, "pyproject.toml"), "w") as f:
        f.write(pyproject_content)

    # Create setup.cfg (for compatibility)
    setup_cfg_content = format_template(SETUP_CFG_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(target_dir, "setup.cfg"), "w") as f:
        f.write(setup_cfg_content)

    # Create MANIFEST.in
    manifest_content = format_template(MANIFEST_IN_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(target_dir, "MANIFEST.in"), "w") as f:
        f.write(manifest_content)

    # Create tox.ini
    tox_content = format_template(TOX_INI_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(target_dir, "tox.ini"), "w") as f:
        f.write(tox_content)

    # Create .pre-commit-config.yaml
    precommit_content = format_template(PRE_COMMIT_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(target_dir, ".pre-commit-config.yaml"), "w") as f:
        f.write(precommit_content)

    # Create GitHub workflow
    os.makedirs(os.path.join(target_dir, ".github", "workflows"), exist_ok=True)
    workflow_content = format_template(GITHUB_WORKFLOW_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(target_dir, ".github", "workflows", "tests.yml"), "w") as f:
        f.write(workflow_content)

    # Create modern README
    readme_content = format_template(README_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(target_dir, "README.md"), "w") as f:
        f.write(readme_content)

    # Create src package structure with modern __init__.py
    src_package_dir = os.path.join(target_dir, "src", package_module)
    os.makedirs(src_package_dir, exist_ok=True)

    init_content = format_template(INIT_PY_SRC_TEMPLATE, **template_vars)
    with open(os.path.join(src_package_dir, "__init__.py"), "w") as f:
        f.write(init_content)
