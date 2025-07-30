# SCHISM Test Suite

This directory contains the comprehensive test suite for the SCHISM implementation in ROMPY.

## Test Structure

The test suite is organized into three main categories:

- **Unit Tests**: Testing individual components in isolation
- **Integration Tests**: Testing how components work together
- **Functional Tests**: End-to-end tests of complete model setups

## Running Tests

You can run the test suite using pytest:

```bash
# Run all SCHISM tests
python -m pytest tests/schism

# Run only unit tests
python -m pytest tests/schism/unit

# Run a specific test module
python -m pytest tests/schism/unit/test_grid.py

# Run tests with verbose output
python -m pytest tests/schism -v

# Run tests and show output
python -m pytest tests/schism -v --capture=no
```

## Test Data

Test data is stored in the `data` directory. This includes:

- Grid files (hgrid.gr3)
- Sample hydrodynamic data
- Atmospheric data files
- Reference files for validation

## Fixtures

Common fixtures are defined in `conftest.py`. These provide:

- Grid definitions
- Sample data sources
- Helper functions for boundary preparation
- Temporary directory management

## Adding New Tests

When adding new tests:

1. Place unit tests in the appropriate module under `unit/`
2. Integration tests should go under `integration/`
3. End-to-end functional tests should go under `functional/`
4. Add any new fixtures to `conftest.py`
5. Follow the naming convention: `test_*.py` for files and `test_*` for functions
6. Add helpful docstrings explaining what each test is checking

## File Formats

The test suite handles various file formats:

- GR3 files: Grid and parameter files
- NML files: Namelist configuration files
- YAML files: Declarative model specification
- NC files: NetCDF data files for boundaries and forcing
