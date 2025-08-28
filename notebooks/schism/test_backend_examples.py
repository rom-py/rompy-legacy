#!/usr/bin/env python3
"""
Test script for the new SCHISM backend framework implementation.

This script demonstrates how to use the new Python-based boundary conditions
examples runner and validates that it works correctly.

Usage: python test_backend_examples.py
"""

import sys
from pathlib import Path

# Add the rompy directory to Python path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from run_boundary_conditions_examples import SchismExampleRunner

def test_runner_initialization():
    """Test that the runner initializes correctly."""
    print("Testing SchismExampleRunner initialization...")

    try:
        runner = SchismExampleRunner()
        print("✓ Runner initialized successfully")
        print(f"✓ Project root: {runner.project_root}")
        print(f"✓ Examples directory: {runner.examples_dir}")
        print(f"✓ Found {len(runner.examples)} example configurations")
        return True
    except Exception as e:
        print(f"✗ Runner initialization failed: {e}")
        return False

def test_example_discovery():
    """Test that examples are discovered correctly."""
    print("\nTesting example discovery...")

    try:
        runner = SchismExampleRunner()

        # Test categories
        categories = ["tidal", "hybrid", "river", "nested"]
        for category in categories:
            examples = runner._get_examples_to_run(category)
            print(f"✓ Found {len(examples)} examples in category '{category}'")

        # Test all examples
        all_examples = runner._get_examples_to_run("all")
        print(f"✓ Found {len(all_examples)} total examples")

        # Test single example
        single_example = runner._get_examples_to_run("all", "basic_tidal")
        if single_example == ["basic_tidal"]:
            print("✓ Single example selection works")
        else:
            print("✗ Single example selection failed")
            return False

        return True
    except Exception as e:
        print(f"✗ Example discovery failed: {e}")
        return False

def test_configuration_validation():
    """Test that example configurations are valid."""
    print("\nTesting configuration validation...")

    try:
        runner = SchismExampleRunner()

        valid_configs = 0
        invalid_configs = []

        for name, config in runner.examples.items():
            config_file = config["file"]

            if config_file.exists():
                try:
                    import yaml
                    with open(config_file, 'r') as f:
                        yaml_data = yaml.safe_load(f)

                    # Check for required fields
                    required_fields = ["run_id", "period", "config"]
                    if all(field in yaml_data for field in required_fields):
                        valid_configs += 1
                        print(f"✓ {name}: Configuration valid")
                    else:
                        invalid_configs.append(f"{name}: Missing required fields")
                        print(f"✗ {name}: Missing required fields")

                except Exception as e:
                    invalid_configs.append(f"{name}: {e}")
                    print(f"✗ {name}: {e}")
            else:
                invalid_configs.append(f"{name}: File not found")
                print(f"✗ {name}: Configuration file not found")

        print(f"\n✓ {valid_configs} valid configurations")
        if invalid_configs:
            print(f"✗ {len(invalid_configs)} invalid configurations")
            return False

        return True

    except Exception as e:
        print(f"✗ Configuration validation failed: {e}")
        return False

def test_dry_run():
    """Test dry run functionality."""
    print("\nTesting dry run functionality...")

    try:
        runner = SchismExampleRunner()

        # Test dry run with single example
        successful_runs, failed_runs = runner.run_examples(
            category="all",
            single_example="basic_tidal",
            dry_run=True,
            keep_outputs=True
        )

        if successful_runs == ["basic_tidal"] and not failed_runs:
            print("✓ Dry run completed successfully")
            return True
        else:
            print("✗ Dry run failed")
            return False

    except Exception as e:
        print(f"✗ Dry run failed: {e}")
        return False

def test_docker_config_creation():
    """Test Docker configuration creation."""
    print("\nTesting Docker configuration creation...")

    try:
        runner = SchismExampleRunner()

        # Create a test path (create it so Docker validation passes)
        test_path = Path("/tmp/test_schism")
        test_path.mkdir(parents=True, exist_ok=True)

        # Test with normal executable
        config = runner._create_docker_config(test_path, "")
        print("✓ Docker config created for standard executable")
        print(f"  - Dockerfile: {config.dockerfile}")
        print(f"  - Build context: {config.build_context}")
        print(f"  - CPU: {config.cpu}")
        print(f"  - Memory: {config.memory}")
        print(f"  - Timeout: {config.timeout}")

        # Test with WWM executable
        runner._create_docker_config(test_path, "_WWM")
        print("✓ Docker config created for WWM executable")

        # Clean up test directory
        import shutil
        if test_path.exists():
            shutil.rmtree(test_path)

        return True

    except Exception as e:
        print(f"✗ Docker config creation failed: {e}")
        return False

def test_prerequisites():
    """Test prerequisites checking."""
    print("\nTesting prerequisites...")

    try:
        runner = SchismExampleRunner()

        # Check project structure
        required_paths = [
            runner.project_root / "docker" / "schism" / "Dockerfile",
            runner.project_root / "docker" / "schism",  # Docker context
            runner.project_root / "tests" / "schism" / "test_data" / "tides" / "oceanum-atlas.tar.gz",
            runner.examples_dir
        ]

        missing_paths = []
        for path in required_paths:
            if path.exists():
                print(f"✓ Found: {path}")
            else:
                missing_paths.append(path)
                print(f"✗ Missing: {path}")

        if missing_paths:
            print(f"\n✗ Missing {len(missing_paths)} required files/directories")
            return False

        print("\n✓ All required files and directories found")
        return True

    except Exception as e:
        print(f"✗ Prerequisites check failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("SCHISM Backend Framework Test Suite")
    print("=" * 60)

    tests = [
        test_runner_initialization,
        test_example_discovery,
        test_configuration_validation,
        test_dry_run,
        test_docker_config_creation,
        test_prerequisites
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test {test_func.__name__} crashed: {e}")
            failed += 1

        print("-" * 40)

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total:  {len(tests)}")

    if failed == 0:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print(f"\n❌ {failed} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
