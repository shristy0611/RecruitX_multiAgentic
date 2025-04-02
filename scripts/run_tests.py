#!/usr/bin/env python3
"""
Script to run the test suite for the RecruitX application.
Provides a simplified interface for running tests with various options.
"""

import sys
import os
import argparse
import subprocess


def run_tests(test_path=None, coverage=False, verbose=True, markers=None):
    """Run tests with specified options."""
    # Ensure we're in the project root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    os.chdir(project_root)
    
    # Prepare command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path if specified
    if test_path:
        cmd.append(test_path)
    
    # Add coverage if requested
    if coverage:
        cmd.extend(["--cov=recruitx_app", "--cov-report=term", "--cov-report=html"])
    
    # Add verbosity
    if verbose:
        cmd.append("-v")
    
    # Add markers if specified
    if markers:
        cmd.append(f"-m {markers}")
    
    # Print command
    print(f"Running: {' '.join(cmd)}")
    
    # Run the tests
    try:
        result = subprocess.run(cmd)
        return result.returncode
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(description="Run RecruitX tests")
    
    # Add arguments
    parser.add_argument("--path", "-p", help="Test path to run (e.g. tests/unit)")
    parser.add_argument("--coverage", "-c", action="store_true", help="Generate coverage report")
    parser.add_argument("--quiet", "-q", action="store_true", help="Run in quiet mode")
    parser.add_argument("--markers", "-m", help="Run tests with specific markers (e.g. unit or integration)")
    
    # Add convenience arguments for common test runs
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument("--integration", action="store_true", help="Run only integration tests")
    parser.add_argument("--agents", action="store_true", help="Run only agent tests")
    
    args = parser.parse_args()
    
    # Handle convenience arguments
    if args.unit:
        args.path = "tests/unit"
    elif args.integration:
        args.path = "tests/integration"
    elif args.agents:
        args.path = "tests/unit/agents"
    
    # Run the tests
    return run_tests(
        test_path=args.path,
        coverage=args.coverage,
        verbose=not args.quiet,
        markers=args.markers
    )


if __name__ == "__main__":
    sys.exit(main()) 