#!/usr/bin/env python3
"""
Test runner script for SIMS problem PLS integration tests.

Usage:
    python run_tests.py --help
    python run_tests.py --quick    # Run only small instances
    python run_tests.py --full     # Run all tests including large instances
    python run_tests.py --data     # Test only data loading
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd, description):
    """Run a command and handle errors."""
    print(f"\n{description}")
    print("=" * len(description))
    print(f"Running: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
        return False
    else:
        print(f"✅ {description} completed successfully")
        return True


def main():
    parser = argparse.ArgumentParser(description="Run SIMS PLS integration tests")
    parser.add_argument("--quick", action="store_true", 
                       help="Run only quick tests (small instances)")
    parser.add_argument("--full", action="store_true", 
                       help="Run all tests including slow ones")
    parser.add_argument("--data", action="store_true", 
                       help="Test only data loading functionality")
    parser.add_argument("--verbose", "-v", action="store_true", 
                       help="Verbose output")
    
    args = parser.parse_args()
    
    if not any([args.quick, args.full, args.data]):
        args.quick = True  # Default to quick tests
    
    success = True
    
    # Base pytest command
    pytest_cmd = ["python", "-m", "pytest"]
    if args.verbose:
        pytest_cmd.append("-v")
    
    if args.data:
        # Test data loading functionality
        cmd = pytest_cmd + ["tests/test_data_loader_unit.py"]
        success &= run_command(cmd, "Data Loader Tests")
    
    if args.quick:
        # Run quick tests (exclude slow marker)
        cmd = pytest_cmd + ["-m", "not slow", "tests/test_pls_real_data_integration.py"]
        success &= run_command(cmd, "Quick PLS Integration Tests")
    
    if args.full:
        # Run all tests including slow ones
        cmd = pytest_cmd + ["tests/test_pls_real_data_integration.py"]
        success &= run_command(cmd, "Full PLS Integration Tests")
        
        # Also run existing tests to ensure we didn't break anything
        cmd = pytest_cmd + ["tests/test_pls_integration.py"]
        success &= run_command(cmd, "Existing PLS Tests")
    
    if success:
        print("\n🎉 All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Some tests failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()
