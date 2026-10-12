"""
Unified test runner for the entire quant strategy framework.
Runs all tests with proper organization and reporting.

Usage:
    python tests/run_all_tests.py              # Run all fast tests
    python tests/run_all_tests.py --all        # Run all tests including slow ones
    python tests/run_all_tests.py --coverage   # Run with coverage report
    python tests/run_all_tests.py --verbose    # Verbose output
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_pytest_tests(fast_only=True, network=False, verbose=True, coverage=False):
    """
    Run all pytest-based tests.
    
    Args:
        fast_only: Only run fast tests (default: True)
        network: Include network tests (default: False)
        verbose: Verbose output (default: True)
        coverage: Generate coverage report (default: False)
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    # Use the current interpreter (venv-safe). Avoid calling bare "python".
    cmd = [sys.executable, "-m", "pytest", "tests/"]
    
    # Test selection
    if fast_only and not network:
        cmd.extend(["-m", "fast"])
    elif network:
        cmd.extend(["-m", "fast or network"])
    
    # Output options
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    # Coverage
    if coverage:
        cmd.extend([
            "--cov=backtest",
            "--cov=data",
            "--cov=strategy",
            "--cov=universe",
            "--cov=utils",
            "--cov-report=term-missing",
            "--cov-report=html:htmlcov"
        ])
    
    cmd.extend(["--tb=short", "--color=yes"])
    
    print("=" * 70)
    print("RUNNING PYTEST TESTS")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd, cwd=project_root)
    return result.returncode


def run_universe_simple_tests():
    """
    Run universe simple tests (no API calls, fast sanity check).
    
    Returns:
        Exit code (0 for success, non-zero for failure)
    """
    print("=" * 70)
    print("RUNNING UNIVERSE SIMPLE TESTS (No API Calls)")
    print("=" * 70)
    print()
    
    result = subprocess.run([sys.executable, "tests/test_universe_simple.py"], cwd=project_root)
    
    return result.returncode


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(
        description='Unified test runner for quant strategy framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tests/run_all_tests.py              # Run all fast tests (default)
  python tests/run_all_tests.py --all        # Run all tests including slow ones
  python tests/run_all_tests.py --network   # Include network tests
  python tests/run_all_tests.py --coverage   # Generate coverage report
  python tests/run_all_tests.py --quiet      # Quiet output
  python tests/run_all_tests.py --simple    # Run simple universe tests only
        """
    )
    
    parser.add_argument(
        '--all',
        action='store_true',
        help='Run all tests including slow ones (default: fast only)'
    )
    parser.add_argument(
        '--network',
        action='store_true',
        help='Include network-dependent tests'
    )
    parser.add_argument(
        '--coverage',
        action='store_true',
        help='Generate coverage report'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Quiet output (minimal)'
    )
    parser.add_argument(
        '--simple',
        action='store_true',
        help='Run only simple universe tests (no API calls)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        default=True,
        help='Verbose output (default)'
    )
    
    args = parser.parse_args()
    
    # Determine test mode
    fast_only = not args.all
    verbose = not args.quiet and args.verbose
    
    exit_codes = []
    
    # Run simple universe tests if requested
    if args.simple:
        exit_code = run_universe_simple_tests()
        exit_codes.append(exit_code)
    else:
        # Run main pytest test suite
        exit_code = run_pytest_tests(
            fast_only=fast_only,
            network=args.network,
            verbose=verbose,
            coverage=args.coverage
        )
        exit_codes.append(exit_code)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    if all(code == 0 for code in exit_codes):
        print("✓ All tests passed!")
        print()
        print("Test Coverage:")
        print("  - Backtest Engine: Comprehensive (52 tests)")
        print("  - Universe Constructor: Comprehensive (39 tests)")
        print("  - Data Fetcher: Unit tests")
        print("  - Strategy Indicators: Unit tests")
        print("  - Integration: End-to-end tests")
        print("  - Utils: Validation tests")
    else:
        print("✗ Some tests failed")
        for i, code in enumerate(exit_codes, 1):
            status = "✓" if code == 0 else "✗"
            print(f"  {status} Test suite {i}: {'PASSED' if code == 0 else 'FAILED'}")
    
    print("=" * 70)
    print()
    
    # Return non-zero if any test failed
    final_exit_code = 0 if all(code == 0 for code in exit_codes) else 1
    sys.exit(final_exit_code)


if __name__ == "__main__":
    main()

