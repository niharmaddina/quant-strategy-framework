"""
Quick test runner script for fast testing.
Usage: python tests/run_tests.py [--fast] [--all] [--network]
"""

import sys
import subprocess
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_tests(fast_only=True, network=False, verbose=True):
    """
    Run tests with specified options.
    
    Args:
        fast_only: Only run fast tests (default: True)
        network: Include network tests (default: False)
        verbose: Verbose output (default: True)
    """
    cmd = ["python", "-m", "pytest", "tests/"]
    
    if fast_only and not network:
        cmd.extend(["-m", "fast"])
    
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
    
    cmd.extend(["--tb=short", "--color=yes"])
    
    print(f"Running: {' '.join(cmd)}")
    print("=" * 60)
    
    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run test suite')
    parser.add_argument('--fast', action='store_true', default=True,
                        help='Run only fast tests (default)')
    parser.add_argument('--all', action='store_true',
                        help='Run all tests including slow ones')
    parser.add_argument('--network', action='store_true',
                        help='Include network-dependent tests')
    parser.add_argument('--quiet', action='store_true',
                        help='Quiet output')
    
    args = parser.parse_args()
    
    fast_only = not args.all
    network = args.network
    verbose = not args.quiet
    
    exit_code = run_tests(fast_only=fast_only, network=network, verbose=verbose)
    sys.exit(exit_code)

