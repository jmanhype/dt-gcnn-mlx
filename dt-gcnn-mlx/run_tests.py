#!/usr/bin/env python3
"""Run all tests for DT-GCNN MLX implementation."""

import unittest
import sys
import os
import argparse
from io import StringIO


def run_tests(test_module=None, verbose=2):
    """Run tests with optional module filtering."""
    
    # Discover all tests
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    if test_module:
        # Run specific test module
        pattern = f'test_{test_module}.py'
    else:
        # Run all tests
        pattern = 'test_*.py'
        
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir, pattern=pattern)
    
    # Count total tests
    total_tests = suite.countTestCases()
    print(f"Discovered {total_tests} tests")
    print("=" * 70)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=verbose, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"\n- {test}")
            print(f"  {traceback.split(chr(10))[-2]}")
            
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"\n- {test}")
            print(f"  {traceback.split(chr(10))[-2]}")
            
    return result.wasSuccessful()


def run_specific_test(test_path, verbose=2):
    """Run a specific test method."""
    # Format: module.TestClass.test_method
    parts = test_path.split('.')
    
    if len(parts) < 2:
        print(f"Invalid test path: {test_path}")
        print("Expected format: module.TestClass.test_method")
        return False
        
    module_name = f"test_{parts[0]}"
    test_class = parts[1] if len(parts) > 1 else None
    test_method = parts[2] if len(parts) > 2 else None
    
    # Load the test
    loader = unittest.TestLoader()
    
    try:
        if test_method:
            # Specific test method
            suite = loader.loadTestsFromName(f"tests.{module_name}.{test_class}.{test_method}")
        elif test_class:
            # All tests in a class
            suite = loader.loadTestsFromName(f"tests.{module_name}.{test_class}")
        else:
            # All tests in a module
            suite = loader.loadTestsFromModule(f"tests.{module_name}")
            
    except Exception as e:
        print(f"Error loading test: {e}")
        return False
        
    # Run the test
    runner = unittest.TextTestRunner(verbosity=verbose)
    result = runner.run(suite)
    
    return result.wasSuccessful()


def list_tests():
    """List all available tests."""
    test_dir = os.path.join(os.path.dirname(__file__), 'tests')
    
    print("Available test modules:")
    print("-" * 40)
    
    for filename in sorted(os.listdir(test_dir)):
        if filename.startswith('test_') and filename.endswith('.py'):
            module_name = filename[5:-3]  # Remove 'test_' and '.py'
            print(f"  {module_name}")
            
            # Load module to list test classes
            module_path = os.path.join(test_dir, filename)
            with open(module_path, 'r') as f:
                content = f.read()
                
            # Find test classes (simple parsing)
            import re
            classes = re.findall(r'class\s+(\w+)\s*\(.*TestCase\):', content)
            for class_name in classes:
                print(f"    - {class_name}")


def main():
    parser = argparse.ArgumentParser(description='Run DT-GCNN tests')
    parser.add_argument('test', nargs='?', help='Specific test to run (e.g., model.TestDTGCNN)')
    parser.add_argument('-v', '--verbose', type=int, default=2, choices=[0, 1, 2],
                        help='Verbosity level (0=quiet, 1=normal, 2=verbose)')
    parser.add_argument('-l', '--list', action='store_true',
                        help='List all available tests')
    parser.add_argument('-m', '--module', help='Run all tests in a specific module')
    
    args = parser.parse_args()
    
    if args.list:
        list_tests()
        return
        
    print("DT-GCNN MLX Test Suite")
    print("=" * 70)
    
    if args.test:
        # Run specific test
        success = run_specific_test(args.test, args.verbose)
    elif args.module:
        # Run specific module
        success = run_tests(args.module, args.verbose)
    else:
        # Run all tests
        success = run_tests(verbose=args.verbose)
        
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()