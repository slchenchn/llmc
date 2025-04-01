#!/usr/bin/env python3

from tests.test_quant_functions import TestQuantFunctions
import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Import the test modules

if __name__ == "__main__":
    # Create the test suite
    test_suite = unittest.TestSuite()

    # Add the test cases
    test_suite.addTest(
        unittest.defaultTestLoader.loadTestsFromTestCase(TestQuantFunctions))

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)

    # Exit with non-zero status if tests failed
    sys.exit(not result.wasSuccessful())
