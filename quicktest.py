#!/usr/bin/env python3

from contextlib import redirect_stdout
import io
from tests.test_quicktests import (
    test_basic_kernel,
    test_basic_blocking_staggered,
    test_basic_vectorization,
)

quick_tests = [
    test_basic_kernel,
    test_basic_blocking_staggered,
    test_basic_vectorization,
]

if __name__ == "__main__":
    print("Running pystencils quicktests")
    for qt in quick_tests:
        print(f"   -> {qt.__name__}")
        with redirect_stdout(io.StringIO()):
            qt()
