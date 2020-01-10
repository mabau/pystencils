"""
Automatically initializes OpenCL context using any device.

Use `pystencils.opencl.{init_globally_with_context,init_globally}` if you want to use a specific device.
"""

from pystencils.opencl.opencljit import (
    clear_global_ctx, init_globally, init_globally_with_context, make_python_function)

__all__ = ['init_globally', 'init_globally_with_context', 'clear_global_ctx', 'make_python_function']

try:
    init_globally()
except Exception as e:
    import warnings
    warnings.warn(str(e))
