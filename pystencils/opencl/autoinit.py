"""
Automatically initializes OpenCL context using any device.

Use `pystencils.opencl.{init_globally_with_context,init_globally}` if you want to use a specific device.
"""

from pystencils.opencl import *  # noqa
from pystencils.opencl.opencljit import *  # noqa
from pystencils.opencl.opencljit import init_globally

init_globally()
