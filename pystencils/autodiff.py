"""
Provides tools for generation of auto-differentiable operations.

See https://github.com/theHamsta/pystencils_autodiff

Installation:

.. code-block:: bash
    pip install pystencils-autodiff
"""
import os

if 'CI' not in os.environ:
    raise NotImplementedError('pystencils-autodiff is not installed. Run `pip install pystencils-autodiff`')
