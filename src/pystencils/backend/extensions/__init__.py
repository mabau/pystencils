"""
The module `pystencils.backend.extensions` contains extensions to the pystencils code generator
beyond its core functionality.

The tools and classes of this module are considered experimental;
their support by the remaining code generator is limited.
They can be used to model and generate code outside of the usual scope of pystencils,
such as non-standard syntax and types.
At the moment, the primary use case is the modelling of C++ syntax.


Foreign Syntax Support
======================

.. automodule:: pystencils.backend.extensions.foreign_ast
    :members:


C++ Language Support
====================

.. automodule:: pystencils.backend.extensions.cpp
    :members:

"""

from .foreign_ast import PsForeignExpression

__all__ = ["PsForeignExpression"]
