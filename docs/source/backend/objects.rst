****************************
Constants and Memory Objects
****************************

Memory Objects: Symbols and Field Arrays
========================================

The Memory Model
----------------

In order to reason about memory accesses, mutability, invariance, and aliasing, the *pystencils* backend uses
a very simple memory model. There are three types of memory objects:

- Symbols (`PsSymbol`), which act as registers for data storage within the scope of a kernel
- Field arrays (`PsLinearizedArray`), which represent a contiguous block of memory the kernel has access to, and
- the *unmanaged heap*, which is a global catch-all memory object which all pointers not belonging to a field
  array point into.

All of these objects are disjoint, and cannot alias each other.
Each symbol exists in isolation,
field arrays do not overlap,
and raw pointers are assumed not to point into memory owned by a symbol or field array.
Instead, all raw pointers point into unmanaged heap memory, and are assumed to *always* alias one another:
Each change brought to unmanaged memory by one raw pointer is assumed to affect the memory pointed to by
another raw pointer.

Classes
-------

.. autoclass:: pystencils.backend.symbols.PsSymbol
    :members:

.. automodule:: pystencils.backend.arrays
    :members:


Constants and Literals
======================

.. autoclass:: pystencils.backend.constants.PsConstant
    :members:

.. autoclass:: pystencils.backend.literals.PsLiteral
    :members:
