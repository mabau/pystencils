.. _page_symbolic_language:

*****************
Symbolic Language
*****************

Pystencils allows you to define near-arbitrarily complex numerical kernels in its symbolic
language, which is based on the computer algebra system `SymPy <https://www.sympy.org>`_.
The pystencils code generator is able to parse and translate a large portion of SymPy's
symbolic expression toolkit, and furthermore extends it with its own features.
Among the supported SymPy features are: symbols, constants, arithmetic and logical expressions,
trigonometric and most transcendental functions, as well as piecewise definitions.

Symbols
=======

Mathematical variables are generally represented using `sympy.Symbol <sympy.core.symbol.Symbol>`.

Fields
======

The most important extension to SymPy brought by pystencils are *fields*.
Fields are a symbolic representation of multidimensional cartesian numerical arrays,
as used in many stencil algorithms.
They are represented by the `Field` class.
Fields can be created from a textual description using the `fields <pystencils.field.fields>` function,
or, more concisely, using the factory methods `Field.create_generic` and `Field.create_fixed_size`.
It is also possible to create a field representing an existing numpy array,
including its shape, data type, and memory layout, using `Field.create_from_numpy_array`.

.. autosummary::
    :nosignatures:

    pystencils.Field


Assignments and Assignment Collections
======================================

Pystencils relies heavily on SymPy's `Assignment <sympy.codegen.ast.Assignment>` class.
Assignments are the fundamental components of pystencils kernels;
they are used both for assigning expressions to symbols
and for writing values to fields.

Assignments are combined and structured inside `assignment collections <pystencils.AssignmentCollection>`.
An assignment collection contains two separate lists of assignments:

- The **subexpressions** list contains assignments to symbols which can be reused in all subsequent assignments.
  These are typically used to structure computations into parts
  by precomputing (common) subexpressions
- The **main assignments** represent the actual effect of the kernel by storing the computation's results
  into fields.

.. autosummary::
    :nosignatures:

    pystencils.Assignment
    pystencils.AssignmentCollection


Restrictions on SymPy Expressions
=================================

In order to produce valid C code, the *pystencils* code generator places some restrictions 
on the SymPy expressions it consumes.

Piecewise Definitions
---------------------

Pystencils can parse and translate piecewise function definitions using
`sympy.Piecewise <sympy.functions.elementary.piecewise.Piecewise>`
*only if* they have a default case.
Any incomplete piecewise definition, such as the following, will result in an error from pystencils:

.. code-block:: Python

    sp.Piecewise((0, x < 0), (1, sp.And(x >= 0, x < 1)))

To avoid this, you may explicitly mark the final branch as the default case by
setting its condition to ``True``:

.. code-block:: Python

    sp.Piecewise((0, x < 0), (1, True))

This is not always necessary; if SymPy can prove that the range of possible values is covered completely,
it might simplify the final condition to ``True`` automatically.

Integer Operations
==================

Division and Remainder
----------------------

Care has to be taken when working with integer division operations in pystencils.
The python operators ``//`` and ``%`` work differently from their counterparts in the C family of languages.
Where in C, integer division always rounds toward zero, ``//`` performs a floor-divide (or euclidean division)
which rounds toward negative infinity.
These two operations differ whenever one of the operands is negative.
Accordingly, in Python ``a % b`` returns the *euclidean modulus*,
while C ``a % b`` computes the *remainder* of division.
The euclidean modulus is always nonnegative, while the remainder, if nonzero, always has the same sign as ``a``.

When ``//`` and ``%`` occur in symbolic expressions given to pystencils, they are interpreted the Python-way.
This can lead to inefficient generated code, since Pythonic integer division does not map to the corresponding C
operators.
To achieve C behaviour (and efficient code), you can use
`pystencils.symb.int_div <pystencils.sympyextensions.integer_functions.int_div>` and
`pystencils.symb.int_rem <pystencils.sympyextensions.integer_functions.int_rem>`
which translate to C ``/`` and ``%``, respectively.

When expressions are translated in an integer type context, the Python ``/`` operator
will also be converted to C-style ``/`` integer division.
Still, use of ``/`` for integers is discouraged, as it is designed to return a floating-point value in Python.
