*****************
Symbolic Language
*****************

.. toctree::
    :maxdepth: 1

    field
    astnodes
    sympyextensions

Pystencils allows you to define near-arbitrarily complex numerical kernels in its symbolic
language, which is based on the computer algebra system `SymPy <https://www.sympy.org>`_.
The pystencils code generator is able to parse and translate a large portion of SymPy's
symbolic expression toolkit, and furthermore extends it with its own features.
Among the supported SymPy features are: symbols, constants, arithmetic and logical expressions,
trigonometric and most transcendental functions, as well as piecewise definitions.

Fields
======

The most important extension to SymPy brought by pystencils are *fields*.
Fields are a symbolic representation of multidimensional cartesian numerical arrays,
as used in many stencil algorithms.
They are represented by the `Field` class.

Piecewise Definitions
=====================

Pystencils can parse and translate piecewise function definitions using `sympy.Piecewise`
*only if* they have a default case.
So, for instance,

.. code-block:: Python

    sp.Piecewise((0, x < 0), (1, x >= 0))

will result in an error from pystencils, while the equivalent

.. code-block:: Python

    sp.Piecewise((0, x < 0), (1, True))

will be accepted. This is because pystencils cannot reason about whether or not
the given cases completely cover the entire possible input range.

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
To achieve C behaviour (and efficient code), you can use `pystencils.symb.int_div` and `pystencils.symb.int_rem`
which translate to C ``/`` and ``%``, respectively.

When expressions are translated in an integer type context, the Python ``/`` operator (or `sympy.Div`)
will also be converted to C-style ``/`` integer division.
Still, use of ``/`` for integers is discouraged, as it is designed to return a floating-point value in Python.
