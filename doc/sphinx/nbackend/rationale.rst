***********************
Rationale and Key Ideas
***********************

Expression Manipulation
^^^^^^^^^^^^^^^^^^^^^^^

The pystencils code generator was originally built based entirely on the computer algebra system SymPy.
SymPy itself is ideal for the front-end representation of kernels using its mathematical language.
In pystencils, however, SymPy was long used to model all mathematical expressions, from the continuous equations
down to the bare C assignments, loop counters, and even pointer arithmetic.
SymPy's unique properties, especially regarding automatic rewriting and simplification of expressions,
while perfect for doing symbolic mathematics, have proven to be very problematic when used as the basis of
an intermediate code representation.

The primary problems caused by using SymPy for expression manipulation are these:

 - Assigning and checking types on SymPy expressions is not possible in a stable way. While a type checking
   pass over the expression trees may validate types early in the code generation process, often SymPy's auto-
   rewriting system will be triggered by changes to the AST at a later stage, silently invalidating type
   information.
 - SymPy will aggressively simplify constant expressions in a strictly mathematical way, which leads to
   semantically invalid transformations in contexts with fixed types. This problem especially concerns
   integer types, and division in integer contexts.
 - SymPy aggressively flattens expressions according to associativity, and freely reorders operands in commutative
   operations. While perfectly fine in symbolic mathematics, this behaviour makes it impossible to group
   and parenthesize operations for numerical or performance benefits. Another often-observed effect is that
   SymPy distributes constant factors across sums, strongly increasing the number of FLOPs.

To avoid these problems, ``nbackend`` uses the [pymbolic](https://pypi.org/project/pymbolic/) package for expression
manipulation. Pymblic has similar capabilities for writing mathematic expressions as SymPy, however its expression
trees are much simpler, completely static, and easier to extend.

Structure and Architecture of the Code Generator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The code generation flow of *pystencils* has grown significantly over the years, to accomodate various different
kinds of kernels, output dialects, and target platforms. Very often, extensions were retroactively integrated with
a system that was not originally designed to support them. As a result, the code generator is now
a very convoluted set of functions and modules, containing large volumes of hard-to-read code, much of it
duplicated for several platforms.

The design of the ``nbackend`` takes the benefit of hindsight to provide the same (and, in some cases, a broader) set of
functionality through a much better structured software system. While the old code generator was implemented in an almost
entirely imperative manner, the ``nbackend`` makes extensive use of object-oriented programming for knowledge representation,
construction and internal representation of code, as well as analysis, transformation, and code generation tasks.
As a result, the ``nbackend`` is much more modular, concise, easier to extend, and implemented in a much smaller volume of
code.

