##############################################
Developer's Reference: Code Generation Backend
##############################################

These pages provide a detailed overview of the pystencils code generation backend
as a reference for current and future developers of pystencils, as well as users
who wish to customize or extend the behaviour of the code generator in their applications.

.. toctree::
    :maxdepth: 1
    
    symbols
    ast
    iteration_space
    translation
    platforms
    transformations
    jit

Internal Representation
-----------------------

The code generator translates the kernel from the SymPy frontend's symbolic language to an internal
representation (IR), which is then emitted as code in the required dialect of C.
All names of classes associated with the internal kernel representation are prefixed ``Ps...``
to distinguis them from identically named front-end and SymPy classes.
The IR comprises *symbols*, *constants*, *arrays*, the *iteration space* and the *abstract syntax tree*:

* `PsSymbol` represents a single symbol in the kernel, annotated with a type. Other than in the frontend,
  uniqueness of symbols is enforced by the backend: of each symbol, at most one instance may exist.
* `PsConstant` provides a type-safe representation of constants.
* `PsLinearizedArray` is the backend counterpart to the ubiquitous `Field`, representing a contiguous
  n-dimensional array.
  These arrays do not occur directly in the IR, but are represented through their *associated symbols*,
  which are base pointers, shapes, and strides.
* The iteration space (`IterationSpace`) represents the kernel's iteration domain.
  Currently, dense iteration spaces (`FullIterationSpace`) and an index list-based
  sparse iteration spaces (`SparseIterationSpace`) are available.
* The *Abstract Syntax Tree* (AST) is implemented in the `pystencils.backend.ast` module.
  It represents a subset of standard C syntax, as required for pystencils kernels.


Kernel Creation
---------------

Translating a kernel's symbolic representation to compilable code takes various analysis, transformation, and
optimization passes. These are implemented modularily, each represented by its own class.
They are tied together in the kernel translation *driver* and communicate with each other through the
`KernelCreationContext`, which assembles all relevant information.
The primary translation driver implemented in pystencils is the ubiquitous `create_kernel`.
However, the backend is designed to make it easy for users and developers to implement custom translation
drivers if necessary.

The various functional components of the kernel translator are best explained in the order they are invoked
by `create_kernel`.

Analysis and Constraint Checks
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The `KernelAnalysis` pass parses the SymPy assignment list and checks it for the consistency requirements
of the code generator, including the absence of loop-carried dependencies and the static single-assignment form.
Furthermore, it populates the `KernelCreationContext` with information about all fields encountered in the kernel.

Creation of the Iteration Space
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Before the actual translation can begin, the kernel's iteration space must be defined.
The `pystencils.backend.kernelcreation.iteration_space` module provides various means of creating iteration spaces,
which are used by *create_kernel* according to its input configuration.
To communicate the presence of an iteration space to other components, it must be set in the context using
`KernelCreationContext.set_iteration_space`.
It will be used during the *freeze* pass, and later be materialized to a loop nest or GPU index translation.

Freeze and Typification
^^^^^^^^^^^^^^^^^^^^^^^

The transformation of the SymPy expressions to the backend's AST is handled by `FreezeExpressions`.
This class instantiates field accesses according to the iteration space, maps SymPy operators and functions to their
backend instances, and raises an exception if asked to translate something the backend can't handle.

Constants and untyped symbols in the frozen expressions now need to be assigned a data type, and expression types
need to be checked against the C typing rules. This is the task of the `Typifier`. It assigns a default type to
every untyped symbol, attempts to infer the type of constants from their context in the expression,
and checks expression types using a stricter subset of the C typing rules,
allowing for no implicit type casts even between closely related types.
After the typification pass, the code generator either has a fully and correctly typed kernel body in hand,
or it has raised an exception.

Platform Selection and Materialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The various hardware platforms supported by pystencils are implemented in the `pystencils.backend.platforms` module.
Each implements a target-specific materialization of generic backend components, including:

- The iteration space, which is materialized to a specific index source. This might be a loop nest for CPU kernels, or
  a thread index translation for GPU kernels
- Mathematical functions, which might have to be mapped to concrete library functions
- Vector data types and operations, which are mapped to intrinsics on vector CPU architectures

Transformations
^^^^^^^^^^^^^^^

TODO

Target-Specific Optimization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

TODO

Finalization
^^^^^^^^^^^^

TODO
