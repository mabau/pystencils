"""
The `kernelcreation` module contains the actual translation logic of the pystencils code generator.
It provides a number of classes and submodules providing the various parts and passes of the code
generation process:

 - Parameterization of the translation process
 - Knowledge collection and management
 - Kernel analysis and constraints checks
 - Expression parsing and AST construction
 - Platform-specific code materialization
 - General and platform-specific optimizations

These components are designed to be combined and composed in a variety of ways, depending
on the actual code generation flow required.
The ``nbackend`` currently provides one native code generation driver:
`create_kernel` takes an `AssignmentCollection` and translates it to a simple loop kernel.
The code generator's components are perhaps most easily introduced in the context of that particular driver.

Exemplary Code Generation Driver: `create_kernel`
-------------------------------------------------

Generator Arguments
^^^^^^^^^^^^^^^^^^^

The driver accepts two parameters: an `AssignmentCollection` whose assignments represent the code of a single
kernel iteration without recurrences or other loop-carried dependencies; and a `CreateKernelConfig` which configures
the translation process.

Context Creation
^^^^^^^^^^^^^^^^

The primary object that stores all information and knowledge required during the translation process is the
`KernelCreationContext`. It is created in the beginning from the configuration parameter.
It will be responsible for managing all fields and arrays encountered during translation,
the kernel's iteration space,
and any parameter constraints introduced by later transformation passes.

Analysis Passes
^^^^^^^^^^^^^^^

Before the actual translation of the SymPy-based assignment collection to the backend's AST begins,
the kernel's assignments are checked for consistency with the translator's prequesites.
In this case, the `KernelAnalysis` pass
checks the static single assignment-form (SSA) requirement and the absence of loop-carried dependencies.
At the same time, it collects the set of all fields used in the assignments.

Iteration Space Creation
^^^^^^^^^^^^^^^^^^^^^^^^

The kernel's `IterationSpace` is inferred from a combination of configuration parameters and the set of field accesses
encountered in the kernel. Two kinds of iteration spaces are available: A sparse iteration space
(`SparseIterationSpace`) encompasses singular points in the cartesian coordinate space, provided by an index list.
A full iteration space (`FullIterationSpace`), on the other hand, represents a full cuboid cartesian coordinate space,
which may optionally be sliced.

The iteration space is used during the following translation passes to translate field accesses with respect to
the current iteration. It will only be instantiated in the form of a loop nest or GPU index calculation much later.

Freeze and Typification
^^^^^^^^^^^^^^^^^^^^^^^

The transformation of the SymPy-expressions to the backend's expression trees is handled by `FreezeExpressions`.
This class instantiates field accesses according to the iteration space, maps SymPy operators and functions to their
backend instances if supported, and raises an exception if asked to translate something the backend can't handle.

Constants and untyped symbols in the frozen expressions now need to be assigned a data type, and expression types
need to be checked against the C typing rules. This is the task of the `Typifier`. It assigns a default type to
every untyped symbol, attempts to infer the type of constants from their context in the expression,
and checks expression types using a much stricter
subset of the C typing rules, allowing for no implicit type casts even between closely related types.
After the typification pass, the code generator either has a fully and correctly typed kernel body in hand,
or it has raised an exception.

Platform-Specific Iteration Space Materialization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

At this point, most remaining transformations are specific to the target platform. Hardware platforms are modelled
using subclasses of the `Platform` class, which implement all platform-specific transformations.
The platform for the current code generation flow is instantiated from the target specification passed
by the user in `CreateKernelConfig`.
Then, the platform is asked to materialize the iteration space (e.g. by turning it into a loop nest
for CPU code) and to materialize any functions for which it provides specific implementations.

Platform-Specific Optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Technically, the kernel is now complete, but it may still be optimized.
This is also the task of the platform instance. Potential optimizations include the inclusion of OpenMP,
loop splitting, slicing and blocking on CPUs,
and vectorization on CPU platforms with vector capabilities.

Finalization
^^^^^^^^^^^^

At last, the kernel is packed up as a `PsKernelFunction`.
It is furthermore annotated with constraints collected during the translation, and returned to the user.

"""

from .context import KernelCreationContext
from .analysis import KernelAnalysis
from .freeze import FreezeExpressions
from .typification import Typifier

from .iteration_space import (
    FullIterationSpace,
    SparseIterationSpace,
    create_full_iteration_space,
    create_sparse_iteration_space,
)

__all__ = [
    "KernelCreationContext",
    "KernelAnalysis",
    "FreezeExpressions",
    "Typifier",
    "FullIterationSpace",
    "SparseIterationSpace",
    "create_full_iteration_space",
    "create_sparse_iteration_space",
]
