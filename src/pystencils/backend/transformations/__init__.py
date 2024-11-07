"""
This module contains various transformation and optimization passes that can be
executed on the backend AST.

Canonical Form
==============

Many transformations in this module require that their input AST is in *canonical form*.
This means that:

- Each symbol, constant, and expression node is annotated with a data type;
- Each symbol has at most one declaration;
- Each symbol that is never written to apart from its declaration has a ``const`` type; and
- Each symbol whose type is *not* ``const`` has at least one non-declaring assignment.

The first requirement can be ensured by running the `Typifier` on each newly constructed subtree.
The other three requirements are ensured by the `CanonicalizeSymbols` pass,
which should be run first before applying any optimizing transformations.
All transformations in this module retain canonicality of the AST.

Canonicality allows transformations to forego various checks that would otherwise be necessary
to prove their legality.

Certain transformations, like the `LoopVectorizer`, state additional requirements, e.g.
the absence of loop-carried dependencies.

Transformations
===============

Canonicalization
----------------

.. autoclass:: CanonicalizeSymbols
    :members: __call__

AST Cloning
-----------

.. autoclass:: CanonicalClone
    :members: __call__

Simplifying Transformations
---------------------------

.. autoclass:: EliminateConstants
    :members: __call__

.. autoclass:: EliminateBranches
    :members: __call__

    
Code Rewriting
--------------

.. autofunction:: substitute_symbols
    
Code Motion
-----------

.. autoclass:: HoistLoopInvariantDeclarations
    :members: __call__

Loop Reshaping Transformations
------------------------------

.. autoclass:: ReshapeLoops
    :members:

.. autoclass:: InsertPragmasAtLoops
    :members:

.. autoclass:: AddOpenMP
    :members:

Vectorization
-------------

.. autoclass:: VectorizationAxis
    :members:

.. autoclass:: VectorizationContext
    :members:

.. autoclass:: AstVectorizer
    :members:

.. autoclass:: LoopVectorizer
    :members:
    
Code Lowering and Materialization
---------------------------------

.. autoclass:: LowerToC
    :members: __call__

.. autoclass:: SelectFunctions
    :members: __call__

.. autoclass:: SelectIntrinsics
    :members:

"""

from .canonicalize_symbols import CanonicalizeSymbols
from .canonical_clone import CanonicalClone
from .rewrite import substitute_symbols
from .eliminate_constants import EliminateConstants
from .eliminate_branches import EliminateBranches
from .hoist_loop_invariant_decls import HoistLoopInvariantDeclarations
from .reshape_loops import ReshapeLoops
from .add_pragmas import InsertPragmasAtLoops, LoopPragma, AddOpenMP
from .ast_vectorizer import VectorizationAxis, VectorizationContext, AstVectorizer
from .loop_vectorizer import LoopVectorizer
from .lower_to_c import LowerToC
from .select_functions import SelectFunctions
from .select_intrinsics import SelectIntrinsics

__all__ = [
    "CanonicalizeSymbols",
    "CanonicalClone",
    "substitute_symbols",
    "EliminateConstants",
    "EliminateBranches",
    "HoistLoopInvariantDeclarations",
    "ReshapeLoops",
    "InsertPragmasAtLoops",
    "LoopPragma",
    "AddOpenMP",
    "VectorizationAxis",
    "VectorizationContext",
    "AstVectorizer",
    "LoopVectorizer",
    "LowerToC",
    "SelectFunctions",
    "SelectIntrinsics",
]
