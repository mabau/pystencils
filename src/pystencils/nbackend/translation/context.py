from ...field import Field
from ..arrays import PsLinearizedArray, PsArrayBasePointer
from ..types import PsIntegerType
from ..constraints import PsKernelConstraint

from .iteration_domain import PsIterationDomain

class PsTranslationContext:
    """The `PsTranslationContext` manages the translation process from the SymPy frontend
    to the backend AST.
    
    It does the following things:

      - Default data types: The context knows the data types that should be applied by default
        to SymPy expressions.
      - Management of fields. The context manages all mappings from front-end `Field`s to their
        underlying `PsLinearizedArray`s.
      - Collection of constraints. All constraints that arise during translation are collected in the
        context, and finally attached to the kernel function object once translation is complete.
    
    Data Types
    ----------

     - The `index_dtype` is the data type used throughout translation for all loop counters and array indexing.
     - The `default_numeric_dtype` is the data type assigned by default to all symbols occuring in SymPy assignments
    
    Fields and Arrays
    -----------------

    There's several types of fields that need to be mapped to arrays.

    - `FieldType.GENERIC` corresponds to domain fields. 
      Domain fields can only be accessed by relative offsets, and therefore must always
      be associated with an *iteration domain* that provides a spatial index tuple.
      All domain fields associated with the same domain must have the same spatial shape, modulo ghost layers.
    - `FieldType.INDEXED` are 1D arrays of index structures. They must be accessed by a single running index.
      If there is at least one indexed field present there must also exist an index source for that field
      (loop or device indexing).
      An indexed field may itself be an index source for domain fields.
    - `FieldType.BUFFER` are 1D arrays whose indices must be incremented with each access.
      Within a domain, a buffer may be either written to or read from, never both.


    In the translator, frontend fields and backend arrays are managed together using the `PsFieldArrayPair` class.
    """

    def __init__(self, index_dtype: PsIntegerType):
        self._index_dtype = index_dtype
        self._constraints: list[PsKernelConstraint] = []

    @property
    def index_dtype(self) -> PsIntegerType:
        return self._index_dtype
    
    def add_constraints(self, *constraints: PsKernelConstraint):
        self._constraints += constraints

    @property
    def constraints(self) -> tuple[PsKernelConstraint, ...]:
        return tuple(self._constraints)

