from __future__ import annotations

from typing import TYPE_CHECKING, cast
from types import EllipsisType

from ...field import Field
from ...typing import TypedSymbol, BasicType
from ..arrays import PsLinearizedArray, PsArrayBasePointer
from ..types.quick import make_type
from ..typed_expressions import PsTypedVariable, PsTypedConstant, VarOrConstant
from .field_array_pair import PsDomainFieldArrayPair

if TYPE_CHECKING:
    from .context import PsTranslationContext

class PsIterationDomain:
    """Represents the n-dimensonal spatial iteration domain of a pystencils kernel.
    
    Domain Shape
    ------------

    A domain may have either constant or variable, n-dimensional shape, where n = 1, 2, 3.
    If the shape is variable, the domain object manages variables for each shape entry.

    The domain provides index variables for each dimension which may be used to access fields
    associated with the domain.
    In the kernel, these index variables must be provided by some index source.
    Index sources differ between two major types of domains: full and sparse domains.

    In a full domain, it is guaranteed that each interior point is processed by the kernel.
    The index source may therefore be a full n-fold loop nest, or a device index calculation.

    In a sparse domain, the iteration is controlled by an index vector, which acts as the index
    source.

    Arrays
    ------

    Any number of domain arrays may be associated with each domain.
    Each array is annotated with a number of ghost layers for each spatial coordinate.

    ### Shape Compatibility

    When an array is associated with a domain, it must be ensured that the array's shape
    is compatible with the domain.
    The first n shape entries are considered the array's spatial shape.
    These spatial shapes, after subtracting ghost layers, must all be equal, and are further
    constrained by a constant domain shape.
    For each spatial coordinate, shape compatibility is ensured as described by the following table.

    |                           |  Constant Array Shape       |   Variable Array Shape |
    |---------------------------|-----------------------------|------------------------|
    | **Constant Domain Shape** | Compile-Time Equality Check |  Kernel Constraints    |
    | **Variable Domain Shape** | Invalid, Compiler Error     |  Kernel Constraints    |

    ### Base Pointers and Array Accesses

    In the kernel's public interface, each array is represented at least through its base pointer,
    which represents the starting address of the array's data in memory.
    Since the iteration domain models arrays as being surrounded by ghost layers, it provides for each
    array a second, *interior* base pointer, which points to the first interior point after skipping the
    ghost layers, e.g. in three dimensions with one index dimension:

    ```
    addr(interior_base_ptr[0, 0, 0, 0]) == addr(base_ptr[gls, gls, gls, 0])
    ```

    To access domain arrays using the domain's index variables, the interior base pointer should be used,
    since the domain index variables always count up from zero.

    """

    def __init__(self, ctx: PsTranslationContext, shape: tuple[int | EllipsisType, ...]):
        self._ctx = ctx
        
        if len(shape) == 0:
            raise ValueError("Domain shape must be at least one-dimensional.")
        
        if len(shape) > 3:
            raise ValueError("Iteration domain can be at most three-dimensional.")
        
        self._shape: tuple[VarOrConstant, ...] = tuple(
            (
                PsTypedVariable(f"domain_size_{i}", self._ctx.index_dtype)
                if s == Ellipsis
                else PsTypedConstant(s, self._ctx.index_dtype)
            )
            for i, s in enumerate(shape)
        )

        self._archetype_field: PsDomainFieldArrayPair | None = None
        self._fields: dict[str, PsDomainFieldArrayPair] = dict()

    @property
    def shape(self) -> tuple[VarOrConstant, ...]:
        return self._shape
    
    def add_field(self, field: Field, ghost_layers: int) -> PsDomainFieldArrayPair:
        arr_shape = tuple(
            (Ellipsis if isinstance(s, TypedSymbol) else s) # TODO: Field should also use ellipsis
            for s in field.shape
        )

        arr_strides = tuple(
            (Ellipsis if isinstance(s, TypedSymbol) else s) # TODO: Field should also use ellipsis
            for s in field.strides
        )

        # TODO: frontend should use new type system
        element_type = make_type(cast(BasicType, field.dtype).numpy_dtype.type) 

        arr = PsLinearizedArray(field.name, element_type, arr_shape, arr_strides, self._ctx.index_dtype)

        fa_pair = PsDomainFieldArrayPair(
            field=field,
            array=arr,
            base_ptr=PsArrayBasePointer("arr_data", arr),
            ghost_layers=ghost_layers,
            interior_base_ptr=PsArrayBasePointer("arr_interior_data", arr),
            domain=self
        )
        
        #   Check shape compatibility
        #   TODO
        for domain_s, field_s in zip(self.shape, field.shape):
            if isinstance(domain_s, PsTypedConstant):
                pass

        raise NotImplementedError()

