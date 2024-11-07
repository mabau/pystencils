from __future__ import annotations

from typing import cast

from .astnode import PsAstNode
from .expressions import PsExpression, PsLvalue, PsUnOp
from .util import failing_cast

from ...types import PsVectorType


class PsVectorOp:
    """Mix-in for vector operations"""


class PsVecBroadcast(PsUnOp, PsVectorOp):
    """Broadcast a scalar value to N vector lanes."""

    __match_args__ = ("lanes", "operand")
    
    def __init__(self, lanes: int, operand: PsExpression):
        super().__init__(operand)
        self._lanes = lanes

    @property
    def lanes(self) -> int:
        return self._lanes
    
    @lanes.setter
    def lanes(self, n: int):
        self._lanes = n

    def _clone_expr(self) -> PsVecBroadcast:
        return PsVecBroadcast(self._lanes, self._operand.clone())
    
    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsVecBroadcast):
            return False
        return (
            super().structurally_equal(other)
            and self._lanes == other._lanes
        )


class PsVecMemAcc(PsExpression, PsLvalue, PsVectorOp):
    """Pointer-based vectorized memory access.
    
    Args:
        base_ptr: Pointer identifying the accessed memory region
        offset: Offset inside the memory region
        vector_entries: Number of elements to access
        stride: Optional integer step size for strided access, or ``None`` for contiguous access
        aligned: For contiguous accesses, whether the access is guaranteed to be naturally aligned
            according to the vector data type
    """

    __match_args__ = ("pointer", "offset", "vector_entries", "stride", "aligned")

    def __init__(
        self,
        base_ptr: PsExpression,
        offset: PsExpression,
        vector_entries: int,
        stride: PsExpression | None = None,
        aligned: bool = False,
    ):
        super().__init__()

        self._ptr = base_ptr
        self._offset = offset
        self._vector_entries = vector_entries
        self._stride = stride
        self._aligned = aligned

    @property
    def pointer(self) -> PsExpression:
        return self._ptr

    @pointer.setter
    def pointer(self, expr: PsExpression):
        self._ptr = expr

    @property
    def offset(self) -> PsExpression:
        return self._offset

    @offset.setter
    def offset(self, expr: PsExpression):
        self._offset = expr

    @property
    def vector_entries(self) -> int:
        return self._vector_entries

    @property
    def stride(self) -> PsExpression | None:
        return self._stride
    
    @stride.setter
    def stride(self, expr: PsExpression | None):
        self._stride = expr

    @property
    def aligned(self) -> bool:
        return self._aligned

    def get_vector_type(self) -> PsVectorType:
        return cast(PsVectorType, self._dtype)
    
    def get_children(self) -> tuple[PsAstNode, ...]:
        return (self._ptr, self._offset) + (() if self._stride is None else (self._stride,))
    
    def set_child(self, idx: int, c: PsAstNode):
        idx = [0, 1, 2][idx]
        match idx:
            case 0:
                self._ptr = failing_cast(PsExpression, c)
            case 1:
                self._offset = failing_cast(PsExpression, c)
            case 2:
                self._stride = failing_cast(PsExpression, c)

    def _clone_expr(self) -> PsVecMemAcc:
        return PsVecMemAcc(
            self._ptr.clone(),
            self._offset.clone(),
            self.vector_entries,
            self._stride.clone() if self._stride is not None else None,
            self._aligned,
        )

    def structurally_equal(self, other: PsAstNode) -> bool:
        if not isinstance(other, PsVecMemAcc):
            return False

        return (
            super().structurally_equal(other)
            and self._vector_entries == other._vector_entries
            and self._aligned == other._aligned
        )
    
    def __repr__(self) -> str:
        return (
            f"PsVecMemAcc({repr(self._ptr)}, {repr(self._offset)}, {repr(self._vector_entries)}, "
            f"stride={repr(self._stride)}, aligned={repr(self._aligned)})"
        )
