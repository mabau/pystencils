from __future__ import annotations

from typing import TypeVar

import pymbolic.primitives as pb
from pymbolic.mapper import IdentityMapper

from .context import KernelCreationContext

from ..ast import PsAstNode, PsExpression
from ..arrays import PsArrayAccess, TypeErasedBasePointer
from ..typed_expressions import PsTypedConstant
from ..types import PsStructType, PsPointerType
from ..functions import deref, address_of, Cast

NodeT = TypeVar("NodeT", bound=PsAstNode)


class EraseAnonymousStructTypes(IdentityMapper):
    """Lower anonymous struct arrays to a byte-array representation.

    For arrays whose element type is an anonymous struct, the struct type is erased from the base pointer,
    making it a pointer to uint8_t.
    Member lookups on accesses into these arrays are then transformed using type casts.
    """

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx

    def __call__(self, node: NodeT) -> NodeT:
        match node:
            case PsExpression(expr):
                # descend into expr
                node.expression = self.rec(expr)
            case other:
                for c in other.children:
                    self(c)

        return node

    def map_lookup(self, lookup: pb.Lookup) -> pb.Expression:
        aggr = lookup.aggregate
        if not isinstance(aggr, PsArrayAccess):
            return lookup

        arr = aggr.array
        if (
            not isinstance(arr.element_type, PsStructType)
            or not arr.element_type.anonymous
        ):
            return lookup

        struct_type = arr.element_type
        struct_size = struct_type.itemsize

        bp = aggr.base_ptr
        type_erased_bp = TypeErasedBasePointer(bp.name, arr)
        base_index = aggr.index_tuple[0] * PsTypedConstant(struct_size, self._ctx.index_dtype)

        member_name = lookup.name
        member = struct_type.get_member(member_name)
        assert member is not None

        np_struct = struct_type.numpy_dtype
        assert np_struct is not None
        assert np_struct.fields is not None
        member_offset = np_struct.fields[member_name][1]

        byte_index = base_index + PsTypedConstant(member_offset, self._ctx.index_dtype)
        type_erased_access = PsArrayAccess(type_erased_bp, byte_index)

        cast = Cast(PsPointerType(member.dtype))

        return deref(cast(address_of(type_erased_access)))
