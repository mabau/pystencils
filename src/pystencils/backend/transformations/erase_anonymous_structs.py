from __future__ import annotations

from ..kernelcreation.context import KernelCreationContext

from ..constants import PsConstant
from ..ast.structural import PsAstNode
from ..ast.expressions import (
    PsArrayAccess,
    PsLookup,
    PsExpression,
    PsMemAcc,
    PsAddressOf,
    PsCast,
)
from ..kernelcreation import Typifier
from ..arrays import PsArrayBasePointer, TypeErasedBasePointer
from ...types import PsStructType, PsPointerType


class EraseAnonymousStructTypes:
    """Lower anonymous struct arrays to a byte-array representation.

    For arrays whose element type is an anonymous struct, the struct type is erased from the base pointer,
    making it a pointer to uint8_t.
    Member lookups on accesses into these arrays are then transformed using type casts.
    """

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx

        self._substitutions: dict[PsArrayBasePointer, TypeErasedBasePointer] = dict()

    def __call__(self, node: PsAstNode) -> PsAstNode:
        self._substitutions = dict()

        #   Check if AST traversal is even necessary
        if not any(
            (isinstance(arr.element_type, PsStructType) and arr.element_type.anonymous)
            for arr in self._ctx.arrays
        ):
            return node

        node = self.visit(node)

        for old, new in self._substitutions.items():
            self._ctx.replace_symbol(old, new)

        return node

    def visit(self, node: PsAstNode) -> PsAstNode:
        match node:
            case PsLookup():
                # descend into expr
                return self.handle_lookup(node)
            case _:
                node.children = [self.visit(c) for c in node.children]

        return node

    def handle_lookup(self, lookup: PsLookup) -> PsExpression:
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

        #   Need to keep track of base pointers already seen, since symbols must be unique
        if bp not in self._substitutions:
            type_erased_bp = TypeErasedBasePointer(bp.name, arr)
            self._substitutions[bp] = type_erased_bp
        else:
            type_erased_bp = self._substitutions[bp]

        base_index = aggr.index * PsExpression.make(
            PsConstant(struct_size, self._ctx.index_dtype)
        )

        member_name = lookup.member_name
        member = struct_type.find_member(member_name)
        assert member is not None

        np_struct = struct_type.numpy_dtype
        assert np_struct is not None
        assert np_struct.fields is not None
        member_offset = np_struct.fields[member_name][1]

        byte_index = base_index + PsExpression.make(
            PsConstant(member_offset, self._ctx.index_dtype)
        )
        type_erased_access = PsArrayAccess(type_erased_bp, byte_index)

        deref = PsMemAcc(
            PsCast(PsPointerType(member.dtype), PsAddressOf(type_erased_access)),
            PsExpression.make(PsConstant(0))
        )

        typify = Typifier(self._ctx)
        deref = typify(deref)
        return deref
