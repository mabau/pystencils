from __future__ import annotations

from typing import cast
from functools import reduce
import operator

from ..kernelcreation import KernelCreationContext, Typifier

from ..constants import PsConstant
from ..memory import PsSymbol, PsBuffer, BufferBasePtr
from ..ast.structural import PsAstNode
from ..ast.expressions import (
    PsBufferAcc,
    PsLookup,
    PsExpression,
    PsMemAcc,
    PsAddressOf,
    PsCast,
    PsSymbolExpr,
)
from ...types import PsStructType, PsPointerType, PsUnsignedIntegerType


class LowerToC:
    """Lower high-level IR constructs to C language concepts.

    This pass will replace a number of IR constructs that have no direct counterpart in the C language
    to lower-level AST nodes. These include:

    - *Linearization of Buffer Accesses:* `PsBufferAcc` buffer accesses are linearized according to
      their buffers' stride information and replaced by `PsMemAcc`.
    - *Erasure of Anonymous Structs:*
      For buffers whose element type is an anonymous struct, the struct type is erased from the base pointer,
      making it a pointer to uint8_t.
      Member lookups on accesses into these buffers are then transformed using type casts.
    """

    def __init__(self, ctx: KernelCreationContext) -> None:
        self._ctx = ctx
        self._substitutions: dict[PsSymbol, PsSymbol] = dict()

        self._typify = Typifier(ctx)

        from .eliminate_constants import EliminateConstants

        self._fold = EliminateConstants(self._ctx)

    def __call__(self, node: PsAstNode) -> PsAstNode:
        self._substitutions = dict()

        node = self.visit(node)

        for old, new in self._substitutions.items():
            self._ctx.replace_symbol(old, new)

        return node

    def visit(self, node: PsAstNode) -> PsAstNode:
        match node:
            case PsBufferAcc(bptr, indices):
                #   Linearize
                buf = node.buffer

                #   Typifier allows different data types in each index
                def maybe_cast(i: PsExpression):
                    if i.get_dtype() != buf.index_type:
                        return PsCast(buf.index_type, i)
                    else:
                        return i

                summands: list[PsExpression] = [
                    maybe_cast(cast(PsExpression, self.visit(idx).clone()))
                    * PsExpression.make(stride)
                    for idx, stride in zip(indices, buf.strides, strict=True)
                ]

                linearized_idx: PsExpression = (
                    summands[0]
                    if len(summands) == 1
                    else reduce(operator.add, summands)
                )

                mem_acc = PsMemAcc(bptr.clone(), linearized_idx)

                return self._fold(
                    self._typify.typify_expression(
                        mem_acc, target_type=buf.element_type
                    )[0]
                )

            case PsLookup(aggr, member_name) if isinstance(
                aggr, PsBufferAcc
            ) and isinstance(
                aggr.buffer.element_type, PsStructType
            ) and aggr.buffer.element_type.anonymous:
                #   Need to lower this buffer-lookup
                linearized_acc = self.visit(aggr)
                return self._lower_anon_lookup(
                    cast(PsMemAcc, linearized_acc), aggr.buffer, member_name
                )

            case _:
                node.children = [self.visit(c) for c in node.children]

        return node

    def _lower_anon_lookup(
        self, aggr: PsMemAcc, buf: PsBuffer, member_name: str
    ) -> PsExpression:
        struct_type = cast(PsStructType, buf.element_type)
        struct_size = struct_type.itemsize

        assert isinstance(aggr.pointer, PsSymbolExpr)
        bp = aggr.pointer.symbol
        bp_type = bp.get_dtype()
        assert isinstance(bp_type, PsPointerType)

        #   Need to keep track of base pointers already seen, since symbols must be unique
        if bp not in self._substitutions:
            erased_type = PsPointerType(
                PsUnsignedIntegerType(8, const=bp_type.base_type.const),
                const=bp_type.const,
                restrict=bp_type.restrict,
            )
            type_erased_bp = PsSymbol(bp.name, erased_type)
            type_erased_bp.add_property(BufferBasePtr(buf))
            self._substitutions[bp] = type_erased_bp
        else:
            type_erased_bp = self._substitutions[bp]

        base_index = aggr.offset * PsExpression.make(
            PsConstant(struct_size, self._ctx.index_dtype)
        )

        member = struct_type.find_member(member_name)
        assert member is not None

        np_struct = struct_type.numpy_dtype
        assert np_struct is not None
        assert np_struct.fields is not None
        member_offset = np_struct.fields[member_name][1]

        byte_index = base_index + PsExpression.make(
            PsConstant(member_offset, self._ctx.index_dtype)
        )
        type_erased_access = PsMemAcc(PsExpression.make(type_erased_bp), byte_index)

        deref = PsMemAcc(
            PsCast(PsPointerType(member.dtype), PsAddressOf(type_erased_access)),
            PsExpression.make(PsConstant(0)),
        )

        deref = self._typify(deref)
        return deref
