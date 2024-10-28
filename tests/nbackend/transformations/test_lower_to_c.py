from functools import reduce
from operator import add

from pystencils import fields, Assignment, make_slice, Field, FieldType
from pystencils.types import PsStructType, create_type

from pystencils.backend.memory import BufferBasePtr
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
)
from pystencils.backend.transformations import LowerToC

from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.expressions import (
    PsBufferAcc,
    PsMemAcc,
    PsSymbolExpr,
    PsExpression,
    PsLookup,
    PsAddressOf,
    PsCast,
)
from pystencils.backend.ast.structural import PsAssignment


def test_lower_buffer_accesses():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:42, :31])
    ctx.set_iteration_space(ispace)

    lower = LowerToC(ctx)

    f, g = fields("f(2), g(3): [2D]")
    asm = Assignment(f.center(1), g[-1, 1](2))

    f_buf = ctx.get_buffer(f)
    g_buf = ctx.get_buffer(g)

    fasm = factory.parse_sympy(asm)
    assert isinstance(fasm.lhs, PsBufferAcc)
    assert isinstance(fasm.rhs, PsBufferAcc)

    fasm_lowered = lower(fasm)
    assert isinstance(fasm_lowered, PsAssignment)

    assert isinstance(fasm_lowered.lhs, PsMemAcc)
    assert isinstance(fasm_lowered.lhs.pointer, PsSymbolExpr)
    assert fasm_lowered.lhs.pointer.symbol == f_buf.base_pointer

    zero = factory.parse_index(0)
    expected_offset = reduce(
        add,
        (
            (PsExpression.make(dm.counter) + zero) * PsExpression.make(stride)
            for dm, stride in zip(ispace.dimensions, f_buf.strides)
        ),
    ) + factory.parse_index(1) * PsExpression.make(f_buf.strides[-1])
    assert fasm_lowered.lhs.offset.structurally_equal(expected_offset)

    assert isinstance(fasm_lowered.rhs, PsMemAcc)
    assert isinstance(fasm_lowered.rhs.pointer, PsSymbolExpr)
    assert fasm_lowered.rhs.pointer.symbol == g_buf.base_pointer

    expected_offset = (
        (PsExpression.make(ispace.dimensions[0].counter) + factory.parse_index(-1))
        * PsExpression.make(g_buf.strides[0])
        + (PsExpression.make(ispace.dimensions[1].counter) + factory.parse_index(1))
        * PsExpression.make(g_buf.strides[1])
        + factory.parse_index(2) * PsExpression.make(g_buf.strides[-1])
    )
    assert fasm_lowered.rhs.offset.structurally_equal(expected_offset)


def test_lower_anonymous_structs():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    ispace = FullIterationSpace.create_from_slice(ctx, make_slice[:12])
    ctx.set_iteration_space(ispace)

    lower = LowerToC(ctx)

    stype = PsStructType(
        [
            ("val", ctx.default_dtype),
            ("x", ctx.index_dtype),
        ]
    )
    sfield = Field.create_generic("s", spatial_dimensions=1, dtype=stype)
    f = Field.create_generic("f", 1, ctx.default_dtype, field_type=FieldType.CUSTOM)

    asm = Assignment(sfield.center("val"), f.absolute_access((sfield.center("x"),), (0,)))

    fasm = factory.parse_sympy(asm)

    sbuf = ctx.get_buffer(sfield)

    assert isinstance(fasm, PsAssignment)
    assert isinstance(fasm.lhs, PsLookup)

    lowered_fasm = lower(fasm.clone())
    assert isinstance(lowered_fasm, PsAssignment)

    #   Check type of sfield data pointer
    for expr in dfs_preorder(lowered_fasm, lambda n: isinstance(n, PsSymbolExpr)):
        if expr.symbol.name == sbuf.base_pointer.name:
            assert expr.symbol.dtype == create_type("uint8_t * restrict")

    #   Check LHS
    assert isinstance(lowered_fasm.lhs, PsMemAcc)
    assert isinstance(lowered_fasm.lhs.pointer, PsCast)
    assert isinstance(lowered_fasm.lhs.pointer.operand, PsAddressOf)
    assert isinstance(lowered_fasm.lhs.pointer.operand.operand, PsMemAcc)
    type_erased_pointer = lowered_fasm.lhs.pointer.operand.operand.pointer

    assert isinstance(type_erased_pointer, PsSymbolExpr)
    assert BufferBasePtr(sbuf) in type_erased_pointer.symbol.properties
    assert type_erased_pointer.symbol.dtype == create_type("uint8_t * restrict")
