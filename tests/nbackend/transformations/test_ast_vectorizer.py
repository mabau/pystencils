import sympy as sp
import pytest

from pystencils import Assignment, TypedSymbol, fields, FieldType, make_slice
from pystencils.sympyextensions import CastFunc, mem_acc
from pystencils.sympyextensions.pointers import AddressOf

from pystencils.backend.constants import PsConstant
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    AstFactory,
    FullIterationSpace,
    Typifier,
)
from pystencils.backend.transformations import (
    VectorizationAxis,
    VectorizationContext,
    AstVectorizer,
)
from pystencils.backend.ast import dfs_preorder
from pystencils.backend.ast.structural import (
    PsBlock,
    PsDeclaration,
    PsAssignment,
    PsLoop,
)
from pystencils.backend.ast.expressions import (
    PsSymbolExpr,
    PsConstantExpr,
    PsExpression,
    PsCast,
    PsMemAcc,
    PsCall,
    PsSubscript,
)
from pystencils.backend.functions import CFunction
from pystencils.backend.ast.vector import PsVecBroadcast, PsVecMemAcc
from pystencils.backend.exceptions import VectorizationError
from pystencils.types import PsArrayType, PsVectorType, deconstify, create_type


def test_vectorize_expressions():
    x, y, z, w = sp.symbols("x, y, z, w")

    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    typify = Typifier(ctx)

    for s in (x, y, z, w):
        _ = factory.parse_sympy(s)

    ctr = ctx.get_symbol("ctr", ctx.index_dtype)

    axis = VectorizationAxis(ctr)
    vc = VectorizationContext(ctx, 4, axis)
    vc.vectorize_symbol(ctx.get_symbol("x"))
    vc.vectorize_symbol(ctx.get_symbol("w"))

    vectorize = AstVectorizer(ctx)

    for expr in [
        factory.parse_sympy(-x * y + 13 * z - 4 * (x / w) * (x + z)),
        factory.parse_sympy(sp.sin(x + z) - sp.cos(w)),
        factory.parse_sympy(y**2 - x**2),
        typify(
            -factory.parse_sympy(x / (w**2))
        ),  # place the negation outside, since SymPy would remove it
        factory.parse_sympy(13 + (1 / w) - sp.exp(x) * 24),
    ]:
        vec_expr = vectorize.visit(expr, vc)

        #   Must be a clone
        assert vec_expr is not expr

        scalar_type = ctx.default_dtype
        vector_type = PsVectorType(scalar_type, 4)

        for subexpr in dfs_preorder(vec_expr):
            match subexpr:
                case PsSymbolExpr(symb) if symb.name in "yz":
                    #   These are not vectorized, but broadcast
                    assert symb.dtype == scalar_type
                    assert subexpr.dtype == scalar_type
                case PsConstantExpr(c):
                    assert deconstify(c.get_dtype()) == scalar_type
                    assert subexpr.dtype == scalar_type
                case PsSymbolExpr(symb):
                    assert symb.name not in "xw"
                    assert symb.get_dtype() == vector_type
                    assert subexpr.dtype == vector_type
                case PsVecBroadcast(lanes, operand):
                    assert lanes == 4
                    assert subexpr.dtype == vector_type
                    assert subexpr.dtype.scalar_type == operand.dtype
                case PsExpression():
                    #   All other expressions are vectorized
                    assert subexpr.dtype == vector_type


def test_vectorize_casts_and_counter():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    ctr = ctx.get_symbol("ctr", ctx.index_dtype)
    vec_ctr = ctx.get_symbol("vec_ctr", PsVectorType(ctx.index_dtype, 4))

    vectorize = AstVectorizer(ctx)

    axis = VectorizationAxis(ctr, vec_ctr)
    vc = VectorizationContext(ctx, 4, axis)

    expr = factory.parse_sympy(CastFunc(sp.Symbol("ctr"), create_type("float32")))
    vec_expr = vectorize.visit(expr, vc)

    assert isinstance(vec_expr, PsCast)
    assert (
        vec_expr.dtype
        == vec_expr.target_type
        == PsVectorType(create_type("float32"), 4)
    )

    assert isinstance(vec_expr.operand, PsSymbolExpr)
    assert vec_expr.operand.symbol == vec_ctr
    assert vec_expr.operand.dtype == PsVectorType(ctx.index_dtype, 4)


def test_invalid_vectorization():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    typify = Typifier(ctx)

    ctr = ctx.get_symbol("ctr", ctx.index_dtype)

    vectorize = AstVectorizer(ctx)

    axis = VectorizationAxis(ctr)
    vc = VectorizationContext(ctx, 4, axis)

    expr = factory.parse_sympy(CastFunc(sp.Symbol("ctr"), create_type("float32")))

    with pytest.raises(VectorizationError):
        #   Fails since no vectorized counter was specified
        _ = vectorize.visit(expr, vc)

    expr = PsExpression.make(
        ctx.get_symbol("x_v", PsVectorType(create_type("float32"), 4))
    )

    with pytest.raises(VectorizationError):
        #   Fails since this symbol is already vectorial
        _ = vectorize.visit(expr, vc)

    func = CFunction("compute", [ctx.default_dtype], ctx.default_dtype)
    expr = typify(PsCall(func, [PsExpression.make(ctx.get_symbol("x"))]))

    with pytest.raises(VectorizationError):
        #   Can't vectorize unknown function
        _ = vectorize.visit(expr, vc)


def test_vectorize_declarations():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    x, y, z, w = sp.symbols("x, y, z, w")
    ctr = TypedSymbol("ctr", ctx.index_dtype)

    vectorize = AstVectorizer(ctx)

    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
        ctx.get_symbol("vec_ctr", PsVectorType(ctx.index_dtype, 4)),
    )
    vc = VectorizationContext(ctx, 4, axis)

    block = PsBlock(
        [
            factory.parse_sympy(asm)
            for asm in [
                Assignment(x, CastFunc.as_numeric(ctr)),
                Assignment(y, sp.cos(x)),
                Assignment(z, x**2 + 2 * y / 4),
                Assignment(w, -x + y - z),
            ]
        ]
    )

    vec_block = vectorize.visit(block, vc)
    assert vec_block is not block
    assert isinstance(vec_block, PsBlock)

    for symb_name, decl in zip("xyzw", vec_block.statements):
        symb = ctx.get_symbol(symb_name)
        assert symb in vc.vectorized_symbols

        assert isinstance(decl, PsDeclaration)
        assert decl.declared_symbol == vc.vectorized_symbols[symb]
        assert (
            decl.lhs.dtype
            == decl.declared_symbol.dtype
            == PsVectorType(ctx.default_dtype, 4)
        )


def test_duplicate_declarations():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    x, y = sp.symbols("x, y")

    vectorize = AstVectorizer(ctx)

    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
    )
    vc = VectorizationContext(ctx, 4, axis)

    block = PsBlock(
        [
            factory.parse_sympy(asm)
            for asm in [
                Assignment(y, sp.cos(x)),
                Assignment(y, 21),
            ]
        ]
    )

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(block, vc)


def test_reject_symbol_assignments():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    x, y = sp.symbols("x, y")

    vectorize = AstVectorizer(ctx)

    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
    )
    vc = VectorizationContext(ctx, 4, axis)

    asm = PsAssignment(factory.parse_sympy(x), factory.parse_sympy(3 + y))

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(asm, vc)


def test_vectorize_assignments():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    x, y = sp.symbols("x, y")

    vectorize = AstVectorizer(ctx)

    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
    )
    vc = VectorizationContext(ctx, 4, axis)

    decl = PsDeclaration(factory.parse_sympy(x), factory.parse_sympy(sp.sympify(0)))
    asm = PsAssignment(factory.parse_sympy(x), factory.parse_sympy(3 + y))
    ast = PsBlock([decl, asm])

    vec_ast = vectorize.visit(ast, vc)
    vec_asm = vec_ast.statements[1]

    assert isinstance(vec_asm, PsAssignment)
    assert isinstance(vec_asm.lhs.symbol.dtype, PsVectorType)


def test_vectorize_memory_assignments():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    typify = Typifier(ctx)
    vectorize = AstVectorizer(ctx)

    x, y = sp.symbols("x, y")

    ctr = TypedSymbol("ctr", ctx.index_dtype)
    i = TypedSymbol("i", ctx.index_dtype)
    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
    )
    vc = VectorizationContext(ctx, 4, axis)

    ptr = TypedSymbol("ptr", create_type("float64 *"))

    asm = typify(
        PsAssignment(
            factory.parse_sympy(mem_acc(ptr, 3 * ctr + 2)),
            factory.parse_sympy(x + y * mem_acc(ptr, ctr + 3)),
        )
    )

    vec_asm = vectorize.visit(asm, vc)
    assert isinstance(vec_asm, PsAssignment)
    assert isinstance(vec_asm.lhs, PsVecMemAcc)

    field = fields("field(1): [2D]", field_type=FieldType.CUSTOM)
    asm = factory.parse_sympy(
        Assignment(
            field.absolute_access((ctr, i), (0,)),
            x + y * field.absolute_access((ctr + 1, i), (0,)),
        )
    )

    vec_asm = vectorize.visit(asm, vc)
    assert isinstance(vec_asm, PsAssignment)
    assert isinstance(vec_asm.lhs, PsVecMemAcc)


def test_invalid_memory_assignments():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    typify = Typifier(ctx)
    vectorize = AstVectorizer(ctx)

    x, y = sp.symbols("x, y")

    ctr = TypedSymbol("ctr", ctx.index_dtype)
    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
    )
    vc = VectorizationContext(ctx, 4, axis)

    i = TypedSymbol("i", ctx.index_dtype)

    ptr = TypedSymbol("ptr", create_type("float64 *"))

    #   Cannot vectorize assignment to LHS that does not depend on axis counter
    asm = typify(
        PsAssignment(
            factory.parse_sympy(mem_acc(ptr, 3 * i + 2)),
            factory.parse_sympy(x + y * mem_acc(ptr, ctr + 3)),
        )
    )

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(asm, vc)


def test_vectorize_mem_acc():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    typify = Typifier(ctx)
    vectorize = AstVectorizer(ctx)

    ctr = TypedSymbol("ctr", ctx.index_dtype)
    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
    )
    vc = VectorizationContext(ctx, 4, axis)

    i = TypedSymbol("i", ctx.index_dtype)
    j = TypedSymbol("j", ctx.index_dtype)

    ptr = TypedSymbol("ptr", create_type("float64 *"))

    #   Lane-invariant index
    acc = factory.parse_sympy(mem_acc(ptr, 3 * i + 5 * j))

    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecBroadcast)
    assert vec_acc.operand is not acc
    assert vec_acc.operand.structurally_equal(acc)

    #   Counter as index
    acc = factory.parse_sympy(mem_acc(ptr, ctr))
    assert isinstance(acc, PsMemAcc)

    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecMemAcc)
    assert vec_acc.pointer is not acc.pointer
    assert vec_acc.pointer.structurally_equal(acc.pointer)
    assert vec_acc.offset is not acc.offset
    assert vec_acc.offset.structurally_equal(acc.offset)
    assert vec_acc.stride is None
    assert vec_acc.vector_entries == 4

    #   Simple affine
    acc = factory.parse_sympy(mem_acc(ptr, 3 * i + 5 * ctr))
    assert isinstance(acc, PsMemAcc)

    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecMemAcc)
    assert vec_acc.pointer is not acc.pointer
    assert vec_acc.pointer.structurally_equal(acc.pointer)
    assert vec_acc.offset is not acc.offset
    assert vec_acc.offset.structurally_equal(acc.offset)
    assert vec_acc.stride.structurally_equal(factory.parse_index(5))
    assert vec_acc.vector_entries == 4

    #   More complex, nested affine
    acc = factory.parse_sympy(mem_acc(ptr, j * i + 2 * (5 + j * ctr) + 2 * ctr))
    assert isinstance(acc, PsMemAcc)

    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecMemAcc)
    assert vec_acc.pointer is not acc.pointer
    assert vec_acc.pointer.structurally_equal(acc.pointer)
    assert vec_acc.offset is not acc.offset
    assert vec_acc.offset.structurally_equal(acc.offset)
    assert vec_acc.stride.structurally_equal(factory.parse_index(2 * j + 2))
    assert vec_acc.vector_entries == 4

    #   Even more complex affine
    idx = -factory.parse_index(ctr) / factory.parse_index(i) - factory.parse_index(
        ctr
    ) * factory.parse_index(j)
    acc = typify(PsMemAcc(factory.parse_sympy(ptr), idx))
    assert isinstance(acc, PsMemAcc)

    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecMemAcc)
    assert vec_acc.pointer is not acc.pointer
    assert vec_acc.pointer.structurally_equal(acc.pointer)
    assert vec_acc.offset is not acc.offset
    assert vec_acc.offset.structurally_equal(acc.offset)
    assert vec_acc.stride.structurally_equal(
        factory.parse_index(-1) / factory.parse_index(i) - factory.parse_index(j)
    )
    assert vec_acc.vector_entries == 4

    #   Mixture of strides in affine and axis
    vc = VectorizationContext(
        ctx, 4, VectorizationAxis(ctx.get_symbol("ctr"), step=factory.parse_index(3))
    )

    acc = factory.parse_sympy(mem_acc(ptr, 3 * i + 5 * ctr))
    assert isinstance(acc, PsMemAcc)

    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecMemAcc)
    assert vec_acc.pointer is not acc.pointer
    assert vec_acc.pointer.structurally_equal(acc.pointer)
    assert vec_acc.offset is not acc.offset
    assert vec_acc.offset.structurally_equal(acc.offset)
    assert vec_acc.stride.structurally_equal(factory.parse_index(15))
    assert vec_acc.vector_entries == 4


def test_invalid_mem_acc():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    vectorize = AstVectorizer(ctx)

    ctr = TypedSymbol("ctr", ctx.index_dtype)
    axis = VectorizationAxis(
        ctx.get_symbol("ctr", ctx.index_dtype),
    )
    vc = VectorizationContext(ctx, 4, axis)

    i = TypedSymbol("i", ctx.index_dtype)
    j = TypedSymbol("j", ctx.index_dtype)
    ptr = TypedSymbol("ptr", create_type("float64 *"))

    #   Non-symbol pointer
    acc = factory.parse_sympy(
        mem_acc(AddressOf(mem_acc(ptr, 10)), 3 * i + ctr * (3 + ctr))
    )

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(acc, vc)

    #   Non-affine index
    acc = factory.parse_sympy(mem_acc(ptr, 3 * i + ctr * (3 + ctr)))

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(acc, vc)

    #   Non lane-invariant index
    vc.vectorize_symbol(ctx.get_symbol("j", ctx.index_dtype))

    acc = factory.parse_sympy(mem_acc(ptr, 3 * j + ctr))

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(acc, vc)


def test_vectorize_buffer_acc():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    vectorize = AstVectorizer(ctx)

    field = fields("f(3): [3D]", layout="fzyx")
    ispace = FullIterationSpace.create_with_ghost_layers(ctx, 0, archetype_field=field)
    ctx.set_iteration_space(ispace)

    ctr = ispace.dimensions_in_loop_order()[-1].counter

    axis = VectorizationAxis(ctr)
    vc = VectorizationContext(ctx, 4, axis)

    buf = ctx.get_buffer(field)

    acc = factory.parse_sympy(field[-1, -1, -1](2))

    #   Buffer strides are symbolic -> expect strided access
    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecMemAcc)
    assert vec_acc.stride is not None
    assert vec_acc.stride.structurally_equal(PsExpression.make(buf.strides[0]))

    #   Set buffer stride to one
    buf.strides[0] = PsConstant(1, dtype=ctx.index_dtype)

    #   Expect non-strided access
    vec_acc = vectorize.visit(acc, vc)
    assert isinstance(vec_acc, PsVecMemAcc)
    assert vec_acc.stride is None


def test_invalid_buffer_acc():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    vectorize = AstVectorizer(ctx)

    field = fields("field(3): [3D]", field_type=FieldType.CUSTOM)

    ctr, i, j = [TypedSymbol(n, ctx.index_dtype) for n in ("ctr", "i", "j")]

    axis = VectorizationAxis(ctx.get_symbol("ctr", ctx.index_dtype))
    vc = VectorizationContext(ctx, 4, axis)

    #   Counter occurs in more than one index
    acc = factory.parse_sympy(field.absolute_access((ctr, i, ctr + j), (1,)))

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(acc, vc)

    #   Counter occurs in index dimension
    acc = factory.parse_sympy(field.absolute_access((ctr, i, j), (ctr,)))

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(acc, vc)

    #   Counter occurs quadratically
    acc = factory.parse_sympy(field.absolute_access(((ctr + i) * ctr, i, j), (1,)))

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(acc, vc)


def test_vectorize_subscript():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    vectorize = AstVectorizer(ctx)

    ctr = ctx.get_symbol("ctr", ctx.index_dtype)

    axis = VectorizationAxis(ctr)
    vc = VectorizationContext(ctx, 4, axis)

    acc = PsSubscript(
        PsExpression.make(ctx.get_symbol("arr", PsArrayType(ctx.default_dtype, 42))),
        [PsExpression.make(ctx.get_symbol("i", ctx.index_dtype))],
    )  # independent of vectorization axis

    vec_acc = vectorize.visit(factory._typify(acc), vc)
    assert isinstance(vec_acc, PsVecBroadcast)
    assert isinstance(vec_acc.operand, PsSubscript)


def test_invalid_subscript():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    vectorize = AstVectorizer(ctx)

    ctr = ctx.get_symbol("ctr", ctx.index_dtype)

    axis = VectorizationAxis(ctr)
    vc = VectorizationContext(ctx, 4, axis)

    acc = PsSubscript(
        PsExpression.make(ctx.get_symbol("arr", PsArrayType(ctx.default_dtype, 42))),
        [PsExpression.make(ctr)],  # depends on vectorization axis
    )

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(factory._typify(acc), vc)


def test_vectorize_nested_loop():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    vectorize = AstVectorizer(ctx)

    ctr = ctx.get_symbol("i", ctx.index_dtype)

    axis = VectorizationAxis(ctr)
    vc = VectorizationContext(ctx, 4, axis)

    ast = factory.loop_nest(
        ("i", "j"),
        make_slice[:8, :8],  # inner loop does not depend on vectorization axis
        PsBlock(
            [
                PsDeclaration(
                    PsExpression.make(ctx.get_symbol("x", ctx.default_dtype)),
                    PsExpression.make(PsConstant(42, ctx.default_dtype)),
                )
            ]
        ),
    )

    vec_ast = vectorize.visit(ast, vc)
    inner_loop = next(
        dfs_preorder(
            vec_ast,
            lambda node: isinstance(node, PsLoop) and node.counter.symbol.name == "j",
        )
    )
    decl = inner_loop.body.statements[0]

    assert inner_loop.step.structurally_equal(
        PsExpression.make(PsConstant(1, ctx.index_dtype))
    )
    assert isinstance(decl.lhs.symbol.dtype, PsVectorType)


def test_invalid_nested_loop():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)
    vectorize = AstVectorizer(ctx)

    ctr = ctx.get_symbol("i", ctx.index_dtype)

    axis = VectorizationAxis(ctr)
    vc = VectorizationContext(ctx, 4, axis)

    ast = factory.loop_nest(
        ("i", "j"),
        make_slice[:8, :ctr],  # inner loop depends on vectorization axis
        PsBlock(
            [
                PsDeclaration(
                    PsExpression.make(ctx.get_symbol("x", ctx.default_dtype)),
                    PsExpression.make(PsConstant(42, ctx.default_dtype)),
                )
            ]
        ),
    )

    with pytest.raises(VectorizationError):
        _ = vectorize.visit(ast, vc)
