import pytest

from pystencils import make_slice, Field, create_type
from pystencils.sympyextensions.typed_sympy import TypedSymbol

from pystencils.backend.constants import PsConstant
from pystencils.backend.kernelcreation import KernelCreationContext, FullIterationSpace
from pystencils.backend.ast.expressions import PsAdd, PsConstantExpr, PsExpression
from pystencils.backend.kernelcreation.typification import TypificationError
from pystencils.types.quick import Int


def test_slices():
    ctx = KernelCreationContext()

    archetype_field = Field.create_generic("f", spatial_dimensions=3, layout="fzyx")
    ctx.add_field(archetype_field)

    islice = (slice(1, -1, 1), slice(3, -3, 3), slice(0, None, -1))
    ispace = FullIterationSpace.create_from_slice(ctx, islice, archetype_field)

    archetype_arr = ctx.get_array(archetype_field)

    dims = ispace.dimensions

    for sl, size, dim in zip(islice, archetype_arr.shape, dims):
        assert (
            isinstance(dim.start, PsConstantExpr)
            and dim.start.constant.value == sl.start
        )
        assert (
            isinstance(dim.step, PsConstantExpr) and dim.step.constant.value == sl.step
        )

    assert isinstance(dims[0].stop, PsAdd) and any(
        op.structurally_equal(PsExpression.make(archetype_arr.shape[0]))
        for op in dims[0].stop.children
    )

    assert isinstance(dims[1].stop, PsAdd) and any(
        op.structurally_equal(PsExpression.make(archetype_arr.shape[1]))
        for op in dims[1].stop.children
    )

    assert dims[2].stop.structurally_equal(PsExpression.make(archetype_arr.shape[2]))


def test_invalid_slices():
    ctx = KernelCreationContext()

    archetype_field = Field.create_generic("f", spatial_dimensions=1, layout="fzyx")
    ctx.add_field(archetype_field)

    islice = (slice(1, -1, 0.5),)
    with pytest.raises(TypeError):
        FullIterationSpace.create_from_slice(ctx, islice, archetype_field)

    islice = (slice(1, -1, TypedSymbol("w", dtype=create_type("double"))),)
    with pytest.raises(TypificationError):
        FullIterationSpace.create_from_slice(ctx, islice, archetype_field)


def test_iteration_count():
    ctx = KernelCreationContext()

    i, j, k = [PsExpression.make(ctx.get_symbol(x, ctx.index_dtype)) for x in "ijk"]
    zero = PsExpression.make(PsConstant(0, ctx.index_dtype))
    two = PsExpression.make(PsConstant(2, ctx.index_dtype))
    three = PsExpression.make(PsConstant(3, ctx.index_dtype))

    ispace = FullIterationSpace.create_from_slice(
        ctx, make_slice[three : i-two, 1:8:3]
    )

    iters = [ispace.actual_iterations(coord) for coord in range(2)]
    assert iters[0].structurally_equal((i - two) - three)
    assert iters[1].structurally_equal(three)

    empty_ispace = FullIterationSpace.create_from_slice(
        ctx, make_slice[4:4:1, 4:4:7]
    )

    iters = [empty_ispace.actual_iterations(coord) for coord in range(2)]
    assert iters[0].structurally_equal(zero)
    assert iters[1].structurally_equal(zero)
