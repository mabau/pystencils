import pytest
import numpy as np

from pystencils import make_slice, Field, create_type
from pystencils.sympyextensions.typed_sympy import TypedSymbol

from pystencils.backend.constants import PsConstant
from pystencils.backend.kernelcreation import (
    KernelCreationContext,
    FullIterationSpace,
    AstFactory,
)
from pystencils.backend.ast.expressions import PsAdd, PsConstantExpr, PsExpression
from pystencils.backend.kernelcreation.typification import TypificationError


def test_slices_over_field():
    ctx = KernelCreationContext()

    archetype_field = Field.create_generic("f", spatial_dimensions=3, layout="fzyx")
    ctx.add_field(archetype_field)

    islice = (slice(1, -1, 1), slice(3, -3, 3), slice(0, None, 1))
    ispace = FullIterationSpace.create_from_slice(ctx, islice, archetype_field)

    archetype_arr = ctx.get_buffer(archetype_field)

    dims = ispace.dimensions

    for sl, dim in zip(islice, dims):
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


def test_slices_with_fixed_size_field():
    ctx = KernelCreationContext()

    archetype_field = Field.create_fixed_size("f", (4, 5, 6), layout="fzyx")
    ctx.add_field(archetype_field)

    islice = (slice(1, -1, 1), slice(3, -3, 3), slice(0, None, 1))
    ispace = FullIterationSpace.create_from_slice(ctx, islice, archetype_field)

    archetype_arr = ctx.get_buffer(archetype_field)

    dims = ispace.dimensions

    for sl, size, dim in zip(islice, archetype_arr.shape, dims):
        assert (
            isinstance(dim.start, PsConstantExpr)
            and dim.start.constant.value == sl.start
        )

        assert isinstance(size, PsConstant)

        assert isinstance(
            dim.stop, PsConstantExpr
        ) and dim.stop.constant.value == np.int64(
            size.value + sl.stop if sl.stop is not None else size.value
        )

        assert (
            isinstance(dim.step, PsConstantExpr) and dim.step.constant.value == sl.step
        )


def test_singular_slice_over_field():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    archetype_field = Field.create_generic("f", spatial_dimensions=2, layout="fzyx")
    ctx.add_field(archetype_field)
    archetype_arr = ctx.get_buffer(archetype_field)

    islice = (4, -3)
    ispace = FullIterationSpace.create_from_slice(ctx, islice, archetype_field)

    dims = ispace.dimensions

    assert dims[0].start.structurally_equal(factory.parse_index(4))

    assert dims[0].stop.structurally_equal(factory.parse_index(5))

    assert dims[1].start.structurally_equal(
        PsExpression.make(archetype_arr.shape[1]) + factory.parse_index(-3)
    )

    assert dims[1].stop.structurally_equal(
        PsExpression.make(archetype_arr.shape[1]) + factory.parse_index(-2)
    )


def test_slices_with_negative_start():
    ctx = KernelCreationContext()
    factory = AstFactory(ctx)

    archetype_field = Field.create_generic("f", spatial_dimensions=2, layout="fzyx")
    ctx.add_field(archetype_field)
    archetype_arr = ctx.get_buffer(archetype_field)

    islice = (slice(-3, -1, 1), slice(-4, None, 1))
    ispace = FullIterationSpace.create_from_slice(ctx, islice, archetype_field)

    dims = ispace.dimensions

    assert dims[0].start.structurally_equal(
        PsExpression.make(archetype_arr.shape[0]) + factory.parse_index(-3)
    )

    assert dims[1].start.structurally_equal(
        PsExpression.make(archetype_arr.shape[1]) + factory.parse_index(-4)
    )


def test_field_independent_slices():
    ctx = KernelCreationContext()

    islice = (slice(-3, -1, 1), slice(-4, 7, 2))
    ispace = FullIterationSpace.create_from_slice(ctx, islice)

    dims = ispace.dimensions

    for sl, dim in zip(islice, dims):
        assert isinstance(dim.start, PsConstantExpr)
        assert dim.start.constant.value == np.int64(sl.start)

        assert isinstance(dim.stop, PsConstantExpr)
        assert dim.stop.constant.value == np.int64(sl.stop)

        assert isinstance(dim.step, PsConstantExpr)
        assert dim.step.constant.value == np.int64(sl.step)


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

    islice = (slice(1, 3, 0),)
    with pytest.raises(ValueError):
        FullIterationSpace.create_from_slice(ctx, islice, archetype_field)

    islice = (slice(1, 3, -1),)
    with pytest.raises(ValueError):
        FullIterationSpace.create_from_slice(ctx, islice, archetype_field)


def test_iteration_count():
    ctx = KernelCreationContext()

    i, j, k = [PsExpression.make(ctx.get_symbol(x, ctx.index_dtype)) for x in "ijk"]
    zero = PsExpression.make(PsConstant(0, ctx.index_dtype))
    two = PsExpression.make(PsConstant(2, ctx.index_dtype))
    three = PsExpression.make(PsConstant(3, ctx.index_dtype))

    ispace = FullIterationSpace.create_from_slice(
        ctx, make_slice[three : i - two, 1:8:3]
    )

    iters = [ispace.actual_iterations(coord) for coord in range(2)]
    assert iters[0].structurally_equal((i - two) - three)
    assert iters[1].structurally_equal(three)

    empty_ispace = FullIterationSpace.create_from_slice(ctx, make_slice[4:4:1, 4:4:7])

    iters = [empty_ispace.actual_iterations(coord) for coord in range(2)]
    assert iters[0].structurally_equal(zero)
    assert iters[1].structurally_equal(zero)
