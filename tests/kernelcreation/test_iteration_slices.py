import numpy as np
import sympy as sp
import pytest

from dataclasses import replace

from pystencils import (
    DEFAULTS,
    Assignment,
    Field,
    TypedSymbol,
    create_kernel,
    make_slice,
    Target,
    CreateKernelConfig,
    DynamicType,
)
from pystencils.sympyextensions.integer_functions import int_rem
from pystencils.simp import sympy_cse_on_assignment_list
from pystencils.slicing import normalize_slice
from pystencils.jit.gpu_cupy import CupyKernelWrapper


def test_sliced_iteration():
    size = (4, 4)
    src_arr = np.ones(size)
    dst_arr = np.zeros_like(src_arr)
    src_field = Field.create_from_numpy_array("src", src_arr)
    dst_field = Field.create_from_numpy_array("dst", dst_arr)

    a, b = sp.symbols("a b")
    update_rule = Assignment(
        dst_field[0, 0],
        (
            a * src_field[0, 1]
            + a * src_field[0, -1]
            + b * src_field[1, 0]
            + b * src_field[-1, 0]
        )
        / 4,
    )

    x_end = TypedSymbol("x_end", "int")
    s = make_slice[1:x_end, 1]
    x_end_value = size[1] - 1
    kernel = create_kernel(
        sympy_cse_on_assignment_list([update_rule]), iteration_slice=s
    ).compile()

    kernel(src=src_arr, dst=dst_arr, a=1.0, b=1.0, x_end=x_end_value)

    expected_result = np.zeros(size)
    expected_result[1:x_end_value, 1] = 1
    np.testing.assert_almost_equal(expected_result, dst_arr)


@pytest.mark.parametrize(
    "islice",
    [
        make_slice[1:-1, 1:-1],
        make_slice[3, 2:-2],
        make_slice[2:-2:2, ::3],
        make_slice[10:, :-5:2],
        make_slice[-5:-1, -1],
        make_slice[-3, -1]
    ],
)
def test_numerical_slices(gen_config: CreateKernelConfig, xp, islice):
    shape = (16, 16)

    f_arr = xp.zeros(shape)
    expected = xp.zeros_like(f_arr)
    expected[islice] = 1.0

    f = Field.create_from_numpy_array("f", f_arr)

    update = Assignment(f.center(), 1)
    gen_config = replace(gen_config, iteration_slice=islice)

    try:
        kernel = create_kernel(update, gen_config).compile()
    except NotImplementedError:
        if gen_config.get_target().is_vector_cpu():
            #   TODO Gather/Scatter not implemented yet
            pytest.xfail("Gather/Scatter not available yet")

    kernel(f=f_arr)

    xp.testing.assert_array_equal(f_arr, expected)


def test_symbolic_slice(gen_config: CreateKernelConfig, xp):
    shape = (16, 16)

    sx, sy, ex, ey = [
        TypedSymbol(n, DynamicType.INDEX_TYPE) for n in ("sx", "sy", "ex", "ey")
    ]

    f_arr = xp.zeros(shape)

    f = Field.create_from_numpy_array("f", f_arr)

    update = Assignment(f.center(), 1)
    islice = make_slice[sy:ey, sx:ex]
    gen_config = replace(gen_config, iteration_slice=islice)

    print(repr(gen_config))

    kernel = create_kernel(update, gen_config).compile()

    for slic in [make_slice[:, :], make_slice[1:-1, 2:-2], make_slice[8:14, 7:11]]:
        slic = normalize_slice(slic, shape)
        expected = xp.zeros_like(f_arr)
        expected[slic] = 1.0

        f_arr[:] = 0.0

        kernel(
            f=f_arr,
            sy=slic[0].start,
            ey=slic[0].stop,
            sx=slic[1].start,
            ex=slic[1].stop,
        )

        xp.testing.assert_array_equal(f_arr, expected)


def test_triangle_pattern(gen_config: CreateKernelConfig, xp):
    shape = (16, 16)

    f_arr = xp.zeros(shape)
    f = Field.create_from_numpy_array("f", f_arr)

    expected = xp.zeros_like(f_arr)
    for r in range(shape[0]):
        expected[r, r:] = 1.0

    update = Assignment(f.center(), 1)
    outer_counter = DEFAULTS.spatial_counters[0]
    islice = make_slice[:, outer_counter:]
    gen_config = replace(gen_config, iteration_slice=islice)

    if gen_config.target == Target.CUDA:
        gen_config.gpu.manual_launch_grid = True

    kernel = create_kernel(update, gen_config).compile()

    if isinstance(kernel, CupyKernelWrapper):
        kernel.block_size = shape + (1,)
        kernel.num_blocks = (1, 1, 1)

    kernel(f=f_arr)

    xp.testing.assert_array_equal(f_arr, expected)


def test_red_black_pattern(gen_config: CreateKernelConfig, xp):
    shape = (16, 16)

    f_arr = xp.zeros(shape)
    f = Field.create_from_numpy_array("f", f_arr)

    expected = xp.zeros_like(f_arr)
    for r in range(shape[0]):
        start = 0 if r % 2 == 0 else 1
        expected[r, start::2] = 1.0

    update = Assignment(f.center(), 1)
    outer_counter = DEFAULTS.spatial_counters[0]
    start = sp.Piecewise((0, sp.Eq(int_rem(outer_counter, 2), 0)), (1, True))
    islice = make_slice[:, start::2]
    gen_config.iteration_slice = islice

    if gen_config.target == Target.CUDA:
        gen_config.gpu.manual_launch_grid = True

    try:
        kernel = create_kernel(update, gen_config).compile()
    except NotImplementedError:
        if gen_config.get_target().is_vector_cpu():
            pytest.xfail("Gather/Scatter not implemented yet")

    if isinstance(kernel, CupyKernelWrapper):
        kernel.block_size = (8, 16, 1)
        kernel.num_blocks = (1, 1, 1)

    kernel(f=f_arr)

    xp.testing.assert_array_equal(f_arr, expected)
