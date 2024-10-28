import numpy as np
import sympy as sp
import pytest

from pystencils import (
    Assignment,
    Field,
    TypedSymbol,
    create_kernel,
    make_slice,
    Target,
    create_data_handling,
)
from pystencils.simp import sympy_cse_on_assignment_list


@pytest.mark.parametrize("target", [Target.CPU, Target.GPU])
def test_sliced_iteration(target):
    if target == Target.GPU:
        pytest.importorskip("cupy")

    size = (4, 4)

    dh = create_data_handling(size, default_target=target, default_ghost_layers=0)

    src_field = dh.add_array("src", 1)
    dst_field = dh.add_array("dst", 1)

    dh.fill(src_field.name, 1.0, ghost_layers=True)
    dh.fill(dst_field.name, 0.0, ghost_layers=True)

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

    s = make_slice[1:3, 1]
    kernel = create_kernel(
        sympy_cse_on_assignment_list([update_rule]), iteration_slice=s, target=target
    ).compile()

    if target == Target.GPU:
        dh.all_to_gpu()

    dh.run_kernel(kernel, a=1.0, b=1.0)

    if target == Target.GPU:
        dh.all_to_cpu()

    expected_result = np.zeros(size)
    expected_result[1:3, 1] = 1
    np.testing.assert_almost_equal(dh.gather_array(dst_field.name), expected_result)


@pytest.mark.parametrize("target", [Target.CPU, Target.GPU])
def test_symbols_in_slice(target):
    if target == Target.GPU:
        pytest.xfail("Iteration slices including arbitrary symbols are currently broken on GPU")

    size = (4, 4)

    dh = create_data_handling(size, default_target=target, default_ghost_layers=0)

    src_field = dh.add_array("src", 1)
    dst_field = dh.add_array("dst", 1)

    dh.fill(src_field.name, 1.0, ghost_layers=True)
    dh.fill(dst_field.name, 0.0, ghost_layers=True)

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
        sympy_cse_on_assignment_list([update_rule]), iteration_slice=s, target=target
    ).compile()

    if target == Target.GPU:
        dh.all_to_gpu()

    dh.run_kernel(kernel, a=1.0, b=1.0, x_end=x_end_value)

    if target == Target.GPU:
        dh.all_to_cpu()

    expected_result = np.zeros(size)
    expected_result[1:x_end_value, 1] = 1
    np.testing.assert_almost_equal(dh.gather_array(dst_field.name), expected_result)
