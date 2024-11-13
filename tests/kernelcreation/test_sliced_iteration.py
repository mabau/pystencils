import numpy as np
import sympy as sp

from pystencils import Assignment, Field, TypedSymbol, create_kernel, make_slice
from pystencils.simp import sympy_cse_on_assignment_list


def test_sliced_iteration():
    size = (4, 4)
    src_arr = np.ones(size)
    dst_arr = np.zeros_like(src_arr)
    src_field = Field.create_from_numpy_array('src', src_arr)
    dst_field = Field.create_from_numpy_array('dst', dst_arr)

    a, b = sp.symbols("a b")
    update_rule = Assignment(dst_field[0, 0],
                             (a * src_field[0, 1] + a * src_field[0, -1] +
                              b * src_field[1, 0] + b * src_field[-1, 0]) / 4)

    x_end = TypedSymbol("x_end", "int")
    s = make_slice[1:x_end, 1]
    x_end_value = size[1] - 1
    kernel = create_kernel(sympy_cse_on_assignment_list([update_rule]), iteration_slice=s).compile()

    kernel(src=src_arr, dst=dst_arr, a=1.0, b=1.0, x_end=x_end_value)

    expected_result = np.zeros(size)
    expected_result[1:x_end_value, 1] = 1
    np.testing.assert_almost_equal(expected_result, dst_arr)
