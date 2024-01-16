import pytest
import numpy as np

import pystencils as ps
from pystencils import Field, Assignment, create_kernel
from pystencils.bit_masks import flag_cond


@pytest.mark.parametrize('mask_type', [np.uint8, np.uint16, np.uint32, np.uint64])
def test_flag_condition(mask_type):
    f_arr = np.zeros((2, 2, 2), dtype=np.float64)
    mask_arr = np.zeros((2, 2), dtype=mask_type)

    mask_arr[0, 1] = (1 << 3)
    mask_arr[1, 0] = (1 << 5)
    mask_arr[1, 1] = (1 << 3) + (1 << 5)

    f = Field.create_from_numpy_array('f', f_arr, index_dimensions=1)
    mask = Field.create_from_numpy_array('mask', mask_arr)

    v1 = 42.3
    v2 = 39.7
    v3 = 119

    assignments = [
        Assignment(f(0), flag_cond(3, mask(0), v1)),
        Assignment(f(1), flag_cond(5, mask(0), v2, v3))
    ]

    kernel = create_kernel(assignments).compile()
    kernel(f=f_arr, mask=mask_arr)
    code = ps.get_code_str(kernel)
    assert '119.0' in code

    reference = np.zeros((2, 2, 2), dtype=np.float64)
    reference[0, 1, 0] = v1
    reference[1, 1, 0] = v1

    reference[0, 0, 1] = v3
    reference[0, 1, 1] = v3

    reference[1, 0, 1] = v2
    reference[1, 1, 1] = v2

    np.testing.assert_array_equal(f_arr, reference)
