import pytest

import pystencils as ps
import numpy as np


# This test aims to trigger deprication warnings. Thus the warnings should not be displayed in the warning summary.
def test_create_kernel_backwards_compatibility():
    size = (30, 20)

    src_field_string = np.random.rand(*size)
    src_field_enum = np.copy(src_field_string)
    src_field_config = np.copy(src_field_string)
    dst_field_string = np.zeros(size)
    dst_field_enum = np.zeros(size)
    dst_field_config = np.zeros(size)

    f = ps.Field.create_from_numpy_array("f", src_field_enum)
    d = ps.Field.create_from_numpy_array("d", dst_field_enum)

    jacobi = ps.Assignment(d[0, 0], (f[1, 0] + f[-1, 0] + f[0, 1] + f[0, -1]) / 4)
    ast_enum = ps.create_kernel(jacobi, target=ps.Target.CPU).compile()
    with pytest.warns(DeprecationWarning):
        ast_string = ps.create_kernel(jacobi, target='cpu').compile()
    # noinspection PyTypeChecker
    with pytest.warns(DeprecationWarning):
        ast_config = ps.create_kernel(jacobi, config=ps.CreateKernelConfig(target='cpu')).compile()
    ast_enum(f=src_field_enum, d=dst_field_enum)
    ast_string(f=src_field_string, d=dst_field_string)
    ast_config(f=src_field_config, d=dst_field_config)

    error = np.sum(np.abs(dst_field_enum - dst_field_string))
    np.testing.assert_almost_equal(error, 0.0)
    error = np.sum(np.abs(dst_field_enum - dst_field_config))
    np.testing.assert_almost_equal(error, 0.0)
