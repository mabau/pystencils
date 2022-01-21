import pytest
import sympy as sp
import numpy as np
import math
import pystencils as ps


@pytest.mark.parametrize('dtype', ["float64", "float32"])
def test_trigonometric_functions(dtype):
    dh = ps.create_data_handling(domain_size=(10, 10), periodicity=True)

    x = dh.add_array('x', values_per_cell=1, dtype=dtype)
    dh.fill("x", 0.0, ghost_layers=True)
    y = dh.add_array('y', values_per_cell=1, dtype=dtype)
    dh.fill("y", 1.0, ghost_layers=True)
    z = dh.add_array('z', values_per_cell=1, dtype=dtype)
    dh.fill("z", 2.0, ghost_layers=True)

    # config = pystencils.CreateKernelConfig(default_number_float=dtype)

    # test sp.Max with one argument
    up = ps.Assignment(x.center, sp.atan2(y.center, z.center))
    ast = ps.create_kernel(up)
    code = ps.get_code_str(ast)
    kernel = ast.compile()
    dh.run_kernel(kernel)

    np.testing.assert_allclose(dh.gather_array("x")[0, 0], math.atan2(1.0, 2.0))
