import pytest
import numpy as np
import sympy as sp

import pystencils as ps


@pytest.mark.parametrize('dtype', ["float64", "float32"])
def test_log(dtype):
    a = sp.Symbol("a")
    x = ps.fields(f'x: {dtype}[1d]')

    assignments = ps.AssignmentCollection({x.center(): sp.log(a)})

    ast = ps.create_kernel(assignments)
    code = ps.get_code_str(ast)
    kernel = ast.compile()

    # ps.show_code(ast)

    if dtype == "float64":
        assert "float" not in code

    array = np.zeros((10,), dtype=dtype)
    kernel(x=array, a=100)
    assert np.allclose(array, 4.60517019)
