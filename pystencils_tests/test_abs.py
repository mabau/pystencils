import sympy

import pystencils
from pystencils.data_types import cast_func, create_type


def test_abs():
    x, y, z = pystencils.fields('x, y, z:  float64[2d]')

    default_int_type = create_type('int64')

    assignments = pystencils.AssignmentCollection({
        x[0, 0]: sympy.Abs(cast_func(y[0, 0], default_int_type))
    })

    ast = pystencils.create_kernel(assignments, target="gpu")
    code = pystencils.get_code_str(ast)
    print(code)
    assert 'fabs(' not in code
