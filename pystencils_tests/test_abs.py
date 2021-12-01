import pystencils.config
import sympy

import pystencils as ps
from pystencils.typing import CastFunc, create_type


def test_abs():
    x, y, z = ps.fields('x, y, z:  float64[2d]')

    default_int_type = create_type('int64')

    assignments = ps.AssignmentCollection({x[0, 0]: sympy.Abs(CastFunc(y[0, 0], default_int_type))})

    config = pystencils.config.CreateKernelConfig(target=ps.Target.GPU)
    ast = ps.create_kernel(assignments, config=config)
    code = ps.get_code_str(ast)
    print(code)
    assert 'fabs(' not in code
