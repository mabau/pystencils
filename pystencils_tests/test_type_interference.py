from sympy.abc import a, b, c, d, e, f

import pystencils
from pystencils.typing import CastFunc, create_type


def test_type_interference():
    x = pystencils.fields('x:  float32[3d]')
    assignments = pystencils.AssignmentCollection({
        a: CastFunc(10, create_type('float64')),
        b: CastFunc(10, create_type('uint16')),
        e: 11,
        c: b,
        f: c + b,
        d: c + b + x.center + e,
        x.center: c + b + x.center
    })

    ast = pystencils.create_kernel(assignments)

    code = str(pystencils.get_code_str(ast))
    assert 'double a' in code
    assert 'uint16_t b' in code
    assert 'uint16_t f' in code
    assert 'int64_t e' in code
