from sympy.abc import a, b, c, d, e, f, g

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
        x.center: c + b + x.center,
        g: a + b + d
    })

    ast = pystencils.create_kernel(assignments)
    code = pystencils.get_code_str(ast)
    # print(code)

    assert 'const double a' in code
    assert 'const uint16_t b' in code
    assert 'const uint16_t f' in code
    assert 'const int64_t e' in code

    assert 'const float d = ((float)(b)) + ((float)(c)) + ((float)(e)) + _data_x_00_10[_stride_x_2*ctr_2];' in code
    assert '_data_x_00_10[_stride_x_2*ctr_2] = ((float)(b)) + ((float)(c)) + _data_x_00_10[_stride_x_2*ctr_2];' in code
    assert 'const double g = a + ((double)(b)) + ((double)(d));' in code
