import pytest

import pystencils
from sympy import oo


@pytest.mark.parametrize('type', ('float32', 'float64', 'int64'))
@pytest.mark.parametrize('negative', (False, 'Negative'))
@pytest.mark.parametrize('target', ('cpu', 'gpu'))
def test_print_infinity(type, negative, target):

    x = pystencils.fields(f'x:  {type}[1d]')

    if negative:
        assignment = pystencils.Assignment(x.center, -oo)
    else:
        assignment = pystencils.Assignment(x.center, oo)
    ast = pystencils.create_kernel(assignment, data_type=type, target=target)

    if target == 'gpu':
        pytest.importorskip('pycuda')

    ast.compile()

    print(ast.compile().code)
