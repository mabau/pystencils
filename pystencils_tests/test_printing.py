import pytest
import sympy as sp

import pystencils
from pystencils.backends.cbackend import CBackend


class UnsupportedNode(pystencils.astnodes.Node):

    def __init__(self):
        super().__init__()


@pytest.mark.parametrize('type', ('float32', 'float64', 'int64'))
@pytest.mark.parametrize('negative', (False, 'Negative'))
@pytest.mark.parametrize('target', (pystencils.Target.CPU, pystencils.Target.GPU))
def test_print_infinity(type, negative, target):

    x = pystencils.fields(f'x:  {type}[1d]')

    if negative:
        assignment = pystencils.Assignment(x.center, -sp.oo)
    else:
        assignment = pystencils.Assignment(x.center, sp.oo)
    ast = pystencils.create_kernel(assignment, data_type=type, target=target)

    if target == pystencils.Target.GPU:
        pytest.importorskip('cupy')

    ast.compile()

    print(ast.compile().code)


def test_print_unsupported_node():
    with pytest.raises(NotImplementedError, match='CBackend does not support node of type UnsupportedNode'):
        CBackend()(UnsupportedNode())


@pytest.mark.parametrize('dtype', ('float32', 'float64'))
@pytest.mark.parametrize('target', (pystencils.Target.CPU, pystencils.Target.GPU))
def test_print_subtraction(dtype, target):
    a, b, c = sp.symbols("a b c")

    x = pystencils.fields(f'x: {dtype}[3d]')
    y = pystencils.fields(f'y: {dtype}[3d]')

    config = pystencils.CreateKernelConfig(target=target, data_type=dtype)
    update = pystencils.Assignment(x.center, y.center - a * b ** 8 + b * -1 / 42.0 - 2 * c ** 4)
    ast = pystencils.create_kernel(update, config=config)

    code = pystencils.get_code_str(ast)
    assert "-1.0" not in code
