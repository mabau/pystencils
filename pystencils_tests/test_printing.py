import pytest
import re
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


def test_print_small_integer_pow():
    printer = pystencils.backends.cbackend.CBackend()

    x = sp.Symbol("x")
    y = sp.Symbol("y")
    n = pystencils.TypedSymbol("n", "int")
    t = pystencils.TypedSymbol("t", "float32")
    s = pystencils.TypedSymbol("s", "float32")

    equs = [
        pystencils.astnodes.SympyAssignment(y, 1/x),
        pystencils.astnodes.SympyAssignment(y, x*x),
        pystencils.astnodes.SympyAssignment(y, 1/(x*x)),
        pystencils.astnodes.SympyAssignment(y, x**8),
        pystencils.astnodes.SympyAssignment(y, x**(-8)),
        pystencils.astnodes.SympyAssignment(y, x**9),
        pystencils.astnodes.SympyAssignment(y, x**(-9)),
        pystencils.astnodes.SympyAssignment(y, x**n),
        pystencils.astnodes.SympyAssignment(y, sp.Pow(4, 4, evaluate=False)),
        pystencils.astnodes.SympyAssignment(y, x**0.25),
        pystencils.astnodes.SympyAssignment(y, x**y),
        pystencils.astnodes.SympyAssignment(y, pystencils.typing.cast_functions.CastFunc(1/x, "float32")),
        pystencils.astnodes.SympyAssignment(y, pystencils.typing.cast_functions.CastFunc(x*x, "float32")),
        pystencils.astnodes.SympyAssignment(y, (t+s)**(-8)),
        pystencils.astnodes.SympyAssignment(y, (t+s)**(-9)),
    ]
    typed = pystencils.typing.transformations.add_types(equs, pystencils.CreateKernelConfig())

    regexes = [
        r"1\.0\s*/\s*\(?\s*x\s*\)?",
        r"x\s*\*\s*x",
        r"1\.0\s*/\s*\(\s*x\s*\*x\s*\)",
        r"x(\s*\*\s*x){7}",
        r"1\.0\s*/\s*\(\s*x(\s*\*\s*x){7}\s*\)",
        r"pow\(\s*x\s*,\s*9(\.0)?\s*\)",
        r"pow\(\s*x\s*,\s*-9(\.0)?\s*\)",
        r"pow\(\s*x\s*,\s*\(?\s*\(\s*double\s*\)\s*\(\s*n\s*\)\s*\)?\s*\)",
        r"\(\s*int[a-zA-Z0-9_]*\s*\)\s*\(+\s*4(\s*\*\s*4){3}\s*\)+",
        r"pow\(\s*x\s*,\s*0\.25\s*\)",
        r"pow\(\s*x\s*,\s*y\s*\)",
        r"\(\s*float\s*\)[ ()]*1\.0\s*/\s*\(?\s*x\s*\)?",
        r"\(\s*float\s*\)[ ()]*x\s*\*\s*x",
        r"\(\s*float\s*\)\s*\(\s*1\.0f\s*/\s*\(\s*\(\s*s\s*\+\s*t\s*\)(\s*\*\s*\(\s*s\s*\+\s*t\s*\)){7}\s*\)",
        r"powf\(\s*s\s*\+\s*t\s*,\s*-9\.0f\s*\)",
    ]

    for r, e in zip(regexes, typed):
        assert re.search(r, printer(e))
