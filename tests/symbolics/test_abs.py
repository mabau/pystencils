import pytest

import pystencils as ps
import sympy


@pytest.mark.parametrize('target', (ps.Target.CPU, ps.Target.GPU))
def test_abs(target):
    if target == ps.Target.GPU:
        #   FIXME
        pytest.skip("GPU target not ready yet")

    x, y, z = ps.fields('x, y, z:  int64[2d]')

    assignments = ps.AssignmentCollection({x[0, 0]: sympy.Abs(y[0, 0])})

    config = ps.CreateKernelConfig(target=target)
    ast = ps.create_kernel(assignments, config=config)
    code = ps.get_code_str(ast)
    print(code)
    assert 'fabs(' not in code
