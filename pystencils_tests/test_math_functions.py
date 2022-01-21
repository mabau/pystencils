import pytest
import sympy as sp
import numpy as np
import pystencils as ps


@pytest.mark.parametrize('dtype', ["float64", "float32"])
@pytest.mark.parametrize('func', [sp.Pow, sp.atan2])
@pytest.mark.parametrize('target', [ps.Target.CPU, ps.Target.GPU])
def test_two_arguments(dtype, func, target):
    if target == ps.Target.GPU:
        pytest.importorskip("pycuda")
    dh = ps.create_data_handling(domain_size=(10, 10), periodicity=True, default_target=target)

    x = dh.add_array('x', values_per_cell=1, dtype=dtype)
    dh.fill("x", 0.0, ghost_layers=True)
    y = dh.add_array('y', values_per_cell=1, dtype=dtype)
    dh.fill("y", 1.0, ghost_layers=True)
    z = dh.add_array('z', values_per_cell=1, dtype=dtype)
    dh.fill("z", 2.0, ghost_layers=True)

    config = ps.CreateKernelConfig(target=target)

    # test sp.Max with one argument
    up = ps.Assignment(x.center, func(y.center, z.center))
    ast = ps.create_kernel(up, config=config)
    code = ps.get_code_str(ast)
    if dtype == 'float32':
        assert func.__name__.lower() in code
    kernel = ast.compile()

    dh.all_to_gpu()
    dh.run_kernel(kernel)
    dh.all_to_cpu()

    np.testing.assert_allclose(dh.gather_array("x")[0, 0], float(func(1.0, 2.0).evalf()))


@pytest.mark.parametrize('dtype', ["float64", "float32"])
@pytest.mark.parametrize('func', [sp.sin, sp.cos, sp.sinh, sp.cosh, sp.atan])
@pytest.mark.parametrize('target', [ps.Target.CPU, ps.Target.GPU])
def test_single_arguments(dtype, func, target):
    if target == ps.Target.GPU:
        pytest.importorskip("pycuda")
    dh = ps.create_data_handling(domain_size=(10, 10), periodicity=True, default_target=target)

    x = dh.add_array('x', values_per_cell=1, dtype=dtype)
    dh.fill("x", 0.0, ghost_layers=True)
    y = dh.add_array('y', values_per_cell=1, dtype=dtype)
    dh.fill("y", 1.0, ghost_layers=True)

    config = ps.CreateKernelConfig(target=target)

    # test sp.Max with one argument
    up = ps.Assignment(x.center, func(y.center))
    ast = ps.create_kernel(up, config=config)
    code = ps.get_code_str(ast)
    if dtype == 'float32':
        assert func.__name__.lower() in code
    kernel = ast.compile()

    dh.all_to_gpu()
    dh.run_kernel(kernel)
    dh.all_to_cpu()

    np.testing.assert_allclose(dh.gather_array("x")[0, 0], float(func(1.0).evalf()),
                               rtol=10**-3 if dtype == 'float32' else 10**-5)
