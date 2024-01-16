import pytest

import sympy as sp
import numpy
import pystencils
from pystencils.datahandling import create_data_handling


@pytest.mark.parametrize('dtype', ["float64", "float32"])
@pytest.mark.parametrize('sympy_function', [sp.Min, sp.Max])
def test_max(dtype, sympy_function):
    dh = create_data_handling(domain_size=(10, 10), periodicity=True)

    x = dh.add_array('x', values_per_cell=1, dtype=dtype)
    dh.fill("x", 0.0, ghost_layers=True)
    y = dh.add_array('y', values_per_cell=1, dtype=dtype)
    dh.fill("y", 1.0, ghost_layers=True)
    z = dh.add_array('z', values_per_cell=1, dtype=dtype)
    dh.fill("z", 2.0, ghost_layers=True)

    config = pystencils.CreateKernelConfig(default_number_float=dtype)

    # test sp.Max with one argument
    assignment_1 = pystencils.Assignment(x.center, sympy_function(y.center + 3.3))
    ast_1 = pystencils.create_kernel(assignment_1, config=config)
    kernel_1 = ast_1.compile()
    # pystencils.show_code(ast_1)

    # test sp.Max with two arguments
    assignment_2 = pystencils.Assignment(x.center, sympy_function(0.5, y.center - 1.5))
    ast_2 = pystencils.create_kernel(assignment_2, config=config)
    kernel_2 = ast_2.compile()
    # pystencils.show_code(ast_2)

    # test sp.Max with many arguments
    assignment_3 = pystencils.Assignment(x.center, sympy_function(z.center, 4.5, y.center - 1.5, y.center + z.center))
    ast_3 = pystencils.create_kernel(assignment_3, config=config)
    kernel_3 = ast_3.compile()
    # pystencils.show_code(ast_3)

    if sympy_function is sp.Max:
        results = [4.3, 0.5, 4.5]
    else:
        results = [4.3, -0.5, -0.5]

    dh.run_kernel(kernel_1)
    assert numpy.all(dh.gather_array('x') == results[0])
    dh.run_kernel(kernel_2)
    assert numpy.all(dh.gather_array('x') == results[1])
    dh.run_kernel(kernel_3)
    assert numpy.all(dh.gather_array('x') == results[2])


@pytest.mark.parametrize('dtype', ["int64", 'int32'])
@pytest.mark.parametrize('sympy_function', [sp.Min, sp.Max])
def test_max_integer(dtype, sympy_function):
    dh = create_data_handling(domain_size=(10, 10), periodicity=True)

    x = dh.add_array('x', values_per_cell=1, dtype=dtype)
    dh.fill("x", 0, ghost_layers=True)
    y = dh.add_array('y', values_per_cell=1, dtype=dtype)
    dh.fill("y", 1, ghost_layers=True)
    z = dh.add_array('z', values_per_cell=1, dtype=dtype)
    dh.fill("z", 2, ghost_layers=True)

    config = pystencils.CreateKernelConfig(default_number_int=dtype)

    # test sp.Max with one argument
    assignment_1 = pystencils.Assignment(x.center, sympy_function(y.center + 3))
    ast_1 = pystencils.create_kernel(assignment_1, config=config)
    kernel_1 = ast_1.compile()
    # pystencils.show_code(ast_1)

    # test sp.Max with two arguments
    assignment_2 = pystencils.Assignment(x.center, sympy_function(1, y.center - 1))
    ast_2 = pystencils.create_kernel(assignment_2, config=config)
    kernel_2 = ast_2.compile()
    # pystencils.show_code(ast_2)

    # test sp.Max with many arguments
    assignment_3 = pystencils.Assignment(x.center, sympy_function(z.center, 4, y.center - 1, y.center + z.center))
    ast_3 = pystencils.create_kernel(assignment_3, config=config)
    kernel_3 = ast_3.compile()
    # pystencils.show_code(ast_3)

    if sympy_function is sp.Max:
        results = [4, 1, 4]
    else:
        results = [4, 0, 0]

    dh.run_kernel(kernel_1)
    assert numpy.all(dh.gather_array('x') == results[0])
    dh.run_kernel(kernel_2)
    assert numpy.all(dh.gather_array('x') == results[1])
    dh.run_kernel(kernel_3)
    assert numpy.all(dh.gather_array('x') == results[2])
