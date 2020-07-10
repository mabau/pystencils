import sympy
import numpy
import pystencils
from pystencils.datahandling import create_data_handling


def test_max():
    dh = create_data_handling(domain_size=(10, 10), periodicity=True)

    x = dh.add_array('x', values_per_cell=1)
    dh.fill("x", 0.0, ghost_layers=True)
    y = dh.add_array('y', values_per_cell=1)
    dh.fill("y", 1.0, ghost_layers=True)
    z = dh.add_array('z', values_per_cell=1)
    dh.fill("z", 2.0, ghost_layers=True)

    # test sp.Max with one argument
    assignment_1 = pystencils.Assignment(x.center, sympy.Max(y.center + 3.3))
    ast_1 = pystencils.create_kernel(assignment_1)
    kernel_1 = ast_1.compile()

    # test sp.Max with two arguments
    assignment_2 = pystencils.Assignment(x.center, sympy.Max(0.5, y.center - 1.5))
    ast_2 = pystencils.create_kernel(assignment_2)
    kernel_2 = ast_2.compile()

    # test sp.Max with many arguments
    assignment_3 = pystencils.Assignment(x.center, sympy.Max(z.center, 4.5, y.center - 1.5, y.center + z.center))
    ast_3 = pystencils.create_kernel(assignment_3)
    kernel_3 = ast_3.compile()

    dh.run_kernel(kernel_1)
    assert numpy.all(dh.cpu_arrays["x"] == 4.3)
    dh.run_kernel(kernel_2)
    assert numpy.all(dh.cpu_arrays["x"] == 0.5)
    dh.run_kernel(kernel_3)
    assert numpy.all(dh.cpu_arrays["x"] == 4.5)


def test_min():
    dh = create_data_handling(domain_size=(10, 10), periodicity=True)

    x = dh.add_array('x', values_per_cell=1)
    dh.fill("x", 0.0, ghost_layers=True)
    y = dh.add_array('y', values_per_cell=1)
    dh.fill("y", 1.0, ghost_layers=True)
    z = dh.add_array('z', values_per_cell=1)
    dh.fill("z", 2.0, ghost_layers=True)

    # test sp.Min with one argument
    assignment_1 = pystencils.Assignment(x.center, sympy.Min(y.center + 3.3))
    ast_1 = pystencils.create_kernel(assignment_1)
    kernel_1 = ast_1.compile()

    # test sp.Min with two arguments
    assignment_2 = pystencils.Assignment(x.center, sympy.Min(0.5, y.center - 1.5))
    ast_2 = pystencils.create_kernel(assignment_2)
    kernel_2 = ast_2.compile()

    # test sp.Min with many arguments
    assignment_3 = pystencils.Assignment(x.center, sympy.Min(z.center, 4.5, y.center - 1.5, y.center + z.center))
    ast_3 = pystencils.create_kernel(assignment_3)
    kernel_3 = ast_3.compile()

    dh.run_kernel(kernel_1)
    assert numpy.all(dh.cpu_arrays["x"] == 4.3)
    dh.run_kernel(kernel_2)
    assert numpy.all(dh.cpu_arrays["x"] == - 0.5)
    dh.run_kernel(kernel_3)
    assert numpy.all(dh.cpu_arrays["x"] == - 0.5)
