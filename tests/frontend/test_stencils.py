import pystencils as ps
import sympy as sp

from pystencils.stencil import coefficient_list, plot_expression
import pystencils.plot as plt


def test_coefficient_list():
    f = ps.fields("f: double[1D]")
    expr = 2 * f[1] + 3 * f[-1]
    coff = coefficient_list(expr)
    assert coff == [3, 0, 2]
    figure = plt.figure()
    plot_expression(expr, matrix_form=True, figure=figure)

    f = ps.fields("f: double[3D]")
    expr = 2 * f[1, 0, 0] + 3 * f[0, -1, 0]
    coff = coefficient_list(expr)
    assert coff == [[[0, 3, 0], [0, 0, 2], [0, 0, 0]]]

    expr = 2 * f[1, 0, 0] + 3 * f[0, -1, 0] + 4 * f[0, 0, 1]
    coff = coefficient_list(expr, matrix_form=True)
    assert coff[0] == sp.zeros(3, 3)

    # in 3D plot only works if there are entries on every of the three 2D planes. In the above examples z-1 was empty
    expr = 2 * f[1, 0, 0] + 1 * f[0, -1, 0] + 1 * f[0, 0, 1] + f[0, 0, -1]
    figure = plt.figure()
    plot_expression(expr, figure=figure)


def test_plot_expression():
    f = ps.fields("f: double[2D]")
    figure = plt.figure()
    plot_expression(2 * f[1, 0] + 3 * f[0, -1], matrix_form=True, figure=figure)
