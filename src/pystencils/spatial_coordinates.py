import sympy
from pystencils.sympyextensions.typed_sympy import get_loop_counter_symbol


x_, y_, z_ = tuple(get_loop_counter_symbol(i) for i in range(3))
x_staggered, y_staggered, z_staggered = x_ + 0.5, y_ + 0.5, z_ + 0.5


def x_vector(ndim):
    return sympy.Matrix(tuple(get_loop_counter_symbol(i) for i in range(ndim)))


def x_staggered_vector(ndim):
    return sympy.Matrix(tuple(get_loop_counter_symbol(i) + 0.5 for i in range(ndim)))
