import sympy
from .defaults import DEFAULTS


x_, y_, z_ = DEFAULTS.spatial_counters
x_staggered, y_staggered, z_staggered = x_ + 0.5, y_ + 0.5, z_ + 0.5


def x_vector(ndim):
    return sympy.Matrix(DEFAULTS.spatial_counters[:ndim])


def x_staggered_vector(ndim):
    return sympy.Matrix(tuple(DEFAULTS.spatial_counters[i] + 0.5 for i in range(ndim)))
