import sympy as sp
import pystencils as ps
from pystencils.stencils import stencil_coefficients
from pystencils.fd.spatial import fd_stencils_standard, fd_stencils_isotropic, discretize_spatial
from pystencils.fd import diff


def test_spatial_2d_unit_sum():
    f = ps.fields("f: double[2D]")
    h = sp.symbols("h")

    terms = [diff(f, 0),
             diff(f, 1),
             diff(f, 0, 1),
             diff(f, 0, 0),
             diff(f, 1, 1),
             diff(f, 0, 0) + diff(f, 1, 1)]

    schemes = [fd_stencils_standard, fd_stencils_isotropic]

    for term in terms:
        for scheme in schemes:
            discretized = discretize_spatial(term, dx=h, stencil=scheme)
            _, coefficients = stencil_coefficients(discretized)
            assert sum(coefficients) == 0

def test_spatial_1d_unit_sum():
    f = ps.fields("f: double[1D]")
    h = sp.symbols("h")

    terms = [diff(f, 0),
             diff(f, 0, 0)]

    schemes = [fd_stencils_standard, fd_stencils_isotropic]

    for term in terms:
        for scheme in schemes:
            discretized = discretize_spatial(term, dx=h, stencil=scheme)
            _, coefficients = stencil_coefficients(discretized)
            assert sum(coefficients) == 0
