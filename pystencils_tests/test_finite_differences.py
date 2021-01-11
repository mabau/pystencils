import sympy as sp
import pytest

import pystencils as ps
from pystencils.astnodes import LoopOverCoordinate
from pystencils.fd import diff, diffusion, Discretization2ndOrder
from pystencils.fd.spatial import discretize_spatial, fd_stencils_isotropic, fd_stencils_standard, \
    fd_stencils_forth_order_isotropic


def test_spatial_2d_unit_sum():
    f = ps.fields("f: double[2D]")
    h = sp.symbols("h")

    terms = [diff(f, 0),
             diff(f, 1),
             diff(f, 0, 1),
             diff(f, 0, 0),
             diff(f, 1, 1),
             diff(f, 0, 0) + diff(f, 1, 1)]

    schemes = [fd_stencils_standard, fd_stencils_isotropic, 'standard', 'isotropic']

    for term in terms:
        for scheme in schemes:
            discretized = discretize_spatial(term, dx=h, stencil=scheme)
            _, coefficients = ps.stencil.coefficients(discretized)
            assert sum(coefficients) == 0


def test_spatial_1d_unit_sum():
    f = ps.fields("f: double[1D]")
    h = sp.symbols("h")

    terms = [diff(f, 0),
             diff(f, 0, 0)]

    schemes = [fd_stencils_standard, fd_stencils_isotropic, 'standard', 'isotropic']

    for term in terms:
        for scheme in schemes:
            discretized = discretize_spatial(term, dx=h, stencil=scheme)
            _, coefficients = ps.stencil.coefficients(discretized)
            assert sum(coefficients) == 0


def test_fd_stencils_forth_order_isotropic():
    f = ps.fields("f: double[2D]")
    a = fd_stencils_forth_order_isotropic([0], 1, f[0, 0](0))
    sten, coefficients = ps.stencil.coefficients(a)
    assert sum(coefficients) == 0

    for i, direction in enumerate(sten):
        counterpart = sten.index((direction[0] * -1, direction[1] * -1))
        assert coefficients[i] + coefficients[counterpart] == 0


def test_staggered_laplacian():
    f = ps.fields("f : double[2D]")
    a, dx = sp.symbols("a, dx")

    factored_version = sum(ps.fd.Diff(a * ps.fd.Diff(f[0, 0], i), i)
                           for i in range(2))
    expanded = ps.fd.expand_diff_full(factored_version, constants=[a])

    reference = ps.fd.discretize_spatial(expanded, dx).factor()
    to_test = ps.fd.discretize_spatial_staggered(factored_version, dx).factor()
    assert reference == to_test


def test_staggered_combined():
    from pystencils.fd import diff
    f = ps.fields("f : double[2D]")
    x, y = [LoopOverCoordinate.get_loop_counter_symbol(i) for i in range(2)]
    dx = sp.symbols("dx")

    expr = diff(x * diff(f, 0) + y * diff(f, 1), 0)

    right = (x + sp.Rational(1, 2)) * (f[1, 0] - f[0, 0]) + y * (f[1, 1] - f[1, -1] + f[0, 1] - f[0, -1]) / 4
    left = (x - sp.Rational(1, 2)) * (f[0, 0] - f[-1, 0]) + y * (f[-1, 1] - f[-1, -1] + f[0, 1] - f[0, -1]) / 4
    reference = (right - left) / (dx ** 2)

    to_test = ps.fd.discretize_spatial_staggered(expr, dx)
    assert sp.expand(reference - to_test) == 0


def test_diffusion():
    f = ps.fields("f(3): [2D]")
    d = sp.Symbol("d")
    dx = sp.Symbol("dx")
    idx = 2
    diffusion_term = diffusion(scalar=f, diffusion_coeff=sp.Symbol("d"), idx=idx)
    discretization = Discretization2ndOrder()
    expected_output = ((f[-1, 0](idx) + f[0, -1](idx) - 4 * f[0, 0](idx) + f[0, 1](idx) + f[1, 0](idx)) * d) / dx ** 2
    assert sp.simplify(discretization(diffusion_term) - expected_output) == 0

    with pytest.raises(ValueError):
        diffusion(scalar=d, diffusion_coeff=sp.Symbol("d"), idx=idx)
