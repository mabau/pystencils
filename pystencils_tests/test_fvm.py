import sympy as sp
import pystencils as ps


def test_ek():

    # parameters

    L = (40, 40)
    D = sp.Symbol("D")
    z = sp.Symbol("z")

    # data structures

    dh = ps.create_data_handling(L, periodicity=True, default_target='cpu')
    c = dh.add_array('c', values_per_cell=1)
    v = dh.add_array('v', values_per_cell=dh.dim)
    j = dh.add_array('j', values_per_cell=dh.dim * 2, field_type=ps.FieldType.STAGGERED_FLUX)
    Phi = dh.add_array('Φ', values_per_cell=1)

    # perform automatic discretization

    def Gradient(f):
        return sp.Matrix([ps.fd.diff(f, i) for i in range(dh.dim)])

    flux_eq = -D * Gradient(c) + D * z * c.center * Gradient(Phi)

    disc = ps.fd.FVM1stOrder(c, flux_eq)
    flux_assignments = disc.discrete_flux(j)
    advection_assignments = ps.fd.VOF(j, v, c)
    continuity_assignments = disc.discrete_continuity(j)

    # manual discretization

    x_staggered = - c[-1, 0] + c[0, 0] + z * (c[-1, 0] + c[0, 0]) / 2 * (Phi[-1, 0] - Phi[0, 0])
    y_staggered = - c[0, -1] + c[0, 0] + z * (c[0, -1] + c[0, 0]) / 2 * (Phi[0, -1] - Phi[0, 0])
    xy_staggered = - c[-1, -1] + c[0, 0] + z * (c[-1, -1] + c[0, 0]) / 2 * (Phi[-1, -1] - Phi[0, 0])
    xY_staggered = - c[-1, 1] + c[0, 0] + z * (c[-1, 1] + c[0, 0]) / 2 * (Phi[-1, 1] - Phi[0, 0])

    jj = j.staggered_access
    divergence = -1 / (1 + sp.sqrt(2) if j.index_shape[0] == 4 else 1) * \
        sum([jj(d) / sp.Matrix(ps.stencil.direction_string_to_offset(d)).norm() for d in j.staggered_stencil
            + [ps.stencil.inverse_direction_string(d) for d in j.staggered_stencil]])

    update = [ps.Assignment(c.center, c.center + divergence)]
    flux = [ps.Assignment(j.staggered_access("W"), D * x_staggered),
            ps.Assignment(j.staggered_access("S"), D * y_staggered)]
    if j.index_shape[0] == 4:
        flux += [ps.Assignment(j.staggered_access("SW"), D * xy_staggered),
                 ps.Assignment(j.staggered_access("NW"), D * xY_staggered)]

    # compare

    for a, b in zip(flux, flux_assignments):
        assert a.lhs == b.lhs
        assert sp.simplify(a.rhs - b.rhs) == 0
    for a, b in zip(update, continuity_assignments):
        assert a.lhs == b.lhs
        assert a.rhs == b.rhs

    # TODO: test advection and source
