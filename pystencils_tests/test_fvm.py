import sympy as sp
import pystencils as ps
import numpy as np
import pytest
from itertools import product


@pytest.mark.parametrize("dim", [2, 3])
def test_advection_diffusion(dim: int):
    # parameters
    if dim == 2:
        domain_size = (32, 32)
        flux_neighbors = 4
    elif dim == 3:
        domain_size = (16, 16, 16)
        flux_neighbors = 13

    dh = ps.create_data_handling(
        domain_size=domain_size, periodicity=True, default_target='cpu')

    n_field = dh.add_array('n', values_per_cell=1)
    j_field = dh.add_array('j', values_per_cell=flux_neighbors,
                           field_type=ps.FieldType.STAGGERED_FLUX)
    velocity_field = dh.add_array('v', values_per_cell=dim)

    D = 0.0666
    time = 200

    def grad(f):
        return sp.Matrix([ps.fd.diff(f, i) for i in range(dim)])

    flux_eq = - D * grad(n_field)
    fvm_eq = ps.fd.FVM1stOrder(n_field, flux=flux_eq)

    vof_adv = ps.fd.VOF(j_field, velocity_field, n_field)

    # merge calculation of advection and diffusion terms
    flux = []
    for adv, div in zip(vof_adv, fvm_eq.discrete_flux(j_field)):
        assert adv.lhs == div.lhs
        flux.append(ps.Assignment(adv.lhs, adv.rhs + div.rhs))

    flux_kernel = ps.create_staggered_kernel(flux).compile()

    pde_kernel = ps.create_kernel(
        fvm_eq.discrete_continuity(j_field)).compile()

    sync_conc = dh.synchronization_function([n_field.name])

    # analytical density calculation
    def density(pos: np.ndarray, time: int):
        return (4 * np.pi * D * time)**(-1.5) * \
            np.exp(-np.sum(np.square(pos), axis=dim) / (4 * D * time))

    pos = np.zeros((*domain_size, dim))
    xpos = np.arange(-domain_size[0] // 2, domain_size[0] // 2)
    ypos = np.arange(-domain_size[1] // 2, domain_size[1] // 2)

    if dim == 2:
        pos[..., 1], pos[..., 0] = np.meshgrid(xpos, ypos)
    elif dim == 3:
        zpos = np.arange(-domain_size[2] // 2, domain_size[2] // 2)
        pos[..., 2], pos[..., 1], pos[..., 0] = np.meshgrid(xpos, ypos, zpos)

    def run(velocity: np.ndarray, time: int):
        print(f"{velocity}, {time}")
        dh.fill(n_field.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(j_field.name, np.nan, ghost_layers=True, inner_ghost_layers=True)

        # set initial values for velocity and density
        for i in range(dim):
            dh.fill(velocity_field.name, velocity[i], i, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(n_field.name, 0)
        dh.fill(n_field.name, 1, slice_obj=ps.make_slice[[
                dom // 2 for dom in domain_size]])

        sync_conc()
        for i in range(time):
            dh.run_kernel(flux_kernel)
            dh.run_kernel(pde_kernel)
            sync_conc()

        calc_density = density(pos - velocity * time, time)

        np.testing.assert_allclose(dh.gather_array(
            n_field.name), calc_density, atol=1e-2, rtol=0)

    for vel in product([0, -0.08, 0.08], repeat=dim):
        run(np.array(vel), time)


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

# TODO: test source
