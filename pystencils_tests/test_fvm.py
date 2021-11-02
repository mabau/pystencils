import sympy as sp
import pystencils as ps
import numpy as np
import pytest
from itertools import product


def advection_diffusion(dim: int):
    # parameters
    if dim == 2:
        L = (32, 32)
    elif dim == 3:
        L = (16, 16, 16)

    dh = ps.create_data_handling(domain_size=L, periodicity=True, default_target=ps.Target.CPU)

    n_field = dh.add_array('n', values_per_cell=1)
    j_field = dh.add_array('j', values_per_cell=3 ** dim // 2, field_type=ps.FieldType.STAGGERED_FLUX)
    velocity_field = dh.add_array('v', values_per_cell=dim)

    D = 0.0666
    time = 100

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

    pde_kernel = ps.create_kernel(fvm_eq.discrete_continuity(j_field)).compile()

    sync_conc = dh.synchronization_function([n_field.name])

    # analytical density calculation
    def density(pos: np.ndarray, time: int, D: float):
        return (4 * np.pi * D * time)**(-dim / 2) * \
            np.exp(-np.sum(np.square(pos), axis=-1) / (4 * D * time))

    pos = np.zeros((*L, dim))
    xpos = np.arange(-L[0] // 2, L[0] // 2)
    ypos = np.arange(-L[1] // 2, L[1] // 2)

    if dim == 2:
        pos[..., 1], pos[..., 0] = np.meshgrid(xpos, ypos)
    elif dim == 3:
        zpos = np.arange(-L[2] // 2, L[2] // 2)
        pos[..., 2], pos[..., 1], pos[..., 0] = np.meshgrid(xpos, ypos, zpos)
    pos += 0.5

    def run(velocity: np.ndarray, time: int):
        dh.fill(n_field.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(j_field.name, np.nan, ghost_layers=True, inner_ghost_layers=True)

        # set initial values for velocity and density
        for i in range(dim):
            dh.fill(velocity_field.name, velocity[i], i, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(n_field.name, 0)
        if dim == 2:
            start = ps.make_slice[L[0] // 2 - 1:L[0] // 2 + 1, L[1] // 2 - 1:L[1] // 2 + 1]
        else:
            start = ps.make_slice[L[0] // 2 - 1:L[0] // 2 + 1, L[1] // 2 - 1:L[1] // 2 + 1,
                                  L[2] // 2 - 1:L[2] // 2 + 1]
        dh.fill(n_field.name, 2**-dim, slice_obj=start)

        sync_conc()
        for i in range(time):
            dh.run_kernel(flux_kernel)
            dh.run_kernel(pde_kernel)
            sync_conc()

        sim_density = dh.gather_array(n_field.name)
        
        # check that mass was conserved
        assert np.isclose(sim_density.sum(), 1)
        assert np.all(sim_density > 0)
        
        # check that the maximum is in the right place
        peak = np.unravel_index(np.argmax(sim_density, axis=None), sim_density.shape)
        assert np.allclose(peak, np.array(L) // 2 - 0.5 + velocity * time, atol=0.5)
        
        # check the concentration profile
        if np.linalg.norm(velocity) == 0:
            calc_density = density(pos - velocity * time, time, D)
            target = [time, D]
        
            pytest.importorskip('scipy.optimize')
            from scipy.optimize import curve_fit
            popt, _ = curve_fit(lambda x, t, D: density(x - velocity * time, t, D),
                                pos.reshape(-1, dim),
                                sim_density.reshape(-1),
                                p0=target)
        
            assert np.isclose(popt[0], time, rtol=0.1)
            assert np.isclose(popt[1], D, rtol=0.1)
            assert np.allclose(calc_density, sim_density, atol=1e-4)

    return lambda v: run(np.array(v), time)


advection_diffusion.runners = {}


@pytest.mark.parametrize("velocity", list(product([0, -0.047, 0.041], [0, -0.031, 0.023])))
def test_advection_diffusion_2d(velocity):
    if 2 not in advection_diffusion.runners:
        advection_diffusion.runners[2] = advection_diffusion(2)
    advection_diffusion.runners[2](velocity)


@pytest.mark.parametrize("velocity", list(product([0, -0.047, 0.041], [0, -0.031, 0.023], [0, -0.017, 0.011])))
@pytest.mark.longrun
def test_advection_diffusion_3d(velocity):
    if 3 not in advection_diffusion.runners:
        advection_diffusion.runners[3] = advection_diffusion(3)
    advection_diffusion.runners[3](velocity)


def VOF2(j: ps.field.Field, v: ps.field.Field, ρ: ps.field.Field, simplify=True):
    """Volume-of-fluid discretization of advection

    Args:
        j: the staggered field to write the fluxes to. Should have a D2Q9/D3Q27 stencil. Other stencils work too, but
           incur a small error (D2Q5/D3Q7: v^2, D3Q19: v^3).
        v: the flow velocity field
        ρ: the quantity to advect
        simplify: whether to simplify the generated expressions (slow, but makes them much more readable and faster)
    """
    dim = j.spatial_dimensions
    assert ps.FieldType.is_staggered(j)
    
    def assume_velocity(e):
        if not simplify:
            return e
        repl = {}
        for c in e.atoms(sp.StrictGreaterThan, sp.GreaterThan):
            if isinstance(c.lhs, ps.field.Field.Access) and c.lhs.field == v and isinstance(c.rhs, sp.Number):
                if c.rhs <= -1:
                    repl[c] = True
                elif c.rhs >= 1:
                    repl[c] = False
        for c in e.atoms(sp.StrictLessThan, sp.LessThan):
            if isinstance(c.lhs, ps.field.Field.Access) and c.lhs.field == v and isinstance(c.rhs, sp.Number):
                if c.rhs >= 1:
                    repl[c] = True
                elif c.rhs <= -1:
                    repl[c] = False
        for c in e.atoms(sp.Equality):
            if isinstance(c.lhs, ps.field.Field.Access) and c.lhs.field == v and isinstance(c.rhs, sp.Number):
                if c.rhs <= -1 or c.rhs >= 1:
                    repl[c] = False
        return e.subs(repl)
    
    class AABB:
        def __init__(self, corner0, corner1):
            self.dim = len(corner0)
            self.minCorner = sp.zeros(self.dim, 1)
            self.maxCorner = sp.zeros(self.dim, 1)
            for i in range(self.dim):
                self.minCorner[i] = sp.Piecewise((corner0[i], corner0[i] < corner1[i]), (corner1[i], True))
                self.maxCorner[i] = sp.Piecewise((corner1[i], corner0[i] < corner1[i]), (corner0[i], True))

        def intersect(self, other):
            minCorner = [sp.Max(self.minCorner[d], other.minCorner[d]) for d in range(self.dim)]
            maxCorner = [sp.Max(minCorner[d], sp.Min(self.maxCorner[d], other.maxCorner[d]))
                         for d in range(self.dim)]
            return AABB(minCorner, maxCorner)

        @property
        def volume(self):
            v = sp.prod([self.maxCorner[d] - self.minCorner[d] for d in range(self.dim)])
            if simplify:
                return sp.simplify(assume_velocity(v.rewrite(sp.Piecewise)))
            else:
                return v
    
    fluxes = []
    cell = AABB([-0.5] * dim, [0.5] * dim)
    cell_s = AABB(sp.Matrix([-0.5] * dim) + v.center_vector, sp.Matrix([0.5] * dim) + v.center_vector)
    for d, neighbor in enumerate(j.staggered_stencil):
        c = sp.Matrix(ps.stencil.direction_string_to_offset(neighbor)[:dim])
        cell_n = AABB(sp.Matrix([-0.5] * dim) + c, sp.Matrix([0.5] * dim) + c)
        cell_ns = AABB(sp.Matrix([-0.5] * dim) + c + v.neighbor_vector(neighbor),
                       sp.Matrix([0.5] * dim) + c + v.neighbor_vector(neighbor))
        fluxes.append(assume_velocity(ρ.center_vector * cell_s.intersect(cell_n).volume
                                      - ρ.neighbor_vector(neighbor) * cell_ns.intersect(cell).volume))
    
    assignments = []
    for i, d in enumerate(j.staggered_stencil):
        for lhs, rhs in zip(j.staggered_vector_access(d).values(), fluxes[i].values()):
            assignments.append(ps.Assignment(lhs, rhs))
    return assignments


@pytest.mark.parametrize("dim", [2, 3])
def test_advection(dim):
    L = (8,) * dim
    dh = ps.create_data_handling(L, periodicity=True, default_target=ps.Target.CPU)
    c = dh.add_array('c', values_per_cell=1)
    j = dh.add_array('j', values_per_cell=3 ** dh.dim // 2, field_type=ps.FieldType.STAGGERED_FLUX)
    u = dh.add_array('u', values_per_cell=dh.dim)
    
    dh.cpu_arrays[c.name][:] = (np.random.random([l + 2 for l in L]))
    dh.cpu_arrays[u.name][:] = (np.random.random([l + 2 for l in L] + [dim]) - 0.5) / 5
    
    vof1 = ps.create_kernel(ps.fd.VOF(j, u, c)).compile()
    dh.fill(j.name, np.nan, ghost_layers=True)
    dh.run_kernel(vof1)
    j1 = dh.gather_array(j.name).copy()
    
    vof2 = ps.create_kernel(VOF2(j, u, c, simplify=False)).compile()
    dh.fill(j.name, np.nan, ghost_layers=True)
    dh.run_kernel(vof2)
    j2 = dh.gather_array(j.name)
    
    assert np.allclose(j1, j2)


@pytest.mark.parametrize("stencil", ["D2Q5", "D2Q9"])
def test_ek(stencil):

    # parameters

    L = (40, 40)
    D = sp.Symbol("D")
    z = sp.Symbol("z")

    # data structures

    dh = ps.create_data_handling(L, periodicity=True, default_target=ps.Target.CPU)
    c = dh.add_array('c', values_per_cell=1)
    j = dh.add_array('j', values_per_cell=int(stencil[-1]) // 2, field_type=ps.FieldType.STAGGERED_FLUX)
    Phi = dh.add_array('Φ', values_per_cell=1)

    # perform automatic discretization

    def Gradient(f):
        return sp.Matrix([ps.fd.diff(f, i) for i in range(dh.dim)])

    flux_eq = -D * Gradient(c) + D * z * c.center * Gradient(Phi)

    disc = ps.fd.FVM1stOrder(c, flux_eq)
    flux_assignments = disc.discrete_flux(j)
    continuity_assignments = disc.discrete_continuity(j)

    # manual discretization

    x_staggered = - c[-1, 0] + c[0, 0] + z * (c[-1, 0] + c[0, 0]) / 2 * (Phi[-1, 0] - Phi[0, 0])
    y_staggered = - c[0, -1] + c[0, 0] + z * (c[0, -1] + c[0, 0]) / 2 * (Phi[0, -1] - Phi[0, 0])
    xy_staggered = (- c[-1, -1] + c[0, 0]) / sp.sqrt(2) + \
        z * (c[-1, -1] + c[0, 0]) / 2 * (Phi[-1, -1] - Phi[0, 0]) / sp.sqrt(2)
    xY_staggered = (- c[-1, 1] + c[0, 0]) / sp.sqrt(2) + \
        z * (c[-1, 1] + c[0, 0]) / 2 * (Phi[-1, 1] - Phi[0, 0]) / sp.sqrt(2)
    A0 = (1 + sp.sqrt(2) if j.index_shape[0] == 4 else 1)

    jj = j.staggered_access
    divergence = -1 * sum([jj(d) for d in j.staggered_stencil
                          + [ps.stencil.inverse_direction_string(d) for d in j.staggered_stencil]])

    update = [ps.Assignment(c.center, c.center + divergence)]
    flux = [ps.Assignment(j.staggered_access("W"), D * x_staggered / A0),
            ps.Assignment(j.staggered_access("S"), D * y_staggered / A0)]
    if j.index_shape[0] == 4:
        flux += [ps.Assignment(j.staggered_access("SW"), D * xy_staggered / A0),
                 ps.Assignment(j.staggered_access("NW"), D * xY_staggered / A0)]

    # compare

    for a, b in zip(flux, flux_assignments):
        assert a.lhs == b.lhs
        assert sp.simplify(a.rhs - b.rhs) == 0
    for a, b in zip(update, continuity_assignments):
        assert a.lhs == b.lhs
        assert a.rhs == b.rhs

# TODO: test source


@pytest.mark.parametrize("stencil", ["D2Q5", "D2Q9", "D3Q7", "D3Q19", "D3Q27"])
@pytest.mark.parametrize("derivative", [0, 1])
def test_flux_stencil(stencil, derivative):
    L = (40, ) * int(stencil[1])
    dh = ps.create_data_handling(L, periodicity=True, default_target=ps.Target.CPU)
    c = dh.add_array('c', values_per_cell=1)
    j = dh.add_array('j', values_per_cell=int(stencil[3:]) // 2, field_type=ps.FieldType.STAGGERED_FLUX)

    def Gradient(f):
        return sp.Matrix([ps.fd.diff(f, i) for i in range(dh.dim)])

    eq = [sp.Matrix([sp.Symbol(f"a_{i}") * c.center for i in range(dh.dim)]), Gradient(c)][derivative]
    disc = ps.fd.FVM1stOrder(c, flux=eq)

    # check the continuity
    continuity_assignments = disc.discrete_continuity(j)
    assert [len(a.rhs.atoms(ps.field.Field.Access)) for a in continuity_assignments] == \
           [int(stencil[3:])] * len(continuity_assignments)

    # check the flux
    flux_assignments = disc.discrete_flux(j)
    assert [len(a.rhs.atoms(ps.field.Field.Access)) for a in flux_assignments] == [2] * len(flux_assignments)


@pytest.mark.parametrize("stencil", ["D2Q5", "D2Q9", "D3Q7", "D3Q19", "D3Q27"])
def test_source_stencil(stencil):
    L = (40, ) * int(stencil[1])
    dh = ps.create_data_handling(L, periodicity=True, default_target=ps.Target.CPU)
    c = dh.add_array('c', values_per_cell=1)
    j = dh.add_array('j', values_per_cell=int(stencil[3:]) // 2, field_type=ps.FieldType.STAGGERED_FLUX)

    continuity_ref = ps.fd.FVM1stOrder(c).discrete_continuity(j)

    for eq in [c.center] + [ps.fd.diff(c, i) for i in range(dh.dim)]:
        disc = ps.fd.FVM1stOrder(c, source=eq)
        diff = sp.simplify(disc.discrete_continuity(j)[0].rhs - continuity_ref[0].rhs)
        if type(eq) is ps.field.Field.Access:
            assert len(diff.atoms(ps.field.Field.Access)) == 1
        else:
            assert len(diff.atoms(ps.field.Field.Access)) == 2
