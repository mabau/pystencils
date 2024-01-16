import sympy as sp
import pystencils as ps
import numpy as np
import pytest
from itertools import product
from pystencils.rng import random_symbol
from pystencils.astnodes import SympyAssignment
from pystencils.node_collection import NodeCollection


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


def advection_diffusion_fluctuations(dim: int):
    # parameters
    if dim == 2:
        L = (32, 32)
        stencil_factor = np.sqrt(1 / (1 + np.sqrt(2)))
    elif dim == 3:
        L = (16, 16, 16)
        stencil_factor = np.sqrt(1 / (1 + 2 * np.sqrt(2) + 4.0 / 3.0 * np.sqrt(3)))

    dh = ps.create_data_handling(domain_size=L, periodicity=True, default_target=ps.Target.CPU)

    n_field = dh.add_array('n', values_per_cell=1)
    j_field = dh.add_array('j', values_per_cell=3 ** dim // 2, field_type=ps.FieldType.STAGGERED_FLUX)
    velocity_field = dh.add_array('v', values_per_cell=dim)

    D = 0.00666
    time = 10000

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
    flux = ps.AssignmentCollection(flux)

    rng_symbol_gen = random_symbol(flux.subexpressions, dim=dh.dim)
    for i in range(len(flux.main_assignments)):
        n = j_field.staggered_stencil[i]
        assert flux.main_assignments[i].lhs == j_field.staggered_access(n)
        
        # calculate mean density
        dens = (n_field.neighbor_vector(n) + n_field.center_vector)[0] / 2
        # multyply by smoothed haviside function so that fluctuation will not get bigger that the density
        dens *= sp.Max(0, sp.Min(1.0, n_field.neighbor_vector(n)[0]) * sp.Min(1.0, n_field.center_vector[0]))
        
        # lenght of the vector
        length = sp.sqrt(len(j_field.staggered_stencil[i]))
        
        # amplitude of the random fluctuations
        fluct = sp.sqrt(2 * dens * D) * sp.sqrt(1 / length) * stencil_factor
        # add fluctuations
        fluct *= 2 * (next(rng_symbol_gen) - 0.5) * sp.sqrt(3)
        
        flux.main_assignments[i] = ps.Assignment(flux.main_assignments[i].lhs, flux.main_assignments[i].rhs + fluct)
    
    # Add the folding to the flux, so that the random numbers persist through the ghostlayers.
    fold = {ps.astnodes.LoopOverCoordinate.get_loop_counter_symbol(i):
            ps.astnodes.LoopOverCoordinate.get_loop_counter_symbol(i) % L[i] for i in range(len(L))}
    flux.subs(fold)

    flux_kernel = ps.create_staggered_kernel(flux).compile()

    pde_kernel = ps.create_kernel(fvm_eq.discrete_continuity(j_field)).compile()

    sync_conc = dh.synchronization_function([n_field.name])

    # analytical density distribution calculation
    def P(rho, density_init):
        res = []
        for r in rho:
            res.append(np.power(density_init, r) * np.exp(-density_init) / np.math.gamma(r + 1))
        return np.array(res)

    def run(density_init: float, velocity: np.ndarray, time: int):
        dh.fill(n_field.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(j_field.name, np.nan, ghost_layers=True, inner_ghost_layers=True)

        # set initial values for velocity and density
        for i in range(dim):
            dh.fill(velocity_field.name, velocity[i], i, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(n_field.name, density_init)

        measurement_intervall = 10
        warm_up = 1000
        data = []

        sync_conc()
        for i in range(warm_up):
            dh.run_kernel(flux_kernel, seed=42, time_step=i)
            dh.run_kernel(pde_kernel)
            sync_conc()

        for i in range(time):
            dh.run_kernel(flux_kernel, seed=42, time_step=i + warm_up)
            dh.run_kernel(pde_kernel)
            sync_conc()
            if(i % measurement_intervall == 0):
                data = np.append(data, dh.gather_array(n_field.name).ravel(), 0)

        # test mass conservation
        np.testing.assert_almost_equal(dh.gather_array(n_field.name).mean(), density_init)

        n_bins = 50

        density_value, bins = np.histogram(data, density=True, bins=n_bins)
        bins_mean = bins[:-1] + (bins[1:] - bins[:-1]) / 2
        analytical_value = P(bins_mean, density_init)
        print(density_value - analytical_value)
        np.testing.assert_allclose(density_value, analytical_value, atol=2e-3)

    return lambda density_init, v: run(density_init, np.array(v), time)


advection_diffusion_fluctuations.runners = {}


@pytest.mark.parametrize("velocity", list(product([0, 0.00041], [0, -0.00031])))
@pytest.mark.parametrize("density", [27.0, 56.5])
@pytest.mark.longrun
def test_advection_diffusion_fluctuation_2d(density, velocity):
    if 2 not in advection_diffusion_fluctuations.runners:
        advection_diffusion_fluctuations.runners[2] = advection_diffusion_fluctuations(2)
    advection_diffusion_fluctuations.runners[2](density, velocity)


@pytest.mark.parametrize("velocity", [(0.0, 0.0, 0.0), (0.00043, -0.00017, 0.00028)])
@pytest.mark.parametrize("density", [27.0, 56.5])
@pytest.mark.longrun
def test_advection_diffusion_fluctuation_3d(density, velocity):
    if 3 not in advection_diffusion_fluctuations.runners:
        advection_diffusion_fluctuations.runners[3] = advection_diffusion_fluctuations(3)
    advection_diffusion_fluctuations.runners[3](density, velocity)


def diffusion_reaction(fluctuations: bool):
    # parameters
    L = (32, 32)
    stencil_factor = np.sqrt(1 / (1 + np.sqrt(2)))

    dh = ps.create_data_handling(domain_size=L, periodicity=True, default_target=ps.Target.CPU)

    species = 2
    n_fields = []
    j_fields = []
    r_flux_fields = []
    for i in range(species):
        n_fields.append(dh.add_array(f'n_{i}', values_per_cell=1))
        j_fields.append(dh.add_array(f'j_{i}', values_per_cell=3 ** dh.dim // 2,
                                     field_type=ps.FieldType.STAGGERED_FLUX))
        r_flux_fields.append(dh.add_array(f'r_{i}', values_per_cell=1))
    velocity_field = dh.add_array('v', values_per_cell=dh.dim)

    D = 0.00666
    time = 1000
    r_order = [2.0, 0.0]
    r_rate_const = 0.00001
    r_coefs = [-2, 1]

    def grad(f):
        return sp.Matrix([ps.fd.diff(f, i) for i in range(dh.dim)])

    flux_eq = - D * grad(n_fields[0])
    fvm_eq = ps.fd.FVM1stOrder(n_fields[0], flux=flux_eq)
    vof_adv = ps.fd.VOF(j_fields[0], velocity_field, n_fields[0])
    continuity_assignments = fvm_eq.discrete_continuity(j_fields[0])
    # merge calculation of advection and diffusion terms
    flux = []
    for adv, div in zip(vof_adv, fvm_eq.discrete_flux(j_fields[0])):
        assert adv.lhs == div.lhs
        flux.append(ps.Assignment(adv.lhs, adv.rhs + div.rhs))
    flux = ps.AssignmentCollection(flux)

    if(fluctuations):
        rng_symbol_gen = random_symbol(flux.subexpressions, dim=dh.dim)
        for i in range(len(flux.main_assignments)):
            n = j_fields[0].staggered_stencil[i]
            assert flux.main_assignments[i].lhs == j_fields[0].staggered_access(n)
            
            # calculate mean density
            dens = (n_fields[0].neighbor_vector(n) + n_fields[0].center_vector)[0] / 2
            # multyply by smoothed haviside function so that fluctuation will not get bigger that the density
            dens *= sp.Max(0,
                           sp.Min(1.0, n_fields[0].neighbor_vector(n)[0]) * sp.Min(1.0, n_fields[0].center_vector[0]))
            
            # lenght of the vector
            length = sp.sqrt(len(j_fields[0].staggered_stencil[i]))
            
            # amplitude of the random fluctuations
            fluct = sp.sqrt(2 * dens * D) * sp.sqrt(1 / length) * stencil_factor
            # add fluctuations
            fluct *= 2 * (next(rng_symbol_gen) - 0.5) * sp.sqrt(3)
            flux.main_assignments[i] = ps.Assignment(flux.main_assignments[i].lhs, flux.main_assignments[i].rhs + fluct)
        
        # Add the folding to the flux, so that the random numbers persist through the ghostlayers.
        fold = {ps.astnodes.LoopOverCoordinate.get_loop_counter_symbol(i):
                ps.astnodes.LoopOverCoordinate.get_loop_counter_symbol(i) % L[i] for i in range(len(L))}
        flux.subs(fold)

    r_flux = NodeCollection([SympyAssignment(j_fields[i].center, 0) for i in range(species)])
    reaction = r_rate_const
    for i in range(species):
        reaction *= sp.Pow(n_fields[i].center, r_order[i])
    new_assignments = []
    if fluctuations:
        rng_symbol_gen = random_symbol(new_assignments, dim=dh.dim)
        reaction_fluctuations = sp.sqrt(sp.Abs(reaction)) * 2 * (next(rng_symbol_gen) - 0.5) * sp.sqrt(3)
        reaction_fluctuations *= sp.Min(1, sp.Abs(reaction**2))
    else:
        reaction_fluctuations = 0.0
    for i in range(species):
        r_flux.all_assignments[i] = SympyAssignment(
            r_flux_fields[i].center, (reaction + reaction_fluctuations) * r_coefs[i])
    [r_flux.all_assignments.insert(0, new) for new in new_assignments]

    continuity_assignments = [SympyAssignment(*assignment.args) for assignment in continuity_assignments]
    continuity_assignments.append(SympyAssignment(n_fields[0].center, n_fields[0].center + r_flux_fields[0].center))

    flux_kernel = ps.create_staggered_kernel(flux).compile()
    reaction_kernel = ps.create_kernel(r_flux).compile()

    config = ps.CreateKernelConfig(allow_double_writes=True)
    pde_kernel = ps.create_kernel(continuity_assignments, config=config).compile()

    sync_conc = dh.synchronization_function([n_fields[0].name, n_fields[1].name])

    def f(t, r, n0, fac, fluctuations):
        """Calculates the amount of product created after a certain time of a reaction with form xA -> B

        Args:
            t: Time of the reation
            r: Reaction rate constant
            n0: Initial density of the 
            fac: Reaction order of A (this in most cases equals the stochometric coefficient x)
            fluctuations: Boolian whether fluctuations were included during the reaction.
        """
        if fluctuations:
            return 1 / fac * (n0 + n0 / (n0 - (n0 + 1) * np.exp(fac * r * t)))
        return 1 / fac * (n0 - (1 / (fac * r * t + (1 / n0))))

    def run(density_init: float, velocity: np.ndarray, time: int):
        for i in range(species):
            dh.fill(n_fields[i].name, np.nan, ghost_layers=True, inner_ghost_layers=True)
            dh.fill(j_fields[i].name, 0.0, ghost_layers=True, inner_ghost_layers=True)
            dh.fill(r_flux_fields[i].name, 0.0, ghost_layers=True, inner_ghost_layers=True)

        # set initial values for velocity and density
        for i in range(dh.dim):
            dh.fill(velocity_field.name, velocity[i], i, ghost_layers=True, inner_ghost_layers=True)
        dh.fill(n_fields[0].name, density_init)
        dh.fill(n_fields[1].name, 0.0)

        measurement_intervall = 10
        data = []
        
        sync_conc()
        for i in range(time):
            if(i % measurement_intervall == 0):
                data.append([i, dh.gather_array(n_fields[1].name).mean(), dh.gather_array(n_fields[0].name).mean()])
            dh.run_kernel(reaction_kernel, seed=41, time_step=i)
            for s_idx in range(species):
                flux_kernel(n_0=dh.cpu_arrays[n_fields[s_idx].name],
                            j_0=dh.cpu_arrays[j_fields[s_idx].name],
                            v=dh.cpu_arrays[velocity_field.name], seed=42 + s_idx, time_step=i)
                pde_kernel(n_0=dh.cpu_arrays[n_fields[s_idx].name],
                           j_0=dh.cpu_arrays[j_fields[s_idx].name],
                           r_0=dh.cpu_arrays[r_flux_fields[s_idx].name])
            sync_conc()

        data = np.array(data).transpose()
        x = data[0]
        analytical_value = f(x, r_rate_const, density_init, abs(r_coefs[0]), fluctuations)

        # test mass conservation
        np.testing.assert_almost_equal(
            dh.gather_array(n_fields[0].name).mean() + 2 * dh.gather_array(n_fields[1].name).mean(), density_init)

        r_tol = 2e-3
        if fluctuations:
            r_tol = 3e-2
        np.testing.assert_allclose(data[1], analytical_value, rtol=r_tol)

    return lambda density_init, v: run(density_init, np.array(v), time)


advection_diffusion_fluctuations.runners = {}


@pytest.mark.parametrize("velocity", list(product([0, 0.0041], [0, -0.0031])))
@pytest.mark.parametrize("density", [27.0, 56.5])
@pytest.mark.parametrize("fluctuations", [False, True])
@pytest.mark.longrun
def test_diffusion_reaction(fluctuations, density, velocity):
    diffusion_reaction.runner = diffusion_reaction(fluctuations)
    diffusion_reaction.runner(density, velocity)


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


def test_fvm_staggered_simplification():
    D = sp.Symbol("D")
    data_type = "float64"

    c = ps.fields(f"c: {data_type}[2D]", layout='fzyx')
    j = ps.fields(f"j(2): {data_type}[2D]", layout='fzyx', field_type=ps.FieldType.STAGGERED_FLUX)

    grad_c = sp.Matrix([ps.fd.diff(c, i) for i in range(c.spatial_dimensions)])

    ek = ps.fd.FVM1stOrder(c, flux=-D * grad_c)

    ast = ps.create_staggered_kernel(ek.discrete_flux(j))
    code = ps.get_code_str(ast)

    assert '_size_c_0 - 1 < _size_c_0 - 1' not in code
