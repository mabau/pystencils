import numpy as np
import sympy as sp

import pytest

import pystencils as ps
from pystencils import x_staggered_vector, TypedSymbol
from pystencils.enums import Target


class TestStaggeredDiffusion:
    def _run(self, num_neighbors, target=ps.Target.CPU, openmp=False):
        L = (40, 40)
        D = 0.066
        dt = 1
        T = 100

        dh = ps.create_data_handling(L, periodicity=True, default_target=target)

        c = dh.add_array('c', values_per_cell=1)
        j = dh.add_array('j', values_per_cell=num_neighbors, field_type=ps.FieldType.STAGGERED_FLUX)

        x_staggered = - c[-1, 0] + c[0, 0]
        y_staggered = - c[0, -1] + c[0, 0]
        xy_staggered = - c[-1, -1] + c[0, 0]
        xY_staggered = - c[-1, 1] + c[0, 0]

        jj = j.staggered_access
        divergence = -1 * D / (1 + sp.sqrt(2) if j.index_shape[0] == 4 else 1) * \
            sum([jj(d) / sp.Matrix(ps.stencil.direction_string_to_offset(d)).norm() for d in j.staggered_stencil
                 + [ps.stencil.inverse_direction_string(d) for d in j.staggered_stencil]])

        update = [ps.Assignment(c.center, c.center + dt * divergence)]
        flux = [ps.Assignment(j.staggered_access("W"), x_staggered),
                ps.Assignment(j.staggered_access("S"), y_staggered)]
        if j.index_shape[0] == 4:
            flux += [ps.Assignment(j.staggered_access("SW"), xy_staggered),
                     ps.Assignment(j.staggered_access("NW"), xY_staggered)]

        staggered_kernel = ps.create_staggered_kernel(flux, target=dh.default_target, cpu_openmp=openmp).compile()
        div_kernel = ps.create_kernel(update, target=dh.default_target, cpu_openmp=openmp).compile()

        def time_loop(steps):
            sync = dh.synchronization_function([c.name])
            dh.all_to_gpu()
            for i in range(steps):
                sync()
                dh.run_kernel(staggered_kernel)
                dh.run_kernel(div_kernel)
            dh.all_to_cpu()

        def init():
            dh.fill(c.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
            dh.fill(c.name, 0)
            dh.fill(j.name, np.nan, ghost_layers=True, inner_ghost_layers=True)
            dh.cpu_arrays[c.name][L[0] // 2:L[0] // 2 + 2, L[1] // 2:L[1] // 2 + 2] = 1.0

        init()
        time_loop(T)

        reference = np.empty(L)
        for x in range(L[0]):
            for y in range(L[1]):
                r = np.array([x, y]) - L[0] / 2 + 0.5
                reference[x, y] = (4 * np.pi * D * T)**(-dh.dim / 2) * np.exp(-np.dot(r, r) / (4 * D * T)) * (2**dh.dim)

        assert np.abs(dh.gather_array(c.name) - reference).max() < 5e-4

    def test_diffusion_2(self):
        self._run(2)

    def test_diffusion_4(self):
        self._run(4)

    def test_diffusion_openmp(self):
        self._run(4, openmp=True)


def test_staggered_subexpressions():
    dh = ps.create_data_handling((10, 10), periodicity=True, default_target=Target.CPU)
    j = dh.add_array('j', values_per_cell=2, field_type=ps.FieldType.STAGGERED)
    c = sp.symbols("c")
    assignments = [ps.Assignment(j.staggered_access("W"), c),
                   ps.Assignment(c, 1)]
    ps.create_staggered_kernel(assignments, target=dh.default_target).compile()


def test_staggered_loop_cutting():
    pytest.importorskip('islpy')
    dh = ps.create_data_handling((4, 4), periodicity=True, default_target=Target.CPU)
    j = dh.add_array('j', values_per_cell=4, field_type=ps.FieldType.STAGGERED)
    assignments = [ps.Assignment(j.staggered_access("SW"), 1)]
    ast = ps.create_staggered_kernel(assignments, target=dh.default_target)
    assert not ast.atoms(ps.astnodes.Conditional)


def test_staggered_vector():
    dim = 2
    v = x_staggered_vector(dim)
    ctr0 = TypedSymbol('ctr_0', 'int', nonnegative=True)
    ctr1 = TypedSymbol('ctr_1', 'int', nonnegative=True)
    expected_result = sp.Matrix(tuple((ctr0 + 0.5, ctr1 + 0.5)))
    assert v == expected_result