import os
from tempfile import TemporaryDirectory

import numpy as np
import pytest

import pystencils
from pystencils import Assignment, create_kernel
from pystencils.boundaries import BoundaryHandling, Dirichlet, Neumann, add_neumann_boundary
from pystencils.datahandling import SerialDataHandling
from pystencils.enums import Target
from pystencils.slicing import slice_from_direction
from pystencils.timeloop import TimeLoop


def test_kernel_vs_copy_boundary():
    dh = SerialDataHandling(domain_size=(7, 7))
    src = dh.add_array('src')
    dst_builtin = dh.add_array_like('dst_builtin', 'src')
    dst_python_copy = dh.add_array_like('dst_python_copy', 'src')
    dst_handling = dh.add_array_like('dst_handling', 'src')

    src_arr = np.arange(dh.shape[0] * dh.shape[1]).reshape(dh.shape)

    def reset_src():
        for block in dh.iterate(ghost_layers=True, inner_ghost_layers=True):
            np.copyto(block['src'], np.random.rand(*block.shape))

        for block in dh.iterate(ghost_layers=False, inner_ghost_layers=True):
            np.copyto(block['src'], src_arr)

    for b in dh.iterate(ghost_layers=False, inner_ghost_layers=True):
        np.copyto(b['dst_builtin'], 42)
        np.copyto(b['dst_python_copy'], 43)
        np.copyto(b['dst_handling'], 44)

    flags = dh.add_array('flags', dtype=np.uint8)
    dh.fill(flags.name, 0)
    borders = ['N', 'S', 'E', 'W']
    for d in borders:
        dh.fill(flags.name, 1, slice_obj=slice_from_direction(d, dim=2), ghost_layers=True, inner_ghost_layers=True)

    rhs = sum(src.neighbors([(1, 0), (-1, 0), (0, 1), (0, -1)]))

    simple_kernel = create_kernel([Assignment(dst_python_copy.center, rhs)]).compile()
    kernel_handling = create_kernel([Assignment(dst_handling.center, rhs)]).compile()

    assignments_with_boundary = add_neumann_boundary([Assignment(dst_builtin.center, rhs)],
                                                     fields=[src], flag_field=flags, boundary_flag=1)
    kernel_with_boundary = create_kernel(assignments_with_boundary).compile()

    # ------ Method 1: Built-in boundary
    reset_src()
    dh.run_kernel(kernel_with_boundary)

    # ------ Method 2: Using python to copy out the values (reference)
    reset_src()
    for b in dh.iterate():
        arr = b['src']
        arr[:, 0] = arr[:, 1]
        arr[:, -1] = arr[:, -2]
        arr[0, :] = arr[1, :]
        arr[-1, :] = arr[-2, :]
    dh.run_kernel(simple_kernel)

    # ------ Method 3: Using boundary handling to copy out the values
    reset_src()
    boundary_stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    boundary_handling = BoundaryHandling(dh, src.name, boundary_stencil)
    neumann = Neumann()
    assert neumann.name == 'Neumann'
    neumann.name = "wall"
    assert neumann.name == 'wall'
    assert neumann.additional_data_init_callback is None
    assert len(neumann.additional_data) == 0

    for d in ('N', 'S', 'W', 'E'):
        boundary_handling.set_boundary(neumann, slice_from_direction(d, dim=2))
    boundary_handling()
    dh.run_kernel(kernel_handling)

    python_copy_result = dh.gather_array('dst_python_copy')
    builtin_result = dh.gather_array('dst_builtin')
    handling_result = dh.gather_array('dst_handling')

    np.testing.assert_almost_equal(python_copy_result, builtin_result)
    np.testing.assert_almost_equal(python_copy_result, handling_result)

    with TemporaryDirectory() as tmp_dir:
        pytest.importorskip('pyevtk')
        boundary_handling.geometry_to_vtk(file_name=os.path.join(tmp_dir, 'test_output1'), ghost_layers=False)
        boundary_handling.geometry_to_vtk(file_name=os.path.join(tmp_dir, 'test_output2'), ghost_layers=True)

        boundaries = list(boundary_handling._boundary_object_to_boundary_info.keys()) + ['domain']
        boundary_handling.geometry_to_vtk(file_name=os.path.join(tmp_dir, 'test_output3'),
                                          boundaries=boundaries[0], ghost_layers=False)


def test_boundary_gpu():
    pytest.importorskip('pycuda')
    dh = SerialDataHandling(domain_size=(7, 7), default_target=Target.GPU)
    src = dh.add_array('src')
    dh.fill("src", 0.0, ghost_layers=True)
    dh.fill("src", 1.0, ghost_layers=False)
    src_cpu = dh.add_array('src_cpu', gpu=False)
    dh.fill("src_cpu", 0.0, ghost_layers=True)
    dh.fill("src_cpu", 1.0, ghost_layers=False)

    boundary_stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    boundary_handling_cpu = BoundaryHandling(dh, src_cpu.name, boundary_stencil,
                                             name="boundary_handling_cpu", target=Target.CPU)

    boundary_handling = BoundaryHandling(dh, src.name, boundary_stencil,
                                         name="boundary_handling_gpu", target=Target.GPU)

    neumann = Neumann()
    for d in ('N', 'S', 'W', 'E'):
        boundary_handling.set_boundary(neumann, slice_from_direction(d, dim=2))
        boundary_handling_cpu.set_boundary(neumann, slice_from_direction(d, dim=2))

    boundary_handling.prepare()
    boundary_handling_cpu.prepare()

    boundary_handling_cpu()

    dh.all_to_gpu()
    boundary_handling()
    dh.all_to_cpu()
    np.testing.assert_almost_equal(dh.cpu_arrays["src_cpu"], dh.cpu_arrays["src"])


def test_boundary_utility():
    dh = SerialDataHandling(domain_size=(7, 7))
    src = dh.add_array('src')
    dh.fill("src", 0.0, ghost_layers=True)

    boundary_stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    boundary_handling = BoundaryHandling(dh, src.name, boundary_stencil,
                                         name="boundary_handling", target=Target.CPU)

    neumann = Neumann()
    dirichlet = Dirichlet(2)
    for d in ('N', 'S', 'W', 'E'):
        boundary_handling.set_boundary(neumann, slice_from_direction(d, dim=2))

    boundary_handling.set_boundary(neumann, (slice(2, 4, None), slice(2, 4, None)))

    boundary_handling.prepare()

    assert boundary_handling.get_flag(boundary_handling.boundary_objects[0]) == 2
    assert boundary_handling.shape == dh.shape
    assert boundary_handling.flag_array_name == 'boundary_handlingFlags'
    mask_neumann = boundary_handling.get_mask((slice(0, 7), slice(0, 7)), boundary_handling.boundary_objects[0])
    np.testing.assert_almost_equal(mask_neumann[1:3, 1:3], 2)

    mask_domain = boundary_handling.get_mask((slice(0, 7), slice(0, 7)), "domain")
    assert np.sum(mask_domain) == 7 ** 2 - 4

    def set_sphere(x, y):
        mid = (4, 4)
        radius = 2
        return (x - mid[0]) ** 2 + (y - mid[1]) ** 2 < radius ** 2

    boundary_handling.set_boundary(dirichlet, mask_callback=set_sphere, force_flag_value=4)
    mask_dirichlet = boundary_handling.get_mask((slice(0, 7), slice(0, 7)), boundary_handling.boundary_objects[1])
    assert np.sum(mask_dirichlet) == 48

    assert boundary_handling.set_boundary("domain") == 1

    assert boundary_handling.set_boundary(dirichlet, mask_callback=set_sphere, force_flag_value=8, replace=False) == 4
    assert boundary_handling.set_boundary(dirichlet, force_flag_value=16, replace=False) == 4

    assert boundary_handling.set_boundary_where_flag_is_set(boundary_handling.boundary_objects[0], 16) == 16


def test_add_fix_steps():
    dh = SerialDataHandling(domain_size=(7, 7))
    src = dh.add_array('src')
    dh.fill("src", 0.0, ghost_layers=True)
    dh.fill("src", 1.0, ghost_layers=False)
    boundary_stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    boundary_handling = BoundaryHandling(dh, src.name, boundary_stencil,
                                         name="boundary_handling", target=pystencils.Target.CPU)

    neumann = Neumann()
    for d in ('N', 'S', 'W', 'E'):
        boundary_handling.set_boundary(neumann, slice_from_direction(d, dim=2))

    timeloop = TimeLoop(steps=1)
    boundary_handling.add_fixed_steps(timeloop)

    timeloop.run()
    assert np.sum(dh.cpu_arrays['src']) == 7 * 7 + 7 * 4


def test_boundary_data_setter():
    dh = SerialDataHandling(domain_size=(7, 7))
    src = dh.add_array('src')
    dh.fill("src", 0.0, ghost_layers=True)
    dh.fill("src", 1.0, ghost_layers=False)
    boundary_stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    boundary_handling = BoundaryHandling(dh, src.name, boundary_stencil,
                                         name="boundary_handling", target=Target.CPU)

    neumann = Neumann()
    for d in 'N':
        boundary_handling.set_boundary(neumann, slice_from_direction(d, dim=2))

    boundary_handling.prepare()

    for b in dh.iterate(ghost_layers=True):
        index_array_bd = b[boundary_handling._index_array_name]
        data_setter = index_array_bd.boundary_object_to_data_setter[boundary_handling.boundary_objects[0]]

        y_pos = data_setter.boundary_cell_positions(1)

        assert all(y_pos == 5.5)
        assert np.all(data_setter.link_offsets() == [0, -1])
        assert np.all(data_setter.link_positions(1) == 6.)


@pytest.mark.parametrize('with_indices', ('with_indices', False))
def test_dirichlet(with_indices):
    value = (1, 20, 3) if with_indices else 1

    dh = SerialDataHandling(domain_size=(7, 7))
    src = dh.add_array('src', values_per_cell=3 if with_indices else 1)
    dh.cpu_arrays.src[...] = np.random.rand(*src.shape)
    boundary_stencil = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    boundary_handling = BoundaryHandling(dh, src.name, boundary_stencil)
    dirichlet = Dirichlet(value)
    assert dirichlet.name == 'Dirichlet'
    dirichlet.name = "wall"
    assert dirichlet.name == 'wall'

    for d in ('N', 'S', 'W', 'E'):
        boundary_handling.set_boundary(dirichlet, slice_from_direction(d, dim=2))
    boundary_handling()

    assert all([np.allclose(a, np.array(value)) for a in dh.cpu_arrays.src[1:-2, 0]])
    assert all([np.allclose(a, np.array(value)) for a in dh.cpu_arrays.src[1:-2, -1]])
    assert all([np.allclose(a, np.array(value)) for a in dh.cpu_arrays.src[0, 1:-2]])
    assert all([np.allclose(a, np.array(value)) for a in dh.cpu_arrays.src[-1, 1:-2]])
