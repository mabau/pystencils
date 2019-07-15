import os
from tempfile import TemporaryDirectory

import numpy as np

from pystencils import Assignment, create_kernel
from pystencils.boundaries import BoundaryHandling, Neumann, add_neumann_boundary
from pystencils.datahandling import SerialDataHandling
from pystencils.slicing import slice_from_direction


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
        boundary_handling.geometry_to_vtk(file_name=os.path.join(tmp_dir, 'test_output1'), ghost_layers=False)
        boundary_handling.geometry_to_vtk(file_name=os.path.join(tmp_dir, 'test_output2'), ghost_layers=True)
