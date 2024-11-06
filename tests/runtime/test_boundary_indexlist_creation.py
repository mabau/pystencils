import numpy as np
from itertools import product
import pystencils.boundaries.createindexlist as cil

import pytest


@pytest.mark.parametrize('single_link', [False, True])
@pytest.mark.skipif(not cil.cython_funcs_available, reason='Cython functions are not available')
def test_equivalence_cython_python_version(single_link):
    #   D2Q9
    stencil_2d = tuple((x, y) for x, y in product([-1, 0, 1], [-1, 0, 1]))
    #   D3Q19
    stencil_3d = tuple(
        (x, y, z) for x, y, z in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]) if abs(x) + abs(y) + abs(z) < 3)

    for dtype in [int, np.int16, np.uint32]:
        fluid_mask = dtype(1)
        mask = dtype(2)
        flag_field_2d = np.ones([15, 16], dtype=dtype) * fluid_mask
        flag_field_3d = np.ones([15, 16, 17], dtype=dtype) * fluid_mask

        flag_field_2d[0, :] = mask
        flag_field_2d[-1, :] = mask
        flag_field_2d[7, 7] = mask

        flag_field_3d[0, :, :] = mask
        flag_field_3d[-1, :, :] = mask
        flag_field_3d[7, 7, 7] = mask

        result_python_2d = cil._create_index_list_python(flag_field_2d, mask, fluid_mask,
                                                         stencil_2d, single_link, True, 1)

        result_python_3d = cil._create_index_list_python(flag_field_3d, mask, fluid_mask,
                                                         stencil_3d, single_link, True, 1)

        result_cython_2d = cil.create_boundary_index_list(flag_field_2d, stencil_2d, mask,
                                                          fluid_mask, 1, True, single_link)
        result_cython_3d = cil.create_boundary_index_list(flag_field_3d, stencil_3d, mask,
                                                          fluid_mask, 1, True, single_link)

        np.testing.assert_equal(result_python_2d, result_cython_2d)
        np.testing.assert_equal(result_python_3d, result_cython_3d)


@pytest.mark.parametrize('single_link', [False, True])
@pytest.mark.skipif(not cil.cython_funcs_available, reason='Cython functions are not available')
def test_equivalence_cell_idx_list_cython_python_version(single_link):
    #   D2Q9
    stencil_2d = tuple((x, y) for x, y in product([-1, 0, 1], [-1, 0, 1]))
    #   D3Q19
    stencil_3d = tuple(
        (x, y, z) for x, y, z in product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]) if abs(x) + abs(y) + abs(z) < 3)

    for dtype in [int, np.int16, np.uint32]:
        fluid_mask = dtype(1)
        mask = dtype(2)
        flag_field_2d = np.ones([15, 16], dtype=dtype) * fluid_mask
        flag_field_3d = np.ones([15, 16, 17], dtype=dtype) * fluid_mask

        flag_field_2d[0, :] = mask
        flag_field_2d[-1, :] = mask
        flag_field_2d[7, 7] = mask

        flag_field_3d[0, :, :] = mask
        flag_field_3d[-1, :, :] = mask
        flag_field_3d[7, 7, 7] = mask

        result_python_2d = cil._create_index_list_python(flag_field_2d, mask, fluid_mask,
                                                         stencil_2d, single_link, False)

        result_python_3d = cil._create_index_list_python(flag_field_3d, mask, fluid_mask,
                                                         stencil_3d, single_link, False)

        result_cython_2d = cil.create_boundary_index_list(flag_field_2d, stencil_2d, mask, fluid_mask, None,
                                                          False, single_link)
        result_cython_3d = cil.create_boundary_index_list(flag_field_3d, stencil_3d, mask, fluid_mask, None,
                                                          False, single_link)

        np.testing.assert_equal(result_python_2d, result_cython_2d)
        np.testing.assert_equal(result_python_3d, result_cython_3d)


@pytest.mark.parametrize('inner_or_boundary', [False, True])
def test_normal_calculation(inner_or_boundary):
    stencil = tuple((x, y) for x, y in product([-1, 0, 1], [-1, 0, 1]))
    domain_size = (32, 32)
    dtype = np.uint32
    fluid_mask = dtype(1)
    mask = dtype(2)
    flag_field = np.ones([domain_size[0], domain_size[1]], dtype=dtype) * fluid_mask

    radius_inner = domain_size[0] // 4
    radius_outer = domain_size[0] // 2
    y_mid = domain_size[1] / 2
    x_mid = domain_size[0] / 2

    for x in range(0, domain_size[0]):
        for y in range(0, domain_size[1]):
            if (y - y_mid) ** 2 + (x - x_mid) ** 2 < radius_inner ** 2:
                flag_field[x, y] = mask
            if (x - x_mid) ** 2 + (y - y_mid) ** 2 > radius_outer ** 2:
                flag_field[x, y] = mask

    args_no_gl = (flag_field, mask, fluid_mask, np.array(stencil, dtype=np.int32), True)
    index_list = cil._create_index_list_python(*args_no_gl, inner_or_boundary=inner_or_boundary, nr_of_ghost_layers=1)

    checkmask = mask if inner_or_boundary else fluid_mask

    for cell in index_list:
        idx = cell[2]
        cell = tuple((cell[0], cell[1]))
        sum_cells = np.zeros(len(cell))
        for dir_idx, direction in enumerate(stencil):
            neighbor_cell = tuple([cell_i + dir_i for cell_i, dir_i in zip(cell, direction)])
            if any(not 0 <= e < upper for e, upper in zip(neighbor_cell, flag_field.shape)):
                continue
            if flag_field[neighbor_cell] & checkmask:
                sum_cells += np.array(direction)

        assert np.argmax(np.inner(sum_cells, stencil)) == idx
